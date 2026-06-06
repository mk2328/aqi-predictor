import os
import joblib
import numpy as np
import requests
import json
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client

# ── Config ───────────────────────────────────────────────
SUPABASE_URL      = os.environ["SUPABASE_URL"]
SUPABASE_KEY      = os.environ["SUPABASE_KEY"]
OPENWEATHER_TOKEN = os.environ["OPENWEATHER_TOKEN"]

LAT  = 24.8607
LON  = 67.0011
CITY = "karachi"

MODEL_PATH     = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
SCALER_PATH    = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
FEATURE_PATH   = os.path.join(os.path.dirname(__file__), "..", "models", "feature_cols.json")
SHAP_PATH      = os.path.join(os.path.dirname(__file__), "..", "models", "shap_values.json")

app = FastAPI(title="AQI Predictor API — Karachi", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Helpers ──────────────────────────────────────────────
AQI_CATEGORIES = [
    (0,   50,  "Good",                     "#00E400", "🟢"),
    (51,  100, "Moderate",                 "#FFFF00", "🟡"),
    (101, 150, "Unhealthy for Sensitive",  "#FF7E00", "🟠"),
    (151, 200, "Unhealthy",                "#FF0000", "🔴"),
    (201, 300, "Very Unhealthy",           "#8F3F97", "🟣"),
    (301, 500, "Hazardous",                "#7E0023", "⚫"),
]

def aqi_category(aqi: float) -> dict:
    for lo, hi, label, color, emoji in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return {"label": label, "color": color, "emoji": emoji}
    return {"label": "Hazardous", "color": "#7E0023", "emoji": "⚫"}

def pm25_to_aqi(pm25: float) -> float:
    if pm25 <= 0: return 0.0
    bp = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
          (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
    for c0,c1,i0,i1 in bp:
        if c0 <= pm25 <= c1:
            return round(((i1-i0)/(c1-c0))*(pm25-c0)+i0, 1)
    return 500.0

def load_model():
    try:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURE_PATH) as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols
    except Exception as e:
        return None, None, None

def fetch_live_data():
    poll = requests.get(
        f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_TOKEN}",
        timeout=10
    ).json()["list"][0]
    wx = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_TOKEN}&units=metric",
        timeout=10
    ).json()
    c = poll["components"]
    return {
        "pm25": c.get("pm2_5", 0), "pm10": c.get("pm10", 0),
        "o3": c.get("o3", 0), "no2": c.get("no2", 0),
        "co": c.get("co", 0), "so2": c.get("so2", 0),
        "temperature": wx["main"]["temp"], "humidity": wx["main"]["humidity"],
        "pressure": wx["main"]["pressure"], "wind": wx["wind"]["speed"],
    }

# ── Routes ───────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "AQI Predictor API", "city": CITY, "version": "2.0.0"}

@app.get("/current")
def get_current():
    try:
        data = fetch_live_data()
        aqi  = pm25_to_aqi(data["pm25"])
        cat  = aqi_category(aqi)
        return {
            "city": CITY, "timestamp": datetime.now(timezone.utc).isoformat(),
            "aqi": aqi, "category": cat, "hazardous": aqi > 150,
            "pollutants": {k: data[k] for k in ["pm25","pm10","o3","no2","co","so2"]},
            "weather": {k: data[k] for k in ["temperature","humidity","pressure","wind"]},
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/forecast")
def get_forecast():
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        result = (
            supabase.table("aqi_predictions")
            .select("*")
            .eq("city", CITY)
            .order("timestamp")
            .execute()
        )
        return {"city": CITY, "forecasts": result.data, "count": len(result.data)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/history")
def get_history(limit: int = 168):
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        result = (
            supabase.table("aqi_features")
            .select("timestamp,aqi,pm25,pm10,o3,temperature,humidity")
            .eq("city", CITY)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        return {"city": CITY, "data": list(reversed(result.data))}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/model-info")
def get_model_info():
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        result = (
            supabase.table("model_registry")
            .select("*")
            .order("trained_at", desc=True)
            .limit(1)
            .execute()
        )
        shap_data = {}
        try:
            with open(SHAP_PATH) as f:
                shap_data = json.load(f)
        except:
            pass
        return {**result.data[0], "shap": shap_data} if result.data else {}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/stats")
def get_stats():
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        result = supabase.table("aqi_features").select("aqi").eq("city", CITY).execute()
        aqis = [r["aqi"] for r in result.data if r["aqi"]]
        return {
            "total_records": len(aqis),
            "avg_aqi": round(sum(aqis)/len(aqis), 1),
            "max_aqi": round(max(aqis), 1),
            "min_aqi": round(min(aqis), 1),
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/shap")
def get_shap():
    try:
        with open(SHAP_PATH) as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(500, str(e))