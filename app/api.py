"""
app/api.py
──────────
FastAPI backend for the AQI Predictor dashboard.

Endpoints
---------
GET  /                      Health check
GET  /current               Latest real AQI + weather snapshot
GET  /forecast              72-hour predictions from aqi_predictions table
GET  /forecast/daily        3-day daily summaries (avg/min/max AQI)
GET  /history?hours=48      Recent actual AQI readings
GET  /shap                  SHAP feature importance values
GET  /alerts                Active hazardous AQI alerts from forecast
GET  /model/info            Latest model from registry

Run locally:
    uvicorn app.api:app --reload --port 8000
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client

# ── Config ──────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
SHAP_PATH    = "models/shap_values.json"
CITY         = "karachi"

app = FastAPI(
    title="AQI Predictor API",
    description="Real-time + 3-day AQI forecast for Karachi",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Supabase client (lazy init) ──────────────────────────
_supabase = None
def get_supabase():
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase

# ── AQI helpers ──────────────────────────────────────────
AQI_CATEGORIES = [
    (0,   50,  "Good",              "#00e400"),
    (51,  100, "Moderate",          "#ffff00"),
    (101, 150, "Unhealthy for SG",  "#ff7e00"),
    (151, 200, "Unhealthy",         "#ff0000"),
    (201, 300, "Very Unhealthy",    "#8f3f97"),
    (301, 500, "Hazardous",         "#7e0023"),
]

def aqi_category(aqi: float) -> dict:
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return {"label": label, "color": color}
    return {"label": "Hazardous", "color": "#7e0023"}

# ── Routes ───────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "city": CITY, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/current")
def current():
    """Most recent actual AQI reading."""
    sb     = get_supabase()
    result = (
        sb.table("aqi_features")
        .select("*")
        .eq("city", CITY)
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise HTTPException(404, "No data available. Run feature_pipeline.py first.")

    row = result.data[0]
    cat = aqi_category(float(row.get("aqi", 0)))
    return {
        **row,
        "aqi_category": cat["label"],
        "aqi_color":    cat["color"],
    }


@app.get("/forecast")
def forecast():
    """Full 72-hour hourly predictions."""
    sb     = get_supabase()
    result = (
        sb.table("aqi_predictions")
        .select("*")
        .eq("city", CITY)
        .order("forecast_hour")
        .execute()
    )
    if not result.data:
        raise HTTPException(
            404,
            "No predictions found. Run inference_pipeline.py first."
        )
    return {"city": CITY, "count": len(result.data), "predictions": result.data}


@app.get("/forecast/daily")
def forecast_daily():
    """Aggregate forecast into 3 daily summaries."""
    sb     = get_supabase()
    result = (
        sb.table("aqi_predictions")
        .select("timestamp,predicted_aqi,is_alert")
        .eq("city", CITY)
        .order("forecast_hour")
        .execute()
    )
    if not result.data:
        raise HTTPException(404, "No predictions found.")

    from collections import defaultdict
    daily: dict = defaultdict(list)
    for row in result.data:
        date = row["timestamp"][:10]
        daily[date].append(row)

    summary = []
    for date, rows in sorted(daily.items()):
        aqis    = [r["predicted_aqi"] for r in rows]
        avg_aqi = round(sum(aqis) / len(aqis), 1)
        cat     = aqi_category(avg_aqi)
        summary.append({
            "date":       date,
            "avg_aqi":    avg_aqi,
            "min_aqi":    round(min(aqis), 1),
            "max_aqi":    round(max(aqis), 1),
            "category":   cat["label"],
            "color":      cat["color"],
            "alert_hours": sum(1 for r in rows if r["is_alert"]),
        })
    return {"city": CITY, "daily": summary}


@app.get("/history")
def history(hours: int = 48):
    """Recent actual AQI readings (default last 48 h)."""
    hours = min(hours, 720)   # cap at 30 days
    sb    = get_supabase()
    result = (
        sb.table("aqi_features")
        .select("timestamp,aqi,pm25,pm10,temperature,humidity,wind")
        .eq("city", CITY)
        .order("timestamp", desc=True)
        .limit(hours)
        .execute()
    )
    data = list(reversed(result.data)) if result.data else []
    return {"city": CITY, "hours": hours, "readings": data}


@app.get("/shap")
def shap_values():
    """SHAP feature importance values from last training run."""
    path = Path(SHAP_PATH)
    if not path.exists():
        raise HTTPException(
            404,
            "SHAP values not found. Run training_pipeline.py first."
        )
    with open(path) as f:
        data = json.load(f)
    return data


@app.get("/alerts")
def alerts():
    """Hours in the next 72 h where AQI ≥ 150 (Unhealthy)."""
    sb     = get_supabase()
    result = (
        sb.table("aqi_predictions")
        .select("*")
        .eq("city", CITY)
        .eq("is_alert", True)
        .order("forecast_hour")
        .execute()
    )
    alert_list = result.data or []
    return {
        "city":        CITY,
        "total_alerts": len(alert_list),
        "has_alerts":   len(alert_list) > 0,
        "alerts":       alert_list,
    }


@app.get("/model/info")
def model_info():
    sb = get_supabase()
    result = (
        sb.table("model_registry")
        .select("*")
        .order("version", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise HTTPException(404, "No model registered yet.")
    
    model = result.data[0]
    # Add horizon if exists
    model["horizon"] = model.get("horizon", 72)
    return model