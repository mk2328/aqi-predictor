import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from supabase import create_client

# ── Config ──────────────────────────────────────────────
OPENWEATHER_TOKEN = os.environ["OPENWEATHER_TOKEN"]
SUPABASE_URL      = os.environ["SUPABASE_URL"]
SUPABASE_KEY      = os.environ["SUPABASE_KEY"]

LAT, LON = 24.8607, 67.0011
CITY = "karachi"

# ── API Logic ───────────────────────────────────────────
def fetch_with_retry(url: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=25)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            time.sleep(2 ** attempt)
    raise Exception("API Fetch Failed")

def fetch_air_pollution():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_TOKEN}"
    return fetch_with_retry(url)["list"][0]

def fetch_weather():
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_TOKEN}&units=metric"
    return fetch_with_retry(url)

# ── Feature Engineering ─────────────────────────────────
def pm25_to_aqi(pm25: float) -> float:
    # (Existing conversion logic...)
    if pm25 <= 0: return 0.0
    breakpoints = [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
                   (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
                   (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]
    for (c_low, c_high, i_low, i_high) in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low, 1)
    return 500.0

def compute_features(pollution, weather, last_aqi=None):
    now = datetime.now(timezone.utc)
    comp = pollution.get("components", {})
    aqi = pm25_to_aqi(comp.get("pm2_5", 0.0))
    aqi = round(aqi * 0.70, 1)
    
    # Calculate Rolling Mean for Live Data (2-hour window)
    aqi_rolling_mean = (aqi + last_aqi) / 2 if last_aqi is not None else aqi
    
    return {
        "timestamp": now.isoformat(),
        "city": CITY,
        "aqi": aqi,
        "aqi_rolling_mean": round(aqi_rolling_mean, 1),
        "pm25": round(comp.get("pm2_5", 0.0), 2),
        "pm10": round(comp.get("pm10", 0.0), 2),
        "o3": round(comp.get("o3", 0.0), 2),
        "no2": round(comp.get("no2", 0.0), 2),
        "co": round(comp.get("co", 0.0), 2),
        "so2": round(comp.get("so2", 0.0), 2),
        "temperature": round(weather.get("main", {}).get("temp", 0.0), 1),
        "humidity": round(weather.get("main", {}).get("humidity", 0.0), 1),
        "pressure": round(weather.get("main", {}).get("pressure", 0.0), 1),
        "wind": round(weather.get("wind", {}).get("speed", 0.0), 1),
        "hour": now.hour,
        "day": now.weekday(),
        "month": now.month,
        "aqi_change_rate": float(aqi - last_aqi) if last_aqi is not None else 0.0
    }

# ── Main Pipeline ───────────────────────────────────────
if __name__ == "__main__":
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # 1. Fetch last state
    res = supabase.table("aqi_features").select("aqi").order("timestamp", desc=True).limit(1).execute()
    last_aqi = res.data[0]["aqi"] if res.data else None
    
    # 2. Compute
    features = compute_features(fetch_air_pollution(), fetch_weather(), last_aqi)
    
    # 3. Guardrail: Outlier Filter
    if 30 <= features["aqi"] <= 350:
        supabase.table("aqi_features").upsert(features, on_conflict="timestamp,city").execute()
        print(f"✅ Success: Stored clean data. AQI={features['aqi']} | RollingMean={features['aqi_rolling_mean']}")
    else:
        print(f"⚠️ Blocked outlier: AQI={features['aqi']}")