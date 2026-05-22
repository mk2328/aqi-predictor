import os
import requests
import pandas as pd
from datetime import datetime, timezone
from supabase import create_client

# ── Config ──────────────────────────────────────────────
AQICN_TOKEN   = os.environ["AQICN_TOKEN"]
SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_KEY"]

CITY = "karachi"   # change to your city if needed

# ── Step 1: Fetch raw data from AQICN ───────────────────
def fetch_aqi_data(city: str) -> dict:
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    if data["status"] != "ok":
        raise ValueError(f"AQICN error: {data}")
    return data["data"]

# ── Step 2: Compute features ────────────────────────────
def compute_features(raw: dict) -> dict:
    iaqi = raw.get("iaqi", {})
    now  = datetime.now(timezone.utc)

    # pull pollutants & weather (default 0 if missing)
    aqi         = raw.get("aqi", 0)
    pm25        = iaqi.get("pm25", {}).get("v", 0)
    pm10        = iaqi.get("pm10", {}).get("v", 0)
    o3          = iaqi.get("o3",   {}).get("v", 0)
    no2         = iaqi.get("no2",  {}).get("v", 0)
    co          = iaqi.get("co",   {}).get("v", 0)
    so2         = iaqi.get("so2",  {}).get("v", 0)
    temperature = iaqi.get("t",    {}).get("v", 0)
    humidity    = iaqi.get("h",    {}).get("v", 0)
    pressure    = iaqi.get("p",    {}).get("v", 0)
    wind        = iaqi.get("w",    {}).get("v", 0)

    # pm25 should not equal overall AQI
    if pm25 == aqi:
        pm25 = 0.0

    # time-based features
    hour  = now.hour
    day   = now.weekday()   # 0=Monday … 6=Sunday
    month = now.month

    # derived feature — placeholder (real rate needs previous row)
    aqi_change_rate = 0.0

    return {
        "timestamp":       now.isoformat(),
        "city":            CITY,
        "aqi":             aqi,
        "pm25":            pm25,
        "pm10":            pm10,
        "o3":              o3,
        "no2":             no2,
        "co":              co,
        "so2":             so2,
        "temperature":     temperature,
        "humidity":        humidity,
        "pressure":        pressure,
        "wind":            wind,
        "hour":            hour,
        "day":             day,
        "month":           month,
        "aqi_change_rate": aqi_change_rate,
    }

# ── Step 3: Calculate real AQI change rate ──────────────
def add_change_rate(supabase, features: dict) -> dict:
    """Fetch the last row from Supabase and compute change rate."""
    result = (
        supabase.table("aqi_features")
        .select("aqi, timestamp")
        .eq("city", CITY)
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    if result.data:
        last_aqi = result.data[0]["aqi"]
        features["aqi_change_rate"] = float(features["aqi"]) - float(last_aqi)
    return features

# ── Step 4: Store in Supabase ────────────────────────────
def store_features(supabase, features: dict):
    result = (
        supabase.table("aqi_features")
        .upsert(features, on_conflict="timestamp,city")
        .execute()
    )
    print(f"✅ Stored: AQI={features['aqi']} at {features['timestamp']}")
    return result

# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"🌍 Fetching AQI data for {CITY}...")

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    raw      = fetch_aqi_data(CITY)
    features = compute_features(raw)
    features = add_change_rate(supabase, features)
    store_features(supabase, features)

    print("🎉 Feature pipeline complete!")