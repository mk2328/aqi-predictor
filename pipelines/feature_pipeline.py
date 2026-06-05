import os
import requests
import pandas as pd
from datetime import datetime, timezone
from supabase import create_client

# ── Config ──────────────────────────────────────────────
OPENWEATHER_TOKEN = os.environ["OPENWEATHER_TOKEN"]
SUPABASE_URL      = os.environ["SUPABASE_URL"]
SUPABASE_KEY      = os.environ["SUPABASE_KEY"]

# Karachi coordinates
LAT  = 24.8607
LON  = 67.0011
CITY = "karachi"

# ── Step 1A: Fetch Air Pollution data ───────────────────
def fetch_air_pollution() -> dict:
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_TOKEN}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data["list"][0]

# ── Step 1B: Fetch Weather data ──────────────────────────
def fetch_weather() -> dict:
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_TOKEN}&units=metric"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

# ── Step 2: Compute features ────────────────────────────
def compute_features(pollution: dict, weather: dict) -> dict:
    now        = datetime.now(timezone.utc)
    components = pollution.get("components", {})

    # OpenWeather AQI: 1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor
    # Convert to US AQI scale (approximate)
    ow_aqi = pollution.get("main", {}).get("aqi", 1)
    aqi_map = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250}
    aqi = aqi_map.get(ow_aqi, 25)

    # Pollutants (µg/m³)
    pm25 = components.get("pm2_5", 0.0)
    pm10 = components.get("pm10",  0.0)
    o3   = components.get("o3",    0.0)
    no2  = components.get("no2",   0.0)
    co   = components.get("co",    0.0)
    so2  = components.get("so2",   0.0)

    # Weather
    temperature = weather.get("main", {}).get("temp",     0.0)
    humidity    = weather.get("main", {}).get("humidity", 0.0)
    pressure    = weather.get("main", {}).get("pressure", 0.0)
    wind        = weather.get("wind", {}).get("speed",    0.0)

    # Time-based features
    hour  = now.hour
    day   = now.weekday()
    month = now.month

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
        "aqi_change_rate": 0.0,
    }

# ── Step 3: Calculate AQI change rate ───────────────────
def add_change_rate(supabase, features: dict) -> dict:
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
    supabase.table("aqi_features") \
        .upsert(features, on_conflict="timestamp,city") \
        .execute()
    print(f"✅ Stored | AQI={features['aqi']} | "
          f"PM2.5={features['pm25']} | "
          f"Temp={features['temperature']}°C | "
          f"at {features['timestamp']}")

# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"🌍 Fetching real-time data for {CITY}...")

    supabase  = create_client(SUPABASE_URL, SUPABASE_KEY)
    pollution = fetch_air_pollution()
    weather   = fetch_weather()
    features  = compute_features(pollution, weather)
    features  = add_change_rate(supabase, features)
    store_features(supabase, features)

    print("🎉 Feature pipeline complete!")