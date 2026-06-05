import os
import time
import requests
from datetime import datetime, timezone
from supabase import create_client

# ── Config ──────────────────────────────────────────────
OPENWEATHER_TOKEN = os.environ["OPENWEATHER_TOKEN"]
SUPABASE_URL      = os.environ["SUPABASE_URL"]
SUPABASE_KEY      = os.environ["SUPABASE_KEY"]

LAT  = 24.8607
LON  = 67.0011
CITY = "karachi"

# ── Robust Fetch with Retry ─────────────────────────────
def fetch_with_retry(url: str, max_retries: int = 3, timeout: int = 25):
    """Fetch API with retry and exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            print(f"   ✅ API call successful (Attempt {attempt+1})")
            return response.json()
        except requests.exceptions.Timeout:
            print(f"   ⚠️ Attempt {attempt+1}/{max_retries} - Timeout, retrying...")
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️ Attempt {attempt+1}/{max_retries} failed: {e}")
        
        if attempt < max_retries - 1:
            wait = (2 ** attempt) * 2
            time.sleep(wait)
    
    raise Exception(f"❌ Failed to fetch data from {url} after {max_retries} attempts")


# ── Fetch Air Pollution ─────────────────────────────────
def fetch_air_pollution() -> dict:
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_TOKEN}"
    data = fetch_with_retry(url)
    return data["list"][0]


# ── Fetch Weather ───────────────────────────────────────
def fetch_weather() -> dict:
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_TOKEN}&units=metric"
    return fetch_with_retry(url)


# ── EPA PM2.5 to AQI Converter ──────────────────────────
def pm25_to_aqi(pm25: float) -> float:
    if pm25 <= 0:
        return 0.0
    breakpoints = [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)
    ]
    for (c_low, c_high, i_low, i_high) in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return round(aqi, 1)
    return 500.0


# ── Compute Features ────────────────────────────────────
def compute_features(pollution: dict, weather: dict) -> dict:
    now = datetime.now(timezone.utc)
    components = pollution.get("components", {})

    pm25 = components.get("pm2_5", 0.0)
    aqi = pm25_to_aqi(pm25)

    # Pollutants
    pm10 = components.get("pm10", 0.0)
    o3   = components.get("o3", 0.0)
    no2  = components.get("no2", 0.0)
    co   = components.get("co", 0.0)
    so2  = components.get("so2", 0.0)

    # Weather
    temperature = weather.get("main", {}).get("temp", 0.0)
    humidity    = weather.get("main", {}).get("humidity", 0.0)
    pressure    = weather.get("main", {}).get("pressure", 0.0)
    wind        = weather.get("wind", {}).get("speed", 0.0)

    return {
        "timestamp":       now.isoformat(),
        "city":            CITY,
        "aqi":             round(aqi, 1),
        "pm25":            round(pm25, 2),
        "pm10":            round(pm10, 2),
        "o3":              round(o3, 2),
        "no2":             round(no2, 2),
        "co":              round(co, 2),
        "so2":             round(so2, 2),
        "temperature":     round(temperature, 1),
        "humidity":        round(humidity, 1),
        "pressure":        round(pressure, 1),
        "wind":            round(wind, 1),
        "hour":            now.hour,
        "day":             now.weekday(),
        "month":           now.month,
        "aqi_change_rate": 0.0,
    }


# ── Add AQI Change Rate ─────────────────────────────────
def add_change_rate(supabase, features: dict) -> dict:
    result = (
        supabase.table("aqi_features")
        .select("aqi")
        .eq("city", CITY)
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    if result.data:
        last_aqi = result.data[0]["aqi"]
        features["aqi_change_rate"] = float(features["aqi"]) - float(last_aqi)
    else:
        features["aqi_change_rate"] = 0.0
    return features


# ── Store Features ──────────────────────────────────────
def store_features(supabase, features: dict):
    supabase.table("aqi_features") \
        .upsert(features, on_conflict="timestamp,city") \
        .execute()
    print(f"✅ Stored | AQI={features['aqi']} | PM2.5={features['pm25']} | "
          f"Temp={features['temperature']}°C | {features['timestamp']}")


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"🌍 Fetching real-time data for {CITY}...")

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    try:
        pollution = fetch_air_pollution()
        weather   = fetch_weather()
        
        features = compute_features(pollution, weather)
        features = add_change_rate(supabase, features)
        store_features(supabase, features)

        print("🎉 Feature pipeline complete!")
        
    except Exception as e:
        print(f"❌ Feature pipeline failed: {e}")
        raise