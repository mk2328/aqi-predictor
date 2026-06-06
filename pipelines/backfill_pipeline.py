import os
import time
import requests
from datetime import datetime, timezone, timedelta
from supabase import create_client
import pandas as pd

# ── Config ──────────────────────────────────────────────
AQICN_TOKEN       = os.environ["AQICN_TOKEN"]
OPENWEATHER_TOKEN = os.environ["OPENWEATHER_TOKEN"]
SUPABASE_URL      = os.environ["SUPABASE_URL"]
SUPABASE_KEY      = os.environ["SUPABASE_KEY"]

LAT           = 24.8607
LON           = 67.0011
CITY          = "karachi"
BACKFILL_DAYS = 90
CHUNK_DAYS    = 30

# ── Karachi Monthly Weather Averages (Based on historical climate data) ──
# Source: Karachi climate records
KARACHI_WEATHER = {
    1:  {"temp": 19.2, "humidity": 60, "pressure": 1016, "wind": 3.2},
    2:  {"temp": 20.8, "humidity": 58, "pressure": 1014, "wind": 3.5},
    3:  {"temp": 25.1, "humidity": 60, "pressure": 1011, "wind": 4.1},
    4:  {"temp": 29.3, "humidity": 63, "pressure": 1007, "wind": 4.8},
    5:  {"temp": 31.2, "humidity": 68, "pressure": 1005, "wind": 5.1},
    6:  {"temp": 30.8, "humidity": 73, "pressure": 1004, "wind": 5.8},
    7:  {"temp": 29.5, "humidity": 78, "pressure": 1004, "wind": 5.2},
    8:  {"temp": 28.9, "humidity": 77, "pressure": 1005, "wind": 4.8},
    9:  {"temp": 29.1, "humidity": 74, "pressure": 1007, "wind": 4.2},
    10: {"temp": 28.3, "humidity": 68, "pressure": 1010, "wind": 3.8},
    11: {"temp": 24.6, "humidity": 62, "pressure": 1014, "wind": 3.3},
    12: {"temp": 20.9, "humidity": 61, "pressure": 1016, "wind": 3.1},
}

# ── AQI Converter ────────────────────────────────────────
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

# ── Fetch OpenWeather Historical Pollution ──────────────
def fetch_ow_pollution(start_ts: int, end_ts: int) -> list:
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={LAT}&lon={LON}&start={start_ts}&end={end_ts}&appid={OPENWEATHER_TOKEN}"
    )
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json().get("list", [])
        except Exception as e:
            print(f"   ⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return []

# ── Compute Features ────────────────────────────────────
def compute_features(entry: dict, prev_aqi: float = None) -> dict:
    dt = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
    components = entry.get("components", {})

    pm25 = components.get("pm2_5", 0.0)
    pm10 = components.get("pm10", 0.0)
    o3   = components.get("o3", 0.0)
    no2  = components.get("no2", 0.0)
    co   = components.get("co", 0.0)
    so2  = components.get("so2", 0.0)

    aqi = pm25_to_aqi(pm25)
    aqi_change_rate = (aqi - prev_aqi) if prev_aqi is not None else 0.0

    # Use Karachi monthly climate averages instead of 0
    month_weather = KARACHI_WEATHER.get(dt.month, KARACHI_WEATHER[6])

    return {
        "timestamp":       dt.isoformat(),
        "city":            CITY,
        "aqi":             aqi,
        "pm25":            round(pm25, 2),
        "pm10":            round(pm10, 2),
        "o3":              round(o3, 2),
        "no2":             round(no2, 2),
        "co":              round(co, 2),
        "so2":             round(so2, 2),
        "temperature":     month_weather["temp"],
        "humidity":        month_weather["humidity"],
        "pressure":        month_weather["pressure"],
        "wind":            month_weather["wind"],
        "hour":            dt.hour,
        "day":             dt.weekday(),
        "month":           dt.month,
        "aqi_change_rate": aqi_change_rate,
    }

# ── Store Batch ─────────────────────────────────────────
def store_batch(supabase, rows: list):
    if not rows:
        return
    supabase.table("aqi_features") \
        .upsert(rows, on_conflict="timestamp,city") \
        .execute()

if __name__ == "__main__":
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    
    # 1. Initialize variables correctly
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=BACKFILL_DAYS)
    
    print(f"📅 Backfilling {BACKFILL_DAYS} days for Karachi...")
    
    all_data = []
    chunk_start = start_dt
    prev_aqi = None
    
    # 2. Fetch logic
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end_dt)
        print(f"🔄 Fetching: {chunk_start.date()} to {chunk_end.date()}")
        
        pollution_list = fetch_ow_pollution(int(chunk_start.timestamp()), int(chunk_end.timestamp()))
        
        for entry in pollution_list:
            features = compute_features(entry, prev_aqi)
            prev_aqi = features["aqi"]
            all_data.append(features)
        
        chunk_start = chunk_end
        time.sleep(2)

    # 3. Data Cleaning & Rolling Mean (Crucial for model stability)
    df = pd.DataFrame(all_data)
    
    # Filter outliers
    df = df[(df['aqi'] >= 30) & (df['aqi'] <= 350)].copy()
    
    # Rolling Mean for smoother prediction
    df['aqi_rolling_mean'] = df['aqi'].rolling(window=3, min_periods=1).mean()
    
    # 4. Final Upsert (Handles new and updates existing records)
    final_data = df.to_dict(orient='records')
    store_batch(supabase, final_data)
    
    print(f"\n🎉 Success! {len(final_data)} rows cleaned and stored.")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    # ... (rest of config)
    
    # 1. Fetch data into a temporary list first
    all_data = []
    chunk_start = start_dt
    prev_aqi = None
    
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end_dt)
        pollution_list = fetch_ow_pollution(int(chunk_start.timestamp()), int(chunk_end.timestamp()))
        
        for entry in pollution_list:
            features = compute_features(entry, prev_aqi)
            prev_aqi = features["aqi"]
            all_data.append(features)
        
        chunk_start = chunk_end

    # 2. Convert to DataFrame for Cleaning & Rolling Mean
    df = pd.DataFrame(all_data)
    
    # 3. 🧹 Outlier Removal
    df = df[(df['aqi'] >= 30) & (df['aqi'] <= 350)]
    
    # 4. 📈 Add Rolling Mean (The "Trend" Feature)
    # Yeh aapke model ko noise se bachaey ga
    df['aqi_rolling_mean'] = df['aqi'].rolling(window=3, min_periods=1).mean()
    
    # 5. Convert back to list of dicts and Store
    final_data = df.to_dict(orient='records')
    store_batch(supabase, final_data) # Aapka existing store_batch function use karein
    
    print(f"\n🎉 Backfill complete! {len(final_data)} clean rows with rolling mean stored.")