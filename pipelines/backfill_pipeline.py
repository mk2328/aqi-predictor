import os
import time
import requests
from datetime import datetime, timezone, timedelta
from supabase import create_client

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

# ── AQI Converter (Consistent with feature_pipeline) ─────
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
        "temperature":     0.0,      # OpenWeather historical weather paid hai
        "humidity":        0.0,
        "pressure":        0.0,
        "wind":            0.0,
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


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=BACKFILL_DAYS)

    print(f"📅 Backfilling {BACKFILL_DAYS} days for Karachi...")
    print(f"   Period: {start_dt.date()} → {end_dt.date()}\n")

    total_stored = 0
    chunk_start  = start_dt
    prev_aqi     = None

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end_dt)
        print(f"🔄 Fetching chunk: {chunk_start.date()} → {chunk_end.date()}")

        pollution_list = fetch_ow_pollution(
            int(chunk_start.timestamp()),
            int(chunk_end.timestamp())
        )
        print(f"   Got {len(pollution_list)} readings")

        batch = []
        for entry in pollution_list:
            features = compute_features(entry, prev_aqi)
            prev_aqi = features["aqi"]
            batch.append(features)

            if len(batch) >= 100:
                store_batch(supabase, batch)
                total_stored += len(batch)
                batch = []

        if batch:
            store_batch(supabase, batch)
            total_stored += len(batch)

        print(f"   ✅ Chunk stored | Total: {total_stored}")
        chunk_start = chunk_end
        time.sleep(2)

    print(f"\n🎉 Backfill complete! {total_stored} rows stored.")