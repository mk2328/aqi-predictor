import os
import time
import requests
from datetime import datetime, timezone, timedelta
from supabase import create_client

# ── Config ──────────────────────────────────────────────
OPENWEATHER_TOKEN = os.environ["OPENWEATHER_TOKEN"]
SUPABASE_URL      = os.environ["SUPABASE_URL"]
SUPABASE_KEY      = os.environ["SUPABASE_KEY"]

LAT  = 24.8607
LON  = 67.0011
CITY = "karachi"

BACKFILL_DAYS = 90
CHUNK_DAYS    = 30  # fetch 30 days at a time

# ── Fetch historical air pollution ───────────────────────
def fetch_historical_pollution(start_ts: int, end_ts: int) -> list:
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={LAT}&lon={LON}"
        f"&start={start_ts}&end={end_ts}"
        f"&appid={OPENWEATHER_TOKEN}"
    )
    for attempt in range(3):  # retry 3 times
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response.json().get("list", [])
        except Exception as e:
            print(f"   ⚠️  Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return []

# ── Compute features ─────────────────────────────────────
def compute_features(entry: dict) -> dict:
    dt         = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
    components = entry.get("components", {})

    ow_aqi  = entry.get("main", {}).get("aqi", 1)
    aqi_map = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250}
    aqi     = aqi_map.get(ow_aqi, 25)

    return {
        "timestamp":       dt.isoformat(),
        "city":            CITY,
        "aqi":             aqi,
        "pm25":            components.get("pm2_5", 0.0),
        "pm10":            components.get("pm10",  0.0),
        "o3":              components.get("o3",    0.0),
        "no2":             components.get("no2",   0.0),
        "co":              components.get("co",    0.0),
        "so2":             components.get("so2",   0.0),
        "temperature":     0.0,
        "humidity":        0.0,
        "pressure":        0.0,
        "wind":            0.0,
        "hour":            dt.hour,
        "day":             dt.weekday(),
        "month":           dt.month,
        "aqi_change_rate": 0.0,
    }

# ── Store batch in Supabase ──────────────────────────────
def store_batch(supabase, rows: list):
    if not rows:
        return
    supabase.table("aqi_features") \
        .upsert(rows, on_conflict="timestamp,city") \
        .execute()

# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=BACKFILL_DAYS)

    print(f"📅 Backfilling {BACKFILL_DAYS} days in {CHUNK_DAYS}-day chunks")

    total_stored = 0
    chunk_start  = start_dt

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end_dt)

        print(f"\n🔄 Fetching: {chunk_start.date()} → {chunk_end.date()}")

        pollution_list = fetch_historical_pollution(
            int(chunk_start.timestamp()),
            int(chunk_end.timestamp())
        )
        print(f"   Got {len(pollution_list)} readings")

        batch    = []
        prev_aqi = None

        for entry in pollution_list:
            features = compute_features(entry)

            if prev_aqi is not None:
                features["aqi_change_rate"] = float(features["aqi"]) - float(prev_aqi)
            prev_aqi = features["aqi"]

            batch.append(features)

            if len(batch) >= 100:
                store_batch(supabase, batch)
                total_stored += len(batch)
                batch = []

        if batch:
            store_batch(supabase, batch)
            total_stored += len(batch)

        print(f"   ✅ Chunk stored! Total so far: {total_stored}")
        chunk_start = chunk_end
        time.sleep(2)  # small pause between chunks

    print(f"\n🎉 Backfill complete! {total_stored} total rows stored.")