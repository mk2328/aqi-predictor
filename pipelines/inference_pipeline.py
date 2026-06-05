"""
inference_pipeline.py
─────────────────────
Generates a 3-day (72-hour) AQI forecast using the trained lagged-features model.
Compatible with the new training_pipeline.py (Random Forest with lags).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from supabase import create_client

# ── Config ──────────────────────────────────────────────
SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
CITY           = "karachi"
MODEL_PATH     = "models/best_model.pkl"
SCALER_PATH    = "models/scaler.pkl"
FEATURE_COLS_PATH = "models/feature_cols.json"
FORECAST_HOURS = 72

# AQI Categories
AQI_CATEGORIES = [
    (0,   50,  "Good",            "#00e400"),
    (51,  100, "Moderate",        "#ffff00"),
    (101, 150, "Unhealthy for SG","#ff7e00"),
    (151, 200, "Unhealthy",       "#ff0000"),
    (201, 300, "Very Unhealthy",  "#8f3f97"),
    (301, 500, "Hazardous",       "#7e0023"),
]

def aqi_category(aqi: float) -> dict:
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return {"label": label, "color": color}
    return {"label": "Hazardous", "color": "#7e0023"}


# ── Load Model & Features ───────────────────────────────
def load_model():
    scaler = joblib.load(SCALER_PATH)
    model  = joblib.load(MODEL_PATH)
    
    # Load feature columns used during training
    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)
    
    print(f"📦 Loaded model: {type(model).__name__} | Horizon: 72h | Features: {len(feature_cols)}")
    return model, scaler, feature_cols


def predict_one(model, scaler, feature_vec: np.ndarray) -> float:
    scaled = scaler.transform(feature_vec.reshape(1, -1))
    return float(model.predict(scaled)[0])


# ── Fetch Recent History (for lags) ─────────────────────
def fetch_recent_history(supabase, n: int = 24) -> pd.DataFrame:
    result = (
        supabase.table("aqi_features")
        .select("*")
        .eq("city", CITY)
        .order("timestamp", desc=True)
        .limit(n)
        .execute()
    )
    df = pd.DataFrame(result.data)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── Simulate Future Pollutants ──────────────────────────
def decay_pollutants(current: dict, hour_offset: int) -> dict:
    rng = np.random.default_rng(seed=hour_offset)
    decay = 0.985  # Slightly stronger decay
    
    new = dict(current)
    noise = lambda s: rng.normal(0, s)
    
    new["pm25"] = max(0, new["pm25"] * decay + noise(0.8))
    new["pm10"] = max(0, new["pm10"] * decay + noise(1.5))
    new["o3"]   = max(0, new["o3"] * (1 + 0.0015 * new.get("temperature", 25)) + noise(0.4))
    new["no2"]  = max(0, new["no2"] * decay + noise(0.4))
    new["co"]   = max(0, new["co"] * decay + noise(4.0))
    new["so2"]  = max(0, new["so2"] * decay + noise(0.3))
    
    new["temperature"] = new.get("temperature", 25) + noise(0.4)
    new["humidity"]    = max(0, min(100, new.get("humidity", 50) + noise(1.2)))
    new["pressure"]    = new.get("pressure", 1013) + noise(0.6)
    new["wind"]        = max(0, new.get("wind", 2) + noise(0.3))
    
    return new


# ── Generate Forecast with Lagged Features ──────────────
def generate_forecast(model, scaler, feature_cols, history_df: pd.DataFrame) -> list:
    predictions = []
    now = datetime.now(timezone.utc)
    current = history_df.iloc[-1].to_dict().copy()
    prev_aqi = float(current["aqi"])
    
    recent_history = history_df.copy()
    
    print("🔮 Generating 72-hour forecast (with lagged features)...")
    
    for h in range(1, FORECAST_HOURS + 1):
        future_dt = now + timedelta(hours=h)
        
        # Evolve pollutants
        current = decay_pollutants(current, h)
        
        # Temporary history for lag calculation
        temp_row = {
            "aqi": prev_aqi,
            "pm25": current["pm25"],
            "pm10": current["pm10"],
            "o3": current["o3"],
            "no2": current["no2"],
            "aqi_change_rate": current.get("aqi_change_rate", 0.0)
        }
        temp_df = pd.concat([recent_history, pd.DataFrame([temp_row])], ignore_index=True)
        
        # Build feature vector
        feature_dict = {}
        for col in feature_cols:
            if col == "hour":
                feature_dict[col] = future_dt.hour
            elif col == "day":
                feature_dict[col] = future_dt.weekday()      # ← Fixed: added ()
            elif col == "month":
                feature_dict[col] = future_dt.month
            elif "_lag_" in col:
                try:
                    lag = int(col.split("_lag_")[1].split("h")[0])
                    base_col = col.split("_lag_")[0]
                    if base_col in temp_df.columns:
                        feature_dict[col] = float(temp_df[base_col].iloc[-lag])
                    else:
                        feature_dict[col] = 0.0
                except:
                    feature_dict[col] = 0.0
            else:
                # Other base features (pm25, temperature, etc.)
                feature_dict[col] = float(current.get(col, 0.0))
        
        # Build feature vector in exact order
        feat = np.array([feature_dict.get(col, 0.0) for col in feature_cols], dtype=float)
        
        pred_aqi = predict_one(model, scaler, feat)
        pred_aqi = max(0, min(500, pred_aqi))
        
        cat = aqi_category(pred_aqi)
        
        row = {
            "timestamp": future_dt.isoformat(),
            "city": CITY,
            "forecast_hour": h,
            "predicted_aqi": round(pred_aqi, 1),
            "aqi_category": cat["label"],
            "aqi_color": cat["color"],
            "is_alert": pred_aqi >= 150,
            "created_at": now.isoformat(),
        }
        predictions.append(row)
        
        # Update for next iteration
        current["aqi"] = pred_aqi
        prev_aqi = pred_aqi
        recent_history = temp_df.copy()
    
    return predictions

# ── Store Predictions ───────────────────────────────────
def store_predictions(supabase, predictions: list):
    supabase.table("aqi_predictions").delete().eq("city", CITY).execute()
    
    for i in range(0, len(predictions), 100):
        supabase.table("aqi_predictions").insert(predictions[i:i+100]).execute()
    
    alerts = [p for p in predictions if p["is_alert"]]
    print(f"✅ Stored {len(predictions)} predictions")
    if alerts:
        print(f"🚨 ALERT: {len(alerts)} hazardous hours predicted")
    else:
        print("✅ No hazardous AQI predicted in next 3 days")


# ── Summary ─────────────────────────────────────────────
def print_summary(predictions: list):
    print("\n📅 3-Day AQI Forecast Summary:")
    print(f"   {'Date':<12} {'Avg AQI':>8} {'Max AQI':>8} {'Category':<20} {'Alerts':>6}")
    print("   " + "-" * 60)

    df = pd.DataFrame(predictions)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    for date, group in df.groupby("date"):
        avg = group["predicted_aqi"].mean()
        mx  = group["predicted_aqi"].max()
        cat = aqi_category(avg)["label"]
        alerts = group["is_alert"].sum()
        print(f"   {str(date):<12} {avg:>8.1f} {mx:>8.1f} {cat:<20} {alerts:>6}")


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"🔮 Generating 72-hour AQI Forecast for Karachi...")

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    model, scaler, feature_cols = load_model()

    history_df = fetch_recent_history(supabase, n=24)
    if len(history_df) < 5:
        print("❌ Not enough historical data. Run feature_pipeline.py first!")
        exit(1)

    seed_row = history_df.iloc[-1].to_dict()
    print(f"   Seed: AQI={seed_row.get('aqi', 'N/A')} | PM2.5={seed_row.get('pm25', 'N/A')}")

    predictions = generate_forecast(model, scaler, feature_cols, history_df)
    
    store_predictions(supabase, predictions)
    print_summary(predictions)

    print("\n🎉 Inference pipeline complete!")