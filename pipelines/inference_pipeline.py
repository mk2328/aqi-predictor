import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from supabase import create_client

# ── Config ──────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
CITY         = "karachi"
MODEL_PATH   = "models/best_model.pkl"
SCALER_PATH  = "models/scaler.pkl"
FEATURE_COLS_PATH = "models/feature_cols.json"
FORECAST_HOURS = 72

def aqi_category(aqi: float) -> dict:
    categories = [(0, 50, "Good", "#00e400"), (51, 100, "Moderate", "#ffff00"),
                  (101, 150, "Unhealthy for SG", "#ff7e00"), (151, 200, "Unhealthy", "#ff0000"),
                  (201, 300, "Very Unhealthy", "#8f3f97"), (301, 500, "Hazardous", "#7e0023")]
    for lo, hi, label, color in categories:
        if lo <= aqi <= hi: return {"label": label, "color": color}
    return {"label": "Hazardous", "color": "#7e0023"}

# ── Load Model & Features ───────────────────────────────
def load_model():
    scaler = joblib.load(SCALER_PATH)
    model  = joblib.load(MODEL_PATH)
    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

# ── Generate Forecast ───────────────────────────────────
def generate_forecast(model, scaler, feature_cols, history_df: pd.DataFrame) -> list:
    predictions = []
    now = datetime.now(timezone.utc)
    latest = history_df.iloc[-1].to_dict()
    
    for h in range(1, FORECAST_HOURS + 1):
        future_dt = now + timedelta(hours=h)
        feature_dict = {}
        for col in feature_cols:
            if col == "hour": feature_dict[col] = future_dt.hour
            elif col == "day": feature_dict[col] = future_dt.weekday()
            elif col == "month": feature_dict[col] = future_dt.month
            elif "_lag_" in col:
                lag = int(col.split("_lag_")[1].split("h")[0])
                base_col = col.split("_lag_")[0]
                feature_dict[col] = float(history_df[base_col].iloc[-lag] if lag <= len(history_df) else latest.get(base_col, 0))
            else:
                feature_dict[col] = float(latest.get(col, 0.0))
        
        feat_vec = np.array([feature_dict.get(c, 0.0) for c in feature_cols]).reshape(1, -1)
        pred_aqi = max(0.0, min(500.0, float(model.predict(scaler.transform(feat_vec))[0])))
        cat = aqi_category(pred_aqi)
        
        predictions.append({
            "timestamp": future_dt.isoformat(),
            "city": CITY,
            "forecast_hour": h,
            "predicted_aqi": round(pred_aqi, 1),
            "aqi_category": cat["label"],
            "aqi_color": cat["color"],
            "is_alert": bool(pred_aqi >= 150),
            "created_at": now.isoformat()
        })
    return predictions

# ── Terminal Summary ────────────────────────────────────
def print_summary(predictions: list):
    print(f"\n📅 3-Day Forecast for {CITY.capitalize()}:")
    print(f"{'Hour':<6} | {'AQI':<8} | {'Category'}")
    print("-" * 35)
    # Print sample: first, every 12th, and last
    for i in [0, 11, 23, 35, 47, 59, 71]:
        p = predictions[i]
        print(f"{p['forecast_hour']:<6} | {p['predicted_aqi']:<8} | {p['aqi_category']}")

# ── Main Execution ──────────────────────────────────────
if __name__ == "__main__":
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    model, scaler, feature_cols = load_model()
    
    history_data = supabase.table("aqi_features").select("*").eq("city", CITY).order("timestamp", desc=True).limit(24).execute()
    history_df = pd.DataFrame(history_data.data).sort_values("timestamp")
    
    preds = generate_forecast(model, scaler, feature_cols, history_df)
    
    supabase.table("aqi_predictions").delete().eq("city", CITY).execute()
    supabase.table("aqi_predictions").insert(preds).execute()
    
    print("✅ Successfully stored 72-hour forecast in Supabase.")
    print_summary(preds)
    print("\n🎉 Inference pipeline complete!")