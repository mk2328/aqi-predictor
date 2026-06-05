import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from supabase import create_client

from sklearn.linear_model    import Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.neural_network  import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

# ── Config ──────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
CITY         = "karachi"
MODEL_PATH   = "best_model.pkl"
SCALER_PATH  = "scaler.pkl"

# ── Step 1: Fetch data from Supabase ────────────────────
def fetch_features(supabase) -> pd.DataFrame:
    print("📦 Fetching features from Supabase...")
    all_rows = []
    page     = 0
    limit    = 1000

    while True:
        result = (
            supabase.table("aqi_features")
            .select("*")
            .eq("city", CITY)
            .order("timestamp")
            .range(page * limit, (page + 1) * limit - 1)
            .execute()
        )
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < limit:
            break
        page += 1

    df = pd.DataFrame(all_rows)
    print(f"   Got {len(df)} rows")
    return df

# ── Step 2: Prepare features & targets ──────────────────
def prepare_data(df: pd.DataFrame):
    # Features we use
    feature_cols = [
        "pm25", "pm10", "o3", "no2", "co", "so2",
        "temperature", "humidity", "pressure", "wind",
        "hour", "day", "month", "aqi_change_rate"
    ]

    # Drop rows with missing aqi
    df = df.dropna(subset=["aqi"])

    # Fill missing values
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values
    y = df["aqi"].values

    return X, y, feature_cols

# ── Step 3: Evaluate a model ─────────────────────────────
def evaluate(name, model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"   {name:30s} | RMSE={rmse:.2f} | MAE={mae:.2f} | R²={r2:.4f}")
    return {"model_name": name, "rmse": rmse, "mae": mae, "r2": r2, "model": model}

# ── Step 4: Train all models ─────────────────────────────
def train_models(X_train, X_test, y_train, y_test) -> list:
    print("\n🤖 Training models...")
    results = []

    # 1. Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results.append(evaluate("Ridge Regression", ridge, X_test, y_test))

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    # 3. Neural Network (MLP)
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
    mlp.fit(X_train, y_train)
    results.append(evaluate("Neural Network (MLP)", mlp, X_test, y_test))

    return results

# ── Step 5: Save best model to Supabase ─────────────────
def save_best_model(supabase, best: dict, scaler, version: int):
    # Save model + scaler locally
    joblib.dump(best["model"], MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n💾 Saved model: {MODEL_PATH}")

    # Save metadata to Supabase model registry
    supabase.table("model_registry").insert({
        "model_name": best["model_name"],
        "version":    version,
        "rmse":       round(best["rmse"], 4),
        "mae":        round(best["mae"],  4),
        "r2":         round(best["r2"],   4),
        "model_path": MODEL_PATH,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }).execute()
    print(f"✅ Model registered in Supabase!")

# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Fetch data
    df = fetch_features(supabase)

    if len(df) < 50:
        print("❌ Not enough data to train. Run backfill first!")
        exit(1)

    # Prepare
    X, y, feature_cols = prepare_data(df)
    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Scale
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # Train
    results = train_models(X_train, X_test, y_train, y_test)

    # Pick best model (lowest RMSE)
    best = min(results, key=lambda x: x["rmse"])
    print(f"\n🏆 Best model: {best['model_name']} (RMSE={best['rmse']:.2f})")

    # Get version number
    existing = (
        supabase.table("model_registry")
        .select("version")
        .order("version", desc=True)
        .limit(1)
        .execute()
    )
    version = (existing.data[0]["version"] + 1) if existing.data else 1

    # Save
    save_best_model(supabase, best, scaler, version)

    print(f"\n🎉 Training pipeline complete! Version {version} saved.")