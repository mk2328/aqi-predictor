import os
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

import shap
import json

# ── Config ──────────────────────────────────────────────
SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_KEY"]
CITY          = "karachi"
MODEL_PATH    = "models/best_model.pkl"
SCALER_PATH   = "models/scaler.pkl"
SHAP_PATH     = "models/shap_values.json"

FORECAST_HORIZON = 72  # 3 days ahead

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


# ── Prepare Forecasting Data with Lags ──────────────────
def prepare_forecasting_data(df: pd.DataFrame, horizon=72):
    df = df.sort_values("timestamp").copy().reset_index(drop=True)
    df = df.dropna(subset=["aqi"]).reset_index(drop=True)
    
    # Lagged features
    lag_features = ["aqi", "pm25", "pm10", "o3", "no2", "aqi_change_rate"]
    for lag in [1, 3, 6, 12, 24]:
        for col in lag_features:
            if col in df.columns:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
    
    # Future target
    df[f'target_aqi_{horizon}h'] = df['aqi'].shift(-horizon)
    
    df = df.dropna(subset=[f'target_aqi_{horizon}h']).reset_index(drop=True)
    
    # Feature columns
    lag_cols = [col for col in df.columns if '_lag_' in col]
    feature_cols = lag_cols + ["hour", "day", "month"]
    
    X = df[feature_cols].fillna(0).values
    y = df[f'target_aqi_{horizon}h'].values
    
    print(f"📊 Forecasting Dataset: {len(df)} samples | Horizon: {horizon}h | Features: {len(feature_cols)}")
    return X, y, feature_cols


# ── Evaluate ────────────────────────────────────────────
def evaluate(name, model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"   {name:35s} | RMSE={rmse:.2f} | MAE={mae:.2f} | R²={r2:.4f}")
    return {"model_name": name, "rmse": rmse, "mae": mae, "r2": r2, "model": model}


# ── Train Models ────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test) -> list:
    print("\n🤖 Training models...")
    results = []

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results.append(evaluate("Ridge Regression", ridge, X_test, y_test))

    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        learning_rate_init=0.001
    )
    mlp.fit(X_train, y_train)
    results.append(evaluate("Neural Network (MLP)", mlp, X_test, y_test))

    return results


# ── Robust SHAP Function ────────────────────────────────
def compute_shap(best: dict, X_train, X_test, feature_cols) -> dict:
    print("\n🔍 Computing SHAP feature importances...")
    model = best["model"]
    
    try:
        if isinstance(model, RandomForestRegressor):
            print("   Using TreeExplainer for Random Forest...")
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_vals = explainer.shap_values(X_test, check_additivity=False)
            if len(np.array(shap_vals).shape) == 3:
                shap_vals = np.mean(shap_vals, axis=2)
        else:
            explainer = shap.Explainer(model, X_train)
            shap_vals = explainer(X_test).values

        mean_abs = np.abs(shap_vals).mean(axis=0).tolist()
        top5 = sorted(zip(feature_cols, mean_abs), key=lambda x: x[1], reverse=True)[:5]

        shap_dict = {
            "feature_names": feature_cols,
            "mean_abs_shap": mean_abs,
            "top_features": [
                {"feature": f, "importance": round(v, 4)} for f, v in top5
            ]
        }
        print("   Top features:", [f["feature"] for f in shap_dict["top_features"]])
        return shap_dict

    except Exception as e:
        print(f"   ⚠️ SHAP failed: {e}. Using fallback...")
        try:
            if isinstance(model, RandomForestRegressor):
                imp = model.feature_importances_.tolist()
            elif hasattr(model, "coef_"):
                imp = np.abs(model.coef_).mean(axis=0).tolist() if len(model.coef_.shape) > 1 else np.abs(model.coef_).tolist()
            else:
                imp = [0.0] * len(feature_cols)
        except:
            imp = [0.0] * len(feature_cols)

        top5 = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)[:5]
        return {
            "feature_names": feature_cols,
            "mean_abs_shap": imp,
            "top_features": [{"feature": f, "importance": round(v, 4)} for f, v in top5]
        }


# ── Save Best Model ─────────────────────────────────────
def save_best_model(supabase, best: dict, scaler, shap_dict: dict, version: int, feature_cols):
    os.makedirs("models", exist_ok=True)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(best["model"], MODEL_PATH)
    print(f"\n💾 Saved model : {MODEL_PATH}")
    print(f"💾 Saved scaler: {SCALER_PATH}")

    # Save feature columns for inference
    with open("models/feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(SHAP_PATH, "w") as f:
        json.dump(shap_dict, f, indent=2)
    print(f"💾 Saved SHAP  : {SHAP_PATH}")

    supabase.table("model_registry").insert({
        "model_name": best["model_name"],
        "version":    version,
        "rmse":       round(best["rmse"], 4),
        "mae":        round(best["mae"],  4),
        "r2":         round(best["r2"],   4),
        "model_path": MODEL_PATH,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "horizon":    FORECAST_HORIZON
    }).execute()
    print(f"✅ Model v{version} (Horizon: {FORECAST_HORIZON}h) registered in Supabase!")


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    df = fetch_features(supabase)
    if len(df) < 100:
        print("❌ Not enough data. Run backfill_pipeline.py first!")
        exit(1)

    X, y, FEATURE_COLS = prepare_forecasting_data(df, horizon=FORECAST_HORIZON)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    results = train_models(X_train, X_test, y_train, y_test)

    best = min(results, key=lambda x: x["rmse"])
    print(f"\n🏆 Best model: {best['model_name']} (RMSE={best['rmse']:.2f})")

    shap_dict = compute_shap(best, X_train, X_test, FEATURE_COLS)

    existing = (
        supabase.table("model_registry")
        .select("version")
        .order("version", desc=True)
        .limit(1)
        .execute()
    )
    version = (existing.data[0]["version"] + 1) if existing.data else 1

    save_best_model(supabase, best, scaler, shap_dict, version, FEATURE_COLS)
    print(f"\n🎉 Training pipeline complete! Version {version} saved.")