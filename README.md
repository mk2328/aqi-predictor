# 🌫️ AQI Predictor — Karachi

End-to-end ML pipeline for 3-day Air Quality Index forecasting using a fully serverless stack.

---

## Architecture

```
OpenWeather API
      │
      ▼
feature_pipeline.py  ──→  Supabase (aqi_features table)
      │                           │
      │                           ▼
      │                  training_pipeline.py
      │                     Ridge + RandomForest
      │                     MLP (sklearn)
      │                     TensorFlow DNN  ◄── NEW
      │                     SHAP values     ◄── NEW
      │                           │
      │                     models/ (pkl + .keras)
      │                           │
      ▼                           ▼
inference_pipeline.py ◄──────────┘
      │   (72-hour iterative forecast)
      ▼
Supabase (aqi_predictions table)  ◄── NEW
      │
      ▼
app/api.py  (FastAPI)
      │
      ▼
app/streamlit_app.py  (Dashboard)  ◄── NEW
```

---

## Pipelines

| File | Frequency | Purpose |
|------|-----------|---------|
| `pipelines/backfill_pipeline.py` | Once | Fetch 90 days historical data |
| `pipelines/feature_pipeline.py` | Every hour | Fetch live weather + pollution data |
| `pipelines/training_pipeline.py` | Daily | Train Ridge/RF/MLP/TF, compute SHAP |
| `pipelines/inference_pipeline.py` | Every hour | Generate 72-hour AQI forecast |

---

## Setup

### 1. Supabase
Run `supabase/migrations/001_create_aqi_predictions.sql` in your Supabase SQL editor.

Your existing tables (`aqi_features`, `model_registry`) are unchanged.

### 2. Environment Variables

```bash
# .env (local) or GitHub Actions Secrets
OPENWEATHER_TOKEN=your_key
AQICN_TOKEN=your_key
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=your_service_role_key
API_BASE_URL=http://localhost:8000   # for Streamlit
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run pipelines in order

```bash
# First time only
python pipelines/backfill_pipeline.py

# Then
python pipelines/feature_pipeline.py
python pipelines/training_pipeline.py
python pipelines/inference_pipeline.py
```

### 5. Start the API + Dashboard

```bash
# Terminal 1 — API
uvicorn app.api:app --reload --port 8000

# Terminal 2 — Dashboard
streamlit run app/streamlit_app.py
```

---

## GitHub Actions (CI/CD)

| Workflow | Schedule | File |
|----------|----------|------|
| Feature Pipeline | Every hour | `.github/workflows/feature_pipeline.yml` |
| Training Pipeline | Daily | `.github/workflows/training_pipeline.yml` |
| Inference Pipeline | Every hour (+30 min offset) | `.github/workflows/inference_pipeline.yml` |

Add `OPENWEATHER_TOKEN`, `AQICN_TOKEN`, `SUPABASE_URL`, `SUPABASE_KEY` as GitHub repository secrets.

---

## Models

| Model | Type |
|-------|------|
| Ridge Regression | Linear (baseline) |
| Random Forest | Ensemble |
| Neural Network (MLP) | sklearn |
| **TensorFlow DNN** | Keras (128→64→32→1) |

Best model (lowest RMSE) is auto-selected and saved to `models/`.

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /current` | Latest real AQI snapshot |
| `GET /forecast` | 72-hour hourly predictions |
| `GET /forecast/daily` | 3-day daily summaries |
| `GET /history?hours=48` | Recent actual readings |
| `GET /shap` | SHAP feature importances |
| `GET /alerts` | Hazardous AQI alerts |
| `GET /model/info` | Latest model registry entry |

---

## Project Structure

```
AQI-PREDICTOR/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml
│       ├── training_pipeline.yml
│       └── inference_pipeline.yml   ← NEW
├── app/
│   ├── api.py                       ← UPDATED
│   └── streamlit_app.py             ← NEW
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── tf_model.keras               ← NEW
│   └── shap_values.json             ← NEW
├── notebooks/
│   └── AQI_EDA.ipynb
├── pipelines/
│   ├── backfill_pipeline.py
│   ├── feature_pipeline.py
│   ├── training_pipeline.py         ← UPDATED
│   └── inference_pipeline.py        ← NEW
├── supabase/
│   └── migrations/
│       └── 001_create_aqi_predictions.sql  ← NEW
├── .gitignore
├── README.md
└── requirements.txt                 ← UPDATED
```
