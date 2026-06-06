# 🌫️ AQI Predictor — Karachi Air Quality Forecasting System

> A production-grade, end-to-end MLOps pipeline that collects real-time air pollution data, trains machine learning models, and generates 72-hour AQI forecasts for Karachi, Pakistan — fully automated via GitHub Actions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Pipelines](#pipelines)
  - [Backfill Pipeline](#1-backfill-pipeline)
  - [Feature Pipeline](#2-feature-pipeline)
  - [Training Pipeline](#3-training-pipeline)
  - [Inference Pipeline](#4-inference-pipeline)
- [Machine Learning](#machine-learning)
- [Database Schema](#database-schema)
- [CI/CD Automation](#cicd-automation)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [Dashboard](#dashboard)
- [AQI Reference Guide](#aqi-reference-guide)
- [Tech Stack](#tech-stack)

---

## Overview

**AQI Predictor** is a fully automated air quality intelligence system built specifically for Karachi. It ingests live pollution and weather data every hour, retrains ML models nightly, and publishes 72-hour forecasts — all without manual intervention.

### Key Features

- **72-hour AQI forecast** using lag-feature time-series ML
- **Hourly data ingestion** from OpenWeatherMap Air Pollution API
- **3 competing ML models** evaluated and best one auto-selected (Ridge, Random Forest, MLP)
- **SHAP explainability** — understand which features drive each prediction
- **Automated CI/CD** — GitHub Actions runs all pipelines on schedule
- **Supabase backend** — PostgreSQL database + file storage for model artifacts
- **Outlier guardrails** — invalid AQI readings are blocked before storage
- **Rolling mean smoothing** — reduces noise for stable predictions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                            │
│  OpenWeatherMap Air Pollution API  │  OWM Weather API       │
└────────────────────┬────────────────────────────────────────┘
                     │ (hourly)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               FEATURE PIPELINE  (GitHub Actions)            │
│  • Fetch PM2.5, PM10, O3, NO2, CO, SO2                     │
│  • Compute AQI from PM2.5 breakpoints                       │
│  • Engineer rolling mean & change rate features             │
│  • Outlier filter (AQI 30–350)                              │
│  • Upsert to Supabase → aqi_features table                  │
└────────────────────┬────────────────────────────────────────┘
                     │ (triggers after feature pipeline)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               TRAINING PIPELINE  (nightly 2 AM)             │
│  • Fetch all historical data from Supabase                  │
│  • Build lag features (1h, 3h, 6h, 12h, 24h)               │
│  • Train Ridge + Random Forest + MLP                        │
│  • Select best model by RMSE                                │
│  • Compute SHAP feature importances                         │
│  • Save model artifacts to Supabase Storage                 │
└────────────────────┬────────────────────────────────────────┘
                     │ (hourly, after feature pipeline)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               INFERENCE PIPELINE  (GitHub Actions)          │
│  • Download latest model from Supabase Storage              │
│  • Load last 24h of features                                │
│  • Generate 72-hour hourly AQI forecast                     │
│  • Assign AQI category labels & alert flags                 │
│  • Store predictions → aqi_predictions table                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               APP LAYER                                     │
│  FastAPI REST API  │  Streamlit Dashboard                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
AQI-PREDICTOR/
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml      # Runs hourly (data ingestion)
│       ├── inference_pipeline.yml    # Runs after feature pipeline
│       └── training_pipeline.yml     # Runs nightly at 2 AM
│
├── app/
│   ├── __init__.py
│   ├── api.py                        # FastAPI REST endpoints
│   └── app.py                        # Streamlit dashboard
│
├── models/
│   ├── best_model.pkl                # Serialized best-performing model
│   ├── scaler.pkl                    # StandardScaler for feature normalization
│   ├── feature_cols.json             # Ordered list of feature column names
│   └── shap_values.json              # SHAP importance scores
│
├── notebooks/
│   └── AQI_EDA.ipynb                 # Exploratory Data Analysis
│
├── pipelines/
│   ├── __init__.py
│   ├── backfill_pipeline.py          # One-time 90-day historical data load
│   ├── feature_pipeline.py           # Hourly data ingestion & feature engineering
│   ├── inference_pipeline.py         # 72-hour AQI forecast generation
│   └── training_pipeline.py          # Model training, evaluation & registration
│
├── .gitignore
├── env.example                       # Template for required environment variables
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Data Sources

| Source | Data | Endpoint |
|--------|------|----------|
| OpenWeatherMap Air Pollution API | PM2.5, PM10, O3, NO2, CO, SO2 | `/data/2.5/air_pollution` |
| OpenWeatherMap Weather API | Temperature, Humidity, Pressure, Wind Speed | `/data/2.5/weather` |
| Karachi Climate Averages | Monthly weather fallback for backfill | Static lookup table (built-in) |

**Location:** Latitude `24.8607`, Longitude `67.0011` (Karachi, Pakistan)

---

## Pipelines

### 1. Backfill Pipeline

**File:** `pipelines/backfill_pipeline.py`  
**Purpose:** One-time setup to populate 90 days of historical AQI data.

**What it does:**
- Fetches 90 days of OpenWeatherMap historical air pollution data in 30-day chunks
- Uses built-in Karachi monthly climate averages as weather fallback (since historical weather costs extra API calls)
- Converts PM2.5 readings to AQI using EPA breakpoints
- Applies outlier filtering (AQI range: 30–350)
- Computes a 3-point rolling mean for trend smoothing
- Bulk-upserts all cleaned rows to Supabase

**Run once before anything else:**
```bash
python pipelines/backfill_pipeline.py
```

---

### 2. Feature Pipeline

**File:** `pipelines/feature_pipeline.py`  
**Schedule:** Every hour via GitHub Actions (`0 * * * *`)

**What it does:**
- Fetches the latest air pollution reading from OpenWeatherMap
- Fetches current weather (temperature, humidity, pressure, wind)
- Computes AQI from PM2.5 using EPA breakpoints
- Calculates `aqi_change_rate` (difference from last stored AQI)
- Computes `aqi_rolling_mean` (average of current + previous AQI)
- Applies a 0.70 calibration factor to the raw AQI
- Validates the reading (blocks outliers outside 30–350)
- Upserts the feature row to `aqi_features` table

**Feature engineering note:**
```python
# AQI calibration applied in live pipeline
aqi = round(aqi * 0.70, 1)

# Rolling mean (2-hour window for live data)
aqi_rolling_mean = (aqi + last_aqi) / 2 if last_aqi else aqi
```

---

### 3. Training Pipeline

**File:** `pipelines/training_pipeline.py`  
**Schedule:** Nightly at 2:00 AM UTC via GitHub Actions

**What it does:**

1. **Data fetch** — Pulls all historical rows from `aqi_features` (paginated, handles large datasets)
2. **Feature engineering** — Creates lag features at 1h, 3h, 6h, 12h, 24h intervals for: `aqi`, `pm25`, `pm10`, `o3`, `no2`, `aqi_change_rate`
3. **Target creation** — `target_aqi_24h` (24-hour ahead AQI)
4. **Train/test split** — 80/20 random split
5. **Model training** — Trains 3 models in parallel:

| Model | Hyperparameters |
|-------|----------------|
| Ridge Regression | `alpha=1.0` |
| Random Forest | `n_estimators=200`, `max_depth=15` |
| MLP Neural Network | `layers=(128,64,32)`, `relu`, early stopping |

6. **Evaluation** — Compares RMSE, MAE, R² on held-out test set
7. **Best model selection** — Lowest RMSE wins
8. **SHAP explainability** — Computes mean absolute SHAP values for top-5 features
9. **Artifact saving** — Saves `best_model.pkl`, `scaler.pkl`, `feature_cols.json`, `shap_values.json`
10. **Model registry** — Logs version, metrics, and timestamp to `model_registry` Supabase table
11. **Cloud upload** — Uploads all model files to Supabase Storage bucket

---

### 4. Inference Pipeline

**File:** `pipelines/inference_pipeline.py`  
**Schedule:** 30 minutes after each feature pipeline run (`30 * * * *`)

**What it does:**
- Downloads latest model artifacts from Supabase Storage
- Loads last 24 hours of feature data as context
- Generates 72-hour (3-day) hourly AQI forecast
- For each future hour:
  - Constructs feature vector using lag lookups from history
  - Scales features with loaded StandardScaler
  - Predicts AQI and clips to valid range (0–500)
  - Assigns AQI category label and color
  - Flags alerts for readings ≥ 150
- Clears previous predictions and inserts fresh 72-hour forecast
- Logs a summary table to console

---

## Machine Learning

### Feature Set

| Feature | Description |
|---------|-------------|
| `aqi_lag_1h` to `aqi_lag_24h` | AQI values from 1, 3, 6, 12, 24 hours ago |
| `pm25_lag_*` | PM2.5 concentration lags |
| `pm10_lag_*` | PM10 concentration lags |
| `o3_lag_*` | Ozone lags |
| `no2_lag_*` | Nitrogen dioxide lags |
| `aqi_change_rate_lag_*` | Rate of AQI change lags |
| `hour` | Hour of day (0–23) |
| `day` | Day of week (0=Monday) |
| `month` | Month of year (1–12) |

### AQI Computation (EPA Standard)

```
PM2.5 → AQI via piecewise linear interpolation:

PM2.5 Range      AQI Range
0.0  – 12.0   →  0  – 50    (Good)
12.1 – 35.4   →  51 – 100   (Moderate)
35.5 – 55.4   →  101 – 150  (Unhealthy for Sensitive Groups)
55.5 – 150.4  →  151 – 200  (Unhealthy)
150.5 – 250.4 →  201 – 300  (Very Unhealthy)
250.5 – 350.4 →  301 – 400  (Hazardous)
350.5 – 500.4 →  401 – 500  (Hazardous+)
```

### Model Evaluation Metrics

- **RMSE** (Root Mean Squared Error) — primary selection criterion
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

---

## Database Schema

### `aqi_features` table

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | timestamptz | UTC timestamp (PK with city) |
| `city` | text | City name (e.g. "karachi") |
| `aqi` | float | Computed AQI value |
| `aqi_rolling_mean` | float | Smoothed AQI (rolling average) |
| `aqi_change_rate` | float | Delta from previous reading |
| `pm25` | float | PM2.5 concentration (µg/m³) |
| `pm10` | float | PM10 concentration (µg/m³) |
| `o3` | float | Ozone (µg/m³) |
| `no2` | float | Nitrogen dioxide (µg/m³) |
| `co` | float | Carbon monoxide (µg/m³) |
| `so2` | float | Sulfur dioxide (µg/m³) |
| `temperature` | float | Temperature (°C) |
| `humidity` | float | Relative humidity (%) |
| `pressure` | float | Atmospheric pressure (hPa) |
| `wind` | float | Wind speed (m/s) |
| `hour` | int | Hour of day |
| `day` | int | Day of week |
| `month` | int | Month of year |

### `aqi_predictions` table

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | timestamptz | Forecasted hour (UTC) |
| `city` | text | City name |
| `forecast_hour` | int | Hours ahead (1–72) |
| `predicted_aqi` | float | Model's AQI prediction |
| `aqi_category` | text | Human-readable category |
| `aqi_color` | text | Hex color code for UI |
| `is_alert` | bool | True if AQI ≥ 150 |
| `created_at` | timestamptz | When forecast was generated |

### `model_registry` table

| Column | Type | Description |
|--------|------|-------------|
| `model_name` | text | Algorithm name |
| `version` | int | Auto-incremented version |
| `rmse` | float | Test set RMSE |
| `mae` | float | Test set MAE |
| `r2` | float | Test set R² |
| `model_path` | text | Local path to model file |
| `trained_at` | timestamptz | Training timestamp |
| `horizon` | int | Forecast horizon in hours |

---

## CI/CD Automation

All pipelines are automated via GitHub Actions. No server required.

### Workflow Schedule

```
Feature Pipeline    ──►  Every hour on the hour       (0 * * * *)
                              │
                              ▼ (triggers on success)
Inference Pipeline  ──►  30 min past every hour       (30 * * * *)

Training Pipeline   ──►  Every night at 2:00 AM UTC   (0 2 * * *)
```

### Setting Up GitHub Secrets

In your GitHub repository: **Settings → Secrets and variables → Actions**

Add the following secrets:

| Secret Name | Value |
|-------------|-------|
| `OPENWEATHER_TOKEN` | Your OpenWeatherMap API key |
| `AQICN_TOKEN` | Your AQICN API token |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Your Supabase service role key |

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- A [Supabase](https://supabase.com) project (free tier works)
- An [OpenWeatherMap](https://openweathermap.org/api) API key (free tier)
- A GitHub repository (for Actions automation)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/aqi-predictor.git
cd aqi-predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp env.example .env
# Edit .env with your credentials
```

### 3. Set Up Supabase Tables

Create the following tables in your Supabase SQL editor:

```sql
-- Feature store
CREATE TABLE aqi_features (
    id            BIGSERIAL PRIMARY KEY,
    timestamp     TIMESTAMPTZ NOT NULL,
    city          TEXT NOT NULL,
    aqi           FLOAT,
    aqi_rolling_mean FLOAT,
    aqi_change_rate  FLOAT,
    pm25          FLOAT, pm10 FLOAT,
    o3            FLOAT, no2 FLOAT,
    co            FLOAT, so2 FLOAT,
    temperature   FLOAT, humidity FLOAT,
    pressure      FLOAT, wind FLOAT,
    hour          INT, day INT, month INT,
    UNIQUE(timestamp, city)
);

-- Predictions store
CREATE TABLE aqi_predictions (
    id            BIGSERIAL PRIMARY KEY,
    timestamp     TIMESTAMPTZ NOT NULL,
    city          TEXT NOT NULL,
    forecast_hour INT,
    predicted_aqi FLOAT,
    aqi_category  TEXT,
    aqi_color     TEXT,
    is_alert      BOOLEAN,
    created_at    TIMESTAMPTZ
);

-- Model registry
CREATE TABLE model_registry (
    id          BIGSERIAL PRIMARY KEY,
    model_name  TEXT,
    version     INT,
    rmse        FLOAT,
    mae         FLOAT,
    r2          FLOAT,
    model_path  TEXT,
    trained_at  TIMESTAMPTZ,
    horizon     INT
);
```

### 4. Create Supabase Storage Bucket

In Supabase Dashboard → Storage → Create bucket named `models` (set to private).

### 5. Run Backfill (First Time Only)

```bash
python pipelines/backfill_pipeline.py
```

This fetches 90 days of historical data (~2,000 rows).

### 6. Run Training

```bash
python pipelines/training_pipeline.py
```

### 7. Run Inference

```bash
python pipelines/inference_pipeline.py
```

### 8. Start the App

```bash
# FastAPI backend
uvicorn app.api:app --reload --port 8000

# Streamlit dashboard (separate terminal)
streamlit run app/app.py
```

---

## Environment Variables

Create a `.env` file based on `env.example`:

```env
# OpenWeatherMap (https://openweathermap.org/api)
OPENWEATHER_TOKEN=your_openweather_api_key

# AQICN (https://aqicn.org/data-platform/token/)
AQICN_TOKEN=your_aqicn_token

# Supabase (https://supabase.com → Project Settings → API)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your_service_role_secret_key
```

> ⚠️ **Never commit your `.env` file.** It is already in `.gitignore`.

---

## API Endpoints

The FastAPI backend (`app/api.py`) exposes the following endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/current` | Latest AQI reading for Karachi |
| `GET` | `/forecast` | Full 72-hour AQI forecast |
| `GET` | `/forecast/{hours}` | Forecast for next N hours (1–72) |
| `GET` | `/history` | Recent historical AQI readings |
| `GET` | `/model/info` | Latest model version & metrics |
| `GET` | `/model/shap` | SHAP feature importance scores |

---

## Dashboard

The Streamlit dashboard (`app/app.py`) provides:

- **Live AQI gauge** with category color coding
- **72-hour forecast chart** (interactive Plotly)
- **Historical trend** with rolling mean overlay
- **Alert panel** highlighting dangerous forecast hours
- **SHAP feature importance** bar chart
- **Model metadata** (version, RMSE, last trained)

---

## AQI Reference Guide

| AQI Range | Category | Color | Health Implication |
|-----------|----------|-------|-------------------|
| 0 – 50 | Good | 🟢 `#00e400` | Air quality is satisfactory |
| 51 – 100 | Moderate | 🟡 `#ffff00` | Acceptable; some pollutants may concern sensitive people |
| 101 – 150 | Unhealthy for Sensitive Groups | 🟠 `#ff7e00` | Sensitive groups may experience effects |
| 151 – 200 | Unhealthy | 🔴 `#ff0000` | Everyone may begin to experience effects |
| 201 – 300 | Very Unhealthy | 🟣 `#8f3f97` | Health alert — everyone may experience serious effects |
| 301 – 500 | Hazardous | 🟤 `#7e0023` | Emergency conditions; entire population at risk |

> **Alert threshold:** Predictions with AQI ≥ 150 are flagged as `is_alert = true`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11 |
| **Data Ingestion** | OpenWeatherMap API, Requests |
| **Data Storage** | Supabase (PostgreSQL + Storage) |
| **Feature Engineering** | Pandas, NumPy |
| **ML Models** | scikit-learn (Ridge, RandomForest, MLP) |
| **Model Serialization** | joblib |
| **Explainability** | SHAP |
| **API Backend** | FastAPI + Uvicorn |
| **Dashboard** | Streamlit + Plotly |
| **Automation** | GitHub Actions |
| **Environment** | python-dotenv |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with ❤️ for Karachi — because clean air data should be accessible to everyone.*
