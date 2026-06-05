"""
app/streamlit_app.py — Beautiful Light Theme, Sharp Colors, High Contrast
"""

import os
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AQI Predictor — Karachi",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg:           #f5f7fa;
    --bg-white:     #ffffff;
    --bg-soft:      #eef1f6;
    --border:       #e2e6ed;
    --text-1:       #0d1117;
    --text-2:       #4a5568;
    --text-3:       #8896aa;

    /* AQI palette — vivid, saturated */
    --blue:         #1d7afc;
    --teal:         #00bfa5;
    --amber:        #f59e0b;
    --coral:        #f45c5c;
    --violet:       #7c3aed;

    --font:         'Outfit', sans-serif;
    --mono:         'Space Mono', monospace;
    --r:            14px;
}

html, body, [class*="css"] {
    font-family: var(--font) !important;
    background: var(--bg) !important;
    color: var(--text-1) !important;
}
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 5rem !important;
    max-width: 1300px !important;
}

/* ── TOP BAR ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1.25rem;
    border-bottom: 2px solid var(--border);
}
.topbar-title {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--text-1);
    letter-spacing: -0.5px;
    line-height: 1;
}
.topbar-title span { color: var(--blue); }
.topbar-sub {
    font-size: 0.8rem;
    color: var(--text-3);
    font-weight: 500;
    letter-spacing: 0.05em;
    margin-top: 4px;
}
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #e8f5e9;
    color: #1b7a3e;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 7px 16px;
    border-radius: 100px;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    border: 1.5px solid #a5d6a7;
}
.live-dot {
    width: 7px; height: 7px;
    background: #2e7d32;
    border-radius: 50%;
    animation: blink 1.8s infinite;
}
@keyframes blink {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.3; }
}

/* ── AQI HERO ── */
.hero {
    background: var(--bg-white);
    border: 1.5px solid var(--border);
    border-radius: 20px;
    padding: 2rem 2.25rem;
    display: grid;
    grid-template-columns: 220px 1fr 200px;
    gap: 2rem;
    align-items: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 20px rgba(0,0,0,0.05);
}
.hero-aqi-num {
    font-family: var(--mono);
    font-size: 5.8rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -3px;
}
.hero-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 6px;
}
.hero-badge {
    display: inline-block;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 5px 16px;
    border-radius: 100px;
    margin-top: 10px;
    letter-spacing: 0.04em;
}
.hero-metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
.metric-tile {
    background: var(--bg-soft);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
}
.metric-tile-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 5px;
}
.metric-tile-val {
    font-family: var(--mono);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-1);
}
.metric-tile-unit {
    font-size: 0.72rem;
    color: var(--text-3);
    font-weight: 400;
    margin-left: 3px;
}
.hero-right {
    text-align: right;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-3);
    line-height: 1.9;
}
.hero-right strong {
    color: var(--text-2);
    font-weight: 600;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-3);
    margin: 0 0 1rem 2px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1.5px;
    background: var(--border);
}

/* ── DAY CARDS ── */
.day-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 2rem;
}
.day-card {
    background: var(--bg-white);
    border: 1.5px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 1.25rem 1.25rem;
    text-align: center;
    transition: box-shadow 0.2s, transform 0.2s;
}
.day-card:hover {
    box-shadow: 0 8px 30px rgba(0,0,0,0.09);
    transform: translateY(-3px);
}
.day-card-date {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 10px;
}
.day-card-num {
    font-family: var(--mono);
    font-size: 3.6rem;
    font-weight: 700;
    letter-spacing: -2px;
    line-height: 1;
    margin-bottom: 10px;
}
.day-card-badge {
    display: inline-block;
    font-size: 0.73rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 100px;
    letter-spacing: 0.04em;
    margin-bottom: 12px;
}
.day-card-range {
    font-family: var(--mono);
    font-size: 0.73rem;
    color: var(--text-3);
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-white) !important;
    border-radius: 12px 12px 0 0 !important;
    border: 1.5px solid var(--border) !important;
    border-bottom: none !important;
    gap: 0 !important;
    padding: 5px 6px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--text-3) !important;
    background: transparent !important;
    padding: 8px 22px !important;
    border-radius: 8px !important;
    border: none !important;
    transition: all 0.15s !important;
}
.stTabs [aria-selected="true"] {
    background: var(--blue) !important;
    color: #fff !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-white) !important;
    border: 1.5px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 1.5rem !important;
}

/* ── MODEL STRIP ── */
.model-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 2rem;
}
.model-tile {
    background: var(--bg-white);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
}
.model-tile-label {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 5px;
}
.model-tile-val {
    font-family: var(--mono);
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--blue);
}

/* ── EMPTY STATES ── */
.empty {
    padding: 2.5rem;
    text-align: center;
    color: var(--text-3);
    font-size: 0.85rem;
    background: var(--bg-soft);
    border-radius: 12px;
    border: 1.5px dashed var(--border);
}

/* ── FOOTER ── */
.footer {
    border-top: 1.5px solid var(--border);
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: var(--text-3);
    font-weight: 500;
}

/* hide Streamlit built-ins */
.stMetric, div[data-testid="stDivider"],
div[data-testid="stCaption"] p,
div[data-testid="stSubheader"] h3 { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch(endpoint):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def chart_layout(fig, title="", height=420):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(family="Outfit", size=14, color="#0d1117"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit", color="#8896aa", size=12),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(
            showgrid=True, gridcolor="#eef1f6", gridwidth=1,
            zeroline=False, linecolor="#e2e6ed",
            tickfont=dict(size=11, color="#8896aa"),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#eef1f6", gridwidth=1,
            zeroline=False, linecolor="#e2e6ed",
            tickfont=dict(size=11, color="#8896aa"),
        ),
        hoverlabel=dict(
            bgcolor="#0d1117",
            bordercolor="#1d7afc",
            font=dict(family="Space Mono, monospace", size=12, color="#ffffff"),
        ),
    )
    return fig


# ── Fetch ──────────────────────────────────────────────────────────────────────
current  = fetch("/current")
forecast = fetch("/forecast")
daily    = fetch("/forecast/daily")
history  = fetch("/history?hours=48")
shap     = fetch("/shap")
model    = fetch("/model/info")

now_str = datetime.now().strftime("%d %b %Y  %H:%M")

# ── Top Bar ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <div>
        <div class="topbar-title">AQI <span>Dashboard</span> — Karachi</div>
        <div class="topbar-sub">Real-time Air Quality Monitor &amp; 72-Hour Intelligent Forecast</div>
    </div>
    <div class="live-pill"><div class="live-dot"></div>Live</div>
</div>
""", unsafe_allow_html=True)


# ── Current AQI Hero ───────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Current Conditions</div>', unsafe_allow_html=True)

if current:
    aqi   = float(current.get("aqi", 0))
    cat   = current.get("aqi_category", "Unknown")
    color = current.get("aqi_color", "#888888")
    pm25  = current.get("pm25", 0)
    pm10  = current.get("pm10", 0)
    temp  = current.get("temperature", 0)
    hum   = current.get("humidity", 0)
    wind  = current.get("wind", 0)
    ts    = (current.get("timestamp") or "")[:16].replace("T", "  ")

    st.markdown(f"""
    <div class="hero">
        <div>
            <div class="hero-label">AQI Index</div>
            <div class="hero-aqi-num" style="color:{color};">{int(aqi)}</div>
            <div class="hero-badge" style="background:{color}1a; color:{color}; border:2px solid {color}55;">{cat}</div>
        </div>
        <div class="hero-metrics">
            <div class="metric-tile">
                <div class="metric-tile-label">PM 2.5</div>
                <div class="metric-tile-val">{pm25:.1f}<span class="metric-tile-unit">µg/m³</span></div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">PM 10</div>
                <div class="metric-tile-val">{pm10:.1f}<span class="metric-tile-unit">µg/m³</span></div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Temperature</div>
                <div class="metric-tile-val">{temp:.1f}<span class="metric-tile-unit">°C</span></div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Humidity</div>
                <div class="metric-tile-val">{hum:.0f}<span class="metric-tile-unit">%</span></div>
            </div>
        </div>
        <div class="hero-right">
            <strong>Last Updated</strong><br>{ts} UTC<br><br>
            <strong>Wind Speed</strong><br>{wind:.1f} m/s<br><br>
            <span style="color:#c8d0db;">Karachi, Pakistan</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="empty">📡 Current data unavailable — make sure the API and feature pipeline are running.</div>', unsafe_allow_html=True)


# ── 3-Day Forecast ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">3-Day Outlook</div>', unsafe_allow_html=True)

if daily and daily.get("daily"):
    cards_html = '<div class="day-grid">'
    for day in daily["daily"]:
        c = day.get("color", "#888")
        try:
            date_label = datetime.strptime(day["date"], "%Y-%m-%d").strftime("%a, %d %b")
        except Exception:
            date_label = day["date"]
        cards_html += f"""
        <div class="day-card">
            <div class="day-card-date">{date_label}</div>
            <div class="day-card-num" style="color:{c};">{int(day['avg_aqi'])}</div>
            <div class="day-card-badge" style="background:{c}18; color:{c}; border:1.5px solid {c}55;">{day['category']}</div>
            <div class="day-card-range">↓ {day.get('min_aqi',0):.0f} &nbsp;–&nbsp; ↑ {day.get('max_aqi',0):.0f}</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)
else:
    st.markdown('<div class="empty">📅 Run <code>inference_pipeline.py</code> to generate the forecast.</div>', unsafe_allow_html=True)


# ── Charts Tabs ────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Analytics</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "  📈  72-Hour Forecast  ",
    "  🕒  History (48h)  ",
    "  🔍  Feature Importance  ",
    "  📌  EDA Insights  ",
])

with tab1:
    if forecast and forecast.get("predictions"):
        df = pd.DataFrame(forecast["predictions"])
        df["dt"] = pd.to_datetime(df["timestamp"])

        fig = go.Figure()
        # Fill area
        fig.add_trace(go.Scatter(
            x=df["dt"], y=df["predicted_aqi"],
            fill="tozeroy", fillcolor="rgba(29,122,252,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        # Main line — vivid blue
        fig.add_trace(go.Scatter(
            x=df["dt"], y=df["predicted_aqi"],
            mode="lines",
            line=dict(color="#1d7afc", width=3),
            name="Predicted AQI",
            hovertemplate="<b>%{x|%d %b %H:%M}</b><br>AQI  %{y:.0f}<extra></extra>",
        ))
        # Dot markers every 6h
        df6 = df.iloc[::6]
        fig.add_trace(go.Scatter(
            x=df6["dt"], y=df6["predicted_aqi"],
            mode="markers",
            marker=dict(size=6, color="#1d7afc", line=dict(color="#fff", width=2)),
            showlegend=False, hoverinfo="skip",
        ))
        chart_layout(fig, "72-Hour Hourly AQI Forecast")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div class="empty">No forecast data yet.</div>', unsafe_allow_html=True)

with tab2:
    if history and history.get("readings"):
        dfh = pd.DataFrame(history["readings"])
        dfh["dt"] = pd.to_datetime(dfh["timestamp"])

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dfh["dt"], y=dfh["aqi"],
            fill="tozeroy", fillcolor="rgba(0,191,165,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig2.add_trace(go.Scatter(
            x=dfh["dt"], y=dfh["aqi"],
            mode="lines",
            line=dict(color="#00bfa5", width=3),
            name="Actual AQI",
            hovertemplate="<b>%{x|%d %b %H:%M}</b><br>AQI  %{y:.0f}<extra></extra>",
        ))
        chart_layout(fig2, "Last 48 Hours — Actual AQI Readings")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div class="empty">No history data yet.</div>', unsafe_allow_html=True)

with tab3:
    if shap and shap.get("feature_names"):
        labels = [
            f.replace("_lag_", " (lag ") + ")" if "_lag_" in f else f
            for f in shap["feature_names"]
        ]
        df_s = pd.DataFrame({
            "Feature": labels,
            "Importance": shap["mean_abs_shap"]
        }).sort_values("Importance", ascending=True).tail(12)

        n = len(df_s)
        # Gradient: amber → coral → blue  (vivid)
        colors = []
        for i in range(n):
            t = i / max(n - 1, 1)
            if t < 0.5:
                r = int(245 + (244 - 245) * t * 2)
                g = int(158 + (92  - 158) * t * 2)
                b = int(11  + (92  - 11 ) * t * 2)
            else:
                r = int(244 + (29  - 244) * (t - 0.5) * 2)
                g = int(92  + (122 - 92 ) * (t - 0.5) * 2)
                b = int(92  + (252 - 92 ) * (t - 0.5) * 2)
            colors.append(f"rgb({r},{g},{b})")

        fig3 = go.Figure(go.Bar(
            x=df_s["Importance"],
            y=df_s["Feature"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
            hovertemplate="<b>%{y}</b><br>Importance  %{x:.4f}<extra></extra>",
        ))
        chart_layout(fig3, "Top Features — SHAP Importance")
        fig3.update_layout(bargap=0.28)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div class="empty">Run <code>training_pipeline.py</code> to generate SHAP values.</div>', unsafe_allow_html=True)

with tab4:
    st.markdown("""
    <div style="padding: 0.5rem 0; display:grid; grid-template-columns:1fr 1fr; gap:14px;">
        <div style="background:#f5f7fa; border:1.5px solid #e2e6ed; border-radius:12px; padding:18px 20px;">
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#8896aa;margin-bottom:8px;">Key Drivers</div>
            <div style="font-size:0.9rem;color:#0d1117;line-height:1.8;">
                🔵 &nbsp;PM2.5 &amp; PM10 are the strongest AQI drivers<br>
                🟠 &nbsp;Hour of day creates clear daily cycles<br>
                🟢 &nbsp;Month captures seasonal variation<br>
                🔴 &nbsp;Wind speed inversely affects AQI
            </div>
        </div>
        <div style="background:#f5f7fa; border:1.5px solid #e2e6ed; border-radius:12px; padding:18px 20px;">
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#8896aa;margin-bottom:8px;">Patterns Observed</div>
            <div style="font-size:0.9rem;color:#0d1117;line-height:1.8;">
                📊 &nbsp;AQI peaks in evening rush hours<br>
                📅 &nbsp;Weekday AQI consistently higher than weekends<br>
                🌫️ &nbsp;Winter months show highest pollution<br>
                💧 &nbsp;Humidity above 70% amplifies PM readings
            </div>
        </div>
    </div>
    <div style="margin-top:12px; background:#eff6ff; border:1.5px solid #bfdbfe; border-radius:12px; padding:14px 18px; font-size:0.85rem; color:#1e40af; font-weight:500;">
        📓 &nbsp;Full EDA with interactive charts is available in <code style="background:#dbeafe;padding:2px 7px;border-radius:5px;">notebooks/AQI_EDA.ipynb</code>
    </div>
    """, unsafe_allow_html=True)


# ── Model Info ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label" style="margin-top:2rem;">Model Registry</div>', unsafe_allow_html=True)

if model:
    st.markdown(f"""
    <div class="model-strip">
        <div class="model-tile">
            <div class="model-tile-label">Model</div>
            <div class="model-tile-val">{model.get("model_name", "—")}</div>
        </div>
        <div class="model-tile">
            <div class="model-tile-label">Version</div>
            <div class="model-tile-val">v{model.get("version", "—")}</div>
        </div>
        <div class="model-tile">
            <div class="model-tile-label">RMSE</div>
            <div class="model-tile-val">{model.get("rmse", 0):.2f}</div>
        </div>
        <div class="model-tile">
            <div class="model-tile-label">R² Score</div>
            <div class="model-tile-val">{model.get("r2", 0):.3f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="empty">🤖 No model registered yet — run the training pipeline.</div>', unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    <span>AQI Predictor &nbsp;·&nbsp; Karachi, Pakistan &nbsp;·&nbsp; OpenWeather + Supabase</span>
    <span>Auto-refreshes every 60s &nbsp;·&nbsp; {now_str}</span>
</div>
""", unsafe_allow_html=True)