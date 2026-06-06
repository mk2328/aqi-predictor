import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
import os
from supabase import create_client

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Karachi AQI Intelligence",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

* { font-family: 'Space Grotesk', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 50%, #0a0e1a 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1526 !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Radio buttons */
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; }

/* All text white by default */
p, div, span, label, h1, h2, h3, h4 { color: #e2e8f0; }

/* Metric card */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2234 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    margin-bottom: 12px;
}
.metric-label {
    font-size: 0.72rem;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 4px 0;
}
.metric-sub {
    font-size: 0.72rem;
    color: #64748b !important;
    margin-top: 4px;
}

/* AQI Hero */
.aqi-hero {
    background: linear-gradient(135deg, #111827 0%, #1a2234 100%);
    border-radius: 20px;
    padding: 32px 24px;
    text-align: center;
    border: 1px solid #2d3748;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.aqi-number {
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
    margin: 12px 0;
}
.aqi-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 100px;
    font-size: 0.9rem;
    font-weight: 600;
}

/* Alert */
.alert-box {
    background: linear-gradient(90deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 14px 20px;
    margin: 12px 0;
    color: #fecaca !important;
}

/* Pollutant card */
.poll-card {
    background: #111827;
    border: 1px solid #2d3748;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 8px;
}
.poll-name { font-size: 0.8rem; color: #94a3b8 !important; font-weight: 600; text-transform: uppercase; }
.poll-value { font-size: 1.4rem; font-weight: 700; }
.poll-desc { font-size: 0.7rem; color: #64748b !important; margin-top: 4px; }
.progress-bg { background: #1e293b; border-radius: 4px; height: 6px; margin: 8px 0; }

/* Section header */
.sec-header {
    font-size: 0.75rem;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    padding: 12px 0 8px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 12px;
}

/* Info card */
.info-card {
    background: #111827;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 16px;
    margin-top: 8px;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e293b;
}
.stTabs [data-baseweb="tab"] { border-radius: 6px; color: #94a3b8 !important; }
.stTabs [aria-selected="true"] { background: #1e40af !important; color: #fff !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1e293b; border-radius: 8px; }

/* Warnings/info boxes */
.stAlert { background: #111827 !important; border-color: #2d3748 !important; color: #e2e8f0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Config ───────────────────────────────────────────────
SUPABASE_URL      = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY      = os.environ.get("SUPABASE_KEY", "")
OPENWEATHER_TOKEN = os.environ.get("OPENWEATHER_TOKEN", "")
LAT, LON, CITY    = 24.8607, 67.0011, "karachi"

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(family="Space Grotesk", color="#94a3b8", size=12),
    xaxis=dict(gridcolor="#1e293b", linecolor="#2d3748", tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="#1e293b", linecolor="#2d3748", tickfont=dict(color="#64748b")),
    margin=dict(l=8, r=8, t=40, b=8),
)

# ── Helpers ──────────────────────────────────────────────
def aqi_info(v):
    if v <= 50:  return "Good",                 "#22c55e", "#052e16"
    if v <= 100: return "Moderate",             "#eab308", "#1c1917"
    if v <= 150: return "Unhealthy (Sensitive)",  "#f97316", "#1c0a00"
    if v <= 200: return "Unhealthy",              "#ef4444", "#1c0000"
    if v <= 300: return "Very Unhealthy",         "#a855f7", "#1a0030"
    return "Hazardous", "#dc2626", "#1a0000"

def pm25_to_aqi(pm25):
    if pm25 <= 0: return 0.0
    bp = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
          (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
    for c0,c1,i0,i1 in bp:
        if c0 <= pm25 <= c1:
            return round(((i1-i0)/(c1-c0))*(pm25-c0)+i0, 1)
    return 500.0

# ── Data Fetchers ────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_current():
    try:
        poll = requests.get(
            f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_TOKEN}",
            timeout=10).json()["list"][0]
        wx = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_TOKEN}&units=metric",
            timeout=10).json()
        c = poll["components"]
        pm25 = c.get("pm2_5", 0)
        return {
            "aqi": pm25_to_aqi(pm25),
            "pm25": pm25, "pm10": c.get("pm10",0), "o3": c.get("o3",0),
            "no2": c.get("no2",0), "co": c.get("co",0), "so2": c.get("so2",0),
            "temperature": wx["main"]["temp"], "humidity": wx["main"]["humidity"],
            "pressure": wx["main"]["pressure"], "wind": wx["wind"]["speed"],
            "feels_like": wx["main"]["feels_like"],
            "description": wx["weather"][0]["description"].title(),
        }
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_history(limit=2000):
    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        all_rows, page = [], 0
        while True:
            r = sb.table("aqi_features").select("timestamp,aqi,pm25,pm10,o3,temperature,humidity,wind") \
                .eq("city", CITY).order("timestamp", desc=False) \
                .range(page*1000, (page+1)*1000-1).execute()
            if not r.data: break
            all_rows.extend(r.data)
            if len(r.data) < 1000: break
            # Agar limit se zyada data ho jaye toh loop break karein
            if len(all_rows) >= limit: break
            page += 1
            
        if not all_rows:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_rows)
        
        # 1. Safe DateTime Parsing (Handles microseconds & mixed formats)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce', utc=True)
        
        # 2. Drop rows where timestamp parsing failed
        df = df.dropna(subset=["timestamp"])
        
        # 3. Data Cleaning: Remove unrealistic 500 AQI sensor spikes/glitches
        # Karachi ka AQI high zaroor hota hai par 500 exact straight spike outlier hai
        df = df[df["aqi"] < 495]
        
        # 4. Sort by timestamp descending (Latest data on top)
        df = df.sort_values(by="timestamp", ascending=False)
        
        # Enforce exact limit if needed
        return df.head(limit)
        
    except Exception as e:
        st.error(f"History fetch error: {e}")
        return pd.DataFrame()
    
@st.cache_data(ttl=300)
def fetch_forecast():
    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        r = sb.table("aqi_predictions").select("*").eq("city", CITY).order("timestamp").execute()
        if not r.data:
            return pd.DataFrame()
        df = pd.DataFrame(r.data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except Exception as e:
        st.error(f"Forecast fetch error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_model_info():
    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        r = sb.table("model_registry").select("*").order("trained_at", desc=True).limit(1).execute()
        return r.data[0] if r.data else {}
    except:
        return {}

@st.cache_data(ttl=600)
def fetch_shap():
    try:
        import json
        with open("models/shap_values.json") as f:
            return json.load(f)
    except:
        return {}

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 16px'>
        <div style='font-size:2.2rem'>🌬️</div>
        <div style='font-size:1.1rem;font-weight:700;color:#f1f5f9'>AQI Intelligence</div>
        <div style='font-size:0.72rem;color:#64748b;margin-top:4px'>Karachi Air Quality Monitor</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio("", [
        "🏠 Live Dashboard",
        "📈 Historical Analysis",
        "🔮 3-Day Forecast",
        "🧠 Model Insights"
    ], label_visibility="collapsed")

    st.divider()

    st.markdown("""
    <div style='font-size:0.72rem;color:#475569;padding:0 8px'>
        <div style='margin-bottom:6px'>📡 <span style='color:#94a3b8'>Data:</span> OpenWeather API</div>
        <div style='margin-bottom:6px'>🤖 <span style='color:#94a3b8'>Model:</span> Neural Network MLP</div>
        <div>⚡ <span style='color:#94a3b8'>Updates:</span> Every Hour</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ════════════════════════════════════════════════════════
# PAGE 1: LIVE DASHBOARD
# ════════════════════════════════════════════════════════
if "Live Dashboard" in page:
    st.markdown("<h1 style='color:#f1f5f9;font-size:1.8rem;font-weight:700;margin-bottom:2px'>Live Air Quality Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;margin-bottom:20px'>Karachi, Pakistan · Real-time monitoring</p>", unsafe_allow_html=True)

    current = fetch_current()

    if not current:
        st.error("Could not fetch live data. Check your API keys.")
    else:
        aqi = current["aqi"]
        label, color, bg = aqi_info(aqi)

        if aqi > 150:
            st.markdown(f"""
            <div class='alert-box'>
                <span style='font-size:1.2rem'>🚨</span>
                <strong style='color:#fca5a5'> HAZARDOUS AIR QUALITY ALERT</strong><br>
                <span style='color:#fca5a5;font-size:0.85rem'>AQI {aqi:.0f} — Avoid outdoor activities. Wear N95 mask if going outside.</span>
            </div>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown(f"""
            <div class='aqi-hero'>
                <div class='metric-label'>CURRENT AQI — KARACHI</div>
                <div class='aqi-number' style='color:{color}'>{aqi:.0f}</div>
                <div class='aqi-badge' style='background:{bg};color:{color};border:1px solid {color}55'>
                    {label}
                </div>
                <div style='margin-top:14px;font-size:0.78rem;color:#475569'>
                    {datetime.now().strftime("%B %d, %Y · %I:%M %p")}
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=aqi,
                domain={"x": [0,1], "y": [0,1]},
                gauge={
                    "axis": {"range": [0, 300], "tickcolor": "#475569", "tickfont": {"color": "#64748b"}},
                    "bar": {"color": color, "thickness": 0.28},
                    "bgcolor": "#111827",
                    "bordercolor": "#2d3748",
                    "steps": [
                        {"range": [0,50],   "color": "#052e16"},
                        {"range": [50,100],  "color": "#1c1917"},
                        {"range": [100,150], "color": "#1c0a00"},
                        {"range": [150,200], "color": "#1c0000"},
                        {"range": [200,300], "color": "#1a0030"},
                    ],
                },
                number={"font": {"color": color, "size": 44, "family": "Space Grotesk"}},
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    height=210, margin=dict(l=16,r=16,t=16,b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.markdown("<div class='sec-header'>☁️ WEATHER CONDITIONS</div>", unsafe_allow_html=True)
            wc1, wc2 = st.columns(2)
            wc1.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>🌡️ Temperature</div>
                <div class='metric-value' style='color:#f97316'>{current['temperature']:.1f}°C</div>
                <div class='metric-sub'>Feels {current['feels_like']:.1f}°C</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>💨 Wind Speed</div>
                <div class='metric-value' style='color:#14b8a6'>{current['wind']:.1f} m/s</div>
                <div class='metric-sub'>Surface wind</div>
            </div>
            """, unsafe_allow_html=True)
            wc2.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>💧 Humidity</div>
                <div class='metric-value' style='color:#3b82f6'>{current['humidity']:.0f}%</div>
                <div class='metric-sub'>{current['description']}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>📊 Pressure</div>
                <div class='metric-value' style='color:#8b5cf6'>{current['pressure']:.0f} hPa</div>
                <div class='metric-sub'>Atmospheric</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='sec-header'>🔬 POLLUTANT BREAKDOWN</div>", unsafe_allow_html=True)

        pollutants = [
            ("PM2.5", current["pm25"], 35.4, "#ef4444", "Fine particles — main AQI driver · µg/m³"),
            ("PM10",  current["pm10"], 150,  "#f97316", "Coarse particles · µg/m³"),
            ("O₃",    current["o3"],   70,   "#eab308", "Ground-level ozone · µg/m³"),
            ("NO₂",   current["no2"],  40,   "#a855f7", "Nitrogen dioxide · µg/m³"),
            ("CO",    current["co"],   4000, "#14b8a6", "Carbon monoxide · µg/m³"),
            ("SO₂",   current["so2"],  20,   "#3b82f6", "Sulfur dioxide · µg/m³"),
        ]
        pc1, pc2, pc3 = st.columns(3)
        for idx, (name, val, limit, clr, desc) in enumerate(pollutants):
            pct = min(val/limit*100, 100)
            [pc1, pc2, pc3][idx%3].markdown(f"""
            <div class='poll-card'>
                <div style='display:flex;justify-content:space-between;align-items:baseline'>
                    <div class='poll-name'>{name}</div>
                    <div class='poll-value' style='color:{clr}'>{val:.1f}</div>
                </div>
                <div class='progress-bg'>
                    <div style='background:{clr};width:{pct:.1f}%;height:6px;border-radius:4px'></div>
                </div>
                <div class='poll-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Last 48 Hours Trend ─────────────────────────────────
        history_48 = fetch_history(48)
        if not history_48.empty:
            # 🟢 On-the-fly transformation aur dual-bound outlier filtration for 48h trend
            history_48["timestamp"] = pd.to_datetime(history_48["timestamp"], errors='coerce', utc=True)
            history_48 = history_48.dropna(subset=["timestamp"])
            
            # Dual-bound guard rails (Bypassing 500 spikes AND faulty single-digit AQIs)
            history_48 = history_48[(history_48["aqi"] < 495) & (history_48["aqi"] > 30)]

            st.markdown("<br><div class='sec-header'>📉 LAST 48 HOURS AQI TREND</div>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_48["timestamp"], y=history_48["aqi"],
                fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
                line=dict(color="#3b82f6", width=2.5), name="AQI",
                hovertemplate="<b>AQI: %{y:.1f}</b><br>%{x}<extra></extra>"
            ))
            for level, clr, lbl in [(50,"#22c55e","Good"),(100,"#eab308","Moderate"),(150,"#f97316","Unhealthy")]:
                fig.add_hline(y=level, line_dash="dot", line_color=clr, opacity=0.5,
                            annotation_text=lbl, annotation_font_color=clr, annotation_position="right")
            fig.update_layout(**PLOTLY_BASE, height=260, showlegend=False,
                            title=dict(text="AQI — Last 48 Hours", font=dict(color="#64748b", size=13)))
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 2: HISTORICAL ANALYSIS
# ════════════════════════════════════════════════════════
elif page == "📈 Historical Analysis":
    st.markdown("<h1 style='color:#f1f5f9;font-size:1.8rem;font-weight:700;margin-bottom:2px'>Historical Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;margin-bottom:20px'>90 days of Karachi air quality data</p>", unsafe_allow_html=True)

    with st.spinner("Analyzing historical patterns..."):
        raw_history = fetch_history(2000)

    if raw_history.empty:
        st.warning("⚠️ No historical data found. Check Supabase connection.")
    else:
        # 🟢 CRITICAL: Robust Dual-Bound Outlier Data Cleaning
        history = raw_history.copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"], errors='coerce', utc=True)
        history = history.dropna(subset=["timestamp"])
        
        # Super-clean data window bounds
        history = history[(history["aqi"] < 495) & (history["aqi"] > 30)]

        # Time properties binding
        history["hour"]  = history["timestamp"].dt.hour
        history["day"]   = history["timestamp"].dt.day_name()
        history["month"] = history["timestamp"].dt.month_name()

        # Stats Metrics (Ab metrics completely trustworthy hain)
        s1, s2, s3, s4 = st.columns(4)
        for col, (lbl, val, clr) in zip([s1,s2,s3,s4], [
            ("📊 Avg AQI",     f"{history['aqi'].mean():.1f}", "#3b82f6"),
            ("🔴 Max AQI",     f"{history['aqi'].max():.1f}", "#ef4444"),
            ("🟢 Min AQI",     f"{history['aqi'].min():.1f}", "#22c55e"),
            ("📁 Total Records", f"{len(history):,}",          "#8b5cf6"),
        ]):
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{lbl}</div>
                <div style='font-size:1.8rem;font-weight:700;color:{clr};margin:6px 0'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Full history timeline layout
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=history["timestamp"], y=history["aqi"],
            fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
            line=dict(color="#3b82f6", width=1.5), name="AQI",
            hovertemplate="<b>AQI: %{y:.1f}</b><br>%{x}<extra></extra>"
        ))
        for level, clr, lbl in [(50,"#22c55e","Good"),(100,"#eab308","Moderate"),(150,"#f97316","Unhealthy")]:
            fig1.add_hline(y=level, line_dash="dot", line_color=clr, opacity=0.4,
                           annotation_text=lbl, annotation_font_color=clr)
        fig1.update_layout(**PLOTLY_BASE, height=300, showlegend=False,
                           title=dict(text="AQI Trend — Full History", font=dict(color="#94a3b8", size=14)))
        st.plotly_chart(fig1, use_container_width=True)

        # Columns for Hourly and Weekly Distribution
        c1, c2 = st.columns(2)
        with c1:
            hourly = history.groupby("hour")["aqi"].mean().reset_index()
            fig2 = px.bar(hourly, x="hour", y="aqi", title="Avg AQI by Hour of Day",
                          color="aqi", color_continuous_scale=[[0,"#22c55e"],[0.5,"#eab308"],[1,"#ef4444"]])
            fig2.update_layout(**PLOTLY_BASE, height=280, coloraxis_showscale=False,
                               title=dict(text="Avg AQI by Hour of Day", font=dict(color="#94a3b8",size=13)),
                               yaxis_range=[0, hourly["aqi"].max() + 20])
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            # 🟢 FIXED: Safe Column rendering block architecture
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            daily = history.groupby("day")["aqi"].mean().reindex(day_order).reset_index()
            
            fig3 = px.bar(daily, x="day", y="aqi", title="Avg AQI by Day of Week",
                          color="aqi", color_continuous_scale=[[0,"#22c55e"],[0.5,"#eab308"],[1,"#ef4444"]])
            fig3.update_layout(**PLOTLY_BASE, height=280, coloraxis_showscale=False,
                               title=dict(text="Avg AQI by Day of Week", font=dict(color="#94a3b8",size=13)),
                               yaxis_range=[0, daily["aqi"].max() + 20])
            st.plotly_chart(fig3, use_container_width=True)

        # PM2.5 vs AQI scatter
        sample = history.sample(min(500, len(history)))
        fig4 = px.scatter(sample, x="pm25", y="aqi", title="PM2.5 vs AQI Relationship",
                          color="aqi", color_continuous_scale=[[0,"#22c55e"],[0.5,"#eab308"],[1,"#ef4444"]],
                          trendline="ols")
        fig4.update_layout(**PLOTLY_BASE, height=300, coloraxis_showscale=False,
                           title=dict(text="PM2.5 vs AQI Relationship", font=dict(color="#94a3b8",size=13)))
        st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 2: HISTORICAL ANALYSIS
# ════════════════════════════════════════════════════════
elif page == "📈 Historical Analysis":
    st.markdown("<h1 style='color:#f1f5f9;font-size:1.8rem;font-weight:700;margin-bottom:2px'>Historical Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;margin-bottom:20px'>90 days of Karachi air quality data</p>", unsafe_allow_html=True)

    with st.spinner("Analyzing historical patterns..."):
        raw_history = fetch_history(2000)

    if raw_history.empty:
        st.warning("⚠️ No historical data found. Check Supabase connection.")
    else:
        # 🟢 CRITICAL: Outlier filter orchestration (Purane kachre data ko on-the-fly discard karne ke liye)
        history = raw_history.copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"], errors='coerce', utc=True)
        history = history.dropna(subset=["timestamp"])
        history = history[history["aqi"] < 495]

        # Safe tracking properties binding
        history["hour"]  = history["timestamp"].dt.hour
        history["day"]   = history["timestamp"].dt.day_name()
        history["month"] = history["timestamp"].dt.month_name()

        # Stats Metrics (Ab original averages accurately calculated hain bina kachra spikes ke)
        s1, s2, s3, s4 = st.columns(4)
        for col, (lbl, val, clr) in zip([s1,s2,s3,s4], [
            ("📊 Avg AQI",     f"{history['aqi'].mean():.1f}", "#3b82f6"),
            ("🔴 Max AQI",     f"{history['aqi'].max():.1f}", "#ef4444"),
            ("🟢 Min AQI",     f"{history['aqi'].min():.1f}", "#22c55e"),
            ("📁 Total Records", f"{len(history):,}",          "#8b5cf6"),
        ]):
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{lbl}</div>
                <div style='font-size:1.8rem;font-weight:700;color:{clr};margin:6px 0'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Full trend (Vanish the 500 AQI linear spikes line)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=history["timestamp"], y=history["aqi"],
            fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
            line=dict(color="#3b82f6", width=1.5), name="AQI",
            hovertemplate="<b>AQI: %{y:.1f}</b><br>%{x}<extra></extra>"
        ))
        for level, clr, lbl in [(50,"#22c55e","Good"),(100,"#eab308","Moderate"),(150,"#f97316","Unhealthy")]:
            fig1.add_hline(y=level, line_dash="dot", line_color=clr, opacity=0.4,
                           annotation_text=lbl, annotation_font_color=clr)
        fig1.update_layout(**PLOTLY_BASE, height=300, showlegend=False,
                           title=dict(text="AQI Trend — Full History", font=dict(color="#94a3b8", size=14)))
        st.plotly_chart(fig1, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            # Hourly distribution bar chart build dynamically from filtered dataframe
            hourly = history.groupby("hour")["aqi"].mean().reset_index()
            fig2 = px.bar(hourly, x="hour", y="aqi", title="Avg AQI by Hour of Day",
                          color="aqi", color_continuous_scale=[[0,"#22c55e"],[0.5,"#eab308"],[1,"#ef4444"]])
            fig2.update_layout(**PLOTLY_BASE, height=280, coloraxis_showscale=False,
                               title=dict(text="Avg AQI by Hour of Day", font=dict(color="#94a3b8",size=13)),
                               yaxis_range=[0, hourly["aqi"].max() + 20]) # Tight scaling auto-fix
            st.plotly_chart(fig2, use_container_width=True)

        with col2 if 'col2' in locals() else c2:
            # Weekly metrics layout rendering exact variation of closed/weekend schedules
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            daily = history.groupby("day")["aqi"].mean().reindex(day_order).reset_index()
            fig3 = px.bar(daily, x="day", y="aqi", title="Avg AQI by Day of Week",
                          color="aqi", color_continuous_scale=[[0,"#22c55e"],[0.5,"#eab308"],[1,"#ef4444"]])
            fig3.update_layout(**PLOTLY_BASE, height=280, coloraxis_showscale=False,
                               title=dict(text="Avg AQI by Day of Week", font=dict(color="#94a3b8",size=13)),
                               yaxis_range=[0, daily["aqi"].max() + 20]) # Tight scaling auto-fix
            st.plotly_chart(fig3, use_container_width=True)

        # PM2.5 vs AQI scatter (Spotless distribution with accurate standard deviation slope)
        sample = history.sample(min(500, len(history)))
        fig4 = px.scatter(sample, x="pm25", y="aqi", title="PM2.5 vs AQI Relationship",
                          color="aqi", color_continuous_scale=[[0,"#22c55e"],[0.5,"#eab308"],[1,"#ef4444"]],
                          trendline="ols")
        fig4.update_layout(**PLOTLY_BASE, height=300, coloraxis_showscale=False,
                           title=dict(text="PM2.5 vs AQI Relationship", font=dict(color="#94a3b8",size=13)))
        st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 3: 3-DAY FORECAST
# ════════════════════════════════════════════════════════
elif "Forecast" in page:
    st.markdown("<h1 style='color:#f1f5f9;font-size:1.8rem;font-weight:700;margin-bottom:2px'>3-Day AQI Forecast</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;margin-bottom:20px'>ML-powered predictions for Karachi</p>", unsafe_allow_html=True)

    with st.spinner("Loading forecast..."):
        raw_forecast = fetch_forecast()

    if raw_forecast is None or raw_forecast.empty:
        st.info("ℹ️ No forecast data yet. Run the inference pipeline first.")
    else:
        # 🟢 FRONTEND INTEGRATION: Apply robust dual-bound validation cleaning
        forecast = raw_forecast.copy()
        forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], errors='coerce', utc=True)
        forecast = forecast.dropna(subset=["timestamp"])
        
        # Guard rails to prevent rendering legacy training anomalies (8.0 or 500.0)
        forecast = forecast[(forecast["predicted_aqi"] < 495) & (forecast["predicted_aqi"] > 30)]

        # Recalculate safe alerts context after cleaning
        alerts = forecast[forecast["is_alert"] == True]
        if not alerts.empty:
            st.markdown(f"""
            <div class='alert-box'>
                🚨 <strong style='color:#fca5a5'>FORECAST ALERT</strong><br>
                <span style='color:#fca5a5;font-size:0.85rem'>{len(alerts)} hours of unhealthy AQI predicted in the next 3 days</span>
            </div>
            """, unsafe_allow_html=True)

        # 🟢 FIX: Explicit sort by timestamp to guarantee absolute chronographical sequence (Today -> Day 3)
        forecast = forecast.sort_values("timestamp")
        forecast["date"] = forecast["timestamp"].dt.date
        
        # Aggregate maintaining correct operational chronological flow
        days = forecast.groupby("date", sort=False).agg(
            avg=("predicted_aqi", "mean"),
            mx=("predicted_aqi", "max"),
            mn=("predicted_aqi", "min"),
            alerts=("is_alert", "sum")
        ).reset_index().head(3)

        cols = st.columns(3)
        day_labels = ["Today +1", "Today +2", "Today +3"]
        for i, (col, row) in enumerate(zip(cols, days.itertuples())):
            lbl, clr, bg = aqi_info(row.avg)
            alert_txt = f"🚨 {int(row.alerts)}h alert" if row.alerts > 0 else "✅ Safe"
            col.markdown(f"""
            <div class='aqi-hero' style='padding:24px'>
                <div class='metric-label'>{day_labels[i]}</div>
                <div style='font-size:0.78rem;color:#475569;margin-bottom:8px'>{row.date}</div>
                <div style='font-size:3rem;font-weight:700;color:{clr};line-height:1'>{row.avg:.0f}</div>
                <div class='aqi-badge' style='background:{bg};color:{clr};border:1px solid {clr}44;font-size:0.78rem;margin-top:10px'>
                    {lbl}
                </div>
                <div style='margin-top:10px;font-size:0.75rem;color:#64748b'>
                    ↑ {row.mx:.0f} · ↓ {row.mn:.0f} · {alert_txt}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Main Forecast graph wrapper
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast["timestamp"], y=forecast["predicted_aqi"],
            fill="tozeroy", fillcolor="rgba(139,92,246,0.08)",
            line=dict(color="#8b5cf6", width=2), name="Predicted AQI",
            hovertemplate="<b>Predicted AQI: %{y:.1f}</b><br>%{x}<extra></extra>"
        ))
        for level, clr, lbl in [(50,"#22c55e","Good"),(100,"#eab308","Moderate"),(150,"#f97316","Caution")]:
            fig.add_hline(y=level, line_dash="dot", line_color=clr, opacity=0.5,
                          annotation_text=lbl, annotation_font_color=clr, annotation_position="right")
        fig.update_layout(**PLOTLY_BASE, height=320, showlegend=False,
                          title=dict(text="72-Hour AQI Forecast Trend", font=dict(color="#94a3b8", size=14)))
        st.plotly_chart(fig, use_container_width=True)

        # Hourly table presentation layer
        st.markdown("<div class='sec-header'>📋 HOURLY BREAKDOWN</div>", unsafe_allow_html=True)
        table = forecast[["timestamp","predicted_aqi","aqi_category","is_alert"]].copy()
        table["timestamp"] = table["timestamp"].dt.strftime("%b %d · %H:%M")
        table.columns = ["Time", "Predicted AQI", "Category", "Alert"]
        st.dataframe(table.head(24), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════
# PAGE 4: MODEL INSIGHTS
# ════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown("<h1 style='color:#f1f5f9;font-size:1.8rem;font-weight:700;margin-bottom:2px'>Model Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;margin-bottom:20px'>ML model performance & explainability</p>", unsafe_allow_html=True)

    with st.spinner("Fetching performance metrics..."):
        model_info = fetch_model_info()
        shap_data  = fetch_shap()

    # 🟢 FIX: Add explicit null/type safe checks to shield app against database payload delays
    if model_info and isinstance(model_info, dict):
        m1, m2, m3, m4 = st.columns(4)
        for col, (lbl, val, clr) in zip([m1,m2,m3,m4], [
            ("Model",   model_info.get("model_name","N/A"),       "#3b82f6"),
            ("RMSE",    f"{model_info.get('rmse',0):.2f}" if model_info.get('rmse') is not None else "N/A",        "#ef4444"),
            ("MAE",     f"{model_info.get('mae',0):.2f}" if model_info.get('mae') is not None else "N/A",         "#f97316"),
            ("R²",      f"{model_info.get('r2',0):.4f}" if model_info.get('r2') is not None else "N/A",          "#22c55e"),
        ]):
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{lbl}</div>
                <div style='font-size:1.5rem;font-weight:700;color:{clr};margin:8px 0;word-break:break-word'>{val}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Model telemetry parameters currently unavailable.")

    st.markdown("<br>", unsafe_allow_html=True)

    # SHAP chart render execution 
    if shap_data and isinstance(shap_data, dict) and "top_features" in shap_data:
        top = shap_data["top_features"]
        feat_names = [f["feature"].replace("_lag_","↩").replace("h","h ") for f in top]
        importances = [f["importance"] for f in top]

        fig_shap = go.Figure(go.Bar(
            x=importances,
            y=feat_names,
            orientation="h",
            marker=dict(
                color=importances,
                colorscale=[[0,"#1e40af"],[0.5,"#7c3aed"],[1,"#db2777"]],
                showscale=False,
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
        ))
        fig_shap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.6)",
            font=dict(family="Space Grotesk", color="#94a3b8", size=12),
            margin=dict(l=8, r=8, t=44, b=8),
            height=300,
            title=dict(text="🔍 SHAP Feature Importance — Top 5", font=dict(color="#94a3b8", size=14)),
            xaxis=dict(gridcolor="#1e293b", linecolor="#2d3748", tickfont=dict(color="#64748b")),
            yaxis=dict(
                autorange="reversed",
                gridcolor="#1e293b",
                linecolor="#2d3748",
                tickfont=dict(color="#e2e8f0", size=12)
            ),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown("""
        <div class='info-card'>
            <div style='font-size:0.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>
                💡 What does this mean?
            </div>
            <div style='font-size:0.85rem;color:#94a3b8;line-height:1.6'>
                SHAP (SHapley Additive exPlanations) reveals which features most influence AQI predictions.
                <strong style='color:#e2e8f0'>PM2.5 lag features dominate</strong> — meaning recent PM2.5
                readings are the strongest predictors of future AQI in Karachi, confirming PM2.5 as
                the primary pollutant driving air quality changes.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ SHAP explainability matrices are not generated for this model snapshot yet.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec-header'>📊 MODEL COMPARISON</div>", unsafe_allow_html=True)

    models_df = pd.DataFrame([
        {"Model": "Ridge Regression", "Type": "Statistical",  "RMSE": 42.81, "MAE": 19.31, "R²": 0.153},
        {"Model": "Random Forest",    "Type": "ML Ensemble",  "RMSE": 42.12, "MAE": 11.22, "R²": 0.181},
        {"Model": "Neural Network",   "Type": "Deep Learning","RMSE": 39.97, "MAE": 15.01, "R²": 0.262},
    ])

    fig_cmp = go.Figure()
    for i, (row, clr) in enumerate(zip(models_df.itertuples(), ["#3b82f6","#14b8a6","#8b5cf6"])):
        fig_cmp.add_trace(go.Bar(name=row.Model, x=["RMSE","MAE"], y=[row.RMSE, row.MAE],
                                 marker_color=clr))
    fig_cmp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.6)",
        font=dict(family="Space Grotesk", color="#94a3b8", size=12),
        margin=dict(l=8,r=8,t=44,b=8), height=280, barmode="group",
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="RMSE & MAE Comparison", font=dict(color="#94a3b8",size=13)),
        xaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#64748b")),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.dataframe(models_df, use_container_width=True, hide_index=True)