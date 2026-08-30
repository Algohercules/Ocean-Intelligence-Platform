"""
pages/3_ARGO.py
===============
ARGO Ocean Observatory Page.
Focuses on ARGO fleet monitoring, float status, trajectory movement, data recency, observation gaps, and network operational alerts.
"""

import sys
from pathlib import Path

# Ensure repository root and frontend directory are in sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
_frontend_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_root), str(_frontend_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

st.set_page_config(
    page_title="ARGO Ocean Observatory | Indian Ocean Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from components.header import render_header
from components.sidebar import render_sidebar
from components.ocean_map import render_ocean_map
from components.footer import render_footer
from data.mock_data import (
    get_argo_floats,
    get_temperature_profile,
    get_argo_vs_glorys_profile,
    DEPTH_LEVELS
)

@st.cache_data(show_spinner=False)
def fetch_argo_floats_data():
    np.random.seed(42)
    n_floats = 124
    float_ids = [f"WMO_{6903000 + i}" for i in range(n_floats)]
    lats = np.random.uniform(-30.0, 24.0, n_floats)
    lons = np.random.uniform(40.0, 105.0, n_floats)
    surface_temps = np.round(28.5 - 0.18 * np.abs(lats) + np.random.normal(0, 0.4, n_floats), 1)
    
    dates = pd.date_range(end="2024-05-20", periods=30, freq="D")
    last_obs = [dates[np.random.randint(0, len(dates))].strftime("%Y-%m-%d") for _ in range(n_floats)]
    statuses = np.random.choice(['Active', 'Reporting', 'Calibrating', 'Inactive'], n_floats, p=[0.6, 0.25, 0.1, 0.05])
    
    return pd.DataFrame({
        'float_id': float_ids,
        'latitude': np.round(lats, 2),
        'longitude': np.round(lons, 2),
        'surface_temp': surface_temps,
        'last_observation': last_obs,
        'status': statuses
    })

# Helper functions for ARGO trajectories & fleet analytics
def fetch_argo_trajectory(float_id="WMO_6903124", lat=15.2, lon=65.4):
    np.random.seed(int(hash(float_id) % 100000))
    n_points = 18
    dates = pd.date_range(end="2024-05-20", periods=n_points, freq="10D")
    
    # Simulate drift path across Indian Ocean
    drift_lat = lat - np.linspace(3.5, 0, n_points) + np.random.normal(0, 0.2, n_points)
    drift_lon = lon - np.linspace(4.2, 0, n_points) + np.random.normal(0, 0.2, n_points)
    depths = [1000] * n_points
    cycle_nums = list(range(1, n_points + 1))
    
    df_traj = pd.DataFrame({
        'Cycle': cycle_nums,
        'Date': dates,
        'Latitude (°N)': np.round(drift_lat, 3),
        'Longitude (°E)': np.round(drift_lon, 3),
        'Max Depth (m)': depths,
        'Temperature (°C)': np.round(26.4 - 0.15 * np.arange(n_points) + np.random.normal(0, 0.2, n_points), 1)
    })
    
    # Distance calculation
    total_dist = np.round(np.sum(np.sqrt(np.diff(drift_lat)**2 + np.diff(drift_lon)**2)) * 111.0, 1)
    avg_speed = np.round(total_dist / (n_points * 10), 2)
    ns_disp = np.round((drift_lat[-1] - drift_lat[0]) * 111.0, 1)
    ew_disp = np.round((drift_lon[-1] - drift_lon[0]) * 111.0, 1)
    
    stats = {
        'total_distance': f"{total_dist} km",
        'avg_speed': f"{avg_speed} km/day",
        'ns_disp': f"{'+' if ns_disp>=0 else ''}{ns_disp} km",
        'ew_disp': f"{'+' if ew_disp>=0 else ''}{ew_disp} km",
        'dominant_direction': "Northeast Drift (NE)"
    }
    
    return df_traj, stats

def fetch_argo_recency_distribution(df_floats):
    recency_counts = {
        'Today (0-1d)': int(len(df_floats) * 0.45),
        '1–7 Days': int(len(df_floats) * 0.32),
        '8–30 Days': int(len(df_floats) * 0.14),
        '31–90 Days': int(len(df_floats) * 0.06),
        '90+ Days': int(len(df_floats) * 0.03)
    }
    return pd.DataFrame({'Recency Category': list(recency_counts.keys()), 'Float Count': list(recency_counts.values())})

render_header(active_page="Argo")
controls = render_sidebar()

# Page Header
st.markdown(
    """
    <div style="background: rgba(13, 27, 42, 0.7); border: 1px solid #1E3A5F; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="font-family: 'Outfit', sans-serif; color: #38BDF8; margin: 0; font-size: 1.35rem;">
                    📡 ARGO OCEAN OBSERVATORY & FLEET MISSION CONTROL
                </h2>
                <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 0.88rem;">
                    Monitor autonomous profiling floats, observation coverage and oceanographic data across the Indian Ocean.
                </p>
            </div>
            <span class="badge-cyan" style="border-color: #38BDF8; color: #38BDF8; font-weight:700;">SYSTEM STATUS: 124 FLOATS ONLINE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

df_floats = fetch_argo_floats_data()

# Initialize session state for float selection
if 'sel_float_id' not in st.session_state:
    st.session_state['sel_float_id'] = df_floats.iloc[0]['float_id']

# ============================================================
# 2. ARGO FLEET OVERVIEW KPI CARDS
# ============================================================
total_floats = len(df_floats)
active_floats = int(np.sum(df_floats['status'] == 'Active')) + int(np.sum(df_floats['status'] == 'Reporting'))
inactive_floats = total_floats - active_floats

st.markdown(
    f"""
    <div style="display: grid; grid-template-columns: repeat(8, 1fr); gap: 8px; margin-bottom: 16px;">
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#EFF6FF; border-color:#93C5FD;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Total Floats</div>
            <div style="font-size:1.15rem; color:#1D4ED8; font-weight:700;">{total_floats}</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#F0FDF4; border-color:#86EFAC;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Active Floats</div>
            <div style="font-size:1.15rem; color:#15803D; font-weight:700;">{active_floats}</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#FEF2F2; border-color:#FCA5A5;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Inactive Floats</div>
            <div style="font-size:1.15rem; color:#DC2626; font-weight:700;">{inactive_floats}</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#FAF5FF; border-color:#D8B4FE;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Total Profiles</div>
            <div style="font-size:1.15rem; color:#7E22CE; font-weight:700;">5,420</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#ECFEFF; border-color:#67E8F9;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Latest Data</div>
            <div style="font-size:1.15rem; color:#0E7490; font-weight:700;">2024-05-20</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#FFF7ED; border-color:#FDBA74;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Max Depth</div>
            <div style="font-size:1.15rem; color:#C2410C; font-weight:700;">1000 m</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#FEFCE8; border-color:#FDE047;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Observations</div>
            <div style="font-size:1.15rem; color:#A16207; font-weight:700;">14,200</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#F8FAFC; border-color:#CBD5E1;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Regions Covered</div>
            <div style="font-size:1.15rem; color:#0F172A; font-weight:700;">4 Regions</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 3 & 4. MAIN ARGO FLOAT MAP & FLEET STATUS
# ============================================================
c_argo_map, c_status_side = st.columns([2.5, 1.0])

with c_argo_map:
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">🗺️ ARGO FLOAT SPATIAL DISTRIBUTION MAP — INDIAN OCEAN</div>', unsafe_allow_html=True)
    render_ocean_map(
        dataset="ARGO Observations",
        variable=controls['variable'],
        depth=int(controls['depth']),
        date_str=str(controls['date']),
        region=controls['region'],
        target_lat=controls['target_lat'],
        target_lon=controls['target_lon'],
        show_floats=True,
        show_heatmap=True
    )

with c_status_side:
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">📊 FLEET STATUS DISTRIBUTION</div>', unsafe_allow_html=True)
    
    status_counts = df_floats['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    fig_status = px.pie(
        status_counts,
        names='Status',
        values='Count',
        color='Status',
        color_discrete_map={'Active': '#16A34A', 'Reporting': '#2563EB', 'Calibrating': '#EAB308', 'Inactive': '#DC2626'},
        hole=0.4
    )
    fig_status.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=10, b=10),
        height=180,
        showlegend=True,
        legend=dict(font=dict(size=9, color="#0F172A"))
    )
    st.plotly_chart(fig_status, use_container_width=True, config={'displayModeBar': False})
    st.caption("ℹ️ *Status inferred from observation recency and float telemetry.*")
    
    # 5. FIND ARGO FLOAT / SELECTOR TOOL
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:0.95rem; font-weight:700; color:#0F172A; margin-top:12px; margin-bottom:4px;">🔍 SELECT TARGET FLOAT</div>', unsafe_allow_html=True)
    sel_f_id = st.selectbox("Choose Float WMO ID", df_floats['float_id'].tolist(), index=0)
    if sel_f_id:
        st.session_state['sel_float_id'] = sel_f_id

st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

# Selected float data lookup
sel_row = df_floats[df_floats['float_id'] == st.session_state['sel_float_id']].iloc[0]
df_traj, traj_stats = fetch_argo_trajectory(float_id=sel_row['float_id'], lat=sel_row['latitude'], lon=sel_row['longitude'])

# ============================================================
# 6, 7 & 8. FLOAT DETAILS, TRAJECTORY & MISSION TRACKER
# ============================================================
c_row3_det, c_row3_traj = st.columns([1.2, 1.8])

with c_row3_det:
    st.markdown(
        f"""
        <div class="info-card-box">
            <div class="info-card-header">📡 FLOAT MISSION DETAILS — {sel_row['float_id']}</div>
            <table style="width:100%; font-size:0.83rem; color:#334155; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Float WMO ID:</td><td style="text-align:right; font-weight:700; color:#0284C7;">{sel_row['float_id']}</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Operational Status:</td><td style="text-align:right; font-weight:700; color:#16A34A;">{sel_row['status']}</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Last Position:</td><td style="text-align:right; font-weight:600;">{sel_row['latitude']:.2f}° N, {sel_row['longitude']:.2f}° E</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Latest Observation:</td><td style="text-align:right; font-weight:600;">{sel_row['last_observation']}</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Data Recency:</td><td style="text-align:right; font-weight:700; color:#1D4ED8;">2 days ago</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Total Profiles:</td><td style="text-align:right; font-weight:600;">43 Profiles</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Max Profile Depth:</td><td style="text-align:right; font-weight:600;">1000 m</td></tr>
                <tr><td style="padding:4px 0; color:#64748B;">Surface Temp:</td><td style="text-align:right; font-weight:700; color:#DC2626;">{sel_row['surface_temp']} °C</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

with c_row3_traj:
    st.markdown(f'<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">📍 FLOAT DRIFT TRAJECTORY ({sel_row["float_id"]})</div>', unsafe_allow_html=True)
    
    fig_traj = go.Figure()
    
    # Trajectory path line
    fig_traj.add_trace(go.Scatter(
        x=df_traj['Longitude (°E)'],
        y=df_traj['Latitude (°N)'],
        mode='lines+markers',
        name='Float Drift Trajectory',
        line=dict(color='#0284C7', width=2.5),
        marker=dict(size=6, color='#0284C7')
    ))
    
    # Latest position marker (Red)
    fig_traj.add_trace(go.Scatter(
        x=[df_traj['Longitude (°E)'].iloc[-1]],
        y=[df_traj['Latitude (°N)'].iloc[-1]],
        mode='markers+text',
        name='Latest Position',
        text=["LATEST"],
        textposition="top center",
        marker=dict(size=14, color='#DC2626', symbol='star')
    ))

    fig_traj.update_layout(
        title=dict(text=f"HYDRODYNAMIC DRIFT PATHWAY FOR {sel_row['float_id']}", font=dict(family="Outfit", size=11, color="#0F172A")),
        xaxis=dict(title=dict(text="Longitude (°E)", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), gridcolor="#E2E8F0", fixedrange=True),
        yaxis=dict(title=dict(text="Latitude (°N)", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), gridcolor="#E2E8F0", fixedrange=True),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=35, b=35),
        height=280,
        showlegend=False
    )
    st.plotly_chart(fig_traj, use_container_width=True, config={'displayModeBar': 'hover', 'displaylogo': False})

st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

# ============================================================
# 9 & 10. MOVEMENT ANALYSIS & RECENCY MONITOR
# ============================================================
c_row4_mov, c_row4_rec = st.columns([1.4, 1.6])

with c_row4_mov:
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">🧭 FLOAT MOVEMENT KINEMATICS</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:8px;">
            <div style="background:#FFF7ED; border:1px solid #FDBA74; border-radius:6px; padding:10px; text-align:center;">
                <div style="font-size:0.65rem; color:#64748B; font-weight:700;">TOTAL DISTANCE</div>
                <div style="font-size:1.1rem; color:#C2410C; font-weight:700;">{traj_stats['total_distance']}</div>
            </div>
            <div style="background:#F0FDF4; border:1px solid #86EFAC; border-radius:6px; padding:10px; text-align:center;">
                <div style="font-size:0.65rem; color:#64748B; font-weight:700;">AVG SPEED</div>
                <div style="font-size:1.1rem; color:#15803D; font-weight:700;">{traj_stats['avg_speed']}</div>
            </div>
            <div style="background:#EFF6FF; border:1px solid #93C5FD; border-radius:6px; padding:10px; text-align:center;">
                <div style="font-size:0.65rem; color:#64748B; font-weight:700;">N/S DISPLACEMENT</div>
                <div style="font-size:1.1rem; color:#1D4ED8; font-weight:700;">{traj_stats['ns_disp']}</div>
            </div>
            <div style="background:#FAF5FF; border:1px solid #D8B4FE; border-radius:6px; padding:10px; text-align:center;">
                <div style="font-size:0.65rem; color:#64748B; font-weight:700;">E/W DISPLACEMENT</div>
                <div style="font-size:1.1rem; color:#7E22CE; font-weight:700;">{traj_stats['ew_disp']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c_row4_rec:
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">⏱️ ARGO DATA RECENCY MONITOR</div>', unsafe_allow_html=True)
    df_rec = fetch_argo_recency_distribution(df_floats)
    fig_rec = px.bar(
        df_rec,
        x='Recency Category',
        y='Float Count',
        color='Recency Category',
        color_discrete_sequence=['#16A34A', '#2563EB', '#EAB308', '#EA580C', '#DC2626']
    )
    fig_rec.update_layout(
        title=dict(text="OBSERVATION RECENCY AGE DISTRIBUTION", font=dict(family="Outfit", size=11, color="#0F172A")),
        xaxis=dict(title=dict(text="Recency Window", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), fixedrange=True),
        yaxis=dict(title=dict(text="Float Count", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), fixedrange=True),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=35, b=35),
        height=200,
        showlegend=False
    )
    st.plotly_chart(fig_rec, use_container_width=True, config={'displayModeBar': False})

st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

# ============================================================
# 13 & 14. REGIONAL DISTRIBUTION & DEPTH COVERAGE MATRIX
# ============================================================
c_row5_reg, c_row5_depth = st.columns([1.5, 1.5])

with c_row5_reg:
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">🌐 REGIONAL FLOAT DISTRIBUTION MATRIX</div>', unsafe_allow_html=True)
    df_reg = pd.DataFrame([
        {'Region': 'Arabian Sea', 'Float Count': 42, 'Profile Count': '1,840', 'Latest Observation': '2024-05-20', 'Avg Depth': '1000 m'},
        {'Region': 'Bay of Bengal', 'Float Count': 38, 'Profile Count': '1,620', 'Latest Observation': '2024-05-19', 'Avg Depth': '850 m'},
        {'Region': 'Equatorial Indian Ocean', 'Float Count': 28, 'Profile Count': '1,210', 'Latest Observation': '2024-05-20', 'Avg Depth': '1000 m'},
        {'Region': 'Southern Indian Ocean', 'Float Count': 16, 'Profile Count': '750', 'Latest Observation': '2024-05-18', 'Avg Depth': '700 m'}
    ])
    st.dataframe(df_reg, use_container_width=True, hide_index=True)

with c_row5_depth:
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">🌊 DEPTH COVERAGE MATRIX BY REGION</div>', unsafe_allow_html=True)
    df_depth_mat = pd.DataFrame([
        {'Region': 'Arabian Sea', '0m': '✓', '10m': '✓', '50m': '✓', '100m': '✓', '200m': '✓', '500m': '✓', '1000m': '✓'},
        {'Region': 'Bay of Bengal', '0m': '✓', '10m': '✓', '50m': '✓', '100m': '✓', '200m': '✓', '500m': '✓', '1000m': '—'},
        {'Region': 'Equatorial IO', '0m': '✓', '10m': '✓', '50m': '✓', '100m': '✓', '200m': '✓', '500m': '—', '1000m': '—'},
        {'Region': 'Southern IO', '0m': '✓', '10m': '✓', '50m': '✓', '100m': '—', '200m': '—', '500m': '×', '1000m': '×'}
    ])
    st.dataframe(df_depth_mat, use_container_width=True, hide_index=True)

st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

# ============================================================
# 16, 18 & 23. OPERATIONAL ALERTS & ARGO INTELLIGENCE
# ============================================================
c_row6_alt, c_row6_ins = st.columns([1.5, 1.5])

with c_row6_alt:
    st.markdown(
        """
        <div style="background:#EFF6FF; border:1px solid #93C5FD; border-radius:6px; padding:12px;">
            <div style="font-size:0.83rem; font-weight:700; color:#1D4ED8; margin-bottom:6px;">🚨 ARGO NETWORK OPERATIONAL ALERTS</div>
            <ul style="font-size:0.8rem; color:#334155; padding-left:16px; margin:0; line-height:1.5;">
                <li style="margin-bottom:4px;">Float <b>WMO_6903124</b> recently reported a new profile at 1000 m depth.</li>
                <li style="margin-bottom:4px;">Observational coverage is sparse in the Southern Indian Ocean below 500 m.</li>
                <li>42 active profiling floats currently operating in the Arabian Sea sub-basin.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with c_row6_ins:
    st.markdown(
        """
        <div class="info-card-box">
            <div class="info-card-header">🧠 ARGO OBSERVATORY INTELLIGENCE</div>
            <ul style="font-size:0.8rem; color:#334155; padding-left:16px; margin:0; line-height:1.5;">
                <li style="margin-bottom:4px;">High observation density concentrated along major Arabian Sea shipping corridors.</li>
                <li style="margin-bottom:4px;">Average float drift speed is <b>4.2 km/day</b> under prevailing monsoon currents.</li>
                <li>Overall network data quality rated <b>GOOD (98.2% valid measurements)</b>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

# ============================================================
# 17. LATEST ARGO OBSERVATIONS TABLE & DOWNLOAD
# ============================================================
with st.expander("📄 VIEW & DOWNLOAD FULL ARGO OBSERVATIONS DATASET (CSV)", expanded=False):
    st.markdown('<div style="font-weight:700; color:#0F172A; margin-bottom:8px;">ARGO TELEMETRY DATASET</div>', unsafe_allow_html=True)
    st.dataframe(df_floats, use_container_width=True, hide_index=True)
    
    csv_bytes = df_floats.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download ARGO Telemetry Dataset (CSV)",
        data=csv_bytes,
        file_name="ARGO_Telemetry_Observations.csv",
        mime="text/csv",
        use_container_width=True
    )

render_footer()
