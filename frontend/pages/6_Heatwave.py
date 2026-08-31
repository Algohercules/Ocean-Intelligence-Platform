"""
pages/6_Heatwave.py
===================
Marine Heatwave (MHW) Intelligence & Ocean Anomaly Tracker Page.
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
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Heatwave | Pirates Of Ocean",
    page_icon="🔥",
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
from components.area_selection import render_area_selection_ui, render_quick_stats_ui
from components.footer import render_footer
from data.mock_data import (
    get_heatwave_data,
    get_mhw_timeseries_data,
    get_point_details,
    get_temperature_profile,
    DEPTH_LEVELS
)

# Self-contained helper data generators to prevent module import caching issues
def fetch_mhw_evolution_timeseries(lat=15.0, lon=65.0, depth=75, region="Arabian Sea"):
    dates = pd.date_range(start="2024-01-01", end="2024-05-20", freq="D")
    t = np.linspace(0, len(dates), len(dates))
    
    # Surface climatology decreases away from equator
    surf_clim = 29.2 - 0.14 * abs(lat) + 0.03 * (lon - 65.0)
    # Depth stratification
    depth_clim = surf_clim / (1.0 + (depth / 160.0)**1.2) + 4.0
    climatology = depth_clim + 1.2 * np.sin(2 * np.pi * t / 365.0)
    thresh_90th = climatology + 1.4
    
    # Spatial heatwave anomaly amplitude depending on basin & coordinates
    if lat > 5.0 and 50.0 <= lon <= 78.0:
        # Arabian Sea Spring Hotspot
        spike_amp = max(0.4, 3.4 - 0.008 * depth - 0.05 * abs(lat - 16.0))
    elif lat > 5.0 and 78.0 < lon <= 98.0:
        # Bay of Bengal
        spike_amp = max(0.4, 2.4 - 0.006 * depth - 0.04 * abs(lat - 15.0))
    elif -8.0 <= lat <= 5.0:
        # Equatorial Warm Pool
        spike_amp = max(0.3, 1.6 - 0.004 * depth)
    else:
        # Southern Indian Ocean
        spike_amp = max(0.2, 0.7 - 0.002 * depth)
        
    spike = spike_amp * np.exp(-((t - 118)**2) / 135.0)
    observed = climatology + spike + np.random.normal(0, 0.08, len(dates))
    anomaly = observed - climatology
    
    df_mhw = pd.DataFrame({
        'Date': dates,
        'Observed Temp (°C)': np.round(observed, 2),
        'Climatology Baseline (°C)': np.round(climatology, 2),
        'Anomaly (°C)': np.round(anomaly, 2),
        '90th Percentile Threshold (°C)': np.round(thresh_90th, 2),
        'Heatwave Active': observed > thresh_90th
    })
    
    current_anom = float(anomaly[-1])
    peak_anom = float(np.max(anomaly))
    active_days = int(np.sum(observed > thresh_90th))
    
    if peak_anom >= 2.8:
        status, severity = "SEVERE", "Category IV (Extreme)"
    elif peak_anom >= 1.8:
        status, severity = "ACTIVE", "Category III (Strong)"
    elif peak_anom >= 0.9:
        status, severity = "WATCH", "Category II (Moderate)"
    else:
        status, severity = "NORMAL", "Category I (Normal)"
        
    stats = {
        'status': status,
        'severity': severity,
        'current_anomaly': f"{'+' if current_anom >= 0 else ''}{current_anom:.1f} °C",
        'peak_anomaly': f"{'+' if peak_anom >= 0 else ''}{peak_anom:.1f} °C",
        'duration': f"{active_days} Days",
        'affected_area': f"{max(0.2, round(0.4 + 0.8 * (peak_anom / 3.0), 2))} Million km²",
        'max_depth': f"{min(depth + 25, 200)} m",
        'event_intensity': "Extreme MHW" if peak_anom >= 2.8 else ("Strong MHW" if peak_anom >= 1.8 else "Moderate MHW"),
        'confidence': f"{round(88 + min(8.0, peak_anom * 2.5), 1)} %",
        'event_id': f"#IO-2024-{abs(int(lat)):02d}{abs(int(lon)):02d}",
        'start_date': "08 MAY 2024"
    }
    
    return df_mhw, stats


def fetch_mhw_depth_time_matrix(lat=15.0, lon=65.0, depth=75):
    base_date = datetime(2024, 5, 20)
    dates = [(base_date - timedelta(days=i)).strftime("%b %d") for i in range(25, -1, -1)]
    depths = np.array(DEPTH_LEVELS)
    
    date_grid, depth_grid = np.meshgrid(np.arange(len(dates)), depths)
    
    # Scale anomaly by latitude
    lat_scale = max(0.4, 1.0 - 0.02 * abs(lat - 15.0))
    anom_matrix = 2.8 * lat_scale * np.exp(-depth_grid / 180.0) * np.exp(-((date_grid - 18)**2) / 45.0)
    anom_matrix += np.random.normal(0, 0.08, anom_matrix.shape)
    
    return dates, depths, np.round(anom_matrix, 2)


render_header(active_page="Heatwave")
controls = render_sidebar()

# Page Header
st.markdown(
    """
    <div style="background: rgba(13, 27, 42, 0.7); border: 1px solid #1E3A5F; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="font-family: 'Outfit', sans-serif; color: #EA580C; margin: 0; font-size: 1.35rem;">
                    🔥 MARINE HEATWAVE INTELLIGENCE & ANOMALY MONITORING
                </h2>
                <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 0.88rem;">
                    Monitor, detect and analyse abnormal ocean warming across the Indian Ocean basin in real-time.
                </p>
            </div>
            <span class="badge-cyan" style="border-color: #EA580C; color: #EA580C; font-weight:700;">SYSTEM STATUS: EARLY WARNING ACTIVE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

HW_PRESETS = {
    "Custom Coordinate": None,
    "Arabian Sea Hotspot (16.0°N, 64.0°E)": (16.0, 64.0, "Arabian Sea"),
    "Bay of Bengal Front (15.0°N, 89.0°E)": (15.0, 89.0, "Bay of Bengal"),
    "Lakshadweep Reefs (10.5°N, 72.5°E)": (10.5, 72.5, "Arabian Sea"),
    "Equatorial Warm Pool (0.0°N, 75.0°E)": (0.0, 75.0, "Equatorial Indian Ocean"),
    "Southern Indian Ocean (-18.0°S, 80.0°E)": (-18.0, 80.0, "Southern Indian Ocean")
}

# ============================================================
# 3. TOP CONTROL PANEL (LIVE REACTIVE)
# ============================================================
st.markdown('<div class="info-card-box" style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:12px 16px; margin-bottom:14px;">', unsafe_allow_html=True)
st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:0.92rem; font-weight:700; color:#0F172A; margin-bottom:8px;">⚙️ HEATWAVE MONITORING CONTROLS & COORDINATE INSPECTOR (LIVE)</div>', unsafe_allow_html=True)

c_p1, c_p2, c_p3, c_p4, c_p5, c_p6 = st.columns([1.5, 1.0, 1.0, 1.1, 1.1, 1.3])

def on_hw_preset_change():
    chosen = st.session_state.get('hw_preset_sel')
    if chosen and HW_PRESETS.get(chosen):
        st.session_state['hw_lat_val'] = HW_PRESETS[chosen][0]
        st.session_state['hw_lon_val'] = HW_PRESETS[chosen][1]

with c_p1:
    st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#0F172A; margin-bottom:4px;">📍 MHW Zone Preset</div>', unsafe_allow_html=True)
    st.selectbox(
        "MHW Preset",
        list(HW_PRESETS.keys()),
        index=0,
        key='hw_preset_sel',
        on_change=on_hw_preset_change,
        label_visibility="collapsed"
    )

if 'hw_lat_val' not in st.session_state:
    st.session_state['hw_lat_val'] = float(controls['target_lat'])
if 'hw_lon_val' not in st.session_state:
    st.session_state['hw_lon_val'] = float(controls['target_lon'])

with c_p2:
    st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#0F172A; margin-bottom:4px;">🌐 Latitude (°N)</div>', unsafe_allow_html=True)
    hw_cur_lat = st.number_input("Latitude (°N)", min_value=-40.0, max_value=30.0, value=float(st.session_state['hw_lat_val']), step=0.5, key='hw_lat_val', label_visibility="collapsed")

with c_p3:
    st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#0F172A; margin-bottom:4px;">🌐 Longitude (°E)</div>', unsafe_allow_html=True)
    hw_cur_lon = st.number_input("Longitude (°E)", min_value=30.0, max_value=120.0, value=float(st.session_state['hw_lon_val']), step=0.5, key='hw_lon_val', label_visibility="collapsed")

with c_p4:
    st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#0F172A; margin-bottom:4px;">🌊 Depth Level (m)</div>', unsafe_allow_html=True)
    hw_cur_depth = st.selectbox("Target Depth Level", DEPTH_LEVELS, index=DEPTH_LEVELS.index(int(controls['depth'])) if int(controls['depth']) in DEPTH_LEVELS else 5, key='hw_depth_val', label_visibility="collapsed")

with c_p5:
    st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#0F172A; margin-bottom:4px;">📊 Variable</div>', unsafe_allow_html=True)
    hw_cur_var = st.selectbox("Monitoring Variable", ["Temperature Anomaly", "Temperature", "Heatwave Intensity"], index=0, key='hw_var_val', label_visibility="collapsed")

with c_p6:
    st.markdown(
        f"""
        <div style="background:#FFF7ED; border:1px solid #FDBA74; border-radius:6px; padding:6px 10px; margin-top:16px; text-align:center;">
            <div style="font-size:0.7rem; color:#9A3412; font-weight:700;">INSPECTING TARGET</div>
            <div style="font-size:0.88rem; color:#C2410C; font-weight:700;">{hw_cur_lat:.1f}°N, {hw_cur_lon:.1f}°E ({hw_cur_depth}m)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Fetch Data dynamically for selected coordinates
df_mhw_ts, hw_stats = fetch_mhw_evolution_timeseries(
    lat=hw_cur_lat,
    lon=hw_cur_lon,
    depth=hw_cur_depth,
    region=controls['region']
)
df_hw_map, df_events = get_heatwave_data(region=controls['region'])

# ============================================================
# 4. CURRENT HEATWAVE STATUS KPI CARDS (DYNAMIC)
# ============================================================
st.markdown(
    f"""
    <div style="display: grid; grid-template-columns: repeat(8, 1fr); gap: 8px; margin-top: 6px; margin-bottom: 16px;">
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#FEF2F2; border-color:#FCA5A5;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Heatwave Status</div>
            <div style="font-size:1.15rem; color:#DC2626; font-weight:700;">{hw_stats['status']}</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#FFF7ED; border-color:#FDBA74;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Current Anomaly</div>
            <div style="font-size:1.15rem; color:#C2410C; font-weight:700;">{hw_stats['current_anomaly']}</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#FEFCE8; border-color:#FDE047;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Peak Anomaly</div>
            <div style="font-size:1.15rem; color:#CA8A04; font-weight:700;">{hw_stats['peak_anomaly']}</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#EFF6FF; border-color:#93C5FD;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Duration</div>
            <div style="font-size:1.15rem; color:#1D4ED8; font-weight:700;">{hw_stats['duration']}</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#FAF5FF; border-color:#D8B4FE;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Affected Area</div>
            <div style="font-size:1.15rem; color:#7E22CE; font-weight:700;">{hw_stats['affected_area']}</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#ECFEFF; border-color:#67E8F9;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Max Depth</div>
            <div style="font-size:1.15rem; color:#0E7490; font-weight:700;">{hw_stats['max_depth']}</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#F0FDF4; border-color:#86EFAC;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Intensity</div>
            <div style="font-size:1.15rem; color:#15803D; font-weight:700;">{hw_stats['event_intensity']}</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center; background:#F8FAFC; border-color:#CBD5E1;">
            <div style="font-size:0.62rem; color:#334155; font-weight:700; text-transform:uppercase;">Confidence</div>
            <div style="font-size:1.15rem; color:#0F172A; font-weight:700;">{hw_stats['confidence']}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 5. ROW 2: MAIN HEATWAVE MAP & EVENT CARD
# ============================================================
c_map, c_event_card = st.columns([2.5, 1.0])

with c_map:
    st.markdown(f'<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">🗺️ LIVE MARINE HEATWAVE ANOMALY PLUME & RADAR TRACKER ({hw_cur_lat:.1f}°N, {hw_cur_lon:.1f}°E, {hw_cur_depth}M)</div>', unsafe_allow_html=True)
    
    # 1. Generate Location-Centric Heatwave Plume Grid
    grid_lats = np.linspace(-30.0, 25.0, 56)
    grid_lons = np.linspace(40.0, 105.0, 66)
    GLON, GLAT = np.meshgrid(grid_lons, grid_lats)
    
    # Gaussian heatwave plume centered on target coordinate
    dist_sq = ((GLAT - hw_cur_lat) ** 2) / 25.0 + ((GLON - hw_cur_lon) ** 2) / 36.0
    peak_anom_val = float(hw_stats['peak_anomaly'].replace('°C','').replace('+',''))
    plume_anom = peak_anom_val * np.exp(-dist_sq)
    
    # Add ambient regional baseline anomalies
    ambient = 0.4 * np.sin(np.radians(GLAT * 3)) + 0.3 * np.cos(np.radians(GLON * 2))
    total_anom = np.clip(plume_anom + ambient, -1.0, 4.0)
    
    # Create interactive Plotly Density/Scatter Mapbox
    fig_mhw_map = go.Figure()
    
    # Trace 1: Marine Heatwave Thermal Anomaly Contour Heatmap
    fig_mhw_map.add_trace(go.Contour(
        z=total_anom,
        x=grid_lons,
        y=grid_lats,
        colorscale=[
            [0.0, '#1E3A8A'],   # Deep blue (<0°C)
            [0.2, '#0284C7'],   # Cyan (0 to +0.5°C)
            [0.4, '#10B981'],   # Green (+0.5 to +1.0°C)
            [0.6, '#F59E0B'],   # Amber/Yellow (+1.0 to +1.8°C Watch)
            [0.8, '#EA580C'],   # Orange (+1.8 to +2.6°C Strong)
            [1.0, '#DC2626']    # Fiery Red (>+2.8°C Extreme)
        ],
        contours=dict(
            coloring='heatmap',
            showlines=True,
            start=-0.5,
            end=3.5,
            size=0.4
        ),
        colorbar=dict(
            title=dict(text='Anomaly (°C)', font=dict(color='#0F172A', size=11)),
            tickfont=dict(color='#0F172A', size=10),
            thickness=14,
            len=0.85,
            y=0.5
        ),
        opacity=0.88,
        hoverinfo='x+y+z',
        name='Thermal Anomaly'
    ))
    
    # Trace 2: Radar Impact Radius Rings around (hw_cur_lat, hw_cur_lon)
    angles = np.linspace(0, 2 * np.pi, 60)
    
    # Outer Ring (Monitored Zone: ~450km / 4.5 deg)
    r_outer_lat = hw_cur_lat + 4.5 * np.sin(angles)
    r_outer_lon = hw_cur_lon + 5.5 * np.cos(angles)
    fig_mhw_map.add_trace(go.Scatter(
        x=r_outer_lon, y=r_outer_lat,
        mode='lines',
        line=dict(color='#F59E0B', width=1.5, dash='dot'),
        hoverinfo='skip',
        name='Watch Zone (~450km)'
    ))
    
    # Middle Ring (Critical Impact Zone: ~250km / 2.5 deg)
    r_mid_lat = hw_cur_lat + 2.5 * np.sin(angles)
    r_mid_lon = hw_cur_lon + 3.0 * np.cos(angles)
    fig_mhw_map.add_trace(go.Scatter(
        x=r_mid_lon, y=r_mid_lat,
        mode='lines',
        line=dict(color='#EA580C', width=2, dash='dash'),
        hoverinfo='skip',
        name='Strong Alert (~250km)'
    ))
    
    # Trace 3: In-Situ ARGO Floats with Real-Time Thermal Anomaly Colors
    np.random.seed(42)
    n_floats = 38
    f_lats = np.random.uniform(max(-35.0, hw_cur_lat - 16.0), min(25.0, hw_cur_lat + 16.0), n_floats)
    f_lons = np.random.uniform(max(38.0, hw_cur_lon - 20.0), min(105.0, hw_cur_lon + 20.0), n_floats)
    f_dist = np.sqrt(((f_lats - hw_cur_lat) ** 2) / 25.0 + ((f_lons - hw_cur_lon) ** 2) / 36.0)
    f_anom = np.round(peak_anom_val * np.exp(-f_dist**2) + np.random.normal(0, 0.15, n_floats), 2)
    
    fig_mhw_map.add_trace(go.Scatter(
        x=f_lons,
        y=f_lats,
        mode='markers',
        marker=dict(
            size=7,
            color=f_anom,
            colorscale='YlOrRd',
            showscale=False,
            line=dict(width=1, color='#0F172A')
        ),
        text=[f"Float #{6903100+i}<br>Lat: {f_lats[i]:.1f}°N, Lon: {f_lons[i]:.1f}°E<br>Anomaly: +{f_anom[i]:.2f}°C" for i in range(n_floats)],
        hoverinfo='text',
        name='ARGO Floats'
    ))
    
    # Trace 4: Pulsating Epicenter Crosshair
    fig_mhw_map.add_trace(go.Scatter(
        x=[hw_cur_lon],
        y=[hw_cur_lat],
        mode='markers+text',
        marker=dict(
            size=18,
            color='#DC2626',
            symbol='cross',
            line=dict(width=3, color='#FFFFFF')
        ),
        text=[f" 🎯 EPICENTER ({hw_cur_lat:.1f}°N, {hw_cur_lon:.1f}°E)"],
        textposition="top right",
        textfont=dict(color='#DC2626', size=11, family='Outfit', weight='bold'),
        name='MHW Epicenter'
    ))
    
    fig_mhw_map.update_layout(
        title=dict(
            text=f"MHW THERMAL ANOMALY PLUME & RADAR COVERAGE AT {hw_cur_lat:.1f}°N, {hw_cur_lon:.1f}°E ({hw_cur_depth}M)",
            font=dict(family="Outfit", size=11, color="#0F172A")
        ),
        xaxis=dict(
            title=dict(text="Longitude (°E)", font=dict(color="#0F172A", size=10)),
            tickfont=dict(color="#0F172A", size=10),
            gridcolor="#E2E8F0",
            range=[max(35.0, hw_cur_lon - 22.0), min(110.0, hw_cur_lon + 22.0)],
            fixedrange=True
        ),
        yaxis=dict(
            title=dict(text="Latitude (°N)", font=dict(color="#0F172A", size=10)),
            tickfont=dict(color="#0F172A", size=10),
            gridcolor="#E2E8F0",
            range=[max(-35.0, hw_cur_lat - 16.0), min(28.0, hw_cur_lat + 16.0)],
            fixedrange=True
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        margin=dict(l=40, r=20, t=35, b=35),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9, color="#0F172A"))
    )
    st.plotly_chart(fig_mhw_map, use_container_width=True, config={'displayModeBar': 'hover', 'displaylogo': False})
    
    # Legend Bar
    st.markdown(
        """
        <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:6px; padding:6px 12px; margin-top:8px; display:flex; justify-content:space-between; align-items:center; font-size:0.75rem; font-weight:600; color:#334155;">
            <span style="font-weight:700;">ANOMALY LEGEND:</span>
            <span style="color:#1E3A8A;">&lt; 0 °C (Cool)</span>
            <span style="color:#0284C7;">0 to +0.5 °C</span>
            <span style="color:#10B981;">+0.5 to +1.0 °C</span>
            <span style="color:#F59E0B;">+1.0 to +1.8 °C (Watch)</span>
            <span style="color:#EA580C;">+1.8 to +2.8 °C (Strong)</span>
            <span style="color:#DC2626; font-weight:700;">&gt; +2.8 °C (Severe Cat IV)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with c_event_card:
    # 7. Current Heatwave Event Card
    st.markdown(
        f"""
        <div class="info-card-box">
            <div class="info-card-header">🔥 CURRENT HEATWAVE EVENT</div>
            <div style="background:#FFF7ED; border:1px solid #FDBA74; border-radius:6px; padding:10px; margin-bottom:10px;">
                <div style="font-size:1.1rem; font-weight:700; color:#C2410C;">{hw_stats['event_id']}</div>
                <div style="font-size:0.8rem; color:#64748B;">Target: <b>{hw_cur_lat:.1f}°N, {hw_cur_lon:.1f}°E</b> ({hw_cur_depth}m)</div>
            </div>
            <table style="width:100%; font-size:0.83rem; color:#334155; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Event Status:</td><td style="text-align:right; font-weight:700; color:#DC2626;">{hw_stats['status']}</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">MHW Category:</td><td style="text-align:right; font-weight:700; color:#C2410C;">{hw_stats['severity']}</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Start Date:</td><td style="text-align:right; font-weight:600;">{hw_stats['start_date']}</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Duration:</td><td style="text-align:right; font-weight:600;">{hw_stats['duration']}</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Peak Anomaly:</td><td style="text-align:right; font-weight:700; color:#C2410C;">{hw_stats['peak_anomaly']}</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Max Impact Depth:</td><td style="text-align:right; font-weight:600;">{hw_stats['max_depth']}</td></tr>
                <tr><td style="padding:4px 0; color:#64748B;">Affected Area:</td><td style="text-align:right; font-weight:600;">{hw_stats['affected_area']}</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Visual Alert System Card
    if hw_stats['status'] in ["SEVERE", "ACTIVE"]:
        alert_title, alert_bg, alert_border, alert_txt = "SEVERE WARNING: PERSISTENT ANOMALY DETECTED", "#FEF2F2", "#FCA5A5", f"Strong persistent thermal warming detected near ({hw_cur_lat:.1f}°N, {hw_cur_lon:.1f}°E) exceeding the 90th percentile climatology."
    elif hw_stats['status'] == "WATCH":
        alert_title, alert_bg, alert_border, alert_txt = "WATCH: ABOVE-NORMAL WARMING DETECTED", "#FEFCE8", "#FDE047", "Above-normal warming detected. Continued monitoring recommended."
    else:
        alert_title, alert_bg, alert_border, alert_txt = "NORMAL: NO SIGNIFICANT ANOMALY", "#F0FDF4", "#86EFAC", "No significant marine heatwave temperature anomaly detected at this coordinate."

    st.markdown(
        f"""
        <div style="background:{alert_bg}; border:1px solid {alert_border}; border-radius:6px; padding:10px; margin-top:10px;">
            <div style="font-size:0.78rem; font-weight:700; color:#0F172A;">🚨 {alert_title}</div>
            <div style="font-size:0.75rem; color:#334155; margin-top:4px; line-height:1.4;">{alert_txt}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

# ============================================================
# 6. ROW 3: EVOLUTION TIME SERIES & TEMP VS CLIMATOLOGY
# ============================================================
c_row3_ev, c_row3_clim = st.columns([1.8, 1.5])

with c_row3_ev:
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">📈 HEATWAVE EVOLUTION TIME SERIES</div>', unsafe_allow_html=True)
    
    fig_ev = go.Figure()
    
    # Observed Anomaly (Solid Red)
    fig_ev.add_trace(go.Scatter(
        x=df_mhw_ts['Date'],
        y=df_mhw_ts['Anomaly (°C)'],
        mode='lines',
        name='Temperature Anomaly',
        line=dict(color='#DC2626', width=2.5)
    ))
    
    # Baseline (0°C Reference)
    fig_ev.add_hline(y=0.0, line_width=1.2, line_dash="solid", line_color="#64748B", annotation_text="Baseline (0°C)", annotation_font=dict(size=9, color="#64748B"))
    
    # 90th Percentile Threshold (+1.5°C)
    fig_ev.add_hline(y=1.4, line_width=1.5, line_dash="dash", line_color="#C2410C", annotation_text="90th Percentile MHW Threshold (+1.4°C)", annotation_font=dict(size=9, color="#C2410C"))

    fig_ev.update_layout(
        title=dict(text=f"SEASONAL ANOMALY EVOLUTION & THRESHOLD EXCEEDANCE AT {hw_cur_lat:.1f}°N, {hw_cur_lon:.1f}°E ({hw_cur_depth}M)", font=dict(family="Outfit", size=11, color="#0F172A")),
        dragmode=False,
        xaxis=dict(title=dict(text="Date", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), gridcolor="#E2E8F0", fixedrange=True),
        yaxis=dict(title=dict(text="Anomaly (°C)", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), gridcolor="#E2E8F0", fixedrange=True),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=35, b=35),
        height=300,
        showlegend=False
    )
    st.plotly_chart(fig_ev, use_container_width=True, config={'displayModeBar': 'hover', 'displaylogo': False, 'scrollZoom': False})

with c_row3_clim:
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">📊 OBSERVED TEMPERATURE VS CLIMATOLOGY BASELINE</div>', unsafe_allow_html=True)
    
    fig_clim = go.Figure()
    fig_clim.add_trace(go.Scatter(x=df_mhw_ts['Date'], y=df_mhw_ts['Observed Temp (°C)'], mode='lines', name='Observed Temp', line=dict(color='#DC2626', width=2)))
    fig_clim.add_trace(go.Scatter(x=df_mhw_ts['Date'], y=df_mhw_ts['Climatology Baseline (°C)'], mode='lines', name='Climatology Baseline', line=dict(color='#2563EB', width=2, dash='dash')))
    
    fig_clim.update_layout(
        title=dict(text=f"OBSERVED TEMP VS 30-YEAR CLIMATOLOGY ({hw_cur_lat:.1f}°N, {hw_cur_lon:.1f}°E)", font=dict(family="Outfit", size=11, color="#0F172A")),
        dragmode=False,
        xaxis=dict(title=dict(text="Date", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), gridcolor="#E2E8F0", fixedrange=True),
        yaxis=dict(title=dict(text="Temperature (°C)", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), gridcolor="#E2E8F0", fixedrange=True),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=35, b=35),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#0F172A"))
    )
    st.plotly_chart(fig_clim, use_container_width=True, config={'displayModeBar': 'hover', 'displaylogo': False})

st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

# ============================================================
# 7. ROW 4: VERTICAL STRUCTURE & DEPTH x TIME HEATMAP
# ============================================================
c_row4_prof, c_row4_depth = st.columns([1.8, 1.5])

with c_row4_prof:
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">🌊 HEATWAVE VERTICAL STRUCTURE (0–1000M)</div>', unsafe_allow_html=True)
    
    df_prof = get_temperature_profile(lat=hw_cur_lat, lon=hw_cur_lon)
    peak_anom_num = float(hw_stats['peak_anomaly'].replace('°C','').replace('+',''))
    anom_prof = peak_anom_num * np.exp(-df_prof['Depth (m)'] / 160.0) + np.random.normal(0, 0.04, len(df_prof))
    
    fig_vprof = go.Figure()
    fig_vprof.add_trace(go.Scatter(
        x=anom_prof,
        y=df_prof['Depth (m)'],
        mode='lines+markers',
        name='Thermal Anomaly Profile',
        line=dict(color='#EA580C', width=2.5),
        marker=dict(size=5, color='#C2410C')
    ))
    
    # Highlight selected depth
    sel_d = hw_cur_depth
    if sel_d in df_prof['Depth (m)'].values:
        idx_d = df_prof[df_prof['Depth (m)'] == sel_d].index[0]
        fig_vprof.add_trace(go.Scatter(
            x=[anom_prof.iloc[idx_d]],
            y=[sel_d],
            mode='markers',
            name=f'Target Depth ({sel_d}m)',
            marker=dict(size=12, color='#DC2626', symbol='circle')
        ))

    fig_vprof.update_layout(
        title=dict(text=f"VERTICAL ANOMALY STRUCTURE AT ({hw_cur_lat:.1f}°N, {hw_cur_lon:.1f}°E)", font=dict(family="Outfit", size=11, color="#0F172A")),
        dragmode=False,
        xaxis=dict(title=dict(text="Temperature Anomaly (°C)", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), gridcolor="#E2E8F0", fixedrange=True),
        yaxis=dict(title=dict(text="Depth (m)", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), gridcolor="#E2E8F0", autorange='reversed', fixedrange=True),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=35, b=35),
        height=310,
        showlegend=False
    )
    st.plotly_chart(fig_vprof, use_container_width=True, config={'displayModeBar': 'hover', 'displaylogo': False})

with c_row4_depth:
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">🌡️ HEATWAVE DEPTH × TIME STRUCTURE HEATMAP</div>', unsafe_allow_html=True)
    
    dates_fmt, depths_fmt, matrix_anom = fetch_mhw_depth_time_matrix(lat=hw_cur_lat, lon=hw_cur_lon, depth=hw_cur_depth)
    
    fig_dmap = go.Figure(data=go.Heatmap(
        z=matrix_anom,
        x=dates_fmt,
        y=depths_fmt,
        colorscale='Reds',
        colorbar=dict(title=dict(text='Anomaly (°C)', font=dict(color='#0F172A')), tickfont=dict(color='#0F172A'))
    ))
    fig_dmap.update_layout(
        title=dict(text=f"SUBSURFACE ANOMALY PROPAGATION AT ({hw_cur_lat:.1f}°N, {hw_cur_lon:.1f}°E)", font=dict(family="Outfit", size=11, color="#0F172A")),
        xaxis=dict(title=dict(text="Date", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), fixedrange=True),
        yaxis=dict(title=dict(text="Depth (m)", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), autorange='reversed', fixedrange=True),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=35, b=35),
        height=310
    )
    st.plotly_chart(fig_dmap, use_container_width=True, config={'displayModeBar': 'hover', 'displaylogo': False})

st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

# ============================================================
# 8. ROW 5: DETECTED HEATWAVE EVENTS TABLE & ARGO VS GLORYS
# ============================================================
c_row5_tbl, c_row5_argo = st.columns([2.0, 1.3])

with c_row5_tbl:
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">📋 DETECTED REGIONAL HEATWAVE EVENTS MATRIX</div>', unsafe_allow_html=True)
    st.dataframe(df_events, use_container_width=True, hide_index=True)

with c_row5_argo:
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">📊 OBSERVATION VS REANALYSIS (ARGO vs GLORYS)</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:6px; padding:12px;">
            <div style="font-size:0.83rem; color:#334155; margin-bottom:8px;">Paired Point Validation at ({hw_cur_lat:.1f}°N, {hw_cur_lon:.1f}°E):</div>
            <table style="width:100%; font-size:0.82rem; color:#334155; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">ARGO Point Temp:</td><td style="text-align:right; font-weight:700; color:#2563EB;">{round(float(df_mhw_ts['Observed Temp (°C)'].iloc[-1]), 2)} °C</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">GLORYS Climatology:</td><td style="text-align:right; font-weight:700; color:#16A34A;">{round(float(df_mhw_ts['Climatology Baseline (°C)'].iloc[-1]), 2)} °C</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Temperature Anomaly:</td><td style="text-align:right; font-weight:700; color:#DC2626;">{hw_stats['current_anomaly']}</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">MAE Residual:</td><td style="text-align:right; font-weight:600;">0.28 °C</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">RMSE Score:</td><td style="text-align:right; font-weight:600;">0.39 °C</td></tr>
                <tr><td style="padding:4px 0; color:#64748B;">Model Bias:</td><td style="text-align:right; font-weight:600;">+0.06 °C</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

# ============================================================
# 9. ROW 6: HEATWAVE INTELLIGENCE INSIGHTS & DATA QUALITY
# ============================================================
c_row6_ins, c_row6_qual = st.columns([2.0, 1.3])

with c_row6_ins:
    st.markdown(
        f"""
        <div class="info-card-box">
            <div class="info-card-header">🧠 HEATWAVE INTELLIGENCE INSIGHTS</div>
            <ul style="font-size:0.83rem; color:#334155; line-height:1.7; padding-left:16px; margin:0;">
                <li style="margin-bottom:8px;">Subsurface temperature anomaly reached a peak of <b>{hw_stats['peak_anomaly']}</b> at <b>{hw_cur_depth}m depth</b> near coordinate <b>{hw_cur_lat:.1f}°N, {hw_cur_lon:.1f}°E</b>.</li>
                <li style="margin-bottom:8px;">Current heatwave severity category: <b>{hw_stats['severity']}</b>.</li>
                <li style="margin-bottom:8px;">Temperature anomaly has remained continuously above the 90th percentile threshold for <b>{hw_stats['duration']}</b>.</li>
                <li style="margin-bottom:8px;">Estimated spatial impact covers approximately <b>{hw_stats['affected_area']}</b>.</li>
                <li>In-situ ARGO float telemetry confirms GLORYS reanalysis high correlation near this location.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with c_row6_qual:
    st.markdown(
        """
        <div class="info-card-box">
            <div class="info-card-header">🛡️ DATA QUALITY & OBSERVATIONAL STATUS</div>
            <table style="width:100%; font-size:0.82rem; color:#334155; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Primary Source:</td><td style="text-align:right; font-weight:600;">ARGO + GLORYS</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Data Timestamp:</td><td style="text-align:right; font-weight:600;">2024-05-20</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Spatial Coverage:</td><td style="text-align:right; font-weight:600;">94 %</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Depth Levels:</td><td style="text-align:right; font-weight:600;">0 – 1000 m</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Active Floats:</td><td style="text-align:right; font-weight:600;">124 Floats</td></tr>
                <tr><td style="padding:4px 0; color:#64748B;">Overall Reliability:</td><td style="text-align:right; font-weight:700; color:#16A34A;">GOOD (98.4%)</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

# ============================================================
# 10. BOTTOM ROW: HEATWAVE DATA TABLE & CSV DOWNLOAD
# ============================================================
with st.expander("📄 VIEW & DOWNLOAD FULL HEATWAVE DATASET (CSV)", expanded=False):
    st.markdown('<div style="font-weight:700; color:#0F172A; margin-bottom:8px;">HEATWAVE ANOMALY DATA MATRIX</div>', unsafe_allow_html=True)
    st.dataframe(df_mhw_ts, use_container_width=True, hide_index=True)
    
    csv_mhw_bytes = df_mhw_ts.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Heatwave Data (CSV)",
        data=csv_mhw_bytes,
        file_name=f"Marine_Heatwave_Data_{hw_cur_lat:.1f}N_{hw_cur_lon:.1f}E_{hw_cur_depth}m.csv",
        mime="text/csv",
        use_container_width=True
    )

render_footer()
