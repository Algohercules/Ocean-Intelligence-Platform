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
    page_title="Heatwave | Indian Ocean Intelligence Platform",
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
def fetch_mhw_evolution_timeseries(region="Arabian Sea", depth=75):
    dates = pd.date_range(start="2024-01-01", end="2024-05-20", freq="D")
    t = np.linspace(0, len(dates), len(dates))
    
    depth_factor = 26.5 - 0.015 * depth
    climatology = depth_factor + 1.2 * np.sin(2 * np.pi * t / 365.0)
    thresh_90th = climatology + 1.5
    
    # Heatwave spike around mid-April / May
    spike = 3.4 * np.exp(-((t - 115)**2) / 140.0)
    observed = climatology + spike + np.random.normal(0, 0.15, len(dates))
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
    
    if peak_anom >= 3.0:
        status, severity = "SEVERE", "Category IV (Extreme)"
    elif peak_anom >= 2.0:
        status, severity = "ACTIVE", "Category III (Strong)"
    elif peak_anom >= 1.0:
        status, severity = "WATCH", "Category II (Moderate)"
    else:
        status, severity = "NO EVENT", "Category I (Normal)"
        
    stats = {
        'status': status,
        'severity': severity,
        'current_anomaly': f"{'+' if current_anom >= 0 else ''}{current_anom:.1f} °C",
        'peak_anomaly': f"{'+' if peak_anom >= 0 else ''}{peak_anom:.1f} °C",
        'duration': f"{active_days} Days",
        'affected_area': "1.24 Million km²",
        'max_depth': f"{min(depth + 25, 200)} m",
        'event_intensity': "Moderate to Strong",
        'confidence': "92 %",
        'event_id': "#IO-2024-05",
        'start_date': "08 MAY 2024"
    }
    
    return df_mhw, stats


def fetch_mhw_depth_time_matrix(horizon_days=30):
    base_date = datetime(2024, 5, 20)
    dates = [(base_date - timedelta(days=i)).strftime("%b %d") for i in range(25, -1, -1)]
    depths = np.array(DEPTH_LEVELS)
    
    date_grid, depth_grid = np.meshgrid(np.arange(len(dates)), depths)
    anom_matrix = 2.8 * np.exp(-depth_grid / 180.0) * np.exp(-((date_grid - 18)**2) / 45.0)
    anom_matrix += np.random.normal(0, 0.1, anom_matrix.shape)
    
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
                    Monitor, detect and analyse abnormal ocean warming across the Indian Ocean basin.
                </p>
            </div>
            <span class="badge-cyan" style="border-color: #EA580C; color: #EA580C; font-weight:700;">SYSTEM STATUS: EARLY WARNING ACTIVE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Session state initialization for Heatwave controls
if 'hw_region' not in st.session_state:
    st.session_state['hw_region'] = controls['region']
if 'hw_depth' not in st.session_state:
    st.session_state['hw_depth'] = int(controls['depth'])
if 'hw_variable' not in st.session_state:
    st.session_state['hw_variable'] = "Temperature Anomaly"
if 'hw_source' not in st.session_state:
    st.session_state['hw_source'] = "ARGO + GLORYS"

# ============================================================
# 3. TOP CONTROL PANEL
# ============================================================
with st.form("heatwave_control_panel_form"):
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:0.92rem; font-weight:700; color:#0F172A; margin-bottom:8px;">⚙️ HEATWAVE MONITORING & REGIONAL SELECTION CONTROLS</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5 = st.columns([1.4, 1.1, 1.2, 1.2, 1.2])
    with c1:
        st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#0F172A; margin-bottom:4px;">📍 Target Region</div>', unsafe_allow_html=True)
        in_region = st.selectbox("Target Region", ["All Indian Ocean", "Arabian Sea", "Bay of Bengal", "Equatorial Indian Ocean", "Custom Region"], index=1 if controls['region']=="Arabian Sea" else 0, label_visibility="collapsed")
    with c2:
        st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#0F172A; margin-bottom:4px;">🌊 Depth Level (m)</div>', unsafe_allow_html=True)
        in_depth = st.selectbox("Target Depth Level", DEPTH_LEVELS, index=DEPTH_LEVELS.index(st.session_state['hw_depth']) if st.session_state['hw_depth'] in DEPTH_LEVELS else 5, label_visibility="collapsed")
    with c3:
        st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#0F172A; margin-bottom:4px;">📊 Monitoring Variable</div>', unsafe_allow_html=True)
        in_var = st.selectbox("Monitoring Variable", ["Temperature", "Temperature Anomaly", "Heatwave Intensity"], index=1, label_visibility="collapsed")
    with c4:
        st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#0F172A; margin-bottom:4px;">🛡️ Data Source</div>', unsafe_allow_html=True)
        in_src = st.selectbox("Data Source", ["ARGO", "GLORYS", "ARGO + GLORYS", "Copernicus"], index=2, label_visibility="collapsed")
    with c5:
        st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
        btn_hw_run = st.form_submit_button("🚀 ANALYZE HEATWAVE", use_container_width=True)

if btn_hw_run:
    st.session_state['hw_region'] = in_region
    st.session_state['hw_depth'] = in_depth
    st.session_state['hw_variable'] = in_var
    st.session_state['hw_source'] = in_src

# Fetch Data for selected controls
df_mhw_ts, hw_stats = fetch_mhw_evolution_timeseries(
    region=st.session_state['hw_region'],
    depth=st.session_state['hw_depth']
)
df_hw_map, df_events = get_heatwave_data(region=st.session_state['hw_region'])

# ============================================================
# 4. CURRENT HEATWAVE STATUS KPI CARDS
# ============================================================
st.markdown(
    f"""
    <div style="display: grid; grid-template-columns: repeat(8, 1fr); gap: 8px; margin-top: 10px; margin-bottom: 16px;">
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
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">🗺️ HEATWAVE INTENSITY — INDIAN OCEAN</div>', unsafe_allow_html=True)
    
    render_ocean_map(
        dataset="ARGO Observations",
        variable="Temperature",
        depth=st.session_state['hw_depth'],
        date_str=str(controls['date']),
        region=st.session_state['hw_region'],
        target_lat=controls['target_lat'],
        target_lon=controls['target_lon'],
        show_floats=True,
        show_heatmap=True
    )
    
    # Legend Bar
    st.markdown(
        """
        <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:6px; padding:6px 12px; margin-top:8px; display:flex; justify-content:space-between; align-items:center; font-size:0.75rem; font-weight:600; color:#334155;">
            <span style="font-weight:700;">ANOMALY LEGEND:</span>
            <span style="color:#2563EB;">&lt; -2 °C</span>
            <span style="color:#0284C7;">-2 to -1 °C</span>
            <span style="color:#64748B;">-1 to 0 °C</span>
            <span style="color:#EAB308;">0 to +1 °C</span>
            <span style="color:#EA580C;">+1 to +2 °C</span>
            <span style="color:#C2410C;">+2 to +3 °C</span>
            <span style="color:#DC2626; font-weight:700;">&gt; +3 °C</span>
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
                <div style="font-size:0.8rem; color:#64748B;">Region: <b>{st.session_state['hw_region']}</b></div>
            </div>
            <table style="width:100%; font-size:0.83rem; color:#334155; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Event Status:</td><td style="text-align:right; font-weight:700; color:#DC2626;">{hw_stats['status']}</td></tr>
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
    
    # 20. Visual Alert System Card
    if hw_stats['status'] in ["SEVERE", "ACTIVE"]:
        alert_title, alert_bg, alert_border, alert_txt = "SEVERE WARNING: PERSISTENT ANOMALY DETECTED", "#FEF2F2", "#FCA5A5", "Strong persistent thermal warming detected across the selected region exceeding 90th percentile climatology."
    elif hw_stats['status'] == "WATCH":
        alert_title, alert_bg, alert_border, alert_txt = "WATCH: ABOVE-NORMAL WARMING DETECTED", "#FEFCE8", "#FDE047", "Above-normal warming detected. Continued monitoring recommended."
    else:
        alert_title, alert_bg, alert_border, alert_txt = "NORMAL: NO SIGNIFICANT ANOMALY", "#F0FDF4", "#86EFAC", "No significant marine heatwave temperature anomaly detected."

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
    fig_ev.add_hline(y=1.5, line_width=1.5, line_dash="dash", line_color="#C2410C", annotation_text="90th Percentile MHW Threshold (+1.5°C)", annotation_font=dict(size=9, color="#C2410C"))

    fig_ev.update_layout(
        title=dict(text=f"SEASONAL ANOMALY EVOLUTION & THRESHOLD EXCEEDANCE AT {st.session_state['hw_depth']}M", font=dict(family="Outfit", size=11, color="#0F172A")),
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
        title=dict(text="OBSERVED TEMP VS 30-YEAR HISTORICAL CLIMATOLOGY", font=dict(family="Outfit", size=11, color="#0F172A")),
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
    
    df_prof = get_temperature_profile(lat=controls['target_lat'], lon=controls['target_lon'])
    anom_prof = 2.4 * np.exp(-df_prof['Depth (m)'] / 160.0) + np.random.normal(0, 0.05, len(df_prof))
    
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
    sel_d = st.session_state['hw_depth']
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
        title=dict(text=f"VERTICAL ANOMALY STRUCTURE AT ({controls['target_lat']}°N, {controls['target_lon']}°E)", font=dict(family="Outfit", size=11, color="#0F172A")),
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
    
    dates_fmt, depths_fmt, matrix_anom = fetch_mhw_depth_time_matrix()
    
    fig_dmap = go.Figure(data=go.Heatmap(
        z=matrix_anom,
        x=dates_fmt,
        y=depths_fmt,
        colorscale='Reds',
        colorbar=dict(title=dict(text='Anomaly (°C)', font=dict(color='#0F172A')), tickfont=dict(color='#0F172A'))
    ))
    fig_dmap.update_layout(
        title=dict(text="SUBSURFACE ANOMALY PROPAGATION (0–1000M)", font=dict(family="Outfit", size=11, color="#0F172A")),
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
        """
        <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:6px; padding:12px;">
            <div style="font-size:0.83rem; color:#334155; margin-bottom:8px;">Paired Point Validation at Selected Coordinate:</div>
            <table style="width:100%; font-size:0.82rem; color:#334155; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">ARGO Point Temp:</td><td style="text-align:right; font-weight:700; color:#2563EB;">19.4 °C</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">GLORYS Model Temp:</td><td style="text-align:right; font-weight:700; color:#16A34A;">19.28 °C</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Temperature Anomaly:</td><td style="text-align:right; font-weight:700; color:#DC2626;">+1.8 °C</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">MAE Residual:</td><td style="text-align:right; font-weight:600;">0.31 °C</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">RMSE Score:</td><td style="text-align:right; font-weight:600;">0.42 °C</td></tr>
                <tr><td style="padding:4px 0; color:#64748B;">Model Bias:</td><td style="text-align:right; font-weight:600;">+0.08 °C</td></tr>
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
                <li style="margin-bottom:8px;">Subsurface temperature anomaly reached a peak of <b>{hw_stats['peak_anomaly']}</b> at <b>{st.session_state['hw_depth']}m depth</b> in the <b>{st.session_state['hw_region']}</b>.</li>
                <li style="margin-bottom:8px;">The strongest ocean warming is concentrated between <b>50–100 meters</b> depth in the central Arabian Sea.</li>
                <li style="margin-bottom:8px;">Temperature anomaly has remained continuously above the 90th percentile threshold for <b>{hw_stats['duration']}</b>.</li>
                <li style="margin-bottom:8px;">Spatial extent analysis indicates <b>{hw_stats['affected_area']}</b> impacted across the Indian Ocean basin.</li>
                <li>Collocated ARGO float observations confirm GLORYS reanalysis high correlation (R² = 0.94).</li>
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
        file_name=f"Marine_Heatwave_Data_{st.session_state['hw_region'].replace(' ', '_')}_{st.session_state['hw_depth']}m.csv",
        mime="text/csv",
        use_container_width=True
    )

render_footer()
