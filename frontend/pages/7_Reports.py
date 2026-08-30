"""
pages/7_Reports.py
==================
Oceanographic Intelligence Report Builder, Preview & Export Page.
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
    page_title="Reports | Indian Ocean Intelligence Platform",
    page_icon="📄",
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
from components.location_report import render_location_report_ui
from components.footer import render_footer
from data.mock_data import (
    get_avg_temp_by_depth,
    get_point_details,
    get_temperature_profile,
    get_argo_vs_glorys_profile,
    get_time_series_data,
    DEPTH_LEVELS
)

# Initialize Session State for Reports History & Location Report
if 'report_history' not in st.session_state:
    st.session_state['report_history'] = []
if 'current_report' not in st.session_state:
    st.session_state['current_report'] = None
if 'loc_lat' not in st.session_state:
    st.session_state['loc_lat'] = 15.0
if 'loc_lon' not in st.session_state:
    st.session_state['loc_lon'] = 65.0
if 'loc_depth' not in st.session_state:
    st.session_state['loc_depth'] = 75

render_header(active_page="Reports")
controls = render_sidebar()

# Page Header
st.markdown(
    """
    <div style="background: rgba(13, 27, 42, 0.7); border: 1px solid #1E3A5F; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="font-family: 'Outfit', sans-serif; color: #38BDF8; margin: 0; font-size: 1.35rem;">
                    📄 OCEAN INTELLIGENCE REPORTS & LOCATION REPORT GENERATOR
                </h2>
                <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 0.88rem;">
                    Generate point-specific location reports and regional oceanographic intelligence from Indian Ocean observations.
                </p>
            </div>
            <span class="badge-cyan" style="border-color: #38BDF8; color: #38BDF8; font-weight:700;">SYSTEM STATUS: REPORT ENGINE READY</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tab_loc, tab_basin = st.tabs([
    "📍 GENERATE LOCATION REPORT (Point-Based Lat/Lon/Depth)",
    "📄 BASIN & REGIONAL OCEAN INTELLIGENCE REPORT"
])

# ============================================================
# TAB 1: LOCATION REPORT (POINT-BASED LAT / LON / DEPTH)
# ============================================================
with tab_loc:
    with st.form("generate_location_report_form"):
        st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:0.95rem; font-weight:700; color:#0F172A; margin-bottom:12px;">📍 POINT-BASED LOCATION REPORT PARAMETERS</div>', unsafe_allow_html=True)
        
        c_l1, c_l2, c_l3, c_l4, c_l5 = st.columns([1.2, 1.2, 1.1, 1.8, 1.4])
        with c_l1:
            st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#0F172A; margin-bottom:4px;">🌐 Latitude (°N)</div>', unsafe_allow_html=True)
            in_loc_lat = st.number_input("Latitude (°N)", min_value=-30.0, max_value=25.0, value=float(controls['target_lat']), step=0.5, label_visibility="collapsed")
        with c_l2:
            st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#0F172A; margin-bottom:4px;">🌐 Longitude (°E)</div>', unsafe_allow_html=True)
            in_loc_lon = st.number_input("Longitude (°E)", min_value=35.0, max_value=105.0, value=float(controls['target_lon']), step=0.5, label_visibility="collapsed")
        with c_l3:
            st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#0F172A; margin-bottom:4px;">🌊 Depth (m)</div>', unsafe_allow_html=True)
            in_loc_depth = st.selectbox("Depth Level", DEPTH_LEVELS, index=DEPTH_LEVELS.index(int(controls['depth'])) if int(controls['depth']) in DEPTH_LEVELS else 5, label_visibility="collapsed")
        with c_l4:
            st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#0F172A; margin-bottom:4px;">📄 Report Type</div>', unsafe_allow_html=True)
            in_loc_type = st.selectbox("Report Type", ["Location Intelligence Report", "Point Temperature Analysis", "ARGO Float Telemetry Report", "Marine Heatwave Indicator Report", "AI Prediction Point Forecast"], index=0, label_visibility="collapsed")
        with c_l5:
            st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
            btn_gen_loc_rep = st.form_submit_button("🚀 GENERATE REPORT", use_container_width=True)

    if btn_gen_loc_rep:
        st.session_state['loc_lat'] = in_loc_lat
        st.session_state['loc_lon'] = in_loc_lon
        st.session_state['loc_depth'] = in_loc_depth
        st.session_state['loc_type'] = in_loc_type

    if 'loc_type' not in st.session_state:
        st.session_state['loc_type'] = "Location Intelligence Report"

    # Render Point-Based Location Report UI
    render_location_report_ui(
        target_lat=st.session_state['loc_lat'],
        target_lon=st.session_state['loc_lon'],
        target_depth=st.session_state['loc_depth'],
        selected_date=controls['date']
    )

# ============================================================
# TAB 2: BASIN & REGIONAL OCEAN INTELLIGENCE REPORT
# ============================================================
with tab_basin:
    REPORT_TYPES = [
        "1. Ocean State Report",
        "2. Temperature Analysis Report",
        "3. ARGO Observation Report",
        "4. GLORYS Reanalysis Report",
        "5. ARGO vs GLORYS Comparison",
        "6. AI Prediction Report",
        "7. Marine Heatwave Report",
        "8. Regional Ocean Intelligence Report",
        "9. Subsurface Temperature Report",
        "10. Comprehensive Ocean Intelligence Report"
    ]

    with st.form("ocean_report_config_form"):
        st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:0.95rem; font-weight:700; color:#0F172A; margin-bottom:8px;">⚙️ SELECT REPORT TYPE & CONFIGURATION PARAMETERS</div>', unsafe_allow_html=True)
        
        r1_c1, r1_c2, r1_c3 = st.columns([1.8, 1.1, 1.1])
        with r1_c1:
            rep_type = st.selectbox("Report Type", REPORT_TYPES, index=9)
        with r1_c2:
            rep_region = st.selectbox("Target Region", ["All Indian Ocean", "Arabian Sea", "Bay of Bengal", "Equatorial Indian Ocean", "Custom Region"], index=1 if controls['region']=="Arabian Sea" else 0)
        with r1_c3:
            rep_depth = st.selectbox("Subsurface Depth Level", DEPTH_LEVELS, index=DEPTH_LEVELS.index(int(controls['depth'])) if int(controls['depth']) in DEPTH_LEVELS else 5)

        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns([1.0, 1.0, 1.2, 1.2])
        with r2_c1:
            rep_lat = st.number_input("Latitude (°N)", min_value=-40.0, max_value=30.0, value=float(controls['target_lat']), step=0.5, key="b_lat")
        with r2_c2:
            rep_lon = st.number_input("Longitude (°E)", min_value=30.0, max_value=120.0, value=float(controls['target_lon']), step=0.5, key="b_lon")
        with r2_c3:
            rep_start_date = st.date_input("Start Date", value=date(2024, 1, 1))
        with r2_c4:
            rep_end_date = st.date_input("End Date", value=date(2024, 5, 20))

        r3_c1, r3_c2, r3_c3, r3_c4 = st.columns([1.2, 1.2, 1.2, 1.4])
        with r3_c1:
            rep_var = st.selectbox("Primary Variable", ["Temperature", "Salinity", "Current Speed", "Sea Level Anomaly"], index=0)
        with r3_c2:
            rep_src = st.selectbox("Data Sources", ["ARGO + GLORYS", "ARGO Observations", "GLORYS Reanalysis", "Copernicus Marine", "AI Reconstruction"], index=0)
        with r3_c3:
            rep_fmt = st.selectbox("Report Export Format", ["Dashboard Preview", "PDF", "CSV", "Excel"], index=0)
        with r3_c4:
            st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
            btn_gen_rep = st.form_submit_button("🚀 GENERATE REGIONAL REPORT", use_container_width=True)

    if btn_gen_rep:
        with st.spinner("Preparing ocean intelligence report..."):
            now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            rep_id = f"IOI-{now_str}-{np.random.randint(100,999)}"
            gen_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            point_stats = get_point_details(lat=rep_lat, lon=rep_lon, depth=rep_depth)
            
            report_data = {
                'id': rep_id,
                'type': rep_type,
                'region': rep_region,
                'lat': rep_lat,
                'lon': rep_lon,
                'depth': rep_depth,
                'start_date': str(rep_start_date),
                'end_date': str(rep_end_date),
                'variable': rep_var,
                'sources': rep_src,
                'format': rep_fmt,
                'timestamp': gen_timestamp,
                'mean_temp': f"{point_stats['avg_temp']} °C",
                'min_temp': f"{point_stats['min_temp']} °C",
                'max_temp': f"{point_stats['max_temp']} °C",
                'anomaly': f"{'+' if point_stats['anomaly'] >= 0 else ''}{point_stats['anomaly']} °C",
                'coverage': point_stats.get('data_coverage', '94%'),
                'argo_floats': point_stats.get('nearest_argo_id', 'WMO_6903000')
            }
            
            st.session_state['current_report'] = report_data
            st.session_state['report_history'].insert(0, report_data)
            st.success(f"✅ Report **{rep_id}** generated successfully!")

    # Display current report preview if available
    current_rep = st.session_state['current_report']

    if current_rep is None:
        st.info("ℹ️ **No report generated yet.** Configure the report parameters above and click **GENERATE REGIONAL REPORT**.")
    else:
        st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)
        
        st.markdown(
            f"""<div style="background: #FFFFFF; border: 2px solid #0284C7; border-radius: 8px; padding: 24px; color: #0F172A; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #E2E8F0; padding-bottom: 12px; margin-bottom:16px;">
<div>
<h2 style="font-family: 'Outfit', sans-serif; color: #0284C7; margin: 0; font-size:1.45rem;">🌊 INDIAN OCEAN INTELLIGENCE PLATFORM — SCIENTIFIC REPORT</h2>
<div style="font-size: 0.88rem; color: #64748B; margin-top: 4px;">Report Type: <b style="color:#0F172A;">{current_rep['type']}</b> | Generated: <b>{current_rep['timestamp']}</b></div>
</div>
<div style="text-align:right;">
<span style="background:#0284C7; color:#FFFFFF; font-weight:700; font-size:0.75rem; padding:4px 12px; border-radius:4px; text-transform:uppercase;">{current_rep['id']}</span>
<div style="font-size:0.75rem; color:#64748B; margin-top:4px;">CONFIDENTIAL SCIENTIFIC REPORT</div>
</div>
</div>
<div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:12px; margin-bottom:16px;">
<div><span style="font-size:0.72rem; color:#64748B; font-weight:700;">REGION</span><br><strong style="font-size:0.9rem; color:#0F172A;">{current_rep['region']}</strong></div>
<div><span style="font-size:0.72rem; color:#64748B; font-weight:700;">COORDINATES</span><br><strong style="font-size:0.9rem; color:#0F172A;">{current_rep['lat']}° N, {current_rep['lon']}° E</strong></div>
<div><span style="font-size:0.72rem; color:#64748B; font-weight:700;">DEPTH & VARIABLE</span><br><strong style="font-size:0.9rem; color:#0F172A;">{current_rep['depth']} m | {current_rep['variable']}</strong></div>
<div><span style="font-size:0.72rem; color:#64748B; font-weight:700;">DATA SOURCES</span><br><strong style="font-size:0.9rem; color:#0284C7;">{current_rep['sources']}</strong></div>
</div>
<h4 style="color:#0284C7; font-family:'Outfit', sans-serif; margin-top:0; border-bottom:1px solid #E2E8F0; padding-bottom:4px;">1. EXECUTIVE SUMMARY</h4>
<p style="font-size:0.9rem; color:#334155; line-height:1.6; margin-bottom:16px;">
Analysis of the selected <b>{current_rep['region']}</b> region at <b>{current_rep['depth']} m depth</b> indicates an average temperature of <b>{current_rep['mean_temp']}</b> (range: {current_rep['min_temp']} to {current_rep['max_temp']}) with a positive thermal anomaly of <b>{current_rep['anomaly']}</b> during the specified period ({current_rep['start_date']} to {current_rep['end_date']}). Collocated observational data from ARGO profiling floats ({current_rep['argo_floats']}) and Copernicus GLORYS reanalysis confirm <b>{current_rep['coverage']} spatial data coverage</b>.
</p>
<h4 style="color:#0284C7; font-family:'Outfit', sans-serif; border-bottom:1px solid #E2E8F0; padding-bottom:4px;">2. KEY OCEANOGRAPHIC INDICATORS</h4>
<div style="display:grid; grid-template-columns: repeat(8, 1fr); gap:8px; margin-bottom:16px;">
<div style="background:#FFF7ED; border:1px solid #FDBA74; border-radius:6px; padding:8px; text-align:center;">
<div style="font-size:0.62rem; color:#334155; font-weight:700;">MEAN TEMP</div>
<div style="font-size:1.1rem; color:#C2410C; font-weight:700;">{current_rep['mean_temp']}</div>
</div>
<div style="background:#FEF2F2; border:1px solid #FCA5A5; border-radius:6px; padding:8px; text-align:center;">
<div style="font-size:0.62rem; color:#334155; font-weight:700;">MAX TEMP</div>
<div style="font-size:1.1rem; color:#DC2626; font-weight:700;">{current_rep['max_temp']}</div>
</div>
<div style="background:#EFF6FF; border:1px solid #93C5FD; border-radius:6px; padding:8px; text-align:center;">
<div style="font-size:0.62rem; color:#334155; font-weight:700;">MIN TEMP</div>
<div style="font-size:1.1rem; color:#1D4ED8; font-weight:700;">{current_rep['min_temp']}</div>
</div>
<div style="background:#FAF5FF; border:1px solid #D8B4FE; border-radius:6px; padding:8px; text-align:center;">
<div style="font-size:0.62rem; color:#334155; font-weight:700;">MEAN ANOMALY</div>
<div style="font-size:1.1rem; color:#7E22CE; font-weight:700;">{current_rep['anomaly']}</div>
</div>
<div style="background:#ECFEFF; border:1px solid #67E8F9; border-radius:6px; padding:8px; text-align:center;">
<div style="font-size:0.62rem; color:#334155; font-weight:700;">MAX ANOMALY</div>
<div style="font-size:1.1rem; color:#0E7490; font-weight:700;">+2.4 °C</div>
</div>
<div style="background:#F0FDF4; border:1px solid #86EFAC; border-radius:6px; padding:8px; text-align:center;">
<div style="font-size:0.62rem; color:#334155; font-weight:700;">DATA POINTS</div>
<div style="font-size:1.1rem; color:#15803D; font-weight:700;">1,420</div>
</div>
<div style="background:#FEFCE8; border:1px solid #FDE047; border-radius:6px; padding:8px; text-align:center;">
<div style="font-size:0.62rem; color:#334155; font-weight:700;">ARGO FLOATS</div>
<div style="font-size:1.1rem; color:#A16207; font-weight:700;">124</div>
</div>
<div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:6px; padding:8px; text-align:center;">
<div style="font-size:0.62rem; color:#334155; font-weight:700;">GLORYS RECORDS</div>
<div style="font-size:1.1rem; color:#0F172A; font-weight:700;">14.8K</div>
</div>
</div>
</div>""",
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        
        c_rep_map, c_rep_chart = st.columns([2.2, 1.8])
        
        with c_rep_map:
            st.markdown(f'<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">🗺️ SPATIAL CONTEXT MAP — {current_rep["region"].upper()}</div>', unsafe_allow_html=True)
            render_ocean_map(
                dataset="ARGO Observations" if "ARGO" in current_rep['sources'] else "GLORYS Reanalysis",
                variable=current_rep['variable'],
                depth=current_rep['depth'],
                date_str=current_rep['end_date'],
                region=current_rep['region'],
                target_lat=current_rep['lat'],
                target_lon=current_rep['lon'],
                show_floats=True,
                show_heatmap=True
            )

        with c_rep_chart:
            st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">📈 SUBSURFACE TEMPERATURE PROFILE (0–1000M)</div>', unsafe_allow_html=True)
            
            df_comp, comp_stats = get_argo_vs_glorys_profile()
            fig_rep_prof = go.Figure()
            fig_rep_prof.add_trace(go.Scatter(x=df_comp['ARGO (°C)'], y=df_comp['Depth (m)'], mode='lines+markers', name='ARGO In-Situ', line=dict(color='#2563EB', width=2)))
            fig_rep_prof.add_trace(go.Scatter(x=df_comp['GLORYS (°C)'], y=df_comp['Depth (m)'], mode='lines+markers', name='GLORYS Model', line=dict(color='#16A34A', width=2, dash='dash')))
            
            if current_rep['depth'] in df_comp['Depth (m)'].values:
                sel_r = df_comp[df_comp['Depth (m)'] == current_rep['depth']].iloc[0]
                fig_rep_prof.add_trace(go.Scatter(x=[sel_r['ARGO (°C)']], y=[current_rep['depth']], mode='markers', name=f"Target ({current_rep['depth']}m)", marker=dict(size=12, color='#DC2626')))

            fig_rep_prof.update_layout(
                title=dict(text=f"VERTICAL THERMAL STRUCTURE AT ({current_rep['lat']}°N, {current_rep['lon']}°E)", font=dict(family="Outfit", size=11, color="#0F172A")),
                dragmode=False,
                xaxis=dict(title=dict(text="Temperature (°C)", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), gridcolor="#E2E8F0", fixedrange=True),
                yaxis=dict(title=dict(text="Depth (m)", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), gridcolor="#E2E8F0", autorange='reversed', fixedrange=True),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin=dict(l=40, r=20, t=35, b=35),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#0F172A"))
            )
            st.plotly_chart(fig_rep_prof, use_container_width=True, config={'displayModeBar': 'hover', 'displaylogo': False})

        st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

        c_find, c_qual_meth = st.columns([1.8, 1.5])
        
        with c_find:
            st.markdown(
                f"""
                <div class="info-card-box">
                    <div class="info-card-header">🧠 KEY SCIENTIFIC FINDINGS</div>
                    <ul style="font-size:0.83rem; color:#334155; line-height:1.7; padding-left:16px; margin:0;">
                        <li style="margin-bottom:8px;">Subsurface mean temperature at <b>{current_rep['depth']} m depth</b> was recorded at <b>{current_rep['mean_temp']}</b> in the {current_rep['region']}.</li>
                        <li style="margin-bottom:8px;">Thermal anomaly reached a peak of <b>{current_rep['anomaly']}</b> above the 30-year climatological baseline.</li>
                        <li style="margin-bottom:8px;">Validation comparing in-situ ARGO profiles against GLORYS model yield a Mean Absolute Error (MAE) of <b>0.31 °C</b> and R² of <b>0.94</b>.</li>
                        <li style="margin-bottom:8px;">Subsurface thermal gradient is strongest in the thermocline layer between <b>50–150 meters</b>.</li>
                        <li>Observational reliability across the selected region remains high with <b>{current_rep['coverage']} data coverage</b>.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c_qual_meth:
            st.markdown(
                f"""
                <div class="info-card-box">
                    <div class="info-card-header">🛡️ DATA QUALITY & METHODOLOGY</div>
                    <table style="width:100%; font-size:0.82rem; color:#334155; border-collapse:collapse; margin-bottom:8px;">
                        <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Primary Sources:</td><td style="text-align:right; font-weight:700; color:#0284C7;">{current_rep['sources']}</td></tr>
                        <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Spatial Coverage:</td><td style="text-align:right; font-weight:600;">{current_rep['coverage']}</td></tr>
                        <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Depth Levels:</td><td style="text-align:right; font-weight:600;">0 – 1000 m (15 Levels)</td></tr>
                        <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">ARGO Floats:</td><td style="text-align:right; font-weight:600;">{current_rep['argo_floats']}</td></tr>
                        <tr><td style="padding:4px 0; color:#64748B;">Quality Status:</td><td style="text-align:right; font-weight:700; color:#16A34A;">GOOD (98.4%)</td></tr>
                    </table>
                    <p style="font-size:0.75rem; color:#64748B; margin:0; line-height:1.4;">
                        <b>Methodology:</b> In-situ CTD float telemetry matched with Copernicus GLORYS 1/12° reanalysis using spatial nearest-neighbor interpolation.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

        st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:8px;">📥 EXPORT REPORT & UNDERLYING DATASET</div>', unsafe_allow_html=True)
        
        df_report_export = get_avg_temp_by_depth(region=current_rep['region'])
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            pdf_text = f"""INDIAN OCEAN INTELLIGENCE PLATFORM - SCIENTIFIC REPORT
Report ID: {current_rep['id']}
Type: {current_rep['type']}
Region: {current_rep['region']}
Coordinate: {current_rep['lat']}°N, {current_rep['lon']}°E
Depth: {current_rep['depth']} m
Variable: {current_rep['variable']}
Mean Temp: {current_rep['mean_temp']}
Anomaly: {current_rep['anomaly']}
Generated: {current_rep['timestamp']}
Data Sources: {current_rep['sources']}

EXECUTIVE SUMMARY:
Analysis of {current_rep['region']} at {current_rep['depth']}m depth indicates an average temperature of {current_rep['mean_temp']} with anomaly of {current_rep['anomaly']}.
"""
            st.download_button(
                label="📥 Download Report (PDF)",
                data=pdf_text,
                file_name=f"Report_{current_rep['id']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with col_dl2:
            csv_bytes = df_report_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Report Data (CSV)",
                data=csv_bytes,
                file_name=f"Report_Data_{current_rep['id']}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl3:
            st.download_button(
                label="📥 Download Excel Data Package",
                data=csv_bytes,
                file_name=f"Report_Data_{current_rep['id']}.xlsx",
                mime="application/vnd.ms-excel",
                use_container_width=True
            )

# ============================================================
# REPORT HISTORY & RECENT REPORTS
# ============================================================
if len(st.session_state['report_history']) > 0:
    st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:8px;">📋 RECENT GENERATED REPORTS HISTORY (THIS SESSION)</div>', unsafe_allow_html=True)
    
    df_hist_table = pd.DataFrame(st.session_state['report_history'])
    display_cols = ['id', 'type', 'region', 'depth', 'variable', 'timestamp']
    st.dataframe(df_hist_table[display_cols], use_container_width=True, hide_index=True)

render_footer()
