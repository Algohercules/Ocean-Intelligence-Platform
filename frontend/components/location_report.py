"""
components/location_report.py
==============================
Point-Based Location Ocean Intelligence Report Component.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, date

from components.ocean_map import render_ocean_map
from data.mock_data import (
    get_point_details,
    get_temperature_profile,
    get_argo_vs_glorys_profile,
    get_mhw_timeseries_data,
    DEPTH_LEVELS
)

def render_location_report_ui(target_lat=15.0, target_lon=65.0, target_depth=75, selected_date=date(2024, 5, 20), **kwargs):
    report_type = st.session_state.get('loc_type', 'Location Intelligence Report')
    # 2. REPORT HEADER & ID GENERATION
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_id = f"IOI-LR-2024-{np.random.randint(100,999)}"
    
    # Land Check Safety Rule
    if target_lat > 23.5 and target_lon < 68.0:
        st.warning("⚠️ **Selected coordinate appears to be outside the ocean domain (Land Mass).** Please select an ocean coordinate.")
        return
        
    point_stats = get_point_details(lat=target_lat, lon=target_lon, depth=target_depth)
    
    # 3. REGION IDENTIFICATION
    region_name = point_stats['region']
    
    st.markdown(
        f"""<div style="background: #FFFFFF; border: 2px solid #0284C7; border-radius: 8px; padding: 24px; color: #0F172A; box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin-bottom: 20px;">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #E2E8F0; padding-bottom: 12px; margin-bottom:16px;">
<div>
<h2 style="font-family: 'Outfit', sans-serif; color: #0284C7; margin: 0; font-size:1.45rem;">📍 {report_type.upper()}</h2>
<div style="font-size: 0.88rem; color: #64748B; margin-top: 4px;">Region: <b style="color:#0F172A;">{region_name}</b> | Target Coordinate: <b>{target_lat:.2f}° N, {target_lon:.2f}° E</b></div>
</div>
<div style="text-align:right;">
<span style="background:#0284C7; color:#FFFFFF; font-weight:700; font-size:0.75rem; padding:4px 12px; border-radius:4px; text-transform:uppercase;">{report_id}</span>
<div style="font-size:0.75rem; color:#64748B; margin-top:4px;">POINT-BASED ANALYSIS</div>
</div>
</div>
<div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:12px; margin-bottom:16px;">
<div><span style="font-size:0.72rem; color:#64748B; font-weight:700;">REQUESTED LAT/LON</span><br><strong style="font-size:0.9rem; color:#0F172A;">{target_lat:.2f}° N, {target_lon:.2f}° E</strong></div>
<div><span style="font-size:0.72rem; color:#64748B; font-weight:700;">REQUESTED DEPTH</span><br><strong style="font-size:0.9rem; color:#0F172A;">{target_depth} meters</strong></div>
<div><span style="font-size:0.72rem; color:#64748B; font-weight:700;">ANALYSIS DATE</span><br><strong style="font-size:0.9rem; color:#0F172A;">{selected_date}</strong></div>
<div><span style="font-size:0.72rem; color:#64748B; font-weight:700;">NEAREST ARGO FLOAT</span><br><strong style="font-size:0.9rem; color:#0284C7;">{point_stats.get('nearest_argo_id', 'WMO_6903000')} ({point_stats.get('nearest_argo_dist', '12 km')})</strong></div>
</div>
<h4 style="color:#0284C7; font-family:'Outfit', sans-serif; margin-top:0; border-bottom:1px solid #E2E8F0; padding-bottom:4px;">1. OCEAN CONDITIONS AT SELECTED DEPTH ({target_depth}M)</h4>
<div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:8px; margin-bottom:16px;">
<div style="background:#FFF7ED; border:1px solid #FDBA74; border-radius:6px; padding:8px; text-align:center;">
<div style="font-size:0.62rem; color:#334155; font-weight:700;">TEMPERATURE</div>
<div style="font-size:1.15rem; color:#C2410C; font-weight:700;">{point_stats['avg_temp']} °C</div>
</div>
<div style="background:#FEF2F2; border:1px solid #FCA5A5; border-radius:6px; padding:8px; text-align:center;">
<div style="font-size:0.62rem; color:#334155; font-weight:700;">ANOMALY</div>
<div style="font-size:1.15rem; color:#DC2626; font-weight:700;">+{point_stats['anomaly']} °C</div>
</div>
<div style="background:#EFF6FF; border:1px solid #93C5FD; border-radius:6px; padding:8px; text-align:center;">
<div style="font-size:0.62rem; color:#334155; font-weight:700;">SALINITY</div>
<div style="font-size:1.15rem; color:#1D4ED8; font-weight:700;">35.4 PSU</div>
</div>
<div style="background:#F0FDF4; border:1px solid #86EFAC; border-radius:6px; padding:8px; text-align:center;">
<div style="font-size:0.62rem; color:#334155; font-weight:700;">CURRENT SPEED</div>
<div style="font-size:1.15rem; color:#15803D; font-weight:700;">0.24 m/s</div>
</div>
<div style="background:#FAF5FF; border:1px solid #D8B4FE; border-radius:6px; padding:8px; text-align:center;">
<div style="font-size:0.62rem; color:#334155; font-weight:700;">SEA LEVEL ANOMALY</div>
<div style="font-size:1.15rem; color:#7E22CE; font-weight:700;">+0.05 m</div>
</div>
</div>
</div>""",
        unsafe_allow_html=True
    )

    # 5 & 6. TEMPERATURE AT LOCATION & VERTICAL PROFILE
    c_loc_prof, c_loc_depth = st.columns([2.0, 1.3])
    
    with c_loc_prof:
        st.markdown(f'<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">📈 VERTICAL TEMPERATURE PROFILE (0–1000M AT {target_lat:.2f}°N, {target_lon:.2f}°E)</div>', unsafe_allow_html=True)
        
        df_prof = get_temperature_profile(lat=target_lat, lon=target_lon)
        
        fig_loc_prof = go.Figure()
        fig_loc_prof.add_trace(go.Scatter(x=df_prof['Temperature (°C)'], y=df_prof['Depth (m)'], mode='lines+markers', name='Observed (ARGO/GLORYS)', line=dict(color='#2563EB', width=2.5)))
        
        # Highlight target depth point
        if target_depth in df_prof['Depth (m)'].values:
            sel_r = df_prof[df_prof['Depth (m)'] == target_depth].iloc[0]
            fig_loc_prof.add_trace(go.Scatter(x=[sel_r['Temperature (°C)']], y=[target_depth], mode='markers', name=f"Target Depth ({target_depth}m)", marker=dict(size=12, color='#DC2626', symbol='circle')))

        fig_loc_prof.update_layout(
            title=dict(text=f"VERTICAL THERMAL PROFILE ({target_lat:.2f}°N, {target_lon:.2f}°E)", font=dict(family="Outfit", size=11, color="#0F172A")),
            dragmode=False,
            xaxis=dict(title=dict(text="Temperature (°C)", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), gridcolor="#E2E8F0", fixedrange=True),
            yaxis=dict(title=dict(text="Depth (m)", font=dict(color="#0F172A", size=10)), tickfont=dict(color="#0F172A", size=10), gridcolor="#E2E8F0", autorange='reversed', fixedrange=True),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=40, r=20, t=35, b=35),
            height=320,
            showlegend=False
        )
        st.plotly_chart(fig_loc_prof, use_container_width=True, config={'displayModeBar': 'hover', 'displaylogo': False})

    with c_loc_depth:
        # 7. SELECTED DEPTH ANALYSIS CARD
        surf_temp = point_stats['surface_temp']
        curr_temp = point_stats['avg_temp']
        diff_surf = np.round(curr_temp - surf_temp, 1)
        
        st.markdown(
            f"""
            <div class="info-card-box">
                <div class="info-card-header">🎯 SELECTED DEPTH ANALYSIS ({target_depth}M)</div>
                <table style="width:100%; font-size:0.83rem; color:#334155; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Selected Depth:</td><td style="text-align:right; font-weight:700; color:#0F172A;">{target_depth} m</td></tr>
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Temperature:</td><td style="text-align:right; font-weight:700; color:#C2410C;">{curr_temp} °C</td></tr>
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Diff from Surface (0m):</td><td style="text-align:right; font-weight:700; color:#DC2626;">{diff_surf} °C</td></tr>
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Diff from Regional Avg:</td><td style="text-align:right; font-weight:600; color:#15803D;">+0.4 °C</td></tr>
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Salinity:</td><td style="text-align:right; font-weight:600;">35.4 PSU</td></tr>
                    <tr><td style="padding:4px 0; color:#64748B;">Current Speed:</td><td style="text-align:right; font-weight:600;">0.24 m/s</td></tr>
                </table>
            </div>
            
            <!-- 9. HEATWAVE STATUS -->
            <div class="info-card-box" style="margin-top:10px;">
                <div class="info-card-header">🔥 MARINE HEATWAVE INDICATOR</div>
                <div style="display:flex; justify-space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-size:0.83rem; color:#64748B;">Status:</span>
                    <span style="background:#FEF08A; color:#854D0E; font-weight:700; font-size:0.75rem; padding:3px 10px; border-radius:4px;">WATCH (+0.6 °C)</span>
                </div>
                <div style="font-size:0.8rem; color:#334155;">
                    Duration: <b>8 Days</b> &nbsp;|&nbsp; Peak Anomaly: <b>+1.8 °C</b><br>
                    Intensity: <b>Moderate</b> | Max Depth: <b>100 m</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

    # 10, 11, 12, 13, 14. TELEMETRY, ARGO, GLORYS & AI PREDICTION
    c_loc_argo, c_loc_glorys, c_loc_ai = st.columns(3)
    
    with c_loc_argo:
        st.markdown(
            f"""
            <div class="info-card-box">
                <div class="info-card-header">📡 NEAREST ARGO OBSERVATION</div>
                <table style="width:100%; font-size:0.82rem; color:#334155; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Float WMO ID:</td><td style="text-align:right; font-weight:700; color:#0284C7;">{point_stats.get('nearest_argo_id', 'WMO_6903000')}</td></tr>
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Distance:</td><td style="text-align:right; font-weight:600;">{point_stats.get('nearest_argo_dist', '12 km')}</td></tr>
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Observation Date:</td><td style="text-align:right; font-weight:600;">2024-05-20</td></tr>
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Observed Temp:</td><td style="text-align:right; font-weight:700; color:#DC2626;">{curr_temp} °C</td></tr>
                    <tr><td style="padding:4px 0; color:#64748B;">Salinity:</td><td style="text-align:right; font-weight:600;">35.4 PSU</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c_loc_glorys:
        st.markdown(
            f"""
            <div class="info-card-box">
                <div class="info-card-header">🌊 GLORYS REANALYSIS GRID</div>
                <table style="width:100%; font-size:0.82rem; color:#334155; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Grid Snap Lat:</td><td style="text-align:right; font-weight:600;">{target_lat:.3f}° N</td></tr>
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Grid Snap Lon:</td><td style="text-align:right; font-weight:600;">{target_lon:.3f}° E</td></tr>
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Grid Model Temp:</td><td style="text-align:right; font-weight:700; color:#16A34A;">{np.round(curr_temp - 0.12, 1)} °C</td></tr>
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">ARGO vs GLORYS Diff:</td><td style="text-align:right; font-weight:600; color:#0284C7;">+0.12 °C</td></tr>
                    <tr><td style="padding:4px 0; color:#64748B;">Model Resolution:</td><td style="text-align:right; font-weight:600;">1/12° (~9 km)</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c_loc_ai:
        st.markdown(
            f"""
            <div class="info-card-box">
                <div class="info-card-header">🤖 AI PREDICTION AT LOCATION</div>
                <table style="width:100%; font-size:0.82rem; color:#334155; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Current Temp:</td><td style="text-align:right; font-weight:600;">{curr_temp} °C</td></tr>
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">7-Day AI Forecast:</td><td style="text-align:right; font-weight:700; color:#9333EA;">{np.round(curr_temp + 0.7, 1)} °C</td></tr>
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Predicted Change:</td><td style="text-align:right; font-weight:700; color:#DC2626;">+0.7 °C</td></tr>
                    <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:4px 0; color:#64748B;">Forecast Horizon:</td><td style="text-align:right; font-weight:600;">7 Days</td></tr>
                    <tr><td style="padding:4px 0; color:#64748B;">Model Confidence:</td><td style="text-align:right; font-weight:700; color:#15803D;">94 %</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

    # 15. LOCATION MAP & 17. SCIENTIFIC INSIGHTS
    c_loc_map, c_loc_ins = st.columns([2.3, 1.2])
    
    with c_loc_map:
        st.markdown(f'<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">🗺️ TARGET LOCATION SPATIAL MAP ({target_lat:.2f}°N, {target_lon:.2f}°E)</div>', unsafe_allow_html=True)
        render_ocean_map(
            dataset="ARGO Observations",
            variable="Temperature",
            depth=target_depth,
            date_str=str(selected_date),
            region=region_name,
            target_lat=target_lat,
            target_lon=target_lon,
            show_floats=True,
            show_heatmap=True,
            map_key=f"loc_map_{int(target_depth)}_{abs(hash(str(target_lat)+str(target_lon)))%10000}"
        )

    with c_loc_ins:
        st.markdown(
            f"""
            <div class="info-card-box">
                <div class="info-card-header">🧠 LOCATION SCIENTIFIC INSIGHTS</div>
                <ul style="font-size:0.83rem; color:#334155; line-height:1.7; padding-left:16px; margin:0;">
                    <li style="margin-bottom:8px;">Subsurface temperature at <b>{target_depth}m depth</b> is recorded at <b>{curr_temp} °C</b> ({point_stats['anomaly']} °C above climatology).</li>
                    <li style="margin-bottom:8px;">The nearest active ARGO float (<b>{point_stats.get('nearest_argo_id', 'WMO_6903000')}</b>) is located approximately <b>{point_stats.get('nearest_argo_dist', '12 km')}</b> from target coordinates.</li>
                    <li style="margin-bottom:8px;">GLORYS hydrodynamic model reanalysis matches in-situ observations with a minimal residual bias of <b>+0.08 °C</b>.</li>
                    <li style="margin-bottom:8px;">AI prediction forecasts a warming trend of <b>+0.7 °C</b> at {target_depth}m over the next 7 days.</li>
                    <li>Observational data quality at coordinate {target_lat:.2f}°N, {target_lon:.2f}°E is rated <b>GOOD (94% coverage)</b>.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

    # 18. LOCATION DATA SUMMARY TABLE
    st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:8px;">📋 LOCATION PARAMETER SUMMARY TABLE</div>', unsafe_allow_html=True)
    
    loc_summary_df = pd.DataFrame([
        {'Parameter': 'Latitude', 'Value': f"{target_lat:.2f}° N", 'Unit': '°N', 'Depth': f"{target_depth} m", 'Date': str(selected_date), 'Source': 'User Input'},
        {'Parameter': 'Longitude', 'Value': f"{target_lon:.2f}° E", 'Unit': '°E', 'Depth': f"{target_depth} m", 'Date': str(selected_date), 'Source': 'User Input'},
        {'Parameter': 'Region', 'Value': region_name, 'Unit': 'Sub-Basin', 'Depth': 'Surface to Bathymetry', 'Date': str(selected_date), 'Source': 'Spatial Boundary'},
        {'Parameter': 'Temperature', 'Value': f"{curr_temp} °C", 'Unit': '°C', 'Depth': f"{target_depth} m", 'Date': str(selected_date), 'Source': 'Copernicus / ARGO'},
        {'Parameter': 'Temperature Anomaly', 'Value': f"+{point_stats['anomaly']} °C", 'Unit': '°C', 'Depth': f"{target_depth} m", 'Date': str(selected_date), 'Source': 'Climatology Baseline'},
        {'Parameter': 'Salinity', 'Value': '35.4 PSU', 'Unit': 'PSU', 'Depth': f"{target_depth} m", 'Date': str(selected_date), 'Source': 'GLORYS Model'},
        {'Parameter': 'Current Speed', 'Value': '0.24 m/s', 'Unit': 'm/s', 'Depth': f"{target_depth} m", 'Date': str(selected_date), 'Source': 'GLORYS Hydrodynamic'},
        {'Parameter': 'Nearest ARGO Distance', 'Value': point_stats.get('nearest_argo_dist', '12 km'), 'Unit': 'km', 'Depth': 'Water Column', 'Date': str(selected_date), 'Source': 'ARGO Float Array'},
        {'Parameter': 'AI 7-Day Prediction', 'Value': f"{np.round(curr_temp + 0.7, 1)} °C", 'Unit': '°C', 'Depth': f"{target_depth} m", 'Date': '2024-05-27', 'Source': 'Ocean-Net-v4'}
    ])
    
    st.dataframe(loc_summary_df, use_container_width=True, hide_index=True)
    
    # 21. DOWNLOAD BUTTONS
    c_loc_dl1, c_loc_dl2 = st.columns(2)
    with c_loc_dl1:
        loc_csv = loc_summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Location Report (CSV)",
            data=loc_csv,
            file_name=f"Location_Report_{target_lat}N_{target_lon}E_{target_depth}m.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c_loc_dl2:
        st.download_button(
            label="📥 Download Excel Data Package",
            data=loc_csv,
            file_name=f"Location_Report_{target_lat}N_{target_lon}E_{target_depth}m.xlsx",
            mime="application/vnd.ms-excel",
            use_container_width=True
        )
