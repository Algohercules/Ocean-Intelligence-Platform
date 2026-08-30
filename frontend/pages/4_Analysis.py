"""
pages/4_Analysis.py
===================
Regional Comparative Oceanographic Analysis Page.
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

st.set_page_config(
    page_title="Analysis | Indian Ocean Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from components.header import render_header
from components.sidebar import render_sidebar
from components.footer import render_footer
from components.comparison_chart import render_argo_vs_glorys_chart
from components.time_series import render_time_series_chart
from data.mock_data import (
    get_regional_analysis_data,
    get_time_series_data,
    get_argo_vs_glorys_profile
)

render_header(active_page="Analysis")
controls = render_sidebar()

st.markdown(
    """
    <div style="background: rgba(13, 27, 42, 0.7); border: 1px solid #1E3A5F; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px;">
        <h3 style="font-family: 'Outfit', sans-serif; color: #38BDF8; margin: 0; font-size: 1.2rem;">
            📊 REGIONAL OCEANOGRAPHIC COMPARISON & STATISTICAL METRICS
        </h3>
        <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 0.88rem;">
            Compare in-situ ARGO profiling observations against GLORYS hydrodynamic model reanalysis with statistical error metrics (RMSE, MAE, R², Bias).
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

df_comp, comp_stats = get_argo_vs_glorys_profile()
df_ts, ts_stats = get_time_series_data(region=controls['region'])

# Statistical Metrics Summary Cards
st.markdown(
    f"""
    <div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; margin-bottom: 16px;">
        <div class="info-card-box" style="margin-bottom:0; text-align:center;">
            <div style="font-size:0.65rem; color:#64748B; font-weight:700;">RMSE</div>
            <div style="font-size:1.15rem; color:#DC2626; font-weight:700;">{comp_stats['rmse']} °C</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center;">
            <div style="font-size:0.65rem; color:#64748B; font-weight:700;">MAE</div>
            <div style="font-size:1.15rem; color:#EA580C; font-weight:700;">{comp_stats['mae']} °C</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center;">
            <div style="font-size:0.65rem; color:#64748B; font-weight:700;">R² SCORE</div>
            <div style="font-size:1.15rem; color:#16A34A; font-weight:700;">{comp_stats['r2']}</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center;">
            <div style="font-size:0.65rem; color:#64748B; font-weight:700;">BIAS</div>
            <div style="font-size:1.15rem; color:#0284C7; font-weight:700;">{comp_stats['bias']} °C</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center;">
            <div style="font-size:0.65rem; color:#64748B; font-weight:700;">MEAN TEMP</div>
            <div style="font-size:1.15rem; color:#0F172A; font-weight:700;">18.6 °C</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center;">
            <div style="font-size:0.65rem; color:#64748B; font-weight:700;">MIN TEMP</div>
            <div style="font-size:1.15rem; color:#2563EB; font-weight:700;">5.8 °C</div>
        </div>
        <div class="info-card-box" style="margin-bottom:0; text-align:center;">
            <div style="font-size:0.65rem; color:#64748B; font-weight:700;">MAX TEMP</div>
            <div style="font-size:1.15rem; color:#B91C1C; font-weight:700;">{ts_stats['max_temp']}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

c_an1, c_an2 = st.columns(2)

with c_an1:
    render_argo_vs_glorys_chart(df_comp, comp_stats)

with c_an2:
    render_time_series_chart(df_ts, ts_stats)

st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

df_regional = get_regional_analysis_data()
st.markdown('<div style="font-family: \'Outfit\', sans-serif; font-size: 1.05rem; font-weight: 600; color: #0F172A; margin-bottom: 8px;">🌐 REGIONAL SUB-BASIN MATRIX</div>', unsafe_allow_html=True)
st.dataframe(df_regional, use_container_width=True, hide_index=True)

render_footer()
