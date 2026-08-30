"""
app.py
======
Main Entrypoint for Indian Ocean Intelligence Platform.
"""

import os
import sys
from pathlib import Path

# Ensure repository root and frontend directory are in sys.path
_repo_root = Path(__file__).resolve().parent.parent
_frontend_dir = Path(__file__).resolve().parent
for _p in [str(_repo_root), str(_frontend_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

st.set_page_config(
    page_title="Indian Ocean Intelligence Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

css_path = _frontend_dir / "styles" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from components.header import render_header
from components.sidebar import render_sidebar
from components.ocean_map import render_ocean_map, render_selected_area_map
from components.area_selection import render_area_selection_ui, render_quick_stats_ui
from components.metric_cards import render_bottom_kpi_bar
from components.temperature_profile import render_temperature_profile_chart
from components.time_series import render_time_series_chart
from components.comparison_chart import render_argo_vs_glorys_chart
from components.data_table import render_avg_temp_depth_table
from components.ai_panel import render_ai_profile_chart
from components.footer import render_footer

from data.mock_data import (
    get_selected_area_stats,
    get_point_details,
    get_avg_temp_by_depth,
    get_temperature_profile,
    get_time_series_data,
    get_argo_vs_glorys_profile,
    get_ai_reconstruction_data
)

# Header
render_header(active_page="Dashboard")

# Sidebar Controls (~20% width)
controls = render_sidebar()

# Fetch Data for searched coordinate
selected_stats = get_point_details(lat=controls['target_lat'], lon=controls['target_lon'], depth=controls['depth'])
df_depth = get_avg_temp_by_depth(region=controls['region'], date_str=str(controls['date']))
df_profile = get_temperature_profile(lat=controls['target_lat'], lon=controls['target_lon'])
df_ts, ts_stats = get_time_series_data(region=controls['region'])
df_comp, comp_stats = get_argo_vs_glorys_profile()
_, _, _, df_ai_prof = get_ai_reconstruction_data()

# 1. Main Map (~60% width) + Right Stats Card (~20% width)
col_map, col_info = st.columns([3.0, 1.0])

with col_map:
    render_ocean_map(
        dataset=controls['dataset'],
        variable=controls['variable'],
        depth=controls['depth'],
        date_str="20 MAY 2024",
        region=controls['region'],
        target_lat=controls['target_lat'],
        target_lon=controls['target_lon'],
        show_floats=True,
        show_heatmap=True
    )

with col_info:
    render_area_selection_ui(selected_stats, current_depth=controls['depth'])
    render_quick_stats_ui(selected_stats, current_depth=controls['depth'])

# 2. Middle 3 Grid Cards
c_m1, c_m2, c_m3 = st.columns(3)

with c_m1:
    render_avg_temp_depth_table(df_depth, selected_depth=controls['depth'])

with c_m2:
    render_temperature_profile_chart(df_profile, selected_depth=controls['depth'])

with c_m3:
    render_selected_area_map(depth=controls['depth'])

# 3. Bottom-Middle 3 Grid Cards
c_b1, c_b2, c_b3 = st.columns(3)

with c_b1:
    render_time_series_chart(df_ts, ts_stats)

with c_b2:
    render_argo_vs_glorys_chart(df_comp, comp_stats)

with c_b3:
    render_ai_profile_chart(df_ai_prof)

# 4. Bottom Key Indicators Bar
render_bottom_kpi_bar()

# 5. Footer
render_footer()
