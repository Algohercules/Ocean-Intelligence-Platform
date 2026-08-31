"""
pages/2_Explorer.py
===================
Interactive Ocean Data Explorer Page.
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
import plotly.graph_objects as go

st.set_page_config(
    page_title="Explorer | Pirates Of Ocean",
    page_icon="🔍",
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
from data.mock_data import get_transect_data, get_temperature_profile, get_point_details
from components.temperature_profile import render_temperature_profile_chart

render_header(active_page="Explorer")
controls = render_sidebar()

st.markdown(
    """
    <div style="background: rgba(13, 27, 42, 0.7); border: 1px solid #1E3A5F; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px;">
        <h3 style="font-family: 'Outfit', sans-serif; color: #38BDF8; margin: 0; font-size: 1.2rem;">
            🗺️ FULLSCREEN OCEAN DATA EXPLORER & TRANSECT SLICER
        </h3>
        <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 0.88rem;">
            Interactively explore subsurface temperature structures, ARGO float trajectories, and 2D vertical depth cross-sections across the Indian Ocean basin.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

selected_stats = get_point_details(lat=controls['target_lat'], lon=controls['target_lon'], depth=controls['depth'])

col_exp_map, col_exp_tools = st.columns([2.5, 1.1])

with col_exp_map:
    st.markdown(f'<div style="font-weight: 600; color: #0F172A; margin-bottom: 6px;">🗺️ SPATIAL LAYER VIEWER ({controls["region"].upper()})</div>', unsafe_allow_html=True)
    render_ocean_map(
        dataset=controls['dataset'],
        variable=controls['variable'],
        depth=controls['depth'],
        date_str=str(controls['date']),
        region=controls['region'],
        target_lat=controls['target_lat'],
        target_lon=controls['target_lon'],
        show_floats=True,
        show_heatmap=True
    )

with col_exp_tools:
    render_area_selection_ui(selected_stats, current_depth=controls['depth'])
    render_quick_stats_ui(selected_stats, current_depth=controls['depth'])

st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

# Transect Slicer Section
st.markdown(
    f"""
    <div style="font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;">
        🌊 2D VERTICAL OCEAN TRANSECT CONTOUR SLICER
    </div>
    """,
    unsafe_allow_html=True
)

col_ts1, col_ts2 = st.columns([1.0, 2.5])

with col_ts1:
    slice_type = st.radio("Slice Orientation", ["Zonal (Latitude Transect)", "Meridional (Longitude Transect)"], key="exp_slice_type")
    
    if "Latitude" in slice_type:
        target_val = st.slider("Target Latitude (°N)", -30.0, 25.0, float(controls['target_lat']), 0.5, key="exp_lat_slider")
        st.info(f"Extracting zonal transect along **{target_val}° N**")
    else:
        target_val = st.slider("Target Longitude (°E)", 35.0, 105.0, float(controls['target_lon']), 0.5, key="exp_lon_slider")
        st.info(f"Extracting meridional transect along **{target_val}° E**")

coords, depths, temp_grid, coord_name = get_transect_data(slice_type=slice_type, target_val=target_val)

with col_ts2:
    fig_contour = go.Figure(data=go.Contour(
        z=temp_grid,
        x=coords,
        y=depths,
        colorscale='Turbid',
        contours=dict(
            coloring='heatmap',
            showlabels=True,
            labelfont=dict(size=10, color='white')
        ),
        colorbar=dict(title=dict(text='Temp (°C)', font=dict(color='#0F172A')), tickfont=dict(color='#0F172A'))
    ))
    fig_contour.update_layout(
        title=dict(text=f"2D VERTICAL DEPTH PROFILE ({coord_name})", font=dict(family="Outfit", size=12, color="#0F172A")),
        xaxis=dict(title=dict(text="Distance / Coordinate", font=dict(color="#0F172A", size=11)), tickfont=dict(color="#0F172A", size=11), fixedrange=True),
        yaxis=dict(title=dict(text="Depth (m)", font=dict(color="#0F172A", size=11)), tickfont=dict(color="#0F172A", size=11), autorange='reversed', fixedrange=True),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=320,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_contour, use_container_width=True, config={'displayModeBar': 'hover', 'displaylogo': False, 'scrollZoom': False})

render_footer()
