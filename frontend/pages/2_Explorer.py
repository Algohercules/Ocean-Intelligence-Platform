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
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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

# ============================================================
# 3. REGIONAL ARGO FLOAT SLOW-MOTION DRIFT SIMULATOR
# ============================================================
st.markdown(
    """
    <div style="background: rgba(13, 27, 42, 0.7); border: 1px solid #1E3A5F; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="font-family: 'Outfit', sans-serif; color: #38BDF8; margin: 0; font-size: 1.15rem;">
                    🛰️ REGIONAL ARGO FLOAT SLOW-MOTION DRIFT SIMULATOR
                </h3>
                <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 0.85rem;">
                    Watch autonomous robotic ARGO profiling floats drift across ocean currents over a 30-day hydrodynamic cycle.
                </p>
            </div>
            <span class="badge-cyan" style="border-color: #38BDF8; color: #38BDF8; font-weight:700;">IN-SITU DRIFT ANIMATION</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns([1.5, 1.2, 1.2, 1.4])

with c_ctrl1:
    drift_region = st.selectbox(
        "📍 Select Drift Region",
        ["Arabian Sea", "Bay of Bengal", "Equatorial Indian Ocean", "Southern Indian Ocean", "All Indian Ocean"],
        index=0 if controls['region'] == "Arabian Sea" else (1 if controls['region'] == "Bay of Bengal" else 0),
        key="argo_drift_reg_select"
    )

with c_ctrl2:
    drift_speed_mode = st.selectbox(
        "⏱️ Playback Pace",
        ["Slow Motion (Realistic)", "Normal Drift", "Fast Forward"],
        index=0,
        key="argo_drift_speed_select"
    )

with c_ctrl3:
    show_tails = st.checkbox("Show Drift Trajectory Tails", value=True, key="argo_show_tails")

with c_ctrl4:
    st.markdown(
        f"""
        <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:6px; padding:6px 10px; margin-top:14px; text-align:center;">
            <div style="font-size:0.68rem; color:#1E40AF; font-weight:700;">ACTIVE TELEMETRY</div>
            <div style="font-size:0.85rem; color:#1D4ED8; font-weight:700;">30-DAY HYDRODYNAMIC DRIFT</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Generate region-specific floats & trajectories
def generate_regional_float_trajectories(region="Arabian Sea", n_steps=11):
    np.random.seed(42)
    
    if region == "Arabian Sea":
        n_floats = 24
        lat_min, lat_max = 8.0, 23.0
        lon_min, lon_max = 52.0, 75.0
        # Clockwise gyral drift
        drift_u_base = 0.28
        drift_v_base = 0.22
    elif region == "Bay of Bengal":
        n_floats = 22
        lat_min, lat_max = 8.0, 22.0
        lon_min, lon_max = 80.0, 95.0
        # Anticyclonic eddy drift
        drift_u_base = -0.18
        drift_v_base = -0.25
    elif region == "Equatorial Indian Ocean":
        n_floats = 26
        lat_min, lat_max = -6.0, 6.0
        lon_min, lon_max = 55.0, 95.0
        # Zonal Wyrtki jet drift
        drift_u_base = 0.45
        drift_v_base = 0.05
    elif region == "Southern Indian Ocean":
        n_floats = 28
        lat_min, lat_max = -32.0, -10.0
        lon_min, lon_max = 45.0, 100.0
        # Westward drift
        drift_u_base = -0.35
        drift_v_base = -0.12
    else: # All
        n_floats = 55
        lat_min, lat_max = -30.0, 22.0
        lon_min, lon_max = 42.0, 102.0
        drift_u_base = 0.20
        drift_v_base = 0.10

    init_lats = np.random.uniform(lat_min, lat_max, n_floats)
    init_lons = np.random.uniform(lon_min, lon_max, n_floats)
    float_wmos = [f"WMO #{6903200 + i}" for i in range(n_floats)]
    float_temps = np.round(28.0 - 0.15 * np.abs(init_lats) + np.random.normal(0, 0.3, n_floats), 1)

    time_steps = [f"Day {i*3}" for i in range(n_steps)]
    
    # Generate continuous smooth trajectories across time steps
    traj_lats = np.zeros((n_floats, n_steps))
    traj_lons = np.zeros((n_floats, n_steps))

    for i in range(n_floats):
        traj_lats[i, 0] = init_lats[i]
        traj_lons[i, 0] = init_lons[i]
        
        # Individual current vortex perturbations
        vortex_speed = np.random.uniform(0.6, 1.4)
        angle_offset = np.random.uniform(0, 2*np.pi)
        
        for t in range(1, n_steps):
            # Slow realistic drift (~0.1 to 0.3 deg per 3-day step)
            dlat = (drift_v_base + 0.12 * np.sin(t * 0.5 + angle_offset)) * vortex_speed * 0.4
            dlon = (drift_u_base + 0.15 * np.cos(t * 0.5 + angle_offset)) * vortex_speed * 0.45
            
            traj_lats[i, t] = traj_lats[i, t-1] + dlat
            traj_lons[i, t] = traj_lons[i, t-1] + dlon

    return time_steps, float_wmos, float_temps, traj_lats, traj_lons, (lat_min, lat_max, lon_min, lon_max)

t_steps, wmo_ids, temps, t_lats, t_lons, bounds = generate_regional_float_trajectories(region=drift_region)

# Build Animated Plotly Map Figure
fig_drift = go.Figure()

# 1. Base Trajectory Tails Trace (at step 0)
for i in range(len(wmo_ids)):
    fig_drift.add_trace(go.Scatter(
        x=[t_lons[i, 0]],
        y=[t_lats[i, 0]],
        mode='lines',
        line=dict(color='rgba(56, 189, 248, 0.4)', width=1.5, dash='dot'),
        hoverinfo='skip',
        showlegend=False,
        name=f"Path {wmo_ids[i]}"
    ))

# 2. Base Active Float Head Markers (at step 0)
fig_drift.add_trace(go.Scatter(
    x=t_lons[:, 0],
    y=t_lats[:, 0],
    mode='markers+text',
    marker=dict(
        size=11,
        color='#38BDF8',
        symbol='circle',
        line=dict(width=2, color='#FFFFFF')
    ),
    text=[f"  {wmo_ids[i].split()[-1]}" for i in range(len(wmo_ids))],
    textposition="top right",
    textfont=dict(color='#0284C7', size=9, family='Outfit', weight='bold'),
    hovertext=[f"<b>{wmo_ids[i]}</b><br>Lat: {t_lats[i, 0]:.2f}°N<br>Lon: {t_lons[i, 0]:.2f}°E<br>Temp: {temps[i]}°C<br>Status: Drifting at 1000m depth" for i in range(len(wmo_ids))],
    hoverinfo='text',
    name='Active ARGO Floats'
))

# Create animation frames
anim_frames = []
frame_duration = 1000 if "Slow" in drift_speed_mode else (600 if "Normal" in drift_speed_mode else 300)

for step_idx, step_name in enumerate(t_steps):
    f_data = []
    
    # 1. Trajectory lines up to step_idx
    for i in range(len(wmo_ids)):
        f_data.append(go.Scatter(
            x=t_lons[i, :step_idx+1] if show_tails else [t_lons[i, step_idx]],
            y=t_lats[i, :step_idx+1] if show_tails else [t_lats[i, step_idx]],
            mode='lines',
            line=dict(color='rgba(56, 189, 248, 0.55)', width=2, dash='dot'),
            hoverinfo='skip',
            showlegend=False
        ))
        
    # 2. Moving Float Heads at step_idx
    f_data.append(go.Scatter(
        x=t_lons[:, step_idx],
        y=t_lats[:, step_idx],
        mode='markers+text',
        marker=dict(
            size=12,
            color='#0284C7',
            symbol='circle',
            line=dict(width=2.5, color='#FFFFFF')
        ),
        text=[f"  {wmo_ids[i].split()[-1]}" for i in range(len(wmo_ids))],
        textposition="top right",
        textfont=dict(color='#0369A1', size=9, family='Outfit', weight='bold'),
        hovertext=[f"<b>{wmo_ids[i]}</b> ({step_name})<br>Lat: {t_lats[i, step_idx]:.2f}°N<br>Lon: {t_lons[i, step_idx]:.2f}°E<br>Temp: {temps[i]}°C<br>Drift Speed: 0.18 m/s" for i in range(len(wmo_ids))],
        hoverinfo='text',
        name='Active ARGO Floats'
    ))
    
    anim_frames.append(go.Frame(data=f_data, name=step_name))

fig_drift.frames = anim_frames

# Animation Play/Pause Buttons & Slider Layout
fig_drift.update_layout(
    title=dict(
        text=f"ARGO ROBOTIC DRIFT TRAJECTORY SIMULATION ({drift_region.upper()} — 30-DAY HYDRODYNAMIC EVOLUTION)",
        font=dict(family="Outfit", size=12, color="#0F172A")
    ),
    xaxis=dict(
        title=dict(text="Longitude (°E)", font=dict(color="#0F172A", size=11)),
        tickfont=dict(color="#0F172A", size=10),
        gridcolor="#E2E8F0",
        range=[max(34.0, bounds[2] - 3.0), min(108.0, bounds[3] + 4.0)],
        fixedrange=True
    ),
    yaxis=dict(
        title=dict(text="Latitude (°N)", font=dict(color="#0F172A", size=11)),
        tickfont=dict(color="#0F172A", size=10),
        gridcolor="#E2E8F0",
        range=[max(-36.0, bounds[0] - 2.5), min(28.0, bounds[1] + 2.5)],
        fixedrange=True
    ),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8FAFC",
    margin=dict(l=40, r=30, t=50, b=50),
    height=420,
    showlegend=False,
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            x=0.0,
            y=1.16,
            showactive=True,
            buttons=[
                dict(
                    label="▶ PLAY DRIFT",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=frame_duration, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=frame_duration // 2, easing="linear")
                        )
                    ]
                ),
                dict(
                    label="⏸ PAUSE",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0)
                        )
                    ]
                )
            ],
            bgcolor="#EFF6FF",
            bordercolor="#93C5FD",
            font=dict(size=10, color="#1D4ED8", weight='bold')
        )
    ],
    sliders=[
        dict(
            active=0,
            yanchor="top",
            y=-0.12,
            xanchor="left",
            x=0.0,
            currentvalue=dict(
                font=dict(size=11, color="#0F172A", family="Outfit"),
                prefix="Current Trajectory Timeline: ",
                visible=True,
                xanchor="right"
            ),
            transition=dict(duration=300, easing="cubic-in-out"),
            pad=dict(b=10, t=20),
            len=0.98,
            steps=[
                dict(
                    args=[
                        [step],
                        dict(
                            frame=dict(duration=frame_duration, redraw=True),
                            mode="immediate",
                            transition=dict(duration=200)
                        )
                    ],
                    label=step,
                    method="animate"
                )
                for step in t_steps
            ]
        )
    ]
)

st.plotly_chart(fig_drift, use_container_width=True, config={'displayModeBar': 'hover', 'displaylogo': False})

# Bottom Telemetry Stats Bar
st.markdown(
    f"""
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; margin-top:8px; margin-bottom:16px;">
        <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:6px; padding:8px 12px; text-align:center;">
            <div style="font-size:0.68rem; color:#64748B; font-weight:700;">TRACKED IN-SITU FLOATS</div>
            <div style="font-size:1.1rem; color:#1D4ED8; font-weight:700;">{len(wmo_ids)} Active Units</div>
        </div>
        <div style="background:#FAF5FF; border:1px solid #E9D5FF; border-radius:6px; padding:8px 12px; text-align:center;">
            <div style="font-size:0.68rem; color:#64748B; font-weight:700;">MEAN DRIFT VELOCITY</div>
            <div style="font-size:1.1rem; color:#7E22CE; font-weight:700;">0.18 m/s (Zonal)</div>
        </div>
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:6px; padding:8px 12px; text-align:center;">
            <div style="font-size:0.68rem; color:#64748B; font-weight:700;">PARKING DEPTH LEVEL</div>
            <div style="font-size:1.1rem; color:#15803D; font-weight:700;">1000 m Drift Layer</div>
        </div>
        <div style="background:#FFF7ED; border:1px solid #FED7AA; border-radius:6px; padding:8px 12px; text-align:center;">
            <div style="font-size:0.68rem; color:#64748B; font-weight:700;">TELEMETRY UPTIME</div>
            <div style="font-size:1.1rem; color:#C2410C; font-weight:700;">99.2 % Reliable</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr style='border-color: #CBD5E1; margin: 20px 0;'>", unsafe_allow_html=True)

# ============================================================
# 4. 2D VERTICAL OCEAN TRANSECT CONTOUR SLICER
# ============================================================
st.markdown(
    """
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
