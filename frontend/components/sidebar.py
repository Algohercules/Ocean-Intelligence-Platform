"""
components/sidebar.py
=====================
Sidebar Data Controls component with native Streamlit Python toggle state handling.
"""

import streamlit as st
from datetime import date

DEPTH_LEVELS_LABELS = [
    "1.  0 m", "2.  10 m", "3.  20 m", "4.  30 m", "5.  50 m",
    "6.  75 m", "7.  100 m", "8.  150 m", "9.  200 m", "10. 300 m",
    "11. 400 m", "12. 500 m", "13. 700 m", "14. 850 m", "15. 1000 m"
]

DEPTH_VALUES = [
    0, 10, 20, 30, 50, 75, 100, 150, 200, 300,
    400, 500, 700, 850, 1000
]

LAT_LON_PRESETS = {
    "Custom Coordinate": (15.0, 65.0),
    "Arabian Sea Center (15.0°N, 65.0°E)": (15.0, 65.0),
    "Bay of Bengal Center (15.0°N, 88.0°E)": (15.0, 88.0),
    "Equator / Maldives (0.0°N, 73.2°E)": (0.0, 73.2),
    "Mumbai Offshore (18.9°N, 72.8°E)": (18.9, 72.8),
    "Southern Ocean (-15.0°S, 75.0°E)": (-15.0, 75.0)
}


def render_sidebar():

    # ============================================================
    # SESSION STATE DEFAULTS
    # ============================================================

    if 'sidebar_open' not in st.session_state:
        st.session_state['sidebar_open'] = True

    if 'selected_dataset' not in st.session_state:
        st.session_state['selected_dataset'] = "ARGO vs GLORYS"

    if 'selected_variable' not in st.session_state:
        st.session_state['selected_variable'] = "Temperature (°C)"

    if 'selected_depth_idx' not in st.session_state:
        st.session_state['selected_depth_idx'] = 5

    if 'selected_date' not in st.session_state:
        st.session_state['selected_date'] = date(2024, 5, 20)

    if 'selected_region' not in st.session_state:
        st.session_state['selected_region'] = "All Indian Ocean"

    if 'target_lat' not in st.session_state:
        st.session_state['target_lat'] = 15.0

    if 'target_lon' not in st.session_state:
        st.session_state['target_lon'] = 65.0

    current_depth_val = DEPTH_VALUES[
        st.session_state['selected_depth_idx']
    ]

    # ============================================================
    # SIDEBAR STYLING
    # ============================================================

    if not st.session_state['sidebar_open']:

        st.markdown(
            """
            <style>

            /* Completely hide sidebar when closed */
            [data-testid="stSidebar"],
            section[data-testid="stSidebar"] {
                display: none !important;
                width: 0 !important;
                min-width: 0 !important;
                max-width: 0 !important;
            }

            /* Give the main content the full width */
            [data-testid="stAppViewContainer"] .main .block-container,
            .block-container {
                margin-left: 0 !important;
                width: 100% !important;
                max-width: 100% !important;
            }

            </style>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <style>

            /* ====================================================
               SIDEBAR
               ==================================================== */

            [data-testid="stSidebar"],
            section[data-testid="stSidebar"] {
                display: block !important;

                width: 285px !important;
                min-width: 285px !important;
                max-width: 290px !important;

                background-color: #0B192C !important;

                border-right: 1px solid #1E293B !important;

                margin-top: 56px !important;

                padding-top: 0 !important;

                height: calc(100vh - 56px) !important;

                box-shadow: none !important;
            }


            /* ====================================================
               REMOVE ALL TOP GAP INSIDE SIDEBAR
               ==================================================== */

            [data-testid="stSidebarUserContent"],
            [data-testid="stSidebarContent"],
            section[data-testid="stSidebar"] > div,
            section[data-testid="stSidebar"] > div > div {

                padding-top: 0 !important;
                margin-top: 0 !important;
            }


            /* First content wrapper */
            [data-testid="stSidebarUserContent"] > div:first-child {

                padding-top: 0 !important;
                margin-top: 0 !important;
            }


            /* Streamlit vertical blocks */
            [data-testid="stSidebarUserContent"] .block-container {

                padding-top: 0 !important;
                margin-top: 0 !important;
            }


            /* ====================================================
               FIRST DATA CONTROLS HEADING
               ==================================================== */

            [data-testid="stSidebar"] h3:first-of-type {

                margin-top: 0 !important;
                padding-top: 0 !important;
            }


            /* ====================================================
               SIDEBAR TEXT COLORS
               ==================================================== */

            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4 {

                color: #FFFFFF !important;
            }


            /* Descriptions / captions */
            [data-testid="stSidebar"] small,
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {

                color: #94A3B8 !important;
            }


            /* ====================================================
               SELECTBOX / INPUT DARK THEME
               ==================================================== */

            [data-testid="stSidebar"] [data-baseweb="select"] > div {

                background-color: #111827 !important;
                border-color: #334155 !important;
                color: #FFFFFF !important;
            }


            [data-testid="stSidebar"] input {

                background-color: #111827 !important;
                color: #FFFFFF !important;
                border-color: #334155 !important;
            }


            /* ====================================================
               RESET BUTTON
               ==================================================== */

            [data-testid="stSidebar"] button {

                border-color: #334155 !important;
            }

            </style>
            """,
            unsafe_allow_html=True
        )

    # ============================================================
    # DATA CONTROLS
    # ============================================================

    st.sidebar.markdown("### 1. DATA CONTROLS")

    st.sidebar.caption(
        "Select dataset, variable, depth, date and more."
    )

    # ============================================================
    # DATASET
    # ============================================================

    dataset_opts = [
        "ARGO Observations",
        "GLORYS Reanalysis",
        "AI Reconstruction",
        "ARGO vs GLORYS"
    ]

    st.session_state['selected_dataset'] = st.sidebar.radio(
        "Dataset",
        dataset_opts,
        index=(
            dataset_opts.index(
                st.session_state['selected_dataset']
            )
            if st.session_state['selected_dataset']
            in dataset_opts
            else 3
        ),
        key="sb_dataset_key"
    )

    # ============================================================
    # VARIABLE
    # ============================================================

    var_opts = [
        "Temperature (°C)",
        "Salinity (PSU)",
        "Current Speed (m/s)",
        "Sea Level Anomaly (m)"
    ]

    st.session_state['selected_variable'] = st.sidebar.selectbox(
        "Variable",
        var_opts,
        index=(
            var_opts.index(
                st.session_state['selected_variable']
            )
            if st.session_state['selected_variable']
            in var_opts
            else 0
        ),
        key="sb_variable_key"
    )

    # ============================================================
    # DEPTH LEVELS
    # ============================================================

    st.sidebar.markdown("### 2. DEPTH LEVELS")

    st.sidebar.caption(
        "15 standard depth levels from 0 to 1000 m."
    )

    selected_label = st.sidebar.selectbox(
        "Depth Level (15 Levels)",
        DEPTH_LEVELS_LABELS,
        index=st.session_state['selected_depth_idx'],
        key="sb_depth_key"
    )

    st.session_state['selected_depth_idx'] = (
        DEPTH_LEVELS_LABELS.index(selected_label)
    )

    current_depth_val = DEPTH_VALUES[
        st.session_state['selected_depth_idx']
    ]

    # ============================================================
    # LATITUDE / LONGITUDE
    # ============================================================

    st.sidebar.markdown("### 3. LAT & LON SEARCH")

    st.sidebar.caption(
        "Search & inspect any specific coordinate in Indian Ocean."
    )

    preset_choice = st.sidebar.selectbox(
        "Quick Coordinate Preset",
        list(LAT_LON_PRESETS.keys()),
        key="sb_latlon_preset_key"
    )

    if preset_choice != "Custom Coordinate":

        preset_lat, preset_lon = LAT_LON_PRESETS[
            preset_choice
        ]

        st.session_state['target_lat'] = preset_lat
        st.session_state['target_lon'] = preset_lon

    c_lat, c_lon = st.sidebar.columns(2)

    with c_lat:

        st.session_state['target_lat'] = st.sidebar.number_input(
            "Latitude (°N)",
            min_value=-30.0,
            max_value=25.0,
            value=float(
                st.session_state['target_lat']
            ),
            step=0.5,
            format="%.1f",
            key="sb_lat_num_input"
        )

    with c_lon:

        st.session_state['target_lon'] = st.sidebar.number_input(
            "Longitude (°E)",
            min_value=35.0,
            max_value=105.0,
            value=float(
                st.session_state['target_lon']
            ),
            step=0.5,
            format="%.1f",
            key="sb_lon_num_input"
        )

    # ============================================================
    # DATE & REGION
    # ============================================================

    st.sidebar.markdown("### 4. DATE & REGION")

    st.session_state['selected_date'] = st.sidebar.date_input(
        "Date",
        value=st.session_state['selected_date'],
        key="sb_date_key"
    )

    region_opts = [
        "All Indian Ocean",
        "Arabian Sea",
        "Bay of Bengal",
        "Equatorial Indian Ocean"
    ]

    st.session_state['selected_region'] = st.sidebar.selectbox(
        "Region (Optional)",
        region_opts,
        index=(
            region_opts.index(
                st.session_state['selected_region']
            )
            if st.session_state['selected_region']
            in region_opts
            else 0
        ),
        key="sb_region_key"
    )

    # ============================================================
    # RESET FILTERS
    # ============================================================

    st.sidebar.write("")

    if st.sidebar.button(
        "🔄 Reset Filters",
        use_container_width=True
    ):

        st.session_state['selected_dataset'] = (
            "ARGO vs GLORYS"
        )

        st.session_state['selected_variable'] = (
            "Temperature (°C)"
        )

        st.session_state['selected_depth_idx'] = 5

        st.session_state['selected_date'] = (
            date(2024, 5, 20)
        )

        st.session_state['selected_region'] = (
            "All Indian Ocean"
        )

        st.session_state['target_lat'] = 15.0
        st.session_state['target_lon'] = 65.0

        st.rerun()

    # ============================================================
    # RETURN CONTROLS
    # ============================================================

    return {
        "dataset": st.session_state['selected_dataset'],
        "variable": st.session_state['selected_variable'],
        "depth": current_depth_val,
        "date": st.session_state['selected_date'],
        "region": st.session_state['selected_region'],
        "target_lat": st.session_state['target_lat'],
        "target_lon": st.session_state['target_lon'],
        "show_floats": True,
        "show_heatmap": True
    }