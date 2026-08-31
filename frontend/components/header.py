"""
components/header.py
====================
Header navigation bar component matching exact reference layout inside top header banner.
"""

import streamlit as st

def render_header(active_page="Dashboard"):
    if 'sidebar_open' not in st.session_state:
        st.session_state['sidebar_open'] = True

    # Native Python toggle button
    if st.button("☰", key="header_toggle_sidebar_btn"):
        st.session_state['sidebar_open'] = not st.session_state['sidebar_open']
        st.rerun()

    # Active Tab Helper Styles
    def nav_style(page_name):
        is_active = (active_page == page_name)
        if is_active:
            return (
                "background: rgba(56, 189, 248, 0.25); "
                "border: 1.8px solid #38BDF8; "
                "color: #FFFFFF; "
                "font-weight: 700; "
                "padding: 6px 14px; "
                "border-radius: 6px; "
                "text-decoration: none; "
                "display: inline-block;"
            )
        else:
            return (
                "color: #CBD5E1; "
                "padding: 6px 10px; "
                "text-decoration: none; "
                "font-weight: 600; "
                "display: inline-block; "
                "transition: color 0.15s ease-in-out;"
            )

    # Top Header Banner HTML with Clean Streamlit Multipage Routes
    st.markdown(
        f"""<div class="top-header-banner">
<div style="display: flex; flex-direction: column; justify-content: center; flex-shrink: 0; margin-left: 0px;">
<div class="top-header-title">INDIAN OCEAN INTELLIGENCE PLATFORM</div>
<div class="top-header-sub">Explore. Analyze. Predict.</div>
</div>
<div style="display: flex; gap: 6px; align-items: center; font-size: 0.83rem; font-weight: 600; white-space: nowrap; flex-shrink: 0; margin-left: auto; margin-right: 120px;">
<a href="/Dashboard" target="_self" style="{nav_style('Dashboard')}">🏢 Dashboard</a>
<a href="/Explorer" target="_self" style="{nav_style('Explorer')}">🔍 Explorer</a>
<a href="/ARGO" target="_self" style="{nav_style('Argo')}">📡 Argo</a>
<a href="/Analysis" target="_self" style="{nav_style('Analysis')}">📊 Analysis</a>
<a href="/AI_Prediction" target="_self" style="{nav_style('AI Prediction')}">🤖 AI Prediction</a>
<a href="/Heatwave" target="_self" style="{nav_style('Heatwave')}">🔥 Heatwave</a>
<a href="/Reports" target="_self" style="{nav_style('Reports')}">📄 Reports</a>
<a href="/Info" target="_self" style="{nav_style('Info')}">ℹ️ Info</a>
</div>
</div>""",
        unsafe_allow_html=True
    )
