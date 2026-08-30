"""
components/footer.py
====================
Footer component.
"""

import streamlit as st

def render_footer():
    st.markdown(
        """
        <div style="margin-top: 40px; padding: 20px 0; border-top: 1px solid #1E3A5F; text-align: center; font-size: 0.8rem; color: #8B949E;">
            <div style="display: flex; justify-content: center; gap: 24px; margin-bottom: 8px;">
                <span>🌊 <b>Indian Ocean Intelligence Platform</b></span>
                <span>•</span>
                <span>Data Sources: ARGO Floats Array | GLORYS Reanalysis (Copernicus Marine)</span>
                <span>•</span>
                <span>Version 2.4.0 (Frontend Edition)</span>
            </div>
            <div style="color: #00ADB5;">
                ⚡ Built with Streamlit, PyDeck & Plotly &nbsp;|&nbsp; Ready for AI/Backend API Pipeline Integration
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
