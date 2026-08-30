"""
pages/8_Info.py
================
Platform Information, Data Sources, Terminology, & System Architecture Page.
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

st.set_page_config(
    page_title="Info | Indian Ocean Intelligence Platform",
    page_icon="ℹ️",
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

render_header(active_page="Info")
controls = render_sidebar()

# Page Header
st.markdown(
    """
    <div style="background: rgba(13, 27, 42, 0.7); border: 1px solid #1E3A5F; border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;">
        <h2 style="font-family: 'Outfit', sans-serif; color: #38BDF8; margin: 0; font-size: 1.4rem;">
            ℹ️ INDIAN OCEAN INTELLIGENCE PLATFORM - DOCUMENTATION & GUIDE
        </h2>
        <p style="color: #94A3B8; margin: 6px 0 0 0; font-size: 0.9rem;">
            Comprehensive guide to oceanographic datasets, mathematical metrics, machine learning reconstructions, and system architecture.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

c_info1, c_info2 = st.columns([1.8, 1.2])

with c_info1:
    # 1. About the Platform
    st.markdown(
        """
        <div class="info-card-box" style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:16px; margin-bottom:16px;">
            <h3 style="font-family:'Outfit', sans-serif; color:#0F172A; margin-top:0; font-size:1.1rem; border-bottom:2px solid #38BDF8; padding-bottom:6px;">
                1. 🌐 About the Platform
            </h3>
            <p style="color:#334155; font-size:0.92rem; line-height:1.6;">
                The <b>Indian Ocean Intelligence Platform</b> is an advanced oceanographic decision-support platform engineered to visualize, analyze, and predict 3D subsurface ocean temperature structures across the Indian Ocean basin (30°S to 30°N, 30°E to 120°E). By fusing in-situ autonomous float observations, high-resolution numerical reanalysis, and deep learning reconstructions, the platform provides marine scientists and policymakers with real-time ocean intelligence.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Ocean Data Sources & Comparisons
    st.markdown(
        """
        <div class="info-card-box" style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:16px; margin-bottom:16px;">
            <h3 style="font-family:'Outfit', sans-serif; color:#0F172A; margin-top:0; font-size:1.1rem; border-bottom:2px solid #38BDF8; padding-bottom:6px;">
                2. 📡 Ocean Data Sources & Comparisons
            </h3>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-top:10px;">
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:12px;">
                    <h4 style="color:#0284C7; margin:0 0 6px 0; font-size:0.95rem;">📡 ARGO Observations</h4>
                    <p style="color:#475569; font-size:0.85rem; margin:0; line-height:1.5;">
                        <b>In-Situ Direct Measurement:</b> Autonomous profiling floats descending from the surface down to 2,000m depth every 10 days, recording physical CTD (Conductivity, Temperature, Depth) profiles.
                    </p>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:12px;">
                    <h4 style="color:#0284C7; margin:0 0 6px 0; font-size:0.95rem;">🌊 GLORYS Reanalysis</h4>
                    <p style="color:#475569; font-size:0.85rem; margin:0; line-height:1.5;">
                        <b>Numerical Hydrodynamic Model:</b> Copernicus Marine Service global ocean reanalysis (1/12° spatial resolution, 50 vertical depth levels) combining satellite altimetry and ocean physics equations.
                    </p>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:12px;">
                    <h4 style="color:#0284C7; margin:0 0 6px 0; font-size:0.95rem;">🤖 AI Reconstruction</h4>
                    <p style="color:#475569; font-size:0.85rem; margin:0; line-height:1.5;">
                        <b>Deep Learning Interpolation:</b> Convolutional & Transformer Neural Networks trained on historical CTD profiles to estimate continuous 3D subsurface fields from satellite sea surface temperature & anomaly data.
                    </p>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:12px;">
                    <h4 style="color:#0284C7; margin:0 0 6px 0; font-size:0.95rem;">📊 ARGO vs GLORYS</h4>
                    <p style="color:#475569; font-size:0.85rem; margin:0; line-height:1.5;">
                        <b>Validation Benchmark:</b> Direct statistical residual evaluation matching in-situ ARGO point profiles against collocated GLORYS model grid cells to quantify hydrodynamic model bias.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3. Marine Heatwave & Analysis
    st.markdown(
        """
        <div class="info-card-box" style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:16px; margin-bottom:16px;">
            <h3 style="font-family:'Outfit', sans-serif; color:#0F172A; margin-top:0; font-size:1.1rem; border-bottom:2px solid #38BDF8; padding-bottom:6px;">
                3. 🔥 Marine Heatwave (MHW) & Regional Analysis
            </h3>
            <p style="color:#334155; font-size:0.9rem; line-height:1.6;">
                <b>Marine Heatwaves (MHWs)</b> are prolonged discrete warm water anomalies where sea temperatures exceed the 90th percentile of local historical climatology for 5 consecutive days or more. The platform categorizes events into 4 severity tiers: <i>Category I (Moderate)</i>, <i>Category II (Strong)</i>, <i>Category III (Severe)</i>, and <i>Category IV (Extreme)</i>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c_info2:
    # 4. Statistical Metrics Glossary
    st.markdown(
        """
        <div class="info-card-box" style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:16px; margin-bottom:16px;">
            <h3 style="font-family:'Outfit', sans-serif; color:#0F172A; margin-top:0; font-size:1.1rem; border-bottom:2px solid #38BDF8; padding-bottom:6px;">
                4. 📐 Mathematical & Metric Glossary
            </h3>
            <ul style="color:#334155; font-size:0.88rem; line-height:1.8; padding-left:18px; margin:0;">
                <li><b>Temperature (°C):</b> Subsurface thermal reading in Celsius.</li>
                <li><b>Depth (m):</b> Vertical ocean level from surface (0m) to bathymetry (1000m+).</li>
                <li><b>Anomaly (°C):</b> Deviation from 30-year historical climatological baseline.</li>
                <li><b>Data Coverage (%):</b> Percentage of spatial grid cells with valid ocean observations.</li>
                <li><b>RMSE (Root Mean Square Error):</b> Square root of average squared errors between model and observation. Lower is better.</li>
                <li><b>MAE (Mean Absolute Error):</b> Average magnitude of absolute residual errors.</li>
                <li><b>R² (Coefficient of Determination):</b> Measure of variance explained by model (1.0 = perfect fit).</li>
                <li><b>Bias (°C):</b> Mean directional error showing systematic over/underestimation.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 5. Technology Stack & Architecture
    st.markdown(
        """
        <div class="info-card-box" style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:16px;">
            <h3 style="font-family:'Outfit', sans-serif; color:#0F172A; margin-top:0; font-size:1.1rem; border-bottom:2px solid #38BDF8; padding-bottom:6px;">
                5. ⚙️ Technology Stack & Architecture
            </h3>
            <table style="width:100%; font-size:0.85rem; color:#334155; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:6px 0; font-weight:600;">Frontend UI:</td><td>Streamlit 1.31, CSS3, HTML5</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:6px 0; font-weight:600;">Visualization:</td><td>Plotly 5.18, PyDeck 0.8, Mapbox Dark</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:6px 0; font-weight:600;">Data Processing:</td><td>Pandas 2.1, NumPy 1.26</td></tr>
                <tr><td style="padding:6px 0; font-weight:600;">Architecture:</td><td>Modular Component Architecture</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

render_footer()
