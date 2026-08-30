"""
components/metric_cards.py
===========================
Horizontal pastel key indicator cards.
"""

import streamlit as st

def render_bottom_kpi_bar():
    st.markdown(
        """<div class="kpi-row-grid">
<div class="kpi-card kpi-1"><div class="kpi-title">AVG SURFACE TEMP</div><div class="kpi-value">28.6 °C</div></div>
<div class="kpi-card kpi-2"><div class="kpi-title">AVG 100m TEMP</div><div class="kpi-value">23.7 °C</div></div>
<div class="kpi-card kpi-3"><div class="kpi-title">AVG 500m TEMP</div><div class="kpi-value">13.1 °C</div></div>
<div class="kpi-card kpi-4"><div class="kpi-title">AVG 1000m TEMP</div><div class="kpi-value">7.5 °C</div></div>
<div class="kpi-card kpi-5"><div class="kpi-title">MAX TEMP (ALL DEPTHS)</div><div class="kpi-value">30.2 °C</div></div>
<div class="kpi-card kpi-6"><div class="kpi-title">MIN TEMP (ALL DEPTHS)</div><div class="kpi-value">6.7 °C</div></div>
<div class="kpi-card kpi-7"><div class="kpi-title">MEAN TEMP (ALL DEPTHS)</div><div class="kpi-value">18.4 °C</div></div>
<div class="kpi-card kpi-8"><div class="kpi-title">HEAT CONTENT (0–100m)</div><div class="kpi-value" style="font-size: 1.0rem;">1.25 × 10⁹ J/m²</div></div>
</div>""",
        unsafe_allow_html=True
    )


def render_ai_validation_metrics(metrics):
    st.markdown(
        f"""<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 8px 12px; font-size: 0.78rem;">
<div style="font-weight: 700; color: #0F172A; text-transform: uppercase; margin-bottom: 4px; font-size: 0.72rem;">STATISTICS</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 2px;"><span style="color: #64748B;">RMSE</span><strong style="color: #0F172A;">: {metrics['rmse']}</strong></div>
<div style="display: flex; justify-content: space-between; margin-bottom: 2px;"><span style="color: #64748B;">MAE</span><strong style="color: #0F172A;">: {metrics['mae']}</strong></div>
<div style="display: flex; justify-content: space-between; margin-bottom: 2px;"><span style="color: #64748B;">R²</span><strong style="color: #0F172A;">: {metrics['r2']}</strong></div>
<div style="display: flex; justify-content: space-between;"><span style="color: #64748B;">Bias</span><strong style="color: #0F172A;">: {metrics['bias']}</strong></div>
</div>""",
        unsafe_allow_html=True
    )
