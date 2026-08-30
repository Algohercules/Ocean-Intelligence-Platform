"""
components/area_selection.py
=============================
Selected Area Info & Quick Stats component with Lat/Lon Searched Point inspection.
"""

import streamlit as st

def render_area_selection_ui(selected_stats, current_depth=75):
    st.markdown(
        f"""<div class="info-card-box">
<div class="info-card-header">SELECTED AREA INFO</div>
<div style="font-size: 0.8rem; color: #334155; line-height: 1.6;">
<div><span style="color: #64748B;">Target Point</span> <strong style="color: #2563EB;">: {selected_stats['bounds']}</strong></div>
<div><span style="color: #64748B;">Region</span> <strong style="color: #0F172A;">: {selected_stats['region']}</strong></div>
<div><span style="color: #64748B;">Spatial Resolution</span> <strong style="color: #0F172A;">: {selected_stats.get('area_km2', 'Continuous Point')}</strong></div>
</div>
</div>""",
        unsafe_allow_html=True
    )


def render_quick_stats_ui(selected_stats, current_depth=75):
    nearest_argo = selected_stats.get('nearest_argo_id', 'WMO_6903142')
    argo_dist = selected_stats.get('nearest_argo_dist', '24 km')
    
    st.markdown(
        f"""<div class="info-card-box">
<div class="info-card-header">QUICK STATS ({current_depth} M)</div>
<div style="font-size: 0.8rem; color: #334155; line-height: 1.6;">
<div><span style="color: #64748B;">Average Temp</span> <strong style="color: #0F172A;">: {selected_stats['avg_temp']} °C</strong></div>
<div><span style="color: #64748B;">Min Temp</span> <strong style="color: #0F172A;">: {selected_stats['min_temp']} °C</strong></div>
<div><span style="color: #64748B;">Max Temp</span> <strong style="color: #0F172A;">: {selected_stats['max_temp']} °C</strong></div>
<div><span style="color: #64748B;">Anomaly</span> <strong style="color: #0F172A;">: +{selected_stats['anomaly']} °C</strong></div>
<div><span style="color: #64748B;">Status</span> <strong style="color: #DC2626;">: {selected_stats['status']}</strong></div>
</div>
<div style="border-top: 1px solid #E2E8F0; margin-top: 8px; padding-top: 6px; font-size: 0.78rem; color: #334155; line-height: 1.5;">
<div><span style="color: #64748B;">Nearest ARGO</span> <strong style="color: #059669;">: {nearest_argo} ({argo_dist})</strong></div>
<div><span style="color: #64748B;">Data Coverage</span> <strong style="color: #0F172A;">: {selected_stats['data_coverage']}</strong></div>
<div><span style="color: #64748B;">AI Coverage</span> <strong style="color: #0F172A;">: {selected_stats['ai_coverage']}</strong></div>
</div>
</div>""",
        unsafe_allow_html=True
    )
