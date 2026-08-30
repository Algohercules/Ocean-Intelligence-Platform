"""
components/data_table.py
========================
Data table component.
"""

import streamlit as st
import pandas as pd

def render_avg_temp_depth_table(df_depth, selected_depth=75):
    st.markdown(
        """<div style="font-family: 'Outfit', sans-serif; font-size: 0.82rem; font-weight: 700; color: #0F172A; text-transform: uppercase; margin-bottom: 4px;">
AVERAGE TEMPERATURE BY DEPTH (SELECTED AREA)
</div>""",
        unsafe_allow_html=True
    )

    def style_row(row):
        if row['Depth (m)'] == selected_depth:
            return ['background-color: #BAE6FD; color: #0369A1; font-weight: bold;'] * len(row)
        return [''] * len(row)

    styled_df = df_depth.style.apply(style_row, axis=1).format({
        'Avg Temp (°C)': '{:.1f}',
        'Min Temp (°C)': '{:.1f}',
        'Max Temp (°C)': '{:.1f}',
        'Anomaly (°C)': '{:+.1f}'
    })

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=240,
        hide_index=True
    )
