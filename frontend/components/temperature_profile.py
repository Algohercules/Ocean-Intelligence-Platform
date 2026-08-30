"""
components/temperature_profile.py
==================================
Plotly Temperature Profile component with 100% locked non-movable chart and hover mode bar.
"""

import plotly.graph_objects as go
import streamlit as st

def render_temperature_profile_chart(df_profile, selected_depth=75):
    fig = go.Figure()

    # Red profile line with markers
    fig.add_trace(go.Scatter(
        x=df_profile['Depth (m)'],
        y=df_profile['Temperature (°C)'],
        mode='lines+markers',
        name='Avg Profile',
        line=dict(color='#DC2626', width=2),
        marker=dict(size=6, color='#DC2626', symbol='circle'),
        hovertemplate="<b>Depth:</b> %{x} m<br><b>Temp:</b> %{y:.2f} °C<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="TEMPERATURE PROFILE (AVERAGE)", font=dict(family="Outfit", size=11, color="#0F172A")),
        dragmode=False,
        xaxis=dict(
            title=dict(text="Depth (m)", font=dict(color="#0F172A", size=11)),
            tickfont=dict(color="#0F172A", size=11),
            gridcolor="#E2E8F0",
            range=[0, 1050],
            fixedrange=True
        ),
        yaxis=dict(
            title=dict(text="Temperature (°C)", font=dict(color="#0F172A", size=11)),
            tickfont=dict(color="#0F172A", size=11),
            gridcolor="#E2E8F0",
            range=[0, 32],
            fixedrange=True
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=45, r=25, t=40, b=40),
        height=260,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            'displayModeBar': 'hover',
            'displaylogo': False,
            'responsive': True,
            'scrollZoom': False,
            'doubleClick': False,
            'showAxisDragHandles': False,
            'showAxisRangeEntryBoxes': False
        }
    )
