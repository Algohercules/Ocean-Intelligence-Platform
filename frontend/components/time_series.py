"""
components/time_series.py
========================
Plotly Time Series component with 100% locked non-movable chart and hover mode bar.
"""

import plotly.graph_objects as go
import streamlit as st

def render_time_series_chart(df_ts, stats):
    col_chart, col_stats = st.columns([2.0, 1.0])

    with col_chart:
        fig = go.Figure()

        # Red line with markers
        fig.add_trace(go.Scatter(
            x=df_ts['Year'],
            y=df_ts['Temperature (°C)'],
            mode='lines+markers',
            name='Average Temperature',
            line=dict(color='#DC2626', width=2),
            marker=dict(size=5, color='#B91C1C'),
            hovertemplate="<b>Year:</b> %{x}<br><b>Temp:</b> %{y:.2f} °C<extra></extra>"
        ))

        fig.update_layout(
            title=dict(text="TIME SERIES (AVERAGE TEMP)", font=dict(family="Outfit", size=11, color="#0F172A")),
            dragmode=False,
            xaxis=dict(
                title=dict(text="Year", font=dict(color="#0F172A", size=11)),
                tickfont=dict(color="#0F172A", size=11),
                gridcolor="#E2E8F0",
                fixedrange=True
            ),
            yaxis=dict(
                title=dict(text="Temperature (°C)", font=dict(color="#0F172A", size=11)),
                tickfont=dict(color="#0F172A", size=11),
                gridcolor="#E2E8F0",
                range=[25.5, 30.5],
                fixedrange=True
            ),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=45, r=15, t=40, b=40),
            height=240,
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

    with col_stats:
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px; margin-top: 25px; font-size: 0.78rem; font-family: 'Inter', sans-serif;">
                <div style="font-weight: 700; color: #0F172A; text-transform: uppercase; margin-bottom: 6px; font-size: 0.72rem;">TREND STATISTICS<br>(2000–2024)</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #64748B;">Trend</span>
                    <strong style="color: #DC2626;">: +0.68 °C / decade</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #64748B;">Max</span>
                    <strong style="color: #0F172A;">: 29.2 °C (2024)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #64748B;">Min</span>
                    <strong style="color: #0F172A;">: 26.3 °C (2000)</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #64748B;">Mean</span>
                    <strong style="color: #0F172A;">: 27.8 °C</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
