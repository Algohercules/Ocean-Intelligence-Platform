"""
components/comparison_chart.py
==============================
ARGO vs GLORYS comparison component with 100% locked non-movable chart and hover mode bar.
"""

import plotly.graph_objects as go
import streamlit as st

def render_argo_vs_glorys_chart(df_comp, stats):
    col_chart, col_stats = st.columns([2.0, 1.0])

    with col_chart:
        fig = go.Figure()

        # ARGO line (Blue)
        fig.add_trace(go.Scatter(
            x=df_comp['Depth (m)'],
            y=df_comp['ARGO (°C)'],
            mode='lines+markers',
            name='ARGO',
            line=dict(color='#2563EB', width=2),
            marker=dict(size=5, color='#1D4ED8', symbol='diamond')
        ))

        # GLORYS line (Green)
        fig.add_trace(go.Scatter(
            x=df_comp['Depth (m)'],
            y=df_comp['GLORYS (°C)'],
            mode='lines+markers',
            name='GLORYS',
            line=dict(color='#16A34A', width=2),
            marker=dict(size=5, color='#15803D', symbol='circle')
        ))

        fig.update_layout(
            title=dict(text="ARGO VS GLORYS (PROFILE)", font=dict(family="Outfit", size=11, color="#0F172A")),
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
                range=[5, 32],
                fixedrange=True
            ),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=45, r=15, t=40, b=40),
            height=240,
            legend=dict(orientation="h", yanchor="bottom", y=0.75, xanchor="right", x=0.95, font=dict(color="#0F172A", size=9))
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
                <div style="font-weight: 700; color: #0F172A; text-transform: uppercase; margin-bottom: 6px; font-size: 0.72rem;">STATISTICS</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #64748B;">RMSE</span>
                    <strong style="color: #0F172A;">: {stats['rmse']}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #64748B;">MAE</span>
                    <strong style="color: #0F172A;">: {stats['mae']}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #64748B;">R²</span>
                    <strong style="color: #0F172A;">: {stats['r2']}</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #64748B;">Bias</span>
                    <strong style="color: #0F172A;">: {stats['bias']}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
