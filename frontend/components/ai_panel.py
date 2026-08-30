"""
components/ai_panel.py
======================
AI Reconstruction profile component with 100% locked non-movable chart and hover mode bar.
"""

import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go

def render_ai_profile_chart(df_profile):
    col_chart, col_legend = st.columns([2.0, 1.0])

    with col_chart:
        fig = go.Figure()

        # Observed ARGO (Blue Solid)
        fig.add_trace(go.Scatter(
            x=df_profile['Depth (m)'],
            y=df_profile['Observed (0-100m)'],
            mode='lines+markers',
            name='Observed (ARGO)',
            line=dict(color='#2563EB', width=2),
            marker=dict(size=5, color='#1D4ED8')
        ))

        # AI Predicted (Red Dashed)
        fig.add_trace(go.Scatter(
            x=df_profile['Depth (m)'],
            y=df_profile['AI Predicted (150-1000m)'],
            mode='lines+markers',
            name='AI Predicted',
            line=dict(color='#DC2626', width=2, dash='dash'),
            marker=dict(size=5, color='#B91C1C')
        ))

        # Missing (Grey Dotted)
        fig.add_trace(go.Scatter(
            x=df_profile['Depth (m)'],
            y=df_profile['Missing Reference'],
            mode='lines',
            name='Missing',
            line=dict(color='#94A3B8', width=1.5, dash='dot')
        ))

        fig.update_layout(
            title=dict(text="AI RECONSTRUCTION PROFILE", font=dict(family="Outfit", size=11, color="#0F172A")),
            dragmode=False,
            xaxis=dict(
                title=dict(text="Depth (m)", font=dict(color="#0F172A", size=10)),
                tickfont=dict(color="#0F172A", size=10),
                gridcolor="#E2E8F0",
                range=[0, 1050],
                fixedrange=True
            ),
            yaxis=dict(
                title=dict(text="Temperature (°C)", font=dict(color="#0F172A", size=10)),
                tickfont=dict(color="#0F172A", size=10),
                gridcolor="#E2E8F0",
                range=[5, 32],
                fixedrange=True
            ),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=40, r=15, t=35, b=35),
            height=220,
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

    with col_legend:
        st.markdown(
            """<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 8px; margin-top: 40px; font-size: 0.74rem;">
<div><span style="color: #64748B;">Observed Depths</span><br><strong style="color: #0F172A;">0 – 100 m</strong></div>
<div style="margin-top: 6px;"><span style="color: #64748B;">Predicted Depths</span><br><strong style="color: #0F172A;">150 – 1000 m</strong></div>
</div>""",
            unsafe_allow_html=True
        )


def render_ai_reconstruction_maps(df_obs, df_ai, df_diff):
    st.markdown('<div style="font-family: \'Outfit\', sans-serif; font-size: 0.9rem; font-weight: 700; color: #0F172A; margin-bottom: 6px;">🤖 SPATIAL RECONSTRUCTION GRID COMPARISON</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("1️⃣ OBSERVED (ARGO SPARSE)")
        v1 = pdk.ViewState(latitude=14.0, longitude=77.0, zoom=3.8)
        r1 = pdk.Deck(layers=[pdk.Layer("ScatterplotLayer", df_obs.dropna(), get_position=["longitude", "latitude"], get_color="[37, 99, 235, 200]", get_radius=35000)], initial_view_state=v1)
        st.pydeck_chart(r1, use_container_width=True)
    with c2:
        st.caption("2️⃣ AI RECONSTRUCTION (DENSE)")
        r2 = pdk.Deck(layers=[pdk.Layer("ScatterplotLayer", df_ai, get_position=["longitude", "latitude"], get_color="[220, 38, 38, 200]", get_radius=35000)], initial_view_state=v1)
        st.pydeck_chart(r2, use_container_width=True)
    with c3:
        st.caption("3️⃣ DIFFERENCE (AI - OBSERVED)")
        r3 = pdk.Deck(layers=[pdk.Layer("ScatterplotLayer", df_diff, get_position=["longitude", "latitude"], get_color="[234, 88, 12, 200]", get_radius=35000)], initial_view_state=v1)
        st.pydeck_chart(r3, use_container_width=True)
