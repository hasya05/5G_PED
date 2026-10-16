import math
import random
import time
from collections import deque

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="5G PED Monitoring", page_icon="📡", layout="wide")

st.title("📡 5G PED Monitoring Framework")
st.caption("Prediction • Explanation • Decision | Synthetic 5G handover simulation")

st.info(
    "Demo note: all values are simulated. The handover score and feature contributions "
    "are illustrative—not a trained prediction model and not actual SHAP explanations."
)

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=80)
if "step" not in st.session_state:
    st.session_state.step = 0
if "serving" not in st.session_state:
    st.session_state.serving = -78.0
    st.session_state.neighbor = -83.0
    st.session_state.speed = 34.0
    st.session_state.load = 55.0

with st.sidebar:
    st.header("Simulation controls")
    auto_refresh = st.toggle("Auto-refresh every 3 seconds", value=False)
    if st.button("Generate new sample", type="primary", use_container_width=True):
        st.session_state.step += 1
        st.session_state.serving = max(-115, min(-55, st.session_state.serving + random.uniform(-7, 5)))
        st.session_state.neighbor = max(-115, min(-55, st.session_state.neighbor + random.uniform(-6, 6)))
        st.session_state.speed = max(0, min(140, st.session_state.speed + random.uniform(-15, 15)))
        st.session_state.load = max(5, min(100, st.session_state.load + random.uniform(-12, 12)))
    st.markdown("---")
    st.caption("Use **Generate new sample** to change the simulated radio conditions.")

serving = st.session_state.serving
neighbor = st.session_state.neighbor
speed = st.session_state.speed
load = st.session_state.load

signal_risk = max(0.0, min(1.0, (-serving - 70) / 35))
neighbor_advantage = max(0.0, min(1.0, (neighbor - serving + 12) / 24))
load_pressure = load / 100
speed_factor = min(1.0, speed / 120)
score = max(0.0, min(1.0, 0.45 * signal_risk + 0.30 * neighbor_advantage + 0.15 * load_pressure + 0.10 * speed_factor))
recommended = score >= 0.55 and neighbor > serving - 3

st.session_state.history.append({
    "Sample": st.session_state.step,
    "Serving signal (dBm)": round(serving, 1),
    "Neighbor signal (dBm)": round(neighbor, 1),
    "Speed (km/h)": round(speed, 1),
    "Cell load (%)": round(load, 1),
    "Handover score": round(score, 3),
})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Serving-cell signal", f"{serving:.1f} dBm", help="Less negative is stronger.")
c2.metric("Neighbor-cell signal", f"{neighbor:.1f} dBm")
c3.metric("UE speed", f"{speed:.1f} km/h")
c4.metric("Serving-cell load", f"{load:.1f}%")

st.markdown("### PED decision")
left, right = st.columns([1, 1])
with left:
    st.metric("Illustrative handover score", f"{score:.0%}")
    st.progress(score)
    if recommended:
        st.success("Policy recommendation: consider handover to the neighbor cell.")
    else:
        st.warning("Policy recommendation: remain on the serving cell for this sample.")
    st.caption("Threshold: score ≥ 55% and neighbor signal is not materially weaker.")
with right:
    st.markdown("**Current context**")
    reasons = []
    if signal_risk > 0.55:
        reasons.append("Serving-cell signal is weak.")
    if neighbor_advantage > 0.55:
        reasons.append("Neighbor-cell signal has an advantage.")
    if load_pressure > 0.75:
        reasons.append("Serving-cell load is high.")
    if speed_factor > 0.65:
        reasons.append("User equipment is moving relatively fast.")
    if not reasons:
        reasons.append("No single risk indicator is especially high in this sample.")
    for reason in reasons:
        st.write(f"• {reason}")

st.markdown("### Explanation (illustrative feature contributions)")
contrib = pd.DataFrame({
    "Feature": ["Signal weakness", "Neighbor advantage", "Cell load", "UE speed"],
    "Contribution": [0.45 * signal_risk, 0.30 * neighbor_advantage, 0.15 * load_pressure, 0.10 * speed_factor],
})
fig = go.Figure(go.Bar(
    x=contrib["Contribution"], y=contrib["Feature"], orientation="h",
    text=[f"{v:.2f}" for v in contrib["Contribution"]], textposition="outside"
))
fig.update_layout(height=300, margin=dict(l=10, r=30, t=10, b=10), xaxis_title="Contribution to score", yaxis_title="")
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Recent simulation history")
history_df = pd.DataFrame(list(st.session_state.history))
if not history_df.empty:
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=history_df["Sample"], y=history_df["Serving signal (dBm)"], mode="lines+markers", name="Serving signal"))
    chart.add_trace(go.Scatter(x=history_df["Sample"], y=history_df["Neighbor signal (dBm)"], mode="lines+markers", name="Neighbor signal"))
    chart.update_layout(height=350, xaxis_title="Sample", yaxis_title="Signal (dBm)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(history_df.tail(12), use_container_width=True, hide_index=True)
    st.download_button("Download history as CSV", history_df.to_csv(index=False), "ped_simulation_history.csv", "text/csv")
else:
    st.write("Generate a sample to start collecting history.")

st.markdown("---")
st.caption("Educational prototype only. Do not use for live network decisions.")
if auto_refresh:
    time.sleep(3)
    st.rerun()
