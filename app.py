chart_left, chart_right = st.columns(2)
with chart_left:
    st.markdown("#### Signal-strength distribution")
    if rsrp_col:
        fig = px.histogram(
            visible, x=rsrp_col, nbins=35, color="Signal_Level",
            labels={rsrp_col: "RSRP (dBm)", "Signal_Level": "Signal level"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No RSRP column detected.")
with chart_right:
    st.markdown("#### Signal-level breakdown")
    if "Signal_Level" in visible.columns:
        counts = (
            visible["Signal_Level"].fillna("Unknown").value_counts()
            .rename_axis("Signal level").reset_index(name="Observations")
        )
        fig = px.bar(counts, x="Signal level", y="Observations", color="Signal level")
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

chart_left, chart_right = st.columns(2)
with chart_left:
    st.markdown("#### Throughput")
    throughput_cols = [c for c in [dl_col, ul_col] if c]
    if throughput_cols:
        long_df = visible[[time_col] + throughput_cols].copy() if has_time else visible[throughput_cols].copy()
        if has_time:
            long_df = long_df[long_df[time_col].notna()]
            long_df = long_df.melt(id_vars=[time_col], var_name="Metric", value_name="Bitrate")
            long_df["Metric"] = long_df["Metric"].map({
                dl_col: "Download", ul_col: "Upload"
            }).fillna(long_df["Metric"])
            fig = px.line(long_df, x=time_col, y="Bitrate", color="Metric", title="Download / upload bitrate")
        else:
            long_df = long_df.melt(var_name="Metric", value_name="Bitrate")
            long_df["Metric"] = long_df["Metric"].map({
                dl_col: "Download", ul_col: "Upload"
            }).fillna(long_df["Metric"])
            fig = px.box(long_df, x="Metric", y="Bitrate", title="Bitrate distribution")
        fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Download/upload bitrate columns not found.")
with chart_right:
    st.markdown("#### Neighbor vs serving signal")
    neighbor_col = available_col(visible, "neighbor_rsrp")
    if rsrp_col and neighbor_col:
        compare = visible[[rsrp_col, neighbor_col]].copy()
        compare = compare.rename(columns={rsrp_col: "Serving RSRP", neighbor_col: "Neighbor RSRP"})
        compare = compare.apply(numeric_series).dropna(how="all")
        compare_long = compare.melt(var_name="Measurement", value_name="RSRP (dBm)")
        fig = px.box(compare_long, x="Measurement", y="RSRP (dBm)", color="Measurement")
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Neighbor RSRP is not guaranteed to identify a specific target cell.")
    else:
        st.info("Serving RSRP and neighbor RSRP are both required for this comparison.")

# ----------------------------- Mobility map ---------------------------------
st.subheader("Mobility track")
lat_col = available_col(visible, "latitude")
lon_col = available_col(visible, "longitude")
if lat_col and lon_col:
    map_df = visible.copy()
    map_df[lat_col] = numeric_series(map_df[lat_col])
    map_df[lon_col] = numeric_series(map_df[lon_col])
    map_df = map_df.dropna(subset=[lat_col, lon_col])
    map_df = map_df[
        map_df[lat_col].between(-90, 90) & map_df[lon_col].between(-180, 180)
    ]
    if not map_df.empty:
        color_col = rsrp_col if rsrp_col else None
        fig = px.scatter_mapbox(
            map_df,
            lat=lat_col,
            lon=lon_col,
            color=color_col,
            hover_name="Source_Trace",
            hover_data=[c for c in [time_col, cell_col, rsrp_col, sinr_col] if c and c in map_df.columns],
            zoom=13,
            height=450,
            mapbox_style="open-street-map",
            title="Recorded GPS points",
            labels={rsrp_col: "RSRP (dBm)"} if rsrp_col else None,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=45, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Latitude/longitude columns were found, but no valid coordinates are available.")
else:
    st.info("GPS columns not found; mobility map is unavailable.")

# ----------------------------- Data table & export --------------------------
st.subheader("Filtered measurements")
with st.expander("View data", expanded=False):
    st.dataframe(visible, use_container_width=True, height=350)

csv_bytes = visible.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered measurements (CSV)",
    data=csv_bytes,
    file_name="5g_monitoring_filtered.csv",
    mime="text/csv",
)

# Show detected mappings for debugging and reproducibility.
with st.expander("Detected source-column mapping"):
    mapping_rows = [
        {
            "Dashboard field": DISPLAY_NAMES.get(key, key),
            "Detected source column": value if value is not None else "Not found",
        }
        for key, value in column_mapping.items()
    ]
    st.dataframe(pd.DataFrame(mapping_rows), use_container_width=True, hide_index=True)

st.caption(
    "Note: A stronger neighboring measurement can be a handover candidate indicator, but "
    "actual handover decisions depend on target-cell identity, radio conditions, mobility, "
    "network configuration, hysteresis, time-to-trigger and network-side signaling."
)
