
# ---------------------------------------------------------
# AUTOMATIC SINGLE-CELL MONITORING MODE
# Insert after: columns = list(raw_df.columns)
# and the empty-columns check, before COLUMN MAPPING.
# ---------------------------------------------------------

column_lookup = {str(c).strip().lower(): c for c in raw_df.columns}

def detect_column(*names):
    for name in names:
        if name.lower() in column_lookup:
            return column_lookup[name.lower()]
    return None

single_rsrp_col = detect_column("RSRP")
single_rsrq_col = detect_column("RSRQ")
single_sinr_col = detect_column("SNR", "SINR")
single_cqi_col = detect_column("CQI")
single_rssi_col = detect_column("RSSI")
single_cell_col = detect_column("CellID", "Cell_ID", "Cell ID")
single_time_col = detect_column("Timestamp", "Time", "Datetime")
single_mode_col = detect_column("NetworkMode", "Network_Mode")
single_dl_col = detect_column("DL_bitrate", "DL bitrate", "Downlink")
single_ul_col = detect_column("UL_bitrate", "UL bitrate", "Uplink")
single_speed_col = detect_column("Speed", "Velocity")

# A target-cell column must explicitly identify a target or neighbour.
target_rsrp_col = detect_column(
    "Target_RSRP",
    "Target RSRP",
    "Neighbour_RSRP",
    "Neighbor_RSRP",
    "TargetCellRSRP",
    "NeighborCellRSRP",
)

# Enter single-cell mode when RSRP exists but target RSRP does not.
if single_rsrp_col is not None and target_rsrp_col is None:

    st.header("📶 Single-cell radio monitoring")
    st.info(
        "Single-cell mode enabled automatically. This dataset contains "
        "radio measurements for individual observations, but no explicit "
        "target-cell RSRP. The dashboard monitors signal quality and "
        "possible risk indicators; it does not claim that a handover "
        "should occur."
    )

    monitor = raw_df.copy()

    def get_numeric(column):
        if column is None:
            return pd.Series(np.nan, index=monitor.index)
        return pd.to_numeric(
            monitor[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

    monitor["Monitor_RSRP"] = get_numeric(single_rsrp_col)
    monitor["Monitor_RSRQ"] = get_numeric(single_rsrq_col)
    monitor["Monitor_SINR"] = get_numeric(single_sinr_col)
    monitor["Monitor_CQI"] = get_numeric(single_cqi_col)
    monitor["Monitor_RSSI"] = get_numeric(single_rssi_col)
    monitor["Monitor_DL"] = get_numeric(single_dl_col)
    monitor["Monitor_UL"] = get_numeric(single_ul_col)
    monitor["Monitor_Speed"] = get_numeric(single_speed_col)

    if single_time_col is not None:
        monitor["Monitor_Time"] = pd.to_datetime(
            monitor[single_time_col],
            errors="coerce",
        )
    else:
        monitor["Monitor_Time"] = pd.NaT

    if single_cell_col is not None:
        monitor["Monitor_Cell"] = monitor[single_cell_col].astype(str)
    else:
        monitor["Monitor_Cell"] = "Cell ID unavailable"

    if single_mode_col is not None:
        monitor["Monitor_Network"] = monitor[single_mode_col].astype(str)
    else:
        monitor["Monitor_Network"] = "Network mode unavailable"

    monitor = monitor.dropna(subset=["Monitor_RSRP"]).copy()

    if monitor.empty:
        st.error(
            "The RSRP column was detected, but no numeric RSRP values "
            "could be read. Check the dataset and its units."
        )
        st.stop()

    # Filters
    st.subheader("Filter observations")

    filter1, filter2 = st.columns(2)

    cell_options = sorted(monitor["Monitor_Cell"].dropna().unique().tolist())

    with filter1:
        chosen_cells = st.multiselect(
            "Cell ID",
            options=cell_options,
            default=cell_options,
        )

    filtered_monitor = monitor[
        monitor["Monitor_Cell"].isin(chosen_cells)
    ].copy()

    if filtered_monitor.empty:
        st.warning("No observations match the selected cell IDs.")
        st.stop()

    with filter2:
        if single_time_col is not None:
            available_times = filtered_monitor["Monitor_Time"].dropna()
            if not available_times.empty:
                time_min = available_times.min().to_pydatetime()
                time_max = available_times.max().to_pydatetime()

                time_range = st.slider(
                    "Observation time range",
                    min_value=time_min,
                    max_value=time_max,
                    value=(time_min, time_max),
                    format="YYYY-MM-DD HH:mm:ss",
                )

                filtered_monitor = filtered_monitor[
                    filtered_monitor["Monitor_Time"].isna()
                    | (
                        (filtered_monitor["Monitor_Time"] >= pd.Timestamp(time_range[0]))
                        & (filtered_monitor["Monitor_Time"] <= pd.Timestamp(time_range[1]))
                    )
                ]

    if filtered_monitor.empty:
        st.warning("No observations remain after filtering.")
        st.stop()

    # Signal classification is a configurable monitoring heuristic.
    # Confirm thresholds and measurement units for your dataset.
    def classify_signal(rsrp):
        if pd.isna(rsrp):
            return "Unknown"
        if rsrp >= -90:
            return "Strong"
        if rsrp >= -105:
            return "Moderate"
        if rsrp >= -115:
            return "Weak"
        return "Very weak"

    filtered_monitor["Signal_Level"] = (
        filtered_monitor["Monitor_RSRP"].apply(classify_signal)
    )

    filtered_monitor["Monitoring_Flag"] = np.select(
        [
            filtered_monitor["Monitor_RSRP"] < -115,
            filtered_monitor["Monitor_RSRP"] < -105,
        ],
        [
            "Very weak RSRP — investigate coverage",
            "Weak RSRP — monitor signal trend",
        ],
        default="No RSRP alert from configured thresholds",
    )

    # KPI cards
    st.subheader("Radio overview")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Observations",
        f"{len(filtered_monitor):,}",
    )
    k2.metric(
        "Average RSRP",
        f"{filtered_monitor['Monitor_RSRP'].mean():.1f} dBm",
    )
    k3.metric(
        "Weak-signal observations",
        f"{(filtered_monitor['Monitor_RSRP'] < -105).sum():,}",
    )

    if filtered_monitor["Monitor_SINR"].notna().any():
        k4.metric(
            "Average SNR/SINR",
            f"{filtered_monitor['Monitor_SINR'].mean():.1f} dB",
        )
    else:
        k4.metric("Average SNR/SINR", "Unavailable")

    # Signal distribution
    chart1, chart2 = st.columns(2)

    with chart1:
        st.subheader("Signal-strength distribution")
        fig = px.histogram(
            filtered_monitor,
            x="Monitor_RSRP",
            nbins=35,
            color="Signal_Level",
            labels={"Monitor_RSRP": "RSRP (dBm)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        st.subheader("Signal-level breakdown")
        level_counts = (
            filtered_monitor["Signal_Level"]
            .value_counts()
            .rename_axis("Signal level")
            .reset_index(name="Observations")
        )
        fig = px.bar(
            level_counts,
            x="Signal level",
            y="Observations",
            color="Signal level",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Signal trends
    st.subheader("Signal trend over time")

    trend = filtered_monitor.dropna(subset=["Monitor_Time"]).sort_values(
        "Monitor_Time"
    )

    if not trend.empty:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=trend["Monitor_Time"],
                y=trend["Monitor_RSRP"],
                mode="lines",
                name="RSRP",
            )
        )

        if trend["Monitor_RSRQ"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=trend["Monitor_Time"],
                    y=trend["Monitor_RSRQ"],
                    mode="lines",
                    name="RSRQ",
                    yaxis="y2",
                )
            )

        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis=dict(title="RSRP (dBm)"),
            yaxis2=dict(
                title="RSRQ (dB)",
                overlaying="y",
                side="right",
            ),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "A usable timestamp column is not available for trend analysis."
        )

    # Other radio indicators
    available_metrics = {
        "SNR / SINR (dB)": "Monitor_SINR",
        "CQI": "Monitor_CQI",
        "RSSI": "Monitor_RSSI",
        "Downlink bitrate": "Monitor_DL",
        "Uplink bitrate": "Monitor_UL",
        "Speed": "Monitor_Speed",
    }

    metric_names = [
        label
        for label, column in available_metrics.items()
        if filtered_monitor[column].notna().any()
    ]

    if metric_names:
        st.subheader("Additional radio and network indicators")

        selected_metric_name = st.selectbox(
            "Choose a metric",
            metric_names,
        )
        selected_metric_col = available_metrics[selected_metric_name]

        if not trend.empty:
            metric_trend = filtered_monitor.dropna(
                subset=[selected_metric_col]
            ).sort_values("Monitor_Time")

            if not metric_trend.empty and metric_trend["Monitor_Time"].notna().any():
                fig = px.line(
                    metric_trend.dropna(subset=["Monitor_Time"]),
                    x="Monitor_Time",
                    y=selected_metric_col,
                    labels={
                        selected_metric_col: selected_metric_name,
                        "Monitor_Time": "Timestamp",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

        metric_values = filtered_monitor[selected_metric_col].dropna()

        if not metric_values.empty:
            st.caption(
                f"{selected_metric_name}: "
                f"minimum {metric_values.min():.2f}, "
                f"average {metric_values.mean():.2f}, "
                f"maximum {metric_values.max():.2f}."
            )

    # Monitoring flags
    st.subheader("Observations requiring attention")

    flagged = filtered_monitor[
        filtered_monitor["Monitor_RSRP"] < -105
    ].copy()

    if flagged.empty:
        st.success(
            "No observations crossed the configured weak-RSRP threshold."
        )
    else:
        st.warning(
            f"{len(flagged):,} observations have RSRP below -105 dBm. "
            "This is a monitoring flag, not proof of a dropped connection "
            "or a required handover."
        )

        columns_to_show = [
            c for c in [
                single_time_col,
                single_cell_col,
                single_mode_col,
                single_rsrp_col,
                single_rsrq_col,
                single_sinr_col,
                single_cqi_col,
                single_dl_col,
                single_ul_col,
            ]
            if c is not None
        ]

        st.dataframe(
            flagged[columns_to_show],
            use_container_width=True,
            hide_index=True,
        )

    # Full data and download
    st.subheader("Filtered measurement data")

    st.dataframe(
        filtered_monitor,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download single-cell monitoring results",
        data=filtered_monitor.to_csv(index=False).encode("utf-8"),
        file_name="5g_single_cell_monitoring.csv",
        mime="text/csv",
    )

    st.caption(
        "Monitoring thresholds are heuristic and configurable. "
        "RSRP, RSRQ, SNR/SINR and bitrate units should be verified "
        "against the dataset documentation. This mode does not predict "
        "a target cell or execute a handover."
    )

    # Prevent the old two-cell workflow from running on this dataset.
    st.stop()
