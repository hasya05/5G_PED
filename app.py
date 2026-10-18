"""
5G Network Monitoring & Handover Recommendation Dashboard
---------------------------------------------------------
Run:
    pip install -r requirements.txt
    streamlit run app.py

This dashboard analyzes uploaded historical 5G CSV traces. Its "playback" view
simulates a live feed by revealing records up to a selected timestamp; it does
not connect to a live cellular network or execute real handovers.
"""

from __future__ import annotations

import io
import zipfile
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="5G Network Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styling ---------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.18);
        padding: 14px 16px;
        border-radius: 12px;
      }
      .small-note {font-size: 0.88rem; opacity: 0.78;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📡 5G Network Monitoring Dashboard")
st.caption(
    "Analyze signal quality, mobility, throughput and handover indicators from uploaded CSV traces."
)

# ----------------------------- Column mapping -------------------------------
ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "date_time", "monitor_time"],
    "rsrp": ["rsrp", "monitor_rsrp", "ss_rsrp"],
    "rsrq": ["rsrq", "monitor_rsrq"],
    "sinr": ["sinr", "snr", "monitor_sinr", "monitor_snr"],
    "rssi": ["rssi", "monitor_rssi"],
    "cqi": ["cqi", "monitor_cqi"],
    "cell_id": ["cellid", "cell_id", "cell id", "serving_cell_id", "eci"],
    "neighbor_rsrp": ["nrxrsrp", "nrx_rsrp", "neighbor_rsrp", "neighbour_rsrp"],
    "neighbor_rsrq": ["nrxrsrq", "nrx_rsrq", "neighbor_rsrq", "neighbour_rsrq"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng", "long"],
    "speed": ["speed", "velocity", "speed_kmh", "velocity_kmh"],
    "dl_bitrate": ["dl_bitrate", "download_bitrate", "downlink_bitrate", "dl_throughput"],
    "ul_bitrate": ["ul_bitrate", "upload_bitrate", "uplink_bitrate", "ul_throughput"],
    "ping_avg": ["pingavg", "ping_avg", "ping_average", "latency", "latency_ms"],
    "ping_loss": ["pingloss", "ping_loss", "packet_loss"],
    "network_mode": ["networkmode", "network_mode", "technology", "rat"],
    "operator": ["operatorname", "operator", "carrier"],
}

DISPLAY_NAMES = {
    "timestamp": "Timestamp",
    "rsrp": "RSRP",
    "rsrq": "RSRQ",
    "sinr": "SNR / SINR",
    "rssi": "RSSI",
    "cqi": "CQI",
    "cell_id": "Serving Cell ID",
    "neighbor_rsrp": "Neighbor RSRP",
    "neighbor_rsrq": "Neighbor RSRQ",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "speed": "Speed",
    "dl_bitrate": "Download bitrate",
    "ul_bitrate": "Upload bitrate",
    "ping_avg": "Average ping",
    "ping_loss": "Ping loss",
    "network_mode": "Network mode",
    "operator": "Operator",
}


def normalize_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def find_column(columns: Iterable[object], key: str):
    normalized = {normalize_name(c): c for c in columns}
    # Exact alias match first.
    for alias in ALIASES.get(key, []):
        if normalize_name(alias) in normalized:
            return normalized[normalize_name(alias)]
    # Allow underscore/space variations.
    compact = {normalize_name(c).replace("_", "").replace(" ", ""): c for c in columns}
    for alias in ALIASES.get(key, []):
        token = normalize_name(alias).replace("_", "").replace(" ", "")
        if token in compact:
            return compact[token]
    return None


def parse_timestamp(series: pd.Series) -> pd.Series:
    """Parse common G-NetTrack timestamp format and other standard timestamps."""
    values = series.astype("string").str.strip()
    parsed = pd.to_datetime(values, format="%Y.%m.%d_%H.%M.%S", errors="coerce")
    remaining = parsed.isna() & values.notna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(values.loc[remaining], errors="coerce")
    return parsed


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.replace("%", "", regex=False),
        errors="coerce",
    )


def read_csv_bytes(data: bytes, filename: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(data), low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), encoding="latin-1", low_memory=False)


def load_uploaded_files(uploaded_files):
    """Read CSV files and CSVs contained in uploaded ZIP archives."""
    frames = {}
    errors = []
    for upload in uploaded_files or []:
        filename = upload.name
        content = upload.getvalue()
        if filename.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    csv_names = [
                        n for n in archive.namelist()
                        if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")
                    ]
                    if not csv_names:
                        errors.append(f"{filename}: ZIP archive contains no CSV files.")
                    for inner_name in csv_names:
                        try:
                            frames[f"{filename} / {inner_name}"] = read_csv_bytes(
                                archive.read(inner_name), inner_name
                            )
                        except Exception as exc:
                            errors.append(f"{inner_name}: {exc}")
            except zipfile.BadZipFile:
                errors.append(f"{filename}: this ZIP file could not be opened.")
        else:
            try:
                frames[filename] = read_csv_bytes(content, filename)
            except Exception as exc:
                errors.append(f"{filename}: {exc}")
    return frames, errors


def prepare_frame(source_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = source_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    mapping = {key: find_column(df.columns, key) for key in ALIASES}

    # Create stable dashboard-friendly columns while preserving original columns.
    for key, source_col in mapping.items():
        target = f"Monitor_{key}"
        if source_col is not None and source_col in df.columns:
            df[target] = df[source_col]

    if mapping["timestamp"] is not None:
        df["Monitor_Time"] = parse_timestamp(df[mapping["timestamp"]])
    else:
        df["Monitor_Time"] = pd.NaT

    numeric_keys = [
        "rsrp", "rsrq", "sinr", "rssi", "cqi", "neighbor_rsrp",
        "neighbor_rsrq", "latitude", "longitude", "speed", "dl_bitrate",
        "ul_bitrate", "ping_avg", "ping_loss",
    ]
    for key in numeric_keys:
        col = f"Monitor_{key}"
        if col in df.columns:
            df[col] = numeric_series(df[col])

    if mapping["cell_id"] is not None:
        df["Monitor_cell_id"] = df[mapping["cell_id"]].astype("string").str.strip()

    # Remove rows that are entirely empty but retain rows with missing measurements.
    df = df.dropna(how="all").reset_index(drop=True)

    # Assign a signal category using RSRP when available.
    if "Monitor_rsrp" in df.columns:
        df["Signal_Level"] = pd.cut(
            df["Monitor_rsrp"],
            bins=[-np.inf, -115, -105, -95, -85, np.inf],
            labels=["Very weak", "Weak", "Fair", "Good", "Excellent"],
            right=False,
        ).astype("string")
        df.loc[df["Monitor_rsrp"].isna(), "Signal_Level"] = "Unknown"
    else:
        df["Signal_Level"] = "Unknown"

    return df, mapping


def available_col(df: pd.DataFrame, key: str) -> str | None:
    candidate = f"Monitor_{key}"
    return candidate if candidate in df.columns else None


def metric_mean(df: pd.DataFrame, col: str | None, suffix: str = "", digits: int = 1) -> str:
    if not col or col not in df.columns:
        return "Unavailable"
    values = numeric_series(df[col]).dropna()
    if values.empty:
        return "Unavailable"
    return f"{values.mean():.{digits}f}{suffix}"


def plot_line(df, x, y, title, y_label=None):
    fig = px.line(df, x=x, y=y, title=title, labels={x: "Time", y: y_label or y})
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), hovermode="x unified")
    return fig


def evaluate_handover(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Conservative evidence-based assessment.
    NRxRSRP is not assumed to identify a particular target cell. Therefore,
    signal-only evidence cannot confirm that a real handover is required.
    """
    rsrp_col = available_col(df, "rsrp")
    cell_col = available_col(df, "cell_id")
    neighbor_col = available_col(df, "neighbor_rsrp")

    if not rsrp_col or df[rsrp_col].dropna().empty:
        return (
            "Insufficient data",
            "No serving-cell RSRP measurements are available.",
            "Upload a trace containing serving-cell RSRP and timestamps.",
        )

    rsrp = numeric_series(df[rsrp_col]).dropna()
    if rsrp.empty:
        return (
            "Insufficient data",
            "Serving-cell RSRP could not be interpreted as numeric measurements.",
            "Check the RSRP column and its units.",
        )

    weak_count = int((rsrp < -105).sum())
    weak_fraction = weak_count / max(len(rsrp), 1)
    cell_changes = 0
    if cell_col and df[cell_col].notna().any():
        cells = df[cell_col].dropna().astype(str)
        cell_changes = int((cells != cells.shift()).sum() - 1) if len(cells) else 0
        cell_changes = max(cell_changes, 0)

    if cell_changes > 0:
        event_text = f"Serving-cell ID changes were observed ({cell_changes} change(s)); this is evidence of cell transitions, not proof of a successful handover procedure."
    else:
        event_text = "No serving-cell ID change was observed in the selected trace."

    if weak_fraction >= 0.25:
        signal_text = f"{weak_fraction:.0%} of valid RSRP samples are below -105 dBm, indicating sustained weak-signal risk."
    elif weak_count:
        signal_text = f"{weak_count:,} valid RSRP sample(s) are below -105 dBm; inspect the time-series and mobility context."
    else:
        signal_text = "No samples are below the configured -105 dBm weak-signal monitoring threshold."

    neighbor_evidence = bool(neighbor_col and df[neighbor_col].notna().any())
    if neighbor_evidence:
        decision = "Handover assessment needed"
        reason = (
            f"{signal_text} {event_text} Neighbor RSRP values exist, but the dataset does not provide a confirmed target-cell identity or network handover configuration. "
            "The trace alone cannot establish that handover is required."
        )
        action = "Compare serving and target-cell measurements, verify candidate-cell identity, and apply network-specific hysteresis/time-to-trigger rules."
    else:
        decision = "Insufficient data"
        reason = (
            f"{signal_text} {event_text} No usable neighbor RSRP values are available for candidate comparison."
        )
        action = "Collect identifiable neighboring-cell measurements and network handover parameters before making a handover recommendation."

    return decision, reason, action


# ----------------------------- Sidebar upload -------------------------------
with st.sidebar:
    st.header("Dataset")
    uploaded_files = st.file_uploader(
        "Upload CSV file(s) or a ZIP containing CSV files",
        type=["csv", "zip"],
        accept_multiple_files=True,
        help="Supports the 5G Production Dataset CSV format, including G-NetTrack timestamps.",
    )
    st.divider()
    st.subheader("Monitoring settings")
    weak_rsrp_threshold = st.slider(
        "Weak RSRP threshold (dBm)", min_value=-125, max_value=-85, value=-105, step=1
    )
    st.caption("This is a configurable alert threshold, not a universal handover trigger.")
    st.divider()
    st.caption("Historical CSV playback only. No connection to a live mobile network is made.")

frames, load_errors = load_uploaded_files(uploaded_files)

if load_errors:
    with st.expander("Upload warnings", expanded=True):
        for error in load_errors:
            st.warning(error)

if not frames:
    st.info(
        "Upload one or more CSV files, or a ZIP containing CSV files, to start. "
        "You can use the supplied 5G Production Dataset traces."
    )
    st.markdown(
        """
        **Expected fields (when available):** `Timestamp`, `CellID`, `RSRP`,
        `RSRQ`, `SNR`, `CQI`, `NRxRSRP`, `NRxRSRQ`, `DL_bitrate`, `UL_bitrate`,
        `PINGAVG`, `Latitude`, `Longitude`, and `Speed`.
        """
    )
    st.stop()

# Build selection list and normalize selected trace.
trace_names = list(frames.keys())
mode = st.sidebar.radio(
    "Analysis mode",
    ["Analyze one trace", "Combine selected traces"],
    help="Combining traces is useful for comparison, but separate journeys may not form one continuous trip.",
)
if mode == "Analyze one trace":
    selected_trace = st.sidebar.selectbox("Select trace", trace_names)
    source_df = frames[selected_trace].copy()
    source_df["Source_Trace"] = selected_trace
else:
    selected_traces = st.sidebar.multiselect(
        "Choose traces to combine", trace_names, default=trace_names
    )
    if not selected_traces:
        st.warning("Select at least one trace.")
        st.stop()
    source_df = pd.concat(
        [
            frames[name].assign(Source_Trace=name)
            for name in selected_traces
        ],
        ignore_index=True,
        sort=False,
    )
    selected_trace = "Combined traces"

data, column_mapping = prepare_frame(source_df)
if "Source_Trace" not in data.columns:
    data["Source_Trace"] = selected_trace

if data.empty:
    st.error("The selected dataset contains no usable rows.")
    st.stop()

# Time ordering is important for playback and time-series plots.
if data["Monitor_Time"].notna().any():
    data = data.sort_values("Monitor_Time", na_position="last").reset_index(drop=True)

# ----------------------------- Filters --------------------------------------
with st.sidebar:
    st.divider()
    st.subheader("Filters")
    if "Monitor_network_mode" in data.columns:
        modes = sorted(data["Monitor_network_mode"].dropna().astype(str).unique().tolist())
        chosen_modes = st.multiselect("Network mode", modes, default=modes)
    else:
        chosen_modes = []
    if "Monitor_cell_id" in data.columns:
        cell_values = sorted(data["Monitor_cell_id"].dropna().astype(str).unique().tolist())
        chosen_cells = st.multiselect("Serving Cell ID", cell_values, default=cell_values)
    else:
        chosen_cells = []
    traces = sorted(data["Source_Trace"].dropna().astype(str).unique().tolist())
    chosen_sources = st.multiselect("Source trace", traces, default=traces)

filtered = data.copy()
if "Monitor_network_mode" in filtered.columns and chosen_modes:
    filtered = filtered[filtered["Monitor_network_mode"].astype(str).isin(chosen_modes)]
if "Monitor_cell_id" in filtered.columns and chosen_cells:
    filtered = filtered[filtered["Monitor_cell_id"].astype(str).isin(chosen_cells)]
if "Source_Trace" in filtered.columns and chosen_sources:
    filtered = filtered[filtered["Source_Trace"].astype(str).isin(chosen_sources)]

if filtered.empty:
    st.warning("No rows match the current filters. Adjust the filters in the sidebar.")
    st.stop()

# ----------------------------- Playback -------------------------------------
st.subheader("Trace playback")
st.caption(
    "Move the slider to simulate incoming measurements from the historical trace. "
    "This is playback, not live network telemetry."
)
playback_on = st.checkbox("Enable playback window", value=False)
if playback_on:
    max_idx = max(len(filtered) - 1, 0)
    selected_idx = st.slider(
        "Replay through record",
        min_value=0,
        max_value=max_idx,
        value=max_idx,
        step=max(1, min(25, max(1, len(filtered) // 100))),
    )
    visible = filtered.iloc[: selected_idx + 1].copy()
    st.caption(f"Showing {len(visible):,} of {len(filtered):,} filtered records.")
else:
    visible = filtered.copy()

# ----------------------------- KPI row --------------------------------------
rsrp_col = available_col(visible, "rsrp")
rsrq_col = available_col(visible, "rsrq")
sinr_col = available_col(visible, "sinr")
dl_col = available_col(visible, "dl_bitrate")
ul_col = available_col(visible, "ul_bitrate")
cell_col = available_col(visible, "cell_id")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Observations", f"{len(visible):,}")
k2.metric("Average RSRP", metric_mean(visible, rsrp_col, " dBm"))
k3.metric("Average RSRQ", metric_mean(visible, rsrq_col, " dB"))
k4.metric("Average SNR / SINR", metric_mean(visible, sinr_col, " dB"))
k5.metric("Average download", metric_mean(visible, dl_col))

k6, k7, k8, k9 = st.columns(4)
if rsrp_col:
    rsrp_values = numeric_series(visible[rsrp_col]).dropna()
    weak_count = int((rsrp_values < weak_rsrp_threshold).sum())
else:
    weak_count = 0
k6.metric("Weak-signal samples", f"{weak_count:,}")
k7.metric("Unique serving cells", str(visible[cell_col].nunique()) if cell_col else "Unavailable")
k8.metric("Average upload", metric_mean(visible, ul_col))
ping_col = available_col(visible, "ping_avg")
k9.metric("Average ping", metric_mean(visible, ping_col, " ms"))

st.divider()

# ----------------------------- Handover panel -------------------------------
st.subheader("Handover assessment")
decision, reason, action = evaluate_handover(visible)
if decision == "Handover assessment needed":
    st.warning(f"**{decision}**\n\n{reason}")
elif decision == "Insufficient data":
    st.info(f"**{decision}**\n\n{reason}")
else:
    st.success(f"**{decision}**\n\n{reason}")
st.markdown(f"**Suggested next step:** {action}")
st.caption(
    "The app provides a measurement-based screening result only. It does not trigger, approve, "
    "or guarantee a cellular handover. RSRP thresholds are heuristics and should be configured "
    "for the specific network and use case."
)

# ----------------------------- Charts ---------------------------------------
st.subheader("Network performance over time")
time_col = "Monitor_Time"
has_time = visible[time_col].notna().any()

if has_time:
    timed = visible[visible[time_col].notna()].copy()
    chart_left, chart_right = st.columns(2)
    with chart_left:
        if rsrp_col:
            fig = px.line(
                timed, x=time_col, y=rsrp_col, color="Source_Trace" if timed["Source_Trace"].nunique() > 1 else None,
                title="Serving-cell RSRP over time",
                labels={time_col: "Timestamp", rsrp_col: "RSRP (dBm)", "Source_Trace": "Trace"},
            )
            fig.add_hline(y=weak_rsrp_threshold, line_dash="dash", annotation_text="Weak-signal threshold")
            fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("RSRP column not found; signal trend is unavailable.")
    with chart_right:
        if sinr_col:
            fig = px.line(
                timed, x=time_col, y=sinr_col, color="Source_Trace" if timed["Source_Trace"].nunique() > 1 else None,
                title="SNR / SINR over time",
                labels={time_col: "Timestamp", sinr_col: "SNR / SINR (dB)", "Source_Trace": "Trace"},
            )
            fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SNR/SINR column not found; quality trend is unavailable.")
else:
    st.info("No valid timestamps were detected. Check the selected trace's timestamp column.")

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
        map_df[lat_col].between(-90, 90)
        & map_df[lon_col].between(-180, 180)
    ]

    if not map_df.empty:
        color_col = rsrp_col if rsrp_col and rsrp_col in map_df.columns else None

       hover_cols = [
    col for col in ["CellId", "NetworkMode"]
    if col in map_df.columns
    ]

        fig = px.scatter_map(
            map_df,
            lat=lat_col,
            lon=lon_col,
            color=color_col,
            hover_data=hover_cols,
            zoom=12,
            height=500,
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=45, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            "Latitude/longitude columns were found, "
            "but no valid coordinates are available."
        )
else:
    st.info("GPS columns not found; mobility map is unavailable.")
