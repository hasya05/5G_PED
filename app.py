
import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="5G PED | Handover Intelligence",
    page_icon="📡",
    layout="wide",
)

st.title("📡 5G PED — Handover Intelligence")
st.caption(
    "Prediction · Explanation · Decision | "
    "Data-driven radio handover analysis"
)

st.info(
    "This dashboard uses a transparent rule-based handover policy. "
    "It is a decision-support prototype, not a live cellular-network "
    "controller or a trained machine-learning model."
)


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def read_table(file_bytes, filename):
    """Read CSV, TSV, whitespace-separated, or Excel data."""
    name = filename.lower()

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))

    if name.endswith((".tsv", ".tab")):
        return pd.read_csv(io.BytesIO(file_bytes), sep="\t")

    # Fixed: engine='python' does not support low_memory.
    try:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=None,
            engine="python",
        )
    except Exception:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=r"\s+",
            engine="python",
        )

    # Handle a tab-separated file that was read as one column.
    if df.shape[1] == 1:
        first_value = str(df.iloc[0, 0]) if len(df) else ""
        if "\t" in first_value:
            df = pd.read_csv(io.BytesIO(file_bytes), sep="\t")

    return df


def load_uploaded_file(uploaded_file):
    """Load a data file or the first supported table in a ZIP."""
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            candidates = [
                item for item in archive.namelist()
                if not item.startswith("__MACOSX/")
                and not item.split("/")[-1].startswith(".")
                and item.lower().endswith(
                    (".csv", ".tsv", ".tab", ".txt", ".xlsx", ".xls")
                )
            ]

            if not candidates:
                raise ValueError(
                    "The ZIP contains no CSV, TSV, TXT, or Excel data file."
                )

            # Prefer CSV/TSV/TXT before Excel.
            candidates.sort(
                key=lambda item: (
                    not item.lower().endswith((".csv", ".tsv", ".tab", ".txt")),
                    item.count("/"),
                )
            )

            selected_name = candidates[0]
            with archive.open(selected_name) as file:
                return read_table(file.read(), selected_name), selected_name

    return read_table(file_bytes, uploaded_file.name), uploaded_file.name


# ---------------------------------------------------------
# COLUMN MAPPING
# ---------------------------------------------------------

def find_column(columns, keywords):
    """Guess a column from its name; user can change the selection."""
    for column in columns:
        name = str(column).lower().replace("_", " ").replace("-", " ")
        if any(keyword in name for keyword in keywords):
            return column
    return columns[0] if columns else None


def numeric_series(df, column):
    if column is None or column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(
        df[column].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


def safe_text(value):
    if pd.isna(value):
        return "Unknown"
    return str(value)


# ---------------------------------------------------------
# DECISION ENGINE
# ---------------------------------------------------------

def evaluate_handover(
    serving_rsrp,
    target_rsrp,
    serving_sinr,
    target_sinr,
    min_target_rsrp,
    min_target_sinr,
    hysteresis,
    weak_serving_rsrp,
):
    """
    Transparent screening policy using radio measurements.

    RSRP is typically expressed in dBm.
    SINR is typically expressed in dB.
    Confirm the units and meanings in your dataset documentation.
    """
    required = [serving_rsrp, target_rsrp]

    if any(pd.isna(value) for value in required):
        return {
            "decision": "INSUFFICIENT DATA",
            "reason": "Serving or target RSRP is missing.",
            "advantage": np.nan,
            "target_suitable": False,
        }

    advantage = target_rsrp - serving_rsrp

    target_rsrp_ok = target_rsrp >= min_target_rsrp
    target_sinr_known = not pd.isna(target_sinr)
    target_sinr_ok = (
        target_sinr >= min_target_sinr if target_sinr_known else None
    )

    if not target_rsrp_ok:
        return {
            "decision": "STAY — TARGET NOT SUITABLE",
            "reason": (
                f"Target RSRP is {target_rsrp:.1f} dBm, below the "
                f"configured minimum of {min_target_rsrp:.1f} dBm."
            ),
            "advantage": advantage,
            "target_suitable": False,
        }

    if target_sinr_known and not target_sinr_ok:
        return {
            "decision": "STAY — TARGET NOT SUITABLE",
            "reason": (
                f"Target SINR is {target_sinr:.1f} dB, below the "
                f"configured minimum of {min_target_sinr:.1f} dB."
            ),
            "advantage": advantage,
            "target_suitable": False,
        }

    if advantage >= hysteresis:
        reason = (
            f"Target RSRP is {advantage:.1f} dB stronger than serving "
            f"RSRP, meeting the {hysteresis:.1f} dB hysteresis threshold."
        )
        if not target_sinr_known:
            reason += " Target SINR is unavailable, so this check was skipped."

        return {
            "decision": "HANDOVER CANDIDATE",
            "reason": reason,
            "advantage": advantage,
            "target_suitable": True,
        }

    if serving_rsrp <= weak_serving_rsrp and advantage >= 0:
        reason = (
            f"Serving RSRP is weak ({serving_rsrp:.1f} dBm), and the "
            f"target is at least as strong ({target_rsrp:.1f} dBm)."
        )
        if not target_sinr_known:
            reason += " Target SINR is unavailable, so this check was skipped."

        return {
            "decision": "HANDOVER CANDIDATE",
            "reason": reason,
            "advantage": advantage,
            "target_suitable": True,
        }

    return {
        "decision": "STAY CONNECTED",
        "reason": (
            f"Target passes the configured suitability checks, but its "
            f"RSRP advantage ({advantage:.1f} dB) does not meet the "
            f"{hysteresis:.1f} dB hysteresis threshold."
        ),
        "advantage": advantage,
        "target_suitable": True,
    }


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Decision policy")

    min_target_rsrp = st.slider(
        "Minimum target RSRP (dBm)",
        min_value=-140,
        max_value=-60,
        value=-110,
        step=1,
    )

    min_target_sinr = st.slider(
        "Minimum target SINR (dB)",
        min_value=-10,
        max_value=30,
        value=0,
        step=1,
    )

    hysteresis = st.slider(
        "Handover hysteresis (dB)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
    )

    weak_serving_rsrp = st.slider(
        "Weak serving threshold (dBm)",
        min_value=-140,
        max_value=-70,
        value=-115,
        step=1,
    )

    st.divider()
    st.caption(
        "These thresholds are configurable prototype settings, "
        "not universal 3GPP requirements."
    )


# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your radio measurement dataset",
    type=["csv", "tsv", "tab", "txt", "xlsx", "xls", "zip"],
    help="Upload a table or a ZIP containing one or more data tables.",
)

if uploaded_file is None:
    st.subheader("Get started")
    st.markdown(
        """
        1. Upload a dataset containing serving-cell and target-cell radio measurements.
        2. Map the dataset's columns to the required fields.
        3. Inspect a row to see the handover decision and its explanation.
        4. Explore the measurements and download the scored results.
        """
    )

    st.warning(
        "The app needs serving-cell RSRP and target-cell RSRP for a "
        "meaningful decision. If your dataset contains only one cell's "
        "measurements, it cannot directly compare serving and target cells."
    )
    st.stop()


# ---------------------------------------------------------
# READ DATASET
# ---------------------------------------------------------

try:
    raw_df, loaded_filename = load_uploaded_file(uploaded_file)
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

if raw_df.empty:
    st.error("The uploaded file contains no data rows.")
    st.stop()

raw_df.columns = [
    str(column).strip() for column in raw_df.columns
]
raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]

st.success(
    f"Loaded **{loaded_filename}** — "
    f"{raw_df.shape[0]:,} rows × {raw_df.shape[1]:,} columns."
)

with st.expander("Preview original data"):
    st.dataframe(raw_df.head(20), use_container_width=True)

columns = list(raw_df.columns)

if len(columns) < 1:
    st.error("No columns were detected in this file.")
    st.stop()


# ---------------------------------------------------------
# MAP DATASET COLUMNS
# ---------------------------------------------------------

st.header("1. Map your dataset columns")
st.caption(
    "Automatic suggestions are only guesses. Verify every mapping "
    "against your dataset's documentation."
)

with st.expander("Column mapping", expanded=True):
    col1, col2 = st.columns(2)

    serving_rsrp_col = col1.selectbox(
        "Serving-cell RSRP",
        columns,
        index=columns.index(
            find_column(columns, ["serving rsrp", "serving_rsrp", "source rsrp"])
        ),
        key="serving_rsrp",
    )

    target_rsrp_col = col2.selectbox(
        "Target-cell RSRP",
        columns,
        index=columns.index(
            find_column(columns, ["target rsrp", "neighbor rsrp", "neighbour rsrp"])
        ),
        key="target_rsrp",
    )

    col3, col4 = st.columns(2)

    serving_sinr_options = ["Not available"] + columns
    target_sinr_options = ["Not available"] + columns

    serving_sinr_guess = find_column(
        columns, ["serving sinr", "serving_sinr"]
    )
    target_sinr_guess = find_column(
        columns, ["target sinr", "neighbor sinr", "neighbour sinr"]
    )

    serving_sinr_col = col3.selectbox(
        "Serving-cell SINR (optional)",
        serving_sinr_options,
        index=(
            serving_sinr_options.index(serving_sinr_guess) + 1
            if serving_sinr_guess in columns
            and any(k in str(serving_sinr_guess).lower()
                    for k in ["serving", "source"])
            else 0
        ),
        key="serving_sinr",
    )

    target_sinr_col = col4.selectbox(
        "Target-cell SINR (optional)",
        target_sinr_options,
        index=(
            target_sinr_options.index(target_sinr_guess) + 1
            if target_sinr_guess in columns
            and any(k in str(target_sinr_guess).lower()
                    for k in ["target", "neighbor", "neighbour"])
            else 0
        ),
        key="target_sinr",
    )

    col5, col6 = st.columns(2)

    ue_options = ["Not available"] + columns
    timestamp_options = ["Not available"] + columns

    ue_guess = find_column(columns, ["ue id", "user id", "device id", "subscriber"])
    time_guess = find_column(columns, ["timestamp", "datetime", "date time", "time"])

    ue_col = col5.selectbox(
        "UE / device identifier (optional)",
        ue_options,
        index=ue_options.index(ue_guess) + 1 if ue_guess in columns else 0,
        key="ue_id",
    )

    timestamp_col = col6.selectbox(
        "Timestamp (optional)",
        timestamp_options,
        index=(
            timestamp_options.index(time_guess) + 1
            if time_guess in columns
            else 0
        ),
        key="timestamp",
    )

    col7, col8 = st.columns(2)

    speed_options = ["Not available"] + columns
    throughput_options = ["Not available"] + columns

    speed_guess = find_column(columns, ["speed", "velocity", "kmh"])
    throughput_guess = find_column(columns, ["throughput", "data rate"])

    speed_col = col7.selectbox(
        "Speed (optional)",
        speed_options,
        index=speed_options.index(speed_guess) + 1 if speed_guess in columns else 0,
        key="speed",
    )

    throughput_col = col8.selectbox(
        "Throughput (optional)",
        throughput_options,
        index=(
            throughput_options.index(throughput_guess) + 1
            if throughput_guess in columns
            else 0
        ),
        key="throughput",
    )


# ---------------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------------

df = raw_df.copy()

df["_serving_rsrp"] = numeric_series(df, serving_rsrp_col)
df["_target_rsrp"] = numeric_series(df, target_rsrp_col)

if serving_sinr_col != "Not available":
    df["_serving_sinr"] = numeric_series(df, serving_sinr_col)
else:
    df["_serving_sinr"] = np.nan

if target_sinr_col != "Not available":
    df["_target_sinr"] = numeric_series(df, target_sinr_col)
else:
    df["_target_sinr"] = np.nan

if ue_col != "Not available":
    df["_ue_id"] = df[ue_col].astype(str)
else:
    df["_ue_id"] = "All devices"

if timestamp_col != "Not available":
    df["_timestamp"] = pd.to_datetime(
        df[timestamp_col], errors="coerce"
    )
else:
    df["_timestamp"] = pd.NaT

if speed_col != "Not available":
    df["_speed"] = numeric_series(df, speed_col)
else:
    df["_speed"] = np.nan

if throughput_col != "Not available":
    df["_throughput"] = numeric_series(df, throughput_col)
else:
    df["_throughput"] = np.nan

df["_source_row"] = np.arange(len(df))

valid_radio_rows = df[
    df["_serving_rsrp"].notna() & df["_target_rsrp"].notna()
].copy()

if valid_radio_rows.empty:
    st.error(
        "No rows contain numeric values in both selected RSRP columns. "
        "Check your column mappings and confirm that the values are in dBm."
    )
    st.stop()


# ---------------------------------------------------------
# FILTERS
# ---------------------------------------------------------

st.header("2. Filter observations")

filtered = valid_radio_rows.copy()

filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    ue_values = sorted(filtered["_ue_id"].dropna().astype(str).unique())
    selected_ues = st.multiselect(
        "Filter by UE / device",
        options=ue_values,
        default=ue_values[: min(10, len(ue_values))],
    )

if ue_col != "Not available":
    filtered = filtered[filtered["_ue_id"].astype(str).isin(selected_ues)]

if timestamp_col != "Not available":
    valid_times = filtered["_timestamp"].dropna()

    if not valid_times.empty:
        with filter_col2:
            date_range = st.date_input(
                "Filter by date",
                value=(valid_times.min().date(), valid_times.max().date()),
            )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered = filtered[
                filtered["_timestamp"].isna()
                | (
                    (filtered["_timestamp"].dt.date >= start_date)
                    & (filtered["_timestamp"].dt.date <= end_date)
                )
            ]

if filtered.empty:
    st.warning("No observations match the selected filters.")
    st.stop()


# ---------------------------------------------------------
# SCORE ALL FILTERED ROWS
# ---------------------------------------------------------

results = []

for _, row in filtered.iterrows():
    outcome = evaluate_handover(
        serving_rsrp=row["_serving_rsrp"],
        target_rsrp=row["_target_rsrp"],
        serving_sinr=row["_serving_sinr"],
        target_sinr=row["_target_sinr"],
        min_target_rsrp=min_target_rsrp,
        min_target_sinr=min_target_sinr,
        hysteresis=hysteresis,
        weak_serving_rsrp=weak_serving_rsrp,
    )

    results.append({
        "source_row": int(row["_source_row"]),
        "ue_id": row["_ue_id"],
        "timestamp": row["_timestamp"],
        "serving_rsrp_dbm": row["_serving_rsrp"],
        "target_rsrp_dbm": row["_target_rsrp"],
        "serving_sinr_db": row["_serving_sinr"],
        "target_sinr_db": row["_target_sinr"],
        "target_advantage_db": outcome["advantage"],
        "decision": outcome["decision"],
        "reason": outcome["reason"],
        "target_suitable": outcome["target_suitable"],
        "speed": row["_speed"],
        "throughput": row["_throughput"],
    })

scored = pd.DataFrame(results)


# ---------------------------------------------------------
# DASHBOARD METRICS
# ---------------------------------------------------------

st.header("3. Handover overview")

candidate_count = int(
    (scored["decision"] == "HANDOVER CANDIDATE").sum()
)
stay_count = int(
    (scored["decision"] == "STAY CONNECTED").sum()
)
unsuitable_count = int(
    (scored["decision"] == "STAY — TARGET NOT SUITABLE").sum()
)
insufficient_count = int(
    (scored["decision"] == "INSUFFICIENT DATA").sum()
)

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Observations analysed", f"{len(scored):,}")
metric2.metric("Handover candidates", f"{candidate_count:,}")
metric3.metric("Stay connected", f"{stay_count:,}")
metric4.metric("Target unsuitable", f"{unsuitable_count:,}")

decision_counts = (
    scored["decision"].value_counts().rename_axis("Decision").reset_index(name="Count")
)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Decision distribution")
    fig = px.pie(
        decision_counts,
        names="Decision",
        values="Count",
        hole=0.45,
        title="Policy decisions",
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.subheader("Serving vs target signal")
    scatter = px.scatter(
        scored,
        x="serving_rsrp_dbm",
        y="target_rsrp_dbm",
        color="decision",
        hover_data=["source_row", "ue_id", "target_advantage_db"],
        labels={
            "serving_rsrp_dbm": "Serving RSRP (dBm)",
            "target_rsrp_dbm": "Target RSRP (dBm)",
            "source_row": "Source row",
            "ue_id": "UE / device",
        },
        title="Signal comparison by decision",
    )
    st.plotly_chart(scatter, use_container_width=True)

st.subheader("Target signal advantage")
adv_fig = px.histogram(
    scored.dropna(subset=["target_advantage_db"]),
    x="target_advantage_db",
    color="decision",
    nbins=35,
    labels={
        "target_advantage_db": "Target RSRP − serving RSRP (dB)"
    },
)
adv_fig.add_vline(
    x=hysteresis,
    line_dash="dash",
    annotation_text=f"Hysteresis = {hysteresis:.1f} dB",
)
st.plotly_chart(adv_fig, use_container_width=True)


# ---------------------------------------------------------
# OBSERVATION INSPECTOR
# ---------------------------------------------------------

st.header("4. Inspect a handover decision")

st.caption(
    "Select an observation to see the decision and the reason "
    "the configured policy produced it."
)

selected_source_row = st.selectbox(
    "Select source row",
    options=scored["source_row"].tolist(),
    format_func=lambda value: (
        f"Row {value + 1} · "
        f"{scored.loc[scored['source_row'] == value, 'decision'].iloc[0]}"
    ),
)

selected = scored[scored["source_row"] == selected_source_row].iloc[0]

st.subheader(selected["decision"])

st.write(f"**Explanation:** {selected['reason']}")

detail1, detail2, detail3 = st.columns(3)

detail1.metric(
    "Serving RSRP",
    f"{selected['serving_rsrp_dbm']:.1f} dBm",
)
detail2.metric(
    "Target RSRP",
    f"{selected['target_rsrp_dbm']:.1f} dBm",
)
if pd.notna(selected["target_advantage_db"]):
    detail3.metric(
        "Target advantage",
        f"{selected['target_advantage_db']:.1f} dB",
    )
else:
    detail3.metric("Target advantage", "N/A")

extra1, extra2, extra3 = st.columns(3)

extra1.metric(
    "Serving SINR",
    (
        f"{selected['serving_sinr_db']:.1f} dB"
        if pd.notna(selected["serving_sinr_db"])
        else "Not available"
    ),
)

extra2.metric(
    "Target SINR",
    (
        f"{selected['target_sinr_db']:.1f} dB"
        if pd.notna(selected["target_sinr_db"])
        else "Not available"
    ),
)

extra3.metric("UE / device", safe_text(selected["ue_id"]))

st.markdown("**Decision logic used**")

st.markdown(
    f"""
    - Target RSRP minimum: **{min_target_rsrp} dBm**
    - Target SINR minimum: **{min_target_sinr} dB**, when target SINR is available
    - Hysteresis: **{hysteresis:.1f} dB**
    - Weak serving-cell threshold: **{weak_serving_rsrp} dBm**
    """
)

with st.expander("Show selected source row"):
    original = raw_df.iloc[int(selected_source_row)]
    st.dataframe(
        original.to_frame(name="Value"),
        use_container_width=True,
    )


# ---------------------------------------------------------
# SIGNAL TREND
# ---------------------------------------------------------

st.header("5. Signal trend")

if timestamp_col != "Not available":
    trend = scored.dropna(subset=["timestamp"]).sort_values("timestamp")

    if not trend.empty:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=trend["timestamp"],
            y=trend["serving_rsrp_dbm"],
            mode="lines+markers",
            name="Serving RSRP",
        ))

        fig.add_trace(go.Scatter(
            x=trend["timestamp"],
            y=trend["target_rsrp_dbm"],
            mode="lines+markers",
            name="Target RSRP",
        ))

        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="RSRP (dBm)",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No valid timestamps were found. Check the timestamp column format."
        )
else:
    st.info(
        "Map a timestamp column to enable time-series analysis."
    )


# ---------------------------------------------------------
# FULL RESULTS AND EXPORT
# ---------------------------------------------------------

st.header("6. Scored observations")

display_columns = [
    "source_row",
    "ue_id",
    "timestamp",
    "serving_rsrp_dbm",
    "target_rsrp_dbm",
    "serving_sinr_db",
    "target_sinr_db",
    "target_advantage_db",
    "decision",
    "reason",
]

st.dataframe(
    scored[display_columns],
    use_container_width=True,
    hide_index=True,
)

csv_bytes = scored.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download scored decisions (CSV)",
    data=csv_bytes,
    file_name="ped_handover_decisions.csv",
    mime="text/csv",
)

st.caption(
    f"Analysis generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
    "Results depend on the selected columns, measurement units, "
    "data quality, and configured thresholds."
)
