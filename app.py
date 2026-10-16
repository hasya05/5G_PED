import io, zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="5G PED | Handover Intelligence", page_icon="📡", layout="wide")
st.title("📡 5G PED — Handover Intelligence")
st.caption("Prediction · Explanation · Decision | Data-driven mobility analysis")
st.info("Upload real measurement data to analyse handover candidates. This dashboard does not connect to a live carrier network. Recommendations are a configurable research policy, not a guarantee of successful handover.")

ALIASES = {
"serving_rsrp":["serving_rsrp","rsrp_serving","serving_cell_rsrp","ss_rsrp","rsrp"],
"target_rsrp":["target_rsrp","neighbor_rsrp","neighbour_rsrp","target_cell_rsrp","candidate_rsrp","best_neighbor_rsrp"],
"serving_rsrq":["serving_rsrq","rsrq_serving","serving_cell_rsrq","rsrq"],
"target_rsrq":["target_rsrq","neighbor_rsrq","neighbour_rsrq","target_cell_rsrq","candidate_rsrq"],
"serving_sinr":["serving_sinr","sinr_serving","serving_cell_sinr","sinr","cinr"],
"target_sinr":["target_sinr","neighbor_sinr","neighbour_sinr","target_cell_sinr","candidate_sinr"],
"timestamp":["timestamp","time","datetime","date_time","time_stamp"],
"ue":["ue_id","imsi","subscriber_id","user_id","vehicle_id","ue"],
"serving_cell":["serving_cell","serving_cell_id","serving_cgi","source_cell","source_cell_id","current_cell","cell_id","cid","cgi"],
"target_cell":["target_cell","target_cell_id","neighbor_cell","neighbour_cell","candidate_cell","target_cgi"],
"speed":["speed","speed_kmh","velocity","ue_speed","velocity_kmh","speed_ms"],
"throughput":["throughput","throughput_mbps","dl_throughput","data_rate","rlc_throughput"],
"latency":["latency","latency_ms","delay","rlc_delay","packet_delay"],
"load":["cell_load","cell_load_percent","load","prb_utilization","prb_utilisation"],
"label":["handover_trigger","handover_label","handover","ho_trigger","is_handover","upcoming_handover","migration","handover_event"]
}
def norm(s): return "".join(x.lower() for x in str(s).strip() if x.isalnum() or x=="_")
def auto_col(cols, key):
    bynorm={norm(c):c for c in cols}
    for a in ALIASES[key]:
        if norm(a) in bynorm: return bynorm[norm(a)]
    return "(not available)"
def read_file(f):
    b=f.getvalue()
    if f.name.lower().endswith(".zip"):
        z=zipfile.ZipFile(io.BytesIO(b))
        candidates=[n for n in z.namelist() if n.lower().endswith((".csv",".txt",".tsv")) and not n.endswith("/")]
        if not candidates: raise ValueError("ZIP has no CSV/TXT/TSV files.")
        choice=st.sidebar.selectbox("File inside ZIP", candidates)
        b=z.read(choice); name=choice
    else: name=f.name
    try: d=pd.read_csv(io.BytesIO(b), sep=None, engine="python", low_memory=False)
    except Exception: d=pd.read_csv(io.BytesIO(b), sep=r"\s+", engine="python", low_memory=False)
    if d.shape[1]==1 and "\t" in str(d.iloc[0,0]): d=pd.read_csv(io.BytesIO(b), sep="\t", low_memory=False)
    d.columns=[str(c).strip() for c in d.columns]
    return d,name
def num(s): return pd.to_numeric(s, errors="coerce")
def value(row, col):
    try:
        v=float(row[col])
        return v if np.isfinite(v) else None
    except Exception: return None

with st.sidebar:
    st.header("Load data")
    upload=st.file_uploader("Upload CSV, TXT, TSV or ZIP", type=["csv","txt","tsv","zip"])
    st.divider()
    st.header("Decision policy")
    hysteresis=st.slider("Target RSRP advantage (dB)",0.0,10.0,3.0,0.5)
    weak=st.slider("Serving weak-signal threshold (dBm)",-120,-70,-105)
    target_min=st.slider("Minimum target RSRP (dBm)",-120,-70,-110)
    sinr_min=st.slider("Minimum target SINR (dB)",-10,25,0)
    use_sinr=st.checkbox("Apply SINR suitability gate when mapped",True)
if not upload:
    st.subheader("Start with a real dataset")
    st.markdown("""
**Recommended: [Mobility Dataset from a 7.2 O-RAN deployment](https://data.mendeley.com/datasets/khxgr6m8wz)**  
A real four-base-station deployment with RSRP, RSRQ, SINR and around 2,300 handover events.

**Alternative: [Drive-Test-Based LTE Handover Dataset](https://data.mendeley.com/datasets/n2pvmtyn2j/2)**  
Real-world LTE drive-test data; note that it is LTE, not 5G NR.

Download a CSV/TXT data file, upload it here, and confirm the column mapping. Review each dataset's data dictionary and license before publishing results.
""")
    st.markdown("### What the decision requires")
    st.write("- Serving and target/neighbor RSRP for the same observation.")
    st.write("- RSRQ/SINR, timestamps and UE identifiers where available.")
    st.write("- Historical handover labels for evaluation; an event label alone does not prove a handover was successful or optimal.")
    st.stop()
try: df, source=read_file(upload)
except Exception as e: st.error(f"Could not read file: {e}"); st.stop()
if df.empty or len(df.columns)<2: st.error("File is empty or has fewer than two columns."); st.stop()
st.success(f"Loaded **{source}** · {len(df):,} rows · {len(df.columns)} columns")
with st.expander("Inspect raw data"): st.dataframe(df.head(30),use_container_width=True)
cols=list(df.columns); options=["(not available)"]+cols
st.subheader("Confirm dataset column mapping")
st.caption("Automatic matching is only a suggestion. Confirm each selection against the dataset documentation.")
mapping={}
layout=st.columns(3)
fields=[
("serving_rsrp","Serving-cell RSRP (dBm)"),("target_rsrp","Target/neighbor RSRP (dBm)"),
("serving_rsrq","Serving-cell RSRQ (dB)"),("target_rsrq","Target/neighbor RSRQ (dB)"),
("serving_sinr","Serving-cell SINR (dB)"),("target_sinr","Target/neighbor SINR (dB)"),
("serving_cell","Serving-cell ID"),("target_cell","Target/neighbor cell ID"),
("timestamp","Timestamp"),("ue","UE/device ID"),("speed","UE speed"),
("throughput","Throughput"),("latency","Latency"),("load","Cell load"),("label","Historical handover label/event")]
for i,(key,label) in enumerate(fields):
    default=auto_col(cols,key); ix=options.index(default) if default in options else 0
    mapping[key]=layout[i%3].selectbox(label,options,index=ix,key="map_"+key)
def series(key):
    c=mapping[key]
    return num(df[c]) if c!="(not available)" else pd.Series(np.nan,index=df.index)
work=df.copy()
for k in ["serving_rsrp","target_rsrp","serving_rsrq","target_rsrq","serving_sinr","target_sinr","speed","throughput","latency","load"]:
    work["_"+k]=series(k)
work["_advantage"]=work["_target_rsrp"]-work["_serving_rsrp"]
work["_decision"]="INSUFFICIENT DATA"
valid=work["_serving_rsrp"].notna() & work["_target_rsrp"].notna()
for ix,row in work.loc[valid].iterrows():
    adv=row["_advantage"]; s=row["_serving_rsrp"]; t=row["_target_rsrp"]
    target_ok=t>=target_min
    why_sinr=use_sinr and mapping["target_sinr"]!="(not available)" and pd.notna(row["_target_sinr"]) and row["_target_sinr"]<sinr_min
    if not target_ok or why_sinr: decision="STAY — TARGET NOT SUITABLE"
    elif adv>=hysteresis or (s<=weak and adv>=0): decision="HANDOVER CANDIDATE"
    else: decision="STAY CONNECTED"
    work.at[ix,"_decision"]=decision

st.subheader("Explore observations")
if mapping["ue"]!="(not available)":
    ues=sorted(df[mapping["ue"]].dropna().astype(str).unique().tolist())[:5000]
    selected=st.selectbox("UE/device",["All UEs"]+ues)
    if selected!="All UEs": work=work[df.loc[work.index,mapping["ue"]].astype(str)==selected].copy()
if mapping["timestamp"]!="(not available)":
    work["_time"]=pd.to_datetime(work[mapping["timestamp"]],errors="coerce",utc=True)
    tv=work["_time"].dropna()
    if len(tv)>1 and tv.min()<tv.max():
        bounds=st.slider("Time window (UTC)",tv.min().to_pydatetime(),tv.max().to_pydatetime(),(tv.min().to_pydatetime(),tv.max().to_pydatetime()),format="YYYY-MM-DD HH:mm")
        work=work[(work["_time"]>=pd.Timestamp(bounds[0]))&(work["_time"]<=pd.Timestamp(bounds[1]))].copy()
if work.empty: st.warning("No observations match these filters."); st.stop()
idx=st.slider("Observation row",0,len(work)-1,min(len(work)//2,len(work)-1))
row=work.iloc[idx]
decision=row["_decision"]; s=value(row,"_serving_rsrp"); t=value(row,"_target_rsrp"); adv=value(row,"_advantage")
st.divider(); st.subheader("PED decision report")
if decision=="HANDOVER CANDIDATE":
    st.success("## HANDOVER CANDIDATE")
    st.write("The candidate passes the configured suitability gate and its RSRP advantage meets the selected policy. This is a candidate for handover evaluation, not a command to execute a live handover.")
elif decision=="STAY — TARGET NOT SUITABLE":
    st.error("## STAY CONNECTED — TARGET NOT SUITABLE")
    st.write("The candidate fails a configured target quality gate. A stronger-looking signal alone is not sufficient.")
elif decision=="STAY CONNECTED":
    st.warning("## STAY CONNECTED")
    st.write("The target does not meet the configured RSRP advantage/weak-serving condition, so this policy does not recommend handover for this observation.")
else:
    st.warning("## INSUFFICIENT DATA")
    st.write("Both serving-cell and target-cell RSRP are required. The app will not guess when measurements are missing.")
a,b,c,d=st.columns(4)
a.metric("Serving RSRP",f"{s:.1f} dBm" if s is not None else "Missing")
b.metric("Target RSRP",f"{t:.1f} dBm" if t is not None else "Missing")
c.metric("Target advantage",f"{adv:+.1f} dB" if adv is not None else "Unavailable")
d.metric("Filtered observations",f"{len(work):,}")
st.markdown("### Why?")
reasons=[]
if s is None or t is None: reasons.append("Missing serving or target RSRP; no decision can be made.")
else:
    reasons.append(f"Target RSRP advantage is {adv:+.1f} dB; configured threshold is {hysteresis:.1f} dB.")
    reasons.append(f"Serving RSRP is {s:.1f} dBm; weak-signal threshold is {weak} dBm.")
    reasons.append(f"Target RSRP is {t:.1f} dBm; minimum target threshold is {target_min} dBm.")
    if use_sinr and mapping["target_sinr"]!="(not available)":
        ts=value(row,"_target_sinr")
        if ts is not None: reasons.append(f"Target SINR is {ts:.1f} dB; configured minimum is {sinr_min} dB.")
    if mapping["serving_rsrq"]!="(not available)" and mapping["target_rsrq"]!="(not available)":
        sr=value(row,"_serving_rsrq"); tr=value(row,"_target_rsrq")
        if sr is not None and tr is not None: reasons.append(f"Quality context: serving RSRQ {sr:.1f} dB; target RSRQ {tr:.1f} dB.")
for r in reasons: st.markdown("- "+r)
with st.expander("Policy assumptions and limitations"):
    st.markdown(f"""
- Target RSRP advantage threshold: **{hysteresis:.1f} dB**
- Serving weak-signal threshold: **{weak} dBm**
- Minimum target RSRP: **{target_min} dBm**
- SINR suitability gate: **{'enabled when available' if use_sinr else 'disabled'}**
- These thresholds are configurable demo settings, not universal 3GPP defaults.
- This version uses transparent rules, not a trained ML model. It does not infer success probability from historical event labels.
- A robust temporal controller requires ordered UE/candidate measurements, time-to-trigger, filtering, target-cell load/capacity and labeled outcomes such as successful/failed handovers and ping-pong events.
""")
st.divider(); st.subheader("Interactive graphs")
plot=work.copy(); plot["_observation"]=np.arange(len(plot))
x="_time" if "_time" in plot.columns and plot["_time"].notna().any() else "_observation"
if valid.any():
    melted=plot[["_serving_rsrp","_target_rsrp",x]].melt(id_vars=x,var_name="Measurement",value_name="RSRP (dBm)")
    melted["Measurement"]=melted["Measurement"].map({"_serving_rsrp":"Serving cell","_target_rsrp":"Target/neighbor"})
    fig=px.line(melted,x=x,y="RSRP (dBm)",color="Measurement",title="Serving vs target radio signal")
    fig.update_layout(hovermode="x unified",xaxis_title="Time (UTC)" if x=="_time" else "Observation order")
    st.plotly_chart(fig,use_container_width=True)
    fig=px.histogram(plot,x="_advantage",color="_decision",nbins=40,title="Target-cell RSRP advantage by decision")
    fig.update_layout(xaxis_title="Target RSRP − serving RSRP (dB)")
    st.plotly_chart(fig,use_container_width=True)
else: st.warning("Map both serving and target RSRP to enable decisions and signal comparison.")
metrics=[(k,label) for k,label in [("_serving_rsrq","Serving RSRQ"),("_target_rsrq","Target RSRQ"),("_serving_sinr","Serving SINR"),("_target_sinr","Target SINR"),("_speed","UE speed"),("_throughput","Throughput"),("_latency","Latency"),("_load","Cell load")] if work[k].notna().any()]
if metrics:
    key,label=st.selectbox("Explore another metric",metrics,format_func=lambda x:x[1])
    st.plotly_chart(px.line(plot,x=x,y=key,title=f"{label} over observations"),use_container_width=True)
st.subheader("Decision summary")
counts=work["_decision"].value_counts().rename_axis("Decision").reset_index(name="Observations")
l,r=st.columns(2)
l.dataframe(counts,use_container_width=True,hide_index=True)
r.plotly_chart(px.pie(counts,names="Decision",values="Observations",title="Decision distribution"),use_container_width=True)
if mapping["label"]!="(not available)":
    st.subheader("Dataset-provided historical label")
    st.caption("This is the source dataset's label, not necessarily a successful handover or proof that handover was optimal.")
    st.dataframe(work[[mapping["label"],"_decision"]].head(100),use_container_width=True,hide_index=True)
st.download_button("Download filtered analysis CSV",work.to_csv(index=False).encode("utf-8"),"ped_handover_analysis.csv","text/csv")
st.caption("Research prototype only. Do not use as a live network controller.")
