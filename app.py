"""
Job Failure Prediction Dashboard
---------------------------------
Streamlit app that loads `job_failure_model.pkl` (trained by train_model.py
on the SMF batch-job extract) and predicts the FAILURE RISK of a job's
next run.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this folder (app.py, job_failure_model.pkl, requirements.txt) to GitHub.
    2. Go to share.streamlit.io -> New app -> point to app.py.
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Job Failure Predictor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Load model artifact
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifact(path: str = "job_failure_model.pkl"):
    return joblib.load(path)


artifact = load_artifact()
model = artifact["model"]
le_servclass = artifact["label_encoder_servclass"]
FEATURE_COLS = artifact["feature_cols"]
NUMERIC_FEATURES = artifact["numeric_features"]
job_profile = artifact["job_profile"].set_index("JOB_NAME")
global_defaults = artifact["global_defaults"]
serv_class_options = artifact["serv_class_options"]
class_options = artifact["class_options"]
job_name_options = artifact["job_name_options"]
metrics = artifact["training_metrics"]
raw_df = artifact["raw_df"]

# ----------------------------------------------------------------------
# Sidebar - custom styling
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #eee;
    }
    .risk-high {color:#d62728; font-weight:700;}
    .risk-low {color:#2ca02c; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚙️ Job Failure Prediction Dashboard")
st.caption(
    "Predicts whether a batch job's **next run** is likely to behave like a "
    "historically risky run (long elapsed time, stuck-in-queue, or "
    "looping/spinning pattern), based on SMF telemetry."
)

with st.expander("ℹ️ How the failure label was defined (read this once)"):
    st.write(
        "The raw SMF extract has no explicit abend/return-code column. "
        "`FAILURE_RISK` is a transparent proxy built from a composite, "
        "rank-based score of **elapsed time**, **queue delay**, and "
        "**EXCP/CPU ratio** (looping signature), computed within each "
        "service class. The worst ~20% of runs are labelled risky. "
        "Swap in real completion codes and retrain when available for a "
        "ground-truth model — the rest of this app stays the same."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Model ROC AUC", f"{metrics['roc_auc']:.2f}" if metrics["roc_auc"] else "n/a")
    c2.metric("Training runs", metrics["n_train"] + metrics["n_test"])
    c3.metric("Historical risk rate", f"{metrics['label_positive_rate']*100:.1f}%")

st.divider()

# ----------------------------------------------------------------------
# Sidebar: Job selection / input
# ----------------------------------------------------------------------
st.sidebar.header("🔧 Predict a Run")

mode = st.sidebar.radio(
    "Job source",
    ["Pick an existing job", "Enter a new / unseen job name"],
    help="Existing jobs are pre-filled with their historical average telemetry, "
         "which you can still adjust before predicting.",
)

if mode == "Pick an existing job":
    job_name = st.sidebar.selectbox("Job Name", job_name_options)
    profile = job_profile.loc[job_name]
    defaults = profile.to_dict()
    serv_class_default = profile["SERV_CLASS"]
    run_count = int(profile["RUN_COUNT"])
    st.sidebar.caption(f"📈 {run_count} historical run(s) found for **{job_name}**.")
else:
    job_name = st.sidebar.text_input("New Job Name", value="JN_NEW")
    defaults = global_defaults
    serv_class_default = serv_class_options[0]
    st.sidebar.caption("No history found — using dataset-wide median defaults. Adjust below.")

st.sidebar.subheader("Run parameters")
serv_class = st.sidebar.selectbox(
    "Service Class", serv_class_options,
    index=serv_class_options.index(serv_class_default) if serv_class_default in serv_class_options else 0,
)
job_class = st.sidebar.selectbox(
    "Class", class_options,
    index=class_options.index(int(defaults.get("CLASS", class_options[0]))) if defaults.get("CLASS") in class_options else 0,
)

def num_input(label, key, help_=None):
    default_val = float(defaults.get(key, global_defaults.get(key, 0)))
    return st.sidebar.number_input(label, min_value=0.0, value=max(default_val, 0.0), help=help_)

excp = num_input("EXCP (I/O op count)", "EXCP")
io_conn_sec = num_input("IO_CONN_SEC", "IO_CONN_SEC")
page_in = num_input("PAGE_IN", "PAGE_IN")
page_out = num_input("PAGE_OUT", "PAGE_OUT")
page_swap = num_input("PAGE_SWAP", "PAGE_SWAP")
ssch = num_input("SSCH (start subchannel count)", "SSCH")
tcb_cpu_sec = num_input("TCB_CPU_SEC", "TCB_CPU_SEC")
srb_cpu_sec = num_input("SRB_CPU_SEC", "SRB_CPU_SEC")
serv_unit = num_input("SERV_UNIT", "SERV_UNIT")
mso_unit = num_input("MSO_UNIT", "MSO_UNIT")
queue_sec = num_input("Total Queue Seconds (JQ+HQ)", "TOTAL_QUEUE_SEC")
cpu_sec = num_input("CPU_SEC", "CPU_SEC")
start_hour = st.sidebar.slider("Planned Start Hour", 0, 23, int(defaults.get("START_HOUR", 0)))
start_dow = st.sidebar.selectbox(
    "Planned Start Day of Week",
    options=list(range(7)),
    format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
    index=int(defaults.get("START_DOW", 0)) if not pd.isna(defaults.get("START_DOW", 0)) else 0,
)

predict_btn = st.sidebar.button("🔮 Predict Failure Risk", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# Build feature row + predict
# ----------------------------------------------------------------------
def build_feature_row():
    row = {
        "EXCP": excp, "IO_CONN_SEC": io_conn_sec, "PAGE_IN": page_in,
        "PAGE_OUT": page_out, "PAGE_SWAP": page_swap, "SSCH": ssch,
        "TCB_CPU_SEC": tcb_cpu_sec, "SRB_CPU_SEC": srb_cpu_sec,
        "SERV_UNIT": serv_unit, "MSO_UNIT": mso_unit,
        "TOTAL_QUEUE_SEC": queue_sec, "CPU_SEC": cpu_sec,
        "START_HOUR": start_hour, "START_DOW": start_dow, "CLASS": job_class,
        "SERV_CLASS_ENC": le_servclass.transform([serv_class])[0],
    }
    return pd.DataFrame([row])[FEATURE_COLS]


tab_predict, tab_history, tab_insights = st.tabs(
    ["🎯 Prediction", "📊 Job History Explorer", "🧠 Model Insights"]
)

with tab_predict:
    if predict_btn:
        X_new = build_feature_row()
        proba = model.predict_proba(X_new)[0, 1]
        pred = int(proba >= 0.5)

        col1, col2 = st.columns([1, 1.4])
        with col1:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number={"suffix": "%"},
                    title={"text": f"Failure Risk — {job_name}"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#d62728" if pred else "#2ca02c"},
                        "steps": [
                            {"range": [0, 40], "color": "#e6f4ea"},
                            {"range": [40, 70], "color": "#fff4ce"},
                            {"range": [70, 100], "color": "#fde2e1"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.8,
                            "value": 50,
                        },
                    },
                )
            )
            gauge.update_layout(height=320, margin=dict(t=60, b=10, l=20, r=20))
            st.plotly_chart(gauge, use_container_width=True)

            if pred:
                st.error(f"⚠️ **High risk** — {job_name}'s next run is predicted to behave like a risky run ({proba*100:.1f}% probability).")
            else:
                st.success(f"✅ **Healthy** — {job_name}'s next run looks normal ({proba*100:.1f}% risk probability).")

        with col2:
            st.subheader("What's driving this prediction?")
            importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
            contrib = (X_new.iloc[0] - pd.Series(global_defaults).reindex(FEATURE_COLS).fillna(0)) * importances
            contrib = contrib.sort_values(key=abs, ascending=True).tail(8)
            fig = px.bar(
                x=contrib.values, y=contrib.index, orientation="h",
                labels={"x": "Relative contribution (vs. dataset median)", "y": ""},
                color=contrib.values, color_continuous_scale=["#2ca02c", "#f4d03f", "#d62728"],
            )
            fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Input snapshot")
        st.dataframe(X_new.rename(columns=lambda c: c.replace("_", " ")), use_container_width=True, hide_index=True)

        if mode == "Pick an existing job" and job_name in raw_df["JOB_NAME"].values:
            st.subheader(f"Recent runs of {job_name}")
            hist = raw_df[raw_df["JOB_NAME"] == job_name][
                ["START_DTSTR", "ELAPSED_SEC", "CPU_SEC", "TOTAL_QUEUE_SEC", "EXCP", "FAILURE_RISK"]
            ].sort_values("START_DTSTR")
            fig2 = px.line(hist, x="START_DTSTR", y="ELAPSED_SEC", markers=True,
                            title="Elapsed time trend", labels={"START_DTSTR": "Run start", "ELAPSED_SEC": "Elapsed (sec)"})
            for _, r in hist[hist["FAILURE_RISK"] == 1].iterrows():
                fig2.add_vline(x=r["START_DTSTR"], line_dash="dot", line_color="red", opacity=0.4)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("👈 Set the job parameters in the sidebar and click **Predict Failure Risk**.")

with tab_history:
    st.subheader("Explore historical SMF runs")
    colf1, colf2 = st.columns(2)
    with colf1:
        sc_filter = st.multiselect("Filter by Service Class", serv_class_options, default=serv_class_options)
    with colf2:
        risk_filter = st.selectbox("Filter by risk label", ["All", "Risky only (1)", "Healthy only (0)"])

    view = raw_df[raw_df["SERV_CLASS"].isin(sc_filter)]
    if risk_filter == "Risky only (1)":
        view = view[view["FAILURE_RISK"] == 1]
    elif risk_filter == "Healthy only (0)":
        view = view[view["FAILURE_RISK"] == 0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs shown", len(view))
    c2.metric("Risky runs", int(view["FAILURE_RISK"].sum()))
    c3.metric("Avg elapsed (sec)", f"{view['ELAPSED_SEC'].mean():.1f}")
    c4.metric("Avg queue delay (sec)", f"{view['TOTAL_QUEUE_SEC'].mean():.0f}")

    fig3 = px.scatter(
        view, x="TOTAL_QUEUE_SEC", y="ELAPSED_SEC", color=view["FAILURE_RISK"].map({0: "Healthy", 1: "Risky"}),
        size="EXCP", hover_data=["JOB_NAME", "SERV_CLASS", "CPU_SEC"],
        color_discrete_map={"Healthy": "#2ca02c", "Risky": "#d62728"},
        title="Elapsed time vs Queue delay (bubble size = EXCP)",
    )
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.bar(
        view.groupby("SERV_CLASS")["FAILURE_RISK"].mean().reset_index(),
        x="SERV_CLASS", y="FAILURE_RISK", color="SERV_CLASS",
        title="Historical risk rate by Service Class", labels={"FAILURE_RISK": "Risk rate"},
    )
    fig4.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(
        view[["JOB_NAME", "SERV_CLASS", "START_DTSTR", "ELAPSED_SEC", "CPU_SEC",
              "TOTAL_QUEUE_SEC", "EXCP", "FAILURE_RISK"]].sort_values("START_DTSTR", ascending=False),
        use_container_width=True, hide_index=True,
    )

with tab_insights:
    st.subheader("What the model learned")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
    fig5 = px.bar(
        x=importances.values, y=importances.index, orientation="h",
        labels={"x": "Importance", "y": "Feature"}, title="Global feature importance",
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Model card")
    st.json(
        {
            "algorithm": "RandomForestClassifier (n_estimators=300, max_depth=6)",
            "target": "FAILURE_RISK (proxy label, see explainer above)",
            "roc_auc": metrics["roc_auc"],
            "n_train": metrics["n_train"],
            "n_test": metrics["n_test"],
            "historical_risk_rate": metrics["label_positive_rate"],
            "features": FEATURE_COLS,
        }
    )

st.divider()
st.caption("Built with Streamlit • RandomForest model trained on SMF batch-job telemetry.")
