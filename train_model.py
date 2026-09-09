"""
Training script for Job Failure Prediction model.
Data: z/OS SMF batch-job records (df_smf.csv)

IMPORTANT ASSUMPTION
---------------------
The raw SMF extract has NO explicit failure / abend / return-code column.
To make this actionable, we build a *proxy* failure label using a documented,
transparent rule (an SRE-style SLA breach definition), rather than inventing
random noise:

    A run is labelled FAILURE_RISK = 1 if, relative to its own SERV_CLASS peer group:
        - ELAPSED_SEC is in the worst 25% (SLA breach / long-running), OR
        - Total queue delay (JQ_SEC + HQ_SEC) is in the worst 25% (starved/stuck in queue), OR
        - EXCP (I/O op count) is in the worst 10% while CPU_SEC is near 0
          (classic "looping / spinning without doing real work" abend pattern)

This gives a defensible, explainable proxy for "this run behaved like jobs that
historically struggle" which is exactly what ops teams use SMF for in practice.
If/when real completion codes (e.g. COND_CODE / ABEND) become available, replace
`build_label()` with the real column and retrain -- everything else stays the same.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
df = pd.read_csv("df_smf.csv")

# Parse hour-of-day from START_T (format like 8632527 -> HHMMSSss? actually it's
# seconds-like packed field in this extract). We use START_DTSTR instead, which
# is a clean timestamp string: '2018-07-31-23.58.46.000000'
df["START_DTSTR"] = pd.to_datetime(df["START_DTSTR"], format="%Y-%m-%d-%H.%M.%S.%f")
df["START_HOUR"] = df["START_DTSTR"].dt.hour
df["START_DOW"] = df["START_DTSTR"].dt.dayofweek

df["TOTAL_QUEUE_SEC"] = df["JQ_SEC"] + df["HQ_SEC"]

# ----------------------------------------------------------------------
# 2. Build proxy failure label (see module docstring)
# ----------------------------------------------------------------------
def build_label(data: pd.DataFrame, top_pct: float = 0.20) -> pd.Series:
    """
    Composite, rank-based risk score computed WITHIN each service class
    (so a BATCH_A job is only compared against other BATCH_A jobs, etc.),
    then the worst `top_pct` runs overall are labelled FAILURE_RISK = 1.
    Using ranks (0-1 percentile) instead of raw thresholds keeps the score
    comparable across service classes with very different workloads.
    """
    excp_cpu_ratio = data["EXCP"] / (data["CPU_SEC"] + 1)

    def pct_rank(s):
        return s.groupby(data["SERV_CLASS"]).rank(pct=True)

    score = (
        0.40 * pct_rank(data["ELAPSED_SEC"])
        + 0.40 * pct_rank(data["TOTAL_QUEUE_SEC"])
        + 0.20 * pct_rank(excp_cpu_ratio)
    )
    threshold = score.quantile(1 - top_pct)
    return (score >= threshold).astype(int), score

df["FAILURE_RISK"], df["RISK_SCORE"] = build_label(df)
print("Label distribution:\n", df["FAILURE_RISK"].value_counts(normalize=True))

# ----------------------------------------------------------------------
# 3. Feature engineering
# ----------------------------------------------------------------------
FEATURES_NUMERIC = [
    "EXCP", "IO_CONN_SEC", "PAGE_IN", "PAGE_OUT", "PAGE_SWAP",
    "SSCH", "TCB_CPU_SEC", "SRB_CPU_SEC", "SERV_UNIT", "MSO_UNIT",
    "TOTAL_QUEUE_SEC", "CPU_SEC", "START_HOUR", "START_DOW", "CLASS",
]
FEATURES_CATEGORICAL = ["SERV_CLASS"]

le_servclass = LabelEncoder()
df["SERV_CLASS_ENC"] = le_servclass.fit_transform(df["SERV_CLASS"])

FEATURE_COLS = FEATURES_NUMERIC + ["SERV_CLASS_ENC"]

X = df[FEATURE_COLS]
y = df["FAILURE_RISK"]

# ----------------------------------------------------------------------
# 4. Train / test split + model
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, pred))
try:
    print("ROC AUC:", roc_auc_score(y_test, proba))
except Exception as e:
    print("ROC AUC could not be computed:", e)

# ----------------------------------------------------------------------
# 5. Build a per-job historical profile (used by the app to prefill inputs
#    when the user types an existing job name)
# ----------------------------------------------------------------------
job_profile = (
    df.groupby("JOB_NAME")[FEATURES_NUMERIC + ["ELAPSED_SEC", "FAILURE_RISK"]]
    .mean()
    .reset_index()
)
job_profile["SERV_CLASS"] = df.groupby("JOB_NAME")["SERV_CLASS"].agg(
    lambda s: s.mode().iloc[0]
).values
job_profile["RUN_COUNT"] = df.groupby("JOB_NAME").size().values

# Global medians, used as sensible defaults for a brand-new / unseen job name
global_defaults = df[FEATURES_NUMERIC].median().to_dict()

# ----------------------------------------------------------------------
# 6. Persist everything the Streamlit app needs into ONE pickle
# ----------------------------------------------------------------------
artifact = {
    "model": model,
    "label_encoder_servclass": le_servclass,
    "feature_cols": FEATURE_COLS,
    "numeric_features": FEATURES_NUMERIC,
    "job_profile": job_profile,
    "global_defaults": global_defaults,
    "serv_class_options": sorted(df["SERV_CLASS"].unique().tolist()),
    "class_options": sorted(df["CLASS"].unique().tolist()),
    "job_name_options": sorted(df["JOB_NAME"].unique().tolist()),
    "training_metrics": {
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else None,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_positive_rate": float(y.mean()),
    },
    "raw_df": df,  # kept small (207 rows) so the dashboard can show real charts
}

joblib.dump(artifact, "job_failure_model.pkl")
print("\nSaved job_failure_model.pkl")
print("Feature importances:")
for f, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f}: {imp:.3f}")
