# Job Failure Prediction Dashboard

Predicts whether a batch job's **next run** is likely to fail / misbehave,
based on z/OS SMF telemetry (`df_smf.csv`).

## ⚠️ Important note on the target variable
The raw SMF extract has **no explicit failure/abend/return-code column**.
`train_model.py` builds a transparent proxy label, `FAILURE_RISK`, from a
composite rank-based score of:
- Elapsed time (SLA breach, within each Service Class)
- Total queue delay (JQ_SEC + HQ_SEC — stuck in queue)
- EXCP/CPU ratio (looping/spinning without doing real work)

The worst ~20% of runs (per this score) are labelled risky. This is
documented in code and in the app itself. **If you get access to real
completion/abend codes, replace `build_label()` in `train_model.py` with
that column and retrain — the rest of the pipeline and the app stay
unchanged.**

## Files
- `train_model.py` — loads `df_smf.csv`, engineers features, builds the
  proxy label, trains a `RandomForestClassifier`, and saves everything
  the app needs into `job_failure_model.pkl`.
- `job_failure_model.pkl` — trained model + encoders + historical job
  profiles + dataset (pickled with `joblib`).
- `app.py` — Streamlit dashboard: enter a Job Name, get a failure-risk
  gauge, feature-contribution chart, historical trend, a job-history
  explorer, and model insights.
- `requirements.txt` — pinned dependencies.
- `df_smf.csv` — the original data (needed only for retraining).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model (e.g. after new data / real failure labels)
```bash
python train_model.py
```
This regenerates `job_failure_model.pkl`.

## Deploy to GitHub + Streamlit Community Cloud
```bash
git init
git add app.py train_model.py job_failure_model.pkl requirements.txt README.md df_smf.csv
git commit -m "Job failure prediction dashboard"
git remote add origin <your-repo-url>
git push -u origin main
```
Then on https://share.streamlit.io: **New app** → select your repo/branch →
main file path `app.py` → Deploy.

## Using the dashboard
1. In the sidebar, pick an **existing job name** (auto-fills its historical
   average telemetry, which you can still tweak) or enter a **new job name**
   (uses dataset-wide medians as a starting point).
2. Adjust any run parameters (EXCP, CPU_SEC, queue delay, service class, etc.)
3. Click **Predict Failure Risk** to see:
   - A gauge showing failure probability
   - The top features driving that specific prediction
   - The job's historical elapsed-time trend (risky runs marked in red)
4. Use the **Job History Explorer** tab to filter/inspect all historical runs.
5. Use the **Model Insights** tab for global feature importance and the model card.
