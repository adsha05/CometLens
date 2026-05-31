# AxionAI Demo Walkthrough

This walkthrough presents AxionAI as a portfolio-grade local MVP using synthetic data only.

## 1. Run The Pipeline

```bash
pip install -r requirements.txt
python src/run_axionai_pipeline.py
```

The orchestrator:

1. Generates synthetic model-review artifacts.
2. Validates feature-table and metadata contracts.
3. Runs Mitra monitoring checks.
4. Runs Varuna model diagnostics.
5. Builds the verified evidence packet.
6. Runs Vishwakarma visual generation.
7. Refreshes the packet with matching-run visual references.
8. Runs Aryaman executive synthesis.
9. Runs Samanvaya feedback calibration review.
10. Archives the run under `reports/runs/<run_id>/`.

## 2. Launch The Dashboard

```bash
streamlit run app/streamlit_app.py
```

## 3. Demo Screen Order

### Overview

Explain that the workflow reviews artifacts rather than deploying a production model. The bundled QSR profile is synthetic and replaceable through metadata.

### Mitra: Signal Monitoring

Show:

- Overall signal risk
- High-drift features
- Data-quality gate
- Prediction drift
- Cluster/context findings

The bundled demo highlights `merchant_novelty_rate` and `weekend_dining_frequency`.

### Varuna: Model Diagnostics

Show:

- SHAP global drivers
- Feature-risk matrix
- VIF findings
- Overfitting delta
- Reliability warning when severe Mitra drift is present

The key demonstration is the combination of feature importance and drift, not SHAP in isolation.

### Evidence Store

Open the expandable JSON viewer. Explain that downstream reporting consumes a verified packet instead of recalculating metrics.

### Vishwakarma: Visual Intelligence

Show:

- Feature-risk scatter
- Prediction distribution overlay
- Run-specific lineage SVG

Vishwakarma is read-only for upstream metrics.

### Aryaman: Executive Synthesis

Show the executive summary and Markdown report. Emphasize concise business language and evidence traceability.

### Samanvaya: Feedback Calibration

Submit a structured feedback event or use the bundled synthetic demo feedback. Click **Run Samanvaya Calibration Review**.

Show:

- Feedback table
- Feedback-type counts
- Pending recommendations
- Downloadable `calibration_config_v2_recommended.json`

Explain that the active v1 config remains unchanged until human approval.

## 4. Verify The Project

```bash
pytest tests/
python -m unittest discover -s tests
python -m compileall -q src app tests scripts
```

Current validated result:

- `39` pytest tests passed
- `11` unittest checks passed
- Streamlit smoke test passed
- Python compile check passed

## 5. Regenerate README Assets

```bash
python scripts/render_readme_assets.py
```

This refreshes screenshot-style proof assets from deterministic report files.
