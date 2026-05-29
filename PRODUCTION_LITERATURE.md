# VyaAI Production Literature

## Executive Overview

VyaAI is an agentic model intelligence MVP for reviewing synthetic financial-services and purchase-analytics model artifacts. The system does not train or deploy a production model. Instead, it reviews existing artifact-style inputs: feature tables, labels, predictions, model metadata, and feature metadata.

The current implementation includes a QSR purchase propensity profile as the bundled sample workflow. The agent code is metadata-driven and can review other tabular model outputs when the target, entity id, prediction column, and feature columns are declared in metadata. All generated data is synthetic and designed only for local demonstration.

## Production Problem Framing

Business and data science teams often need a fast, auditable read on whether a model is still healthy enough for use. The core question is not only whether a metric moved, but why the movement matters for decision quality, resource allocation, client communication, and model governance.

VyaAI addresses this with a deterministic-evidence-first workflow:

- Python agents calculate all metrics and diagnostics.
- Evidence is saved as JSON, CSV, Markdown, and figures.
- Executive reporting is produced from saved evidence only.
- Any future LLM layer must summarize evidence, not calculate metrics or override risk.

## Current MVP Workflow

### 1. Sample Artifact Generation

File: `src/generate_sample_artifacts.py`

Generated inputs:

- `data/train_features_sample.csv`
- `data/current_features_sample.csv`
- `data/current_predictions_sample.csv`
- `models/model_metadata.json`
- `models/feature_metadata.json`

The generator creates 200 training rows and 200 current rows. Current-period drift is intentionally injected:

- `weekend_dining_frequency` decreases
- `merchant_novelty_rate` increases
- `competitor_qsr_share_90d` increases
- `fuel_spend_30d` increases slightly

These names belong to the bundled demo profile. For another model, the same files can contain different numeric feature names as long as `models/model_metadata.json` and `models/feature_metadata.json` describe the artifact contract.

### 2. Agent 01: Mitra

File: `src/agents/signal_sentinel_agent.py`

Purpose: identify data and signal movement before high-impact business use.

Implemented checks:

- Missing-value shifts
- Feature drift via PSI
- Feature drift via Kolmogorov-Smirnov test
- Distribution movement via Wasserstein distance
- Prediction score summary and prediction-label mix gap
- Cluster shift using `StandardScaler` and `KMeans`

Outputs:

- `reports/signal_sentinel_output.json`
- `reports/drift_report.csv`
- `reports/cluster_shift_report.csv`
- `reports/figures/drift_top_features.png`

### 3. Agent 02: Varuna

File: `src/agents/model_lens_agent.py`

Purpose: explain model behavior and identify model-level risk.

Implemented checks:

- Small local `XGBClassifier` trained on supplied training features
- SHAP global feature importance on current features
- SHAP global bar plot
- SHAP beeswarm plot
- VIF multicollinearity diagnostics
- Train-validation metric overfitting delta from metadata
- High-risk feature matrix combining SHAP rank, drift level, and VIF warning

Outputs:

- `reports/model_lens_output.json`
- `reports/shap_global_importance.csv`
- `reports/vif_report.csv`
- `reports/figures/shap_global_bar.png`
- `reports/figures/shap_beeswarm.png`

### 4. Evidence Store

File: `src/agents/evidence_store.py`

Purpose: consolidate deterministic outputs into a single packet for reporting.

Output:

- `reports/evidence_packet.json`

The evidence packet includes:

- Model metadata
- Feature metadata
- Mitra summary
- Varuna summary
- Key findings derived only from saved JSON values
- Available plots
- Business context
- Limitations

### 5. Agent 03: Aryaman

File: `src/agents/executive_synthesis_agent.py`

Purpose: create a concise consulting-style model health brief from the evidence packet.

The MVP uses deterministic template logic, not an external LLM.

Outputs:

- `reports/executive_model_report.json`
- `reports/executive_model_report.md`

Risk rules:

- High Risk: 3+ high-drift features or high overfitting risk
- Medium Risk: 1-2 high-drift features, medium overfitting risk, or high VIF
- Low Risk: otherwise

## Current Evidence Summary

The latest local run produces a `Medium Risk` executive model health status.

Key evidence from the bundled QSR demo:

- `merchant_novelty_rate` is a high-drift feature and a high combined-risk feature.
- `weekend_dining_frequency` is a high-drift feature and a medium combined-risk feature.
- Cluster mix shifts by up to 9.5 percentage points.
- Prediction-positive rate differs from actual-positive rate by -0.135.
- Train-validation metric delta is 0.036, indicating medium overfitting risk.
- Top SHAP driver is a demo-profile purchase behavior feature.

## Auditability

Every business-facing statement in the executive report traces back to deterministic artifacts:

- Drift findings come from `reports/signal_sentinel_output.json` and `reports/drift_report.csv`.
- SHAP and VIF findings come from `reports/model_lens_output.json`, `reports/shap_global_importance.csv`, and `reports/vif_report.csv`.
- Executive report content comes from `reports/evidence_packet.json`.

## Production Guardrails

- Synthetic sample data only.
- No real customer or financial data.
- No secrets or production extracts.
- LLMs must not calculate metrics.
- LLMs must not override deterministic risk levels.
- Metrics are simulated for MVP demonstration.
- Executive narratives must cite or derive from saved evidence.

## Local Execution

Run the full MVP:

```bash
python src/run_vyaai_pipeline.py
```

Run checks:

```bash
python -m unittest discover -s tests
python -m compileall -q src app tests
```

Launch dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Remaining Production Work

- Add stronger data contract validation with explicit schemas.
- Add train prediction baseline for direct prediction drift comparisons.
- Add richer segment naming and segment-level calibration checks.
- Add CI execution for tests and compile checks.
- Add optional LLM provider only after deterministic report quality is stable.
- Add versioned evidence packets for multi-run comparison.
