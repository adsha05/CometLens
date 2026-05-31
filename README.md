# AxionAI

**Agentic model intelligence for financial-services ML teams.**

AxionAI reviews model artifacts, detects model-health risks, explains model behavior, and produces an executive-ready model health brief from deterministic evidence.

The bundled demo uses synthetic QSR purchase-propensity data, but the architecture is not QSR-specific. AxionAI works from a generic tabular artifact contract: feature tables, predictions, labels when available, model metadata, and feature metadata.

## 1. What Is AxionAI?

AxionAI is a local model intelligence MVP for teams that need fast, auditable review of deployed or candidate ML models.

It does not need production customer data and it does not let an LLM calculate metrics. Deterministic Python agents calculate drift, explainability, multicollinearity, segment movement, and risk signals. The narrative layer summarizes only verified evidence.

Current MVP strengths:

- Numeric tabular classification artifacts
- Model monitoring and drift review
- SHAP-based feature importance
- VIF and overfitting diagnostics
- Schema validation and graceful pipeline failures
- Timestamped local run archives
- Dashboard feedback capture for future calibration
- Executive reporting for business and data science stakeholders
- Synthetic-data-only local demo

## 2. Why This Matters

Predictive models can degrade silently when input distributions shift, unstable features become important, or population mix changes before headline metrics move.

AxionAI helps teams:

- Review models faster with an automated first-pass evidence package
- Build business trust with reports that trace back to JSON, CSV, and plots
- Detect drift and segment movement earlier
- Improve feature engineering decisions using SHAP + drift + VIF evidence
- Create client-ready model health briefs without manual report assembly

### What Makes AxionAI Flexible Today

AxionAI is flexible because it is artifact-driven, not platform-driven.

| Real Today | Why It Matters |
| --- | --- |
| File-based artifact contracts | CSV + JSON in, CSV + JSON out. No SDK, API handshake, or vendor-specific runtime required. |
| Model-agnostic metadata schema | `model_metadata.json` declares the model name, target, entity id, prediction column, feature columns, and metrics. |
| Deterministic metrics layer | Python owns drift, SHAP, VIF, clustering, validation, and risk evidence. |
| LLM isolated to narrative prep | The LLM path is optional and must summarize verified evidence only. It does not calculate metrics. |
| No framework lock-in | The current orchestrator is plain Python and can later move into Airflow, Dagster, Prefect, or LangGraph. |
| Local-first execution | Runs locally without API keys and is suitable for restricted or air-gapped demo environments. |
| Dual outputs | JSON for machines, Markdown for stakeholders, PNG plots for demos and reports. |
| Streamlit dashboard | Non-technical users can review evidence without reading raw artifacts. |
| Schema validation | The pipeline checks required metadata fields and CSV columns before agent execution. |
| Timestamped archives | Successful runs are copied into `reports/runs/<run_id>/` for future comparison. |
| Conditional reliability gate | Varuna flags or skips SHAP when Mitra detects severe drift. |
| Feedback hook | Dashboard feedback is saved to seed future calibration and organizational intelligence. |

### Gap / Roadmap

AxionAI is still an MVP. The remaining flexibility work is mostly about connectors, scale, and learning from usage.

| Gap | Roadmap Direction |
| --- | --- |
| Only synthetic data has been tested deeply | Add messy real-world test fixtures: null-heavy columns, schema drift, class imbalance, delayed labels, duplicate entities. |
| Connector layer is not built | Add optional adapters for S3, Snowflake, Databricks, MLflow, and model registry exports. |
| Multi-client execution is basic | Add client/model namespaces and portfolio-level run indexes. |
| Run archives exist but trend analysis is limited | Build run-to-run comparison reports for drift, SHAP movement, health status, and recurring alerts. |
| Feedback is captured but not learned from yet | Build a calibration store that tracks false positives, accepted recommendations, and team-specific thresholds. |
| Organizational intelligence layer does not exist yet | Learn team patterns, known pipeline topology, report usage, and preferred business-language outputs. |
| Categorical and mixed-type support is limited | Add type-aware preprocessing and validation for categorical, ordinal, timestamp, and segment columns. |
| Production governance is not complete | Add approval states, owner metadata, model version lineage, and audit exports. |

## 3. Architecture Diagram

```text
Model Artifacts
  - train/current feature tables
  - current predictions
  - model metadata
  - feature metadata
        |
        v
Agent 01: Mitra
  - PSI, KS, Wasserstein drift checks
  - missing-value checks
  - prediction summary
  - cluster/context shift
        |
        v
Agent 02: Varuna
  - local reviewer model
  - SHAP global drivers
  - VIF multicollinearity
  - train-validation metric delta
  - high-risk feature matrix
  - reliability gate from Mitra drift severity
        |
        v
Evidence Store
  - reports/evidence_packet.json
        |
        v
Agent 03: Aryaman
  - executive_model_report.json
  - executive_model_report.md
        |
        v
Streamlit Dashboard
  - stakeholder review interface
  - feedback capture
        |
        v
Agent 04: Samanvaya
  - feedback review
  - proposed calibration changes
  - simulated approval log
```

## 4. Four-Agent Workflow

### Agent 01: Mitra

Mitra monitors signal quality and distribution movement.

- Data sanity checks
- Missing-value shifts
- Feature drift using PSI
- Feature drift using KS test
- Distribution movement using Wasserstein distance
- Prediction score and label mix summary
- Cluster/context shift using `StandardScaler` and `KMeans`

### Agent 02: Varuna

Varuna explains model behavior and identifies model-level risks.

- Trains a small local reviewer model on supplied artifacts
- Computes SHAP global importance
- Generates SHAP bar and beeswarm plots
- Calculates VIF multicollinearity diagnostics
- Calculates train-validation metric delta from metadata
- Combines SHAP rank, drift level, and VIF into a high-risk feature matrix
- Flags SHAP output as unreliable when Mitra detects severe drift

By default Varuna still runs and labels SHAP as directional when severe drift is present. To make Varuna skip explainability under severe drift:

```bash
VARUNA_DRIFT_GATE_MODE=skip python src/run_axionai_pipeline.py
```

### Agent 03: Aryaman

Aryaman converts deterministic evidence into a concise business report.

- Reads `reports/evidence_packet.json`
- Determines model health status
- Summarizes what changed
- Translates technical findings into business risk
- Recommends next actions
- Saves JSON and Markdown reports

### Agent 04: Samanvaya

Samanvaya reads dashboard feedback and proposes calibration changes without applying them automatically.

- Reads `reports/feedback_log.csv`
- Proposes threshold/config changes for human review
- Writes `reports/samanvaya_recommendations.json`
- Writes `reports/config_change_log.json`
- Writes proposed `configs/calibration_config_v2.json`

## 5. Sample Input Artifacts

To review a different model, replace the sample artifacts while preserving this contract:

```text
data/train_features_sample.csv
  entity id + numeric model features + target

data/current_features_sample.csv
  entity id + same numeric model features + target when available

data/current_predictions_sample.csv
  entity id + prediction score + predicted/actual labels when available

data/train_predictions_sample.csv
  optional reference-window prediction scores used for prediction drift checks

models/model_metadata.json
  model_name, model_type, target, entity_id, prediction_column,
  feature_columns, performance metrics, business_use_case

models/feature_metadata.json
  feature names, types, and business definitions
```

The demo metadata declares `target`, `entity_id`, `prediction_column`, and `feature_columns`, so the agents do not need QSR-specific constants.

## 6. Sample Outputs

Running the pipeline creates:

```text
reports/
  artifact_validation.json
  mitra_output.json
  data_quality_report.csv
  prediction_drift_report.json
  drift_report.csv
  cluster_shift_report.csv
  varuna_output.json
  shap_global_importance.csv
  vif_report.csv
  evidence_packet.json
  executive_model_report.json
  executive_model_report.md
  samanvaya_recommendations.json
  config_change_log.json
  sample_end_to_end_agent_report.md
  runs/<timestamp>/
    archived copy of the generated evidence and report artifacts

reports/figures/
  drift_top_features.png
  shap_bar.png
  shap_beeswarm.png
```

Every generated report artifact includes, where applicable, the active `config_version`, source file paths, and deterministic explanations for assigned risk levels. Mitra also writes the ordered rule hierarchy used to determine its overall risk level.

Typical demo findings:

- Medium executive model health status
- High drift in `merchant_novelty_rate`
- High drift in `weekend_dining_frequency`
- Material cluster movement
- SHAP drivers led by synthetic purchase behavior features
- Recommended validation refresh before high-impact business use

For a full walkthrough of one synthetic run, see:

- [`reports/sample_end_to_end_agent_report.md`](reports/sample_end_to_end_agent_report.md)

## 7. Screenshots

### Model Health Summary

![Model health summary](docs/assets/model_health_summary.png)

### Drift Report

![Drift report](docs/assets/drift_report.png)

### SHAP Feature Importance

![SHAP feature importance](docs/assets/shap_feature_importance.png)

### High-Risk Feature Matrix

![High-risk feature matrix](docs/assets/high_risk_feature_matrix.png)

### Executive Report

![Executive report](docs/assets/executive_report.png)

## 8. Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python src/run_axionai_pipeline.py
```

The pipeline validates the artifact contract first. If metadata fields or required columns are missing, it fails gracefully and writes `reports/pipeline_error.json` with a meaningful next step.

Run against your own existing artifacts without regenerating the synthetic demo:

```bash
python src/run_axionai_pipeline.py --use-existing-artifacts
```

Regenerate README screenshot assets:

```bash
python scripts/render_readme_assets.py
```

Launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

Run local checks:

```bash
python -m unittest discover -s tests
python -m compileall -q src app tests
```

## 9. Financial-Services Use Cases

AxionAI is designed for model review workflows common in financial-services and purchase-analytics environments:

- Credit risk model drift review
- Fraud score monitoring
- Card-linked offer propensity model review
- Customer attrition and retention scoring
- Marketing audience quality checks
- Merchant/category behavior shift analysis
- Client-facing model health reporting
- Feature monitoring before model recalibration

### Future Organizational Intelligence Layer

The dashboard now captures simple analyst feedback in `reports/feedback_log.csv`. This is the first seed for a future calibration store, not a full organizational intelligence layer yet.

The next layer should learn:

- Which teams treat specific flags as useful or noisy
- Which reports are used in reviews or stakeholder meetings
- Which pipelines have known seasonal or source-system changes
- Which thresholds should be adjusted by model family or business unit
- How archived runs differ over time

## 10. Disclaimer: Synthetic Data Only

This repository uses **synthetic sample data only**. It is not affiliated with Affinity Solutions or any financial institution. The MVP is for demonstration, education, and local prototyping, not production model validation.
