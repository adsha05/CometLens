# VyaAI

**Agentic model intelligence for financial-services ML teams.**

VyaAI reviews model artifacts, detects model-health risks, explains model behavior, and produces an executive-ready model health brief from deterministic evidence.

The bundled demo uses synthetic QSR purchase-propensity data, but the architecture is not QSR-specific. VyaAI works from a generic tabular artifact contract: feature tables, predictions, labels when available, model metadata, and feature metadata.

## 1. What Is VyaAI?

VyaAI is a local model intelligence MVP for teams that need fast, auditable review of deployed or candidate ML models.

It does not need production customer data and it does not let an LLM calculate metrics. Deterministic Python agents calculate drift, explainability, multicollinearity, segment movement, and risk signals. The narrative layer summarizes only verified evidence.

Current MVP strengths:

- Numeric tabular classification artifacts
- Model monitoring and drift review
- SHAP-based feature importance
- VIF and overfitting diagnostics
- Executive reporting for business and data science stakeholders
- Synthetic-data-only local demo

## 2. Why This Matters

Predictive models can degrade silently when input distributions shift, unstable features become important, or population mix changes before headline metrics move.

VyaAI helps teams:

- Review models faster with an automated first-pass evidence package
- Build business trust with reports that trace back to JSON, CSV, and plots
- Detect drift and segment movement earlier
- Improve feature engineering decisions using SHAP + drift + VIF evidence
- Create client-ready model health briefs without manual report assembly

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
```

## 4. Three-Agent Workflow

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

### Agent 03: Aryaman

Aryaman converts deterministic evidence into a concise business report.

- Reads `reports/evidence_packet.json`
- Determines model health status
- Summarizes what changed
- Translates technical findings into business risk
- Recommends next actions
- Saves JSON and Markdown reports

## 5. Sample Input Artifacts

To review a different model, replace the sample artifacts while preserving this contract:

```text
data/train_features_sample.csv
  entity id + numeric model features + target

data/current_features_sample.csv
  entity id + same numeric model features + target when available

data/current_predictions_sample.csv
  entity id + prediction score + predicted/actual labels when available

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
  signal_sentinel_output.json
  drift_report.csv
  cluster_shift_report.csv
  model_lens_output.json
  shap_global_importance.csv
  vif_report.csv
  evidence_packet.json
  executive_model_report.json
  executive_model_report.md

reports/figures/
  drift_top_features.png
  shap_global_bar.png
  shap_beeswarm.png
```

Typical demo findings:

- Medium executive model health status
- High drift in `merchant_novelty_rate`
- High drift in `weekend_dining_frequency`
- Material cluster movement
- SHAP drivers led by synthetic purchase behavior features
- Recommended validation refresh before high-impact business use

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
python src/run_vyaai_pipeline.py
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

VyaAI is designed for model review workflows common in financial-services and purchase-analytics environments:

- Credit risk model drift review
- Fraud score monitoring
- Card-linked offer propensity model review
- Customer attrition and retention scoring
- Marketing audience quality checks
- Merchant/category behavior shift analysis
- Client-facing model health reporting
- Feature monitoring before model recalibration

## 10. Disclaimer: Synthetic Data Only

This repository uses **synthetic sample data only**. It is not affiliated with Affinity Solutions or any financial institution. The MVP is for demonstration, education, and local prototyping, not production model validation.
