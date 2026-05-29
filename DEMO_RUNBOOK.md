# AxionAI Local Demo Runbook

## Demo Goal

Show a deterministic, auditable model intelligence workflow for generic tabular model artifacts. The bundled demo uses a synthetic QSR purchase-propensity profile, but the agents read schema details from metadata.

## Setup

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Rebuild Artifacts

```bash
python src/run_axionai_pipeline.py
```

## Run Checks

```bash
python -m unittest discover -s tests
python -m compileall -q src app tests
```

## Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Storyline

1. Generate synthetic training/current artifacts for the bundled QSR demo profile.
2. Agent 01: Mitra detects drift, prediction-label mix gap, and cluster movement.
3. Agent 02: Varuna trains a small local XGBoost reviewer model and explains behavior with SHAP.
4. Evidence Store consolidates deterministic facts into `reports/evidence_packet.json`.
5. Agent 03: Aryaman produces a consulting-style model health brief.

## Key Message

AxionAI keeps metrics deterministic and auditable. The executive report is generated from saved evidence only; no external LLM calculates or overrides risk.

## Bring Your Own Model Artifacts

To demo a different model, replace the sample CSV/JSON files and keep the artifact contract intact:

- `models/model_metadata.json` defines `target`, `entity_id`, `prediction_column`, and `feature_columns`.
- `models/feature_metadata.json` defines feature names and business meanings.
- Train/current feature tables contain the metadata-defined numeric features.
- Current predictions contain the metadata-defined prediction column and labels when available.
