# PurchaseIntel Lens Local Demo Runbook

## Demo Goal

Show an ML observability workflow that detects synthetic behavior shift, explains
an XGBoost purchase-propensity model, profiles changing customer segments, and
uses a local LLM to summarize verified evidence.

## One-Time Setup

```bash
cd /Users/adityasharma/Desktop/CometLens
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
ollama serve
ollama pull llama3.1:8b
```

`llama3.1:8b` is already available on the current development machine.

## Rebuild Demo Artifacts

In a terminal with the virtual environment active:

```bash
python src/generate_synthetic_data.py
python src/train_model.py
python src/agents/drift_agent.py
python src/agents/explainability_agent.py
python src/agents/cluster_agent.py
python src/agents/feature_suggestion_agent.py
python src/agents/report_agent.py
python src/agents/llm_context_builder.py
python src/agents/narrative_agent.py
python src/validate_integration.py
python scripts/render_readme_assets.py
```

## Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`.

## Screen Walkthrough

| Dashboard Section | What To Demonstrate |
| --- | --- |
| 1. Project Overview | Two 10,000-row synthetic periods and 15 prediction features. |
| 2. Model Health Summary | `High` risk is driven by feature drift, despite stable current AUC. |
| 3. Model Performance Metrics | Compare validation and current AUC, recall, and F1. |
| 4. Top SHAP Feature Drivers | Explain that QSR spend and transaction frequency drive predictions. |
| 5. Drift Report Table | Highlight merchant novelty, fuel spend, weekend dining, and competitor share. |
| 6. Cluster Shift Report | Show contraction in Loyal QSR Buyers and increase in Value-Seeking Routine Shoppers. |
| 7. Feature Suggestions | Show evidence-linked, rule-based candidate features. |
| 8. Model Review Report | Establish the deterministic, auditable business report baseline. |
| 9. AI Narrative Review | Show the Ollama-generated summary grounded only in validated reports. |

## Key Demo Message

The analytical results are deterministic and auditable. The local LLM does not
calculate metrics or set risk; it turns validated synthetic evidence into a
natural-language review for stakeholders.

## Current Local LLM Output

The generated AI narrative artifacts record their provenance:

```text
Provider: ollama
Model: llama3.1:8b
```

Files:

- `reports/llm_evidence_context.json`
- `reports/llm_model_review.json`
- `reports/llm_model_review.md`
