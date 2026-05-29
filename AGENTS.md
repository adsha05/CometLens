# AGENTS.md

## Project

VyaAI is an agentic model intelligence MVP for financial-services and purchase-analytics ML workflows.

## Project Goal

Build a local, auditable workflow that reviews existing model artifacts instead of training production models.

The system takes:

- Feature tables
- Predictions
- Labels
- Model metadata
- Feature metadata

It then runs deterministic Python agents and an optional LLM narrative layer to produce business-facing model intelligence reports.

## Core Principle

The LLM must not calculate metrics.

Deterministic Python tools calculate metrics, tests, scores, drift values, and plots. The LLM only summarizes verified evidence and turns saved outputs into concise narrative.

## MVP Agents

### Agent 01: Mitra

Purpose: detect data, feature, prediction, and context movement.

Responsibilities:

- Run data sanity checks.
- Detect feature drift.
- Detect prediction drift.
- Detect missing-value shifts.
- Detect cluster/context shifts.
- Save auditable outputs as JSON and/or CSV.

Methods:

- Population Stability Index, or PSI
- Kolmogorov-Smirnov test
- Wasserstein distance
- Missing-rate checks
- Clustering-based context shift analysis

### Agent 02: Varuna

Purpose: explain model behavior and identify model-quality risks.

Responsibilities:

- Explain global and local model behavior.
- Generate SHAP summaries and plots.
- Report feature importance.
- Check VIF multicollinearity.
- Compare train-validation overfitting deltas.
- Save selected plots under `reports/figures/`.

Methods:

- SHAP
- Feature importance
- Variance Inflation Factor, or VIF
- Train-validation metric deltas
- Plotly or Matplotlib visualizations

### Agent 03: Aryaman

Purpose: create a concise executive model-health report.

Responsibilities:

- Read only verified outputs from Mitra and Varuna.
- Generate a concise McKinsey/BCG-style report for business teams, clients, and data science partners.
- Separate deterministic findings from LLM-written narrative.
- Save Markdown and JSON outputs.

Rules:

- Use only supplied evidence.
- Do not invent metrics, root causes, customer behavior, market events, or business facts.
- Do not restate hypotheses as proven causes.
- Include limitations and data assumptions.
- Cite the deterministic artifacts used.

## Data Rules

- Use synthetic sample data only.
- Do not use real financial data.
- Do not use real customer data.
- Do not commit secrets, credentials, raw production extracts, or personally identifiable information.
- Keep sample files small enough to run locally.
- Any realistic-looking data must be explicitly synthetic.

## Engineering Rules

- Keep the MVP small and modular.
- Prefer simple, readable Python over framework-heavy abstractions.
- Each module should be runnable locally without hidden dependencies.
- Keep deterministic calculations separate from LLM summarization.
- Save outputs in auditable formats: JSON, CSV, Markdown, and selected figures.
- Add docstrings where they clarify purpose or contracts.
- Avoid broad refactors unless they directly support the requested workflow.
- Prefer pandas, numpy, scipy, scikit-learn, xgboost, shap, plotly, pydantic, jinja2, duckdb, and joblib.

## LLM Rules

- The LLM is a narrative layer only.
- The LLM receives compact evidence packets, not raw sensitive data.
- The LLM must not calculate PSI, KS, Wasserstein distance, SHAP values, VIF, metrics, thresholds, or drift labels.
- The LLM must not override deterministic risk levels.
- LLM outputs must be schema-validated with Pydantic.
- LLM outputs should be saved with provider, model, timestamp, and evidence-source metadata.
- Local LLM testing should be supported before cloud API integration.

## Expected Folder Structure

```text
data/
models/
reports/
reports/figures/
src/
src/agents/
src/llm/
src/utils/
app/
README.md
requirements.txt
```

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- scipy
- xgboost
- shap
- plotly
- streamlit
- pydantic
- jinja2
- duckdb
- joblib

## Output Expectations

Agents should produce clear artifacts such as:

- `reports/signal_sentinel_report.json`
- `reports/model_lens_report.json`
- `reports/executive_synthesis_report.md`
- `reports/executive_synthesis_report.json`
- `reports/figures/*.png`

Exact filenames may evolve, but outputs must remain auditable, reproducible, and easy to load in Streamlit.

## Local Demo Priority

The MVP should run locally end to end:

1. Load or generate synthetic sample artifacts.
2. Run Mitra checks.
3. Run Varuna explainability and diagnostics.
4. Build a verified evidence packet.
5. Optionally generate an LLM narrative from that packet.
6. Render results in Streamlit.

## Style

- Simple, modular, readable code.
- Clear file boundaries.
- No overengineering.
- No hidden data dependencies.
- No real financial or customer data.
- Business-facing outputs should be concise, evidence-based, and client-ready.
