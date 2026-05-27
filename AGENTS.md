# AGENTS.md

## Project
PurchaseIntel Lens is a Python ML observability project for synthetic consumer purchase analytics.

## Goal
Build an explainability, drift, clustering, and feature recommendation agent around a purchase-propensity model.

## MVP Scope
- Generate synthetic users, transactions, campaign exposure, and features.
- Train an XGBoost model to predict QSR purchase in the next 30 days.
- Run SHAP explainability.
- Run drift checks between training and current periods.
- Cluster customers into behavior segments.
- Generate feature suggestions and a model health report.
- Display results in Streamlit.

## Style
- Keep code simple and modular.
- Avoid overengineering.
- No real financial/consumer data.
- Use synthetic data only.
- Prefer pandas and scikit-learn.
- Add docstrings and comments where useful.
- Each file should run without hidden dependencies.

## Main folders
- src/generate_synthetic_data.py
- src/train_model.py
- src/agents/explainability_agent.py
- src/agents/drift_agent.py
- src/agents/cluster_agent.py
- src/agents/feature_suggestion_agent.py
- src/agents/report_agent.py
- app/streamlit_app.py
