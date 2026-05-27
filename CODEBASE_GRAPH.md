# PurchaseIntel Lens Codebase Graph

## Runtime Architecture

```mermaid
flowchart TD
    G["src/generate_synthetic_data.py<br/>Seeded synthetic data generator"]
    TF["data/train_features.csv<br/>10,000 rows, target + 15 features"]
    CF["data/current_features.csv<br/>10,000 rows with simulated drift"]
    T["src/train_model.py<br/>XGBClassifier"]
    M["models/qsr_xgb_model.joblib"]
    MD["models/model_metadata.json"]
    TP["data/train_predictions.csv"]
    CP["data/current_predictions.csv"]
    D["src/agents/drift_agent.py<br/>PSI + KS"]
    E["src/agents/explainability_agent.py<br/>SHAP TreeExplainer"]
    C["src/agents/cluster_agent.py<br/>StandardScaler + KMeans"]
    DR["reports/drift_report.csv"]
    SR["reports/shap_global_importance.csv<br/>reports/top_features.json<br/>SHAP PNG plots"]
    CR["reports/cluster_shift_report.csv<br/>cluster profiles"]
    F["src/agents/feature_suggestion_agent.py<br/>Rule-based suggestions"]
    FR["reports/feature_suggestions.csv"]
    R["src/agents/report_agent.py<br/>Business report + risk rules"]
    RR["reports/model_review_report.md"]
    X["src/agents/llm_context_builder.py<br/>Validated evidence package"]
    N["src/agents/narrative_agent.py<br/>Optional LLM narrative"]
    O["src/llm/ollama_provider.py<br/>Local Ollama first"]
    NR["reports/llm_model_review.json/.md"]
    UI["app/streamlit_app.py<br/>Plotly + Streamlit dashboard"]

    G --> TF
    G --> CF
    TF --> T
    CF --> T
    T --> M
    T --> MD
    T --> TP
    T --> CP
    TF --> D
    CF --> D
    D --> DR
    M --> E
    MD --> E
    TF --> E
    CF --> E
    E --> SR
    TF --> C
    CF --> C
    MD --> C
    TP --> C
    CP --> C
    C --> CR
    DR --> F
    SR --> F
    CR --> F
    F --> FR
    MD --> R
    DR --> R
    SR --> R
    CR --> R
    FR --> R
    R --> RR
    MD --> X
    DR --> X
    SR --> X
    CR --> X
    FR --> X
    X --> N
    O --> N
    N --> NR
    TF --> UI
    CF --> UI
    MD --> UI
    DR --> UI
    SR --> UI
    CR --> UI
    FR --> UI
    RR --> UI
    NR --> UI
```

## Integration Order

```bash
python src/generate_synthetic_data.py
python src/train_model.py
python src/agents/drift_agent.py
python src/agents/explainability_agent.py
python src/agents/cluster_agent.py
python src/agents/feature_suggestion_agent.py
python src/agents/report_agent.py
python src/agents/llm_context_builder.py
# Optional AI narrative after Ollama is running:
python src/agents/narrative_agent.py
python src/validate_integration.py
streamlit run app/streamlit_app.py
```

## Artifact Contracts

| Producer | Consumer | Contract |
| --- | --- | --- |
| Synthetic generator | Trainer, drift, SHAP, clustering, dashboard | `user_id`, 15 numeric model features, `purchase_qsr_next_30d` |
| Trainer | SHAP, clustering, report, dashboard | XGBoost model, feature order, metrics, user probabilities |
| Drift agent | Suggestions, report, dashboard | One row per trained feature with PSI, KS, mean shift, drift severity |
| Explainability agent | Suggestions, report, dashboard | One row per feature ranked by mean absolute SHAP value plus plot files |
| Cluster agent | Suggestions, report, dashboard | Named cluster shares, shifts, profiles, and optional prediction means |
| Feature suggestion agent | Report, dashboard | Proposed feature, reason, linked evidence, priority |
| Report agent | Dashboard | Markdown review with performance, monitoring, recommendations, and risk |
| LLM context builder | Narrative agent | Compact evidence payload containing validated metrics and findings only |
| Narrative agent | Dashboard | Optional structured LLM narrative with provider/model provenance |

## LLM Usage

The core analytics and risk decision remain deterministic. An optional local
LLM narrative layer can now summarize the verified outputs for the demo.

| Component | Technique Used |
| --- | --- |
| Prediction model | `XGBClassifier` |
| Explainability | SHAP `TreeExplainer` |
| Drift checks | PSI and Kolmogorov-Smirnov tests |
| Segmentation | `StandardScaler` and `KMeans` |
| Feature suggestions | Fixed Python rules over saved reports |
| Model review report | Deterministic Markdown template over saved reports |
| AI narrative review | Optional Ollama model over validated evidence only |
| Dashboard | Streamlit and Plotly rendering |

`reports/model_review_report.md` remains the source-of-truth deterministic
report. When generated, `reports/llm_model_review.md` is explicitly labeled
as AI interpretation and records the provider and model used.

## Verification

Run:

```bash
python src/validate_integration.py
```

The validator checks that feature, prediction, model metadata, monitoring,
recommendation, and final report artifacts exist and agree on the shared
feature and population contracts.
