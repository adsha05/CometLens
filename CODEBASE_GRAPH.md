# VyaAI Codebase Graph

## Runtime Architecture

```mermaid
flowchart TD
    P["src/run_pipeline.py<br/>One-command orchestrator"]
    G["src/generate_sample_artifacts.py<br/>Synthetic tabular artifacts<br/>(QSR demo profile)"]
    TF["data/train_features_sample.csv<br/>200 training rows"]
    CF["data/current_features_sample.csv<br/>200 current rows with drift"]
    PR["data/current_predictions_sample.csv"]
    MM["models/model_metadata.json"]
    FM["models/feature_metadata.json"]

    S["src/agents/signal_sentinel_agent.py<br/>Agent 01: Mitra<br/>PSI, KS, Wasserstein, missingness, cluster shift"]
    SO["reports/signal_sentinel_output.json"]
    DR["reports/drift_report.csv"]
    CR["reports/cluster_shift_report.csv"]
    DF["reports/figures/drift_top_features.png"]

    L["src/agents/model_lens_agent.py<br/>Agent 02: Varuna<br/>XGBoost, SHAP, VIF, overfitting delta"]
    LO["reports/model_lens_output.json"]
    SH["reports/shap_global_importance.csv"]
    VF["reports/vif_report.csv"]
    SF["reports/figures/shap_global_bar.png<br/>reports/figures/shap_beeswarm.png"]

    E["src/agents/evidence_store.py<br/>Evidence packet builder"]
    EP["reports/evidence_packet.json"]

    X["src/agents/executive_synthesis_agent.py<br/>Agent 03: Aryaman<br/>Deterministic executive brief"]
    XO["reports/executive_model_report.json<br/>reports/executive_model_report.md"]

    UI["app/streamlit_app.py<br/>Dashboard"]

    P --> G
    P --> S
    P --> L
    P --> E
    P --> X
    G --> TF
    G --> CF
    G --> PR
    G --> MM
    G --> FM
    TF --> S
    CF --> S
    PR --> S
    FM --> S
    S --> SO
    S --> DR
    S --> CR
    S --> DF
    TF --> L
    CF --> L
    PR --> L
    MM --> L
    FM --> L
    SO --> L
    L --> LO
    L --> SH
    L --> VF
    L --> SF
    SO --> E
    LO --> E
    MM --> E
    FM --> E
    E --> EP
    EP --> X
    X --> XO
    SO --> UI
    LO --> UI
    EP --> UI
    XO --> UI
```

## Integration Order

```bash
python src/run_vyaai_pipeline.py
streamlit run app/streamlit_app.py
```

## Artifact Contracts

| Producer | Consumer | Contract |
| --- | --- | --- |
| Sample generator or external artifact drop | Mitra, Varuna | Train/current feature tables, labels when available, current predictions, model metadata, feature metadata |
| Model metadata | Mitra, Varuna, Evidence Store | `target`, `entity_id`, `prediction_column`, `feature_columns`, performance metrics, business use case |
| Agent 01: Mitra | Evidence Store, Varuna, Dashboard | Feature drift, prediction summary, missingness, cluster share movement |
| Agent 02: Varuna | Evidence Store, Dashboard | SHAP importance, VIF report, overfitting delta, high-risk feature matrix |
| Evidence Store | Agent 03: Aryaman | Single evidence packet containing deterministic outputs only |
| Agent 03: Aryaman | Dashboard, stakeholders | Consulting-style JSON and Markdown model health brief |

## Verification

```bash
python -m unittest discover -s tests
python -m compileall -q src app tests
```
