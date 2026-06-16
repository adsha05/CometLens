# AxionAI Sample Outputs

AxionAI saves machine-readable evidence and stakeholder-facing summaries. The bundled artifacts are generated from synthetic QSR propensity data only.

## Demo Findings

| Area | Synthetic demo result |
| --- | --- |
| Executive model health | `High Risk` |
| Mitra risk | `High` |
| High-drift features | `merchant_novelty_rate`, `weekend_dining_frequency` |
| Prediction score drift | `High` |
| Prediction score mean movement | Approximately `-9.0%` |
| Calibration diagnostics | Brier score, expected calibration error, and calibration bins |
| Lift diagnostics | Top-decile lift and cumulative capture rate |
| Segment diagnostics | Cohort-level score, outcome, lift, Brier score, and calibration gap |
| Leading SHAP driver | `merchant_novelty_rate` |
| Samanvaya recommendations | `2` pending human approval |

## Monitoring Outputs

| Artifact | Purpose |
| --- | --- |
| `reports/mitra_output.json` | Mitra summary, risk explanation, drift lists, cluster findings |
| `reports/data_quality_report.csv` | Missing-rate shifts, boundary checks, cardinality checks |
| `reports/drift_report.csv` | PSI, KS, Wasserstein distance, means, and drift labels |
| `reports/prediction_drift_report.json` | Reference/current score-distribution movement |
| `reports/cluster_shift_report.csv` | Reference/current cluster-share comparison |

## Model Diagnostic Outputs

| Artifact | Purpose |
| --- | --- |
| `reports/varuna_output.json` | Varuna summary and explainability reliability status |
| `reports/shap_global_importance.csv` | Ranked global SHAP importance |
| `reports/vif_report.csv` | Multicollinearity diagnostics |
| `reports/model_diagnostics.json` | Overfitting delta and diagnostic metadata |
| `reports/score_decile_report.csv` | Score-decile performance, lift, and cumulative gains |
| `reports/calibration_report.csv` | Calibration bins, average predicted score, actual rate, and calibration gap |
| `reports/lift_report.csv` | Decile lift and cumulative lift report |
| `reports/segment_performance_report.csv` | Segment-level model performance by top model driver cohorts |
| `reports/feature_risk_matrix.csv` | Combined SHAP, drift, and VIF feature-risk evidence |

## Evidence And Reporting Outputs

| Artifact | Purpose |
| --- | --- |
| `reports/evidence_packet.json` | Verified handoff for downstream synthesis |
| `reports/aryaman_output.json` | Structured executive synthesis |
| `reports/executive_model_report.md` | Business-readable model-health brief |
| `reports/executive_model_report.json` | Schema-validated structured report |

## Visual Intelligence Outputs

| Artifact | Purpose |
| --- | --- |
| `reports/visuals/feature_risk_scatter.html` | Interactive SHAP-versus-PSI feature map |
| `reports/visuals/prediction_distribution_overlay.html` | Interactive reference/current score overlay |
| `reports/visuals/lineage_graph.svg` | Run-specific architecture and risk lineage |
| `reports/visuals/vishwakarma_output.json` | Visual manifest with source files and warnings |

## Model Performance Figures

| Artifact | Purpose |
| --- | --- |
| `reports/figures/calibration_curve.png` | Visual calibration curve against the ideal diagonal |
| `reports/figures/lift_chart.png` | Decile lift and cumulative lift chart |
| `reports/figures/segment_performance_heatmap.png` | Heatmap of segment-level calibration gaps |

## Governed Feedback Outputs

| Artifact | Purpose |
| --- | --- |
| `reports/feedback_log.csv` | Structured analyst and stakeholder feedback events |
| `reports/samanvaya_output.json` | Feedback-calibration summary |
| `reports/calibration_recommendations.json` | Pending human-review recommendations |
| `reports/config_change_log.json` | Audit log with approved and rejected change slots |
| `configs/calibration_config_v2_recommended.json` | Proposed config; never activated automatically |

## Read The Executive Brief

The generated Markdown report is available at [`../reports/executive_model_report.md`](../reports/executive_model_report.md).

The report is intentionally explicit about its limits:

- Synthetic demo data only
- Not validated for production use
- No real customer or financial data used
