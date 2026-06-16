# Agent 03: Aryaman Executive Model Health Brief

## 1. Executive Summary

qsr_purchase_predictor_v3 supports QSR audience targeting and campaign optimization. Deterministic evidence indicates **High Risk**. The leading issue is: merchant_novelty_rate is high drift and appears in the feature risk matrix. Review the recommended actions before activation.

## 2. Model Health Status

**High Risk**

## 3. What Changed

- merchant_novelty_rate is high drift and appears in the feature risk matrix.
- weekend_dining_frequency is high drift and appears in the feature risk matrix.
- merchant_novelty_rate is a high-risk feature because it is model-important and has high drift.
- Prediction drift is High based on score distribution shift and KS p-value.
- Train-validation AUC delta indicates medium overfitting risk.

## 4. Why It Matters

Model decisions depend on stable signals and validated score behavior. Drift in model-important features or prediction distributions can reduce confidence in current-window activation decisions.

## 5. Top Model Drivers

- merchant_novelty_rate is SHAP rank 1 with mean |SHAP| 0.4256.
- qsr_txn_count_30d is SHAP rank 2 with mean |SHAP| 0.4084.
- grocery_spend_30d is SHAP rank 3 with mean |SHAP| 0.3089.
- qsr_spend_30d is SHAP rank 4 with mean |SHAP| 0.2315.
- qsr_recency_days is SHAP rank 5 with mean |SHAP| 0.1798.

## 6. High-Risk Features

- merchant_novelty_rate: High because the feature is a top-5 SHAP driver with High drift.

## 7. Business Risks

- Decision quality may decline if current-window behavior differs from the validated baseline.
- Activation efficiency may decline if score movement is not reviewed before operational use.
- High-impact use should wait for validation and feature-stability review.

## 8. Recommended Actions

- Review high-risk feature stability and consider recalibration before activation.
- Run validation before campaign or model activation because prediction drift is High.
- Review upstream data pipelines for medium or high data-quality issues.
- Review high-drift features for data pipeline or behavior-shift explanations.
- Compare prediction score distribution against the model training baseline when train predictions are available.
- Inspect cluster share movement for population-mix changes before high-impact business use.
- Review prediction score drift against decision thresholds and downstream actions.

## 9. Evidence Appendix

- Model: `qsr_purchase_predictor_v3`
- Config version: `v1`
- Run ID: `20260616T190849Z`
- `mitra_output`: `/Users/adityasharma/Desktop/CometLens/reports/mitra_output.json`
- `varuna_output`: `/Users/adityasharma/Desktop/CometLens/reports/model_lens_output.json`
- `data_quality_report`: `/Users/adityasharma/Desktop/CometLens/reports/data_quality_report.csv`
- `drift_report`: `/Users/adityasharma/Desktop/CometLens/reports/drift_report.csv`
- `prediction_drift_report`: `/Users/adityasharma/Desktop/CometLens/reports/prediction_drift_report.json`
- `feature_risk_matrix`: `/Users/adityasharma/Desktop/CometLens/reports/feature_risk_matrix.csv`
- `model_diagnostics`: `/Users/adityasharma/Desktop/CometLens/reports/model_diagnostics.json`
- `calibration_report`: `/Users/adityasharma/Desktop/CometLens/reports/calibration_report.csv`
- `score_decile_report`: `/Users/adityasharma/Desktop/CometLens/reports/score_decile_report.csv`
- `lift_report`: `/Users/adityasharma/Desktop/CometLens/reports/lift_report.csv`
- `vishwakarma_output`: `/Users/adityasharma/Desktop/CometLens/reports/visuals/vishwakarma_output.json`
- `model_metadata`: `/Users/adityasharma/Desktop/CometLens/models/model_metadata.json`
- `feature_metadata`: `/Users/adityasharma/Desktop/CometLens/models/feature_metadata.json`
- `calibration_config`: `/Users/adityasharma/Desktop/CometLens/configs/calibration_config_v1.json`

### Recommended Visuals

- feature_risk_scatter
- prediction_distribution_overlay
- lineage_graph

## 10. Limitations

- Synthetic demo data only
- Not validated for production use
- No real customer or financial data used
