# Agent 03: Aryaman Client-Ready Purchase Model Intelligence Brief

**Model health status:** Medium Risk
**Config version:** v1

## Executive Summary

qsr_purchase_predictor_v3 supports QSR audience targeting and campaign optimization. Current MVP evidence indicates **Medium Risk**. The top issue is that merchant_novelty_rate is both a monitored drift issue and model-risk feature. The recommended next step is to review affected segments and recalibrate before high-impact business use.

## What Changed

- merchant_novelty_rate is high drift and has combined risk High.
- weekend_dining_frequency is high drift and has combined risk Medium.
- Cluster mix shifted materially; largest movement is 9.5 percentage points.
- Prediction-positive rate differs from actual-positive rate by -0.130.
- Varuna SHAP outputs are flagged as unreliable due to severe Mitra drift.

## Why It Matters

Purchase and audience decisioning depend on stable behavioral signals. When important features drift or segment mix changes, scores can become less aligned with current consumer behavior, increasing the risk of poor prioritization, inefficient spend, or weak client trust.

## Top Model Drivers

- qsr_txn_count_30d is SHAP rank 1 with mean |SHAP| 0.5763
- merchant_novelty_rate is SHAP rank 2 with mean |SHAP| 0.5086
- grocery_spend_30d is SHAP rank 3 with mean |SHAP| 0.4651
- qsr_recency_days is SHAP rank 4 with mean |SHAP| 0.2392
- qsr_spend_30d is SHAP rank 5 with mean |SHAP| 0.2199

## High-Risk Features

- merchant_novelty_rate: combined risk High, drift High, VIF warning Low
- weekend_dining_frequency: combined risk Medium, drift High, VIF warning Low

## Business Risks

- Decision quality may decline if current input behavior differs from training-period behavior.
- Resource allocation may become less efficient if high-drift features are used without validation refresh.
- High-impact business use should wait for validation and segment review.
- Prediction-label mix gap may indicate threshold or calibration pressure.

## Recommended Actions

- Review affected segments and recalibrate before high-impact business use.
- Prototype `merchant_confidence_score` to separate genuine new-merchant behavior from merchant classification uncertainty.
- Prototype `weekend_dining_recovery_index` to capture whether weekend dining behavior is rebounding or still depressed.
- Treat SHAP interpretation as directional until severe drift is resolved or the model is recalibrated.
- Re-run validation before high-impact business use.

## Plots To Include

- /Users/adityasharma/Desktop/CometLens/reports/figures/shap_bar.png
- /Users/adityasharma/Desktop/CometLens/reports/figures/shap_beeswarm.png
- /Users/adityasharma/Desktop/CometLens/reports/figures/drift_top_features.png

## Questions For The Team

- Did any source-system, data-definition, policy, product, or population mix change coincide with the current-window drift?
- Should threshold calibration be reviewed for the observed prediction-positive versus actual-positive gap?
- Do high-risk features require new monitoring thresholds before high-impact business use?

## Client-Safe Summary

The model review is rated Medium Risk based on synthetic MVP evidence. The main findings are feature drift, segment movement, and validation-risk indicators; these should be reviewed before high-impact business decisions.

## Limitations

- Synthetic sample data only
- LLM narrative should not be treated as production validation
- Metrics are simulated for MVP demonstration

## Source Files

- `evidence_packet`: `/Users/adityasharma/Desktop/CometLens/reports/evidence_packet.json`
- `mitra_output`: `/Users/adityasharma/Desktop/CometLens/reports/mitra_output.json`
- `varuna_output`: `/Users/adityasharma/Desktop/CometLens/reports/varuna_output.json`
- `model_metadata`: `/Users/adityasharma/Desktop/CometLens/models/model_metadata.json`
- `feature_metadata`: `/Users/adityasharma/Desktop/CometLens/models/feature_metadata.json`
- `mitra_source_files`: `{'train_features': '/Users/adityasharma/Desktop/CometLens/data/train_features_sample.csv', 'current_features': '/Users/adityasharma/Desktop/CometLens/data/current_features_sample.csv', 'train_predictions': '/Users/adityasharma/Desktop/CometLens/data/train_predictions_sample.csv', 'current_predictions': '/Users/adityasharma/Desktop/CometLens/data/current_predictions_sample.csv', 'model_metadata': '/Users/adityasharma/Desktop/CometLens/models/model_metadata.json', 'feature_metadata': '/Users/adityasharma/Desktop/CometLens/models/feature_metadata.json', 'calibration_config': '/Users/adityasharma/Desktop/CometLens/configs/calibration_config_v1.json'}`
- `varuna_source_files`: `{'train_features': '/Users/adityasharma/Desktop/CometLens/data/train_features_sample.csv', 'current_features': '/Users/adityasharma/Desktop/CometLens/data/current_features_sample.csv', 'current_predictions': '/Users/adityasharma/Desktop/CometLens/data/current_predictions_sample.csv', 'model_metadata': '/Users/adityasharma/Desktop/CometLens/models/model_metadata.json', 'feature_metadata': '/Users/adityasharma/Desktop/CometLens/models/feature_metadata.json', 'mitra_output': '/Users/adityasharma/Desktop/CometLens/reports/mitra_output.json'}`
