# Agent 03: Aryaman Executive Model Health Brief

**Model health status:** Medium Risk

## Executive Summary

qsr_purchase_predictor_v3 supports QSR audience targeting and campaign optimization. Current MVP evidence indicates **Medium Risk**. The top issue is that merchant_novelty_rate is both a monitored drift issue and model-risk feature. The recommended next step is to review affected segments and recalibrate before high-impact business use.

## What Changed

- merchant_novelty_rate is high drift and has combined risk High.
- weekend_dining_frequency is high drift and has combined risk Medium.
- Cluster mix shifted materially; largest movement is 9.5 percentage points.
- Prediction-positive rate differs from actual-positive rate by -0.135.
- Varuna SHAP outputs are flagged as unreliable due to severe Mitra drift.

## Why It Matters

Production model decisions depend on stable inputs, interpretable drivers, and consistent population context. When important features drift or segment mix changes, model scores can become less aligned with the current operating environment, increasing the risk of poor prioritization, inefficient resource allocation, or weak stakeholder trust.

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

- /Users/adityasharma/Desktop/CometLens/reports/figures/shap_global_bar.png
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
