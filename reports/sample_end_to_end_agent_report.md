# AxionAI End-to-End Sample Agent Report

**Run ID:** `20260529T183744Z`  
**Artifact profile:** `qsr_purchase_propensity_sample`  
**Model reviewed:** `qsr_purchase_predictor_v3`  
**Model type:** classification  
**Target:** `purchase_qsr_next_30d`  
**Entity:** `consumer_token`  
**Prediction column:** `propensity_score`

## 1. Demo Objective

This sample report shows how AxionAI reviews a synthetic model output end to end.

The demo starts with synthetic train/current feature tables, synthetic current predictions, model metadata, and feature metadata. AxionAI then runs four agents:

- **Agent 01: Mitra** checks data health, drift, prediction movement, and cluster shift.
- **Agent 02: Varuna** explains model behavior using SHAP, VIF, and model-risk diagnostics.
- **Agent 03: Aryaman** turns verified evidence into an executive model health brief.
- **Agent 04: Samanvaya** reads dashboard feedback and proposes calibration changes for human review.

No real customer, consumer, or financial data is used.

## 2. Input Artifacts

The pipeline generated and reviewed these sample artifacts:

| Artifact | Purpose |
| --- | --- |
| `data/train_features_sample.csv` | Synthetic training-period features and labels |
| `data/current_features_sample.csv` | Synthetic current-period features and labels |
| `data/train_predictions_sample.csv` | Reference prediction scores, predicted labels, and actual labels |
| `data/current_predictions_sample.csv` | Current prediction scores, predicted labels, and actual labels |
| `models/model_metadata.json` | Model contract, target, entity id, prediction column, feature list, and simulated metrics |
| `models/feature_metadata.json` | Business definitions for each feature |

The artifact validator passed:

```text
status: passed
model_name: qsr_purchase_predictor_v3
target: purchase_qsr_next_30d
entity_id: consumer_token
prediction_column: propensity_score
feature_count: 9
```

## 3. Sample Model Output

The synthetic current prediction file contains 200 scored consumers.

Key prediction summary from Mitra:

| Metric | Value |
| --- | ---: |
| Row count | 200 |
| Mean propensity score | 0.403 |
| Score p10 | 0.208 |
| Score p50 | 0.383 |
| Score p90 | 0.652 |
| Predicted positive rate | 0.245 |
| Actual positive rate | 0.380 |
| Prediction vs actual rate gap | -0.135 |

Interpretation: the current scoring output under-predicts positives relative to the observed synthetic actual label rate. This does not prove the model is broken, but it is enough to recommend threshold and calibration review before high-impact use.

## 4. Agent 01: Mitra Findings

Mitra reviewed the feature tables, prediction file, and metadata contract.

Overall Mitra risk level: **High**

High-drift features:

| Feature | Direction | PSI | KS p-value | Drift level |
| --- | --- | ---: | ---: | --- |
| `merchant_novelty_rate` | increased by 38.0% | 0.565 | 0.0000019 | High |
| `weekend_dining_frequency` | decreased by 25.5% | 0.314 | 0.0005552 | High |

Cluster movement:

| Cluster | Train share | Current share | Change |
| --- | ---: | ---: | ---: |
| 0 | 49.0% | 40.5% | -8.5 pts |
| 1 | 23.5% | 33.0% | +9.5 pts |
| 2 | 27.5% | 26.5% | -1.0 pts |

Mitra conclusion: the current population differs materially from the training population. Two features crossed high-drift thresholds, and one cluster increased by 9.5 percentage points.

## 5. Agent 02: Varuna Findings

Varuna trained a small local reviewer model on the synthetic training features and explained current-period behavior.

Top SHAP drivers:

| Rank | Feature | Mean absolute SHAP |
| ---: | --- | ---: |
| 1 | `qsr_txn_count_30d` | 0.5763 |
| 2 | `merchant_novelty_rate` | 0.5086 |
| 3 | `grocery_spend_30d` | 0.4651 |
| 4 | `qsr_recency_days` | 0.2392 |
| 5 | `qsr_spend_30d` | 0.2199 |

High-risk feature matrix:

| Feature | SHAP rank | Drift level | VIF warning | Combined risk |
| --- | ---: | --- | --- | --- |
| `merchant_novelty_rate` | 2 | High | Low | High |
| `weekend_dining_frequency` | 8 | High | Low | Medium |

Overfitting check:

| Metric | Value |
| --- | ---: |
| Train AUC | 0.812 |
| Validation AUC | 0.776 |
| Delta | 0.036 |
| Risk level | Medium |

Explainability reliability gate:

Varuna marked SHAP output as **unreliable / directional only** because Mitra detected severe drift.

Severe drift gate triggers:

- `merchant_novelty_rate`: PSI 0.565, KS p-value 0.0000019
- `weekend_dining_frequency`: PSI 0.314, KS p-value 0.0005552

Varuna conclusion: SHAP is still useful for directional review, but it should not be treated as stable production explanation until severe drift is resolved or the model is recalibrated.

## 6. Agent 03: Aryaman Executive Synthesis

Aryaman generated the final executive model health brief from the evidence packet.

Final model health status: **Medium Risk**

Aryaman’s top summary:

> `qsr_purchase_predictor_v3` supports QSR audience targeting and campaign optimization. Current MVP evidence indicates **Medium Risk**. The top issue is that `merchant_novelty_rate` is both a monitored drift issue and model-risk feature. The recommended next step is to review affected segments and recalibrate before high-impact business use.

Recommended actions:

1. Review affected segments and recalibrate before high-impact business use.
2. Prototype `merchant_confidence_score` to separate genuine new-merchant behavior from merchant classification uncertainty.
3. Prototype `weekend_dining_recovery_index` to capture whether weekend dining behavior is rebounding or still depressed.
4. Treat SHAP interpretation as directional until severe drift is resolved or the model is recalibrated.
5. Re-run validation before high-impact business use.

## 7. Agent 04: Samanvaya Calibration Layer

Samanvaya reads analyst feedback from the dashboard and proposes calibration changes for human review.

In this sample run, no dashboard feedback has been recorded yet, so Samanvaya produces no threshold changes. The important architectural point is that feedback does not directly mutate Mitra, Varuna, or Aryaman. It is converted into a reviewable recommendation package:

- `reports/samanvaya_recommendations.json`
- `reports/config_change_log.json`
- `configs/calibration_config_v2.json`

This keeps calibration auditable and prevents silent threshold changes.

## 8. How The Agents Worked Together

```text
Synthetic artifacts
  -> artifact validation passed
  -> Mitra detected high drift and cluster movement
  -> Varuna explained model drivers and flagged SHAP reliability risk
  -> Evidence Store consolidated verified outputs
  -> Aryaman generated the executive report
  -> Samanvaya generated calibration recommendations
  -> Run archive saved outputs under reports/runs/20260529T183744Z/
```

The key design point is that AxionAI does not let narrative logic invent metrics. Every statement in the final report traces back to deterministic files:

- `reports/mitra_output.json`
- `reports/data_quality_report.csv`
- `reports/prediction_drift_report.json`
- `reports/drift_report.csv`
- `reports/varuna_output.json`
- `reports/shap_global_importance.csv`
- `reports/vif_report.csv`
- `reports/evidence_packet.json`

## 9. Generated Visuals

Key plots from this run:

- `reports/figures/drift_top_features.png`
- `reports/figures/shap_bar.png`
- `reports/figures/shap_beeswarm.png`

README-ready visual assets:

- `docs/assets/model_health_summary.png`
- `docs/assets/drift_report.png`
- `docs/assets/shap_feature_importance.png`
- `docs/assets/high_risk_feature_matrix.png`
- `docs/assets/executive_report.png`

## 10. Final Interpretation

The sample model is not treated as failed, but it should not be used blindly.

The evidence suggests:

- Current behavior has shifted materially versus training data.
- `merchant_novelty_rate` is both highly drifted and highly influential.
- `weekend_dining_frequency` is also high drift.
- The prediction-positive rate is below the actual-positive rate.
- SHAP explanations should be treated as directional because severe drift is present.

Recommended business posture: **pause high-impact activation, review affected segments, recalibrate or validate, then rerun AxionAI on the refreshed artifacts.**

## 11. Limitations

- Synthetic sample data only.
- Metrics are simulated for MVP demonstration.
- The local reviewer model is for explainability diagnostics, not production scoring.
- The feedback and organizational intelligence layer is only a starting hook.
- This report is not production model validation.
