# PurchaseIntel Lens Model Review Report

## 1. Executive Summary

**Model risk level: High.** 4 features have high drift; AUC change is +0.0077.

The model maintains ranking performance on the current synthetic snapshot (AUC 0.7743 versus validation AUC 0.7665), but 4 monitored features show high drift. The main operational concern is changing customer behavior rather than immediate discrimination loss.

## 2. Model Objective

`qsr_purchase_propensity_xgb` is a `XGBClassifier` model that predicts `purchase_qsr_next_30d` using 15 synthetic customer behavior and profile features.

## 3. Model Performance

| Dataset | AUC | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.7665 | 0.8000 | 0.6078 | 0.3133 | 0.4135 |
| Current | 0.7743 | 0.8209 | 0.6215 | 0.2923 | 0.3976 |

Current-period accuracy and precision remain stable, while recall is lower. The default decision threshold may miss likely purchasers.

## 4. Top Feature Drivers

| Rank | Feature | Mean Absolute SHAP Value |
| ---: | --- | ---: |
| 1 | `qsr_spend_30d` | 0.5467 |
| 2 | `qsr_txn_count_30d` | 0.3633 |
| 3 | `weekend_dining_frequency` | 0.1903 |
| 4 | `qsr_recency_days` | 0.1901 |
| 5 | `campaign_exposed` | 0.1561 |
| 6 | `competitor_qsr_share_90d` | 0.1253 |
| 7 | `prior_offer_redemption_rate` | 0.1001 |
| 8 | `fuel_spend_30d` | 0.0909 |
| 9 | `merchant_novelty_rate` | 0.0792 |
| 10 | `avg_ticket_size` | 0.0713 |

QSR spend and transaction frequency are the leading prediction drivers. Weekend dining frequency and competitor share are both influential and drifting.

## 5. Drift Findings

| Feature | Drift Level | PSI | Mean Change |
| --- | --- | ---: | ---: |
| `merchant_novelty_rate` | High | 0.5243 | +30.87% |
| `fuel_spend_30d` | High | 0.1233 | +26.10% |
| `weekend_dining_frequency` | High | 0.1206 | -24.40% |
| `competitor_qsr_share_90d` | High | 0.0572 | +11.26% |

Fuel spend and merchant novelty increased materially, weekend dining declined, and competitor QSR share increased. These shifts should be monitored before model refresh decisions.

## 6. Segment/Cluster Findings

| Segment | Reference Share | Current Share | Shift |
| --- | ---: | ---: | ---: |
| Loyal QSR Buyers | 27.88% | 25.87% | -2.01 pts |
| Value-Seeking Routine Shoppers | 23.69% | 24.90% | +1.21 pts |
| Unstable/New Behavior Segment | 40.44% | 41.03% | +0.59 pts |
| Competitor Switchers | 7.99% | 8.20% | +0.21 pts |

The Loyal QSR Buyers segment contracted while Value-Seeking Routine Shoppers increased, consistent with changing behavioral mix in the current snapshot.

## 7. Feature Recommendations

| Suggested Feature | Priority | Rationale |
| --- | --- | --- |
| `competitor_share_rolling_60d` | High | Measure recent movement toward competing QSR merchants with shorter and directional signals. |
| `competitor_switching_velocity` | High | Measure recent movement toward competing QSR merchants with shorter and directional signals. |
| `merchant_confidence_score` | High | Capture changing merchant behavior and distinguish genuine novelty from uncertain categorization. |
| `merchant_descriptor_novelty_rate` | High | Capture changing merchant behavior and distinguish genuine novelty from uncertain categorization. |
| `weekend_dining_recovery_index` | High | Track whether reduced weekend dining is recovering or persisting for purchase propensity. |
| `cluster_stability_score` | Medium | Account for material movement in customer segment composition between periods. |
| `segment_specific_calibration` | Medium | Account for material movement in customer segment composition between periods. |

## 8. Recommended Actions

1. Continue current model monitoring, but flag the deployment for high drift review.
2. Track high-drift predictors and segment distribution in the next current-period snapshot.
3. Prototype the high-priority candidate features: `competitor_share_rolling_60d`, `competitor_switching_velocity`, `merchant_confidence_score`, `merchant_descriptor_novelty_rate`, `weekend_dining_recovery_index`.
4. Evaluate threshold tuning or calibration because current recall is below validation recall.
5. Retrain or recalibrate only after confirming whether the observed behavioral shifts persist.

All data and findings in this report are based on synthetic observations.
