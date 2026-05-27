# PurchaseIntel Lens Model Results

## Executive Summary

The initial `XGBClassifier` was trained to predict
`purchase_qsr_next_30d` using 15 synthetic behavioral and demographic
features. The model reached a validation AUC of **0.7665** and a
current-period AUC of **0.7743**.

The current synthetic snapshot includes deliberate behavior drift. Its observed
QSR purchase target rate decreased from **22.52%** in the training snapshot to
**20.22%** in the current snapshot.

## Dataset Summary

| Dataset | Rows | Feature Columns | Target Rate |
| --- | ---: | ---: | ---: |
| Training snapshot | 10,000 | 15 | 22.52% |
| Current snapshot | 10,000 | 15 | 20.22% |

The output files each also include `user_id` and the target column.

## Model Details

| Item | Value |
| --- | --- |
| Model name | `qsr_purchase_propensity_xgb` |
| Model type | `XGBClassifier` |
| Target | `purchase_qsr_next_30d` |
| Training window | `synthetic_training_snapshot` |
| Evaluation window | `synthetic_current_snapshot` |
| Classification threshold | `0.50` |

## Performance Metrics

| Evaluation Dataset | AUC | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation split | 0.7665 | 0.8000 | 0.6078 | 0.3133 | 0.4135 |
| Current snapshot | 0.7743 | 0.8209 | 0.6215 | 0.2923 | 0.3976 |

### Interpretation

- AUC is stable across validation and current data, indicating that model
  ranking remains reasonably consistent under the simulated shifts.
- Recall is low at the default `0.50` threshold: the model finds fewer than
  one third of positive purchase outcomes.
- Current-period F1 is lower than validation F1 because recall declined, even
  though accuracy and precision increased.
- Threshold tuning should be evaluated later if capturing likely QSR purchasers
  is more important than limiting false positives.

## Simulated Drift Summary

| Feature | Train Mean | Current Mean | Absolute Change | Percent Change |
| --- | ---: | ---: | ---: | ---: |
| `fuel_spend_30d` | 60.8808 | 76.7677 | +15.8869 | +26.10% |
| `weekend_dining_frequency` | 1.6310 | 1.2331 | -0.3979 | -24.40% |
| `merchant_novelty_rate` | 0.2944 | 0.3852 | +0.0909 | +30.87% |
| `competitor_qsr_share_90d` | 0.3841 | 0.4273 | +0.0432 | +11.26% |

These changes are intentionally generated for drift monitoring development and
do not represent real customer behavior.

## Model Features

| Feature |
| --- |
| `qsr_txn_count_30d` |
| `qsr_spend_30d` |
| `qsr_recency_days` |
| `competitor_qsr_share_90d` |
| `grocery_spend_30d` |
| `fuel_spend_30d` |
| `weekend_dining_frequency` |
| `avg_ticket_size` |
| `dining_category_entropy` |
| `merchant_novelty_rate` |
| `campaign_exposed` |
| `income_band_encoded` |
| `age_band_encoded` |
| `region_encoded` |
| `prior_offer_redemption_rate` |

## Generated Artifacts

| Artifact | Path |
| --- | --- |
| Trained model | `models/qsr_xgb_model.joblib` |
| Model metadata | `models/model_metadata.json` |
| Training predictions | `data/train_predictions.csv` |
| Current predictions | `data/current_predictions.csv` |
| Training features | `data/train_features.csv` |
| Current features | `data/current_features.csv` |

## Reproduce Results

```bash
python src/generate_synthetic_data.py
python src/train_model.py
# Or run the trainer as a module from the project root:
python -m src.train_model
```

All data used for this report is synthetic.
