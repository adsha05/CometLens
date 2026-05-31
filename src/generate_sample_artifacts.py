"""Generate small synthetic tabular model-review artifacts for AxionAI.

The bundled profile is a QSR purchase-propensity example. The downstream agents
read the schema from metadata, so other tabular model outputs can use the same
artifact contract with different feature, target, entity, and prediction names.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TARGET = "purchase_qsr_next_30d"
ENTITY_ID = "consumer_token"
RANDOM_SEED = 42
ROW_COUNT = 200

FEATURE_COLUMNS = [
    "qsr_txn_count_30d",
    "qsr_spend_30d",
    "qsr_recency_days",
    "competitor_qsr_share_90d",
    "grocery_spend_30d",
    "fuel_spend_30d",
    "weekend_dining_frequency",
    "merchant_novelty_rate",
    "campaign_exposed",
]


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities."""
    return 1 / (1 + np.exp(-values))


def build_feature_frame(seed: int, rows: int = ROW_COUNT, drift: bool = False) -> pd.DataFrame:
    """Create one synthetic feature snapshot with optional current-period drift."""
    rng = np.random.default_rng(seed)
    tokens = [f"consumer_{index:04d}" for index in range(1, rows + 1)]

    qsr_txn_count = rng.poisson(lam=3.2, size=rows)
    qsr_spend = np.maximum(0, qsr_txn_count * rng.normal(14.5, 3.0, rows) + rng.normal(8, 10, rows))
    qsr_recency = rng.integers(1, 46, size=rows)
    competitor_share = rng.beta(2.2, 4.2, rows)
    grocery_spend = rng.gamma(shape=3.0, scale=34.0, size=rows)
    fuel_spend = rng.gamma(shape=2.2, scale=24.0, size=rows)
    weekend_frequency = rng.poisson(lam=1.6, size=rows)
    merchant_novelty = rng.beta(2.0, 5.5, rows)
    campaign_exposed = rng.binomial(1, 0.36, rows)

    if drift:
        weekend_frequency = rng.binomial(
            n=np.maximum(weekend_frequency, 1),
            p=0.68,
        )
        merchant_novelty = np.clip(merchant_novelty + rng.normal(0.09, 0.035, rows), 0, 1)
        competitor_share = np.clip(competitor_share + rng.normal(0.055, 0.025, rows), 0, 1)
        fuel_spend = fuel_spend * rng.normal(1.10, 0.04, rows)

    logits = (
        -1.25
        + 0.22 * qsr_txn_count
        + 0.011 * qsr_spend
        - 0.025 * qsr_recency
        - 0.9 * competitor_share
        + 0.28 * weekend_frequency
        + 0.45 * campaign_exposed
        - 0.32 * merchant_novelty
    )
    probabilities = sigmoid(logits)
    labels = rng.binomial(1, probabilities)

    return pd.DataFrame(
        {
            ENTITY_ID: tokens,
            "qsr_txn_count_30d": qsr_txn_count,
            "qsr_spend_30d": qsr_spend.round(2),
            "qsr_recency_days": qsr_recency,
            "competitor_qsr_share_90d": competitor_share.round(4),
            "grocery_spend_30d": grocery_spend.round(2),
            "fuel_spend_30d": fuel_spend.round(2),
            "weekend_dining_frequency": weekend_frequency,
            "merchant_novelty_rate": merchant_novelty.round(4),
            "campaign_exposed": campaign_exposed,
            TARGET: labels,
        }
    )


def build_predictions(features: pd.DataFrame, seed: int, score_shift: float = 0.0) -> pd.DataFrame:
    """Create synthetic prediction scores and labels for one artifact window."""
    rng = np.random.default_rng(seed)
    logits = (
        -1.1
        + 0.19 * features["qsr_txn_count_30d"]
        + 0.010 * features["qsr_spend_30d"]
        - 0.021 * features["qsr_recency_days"]
        - 0.75 * features["competitor_qsr_share_90d"]
        + 0.22 * features["weekend_dining_frequency"]
        + 0.38 * features["campaign_exposed"]
        - 0.26 * features["merchant_novelty_rate"]
        + score_shift
        + rng.normal(0, 0.18, len(features))
    )
    propensity_score = sigmoid(logits)
    return pd.DataFrame(
        {
            ENTITY_ID: features[ENTITY_ID],
            "propensity_score": propensity_score.round(6),
            "predicted_label": (propensity_score >= 0.5).astype(int),
            "actual_label": features[TARGET],
        }
    )


def model_metadata() -> dict:
    """Return static model metadata for the sample review."""
    return {
        "model_name": "qsr_purchase_predictor_v3",
        "artifact_profile": "qsr_purchase_propensity_sample",
        "domain": "synthetic_purchase_analytics",
        "supports_generic_contract": True,
        "model_type": "classification",
        "target": TARGET,
        "feature_columns": FEATURE_COLUMNS,
        "business_use_case": "QSR audience targeting and campaign optimization",
        "decision_supported": "prioritize, suppress, or recalibrate synthetic QSR audience activation",
        "entity_id": ENTITY_ID,
        "prediction_column": "propensity_score",
        "training_window": "2026-01-01_to_2026-03-31_synthetic",
        "current_window": "2026-04-01_to_2026-04-30_synthetic",
        "train_auc": 0.812,
        "validation_auc": 0.776,
        "current_auc": 0.741,
        "precision": 0.64,
        "recall": 0.58,
        "f1": 0.61,
    }


def feature_metadata() -> dict:
    """Return business definitions for the synthetic sample features."""
    definitions = {
        "qsr_txn_count_30d": "Number of synthetic QSR transactions in the last 30 days.",
        "qsr_spend_30d": "Total synthetic QSR spend in the last 30 days.",
        "qsr_recency_days": "Days since the synthetic consumer's most recent QSR transaction.",
        "competitor_qsr_share_90d": "Share of synthetic QSR activity with competitor QSR merchants over 90 days.",
        "grocery_spend_30d": "Total synthetic grocery spend in the last 30 days.",
        "fuel_spend_30d": "Total synthetic fuel spend in the last 30 days.",
        "weekend_dining_frequency": "Count of synthetic weekend dining events in the last 30 days.",
        "merchant_novelty_rate": "Rate of synthetic dining activity at merchants not previously observed for the consumer.",
        "campaign_exposed": "Indicator that the synthetic consumer was exposed to a QSR campaign.",
    }
    return {
        "entity_id": ENTITY_ID,
        "target": TARGET,
        "artifact_profile": "qsr_purchase_propensity_sample",
        "features": [
            {
                "name": name,
                "type": "numeric",
                "business_definition": definition,
                "synthetic_only": True,
            }
            for name, definition in definitions.items()
        ],
    }


def write_json(path: Path, payload: dict) -> None:
    """Write JSON with stable formatting."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """Generate sample feature, prediction, and metadata artifacts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_features = build_feature_frame(RANDOM_SEED, drift=False)
    current_features = build_feature_frame(RANDOM_SEED + 1, drift=True)
    train_predictions = build_predictions(train_features, RANDOM_SEED + 2)
    current_predictions = build_predictions(current_features, RANDOM_SEED + 3, score_shift=-0.08)

    output_paths = {
        "train_features": DATA_DIR / "train_features_sample.csv",
        "current_features": DATA_DIR / "current_features_sample.csv",
        "train_predictions": DATA_DIR / "train_predictions_sample.csv",
        "current_predictions": DATA_DIR / "current_predictions_sample.csv",
        "model_metadata": MODELS_DIR / "model_metadata.json",
        "feature_metadata": MODELS_DIR / "feature_metadata.json",
    }

    train_features.to_csv(output_paths["train_features"], index=False)
    current_features.to_csv(output_paths["current_features"], index=False)
    train_predictions.to_csv(output_paths["train_predictions"], index=False)
    current_predictions.to_csv(output_paths["current_predictions"], index=False)
    write_json(output_paths["model_metadata"], model_metadata())
    write_json(output_paths["feature_metadata"], feature_metadata())

    print("Created synthetic tabular model-review sample artifacts:")
    print("- profile: qsr_purchase_propensity_sample")
    print(f"- {output_paths['train_features']} shape={train_features.shape}")
    print(f"- {output_paths['current_features']} shape={current_features.shape}")
    print(f"- {output_paths['train_predictions']} shape={train_predictions.shape}")
    print(f"- {output_paths['current_predictions']} shape={current_predictions.shape}")
    print(f"- {output_paths['model_metadata']}")
    print(f"- {output_paths['feature_metadata']}")


if __name__ == "__main__":
    main()
