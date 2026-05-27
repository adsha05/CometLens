"""Generate synthetic feature snapshots for QSR purchase prediction."""

from pathlib import Path

import numpy as np
import pandas as pd

N_USERS = 10_000
RANDOM_SEED = 42
TARGET_COLUMN = "purchase_qsr_next_30d"
FEATURE_COLUMNS = [
    "qsr_txn_count_30d",
    "qsr_spend_30d",
    "qsr_recency_days",
    "competitor_qsr_share_90d",
    "grocery_spend_30d",
    "fuel_spend_30d",
    "weekend_dining_frequency",
    "avg_ticket_size",
    "dining_category_entropy",
    "merchant_novelty_rate",
    "campaign_exposed",
    "income_band_encoded",
    "age_band_encoded",
    "region_encoded",
    "prior_offer_redemption_rate",
]


def generate_user_profiles(n_users: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create stable synthetic user attributes and unobserved QSR affinity."""
    return pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1),
            "qsr_affinity": rng.beta(2.5, 2.2, n_users),
            "income_band_encoded": rng.choice(4, n_users, p=[0.22, 0.34, 0.29, 0.15]),
            "age_band_encoded": rng.choice(5, n_users, p=[0.16, 0.25, 0.25, 0.21, 0.13]),
            "region_encoded": rng.choice(4, n_users, p=[0.24, 0.25, 0.31, 0.20]),
        }
    )


def generate_feature_snapshot(
    profiles: pd.DataFrame, rng: np.random.Generator, apply_drift: bool = False
) -> pd.DataFrame:
    """Generate one behavioral snapshot, optionally shifted for drift analysis."""
    n_users = len(profiles)
    affinity = profiles["qsr_affinity"].to_numpy()
    income = profiles["income_band_encoded"].to_numpy()
    region = profiles["region_encoded"].to_numpy()

    qsr_count = rng.poisson(0.8 + (5.5 * affinity) + (0.12 * income))
    avg_ticket = np.clip(
        rng.normal(7.5 + (10.0 * affinity) + (1.4 * income), 3.0, n_users),
        4.0,
        45.0,
    )
    qsr_spend = np.clip(
        (qsr_count * avg_ticket) + rng.normal(0.0, 5.0, n_users), 0.0, None
    )
    qsr_recency = np.clip(
        rng.exponential(50.0 / (qsr_count + 1), n_users), 0.0, 90.0
    ).round()
    competitor_share = rng.beta(2.7 - affinity, 2.0 + (2.8 * affinity), n_users)
    grocery_spend = rng.gamma(3.2 + (0.4 * income), 32.0, n_users)
    fuel_spend = rng.gamma(2.2 + (0.25 * region), 24.0, n_users)
    weekend_frequency = rng.poisson(0.25 + (2.6 * affinity), n_users)
    dining_entropy = np.clip(
        rng.normal(1.45 - (0.20 * affinity), 0.28, n_users), 0.05, 2.5
    )
    merchant_novelty = rng.beta(1.5 + (0.5 * affinity), 4.2, n_users)
    campaign_exposed = rng.binomial(1, 0.34, n_users)
    redemption_rate = rng.beta(
        1.0 + (1.8 * affinity) + (0.5 * campaign_exposed), 4.2, n_users
    )

    if apply_drift:
        fuel_spend *= rng.normal(1.24, 0.04, n_users)
        weekend_frequency = np.maximum(
            weekend_frequency - rng.binomial(1, 0.52, n_users), 0
        )
        merchant_novelty = np.clip(merchant_novelty + rng.normal(0.09, 0.02, n_users), 0, 1)
        competitor_share = np.clip(competitor_share + rng.normal(0.04, 0.01, n_users), 0, 1)

    features = pd.DataFrame(
        {
            "user_id": profiles["user_id"],
            "qsr_txn_count_30d": qsr_count,
            "qsr_spend_30d": qsr_spend.round(2),
            "qsr_recency_days": qsr_recency.astype(int),
            "competitor_qsr_share_90d": competitor_share.round(4),
            "grocery_spend_30d": grocery_spend.round(2),
            "fuel_spend_30d": fuel_spend.round(2),
            "weekend_dining_frequency": weekend_frequency,
            "avg_ticket_size": avg_ticket.round(2),
            "dining_category_entropy": dining_entropy.round(4),
            "merchant_novelty_rate": merchant_novelty.round(4),
            "campaign_exposed": campaign_exposed,
            "income_band_encoded": profiles["income_band_encoded"],
            "age_band_encoded": profiles["age_band_encoded"],
            "region_encoded": profiles["region_encoded"],
            "prior_offer_redemption_rate": redemption_rate.round(4),
        }
    )
    return add_target(features, rng)


def add_target(features: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Draw the next-30-day purchase target from behavior-driven probabilities."""
    logits = (
        -3.15
        + (0.24 * features["qsr_txn_count_30d"])
        + (0.010 * features["qsr_spend_30d"])
        - (0.020 * features["qsr_recency_days"])
        - (0.90 * features["competitor_qsr_share_90d"])
        + (0.16 * features["weekend_dining_frequency"])
        + (0.30 * features["campaign_exposed"])
        + (0.70 * features["prior_offer_redemption_rate"])
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    result = features.copy()
    result[TARGET_COLUMN] = rng.binomial(1, probabilities)
    return result[["user_id", *FEATURE_COLUMNS, TARGET_COLUMN]]


def generate_datasets(n_users: int = N_USERS, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train and drifted current feature datasets for synthetic users."""
    rng = np.random.default_rng(seed)
    profiles = generate_user_profiles(n_users, rng)
    train_features = generate_feature_snapshot(profiles, rng)
    current_features = generate_feature_snapshot(profiles, rng, apply_drift=True)
    return train_features, current_features


def main() -> None:
    """Generate synthetic CSV snapshots and print basic dataset statistics."""
    train_features, current_features = generate_datasets()
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_features.to_csv(data_dir / "train_features.csv", index=False)
    current_features.to_csv(data_dir / "current_features.csv", index=False)

    print(f"Saved synthetic feature datasets to {data_dir}")
    print(
        "train_features.csv: "
        f"shape={train_features.shape}, target_rate={train_features[TARGET_COLUMN].mean():.3%}"
    )
    print(
        "current_features.csv: "
        f"shape={current_features.shape}, target_rate={current_features[TARGET_COLUMN].mean():.3%}"
    )


if __name__ == "__main__":
    main()
