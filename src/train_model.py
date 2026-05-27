"""Train and evaluate the QSR purchase-propensity classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

TARGET_COLUMN = "purchase_qsr_next_30d"
ID_COLUMN = "user_id"
RANDOM_SEED = 42
VALIDATION_SIZE = 0.20
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def load_feature_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load generated training and current-period feature snapshots."""
    train_path = DATA_DIR / "train_features.csv"
    current_path = DATA_DIR / "current_features.csv"
    if not train_path.exists() or not current_path.exists():
        raise FileNotFoundError(
            "Feature CSVs not found. Run `python src/generate_synthetic_data.py` first."
        )

    train_features = pd.read_csv(train_path)
    current_features = pd.read_csv(current_path)
    for name, data in (("train", train_features), ("current", current_features)):
        missing_columns = {ID_COLUMN, TARGET_COLUMN} - set(data.columns)
        if missing_columns:
            raise ValueError(f"{name} features missing required columns: {sorted(missing_columns)}")
    return train_features, current_features


def select_feature_columns(train_features: pd.DataFrame, current_features: pd.DataFrame) -> list[str]:
    """Select numeric predictor columns shared by both datasets."""
    excluded = {ID_COLUMN, TARGET_COLUMN}
    feature_columns = [
        column
        for column in train_features.select_dtypes(include="number").columns
        if column not in excluded
    ]
    if not feature_columns:
        raise ValueError("No numeric model features found in training dataset.")

    missing_current = set(feature_columns) - set(current_features.columns)
    if missing_current:
        raise ValueError(f"Current features missing predictors: {sorted(missing_current)}")
    non_numeric_current = [
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(current_features[column])
    ]
    if non_numeric_current:
        raise ValueError(f"Current predictors must be numeric: {non_numeric_current}")
    return feature_columns


def create_model() -> XGBClassifier:
    """Configure the initial purchase-propensity classifier."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        n_jobs=2,
    )


def calculate_metrics(y_true: pd.Series, y_pred_proba: Any) -> dict[str, float]:
    """Calculate binary-classification metrics from predicted probabilities."""
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_pred_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_predictions(data: pd.DataFrame, y_pred_proba: Any, path: Path) -> None:
    """Persist user-level observed outcomes and predicted probabilities."""
    predictions = pd.DataFrame(
        {
            ID_COLUMN: data[ID_COLUMN],
            "y_true": data[TARGET_COLUMN],
            "y_pred_proba": y_pred_proba,
        }
    )
    predictions.to_csv(path, index=False)


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    """Print a compact metric summary."""
    values = ", ".join(f"{metric}={value:.4f}" for metric, value in metrics.items())
    print(f"{name} metrics: {values}")


def main() -> None:
    """Train the model, evaluate snapshots, and save reusable artifacts."""
    train_features, current_features = load_feature_data()
    feature_columns = select_feature_columns(train_features, current_features)

    train_split, validation_split = train_test_split(
        train_features,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=train_features[TARGET_COLUMN],
    )
    model = create_model()
    model.fit(train_split[feature_columns], train_split[TARGET_COLUMN])

    validation_proba = model.predict_proba(validation_split[feature_columns])[:, 1]
    current_proba = model.predict_proba(current_features[feature_columns])[:, 1]
    train_proba = model.predict_proba(train_features[feature_columns])[:, 1]

    metrics = {
        "validation": calculate_metrics(validation_split[TARGET_COLUMN], validation_proba),
        "current": calculate_metrics(current_features[TARGET_COLUMN], current_proba),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "qsr_xgb_model.joblib")
    save_predictions(train_features, train_proba, DATA_DIR / "train_predictions.csv")
    save_predictions(current_features, current_proba, DATA_DIR / "current_predictions.csv")

    metadata = {
        "model_name": "qsr_purchase_propensity_xgb",
        "target": TARGET_COLUMN,
        "model_type": "XGBClassifier",
        "training_window": "synthetic_training_snapshot",
        "current_window": "synthetic_current_snapshot",
        "feature_list": feature_columns,
        "metrics": metrics,
    }
    with (MODELS_DIR / "model_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"Trained model using {len(feature_columns)} numeric features.")
    print_metrics("Validation", metrics["validation"])
    print_metrics("Current", metrics["current"])
    print(f"Saved model artifacts to {MODELS_DIR}")
    print(f"Saved prediction files to {DATA_DIR}")


if __name__ == "__main__":
    main()
