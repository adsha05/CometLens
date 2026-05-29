"""Artifact contract validation for AxionAI pipeline runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class ArtifactValidationError(ValueError):
    """Raised when model-review artifacts do not satisfy the expected contract."""


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON file and ensure it contains an object."""
    if not path.exists():
        raise ArtifactValidationError(f"Missing required artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactValidationError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected JSON object in {path}.")
    return payload


def require_keys(payload: dict[str, Any], keys: list[str], source: Path) -> None:
    """Validate that required keys are present and non-empty."""
    missing = []
    for key in keys:
        value = payload.get(key)
        if value is None or value == "" or value == []:
            missing.append(key)
    if missing:
        raise ArtifactValidationError(f"{source} is missing required field(s): {', '.join(missing)}")


def validate_input_contract(
    train_features_path: Path,
    current_features_path: Path,
    predictions_path: Path,
    model_metadata_path: Path,
    feature_metadata_path: Path,
) -> dict[str, Any]:
    """Validate the minimum tabular artifact contract before agents run."""
    model_metadata = load_json_object(model_metadata_path)
    feature_metadata = load_json_object(feature_metadata_path)
    require_keys(
        model_metadata,
        ["model_name", "model_type", "target", "entity_id", "prediction_column", "feature_columns"],
        model_metadata_path,
    )

    for path in [train_features_path, current_features_path, predictions_path]:
        if not path.exists():
            raise ArtifactValidationError(f"Missing required artifact: {path}")

    try:
        train_columns = set(pd.read_csv(train_features_path, nrows=5).columns)
        current_columns = set(pd.read_csv(current_features_path, nrows=5).columns)
        prediction_columns = set(pd.read_csv(predictions_path, nrows=5).columns)
    except Exception as error:
        raise ArtifactValidationError(f"Could not read input CSV artifacts: {error}") from error

    target = str(model_metadata["target"])
    entity_id = str(model_metadata["entity_id"])
    prediction_column = str(model_metadata["prediction_column"])
    feature_columns = [str(feature) for feature in model_metadata.get("feature_columns", [])]
    if not feature_columns:
        raise ArtifactValidationError(f"{model_metadata_path} must define at least one feature column.")

    missing_train = [column for column in [entity_id, target, *feature_columns] if column not in train_columns]
    missing_current = [column for column in [entity_id, *feature_columns] if column not in current_columns]
    missing_predictions = [column for column in [entity_id, prediction_column] if column not in prediction_columns]

    messages = []
    if missing_train:
        messages.append(f"{train_features_path} missing: {', '.join(missing_train)}")
    if missing_current:
        messages.append(f"{current_features_path} missing: {', '.join(missing_current)}")
    if missing_predictions:
        messages.append(f"{predictions_path} missing: {', '.join(missing_predictions)}")
    if messages:
        raise ArtifactValidationError("Artifact schema mismatch. " + " | ".join(messages))

    metadata_feature_names = {
        str(feature.get("name"))
        for feature in feature_metadata.get("features", [])
        if isinstance(feature, dict) and feature.get("name")
    }
    missing_feature_definitions = [feature for feature in feature_columns if feature not in metadata_feature_names]
    if missing_feature_definitions:
        raise ArtifactValidationError(
            f"{feature_metadata_path} missing feature definition(s): "
            + ", ".join(missing_feature_definitions)
        )

    return {
        "model_name": model_metadata["model_name"],
        "target": target,
        "entity_id": entity_id,
        "prediction_column": prediction_column,
        "feature_count": len(feature_columns),
    }
