"""Deterministic train-validation overfitting diagnostics for Varuna."""

from __future__ import annotations

from typing import Any


def _thresholds(config: dict[str, Any]) -> tuple[float, float]:
    """Read overfitting thresholds while supporting the prior flat config shape."""
    varuna = config.get("varuna", {})
    nested = varuna.get("overfitting_delta_thresholds", {})
    return (
        float(nested.get("medium", varuna.get("overfit_delta_medium", 0.03))),
        float(nested.get("high", varuna.get("overfit_delta_high", 0.07))),
    )


def calculate_overfitting_delta(
    model_metadata: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Calculate train-validation AUC delta using config-driven thresholds."""
    if "train_auc" not in model_metadata or "validation_auc" not in model_metadata:
        return {
            "train_auc": model_metadata.get("train_auc"),
            "validation_auc": model_metadata.get("validation_auc"),
            "delta": None,
            "risk_level": "Not Available",
            "reason": "Not Available because train_auc and validation_auc are both required.",
        }

    train_auc = float(model_metadata["train_auc"])
    validation_auc = float(model_metadata["validation_auc"])
    delta = train_auc - validation_auc
    medium, high = _thresholds(config)
    if delta >= high:
        risk_level = "High"
        reason = f"High because train-validation AUC delta {delta:.4f} is at or above configured threshold {high:.4f}."
    elif delta >= medium:
        risk_level = "Medium"
        reason = (
            f"Medium because train-validation AUC delta {delta:.4f} is at or above configured threshold {medium:.4f}."
        )
    else:
        risk_level = "Low"
        reason = f"Low because train-validation AUC delta {delta:.4f} is below configured threshold {medium:.4f}."
    return {
        "train_auc": train_auc,
        "validation_auc": validation_auc,
        "delta": delta,
        "risk_level": risk_level,
        "reason": reason,
    }
