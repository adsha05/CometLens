"""Deterministic multicollinearity diagnostics for Agent 02: Varuna."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "calibration_config_v1.json"


def _load_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Load default config when one is not supplied."""
    if config is not None:
        return config
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def _thresholds(config: dict[str, Any]) -> tuple[float, float]:
    """Read VIF thresholds while supporting the previous flat config shape."""
    varuna = config.get("varuna", {})
    nested = varuna.get("vif_thresholds", {})
    return (
        float(nested.get("medium", varuna.get("vif_medium", 5.0))),
        float(nested.get("high", varuna.get("vif_high", 10.0))),
    )


def _risk(vif: float, medium: float, high: float) -> tuple[str, str]:
    """Return VIF risk level and deterministic explanation."""
    if vif >= high:
        return "High", f"High because VIF {vif:.2f} is at or above configured threshold {high:.2f}."
    if vif >= medium:
        return "Medium", f"Medium because VIF {vif:.2f} is at or above configured threshold {medium:.2f}."
    return "Low", f"Low because VIF {vif:.2f} is below configured threshold {medium:.2f}."


def _sklearn_vif(frame: pd.DataFrame, feature: str) -> float:
    """Calculate VIF using sklearn LinearRegression."""
    y = frame[feature]
    others = frame.drop(columns=[feature])
    if others.empty or float(y.var()) == 0.0:
        return float("inf") if float(y.var()) == 0.0 else 1.0
    model = LinearRegression()
    model.fit(others, y)
    r_squared = float(model.score(others, y))
    return float("inf") if r_squared >= 0.999999 else 1.0 / (1.0 - r_squared)


def calculate_vif(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Calculate VIF using statsmodels when installed, otherwise sklearn."""
    loaded_config = _load_config(config)
    medium, high = _thresholds(loaded_config)
    frame = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    frame = frame.fillna(frame.median(numeric_only=True)).fillna(0.0)
    rows = []

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        use_statsmodels = True
    except ImportError:
        use_statsmodels = False

    for index, feature in enumerate(feature_cols):
        if float(frame[feature].var()) == 0.0:
            vif = float("inf")
        elif use_statsmodels:
            try:
                vif = float(variance_inflation_factor(frame.to_numpy(dtype=float), index))
            except Exception:
                vif = _sklearn_vif(frame, feature)
        else:
            vif = _sklearn_vif(frame, feature)
        risk, reason = _risk(vif, medium, high)
        rows.append({"feature": feature, "vif": vif, "vif_risk": risk, "vif_risk_reason": reason})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)
