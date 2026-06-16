"""Calibration diagnostics for binary classification model outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cometlens-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_prediction_columns(
    predictions: pd.DataFrame,
    prediction_column: str | None = None,
    label_column: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve score and label columns from metadata and common prediction names."""
    score_candidates = [
        prediction_column,
        "propensity_score",
        "prediction_score",
        "score",
        "y_pred_proba",
    ]
    label_candidates = [
        label_column,
        "actual_label",
        "y_true",
        "target",
        "label",
    ]
    score_col = next((column for column in score_candidates if column and column in predictions.columns), None)
    label_col = next((column for column in label_candidates if column and column in predictions.columns), None)
    return score_col, label_col


def _clean_binary_predictions(
    predictions: pd.DataFrame,
    prediction_column: str | None,
    label_column: str | None = None,
) -> tuple[pd.DataFrame, str | None, str | None]:
    """Return numeric score/label rows for binary classification diagnostics."""
    score_col, label_col = resolve_prediction_columns(predictions, prediction_column, label_column)
    if not score_col or not label_col:
        return pd.DataFrame(), score_col, label_col
    frame = predictions[[score_col, label_col]].copy()
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    frame[label_col] = pd.to_numeric(frame[label_col], errors="coerce")
    frame = frame.dropna()
    frame = frame.loc[frame[label_col].isin([0, 1])]
    frame[score_col] = frame[score_col].clip(0.0, 1.0)
    frame[label_col] = frame[label_col].astype(int)
    return frame, score_col, label_col


def calculate_brier_score(y_true: pd.Series, y_score: pd.Series) -> float:
    """Calculate Brier score for binary labels and probability scores."""
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_score.to_numpy(dtype=float) - y_true.to_numpy(dtype=float)) ** 2))


def calculate_expected_calibration_error(calibration_report: pd.DataFrame) -> float:
    """Calculate weighted expected calibration error from a bin-level report."""
    if calibration_report.empty or "count" not in calibration_report:
        return float("nan")
    total = float(calibration_report["count"].sum())
    if total == 0:
        return float("nan")
    weighted_gap = calibration_report["abs_calibration_gap"] * calibration_report["count"]
    return float(weighted_gap.sum() / total)


def build_calibration_report(
    predictions: pd.DataFrame,
    prediction_column: str | None,
    label_column: str | None = None,
    bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a bin-level calibration report and summary diagnostics."""
    frame, score_col, label_col = _clean_binary_predictions(predictions, prediction_column, label_column)
    columns = [
        "bin",
        "score_min",
        "score_max",
        "count",
        "avg_predicted_score",
        "actual_rate",
        "calibration_gap",
        "abs_calibration_gap",
    ]
    if frame.empty or not score_col or not label_col:
        return pd.DataFrame(columns=columns), {
            "available": False,
            "reason": "Calibration diagnostics require prediction scores and binary actual labels.",
            "prediction_column": score_col,
            "label_column": label_col,
            "brier_score": None,
            "expected_calibration_error": None,
            "row_count": 0,
        }

    boundaries = np.linspace(0.0, 1.0, bins + 1)
    binned = pd.cut(frame[score_col], bins=boundaries, include_lowest=True, duplicates="drop")
    report = (
        frame.assign(_bin=binned)
        .groupby("_bin", observed=False)
        .agg(
            score_min=(score_col, "min"),
            score_max=(score_col, "max"),
            count=(score_col, "size"),
            avg_predicted_score=(score_col, "mean"),
            actual_rate=(label_col, "mean"),
        )
        .reset_index(drop=True)
    )
    report = report.loc[report["count"] > 0].reset_index(drop=True)
    report.insert(0, "bin", np.arange(1, len(report) + 1))
    report["calibration_gap"] = report["avg_predicted_score"] - report["actual_rate"]
    report["abs_calibration_gap"] = report["calibration_gap"].abs()
    brier_score = calculate_brier_score(frame[label_col], frame[score_col])
    ece = calculate_expected_calibration_error(report)
    summary = {
        "available": True,
        "prediction_column": score_col,
        "label_column": label_col,
        "row_count": int(len(frame)),
        "brier_score": brier_score,
        "expected_calibration_error": ece,
        "mean_predicted_score": float(frame[score_col].mean()),
        "actual_positive_rate": float(frame[label_col].mean()),
    }
    return report[columns], summary


def save_calibration_curve(calibration_report: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a calibration curve figure and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 5.5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Perfect calibration")
    if not calibration_report.empty:
        plt.plot(
            calibration_report["avg_predicted_score"],
            calibration_report["actual_rate"],
            marker="o",
            color="#3855ff",
            label="Observed",
        )
    plt.xlabel("Average predicted score")
    plt.ylabel("Actual positive rate")
    plt.title("Calibration Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path
