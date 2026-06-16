"""Lift, gains, and score-decile diagnostics for binary classifiers."""

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

from src.diagnostics.calibration import resolve_prediction_columns


def _clean_predictions(
    predictions: pd.DataFrame,
    prediction_column: str | None,
    label_column: str | None = None,
) -> tuple[pd.DataFrame, str | None, str | None]:
    """Return rows with numeric score and binary label values."""
    score_col, label_col = resolve_prediction_columns(predictions, prediction_column, label_column)
    if not score_col or not label_col:
        return pd.DataFrame(), score_col, label_col
    frame = predictions[[score_col, label_col]].copy()
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    frame[label_col] = pd.to_numeric(frame[label_col], errors="coerce")
    frame = frame.dropna()
    frame = frame.loc[frame[label_col].isin([0, 1])]
    frame[label_col] = frame[label_col].astype(int)
    return frame, score_col, label_col


def build_score_decile_report(
    predictions: pd.DataFrame,
    prediction_column: str | None,
    label_column: str | None = None,
    deciles: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build descending-score decile, lift, and cumulative gains diagnostics."""
    frame, score_col, label_col = _clean_predictions(predictions, prediction_column, label_column)
    columns = [
        "decile",
        "count",
        "score_min",
        "score_max",
        "avg_score",
        "positives",
        "actual_rate",
        "lift",
        "cumulative_positives",
        "cumulative_capture_rate",
        "cumulative_lift",
    ]
    if frame.empty or not score_col or not label_col:
        return pd.DataFrame(columns=columns), {
            "available": False,
            "reason": "Lift diagnostics require prediction scores and binary actual labels.",
            "prediction_column": score_col,
            "label_column": label_col,
            "row_count": 0,
            "baseline_rate": None,
            "top_decile_lift": None,
        }

    frame = frame.sort_values(score_col, ascending=False).reset_index(drop=True)
    decile_count = max(1, min(deciles, len(frame)))
    frame["_decile"] = np.floor(np.arange(len(frame)) * decile_count / len(frame)).astype(int) + 1
    baseline_rate = float(frame[label_col].mean())
    total_positives = int(frame[label_col].sum())
    report = (
        frame.groupby("_decile", observed=False)
        .agg(
            count=(score_col, "size"),
            score_min=(score_col, "min"),
            score_max=(score_col, "max"),
            avg_score=(score_col, "mean"),
            positives=(label_col, "sum"),
            actual_rate=(label_col, "mean"),
        )
        .reset_index()
        .rename(columns={"_decile": "decile"})
    )
    report["lift"] = report["actual_rate"] / baseline_rate if baseline_rate else np.nan
    report["cumulative_positives"] = report["positives"].cumsum()
    report["cumulative_capture_rate"] = (
        report["cumulative_positives"] / total_positives if total_positives else np.nan
    )
    report["cumulative_lift"] = report["cumulative_capture_rate"] / (report["decile"] / decile_count)
    summary = {
        "available": True,
        "prediction_column": score_col,
        "label_column": label_col,
        "row_count": int(len(frame)),
        "baseline_rate": baseline_rate,
        "top_decile_lift": float(report.loc[report["decile"] == 1, "lift"].iloc[0]),
        "top_decile_actual_rate": float(report.loc[report["decile"] == 1, "actual_rate"].iloc[0]),
        "top_decile_capture_rate": float(report.loc[report["decile"] == 1, "cumulative_capture_rate"].iloc[0]),
    }
    return report[columns], summary


def save_lift_chart(lift_report: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a lift chart and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 5.2))
    if not lift_report.empty:
        plt.plot(lift_report["decile"], lift_report["lift"], marker="o", color="#3855ff", label="Decile lift")
        plt.plot(
            lift_report["decile"],
            lift_report["cumulative_lift"],
            marker="s",
            color="#16a34a",
            label="Cumulative lift",
        )
    plt.axhline(1.0, linestyle="--", color="#6b7280", label="Baseline")
    plt.xlabel("Score decile, descending")
    plt.ylabel("Lift")
    plt.title("Lift by Score Decile")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path
