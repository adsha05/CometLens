"""Segment-level performance diagnostics for binary classification outputs."""

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

from src.diagnostics.calibration import calculate_brier_score, resolve_prediction_columns


def _clean_joined_frame(
    current_features: pd.DataFrame,
    predictions: pd.DataFrame,
    entity_id_col: str | None,
    prediction_column: str | None,
    label_column: str | None = None,
) -> tuple[pd.DataFrame, str | None, str | None]:
    """Join current features to predictions and keep valid binary score rows."""
    score_col, label_col = resolve_prediction_columns(predictions, prediction_column, label_column)
    if not score_col or not label_col:
        return pd.DataFrame(), score_col, label_col

    if entity_id_col and entity_id_col in current_features.columns and entity_id_col in predictions.columns:
        frame = current_features.merge(
            predictions[[entity_id_col, score_col, label_col]],
            on=entity_id_col,
            how="inner",
        )
    else:
        frame = current_features.reset_index(drop=True).join(
            predictions[[score_col, label_col]].reset_index(drop=True),
            how="inner",
        )

    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    frame[label_col] = pd.to_numeric(frame[label_col], errors="coerce")
    frame = frame.dropna(subset=[score_col, label_col])
    frame = frame.loc[frame[label_col].isin([0, 1])].copy()
    frame[score_col] = frame[score_col].clip(0.0, 1.0)
    frame[label_col] = frame[label_col].astype(int)
    return frame, score_col, label_col


def _segment_series(values: pd.Series, bins: int) -> pd.Series:
    """Create stable segment labels for numeric or low-cardinality feature values."""
    clean = pd.to_numeric(values, errors="coerce")
    unique_count = clean.dropna().nunique()
    if unique_count <= 2:
        return clean.map(lambda value: f"value={int(value)}" if pd.notna(value) else "missing")
    try:
        return pd.qcut(clean, q=min(bins, unique_count), duplicates="drop").astype(str)
    except ValueError:
        return pd.cut(clean, bins=min(bins, unique_count), duplicates="drop").astype(str)


def _risk_level(abs_gap: float, count: int, min_count: int, config: dict[str, Any]) -> tuple[str, str]:
    """Classify segment risk from calibration gap and minimum sample size."""
    thresholds = config.get("varuna", {}).get("segment_performance_thresholds", {})
    medium_gap = float(thresholds.get("calibration_gap_medium", 0.10))
    high_gap = float(thresholds.get("calibration_gap_high", 0.20))
    if count < min_count:
        return "Low", f"Low confidence because segment count {count} is below minimum {min_count}."
    if abs_gap >= high_gap:
        return "High", f"High because absolute calibration gap {abs_gap:.3f} is at or above {high_gap:.3f}."
    if abs_gap >= medium_gap:
        return "Medium", f"Medium because absolute calibration gap {abs_gap:.3f} is at or above {medium_gap:.3f}."
    return "Low", "Low because segment calibration gap did not cross configured thresholds."


def build_segment_performance_report(
    current_features: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_cols: list[str],
    entity_id_col: str | None,
    prediction_column: str | None,
    label_column: str | None = None,
    config: dict[str, Any] | None = None,
    bins: int = 4,
    max_features: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build segment-level score, outcome, lift, and calibration diagnostics."""
    config = config or {}
    thresholds = config.get("varuna", {}).get("segment_performance_thresholds", {})
    min_count = int(thresholds.get("minimum_segment_count", 10))
    frame, score_col, label_col = _clean_joined_frame(
        current_features,
        predictions,
        entity_id_col,
        prediction_column,
        label_column,
    )
    columns = [
        "feature",
        "segment",
        "count",
        "count_share",
        "avg_score",
        "actual_rate",
        "calibration_gap",
        "abs_calibration_gap",
        "lift_vs_overall",
        "brier_score",
        "risk_level",
        "risk_reason",
    ]
    if frame.empty or not score_col or not label_col:
        return pd.DataFrame(columns=columns), {
            "available": False,
            "reason": "Segment performance diagnostics require current features, prediction scores, and binary actual labels.",
            "prediction_column": score_col,
            "label_column": label_col,
            "features_evaluated": [],
            "high_risk_segments": 0,
            "medium_risk_segments": 0,
        }

    candidate_features = [
        feature
        for feature in feature_cols
        if feature in frame.columns and pd.api.types.is_numeric_dtype(frame[feature])
    ][:max_features]
    if not candidate_features:
        return pd.DataFrame(columns=columns), {
            "available": False,
            "reason": "No numeric segment features were available for segment performance diagnostics.",
            "prediction_column": score_col,
            "label_column": label_col,
            "features_evaluated": [],
            "high_risk_segments": 0,
            "medium_risk_segments": 0,
        }

    overall_rate = float(frame[label_col].mean())
    rows: list[dict[str, Any]] = []
    for feature in candidate_features:
        segmented = frame.assign(_segment=_segment_series(frame[feature], bins))
        grouped = (
            segmented.groupby("_segment", observed=False)
            .agg(
                count=(score_col, "size"),
                avg_score=(score_col, "mean"),
                actual_rate=(label_col, "mean"),
            )
            .reset_index()
        )
        for row in grouped.to_dict(orient="records"):
            segment_frame = segmented.loc[segmented["_segment"] == row["_segment"]]
            calibration_gap = float(row["avg_score"] - row["actual_rate"])
            abs_gap = abs(calibration_gap)
            risk_level, risk_reason = _risk_level(abs_gap, int(row["count"]), min_count, config)
            rows.append(
                {
                    "feature": feature,
                    "segment": str(row["_segment"]),
                    "count": int(row["count"]),
                    "count_share": float(row["count"] / len(frame)),
                    "avg_score": float(row["avg_score"]),
                    "actual_rate": float(row["actual_rate"]),
                    "calibration_gap": calibration_gap,
                    "abs_calibration_gap": abs_gap,
                    "lift_vs_overall": float(row["actual_rate"] / overall_rate) if overall_rate else np.nan,
                    "brier_score": calculate_brier_score(segment_frame[label_col], segment_frame[score_col]),
                    "risk_level": risk_level,
                    "risk_reason": risk_reason,
                }
            )

    report = pd.DataFrame(rows, columns=columns).sort_values(
        ["risk_level", "abs_calibration_gap"],
        ascending=[True, False],
    )
    risk_order = {"High": 0, "Medium": 1, "Low": 2}
    report = (
        report.assign(_risk_order=report["risk_level"].map(risk_order))
        .sort_values(["_risk_order", "abs_calibration_gap"], ascending=[True, False])
        .drop(columns="_risk_order")
        .reset_index(drop=True)
    )
    summary = {
        "available": True,
        "prediction_column": score_col,
        "label_column": label_col,
        "row_count": int(len(frame)),
        "features_evaluated": candidate_features,
        "minimum_segment_count": min_count,
        "high_risk_segments": int((report["risk_level"] == "High").sum()),
        "medium_risk_segments": int((report["risk_level"] == "Medium").sum()),
        "max_abs_calibration_gap": float(report["abs_calibration_gap"].max()) if not report.empty else 0.0,
    }
    return report, summary


def save_segment_performance_heatmap(segment_report: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a compact feature-by-segment calibration-gap heatmap."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    if not segment_report.empty:
        top_features = (
            segment_report.groupby("feature")["abs_calibration_gap"]
            .max()
            .sort_values(ascending=False)
            .head(5)
            .index
        )
        plot_frame = segment_report.loc[segment_report["feature"].isin(top_features)].copy()
        plot_frame["segment_label"] = plot_frame.groupby("feature").cumcount().map(lambda index: f"S{index + 1}")
        heatmap = plot_frame.pivot(index="feature", columns="segment_label", values="abs_calibration_gap").fillna(0.0)
        image = plt.imshow(heatmap.to_numpy(), aspect="auto", cmap="Reds")
        plt.colorbar(image, label="Absolute calibration gap")
        plt.xticks(range(len(heatmap.columns)), heatmap.columns)
        plt.yticks(range(len(heatmap.index)), heatmap.index)
    plt.title("Segment Performance Risk")
    plt.xlabel("Segment")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path
