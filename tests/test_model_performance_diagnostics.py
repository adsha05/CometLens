"""Focused tests for calibration, lift, and score-decile diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.agents.model_lens_agent import ModelLensAgent
from src.agents.signal_sentinel_agent import SignalSentinelAgent
from src.diagnostics.calibration import build_calibration_report, calculate_brier_score
from src.diagnostics.lift import build_score_decile_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def test_calibration_report_calculates_expected_columns() -> None:
    """Calibration report should expose bin-level prediction and outcome rates."""
    predictions = pd.DataFrame(
        {
            "score": [0.05, 0.15, 0.45, 0.55, 0.85, 0.95],
            "actual_label": [0, 0, 1, 0, 1, 1],
        }
    )
    report, summary = build_calibration_report(predictions, "score", "actual_label", bins=5)
    assert summary["available"] is True
    assert summary["row_count"] == 6
    assert {"avg_predicted_score", "actual_rate", "calibration_gap"}.issubset(report.columns)
    assert summary["expected_calibration_error"] >= 0
    assert calculate_brier_score(predictions["actual_label"], predictions["score"]) > 0


def test_lift_report_calculates_top_decile_lift() -> None:
    """Lift report should rank records by score and calculate cumulative gains."""
    predictions = pd.DataFrame(
        {
            "score": [0.95, 0.90, 0.80, 0.70, 0.40, 0.30, 0.20, 0.10],
            "actual_label": [1, 1, 1, 0, 0, 0, 0, 0],
        }
    )
    report, summary = build_score_decile_report(predictions, "score", "actual_label", deciles=4)
    assert summary["available"] is True
    assert summary["top_decile_lift"] > 1
    assert {"lift", "cumulative_capture_rate", "cumulative_lift"}.issubset(report.columns)


def test_missing_labels_return_unavailable_reports() -> None:
    """Diagnostics should not fail when actual labels are unavailable."""
    predictions = pd.DataFrame({"score": [0.1, 0.2, 0.3]})
    calibration_report, calibration_summary = build_calibration_report(predictions, "score", "actual_label")
    lift_report, lift_summary = build_score_decile_report(predictions, "score", "actual_label")
    assert calibration_report.empty
    assert lift_report.empty
    assert calibration_summary["available"] is False
    assert lift_summary["available"] is False


def test_varuna_writes_model_performance_artifacts() -> None:
    """Varuna should save calibration, lift, and score-decile reports."""
    SignalSentinelAgent().save_outputs()
    paths = ModelLensAgent().save_outputs()
    assert paths["calibration_report"].exists()
    assert paths["score_decile_report"].exists()
    assert paths["lift_report"].exists()
    assert paths["calibration_curve"].exists()
    assert paths["lift_chart"].exists()

    diagnostics = json.loads((REPORTS_DIR / "model_diagnostics.json").read_text(encoding="utf-8"))
    assert "performance_diagnostics" in diagnostics
