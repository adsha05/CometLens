"""Focused tests for Agent 02: Varuna deterministic model diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agents.model_lens_agent import ModelLensAgent
from src.agents.signal_sentinel_agent import SignalSentinelAgent
from src.diagnostics.multicollinearity import calculate_vif
from src.diagnostics.overfitting import calculate_overfitting_delta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


@pytest.fixture(scope="module", autouse=True)
def generated_varuna_outputs() -> dict[str, Path]:
    """Generate deterministic Mitra and Varuna outputs once for this module."""
    SignalSentinelAgent().save_outputs()
    return ModelLensAgent().save_outputs()


def test_varuna_runs_successfully_after_mitra(generated_varuna_outputs: dict[str, Path]) -> None:
    """Varuna should create its complete output set after Mitra runs."""
    assert generated_varuna_outputs["json"].exists()
    assert generated_varuna_outputs["model_diagnostics"].exists()
    assert generated_varuna_outputs["feature_risk_matrix"].exists()


def test_shap_global_importance_contract() -> None:
    """SHAP output should expose the requested stable columns."""
    frame = pd.read_csv(REPORTS_DIR / "shap_global_importance.csv")
    assert {"feature", "mean_abs_shap", "shap_rank"}.issubset(frame.columns)


def test_vif_report_contract() -> None:
    """VIF output should expose config-driven risk labels."""
    frame = pd.read_csv(REPORTS_DIR / "vif_report.csv")
    assert {"feature", "vif", "vif_risk"}.issubset(frame.columns)


def test_feature_risk_matrix_contract() -> None:
    """Feature risk matrix should contain actionable deterministic output."""
    frame = pd.read_csv(REPORTS_DIR / "feature_risk_matrix.csv")
    assert {"feature", "final_risk", "recommended_action"}.issubset(frame.columns)


def test_model_lens_output_identifies_varuna() -> None:
    """Main JSON output should use the Varuna agent identity."""
    payload = json.loads((REPORTS_DIR / "model_lens_output.json").read_text(encoding="utf-8"))
    assert payload["agent"] == "Varuna"


def test_overfitting_thresholds_are_config_driven() -> None:
    """Changing config thresholds should change overfitting classification."""
    metadata = {"train_auc": 0.82, "validation_auc": 0.77}
    strict = {"varuna": {"overfitting_delta_thresholds": {"medium": 0.08, "high": 0.12}}}
    sensitive = {"varuna": {"overfitting_delta_thresholds": {"medium": 0.01, "high": 0.04}}}
    assert calculate_overfitting_delta(metadata, strict)["risk_level"] == "Low"
    assert calculate_overfitting_delta(metadata, sensitive)["risk_level"] == "High"


def test_vif_thresholds_are_config_driven() -> None:
    """Changing config thresholds should change VIF risk labels."""
    rng = np.random.default_rng(42)
    feature_a = rng.normal(size=100)
    frame = pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_a * 0.80 + rng.normal(scale=0.35, size=100),
            "feature_c": rng.normal(size=100),
        }
    )
    strict = {"varuna": {"vif_thresholds": {"medium": 1000.0, "high": 2000.0}}}
    sensitive = {"varuna": {"vif_thresholds": {"medium": 1.0, "high": 1.1}}}
    assert set(calculate_vif(frame, list(frame.columns), strict)["vif_risk"]) == {"Low"}
    assert "High" in set(calculate_vif(frame, list(frame.columns), sensitive)["vif_risk"])
