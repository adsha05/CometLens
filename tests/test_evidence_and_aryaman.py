"""Focused tests for Phase 4 Evidence Store and Agent 03: Aryaman."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.evidence_store import EvidenceStoreBuilder
from src.agents.executive_synthesis_agent import ExecutiveSynthesisAgent
from src.agents.model_lens_agent import ModelLensAgent
from src.agents.signal_sentinel_agent import SignalSentinelAgent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


@pytest.fixture(scope="module", autouse=True)
def generated_phase_four_outputs() -> None:
    """Generate deterministic upstream and Phase 4 outputs once."""
    SignalSentinelAgent().save_outputs()
    ModelLensAgent().save_outputs()
    EvidenceStoreBuilder().save()
    ExecutiveSynthesisAgent().save_outputs()


def test_evidence_packet_is_created() -> None:
    """Evidence Store should write evidence_packet.json."""
    assert (REPORTS_DIR / "evidence_packet.json").exists()


def test_evidence_packet_contains_required_sections() -> None:
    """Evidence packet should preserve audit and summary sections."""
    packet = json.loads((REPORTS_DIR / "evidence_packet.json").read_text(encoding="utf-8"))
    required = {
        "run_id",
        "timestamp",
        "config_version",
        "mitra_summary",
        "varuna_summary",
        "key_findings",
        "limitations",
    }
    assert required.issubset(packet)


def test_high_risk_features_exist_in_metadata_or_matrix() -> None:
    """Every high-risk feature should trace to feature metadata or the risk matrix."""
    packet = json.loads((REPORTS_DIR / "evidence_packet.json").read_text(encoding="utf-8"))
    metadata = json.loads((MODELS_DIR / "feature_metadata.json").read_text(encoding="utf-8"))
    metadata_features = {row["name"] for row in metadata["features"]}
    matrix_features = {row["feature"] for row in packet["feature_risk_matrix"]}
    for row in packet["high_risk_features"]:
        assert row["feature"] in metadata_features or row["feature"] in matrix_features


def test_executive_markdown_is_created() -> None:
    """Aryaman should create a Markdown brief."""
    assert (REPORTS_DIR / "executive_model_report.md").exists()


def test_aryaman_output_identifies_agent() -> None:
    """Aryaman JSON should use the required identity."""
    output = json.loads((REPORTS_DIR / "aryaman_output.json").read_text(encoding="utf-8"))
    assert output["agent"] == "Aryaman"


def test_report_contains_required_sections() -> None:
    """Markdown report should contain every requested consulting-style section."""
    report = (REPORTS_DIR / "executive_model_report.md").read_text(encoding="utf-8")
    for section in [
        "## 1. Executive Summary",
        "## 2. Model Health Status",
        "## 3. What Changed",
        "## 4. Why It Matters",
        "## 5. Top Model Drivers",
        "## 6. High-Risk Features",
        "## 7. Business Risks",
        "## 8. Recommended Actions",
        "## 9. Evidence Appendix",
        "## 10. Limitations",
    ]:
        assert section in report


def test_report_mentions_synthetic_data_limit() -> None:
    """Markdown report should disclose the synthetic data limit."""
    report = (REPORTS_DIR / "executive_model_report.md").read_text(encoding="utf-8")
    assert "Synthetic demo data only" in report


def test_report_avoids_forbidden_phrases() -> None:
    """Aryaman report should avoid unsupported certainty claims."""
    report = (REPORTS_DIR / "executive_model_report.md").read_text(encoding="utf-8").lower()
    for phrase in ["guarantees", "proves causality", "fully accurate", "production validated"]:
        assert phrase not in report
