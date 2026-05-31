"""Focused tests for Agent 05: Vishwakarma visual intelligence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.evidence_store import EvidenceStoreBuilder
from src.agents.model_lens_agent import ModelLensAgent
from src.agents.signal_sentinel_agent import SignalSentinelAgent
from src.agents.vishwakarma_visual_architect import VishwakarmaVisualArchitect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
VISUALS_DIR = REPORTS_DIR / "visuals"


@pytest.fixture(scope="module", autouse=True)
def generated_vishwakarma_outputs() -> None:
    """Generate deterministic upstream and visual outputs once."""
    SignalSentinelAgent().save_outputs()
    ModelLensAgent().save_outputs()
    EvidenceStoreBuilder().save()
    VishwakarmaVisualArchitect().save_outputs()
    EvidenceStoreBuilder().save()


def test_vishwakarma_manifest_identifies_agent() -> None:
    """Manifest should identify Vishwakarma."""
    manifest = json.loads((VISUALS_DIR / "vishwakarma_output.json").read_text(encoding="utf-8"))
    assert manifest["agent"] == "Vishwakarma"


def test_required_visual_outputs_exist() -> None:
    """Interactive risk map and lineage graph should always be written."""
    assert (VISUALS_DIR / "feature_risk_scatter.json").exists()
    assert (VISUALS_DIR / "feature_risk_scatter.html").exists()
    assert (VISUALS_DIR / "lineage_graph.json").exists()
    assert (VISUALS_DIR / "lineage_graph.svg").exists()


def test_prediction_overlay_exists_for_demo_predictions() -> None:
    """Bundled demo includes reference and current prediction logs."""
    assert (VISUALS_DIR / "prediction_distribution_overlay.json").exists()
    assert (VISUALS_DIR / "prediction_distribution_overlay.html").exists()


def test_manifest_contains_recommended_visuals() -> None:
    """Manifest should list report-ready visual names."""
    manifest = json.loads((VISUALS_DIR / "vishwakarma_output.json").read_text(encoding="utf-8"))
    assert "feature_risk_scatter" in manifest["recommended_report_visuals"]
    assert "lineage_graph" in manifest["recommended_report_visuals"]


def test_lineage_graph_contains_agent_nodes() -> None:
    """Lineage graph should show deterministic workflow handoffs."""
    graph = json.loads((VISUALS_DIR / "lineage_graph.json").read_text(encoding="utf-8"))
    node_ids = {node["id"] for node in graph["nodes"]}
    assert {"mitra", "varuna", "evidence_store", "vishwakarma", "aryaman"}.issubset(node_ids)


def test_visual_agent_does_not_modify_upstream_metric_artifacts() -> None:
    """Vishwakarma should remain read-only with respect to verified metrics."""
    paths = [
        REPORTS_DIR / "drift_report.csv",
        REPORTS_DIR / "prediction_drift_report.json",
        REPORTS_DIR / "shap_global_importance.csv",
        REPORTS_DIR / "vif_report.csv",
        REPORTS_DIR / "feature_risk_matrix.csv",
    ]
    before = {path: path.read_bytes() for path in paths}
    VishwakarmaVisualArchitect().save_outputs()
    after = {path: path.read_bytes() for path in paths}
    assert before == after


def test_evidence_store_includes_matching_run_visual_manifest() -> None:
    """Evidence refresh should expose only same-run visual paths."""
    packet = json.loads((REPORTS_DIR / "evidence_packet.json").read_text(encoding="utf-8"))
    assert "feature_risk_scatter" in packet["visuals_available"]
    assert "lineage_graph" in packet["visuals_available"]
