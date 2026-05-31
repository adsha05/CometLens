"""Focused tests for Agent 04: Samanvaya governed feedback calibration."""

from __future__ import annotations

import json
from pathlib import Path

from src.agents.samanvaya_calibration_agent import SamanvayaCalibrationAgent
from src.memory.feedback_store import append_feedback_event, ensure_feedback_log, load_feedback_log


def _write_json(path: Path, payload: dict) -> Path:
    """Write one test JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _agent_paths(tmp_path: Path, *, high_risk: bool = False) -> dict[str, Path]:
    """Create isolated required inputs and output paths."""
    config = {
        "config_version": "v1",
        "mitra": {"psi_high": 0.25},
        "aryaman": {},
        "samanvaya": {"minimum_feedback_events_for_recommendation": 2},
    }
    evidence = {
        "run_id": "TEST_RUN",
        "key_findings": ["Synthetic deterministic finding."],
        "high_risk_features": [{"feature": "merchant_novelty_rate"}] if high_risk else [],
    }
    return {
        "evidence_packet": _write_json(tmp_path / "reports" / "evidence_packet.json", evidence),
        "aryaman_output": _write_json(
            tmp_path / "reports" / "aryaman_output.json",
            {"model_health_status": "High Risk" if high_risk else "Low Risk"},
        ),
        "mitra_output": _write_json(tmp_path / "reports" / "mitra_output.json", {}),
        "varuna_output": _write_json(tmp_path / "reports" / "model_lens_output.json", {}),
        "vishwakarma_output": _write_json(tmp_path / "reports" / "visuals" / "vishwakarma_output.json", {}),
        "calibration_config": _write_json(tmp_path / "configs" / "calibration_config_v1.json", config),
        "feedback_log": tmp_path / "reports" / "feedback_log.csv",
        "samanvaya_output": tmp_path / "reports" / "samanvaya_output.json",
        "calibration_recommendations": tmp_path / "reports" / "calibration_recommendations.json",
        "legacy_recommendations": tmp_path / "reports" / "samanvaya_recommendations.json",
        "config_change_log": tmp_path / "reports" / "config_change_log.json",
        "recommended_config": tmp_path / "configs" / "calibration_config_v2_recommended.json",
    }


def _false_alarm_event(feature: str) -> dict[str, str]:
    """Return one valid false-alarm event."""
    return {
        "run_id": "TEST_RUN",
        "user_role": "model_analyst",
        "finding_id": "MITRA_DRIFT_001",
        "feature": feature,
        "feedback_type": "false_alarm",
        "severity": "Medium",
        "comment": "Expected seasonal movement.",
        "related_agent": "Mitra",
        "action_taken": "review_requested",
    }


def test_feedback_log_is_created_if_missing(tmp_path: Path) -> None:
    """Feedback store should initialize a governed CSV contract."""
    path = ensure_feedback_log(tmp_path / "reports" / "feedback_log.csv")
    assert path.exists()
    assert list(load_feedback_log(path).columns) == [
        "timestamp",
        "run_id",
        "user_role",
        "finding_id",
        "feature",
        "feedback_type",
        "severity",
        "comment",
        "related_agent",
        "action_taken",
    ]


def test_samanvaya_writes_required_outputs_and_preserves_v1(tmp_path: Path) -> None:
    """Samanvaya should write governed artifacts without changing active config."""
    paths = _agent_paths(tmp_path)
    before = paths["calibration_config"].read_bytes()
    output_paths = SamanvayaCalibrationAgent(paths=paths).save_outputs()
    assert set(output_paths) == {
        "samanvaya_output",
        "calibration_recommendations",
        "config_change_log",
        "recommended_config",
    }
    assert all(path.exists() for path in output_paths.values())
    assert paths["calibration_config"].read_bytes() == before
    output = json.loads(paths["samanvaya_output"].read_text(encoding="utf-8"))
    assert output["agent"] == "Samanvaya"


def test_repeated_false_alarm_generates_seasonal_tagging_recommendation(tmp_path: Path) -> None:
    """Repeated feature false alarms should produce a reviewable seasonal tag."""
    paths = _agent_paths(tmp_path)
    ensure_feedback_log(paths["feedback_log"])
    append_feedback_event(paths["feedback_log"], _false_alarm_event("fuel_spend_30d"))
    append_feedback_event(paths["feedback_log"], _false_alarm_event("fuel_spend_30d"))
    agent = SamanvayaCalibrationAgent(paths=paths)
    agent.save_outputs()
    payload = json.loads(paths["calibration_recommendations"].read_text(encoding="utf-8"))
    recommendation = next(item for item in payload["recommendations"] if item["feature"] == "fuel_spend_30d")
    assert recommendation["change_type"] == "seasonal_feature_tagging"
    assert recommendation["requires_human_approval"] is True
    proposed = json.loads(paths["recommended_config"].read_text(encoding="utf-8"))
    assert "fuel_spend_30d" in proposed["mitra"]["known_seasonal_features"]
    assert proposed["approval_status"] == "pending_human_review"


def test_all_recommendations_require_human_approval(tmp_path: Path) -> None:
    """Every recommendation should explicitly require human approval."""
    paths = _agent_paths(tmp_path)
    ensure_feedback_log(paths["feedback_log"])
    append_feedback_event(paths["feedback_log"], _false_alarm_event("fuel_spend_30d"))
    append_feedback_event(paths["feedback_log"], _false_alarm_event("fuel_spend_30d"))
    recommendations = SamanvayaCalibrationAgent(paths=paths).generate_calibration_recommendations()
    assert recommendations["recommendations"]
    assert all(item["requires_human_approval"] is True for item in recommendations["recommendations"])


def test_no_feedback_high_risk_evidence_requests_human_review(tmp_path: Path) -> None:
    """High risk without feedback should request review rather than mutate config."""
    paths = _agent_paths(tmp_path, high_risk=True)
    agent = SamanvayaCalibrationAgent(paths=paths)
    agent.save_outputs()
    payload = json.loads(paths["calibration_recommendations"].read_text(encoding="utf-8"))
    assert any(item["change_type"] == "human_review" for item in payload["recommendations"])
    proposed = json.loads(paths["recommended_config"].read_text(encoding="utf-8"))
    assert proposed["approval_status"] == "pending_human_review"


def test_no_feedback_low_risk_creates_no_change_proposal(tmp_path: Path) -> None:
    """No-feedback low-risk reviews should still produce an auditable no-change config."""
    paths = _agent_paths(tmp_path)
    SamanvayaCalibrationAgent(paths=paths).save_outputs()
    proposed = json.loads(paths["recommended_config"].read_text(encoding="utf-8"))
    assert proposed["approval_status"] == "no_changes_recommended"
    assert proposed["generated_by"] == "Samanvaya"
