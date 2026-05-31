"""Agent 04: Samanvaya governed feedback calibration recommendations."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.memory.feedback_store import ensure_feedback_log, load_feedback_log, summarize_feedback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def default_paths(use_case: str | None = None) -> dict[str, Path]:
    """Return configurable Samanvaya paths with current repo defaults."""
    use_case_reports = REPORTS_DIR / use_case if use_case else REPORTS_DIR
    reports_dir = use_case_reports if use_case and use_case_reports.exists() else REPORTS_DIR
    return {
        "evidence_packet": reports_dir / "evidence_packet.json",
        "aryaman_output": reports_dir / "aryaman_output.json",
        "mitra_output": reports_dir / "mitra_output.json",
        "varuna_output": reports_dir / "model_lens_output.json",
        "vishwakarma_output": reports_dir / "visuals" / "vishwakarma_output.json",
        "calibration_config": CONFIGS_DIR / "calibration_config_v1.json",
        "feedback_log": reports_dir / "feedback_log.csv",
        "samanvaya_output": reports_dir / "samanvaya_output.json",
        "calibration_recommendations": reports_dir / "calibration_recommendations.json",
        "legacy_recommendations": reports_dir / "samanvaya_recommendations.json",
        "config_change_log": reports_dir / "config_change_log.json",
        "recommended_config": CONFIGS_DIR / "calibration_config_v2_recommended.json",
    }


class SamanvayaCalibrationAgent:
    """Analyze human feedback and propose auditable calibration changes."""

    def __init__(
        self,
        paths: dict[str, str | Path] | None = None,
        config_path: str | Path = CONFIGS_DIR / "calibration_config_v1.json",
        *,
        demo_mode: bool = False,
    ) -> None:
        """Configure saved artifact paths and optional synthetic demo feedback."""
        configured = default_paths()
        configured["calibration_config"] = Path(config_path)
        if paths:
            configured.update({key: Path(value) for key, value in paths.items()})
        self.paths = {key: Path(value) for key, value in configured.items()}
        self.demo_mode = demo_mode
        self.inputs: dict[str, Any] = {}
        self.feedback = pd.DataFrame()
        self.patterns: dict[str, Any] = {}
        self.recommendations_payload: dict[str, Any] = {}
        self.change_log: dict[str, Any] = {}
        self.recommended_config: dict[str, Any] = {}
        self.output: dict[str, Any] = {}
        self.warnings: list[str] = []

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Load a required JSON object."""
        if not path.exists():
            raise FileNotFoundError(f"Required Samanvaya input not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}")
        return payload

    def load_inputs(self) -> dict[str, Any]:
        """Load saved evidence, agent outputs, active config, and feedback."""
        ensure_feedback_log(self.paths["feedback_log"], demo_mode=self.demo_mode)
        self.feedback = load_feedback_log(self.paths["feedback_log"])
        self.inputs = {
            "evidence_packet": self._load_json(self.paths["evidence_packet"]),
            "aryaman_output": self._load_json(self.paths["aryaman_output"]),
            "mitra_output": self._load_json(self.paths["mitra_output"]),
            "varuna_output": self._load_json(self.paths["varuna_output"]),
            "vishwakarma_output": self._load_json(self.paths["vishwakarma_output"]),
            "calibration_config": self._load_json(self.paths["calibration_config"]),
            "feedback_log": self.feedback,
        }
        return self.inputs

    def _minimum_events(self) -> int:
        """Return the config-driven minimum feedback event count."""
        config = self.inputs["calibration_config"].get("samanvaya", {})
        return int(
            config.get(
                "minimum_feedback_events_for_recommendation",
                config.get("false_positive_threshold_for_review", 2),
            )
        )

    def analyze_feedback_patterns(self) -> dict[str, Any]:
        """Detect repeat feedback patterns without changing runtime behavior."""
        if not self.inputs:
            self.load_inputs()
        minimum = self._minimum_events()
        feedback = self.feedback
        false_alarms = feedback.loc[feedback["feedback_type"] == "false_alarm"]
        accepted = feedback.loc[feedback["feedback_type"] == "accepted_recommendation"]
        rejected = feedback.loc[feedback["feedback_type"] == "rejected_recommendation"]
        executive_style = feedback.loc[
            (feedback["feedback_type"] == "too_technical")
            & feedback["user_role"].isin(["executive", "client_safe", "client"])
        ]
        self.patterns = {
            "minimum_feedback_events_for_recommendation": minimum,
            "feedback_summary": summarize_feedback(feedback),
            "repeated_false_alarms": {
                str(feature): int(count)
                for feature, count in false_alarms["feature"].value_counts().items()
                if feature and int(count) >= minimum
            },
            "repeated_accepted_recommendations": {
                str(feature): int(count)
                for feature, count in accepted["feature"].value_counts().items()
                if feature and int(count) >= minimum
            },
            "repeated_rejected_recommendations": {
                str(finding_id): int(count)
                for finding_id, count in rejected["finding_id"].value_counts().items()
                if finding_id and int(count) >= minimum
            },
            "executive_too_technical_count": int(len(executive_style)),
            "has_high_risk_evidence": bool(self.inputs["evidence_packet"].get("high_risk_features"))
            or self.inputs["aryaman_output"].get("model_health_status") == "High Risk",
        }
        return self.patterns

    @staticmethod
    def _get_nested(config: dict[str, Any], dotted_path: str, default: Any) -> Any:
        """Read a nested config value from a dotted path."""
        value: Any = config
        for key in dotted_path.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    @staticmethod
    def _set_nested(config: dict[str, Any], dotted_path: str, value: Any) -> None:
        """Set a nested config value from a dotted path."""
        target = config
        keys = dotted_path.split(".")
        for key in keys[:-1]:
            child = target.get(key)
            if not isinstance(child, dict):
                child = {}
                target[key] = child
            target = child
        target[keys[-1]] = value

    @staticmethod
    def _append_unique(values: Any, item: str) -> list[str]:
        """Return a list with an item included exactly once."""
        output = list(values) if isinstance(values, list) else []
        if item not in output:
            output.append(item)
        return output

    def generate_calibration_recommendations(self) -> dict[str, Any]:
        """Generate deterministic human-reviewable recommendations."""
        if not self.patterns:
            self.analyze_feedback_patterns()
        config = self.inputs["calibration_config"]
        findings_evidence = "evidence_packet:key_findings"
        recommendations: list[dict[str, Any]] = []

        def add(
            change_type: str,
            target_config_path: str | None,
            *,
            feature: str = "",
            recommended_value: Any = None,
            reason: str,
            evidence_used: list[str],
            confidence: str = "Medium",
        ) -> None:
            """Append one normalized governed recommendation."""
            recommendations.append(
                {
                    "recommendation_id": f"REC_{len(recommendations) + 1:03d}",
                    "change_type": change_type,
                    "target_config_path": target_config_path,
                    "feature": feature,
                    "current_value": (
                        self._get_nested(config, target_config_path, [])
                        if target_config_path
                        else None
                    ),
                    "recommended_value": recommended_value,
                    "reason": reason,
                    "evidence_used": evidence_used,
                    "confidence": confidence,
                    "requires_human_approval": True,
                    "status": "recommended",
                }
            )

        for feature, count in self.patterns["repeated_false_alarms"].items():
            path = "mitra.known_seasonal_features"
            add(
                "seasonal_feature_tagging",
                path,
                feature=feature,
                recommended_value=self._append_unique(self._get_nested(config, path, []), feature),
                reason=f"Repeated false-alarm feedback for {feature}; review seasonal tagging before suppression.",
                evidence_used=[f"feedback_log:false_alarm_count={count}", findings_evidence],
            )

        for feature, count in self.patterns["repeated_accepted_recommendations"].items():
            path = "mitra.high_priority_features"
            add(
                "increase_monitoring_priority",
                path,
                feature=feature,
                recommended_value=self._append_unique(self._get_nested(config, path, []), feature),
                reason=f"Repeated accepted recommendations for {feature} support continued monitoring priority.",
                evidence_used=[f"feedback_log:accepted_recommendation_count={count}", findings_evidence],
                confidence="High",
            )

        if self.patterns["executive_too_technical_count"] >= self._minimum_events():
            path = "aryaman.report_profile"
            add(
                "report_profile_adjustment",
                path,
                recommended_value={
                    "max_findings": 5,
                    "simplify_language": True,
                    "hide_technical_appendix_by_default": True,
                },
                reason="Repeated executive or client-safe feedback indicates that report language is too technical.",
                evidence_used=[
                    f"feedback_log:executive_too_technical_count={self.patterns['executive_too_technical_count']}",
                    findings_evidence,
                ],
            )

        for finding_id, count in self.patterns["repeated_rejected_recommendations"].items():
            path = "samanvaya.recommendations_requiring_additional_evidence"
            add(
                "require_more_evidence_before_escalation",
                path,
                feature=finding_id,
                recommended_value=self._append_unique(self._get_nested(config, path, []), finding_id),
                reason=f"Repeated rejection of {finding_id} indicates that escalation needs more supporting evidence.",
                evidence_used=[f"feedback_log:rejected_recommendation_count={count}", findings_evidence],
            )

        if self.feedback.empty and self.patterns["has_high_risk_evidence"]:
            add(
                "human_review",
                None,
                reason="High-risk deterministic evidence exists, but no analyst feedback has been recorded.",
                evidence_used=["evidence_packet:high_risk_features", "aryaman_output:model_health_status"],
                confidence="High",
            )

        evidence = self.inputs["evidence_packet"]
        self.recommendations_payload = {
            "agent": "Samanvaya",
            "agent_name": "Agent 04: Samanvaya",
            "run_id": evidence.get("run_id", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_version_reviewed": config.get("config_version", "unknown"),
            "recommendations": recommendations,
            "warnings": list(self.warnings),
            "human_approval_required": True,
            "summary": (
                "Calibration recommendations were generated for human review."
                if recommendations
                else "Feedback exists, but no configured recommendation rule crossed its threshold."
            ),
        }
        return self.recommendations_payload

    def generate_config_change_log(self) -> dict[str, Any]:
        """Build an auditable pending-only config change log."""
        if not self.recommendations_payload:
            self.generate_calibration_recommendations()
        self.change_log = {
            "run_id": self.recommendations_payload["run_id"],
            "timestamp": self.recommendations_payload["timestamp"],
            "config_version_reviewed": self.recommendations_payload["config_version_reviewed"],
            "recommended_changes": self.recommendations_payload["recommendations"],
            "approved_changes": [],
            "rejected_changes": [],
            "human_approval_required": True,
            "note": "No changes are applied automatically.",
            "source_files": self._source_files(),
        }
        return self.change_log

    def create_recommended_config(self) -> dict[str, Any]:
        """Create a pending proposal without overwriting the active v1 config."""
        if not self.recommendations_payload:
            self.generate_calibration_recommendations()
        proposed = deepcopy(self.inputs["calibration_config"])
        for recommendation in self.recommendations_payload["recommendations"]:
            target = recommendation.get("target_config_path")
            if target:
                self._set_nested(proposed, str(target), recommendation["recommended_value"])
        proposed["version"] = "v2_recommended"
        proposed["config_version"] = "v2_recommended"
        proposed["based_on"] = str(self.inputs["calibration_config"].get("config_version", "v1"))
        proposed["approval_status"] = (
            "pending_human_review" if self.recommendations_payload["recommendations"] else "no_changes_recommended"
        )
        proposed["generated_by"] = "Samanvaya"
        self.recommended_config = proposed
        return proposed

    def _source_files(self) -> dict[str, str]:
        """Return existing auditable input paths."""
        return {
            name: str(path)
            for name, path in self.paths.items()
            if name
            in {
                "evidence_packet",
                "aryaman_output",
                "mitra_output",
                "varuna_output",
                "vishwakarma_output",
                "calibration_config",
                "feedback_log",
            }
            and path.exists()
        }

    def save_outputs(self) -> dict[str, Path]:
        """Write governed recommendation artifacts and return their paths."""
        self.load_inputs()
        recommendations = self.generate_calibration_recommendations()
        change_log = self.generate_config_change_log()
        recommended_config = self.create_recommended_config()
        output_paths = {
            "samanvaya_output": self.paths["samanvaya_output"],
            "calibration_recommendations": self.paths["calibration_recommendations"],
            "config_change_log": self.paths["config_change_log"],
            "recommended_config": self.paths["recommended_config"],
        }
        for path in output_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        self.paths["calibration_recommendations"].write_text(json.dumps(recommendations, indent=2), encoding="utf-8")
        self.paths["legacy_recommendations"].write_text(json.dumps(recommendations, indent=2), encoding="utf-8")
        self.paths["config_change_log"].write_text(json.dumps(change_log, indent=2), encoding="utf-8")
        self.paths["recommended_config"].write_text(json.dumps(recommended_config, indent=2), encoding="utf-8")
        self.output = {
            "agent": "Samanvaya",
            "run_id": recommendations["run_id"],
            "timestamp": recommendations["timestamp"],
            "config_version_reviewed": recommendations["config_version_reviewed"],
            "recommendations_count": len(recommendations["recommendations"]),
            "high_confidence_recommendations": sum(
                item["confidence"] == "High" for item in recommendations["recommendations"]
            ),
            "pending_human_approval_count": len(recommendations["recommendations"]),
            "feedback_summary": self.patterns["feedback_summary"],
            "recommendations_path": str(self.paths["calibration_recommendations"]),
            "config_change_log_path": str(self.paths["config_change_log"]),
            "recommended_config_path": str(self.paths["recommended_config"]),
            "warnings": list(self.warnings),
            "source_files": self._source_files(),
        }
        self.paths["samanvaya_output"].write_text(json.dumps(self.output, indent=2), encoding="utf-8")
        return output_paths

    def run(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run Samanvaya as a future LangGraph-compatible state transformation."""
        next_state = dict(state or {})
        output_paths = self.save_outputs()
        next_state["samanvaya"] = self.output
        next_state["samanvaya_output_paths"] = {name: str(path) for name, path in output_paths.items()}
        return next_state


def parse_args() -> argparse.Namespace:
    """Parse Samanvaya CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Agent 04: Samanvaya feedback calibration.")
    parser.add_argument("--use_case", choices=["fraud", "purchase"], help="Optional configured report folder.")
    parser.add_argument("--demo", action="store_true", help="Seed synthetic feedback only when the feedback log is missing.")
    return parser.parse_args()


def main() -> None:
    """Run Samanvaya from the command line."""
    args = parse_args()
    output_paths = SamanvayaCalibrationAgent(paths=default_paths(args.use_case), demo_mode=args.demo).save_outputs()
    print("Saved Agent 04: Samanvaya outputs:")
    for label, path in output_paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
