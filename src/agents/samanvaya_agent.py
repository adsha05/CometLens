"""Agent 04: Samanvaya for feedback-driven calibration recommendations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.feedback_store import FEEDBACK_PATH, load_feedback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"


class SamanvayaAgent:
    """Read feedback and propose human-reviewable calibration changes."""

    def __init__(
        self,
        feedback_path: Path = FEEDBACK_PATH,
        calibration_config_path: Path = CONFIGS_DIR / "calibration_config_v1.json",
    ) -> None:
        """Configure feedback and calibration inputs."""
        self.feedback_path = Path(feedback_path)
        self.calibration_config_path = Path(calibration_config_path)
        self.feedback: pd.DataFrame = pd.DataFrame()
        self.config: dict[str, Any] = {}

    def load_inputs(self) -> None:
        """Load feedback events and active calibration config."""
        self.feedback = load_feedback(self.feedback_path)
        if not self.calibration_config_path.exists():
            raise FileNotFoundError(f"Calibration config not found: {self.calibration_config_path}")
        self.config = json.loads(self.calibration_config_path.read_text(encoding="utf-8"))

    def build_recommendations(self) -> dict[str, Any]:
        """Build non-mutating calibration recommendations."""
        if not self.config:
            self.load_inputs()

        generated_at = datetime.now(timezone.utc).isoformat()
        if self.feedback.empty:
            return {
                "agent_name": "Agent 04: Samanvaya",
                "config_version": self.config.get("config_version", "unknown"),
                "source_files": {
                    "feedback_log": str(self.feedback_path),
                    "calibration_config": str(self.calibration_config_path),
                },
                "generated_at_utc": generated_at,
                "feedback_events": 0,
                "recommendations": [],
                "summary": "No dashboard feedback is available yet. No calibration changes are recommended.",
                "human_approval_required": True,
            }

        signal_counts = self.feedback["signal"].value_counts().to_dict()
        false_positive_count = int(signal_counts.get("false positive", 0))
        useful_count = int(signal_counts.get("useful", 0))
        threshold = int(self.config.get("samanvaya", {}).get("false_positive_threshold_for_review", 2))

        recommendations: list[dict[str, Any]] = []
        if false_positive_count >= threshold:
            recommendations.append(
                {
                    "recommendation_type": "threshold_review",
                    "target_config": "mitra.psi_high",
                    "current_value": self.config["mitra"]["psi_high"],
                    "proposed_value": round(float(self.config["mitra"]["psi_high"]) * 1.10, 4),
                    "reason": "Repeated false-positive feedback suggests the current high-drift PSI threshold may be too sensitive.",
                    "approval_status": "pending_human_review",
                }
            )
        if useful_count >= int(self.config.get("samanvaya", {}).get("useful_signal_threshold_for_keep", 3)):
            recommendations.append(
                {
                    "recommendation_type": "keep_current_thresholds",
                    "target_config": "mitra",
                    "current_value": self.config["mitra"],
                    "proposed_value": self.config["mitra"],
                    "reason": "Useful feedback volume supports keeping current Mitra thresholds for now.",
                    "approval_status": "pending_human_review",
                }
            )

        return {
            "agent_name": "Agent 04: Samanvaya",
            "config_version": self.config.get("config_version", "unknown"),
            "source_files": {
                "feedback_log": str(self.feedback_path),
                "calibration_config": str(self.calibration_config_path),
            },
            "generated_at_utc": generated_at,
            "feedback_events": int(len(self.feedback)),
            "feedback_signal_counts": signal_counts,
            "recommendations": recommendations,
            "summary": (
                "Calibration recommendations were generated for human review."
                if recommendations
                else "Feedback exists, but no calibration threshold change crossed the MVP recommendation rule."
            ),
            "human_approval_required": True,
        }

    def build_v2_config(self, recommendations: dict[str, Any]) -> dict[str, Any]:
        """Create a proposed v2 config without changing runtime behavior automatically."""
        proposed = json.loads(json.dumps(self.config))
        proposed["config_version"] = "v2"
        proposed["status"] = "proposed_pending_human_approval"
        proposed["source"] = "Agent 04: Samanvaya recommendations"
        for recommendation in recommendations.get("recommendations", []):
            if recommendation.get("target_config") == "mitra.psi_high":
                proposed["mitra"]["psi_high"] = recommendation["proposed_value"]
        return proposed

    def save_outputs(self) -> dict[str, Path]:
        """Save recommendations, simulated change log, and proposed v2 config."""
        self.load_inputs()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

        recommendations = self.build_recommendations()
        proposed_config = self.build_v2_config(recommendations)
        change_log = {
            "agent_name": "Agent 04: Samanvaya",
            "config_version": self.config.get("config_version", "unknown"),
            "source_files": {
                "feedback_log": str(self.feedback_path),
                "calibration_config": str(self.calibration_config_path),
            },
            "generated_at_utc": recommendations["generated_at_utc"],
            "approval_mode": "simulated_human_review",
            "approval_status": "pending",
            "note": "MVP records proposed changes but does not automatically apply them to agent runtime.",
            "recommendation_count": len(recommendations.get("recommendations", [])),
        }

        recommendation_path = REPORTS_DIR / "samanvaya_recommendations.json"
        change_log_path = REPORTS_DIR / "config_change_log.json"
        proposed_config_path = CONFIGS_DIR / "calibration_config_v2.json"

        recommendation_path.write_text(json.dumps(recommendations, indent=2), encoding="utf-8")
        change_log_path.write_text(json.dumps(change_log, indent=2), encoding="utf-8")
        proposed_config_path.write_text(json.dumps(proposed_config, indent=2), encoding="utf-8")

        return {
            "recommendations": recommendation_path,
            "config_change_log": change_log_path,
            "proposed_config": proposed_config_path,
        }


def main() -> None:
    """Run Agent 04: Samanvaya from the command line."""
    output_paths = SamanvayaAgent().save_outputs()
    print("Saved Agent 04: Samanvaya outputs:")
    for label, path in output_paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
