"""Agent 03: Aryaman for deterministic executive synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.schemas import ExecutiveModelReport
from src.reports.report_renderer import render_executive_report_markdown, save_markdown_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def default_paths(use_case: str | None = None) -> dict[str, Path]:
    """Return configurable Aryaman paths with current repo defaults."""
    use_case_reports = REPORTS_DIR / use_case if use_case else REPORTS_DIR
    reports_dir = use_case_reports if use_case and use_case_reports.exists() else REPORTS_DIR
    return {
        "evidence_packet": reports_dir / "evidence_packet.json",
        "team_profiles": CONFIGS_DIR / "team_profiles.json",
        "markdown_report": reports_dir / "executive_model_report.md",
        "aryaman_output": reports_dir / "aryaman_output.json",
        "legacy_report_json": reports_dir / "executive_model_report.json",
    }


class ExecutiveSynthesisAgent:
    """Generate a concise executive report using only verified evidence_packet.json."""

    def __init__(
        self,
        paths: dict[str, str | Path] | None = None,
        evidence_packet_path: str | Path | None = None,
    ) -> None:
        """Configure evidence and output paths."""
        configured = default_paths()
        if paths:
            configured.update({key: Path(value) for key, value in paths.items()})
        if evidence_packet_path is not None:
            configured["evidence_packet"] = Path(evidence_packet_path)
        self.paths = {key: Path(value) for key, value in configured.items()}
        self.evidence_packet_path = self.paths["evidence_packet"]
        self.evidence: dict[str, Any] = {}
        self.team_profiles: dict[str, Any] = {}
        self.output: dict[str, Any] = {}

    def load_evidence(self) -> dict[str, Any]:
        """Load the single allowed Aryaman evidence source."""
        if not self.evidence_packet_path.exists():
            raise FileNotFoundError(
                f"Evidence packet not found: {self.evidence_packet_path}. "
                "Run `python src/agents/evidence_store.py` first."
            )
        self.evidence = json.loads(self.evidence_packet_path.read_text(encoding="utf-8"))
        if self.paths["team_profiles"].exists():
            self.team_profiles = json.loads(self.paths["team_profiles"].read_text(encoding="utf-8"))
        return self.evidence

    def determine_model_health_status(self) -> str:
        """Apply deterministic Aryaman health rules from verified evidence."""
        if not self.evidence:
            self.load_evidence()
        mitra_risk = self.evidence.get("mitra_summary", {}).get("overall_risk_level", "Low")
        high_risk_features = self.evidence.get("high_risk_features", [])
        all_feature_risks = self.evidence.get("feature_risk_matrix", [])
        top_five_high_drift = any(
            int(row.get("shap_rank", 999)) <= 5 and row.get("drift_level") == "High"
            for row in high_risk_features
        )
        medium_count = sum(row.get("final_risk") == "Medium" for row in all_feature_risks)
        if mitra_risk == "High" or top_five_high_drift:
            return "High Risk"
        if mitra_risk == "Medium" or medium_count >= 2:
            return "Medium Risk"
        return "Low Risk"

    def _top_issue(self) -> str:
        """Return the leading saved finding without inventing new evidence."""
        findings = self.evidence.get("key_findings", [])
        return findings[0] if findings else "No material issue was flagged by the deterministic evidence."

    def build_recommended_actions(self) -> list[str]:
        """Build deterministic Aryaman actions using verified packet values."""
        actions = list(self.evidence.get("recommended_actions", []))
        feature_risks = self.evidence.get("feature_risk_matrix", [])
        prediction = self.evidence.get("prediction_drift_summary", {})
        quality_counts = self.evidence.get("data_quality_summary", {}).get("issue_counts", {})
        if any(
            int(row.get("shap_rank", 999)) <= 5 and row.get("drift_level") == "High"
            for row in feature_risks
        ):
            actions.append("Review high-risk feature stability and consider recalibration before activation.")
        if prediction.get("prediction_drift_level") == "High":
            actions.append("Run validation before campaign or model activation because prediction drift is High.")
        if int(quality_counts.get("Medium", 0)) + int(quality_counts.get("High", 0)) > 0:
            actions.append("Review upstream data pipelines for medium or high data-quality issues.")
        if any(row.get("vif_risk") == "High" for row in feature_risks):
            actions.append("Review redundancy and consider consolidating high-VIF features.")
        return list(dict.fromkeys(actions))

    def build_output(self) -> dict[str, Any]:
        """Build Aryaman's deterministic JSON payload."""
        if not self.evidence:
            self.load_evidence()
        status = self.determine_model_health_status()
        metadata = self.evidence.get("model_metadata", {})
        model_name = metadata.get("model_name", "the reviewed model")
        use_case = metadata.get("business_use_case", "the configured business use case")
        key_findings = self.evidence.get("key_findings", [])
        high_risk_rows = self.evidence.get("high_risk_features", [])
        top_drivers = [
            f"{row['feature']} is SHAP rank {int(row['shap_rank'])} with mean |SHAP| "
            f"{float(row.get('mean_abs_shap', row.get('mean_abs_shap_value', 0.0))):.4f}."
            for row in self.evidence.get("top_model_drivers", [])[:5]
        ]
        high_risk_features = [
            f"{row['feature']}: {row.get('reason', 'High deterministic feature risk.')}"
            for row in high_risk_rows
        ]
        why_it_matters = (
            "Model decisions depend on stable signals and validated score behavior. Drift in model-important "
            "features or prediction distributions can reduce confidence in current-window activation decisions."
        )
        business_risks = [
            "Decision quality may decline if current-window behavior differs from the validated baseline.",
            "Activation efficiency may decline if score movement is not reviewed before operational use.",
        ]
        if status != "Low Risk":
            business_risks.append("High-impact use should wait for validation and feature-stability review.")
        self.output = {
            "agent": "Aryaman",
            "run_id": self.evidence.get("run_id", "unknown"),
            "timestamp": self.evidence.get("timestamp", "unknown"),
            "config_version": self.evidence.get("config_version", "unknown"),
            "model_health_status": status,
            "executive_summary": (
                f"{model_name} supports {use_case}. Deterministic evidence indicates **{status}**. "
                f"The leading issue is: {self._top_issue()} Review the recommended actions before activation."
            ),
            "key_findings_used": key_findings,
            "recommended_actions": self.build_recommended_actions(),
            "top_model_drivers": top_drivers,
            "high_risk_features": high_risk_features,
            "business_risks": business_risks,
            "why_it_matters": why_it_matters,
            "report_path": str(self.paths["markdown_report"]),
            "evidence_packet_path": str(self.evidence_packet_path),
            "recommended_report_visuals": self.evidence.get("recommended_report_visuals", []),
            "visuals_available": self.evidence.get("visuals_available", {}),
            "warnings": [],
        }
        return self.output

    def build_report(self) -> ExecutiveModelReport:
        """Build backward-compatible structured report data from Aryaman output."""
        output = self.build_output()
        visual_paths: list[str] = []
        for paths in self.evidence.get("visuals_available", {}).values():
            preferred_path = next(
                (paths[format_name] for format_name in ("png", "svg", "html", "json") if paths.get(format_name)),
                None,
            )
            if preferred_path:
                visual_paths.append(preferred_path)
        return ExecutiveModelReport(
            config_version=output["config_version"],
            source_files={"evidence_packet": str(self.evidence_packet_path)},
            report_title="Agent 03: Aryaman Executive Model Health Brief",
            model_health_status=output["model_health_status"],
            executive_summary=output["executive_summary"],
            what_changed=output["key_findings_used"],
            why_it_matters=output["why_it_matters"],
            top_model_drivers=output["top_model_drivers"],
            high_risk_features=output["high_risk_features"],
            business_risks=output["business_risks"],
            recommended_actions=output["recommended_actions"],
            plots_to_include=[
                item["path"]
                for item in self.evidence.get("plots_available", {}).values()
                if item.get("exists")
            ]
            + visual_paths,
            questions_for_team=[
                "Did any source-system, policy, or population-mix change coincide with this review window?",
                "Should activation thresholds be reviewed before the next high-impact use?",
            ],
            client_safe_summary=(
                f"The deterministic model review is rated {output['model_health_status']}. "
                "Review the identified feature and score movement before high-impact use."
            ),
            limitations=self.evidence.get("limitations", []),
        )

    def save_outputs(self) -> tuple[Path, Path]:
        """Save Aryaman JSON, Markdown, and legacy structured report JSON."""
        output = self.build_output()
        markdown = render_executive_report_markdown(self.evidence, output)
        markdown_path = save_markdown_report(markdown, self.paths["markdown_report"])
        self.paths["aryaman_output"].write_text(json.dumps(output, indent=2), encoding="utf-8")
        self.paths["legacy_report_json"].write_text(self.build_report().model_dump_json(indent=2), encoding="utf-8")
        return self.paths["aryaman_output"], markdown_path

    def run(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run Aryaman as a future LangGraph-compatible state transformation."""
        next_state = dict(state or {})
        json_path, markdown_path = self.save_outputs()
        next_state["aryaman"] = self.output
        next_state["aryaman_output_paths"] = {
            "json": str(json_path),
            "markdown": str(markdown_path),
        }
        return next_state


def parse_args() -> argparse.Namespace:
    """Parse Aryaman CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Agent 03: Aryaman executive synthesis.")
    parser.add_argument("--use_case", choices=["fraud", "purchase"], help="Optional configured report folder.")
    return parser.parse_args()


def main() -> None:
    """Run Agent 03: Aryaman from the command line."""
    args = parse_args()
    json_path, markdown_path = ExecutiveSynthesisAgent(paths=default_paths(args.use_case)).save_outputs()
    print(f"Saved Aryaman output JSON to {json_path}")
    print(f"Saved executive model report Markdown to {markdown_path}")


if __name__ == "__main__":
    main()
