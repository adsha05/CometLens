"""Agent 03: Aryaman for deterministic executive synthesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.schemas import ExecutiveModelReport, executive_report_to_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


class ExecutiveSynthesisAgent:
    """Generate a consulting-style model health report from verified evidence."""

    SAMPLE_FEATURE_ACTIONS = {
        "weekend_dining_frequency": "Prototype `weekend_dining_recovery_index` to capture whether weekend dining behavior is rebounding or still depressed.",
        "competitor_qsr_share_90d": "Prototype `competitor_switching_velocity` to measure recent movement toward competing merchants or alternatives.",
        "merchant_novelty_rate": "Prototype `merchant_confidence_score` to separate genuine new-merchant behavior from merchant classification uncertainty.",
    }

    def __init__(self, evidence_packet_path: Path = REPORTS_DIR / "evidence_packet.json") -> None:
        """Configure evidence input location."""
        self.evidence_packet_path = Path(evidence_packet_path)
        self.evidence: dict[str, Any] = {}

    def load_evidence(self) -> dict[str, Any]:
        """Load the deterministic evidence packet."""
        if not self.evidence_packet_path.exists():
            raise FileNotFoundError(
                f"Evidence packet not found: {self.evidence_packet_path}. "
                "Run `python src/agents/evidence_store.py` first."
            )
        self.evidence = json.loads(self.evidence_packet_path.read_text(encoding="utf-8"))
        return self.evidence

    @staticmethod
    def _risk_rank(status: str) -> int:
        """Map risk labels to a sortable rank."""
        return {"High": 3, "Medium": 2, "Low": 1}.get(status, 0)

    def determine_model_health_status(self) -> str:
        """Apply deterministic MVP risk rules."""
        signal = self.evidence.get("signal_sentinel_summary", {})
        model_lens = self.evidence.get("model_lens_summary", {})
        high_drift_count = int(signal.get("high_drift_feature_count", 0))
        overfitting_risk = model_lens.get("overfitting_check", {}).get("risk_level", "Low")
        has_high_vif = any(
            row.get("vif_level") == "High"
            for row in model_lens.get("multicollinearity_findings", [])
        )

        if high_drift_count >= 3 or overfitting_risk == "High":
            return "High Risk"
        if high_drift_count in {1, 2} or overfitting_risk == "Medium" or has_high_vif:
            return "Medium Risk"
        return "Low Risk"

    def _high_drift_features(self) -> list[dict[str, Any]]:
        """Return high-drift feature rows from the high-risk feature matrix."""
        matrix = self.evidence.get("model_lens_summary", {}).get("high_risk_feature_matrix", [])
        return [row for row in matrix if row.get("drift_level") == "High"]

    def _high_or_medium_cluster_shift(self) -> tuple[bool, float]:
        """Return whether cluster movement is material and the largest absolute shift."""
        cluster_rows = self.evidence.get("signal_sentinel_summary", {}).get("cluster_findings", [])
        if not cluster_rows:
            return False, 0.0
        max_shift = max(abs(float(row.get("share_change_pct_points", 0.0))) for row in cluster_rows)
        return max_shift >= 5.0, max_shift

    def _vif_issues(self, level: str | None = None) -> list[dict[str, Any]]:
        """Return VIF findings filtered by level when provided."""
        findings = self.evidence.get("model_lens_summary", {}).get("multicollinearity_findings", [])
        if level is None:
            return [row for row in findings if row.get("vif_level") in {"Medium", "High"}]
        return [row for row in findings if row.get("vif_level") == level]

    def _top_issue(self) -> str:
        """Select the top issue for executive summary text."""
        high_drift = self._high_drift_features()
        if high_drift:
            return f"{high_drift[0]['feature']} is both a monitored drift issue and model-risk feature"
        overfitting = self.evidence.get("model_lens_summary", {}).get("overfitting_check", {})
        if overfitting.get("risk_level") in {"Medium", "High"}:
            metric_name = overfitting.get("metric_name", "metric")
            return f"train-validation {metric_name} delta indicates {str(overfitting['risk_level']).lower()} overfitting risk"
        high_vif = self._vif_issues("High")
        if high_vif:
            return f"{high_vif[0]['feature']} has high VIF"
        reliability = self.evidence.get("model_lens_summary", {}).get("explainability_reliability", {})
        if reliability.get("status") == "unreliable":
            return "Varuna explainability is flagged as unreliable because severe Mitra drift was detected"
        return "no material model-health issue is flagged by the MVP evidence"

    def _next_step(self) -> str:
        """Select the recommended next step for summary text."""
        cluster_material, _ = self._high_or_medium_cluster_shift()
        if cluster_material:
            return "review affected segments and recalibrate before high-impact business use"
        if self._high_drift_features():
            return "review high-drift drivers and run a validation refresh"
        return "continue monitoring with the next synthetic current-period snapshot"

    def build_executive_summary(self, status: str) -> str:
        """Create concise executive summary text."""
        metadata = self.evidence.get("model_metadata", {})
        model_name = metadata.get("model_name", "the model")
        use_case = metadata.get("business_use_case", "the business use case")
        return (
            f"{model_name} supports {use_case}. Current MVP evidence indicates **{status}**. "
            f"The top issue is that {self._top_issue()}. The recommended next step is to "
            f"{self._next_step()}."
        )

    def build_what_changed(self) -> list[str]:
        """Summarize drift, cluster, prediction, and VIF changes."""
        findings: list[str] = []
        for row in self._high_drift_features():
            findings.append(f"{row['feature']} is high drift and has combined risk {row['combined_risk']}.")

        cluster_material, max_shift = self._high_or_medium_cluster_shift()
        if cluster_material:
            findings.append(f"Cluster mix shifted materially; largest movement is {max_shift:.1f} percentage points.")

        prediction = self.evidence.get("signal_sentinel_summary", {}).get("prediction_drift_summary", {})
        if prediction.get("available") and "prediction_actual_rate_gap" in prediction:
            findings.append(
                "Prediction-positive rate differs from actual-positive rate by "
                f"{float(prediction['prediction_actual_rate_gap']):+.3f}."
            )

        for row in self._vif_issues():
            findings.append(f"{row['feature']} has {str(row['vif_level']).lower()} VIF ({float(row['vif']):.2f}).")

        reliability = self.evidence.get("model_lens_summary", {}).get("explainability_reliability", {})
        if reliability.get("status") == "unreliable":
            findings.append("Varuna SHAP outputs are flagged as unreliable due to severe Mitra drift.")

        if not findings:
            findings.append("No material drift, prediction, cluster, or VIF issue was flagged by the MVP evidence.")
        return findings

    def build_why_it_matters(self) -> str:
        """Translate technical findings into business impact."""
        return (
            "Production model decisions depend on stable inputs, interpretable drivers, and consistent "
            "population context. When important features drift or segment mix changes, model scores can "
            "become less aligned with the current operating environment, increasing the risk of poor "
            "prioritization, inefficient resource allocation, or weak stakeholder trust."
        )

    def build_top_model_drivers(self) -> list[str]:
        """Return top driver statements from Varuna evidence."""
        drivers = self.evidence.get("model_lens_summary", {}).get("top_global_drivers", [])
        return [
            f"{row['feature']} is SHAP rank {int(row['shap_rank'])} with mean |SHAP| {float(row['mean_abs_shap_value']):.4f}"
            for row in drivers[:5]
        ]

    def build_high_risk_features(self) -> list[str]:
        """Return high and medium combined-risk feature statements."""
        matrix = self.evidence.get("model_lens_summary", {}).get("high_risk_feature_matrix", [])
        risk_rows = [row for row in matrix if row.get("combined_risk") in {"High", "Medium"}]
        return [
            f"{row['feature']}: combined risk {row['combined_risk']}, drift {row['drift_level']}, VIF warning {row['vif_warning']}"
            for row in risk_rows
        ]

    def build_business_risks(self, status: str) -> list[str]:
        """Return business risks implied by the evidence."""
        risks = [
            "Decision quality may decline if current input behavior differs from training-period behavior.",
            "Resource allocation may become less efficient if high-drift features are used without validation refresh.",
        ]
        if status != "Low Risk":
            risks.append("High-impact business use should wait for validation and segment review.")
        prediction = self.evidence.get("signal_sentinel_summary", {}).get("prediction_drift_summary", {})
        if abs(float(prediction.get("prediction_actual_rate_gap", 0.0))) >= 0.10:
            risks.append("Prediction-label mix gap may indicate threshold or calibration pressure.")
        return risks

    def build_recommended_actions(self) -> list[str]:
        """Build deterministic recommended actions from evidence rules."""
        actions: list[str] = []
        cluster_material, _ = self._high_or_medium_cluster_shift()
        if cluster_material:
            actions.append("Review affected segments and recalibrate before high-impact business use.")

        high_risk_features = self.evidence.get("model_lens_summary", {}).get("high_risk_feature_matrix", [])
        for row in high_risk_features:
            feature = row.get("feature")
            if row.get("combined_risk") in {"High", "Medium"}:
                if feature in self.SAMPLE_FEATURE_ACTIONS:
                    actions.append(self.SAMPLE_FEATURE_ACTIONS[feature])
                elif feature:
                    actions.append(
                        f"Review `{feature}` for stability, business definition changes, and potential monitored variants."
                    )

        high_vif_features = self._vif_issues("High")
        if high_vif_features:
            feature_names = ", ".join(row["feature"] for row in high_vif_features)
            actions.append(f"Review high-VIF features for redundancy or unstable coefficient behavior: {feature_names}.")

        reliability = self.evidence.get("model_lens_summary", {}).get("explainability_reliability", {})
        if reliability.get("status") == "unreliable":
            actions.append("Treat SHAP interpretation as directional until severe drift is resolved or the model is recalibrated.")

        actions.append("Re-run validation before high-impact business use.")
        return list(dict.fromkeys(actions))

    def build_plots_to_include(self) -> list[str]:
        """Select required plot artifacts for the report."""
        selected = [
            REPORTS_DIR / "figures" / "shap_global_bar.png",
            REPORTS_DIR / "figures" / "shap_beeswarm.png",
            REPORTS_DIR / "figures" / "drift_top_features.png",
        ]
        return [str(path) for path in selected if path.exists()]

    def build_questions_for_team(self) -> list[str]:
        """Create follow-up questions for business and data science teams."""
        return [
            "Did any source-system, data-definition, policy, product, or population mix change coincide with the current-window drift?",
            "Should threshold calibration be reviewed for the observed prediction-positive versus actual-positive gap?",
            "Do high-risk features require new monitoring thresholds before high-impact business use?",
        ]

    def build_report(self) -> ExecutiveModelReport:
        """Build the ExecutiveModelReport from the evidence packet."""
        if not self.evidence:
            self.load_evidence()
        status = self.determine_model_health_status()
        limitations = self.evidence.get("limitations", [])
        return ExecutiveModelReport(
            report_title="Agent 03: Aryaman Executive Model Health Brief",
            model_health_status=status,
            executive_summary=self.build_executive_summary(status),
            what_changed=self.build_what_changed(),
            why_it_matters=self.build_why_it_matters(),
            top_model_drivers=self.build_top_model_drivers(),
            high_risk_features=self.build_high_risk_features(),
            business_risks=self.build_business_risks(status),
            recommended_actions=self.build_recommended_actions(),
            plots_to_include=self.build_plots_to_include(),
            questions_for_team=self.build_questions_for_team(),
            client_safe_summary=(
                f"The model review is rated {status} based on synthetic MVP evidence. "
                "The main findings are feature drift, segment movement, and validation-risk indicators; "
                "these should be reviewed before high-impact business decisions."
            ),
            limitations=limitations,
        )

    def save_outputs(self) -> tuple[Path, Path]:
        """Save the executive report as JSON and Markdown."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report = self.build_report()
        json_path = REPORTS_DIR / "executive_model_report.json"
        markdown_path = REPORTS_DIR / "executive_model_report.md"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        markdown_path.write_text(executive_report_to_markdown(report), encoding="utf-8")
        return json_path, markdown_path


def main() -> None:
    """Run Agent 03: Aryaman."""
    json_path, markdown_path = ExecutiveSynthesisAgent().save_outputs()
    print(f"Saved executive model report JSON to {json_path}")
    print(f"Saved executive model report Markdown to {markdown_path}")


if __name__ == "__main__":
    main()
