"""Generate a concise business-facing model review report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


class ReportAgent:
    """Build a model review brief from saved monitoring artifacts."""

    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        """Load model metadata and agent outputs from the project artifact folders."""
        self.project_root = project_root
        self.models_dir = project_root / "models"
        self.reports_dir = project_root / "reports"
        self.metadata = self._load_json(self.models_dir / "model_metadata.json")
        self.drift_report = self._load_csv(self.reports_dir / "drift_report.csv")
        self.shap_importance = self._load_csv(self.reports_dir / "shap_global_importance.csv")
        self.cluster_shift_report = self._load_csv(self.reports_dir / "cluster_shift_report.csv")
        self.feature_suggestions = self._load_csv(self.reports_dir / "feature_suggestions.csv")

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Load one required JSON artifact."""
        if not path.exists():
            raise FileNotFoundError(f"Required report input not found: {path}")
        with path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        """Load one required tabular report artifact."""
        if not path.exists():
            raise FileNotFoundError(f"Required report input not found: {path}")
        return pd.read_csv(path)

    def model_risk_level(self) -> tuple[str, str]:
        """Classify model risk from drift volume and change in current AUC."""
        metrics = self.metadata["metrics"]
        validation_auc = float(metrics["validation"]["auc"])
        current_auc = float(metrics["current"]["auc"])
        auc_drop = validation_auc - current_auc
        high_drift_count = int((self.drift_report["drift_level"] == "High").sum())
        medium_drift_count = int((self.drift_report["drift_level"] == "Medium").sum())

        if high_drift_count >= 3 or auc_drop > 0.05:
            return (
                "High",
                f"{high_drift_count} features have high drift; AUC change is {current_auc - validation_auc:+.4f}.",
            )
        if medium_drift_count >= 2 or auc_drop > 0.02:
            return (
                "Medium",
                f"{medium_drift_count} features have medium drift; AUC change is {current_auc - validation_auc:+.4f}.",
            )
        return (
            "Low",
            f"No material risk threshold was exceeded; AUC change is {current_auc - validation_auc:+.4f}.",
        )

    @staticmethod
    def _metrics_table(metrics: dict[str, dict[str, float]]) -> str:
        """Render validation and current model metrics as Markdown."""
        rows = ["| Dataset | AUC | Accuracy | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for label, values in (("Validation", metrics["validation"]), ("Current", metrics["current"])):
            rows.append(
                f"| {label} | {values['auc']:.4f} | {values['accuracy']:.4f} | "
                f"{values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} |"
            )
        return "\n".join(rows)

    def generate_report(self) -> str:
        """Render the complete model review as Markdown text."""
        risk_level, risk_reason = self.model_risk_level()
        metrics = self.metadata["metrics"]
        validation_auc = float(metrics["validation"]["auc"])
        current_auc = float(metrics["current"]["auc"])
        top_drivers = self.shap_importance.head(10)
        high_drift = self.drift_report.loc[
            self.drift_report["drift_level"] == "High"
        ].sort_values("psi", ascending=False)
        material_clusters = self.cluster_shift_report.assign(
            abs_shift=self.cluster_shift_report["population_shift_pct_points"].abs()
        ).sort_values("abs_shift", ascending=False)

        lines = [
            "# PurchaseIntel Lens Model Review Report",
            "",
            "## 1. Executive Summary",
            "",
            f"**Model risk level: {risk_level}.** {risk_reason}",
            "",
            f"The model maintains ranking performance on the current synthetic snapshot "
            f"(AUC {current_auc:.4f} versus validation AUC {validation_auc:.4f}), but "
            f"{len(high_drift)} monitored features show high drift. The main operational "
            "concern is changing customer behavior rather than immediate discrimination loss.",
            "",
            "## 2. Model Objective",
            "",
            f"`{self.metadata['model_name']}` is a `{self.metadata['model_type']}` model that "
            f"predicts `{self.metadata['target']}` using {len(self.metadata['feature_list'])} "
            "synthetic customer behavior and profile features.",
            "",
            "## 3. Model Performance",
            "",
            self._metrics_table(metrics),
            "",
            "Current-period accuracy and precision remain stable, while recall is lower. "
            "The default decision threshold may miss likely purchasers.",
            "",
            "## 4. Top Feature Drivers",
            "",
            "| Rank | Feature | Mean Absolute SHAP Value |",
            "| ---: | --- | ---: |",
        ]
        for rank, row in enumerate(top_drivers.itertuples(), start=1):
            lines.append(f"| {rank} | `{row.feature}` | {row.mean_abs_shap_value:.4f} |")

        lines.extend(
            [
                "",
                "QSR spend and transaction frequency are the leading prediction drivers. "
                "Weekend dining frequency and competitor share are both influential and drifting.",
                "",
                "## 5. Drift Findings",
                "",
                "| Feature | Drift Level | PSI | Mean Change |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for row in high_drift.itertuples():
            lines.append(
                f"| `{row.feature}` | {row.drift_level} | {row.psi:.4f} | "
                f"{row.mean_change_pct:+.2f}% |"
            )
        lines.extend(
            [
                "",
                "Fuel spend and merchant novelty increased materially, weekend dining declined, "
                "and competitor QSR share increased. These shifts should be monitored before model refresh decisions.",
                "",
                "## 6. Segment/Cluster Findings",
                "",
                "| Segment | Reference Share | Current Share | Shift |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in material_clusters.itertuples():
            lines.append(
                f"| {row.cluster_name} | {row.reference_pct:.2f}% | {row.current_pct:.2f}% | "
                f"{row.population_shift_pct_points:+.2f} pts |"
            )
        lines.extend(
            [
                "",
                "The Loyal QSR Buyers segment contracted while Value-Seeking Routine Shoppers increased, "
                "consistent with changing behavioral mix in the current snapshot.",
                "",
                "## 7. Feature Recommendations",
                "",
                "| Suggested Feature | Priority | Rationale |",
                "| --- | --- | --- |",
            ]
        )
        for row in self.feature_suggestions.itertuples():
            lines.append(f"| `{row.suggested_feature}` | {row.priority} | {row.reason} |")

        high_recommendations = self.feature_suggestions.loc[
            self.feature_suggestions["priority"] == "High", "suggested_feature"
        ].tolist()
        lines.extend(
            [
                "",
                "## 8. Recommended Actions",
                "",
                "1. Continue current model monitoring, but flag the deployment for high drift review.",
                "2. Track high-drift predictors and segment distribution in the next current-period snapshot.",
                f"3. Prototype the high-priority candidate features: {', '.join(f'`{feature}`' for feature in high_recommendations)}.",
                "4. Evaluate threshold tuning or calibration because current recall is below validation recall.",
                "5. Retrain or recalibrate only after confirming whether the observed behavioral shifts persist.",
                "",
                "All data and findings in this report are based on synthetic observations.",
                "",
            ]
        )
        return "\n".join(lines)

    def run(self) -> Path:
        """Write the generated Markdown report and return its path."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.reports_dir / "model_review_report.md"
        output_path.write_text(self.generate_report(), encoding="utf-8")
        return output_path


def main() -> None:
    """Generate and save the complete model review report."""
    agent = ReportAgent()
    output_path = agent.run()
    risk_level, risk_reason = agent.model_risk_level()
    print(f"Model risk level: {risk_level}. {risk_reason}")
    print(f"Saved model review report to {output_path}")


if __name__ == "__main__":
    main()
