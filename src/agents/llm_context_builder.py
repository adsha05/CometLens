"""Build a compact, deterministic evidence payload for narrative generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


class LLMContextBuilder:
    """Package verified analytics outputs for a grounded LLM narrative."""

    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        """Configure artifact locations."""
        self.project_root = project_root
        self.models_dir = project_root / "models"
        self.reports_dir = project_root / "reports"

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Load a required JSON input artifact."""
        if not path.exists():
            raise FileNotFoundError(f"Required narrative input not found: {path}")
        with path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        """Load a required CSV input artifact."""
        if not path.exists():
            raise FileNotFoundError(f"Required narrative input not found: {path}")
        return pd.read_csv(path)

    @staticmethod
    def _risk_level(metadata: dict[str, Any], drift_report: pd.DataFrame) -> tuple[str, str]:
        """Apply the deterministic model risk thresholds used in reporting."""
        validation_auc = float(metadata["metrics"]["validation"]["auc"])
        current_auc = float(metadata["metrics"]["current"]["auc"])
        auc_drop = validation_auc - current_auc
        high_count = int((drift_report["drift_level"] == "High").sum())
        medium_count = int((drift_report["drift_level"] == "Medium").sum())
        if high_count >= 3 or auc_drop > 0.05:
            return "High", f"{high_count} high drift features; AUC change {current_auc - validation_auc:+.4f}."
        if medium_count >= 2 or auc_drop > 0.02:
            return "Medium", f"{medium_count} medium drift features; AUC change {current_auc - validation_auc:+.4f}."
        return "Low", f"No risk threshold exceeded; AUC change {current_auc - validation_auc:+.4f}."

    def build(self) -> dict[str, Any]:
        """Return the evidence payload supplied to an LLM provider."""
        metadata = self._load_json(self.models_dir / "model_metadata.json")
        drift = self._load_csv(self.reports_dir / "drift_report.csv")
        shap_importance = self._load_csv(self.reports_dir / "shap_global_importance.csv")
        cluster_shift = self._load_csv(self.reports_dir / "cluster_shift_report.csv")
        feature_suggestions = self._load_csv(self.reports_dir / "feature_suggestions.csv")
        risk_level, risk_reason = self._risk_level(metadata, drift)

        high_drift = drift.loc[drift["drift_level"] == "High"].sort_values("psi", ascending=False)
        cluster_shift = cluster_shift.assign(
            abs_shift=cluster_shift["population_shift_pct_points"].abs()
        ).sort_values("abs_shift", ascending=False)
        return {
            "scope": "Synthetic QSR purchase propensity model monitoring only.",
            "guardrails": [
                "All observations and customer records are synthetic.",
                "Metrics and risk labels are deterministic inputs and must not be recalculated.",
                "Narrative hypotheses must not be presented as causes.",
            ],
            "model": {
                "model_name": metadata["model_name"],
                "model_type": metadata["model_type"],
                "target": metadata["target"],
                "feature_count": len(metadata["feature_list"]),
                "risk_level": risk_level,
                "risk_reason": risk_reason,
            },
            "performance": metadata["metrics"],
            "high_drift_features": high_drift[
                ["feature", "psi", "ks_pvalue", "mean_change_pct", "drift_level"]
            ].to_dict(orient="records"),
            "top_shap_features": shap_importance.head(10).to_dict(orient="records"),
            "cluster_shifts": cluster_shift[
                ["cluster_name", "reference_pct", "current_pct", "population_shift_pct_points"]
            ].to_dict(orient="records"),
            "deterministic_feature_suggestions": feature_suggestions.to_dict(orient="records"),
        }

    def save(self) -> Path:
        """Persist the exact evidence submitted for narrative generation."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.reports_dir / "llm_evidence_context.json"
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(self.build(), output_file, indent=2)
        return output_path


def main() -> None:
    """Build and save the evidence payload for inspection before LLM execution."""
    output_path = LLMContextBuilder().save()
    print(f"Saved narrative evidence payload to {output_path}")


if __name__ == "__main__":
    main()
