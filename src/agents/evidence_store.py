"""Build a deterministic evidence packet from AxionAI agent outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_EVIDENCE_PATH = REPORTS_DIR / "evidence_store.json"
DEFAULT_PACKET_PATH = REPORTS_DIR / "evidence_packet.json"


class EvidenceStore:
    """Persist agent evidence and build a compact packet for narrative use."""

    def __init__(self, path: Path = DEFAULT_EVIDENCE_PATH) -> None:
        """Create an evidence store at the provided path."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        """Load the append-only evidence store used by agents during execution."""
        if not self.path.exists():
            return {"evidence": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save_section(self, section: str, payload: dict[str, Any]) -> Path:
        """Save one named evidence section and return the store path."""
        evidence = self.load()
        evidence.setdefault("evidence", {})[section] = payload
        self.path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        return self.path


class EvidencePacketBuilder:
    """Create a single auditable evidence packet from deterministic artifacts."""

    def __init__(
        self,
        signal_sentinel_path: Path = REPORTS_DIR / "signal_sentinel_output.json",
        model_lens_path: Path = REPORTS_DIR / "model_lens_output.json",
        model_metadata_path: Path = MODELS_DIR / "model_metadata.json",
        feature_metadata_path: Path = MODELS_DIR / "feature_metadata.json",
        output_path: Path = DEFAULT_PACKET_PATH,
    ) -> None:
        """Configure source and destination artifacts."""
        self.signal_sentinel_path = Path(signal_sentinel_path)
        self.model_lens_path = Path(model_lens_path)
        self.model_metadata_path = Path(model_metadata_path)
        self.feature_metadata_path = Path(feature_metadata_path)
        self.output_path = Path(output_path)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Load a required JSON artifact."""
        if not path.exists():
            raise FileNotFoundError(f"Required evidence input not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _feature_direction(row: dict[str, Any]) -> str:
        """Describe feature mean direction from existing drift values."""
        change = float(row.get("mean_change_pct", 0.0))
        if change > 0:
            return "increased in current data"
        if change < 0:
            return "decreased in current data"
        return "was unchanged in current data"

    @staticmethod
    def _available_plot(path_value: str) -> dict[str, Any]:
        """Describe a plot path and whether it exists locally."""
        path = Path(path_value)
        return {"path": path_value, "exists": path.exists()}

    def _derive_key_findings(
        self,
        signal_sentinel: dict[str, Any],
        model_lens: dict[str, Any],
    ) -> list[str]:
        """Derive plain-English findings only from existing deterministic values."""
        findings: list[str] = []

        for row in signal_sentinel.get("high_drift_features", []):
            feature = row["feature"]
            findings.append(f"{feature} is a high-drift feature")
            findings.append(f"{feature} {self._feature_direction(row)}")

        for row in signal_sentinel.get("medium_drift_features", []):
            feature = row["feature"]
            findings.append(f"{feature} is a medium-drift feature")
            findings.append(f"{feature} {self._feature_direction(row)}")

        for row in model_lens.get("multicollinearity_findings", []):
            feature = row["feature"]
            vif_level = row.get("vif_level", "Low")
            if vif_level == "High":
                findings.append(f"{feature} has high VIF")
            elif vif_level == "Medium":
                findings.append(f"{feature} has medium VIF")

        overfitting = model_lens.get("overfitting_check", {})
        overfitting_level = overfitting.get("risk_level")
        if overfitting_level:
            metric_name = overfitting.get("metric_name", "metric")
            findings.append(
                f"train-validation {metric_name} delta indicates {str(overfitting_level).lower()} overfitting risk"
            )

        top_driver = next(iter(model_lens.get("top_global_drivers", [])), None)
        if top_driver:
            findings.append(f"{top_driver['feature']} is the top global model driver")

        prediction_summary = signal_sentinel.get("prediction_drift_summary", {})
        if "prediction_actual_rate_gap" in prediction_summary:
            gap = float(prediction_summary["prediction_actual_rate_gap"])
            findings.append(f"predicted-positive rate differs from actual-positive rate by {gap:+.3f}")

        return findings

    @staticmethod
    def _signal_summary(signal_sentinel: dict[str, Any]) -> dict[str, Any]:
        """Extract a compact Mitra summary."""
        return {
            "overall_risk_level": signal_sentinel.get("overall_risk_level"),
            "data_health_summary": signal_sentinel.get("data_health_summary", {}),
            "high_drift_feature_count": len(signal_sentinel.get("high_drift_features", [])),
            "medium_drift_feature_count": len(signal_sentinel.get("medium_drift_features", [])),
            "prediction_drift_summary": signal_sentinel.get("prediction_drift_summary", {}),
            "cluster_findings": signal_sentinel.get("cluster_findings", []),
            "recommended_checks": signal_sentinel.get("recommended_checks", []),
        }

    @staticmethod
    def _model_lens_summary(model_lens: dict[str, Any]) -> dict[str, Any]:
        """Extract a compact Varuna summary."""
        return {
            "top_global_drivers": model_lens.get("top_global_drivers", []),
            "high_risk_feature_matrix": model_lens.get("high_risk_feature_matrix", []),
            "multicollinearity_findings": model_lens.get("multicollinearity_findings", []),
            "overfitting_check": model_lens.get("overfitting_check", {}),
            "explainability_reliability": model_lens.get("explainability_reliability", {}),
        }

    def build_packet(self) -> dict[str, Any]:
        """Build the full evidence packet."""
        signal_sentinel = self._load_json(self.signal_sentinel_path)
        model_lens = self._load_json(self.model_lens_path)
        model_metadata = self._load_json(self.model_metadata_path)
        feature_metadata = self._load_json(self.feature_metadata_path)
        plots = {
            name: self._available_plot(path)
            for name, path in model_lens.get("plots_generated", {}).items()
        }

        return {
            "model_metadata": model_metadata,
            "feature_metadata": feature_metadata,
            "signal_sentinel_summary": self._signal_summary(signal_sentinel),
            "model_lens_summary": self._model_lens_summary(model_lens),
            "key_findings": self._derive_key_findings(signal_sentinel, model_lens),
            "available_plots": plots,
            "business_context": {
                "business_use_case": model_metadata.get("business_use_case"),
                "target": model_metadata.get("target"),
                "entity_id": model_metadata.get("entity_id"),
                "prediction_column": model_metadata.get("prediction_column"),
                "training_window": model_metadata.get("training_window"),
                "current_window": model_metadata.get("current_window"),
            },
            "limitations": [
                "Synthetic sample data only",
                "LLM narrative should not be treated as production validation",
                "Metrics are simulated for MVP demonstration",
            ],
        }

    def save(self) -> Path:
        """Write the evidence packet to disk."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        packet = self.build_packet()
        self.output_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
        return self.output_path


def main() -> None:
    """Build and save the deterministic evidence packet."""
    output_path = EvidencePacketBuilder().save()
    print(f"Saved evidence packet to {output_path}")


if __name__ == "__main__":
    main()
