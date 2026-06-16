"""Build a deterministic evidence packet from AxionAI agent artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_EVIDENCE_PATH = REPORTS_DIR / "evidence_store.json"
DEFAULT_PACKET_PATH = REPORTS_DIR / "evidence_packet.json"


def default_paths(use_case: str | None = None) -> dict[str, Path]:
    """Return configurable Evidence Store paths with current repo defaults."""
    use_case_reports = REPORTS_DIR / use_case if use_case else REPORTS_DIR
    reports_dir = use_case_reports if use_case and use_case_reports.exists() else REPORTS_DIR
    return {
        "mitra_output": reports_dir / "mitra_output.json",
        "varuna_output": reports_dir / "model_lens_output.json",
        "data_quality_report": reports_dir / "data_quality_report.csv",
        "drift_report": reports_dir / "drift_report.csv",
        "prediction_drift_report": reports_dir / "prediction_drift_report.json",
        "feature_risk_matrix": reports_dir / "feature_risk_matrix.csv",
        "model_diagnostics": reports_dir / "model_diagnostics.json",
        "calibration_report": reports_dir / "calibration_report.csv",
        "score_decile_report": reports_dir / "score_decile_report.csv",
        "lift_report": reports_dir / "lift_report.csv",
        "segment_performance_report": reports_dir / "segment_performance_report.csv",
        "vishwakarma_output": reports_dir / "visuals" / "vishwakarma_output.json",
        "model_metadata": MODELS_DIR / "model_metadata.json",
        "feature_metadata": MODELS_DIR / "feature_metadata.json",
        "calibration_config": CONFIGS_DIR / "calibration_config_v1.json",
        "output": reports_dir / "evidence_packet.json",
    }


class EvidenceStore:
    """Persist named deterministic evidence sections during agent execution."""

    def __init__(self, path: Path = DEFAULT_EVIDENCE_PATH) -> None:
        """Create an evidence store at the provided path."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        """Load saved agent evidence sections."""
        if not self.path.exists():
            return {"evidence": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save_section(self, section: str, payload: dict[str, Any]) -> Path:
        """Save one named section and return the store path."""
        evidence = self.load()
        evidence.setdefault("evidence", {})[section] = payload
        self.path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        return self.path


class EvidenceStoreBuilder:
    """Combine verified Mitra and Varuna outputs into one auditable evidence packet."""

    def __init__(self, paths: dict[str, str | Path] | None = None, **legacy_paths: str | Path) -> None:
        """Configure input and output paths while accepting earlier keyword paths."""
        configured = default_paths()
        if paths:
            configured.update({key: Path(value) for key, value in paths.items()})
        legacy_mapping = {
            "signal_sentinel_path": "mitra_output",
            "model_lens_path": "varuna_output",
            "model_metadata_path": "model_metadata",
            "feature_metadata_path": "feature_metadata",
            "output_path": "output",
        }
        for legacy_key, target_key in legacy_mapping.items():
            if legacy_key in legacy_paths:
                configured[target_key] = Path(legacy_paths[legacy_key])
        self.paths = {key: Path(value) for key, value in configured.items()}
        self.output_path = self.paths["output"]
        self.packet: dict[str, Any] = {}

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Load a required JSON object."""
        if not path.exists():
            raise FileNotFoundError(f"Required evidence input not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}")
        return payload

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        """Load a required CSV artifact."""
        if not path.exists():
            raise FileNotFoundError(f"Required evidence input not found: {path}")
        return pd.read_csv(path)

    @staticmethod
    def _available_plot(path_value: str) -> dict[str, Any]:
        """Return an auditable plot availability record."""
        path = Path(path_value)
        return {"path": str(path), "exists": path.exists()}

    @staticmethod
    def _risk_rows(frame: pd.DataFrame, risk_column: str, levels: set[str]) -> list[dict[str, Any]]:
        """Return records filtered by requested risk levels."""
        if frame.empty or risk_column not in frame.columns:
            return []
        return frame.loc[frame[risk_column].isin(levels)].to_dict(orient="records")

    def _derive_key_findings(
        self,
        mitra: dict[str, Any],
        varuna: dict[str, Any],
        feature_risk: pd.DataFrame,
        prediction_drift: dict[str, Any],
    ) -> list[str]:
        """Derive concise findings using only saved deterministic outputs."""
        findings: list[str] = []
        risk_features = set(feature_risk.get("feature", pd.Series(dtype=str)).astype(str))
        for row in mitra.get("high_drift_features", []):
            feature = str(row.get("feature"))
            suffix = " and appears in the feature risk matrix" if feature in risk_features else ""
            findings.append(f"{feature} is high drift{suffix}.")

        for row in feature_risk.to_dict(orient="records"):
            if row.get("final_risk") == "High":
                findings.append(
                    f"{row['feature']} is a high-risk feature because it is model-important and has "
                    f"{str(row.get('drift_level', 'Low')).lower()} drift."
                )

        prediction_level = prediction_drift.get("prediction_drift_level")
        if prediction_level and prediction_level != "Unknown":
            findings.append(
                f"Prediction drift is {prediction_level} based on score distribution shift and KS p-value."
            )

        for row in varuna.get("multicollinearity_findings", []):
            vif_risk = row.get("vif_risk", row.get("vif_level"))
            if vif_risk == "High":
                findings.append(f"VIF risk is High for {row['feature']}.")

        overfitting = varuna.get("overfitting_check", {})
        if overfitting.get("risk_level") not in {None, "Low", "Not Available"}:
            findings.append(
                "Train-validation AUC delta indicates "
                f"{str(overfitting['risk_level']).lower()} overfitting risk."
            )
        return list(dict.fromkeys(findings))

    def _recommended_actions(
        self,
        mitra: dict[str, Any],
        varuna: dict[str, Any],
        feature_risk: pd.DataFrame,
        data_quality: pd.DataFrame,
        prediction_drift: dict[str, Any],
    ) -> list[str]:
        """Build actions using deterministic risk rules only."""
        actions: list[str] = []
        if any(feature_risk.get("final_risk", pd.Series(dtype=str)) == "High"):
            actions.append("Review high-risk feature stability and consider recalibration before activation.")
        if prediction_drift.get("prediction_drift_level") == "High":
            actions.append("Run validation before campaign or model activation because prediction drift is High.")
        if not data_quality.empty and any(data_quality.get("issue_level", pd.Series(dtype=str)).isin(["Medium", "High"])):
            actions.append("Review upstream data pipelines for medium or high data-quality issues.")
        if any(
            row.get("vif_risk", row.get("vif_level")) == "High"
            for row in varuna.get("multicollinearity_findings", [])
        ):
            actions.append("Review feature redundancy and consider consolidating high-VIF features.")
        actions.extend(mitra.get("recommended_checks", []))
        return list(dict.fromkeys(actions))

    def build_packet(self) -> dict[str, Any]:
        """Build the complete verified evidence packet."""
        mitra = self._load_json(self.paths["mitra_output"])
        varuna = self._load_json(self.paths["varuna_output"])
        data_quality = self._load_csv(self.paths["data_quality_report"])
        drift = self._load_csv(self.paths["drift_report"])
        prediction_drift = self._load_json(self.paths["prediction_drift_report"])
        feature_risk = self._load_csv(self.paths["feature_risk_matrix"])
        model_diagnostics = self._load_json(self.paths["model_diagnostics"])
        calibration_report = self._load_csv(self.paths["calibration_report"])
        score_decile_report = self._load_csv(self.paths["score_decile_report"])
        lift_report = self._load_csv(self.paths["lift_report"])
        segment_performance_report = self._load_csv(self.paths["segment_performance_report"])
        model_metadata = self._load_json(self.paths["model_metadata"])
        feature_metadata = self._load_json(self.paths["feature_metadata"])
        config = self._load_json(self.paths["calibration_config"])
        run_id = str(varuna.get("run_id") or mitra.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        timestamp = str(varuna.get("timestamp") or datetime.now(timezone.utc).isoformat())
        config_version = str(config.get("config_version", mitra.get("config_version", "unknown")))
        plots_available = {
            name: self._available_plot(path)
            for name, path in varuna.get("plots_generated", {}).items()
        }
        vishwakarma = self._load_json(self.paths["vishwakarma_output"]) if self.paths["vishwakarma_output"].exists() else {}
        visuals_match_run = str(vishwakarma.get("run_id")) == run_id
        visuals_available = vishwakarma.get("visuals_generated", {}) if visuals_match_run else {}
        recommended_report_visuals = vishwakarma.get("recommended_report_visuals", []) if visuals_match_run else []

        mitra_summary = {
            "overall_risk_level": mitra.get("overall_risk_level"),
            "overall_risk_explanation": mitra.get("overall_risk_explanation"),
            "risk_assessment": mitra.get("risk_assessment", {}),
            "high_drift_features": mitra.get("high_drift_features", []),
            "medium_drift_features": mitra.get("medium_drift_features", []),
            "cluster_findings": mitra.get("cluster_findings", []),
            "recommended_checks": mitra.get("recommended_checks", []),
        }
        varuna_summary = {
            "agent": varuna.get("agent", "Varuna"),
            "reference_model_type": varuna.get("reference_model_type"),
            "explanation_method": varuna.get("explanation_method"),
            "explainability_reliability": varuna.get("explainability_reliability", {}),
            "overfitting_check": varuna.get("overfitting_check", {}),
            "multicollinearity_findings": varuna.get("multicollinearity_findings", []),
            "warnings": varuna.get("warnings", []),
        }
        high_risk_features = self._risk_rows(feature_risk, "final_risk", {"High"})
        key_findings = self._derive_key_findings(mitra, varuna, feature_risk, prediction_drift)
        performance_diagnostics = varuna.get(
            "performance_diagnostics",
            model_diagnostics.get("performance_diagnostics", {}),
        )
        limitations = [
            "Synthetic demo data only",
            "Not validated for production use",
            "No real customer or financial data used",
        ]
        source_files = {name: str(path) for name, path in self.paths.items() if name != "output"}

        self.packet = {
            "run_id": run_id,
            "timestamp": timestamp,
            "config_version": config_version,
            "model_metadata": model_metadata,
            "feature_metadata": feature_metadata,
            "source_files": source_files,
            "mitra_summary": mitra_summary,
            "varuna_summary": varuna_summary,
            "data_quality_summary": {
                "issue_counts": data_quality.get("issue_level", pd.Series(dtype=str)).value_counts().to_dict(),
                "findings": data_quality.to_dict(orient="records"),
            },
            "feature_drift_summary": {
                "high_drift_features": self._risk_rows(drift, "drift_level", {"High"}),
                "medium_drift_features": self._risk_rows(drift, "drift_level", {"Medium"}),
                "all_features": drift.to_dict(orient="records"),
            },
            "prediction_drift_summary": prediction_drift,
            "top_model_drivers": varuna.get("top_model_drivers", varuna.get("top_global_drivers", [])),
            "high_risk_features": high_risk_features,
            "feature_risk_matrix": feature_risk.to_dict(orient="records"),
            "model_diagnostics": model_diagnostics,
            "model_performance_summary": performance_diagnostics,
            "calibration_summary": {
                "diagnostics": performance_diagnostics.get("calibration", {}),
                "bins": calibration_report.to_dict(orient="records"),
            },
            "lift_summary": {
                "diagnostics": performance_diagnostics.get("lift", {}),
                "deciles": lift_report.to_dict(orient="records"),
            },
            "score_decile_summary": score_decile_report.to_dict(orient="records"),
            "segment_performance_summary": {
                "diagnostics": performance_diagnostics.get("segment_performance", {}),
                "segments": segment_performance_report.to_dict(orient="records"),
            },
            "key_findings": key_findings,
            "recommended_actions": self._recommended_actions(
                mitra,
                varuna,
                feature_risk,
                data_quality,
                prediction_drift,
            ),
            "plots_available": plots_available,
            "visuals_available": visuals_available,
            "recommended_report_visuals": recommended_report_visuals,
            "limitations": limitations,
            # Backward-compatible aliases for earlier narrative utilities.
            "signal_sentinel_summary": {
                **mitra_summary,
                "data_health_summary": mitra.get("data_health_summary", {}),
                "high_drift_feature_count": len(mitra.get("high_drift_features", [])),
                "medium_drift_feature_count": len(mitra.get("medium_drift_features", [])),
                "prediction_drift_summary": prediction_drift,
            },
            "model_lens_summary": {
                **varuna_summary,
                "top_global_drivers": varuna.get("top_model_drivers", varuna.get("top_global_drivers", [])),
                "high_risk_feature_matrix": feature_risk.to_dict(orient="records"),
            },
            "available_plots": plots_available,
            "business_context": {
                "business_use_case": model_metadata.get("business_use_case"),
                "decision_supported": model_metadata.get("decision_supported"),
                "target": model_metadata.get("target"),
                "entity_id": model_metadata.get("entity_id"),
                "prediction_column": model_metadata.get("prediction_column"),
                "training_window": model_metadata.get("training_window"),
                "current_window": model_metadata.get("current_window"),
            },
        }
        return self.packet

    def save(self) -> Path:
        """Write evidence_packet.json and return its path."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(self.build_packet(), indent=2), encoding="utf-8")
        return self.output_path

    def run(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build evidence as a future LangGraph-compatible state transformation."""
        next_state = dict(state or {})
        output_path = self.save()
        next_state["evidence_packet"] = self.packet
        next_state["evidence_packet_path"] = str(output_path)
        return next_state


# Backward-compatible alias used by earlier pipeline and tests.
EvidencePacketBuilder = EvidenceStoreBuilder


def parse_args() -> argparse.Namespace:
    """Parse Evidence Store CLI arguments."""
    parser = argparse.ArgumentParser(description="Build deterministic AxionAI evidence packet.")
    parser.add_argument("--use_case", choices=["fraud", "purchase"], help="Optional configured report folder.")
    return parser.parse_args()


def main() -> None:
    """Build and save reports/evidence_packet.json."""
    args = parse_args()
    output_path = EvidenceStoreBuilder(paths=default_paths(args.use_case)).save()
    print(f"Saved evidence packet to {output_path}")


if __name__ == "__main__":
    main()
