"""Agent 02: Varuna for deterministic explainability and model-risk diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.evidence_store import EvidenceStore
from src.diagnostics.calibration import build_calibration_report, save_calibration_curve
from src.diagnostics.explainability import (
    compute_shap_importance,
    get_feature_columns,
    save_shap_plots,
    train_reference_model,
)
from src.diagnostics.lift import build_score_decile_report, save_lift_chart
from src.diagnostics.multicollinearity import calculate_vif
from src.diagnostics.overfitting import calculate_overfitting_delta
from src.diagnostics.segment_performance import (
    build_segment_performance_report,
    save_segment_performance_heatmap,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "calibration_config_v1.json"


def default_paths(use_case: str | None = None) -> dict[str, Path]:
    """Return configurable Varuna artifact paths with current repo defaults."""
    use_case_data_dir = DATA_DIR / use_case if use_case else DATA_DIR
    use_case_reports_dir = REPORTS_DIR / use_case if use_case else REPORTS_DIR
    data_dir = use_case_data_dir if use_case and use_case_data_dir.exists() else DATA_DIR
    reports_dir = use_case_reports_dir if use_case and use_case_reports_dir.exists() else REPORTS_DIR
    return {
        "train_features": data_dir / "train_features_sample.csv",
        "current_features": data_dir / "current_features_sample.csv",
        "current_predictions": data_dir / "current_predictions_sample.csv",
        "drift_report": reports_dir / "drift_report.csv",
        "mitra_output": reports_dir / "mitra_output.json",
        "model_metadata": MODELS_DIR / "model_metadata.json",
        "feature_metadata": MODELS_DIR / "feature_metadata.json",
        "reports_dir": reports_dir,
        "figures_dir": reports_dir / "figures",
    }


class ModelLensAgent:
    """Coordinate deterministic explainability, VIF, overfitting, and feature risk checks."""

    def __init__(
        self,
        paths: dict[str, str | Path] | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        **legacy_paths: str | Path,
    ) -> None:
        """Configure Varuna while accepting earlier keyword path arguments."""
        configured = default_paths()
        if paths:
            configured.update({key: Path(value) for key, value in paths.items()})
        legacy_mapping = {
            "train_features_path": "train_features",
            "current_features_path": "current_features",
            "predictions_path": "current_predictions",
            "model_metadata_path": "model_metadata",
            "feature_metadata_path": "feature_metadata",
            "signal_sentinel_path": "mitra_output",
        }
        for legacy_key, target_key in legacy_mapping.items():
            if legacy_key in legacy_paths:
                configured[target_key] = Path(legacy_paths[legacy_key])

        self.paths = {key: Path(value) for key, value in configured.items()}
        self.config_path = Path(config_path)
        self.train_features_path = self.paths["train_features"]
        self.current_features_path = self.paths["current_features"]
        self.predictions_path = self.paths["current_predictions"]
        self.drift_report_path = self.paths["drift_report"]
        self.model_metadata_path = self.paths["model_metadata"]
        self.feature_metadata_path = self.paths["feature_metadata"]
        self.signal_sentinel_path = self.paths["mitra_output"]
        self.reports_dir = self.paths["reports_dir"]
        self.figures_dir = self.paths["figures_dir"]

        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.config: dict[str, Any] = {}
        self.config_version = "unknown"
        self.train_features = pd.DataFrame()
        self.current_features = pd.DataFrame()
        self.predictions = pd.DataFrame()
        self.drift_report = pd.DataFrame()
        self.model_metadata: dict[str, Any] = {}
        self.feature_metadata: dict[str, Any] = {}
        self.signal_sentinel: dict[str, Any] = {}
        self.target_col = ""
        self.entity_id_col = ""
        self.feature_cols: list[str] = []
        self.model: Any = None
        self.reference_model_type = "unknown"
        self.shap_importance = pd.DataFrame()
        self.vif_report = pd.DataFrame()
        self.feature_risk_matrix = pd.DataFrame()
        self.high_risk_feature_matrix = pd.DataFrame()
        self.overfitting_check: dict[str, Any] = {}
        self.calibration_report = pd.DataFrame()
        self.score_decile_report = pd.DataFrame()
        self.lift_report = pd.DataFrame()
        self.segment_performance_report = pd.DataFrame()
        self.performance_diagnostics: dict[str, Any] = {}
        self.explainability_reliability: dict[str, Any] = {}
        self.explanation_method = "unknown"
        self.plots_generated: dict[str, str] = {}
        self.warnings: list[str] = []
        self.output: dict[str, Any] = {}

    def load_inputs(self) -> None:
        """Load model artifacts, Mitra outputs, drift report, metadata, and config."""
        required_paths = [
            self.train_features_path,
            self.current_features_path,
            self.predictions_path,
            self.drift_report_path,
            self.signal_sentinel_path,
            self.model_metadata_path,
            self.feature_metadata_path,
            self.config_path,
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing Agent 02: Varuna inputs. Run sample generation and Agent 01: Mitra first. Missing: "
                + ", ".join(missing)
            )

        self.train_features = pd.read_csv(self.train_features_path)
        self.current_features = pd.read_csv(self.current_features_path)
        self.predictions = pd.read_csv(self.predictions_path)
        self.drift_report = pd.read_csv(self.drift_report_path)
        self.model_metadata = json.loads(self.model_metadata_path.read_text(encoding="utf-8"))
        self.feature_metadata = json.loads(self.feature_metadata_path.read_text(encoding="utf-8"))
        self.signal_sentinel = json.loads(self.signal_sentinel_path.read_text(encoding="utf-8"))
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.config_version = str(self.config.get("config_version", "unknown"))
        self.target_col = str(self.model_metadata.get("target") or self.feature_metadata.get("target", ""))
        self.entity_id_col = str(
            self.model_metadata.get("entity_id") or self.feature_metadata.get("entity_id", "")
        )

        metadata_features = [
            feature
            for feature in self.model_metadata.get("feature_columns", [])
            if feature in self.train_features.columns and feature in self.current_features.columns
        ]
        inferred = get_feature_columns(self.train_features, self.target_col, self.entity_id_col)
        self.feature_cols = [feature for feature in metadata_features if feature in inferred] or inferred
        if not self.feature_cols:
            raise ValueError("Agent 02: Varuna could not identify numeric model feature columns.")

    def _source_files(self) -> dict[str, str]:
        """Return source paths for auditable output metadata."""
        return {
            "train_features": str(self.train_features_path),
            "current_features": str(self.current_features_path),
            "current_predictions": str(self.predictions_path),
            "drift_report": str(self.drift_report_path),
            "mitra_output": str(self.signal_sentinel_path),
            "model_metadata": str(self.model_metadata_path),
            "feature_metadata": str(self.feature_metadata_path),
            "calibration_config": str(self.config_path),
        }

    def assess_explainability_reliability(self) -> dict[str, Any]:
        """Use config-driven severe drift gates to label explainability reliability."""
        if not self.signal_sentinel:
            self.load_inputs()
        varuna = self.config.get("varuna", {})
        severe_psi = float(varuna.get("severe_psi_threshold", 0.50))
        severe_ks = float(varuna.get("severe_ks_pvalue_threshold", 0.001))
        mode = os.getenv("VARUNA_DRIFT_GATE_MODE", str(varuna.get("drift_gate_mode", "flag"))).lower()
        severe_features = []
        for row in self.signal_sentinel.get("high_drift_features", []):
            psi = float(row.get("psi", 0.0))
            ks_pvalue = float(row.get("ks_pvalue", 1.0))
            if psi >= severe_psi or ks_pvalue < severe_ks:
                severe_features.append(
                    {
                        "feature": row.get("feature"),
                        "psi": psi,
                        "ks_pvalue": ks_pvalue,
                        "reason": "PSI or KS p-value crossed configured severe drift gate.",
                    }
                )
        self.explainability_reliability = {
            "status": "unreliable" if severe_features else "reliable",
            "gate_mode": mode,
            "should_skip": bool(severe_features) and mode == "skip",
            "severe_psi_threshold": severe_psi,
            "severe_ks_pvalue_threshold": severe_ks,
            "severe_drift_features": severe_features,
            "message": (
                "SHAP outputs are directional only because Mitra found severe input drift."
                if severe_features
                else "No severe Mitra drift gate was triggered."
            ),
        }
        return self.explainability_reliability

    def run_shap_analysis(self) -> pd.DataFrame:
        """Train the local reviewer and calculate explainability importance."""
        if self.train_features.empty:
            self.load_inputs()
        self.model, self.reference_model_type = train_reference_model(
            self.train_features,
            self.feature_cols,
            self.target_col,
        )
        self.shap_importance = compute_shap_importance(self.model, self.current_features, self.feature_cols)
        self.explanation_method = str(self.shap_importance.attrs.get("explanation_method", "unknown"))
        warning = self.shap_importance.attrs.get("warning")
        if warning:
            self.warnings.append(str(warning))
        plot_result = save_shap_plots(
            self.model,
            self.current_features,
            self.feature_cols,
            self.figures_dir,
            importance_df=self.shap_importance,
        )
        self.plots_generated.update(plot_result["plots_generated"])
        self.warnings.extend(plot_result["warnings"])

        bar_path = self.figures_dir / "shap_bar.png"
        legacy_bar_path = self.figures_dir / "shap_global_bar.png"
        if bar_path.exists():
            shutil.copyfile(bar_path, legacy_bar_path)
            self.plots_generated["shap_global_bar"] = str(legacy_bar_path)
        return self.shap_importance

    def run_vif_analysis(self) -> pd.DataFrame:
        """Run config-driven VIF diagnostics."""
        if self.train_features.empty:
            self.load_inputs()
        self.vif_report = calculate_vif(self.train_features, self.feature_cols, self.config)
        self.vif_report["vif_level"] = self.vif_report["vif_risk"]
        self.vif_report["vif_level_reason"] = self.vif_report["vif_risk_reason"]
        return self.vif_report

    def run_overfitting_check(self) -> dict[str, Any]:
        """Calculate config-driven train-validation AUC delta."""
        if not self.model_metadata:
            self.load_inputs()
        self.overfitting_check = calculate_overfitting_delta(self.model_metadata, self.config)
        self.overfitting_check["risk_level_reason"] = self.overfitting_check["reason"]
        return self.overfitting_check

    def run_performance_diagnostics(self) -> dict[str, Any]:
        """Calculate current-window calibration, lift, score-decile, and segment diagnostics."""
        if self.predictions.empty:
            self.load_inputs()
        prediction_column = str(self.model_metadata.get("prediction_column", ""))
        label_column = "actual_label" if "actual_label" in self.predictions.columns else self.target_col
        self.calibration_report, calibration_summary = build_calibration_report(
            self.predictions,
            prediction_column,
            label_column,
        )
        self.score_decile_report, lift_summary = build_score_decile_report(
            self.predictions,
            prediction_column,
            label_column,
        )
        self.lift_report = self.score_decile_report.copy()
        segment_features = (
            self.shap_importance["feature"].head(5).tolist()
            if not self.shap_importance.empty and "feature" in self.shap_importance.columns
            else self.feature_cols[:5]
        )
        self.segment_performance_report, segment_summary = build_segment_performance_report(
            self.current_features,
            self.predictions,
            segment_features,
            self.entity_id_col,
            prediction_column,
            label_column,
            config=self.config,
        )
        calibration_plot = save_calibration_curve(
            self.calibration_report,
            self.figures_dir / "calibration_curve.png",
        )
        lift_plot = save_lift_chart(
            self.lift_report,
            self.figures_dir / "lift_chart.png",
        )
        segment_plot = save_segment_performance_heatmap(
            self.segment_performance_report,
            self.figures_dir / "segment_performance_heatmap.png",
        )
        self.plots_generated["calibration_curve"] = str(calibration_plot)
        self.plots_generated["lift_chart"] = str(lift_plot)
        self.plots_generated["segment_performance_heatmap"] = str(segment_plot)
        self.performance_diagnostics = {
            "calibration": calibration_summary,
            "lift": lift_summary,
            "segment_performance": segment_summary,
            "score_deciles_available": bool(not self.score_decile_report.empty),
        }
        if not calibration_summary.get("available"):
            self.warnings.append(str(calibration_summary.get("reason")))
        if not lift_summary.get("available"):
            self.warnings.append(str(lift_summary.get("reason")))
        if not segment_summary.get("available"):
            self.warnings.append(str(segment_summary.get("reason")))
        return self.performance_diagnostics

    def build_feature_risk_matrix(self) -> pd.DataFrame:
        """Combine SHAP, Mitra drift, VIF, and feature metadata into one risk matrix."""
        if self.shap_importance.empty:
            self.run_shap_analysis()
        if self.vif_report.empty:
            self.run_vif_analysis()
        metadata = {
            row.get("name"): row
            for row in self.feature_metadata.get("features", [])
            if isinstance(row, dict) and row.get("name")
        }
        drift_columns = ["feature", "psi", "ks_pvalue", "wasserstein_distance", "drift_level"]
        available_drift_columns = [column for column in drift_columns if column in self.drift_report.columns]
        matrix = self.shap_importance.merge(
            self.drift_report[available_drift_columns],
            on="feature",
            how="left",
        ).merge(
            self.vif_report[["feature", "vif", "vif_risk"]],
            on="feature",
            how="left",
        )
        matrix["feature_group"] = matrix["feature"].map(
            lambda feature: metadata.get(feature, {}).get("feature_group", "model_feature")
        )
        matrix["business_definition"] = matrix["feature"].map(
            lambda feature: metadata.get(feature, {}).get("business_definition", "")
        )
        matrix["psi"] = matrix["psi"].fillna(0.0)
        matrix["ks_pvalue"] = matrix["ks_pvalue"].fillna(1.0)
        matrix["wasserstein_distance"] = matrix["wasserstein_distance"].fillna(0.0)
        matrix["drift_level"] = matrix["drift_level"].fillna("Low")
        matrix["vif"] = matrix["vif"].fillna(1.0)
        matrix["vif_risk"] = matrix["vif_risk"].fillna("Low")

        def classify(row: pd.Series) -> tuple[str, str, str]:
            top_five = int(row["shap_rank"]) <= 5
            top_ten = int(row["shap_rank"]) <= 10
            if top_five and row["drift_level"] == "High":
                return (
                    "High",
                    "High because the feature is a top-5 SHAP driver with High drift.",
                    "Review feature stability and consider recalibration before activation.",
                )
            if top_five and row["vif_risk"] == "High":
                return (
                    "High",
                    "High because the feature is a top-5 SHAP driver with High VIF risk.",
                    "Review redundancy and consider combining or removing correlated features.",
                )
            if top_ten and row["drift_level"] == "Medium":
                return (
                    "Medium",
                    "Medium because the feature is a top-10 SHAP driver with Medium drift.",
                    "Monitor feature drift and validate stability in the next review window.",
                )
            if not top_five and row["drift_level"] == "High":
                return (
                    "Medium",
                    "Medium because the feature has High drift but is not a top-5 SHAP driver.",
                    "Monitor feature drift; lower immediate model-risk impact.",
                )
            return "Low", "Low because no configured feature-risk rule fired.", "No immediate action."

        classified = matrix.apply(classify, axis=1, result_type="expand")
        matrix[["final_risk", "reason", "recommended_action"]] = classified
        matrix["combined_risk"] = matrix["final_risk"]
        matrix["combined_risk_reason"] = matrix["reason"]
        matrix["vif_warning"] = matrix["vif_risk"]
        risk_order = {"High": 0, "Medium": 1, "Low": 2}
        self.feature_risk_matrix = (
            matrix[
                [
                    "feature",
                    "feature_group",
                    "business_definition",
                    "shap_rank",
                    "mean_abs_shap",
                    "mean_abs_shap_value",
                    "psi",
                    "ks_pvalue",
                    "wasserstein_distance",
                    "drift_level",
                    "vif",
                    "vif_risk",
                    "vif_warning",
                    "final_risk",
                    "combined_risk",
                    "reason",
                    "combined_risk_reason",
                    "recommended_action",
                ]
            ]
            .assign(_risk_order=lambda frame: frame["final_risk"].map(risk_order))
            .sort_values(["_risk_order", "shap_rank"])
            .drop(columns="_risk_order")
            .reset_index(drop=True)
        )
        self.high_risk_feature_matrix = self.feature_risk_matrix
        return self.feature_risk_matrix

    def build_high_risk_feature_matrix(self) -> pd.DataFrame:
        """Backward-compatible alias for the richer feature risk matrix."""
        return self.build_feature_risk_matrix()

    def build_output(self) -> dict[str, Any]:
        """Build final auditable Varuna output."""
        if self.train_features.empty:
            self.load_inputs()
        reliability = self.assess_explainability_reliability()
        if not reliability["should_skip"]:
            if self.shap_importance.empty:
                self.run_shap_analysis()
            if self.vif_report.empty:
                self.run_vif_analysis()
            if self.feature_risk_matrix.empty:
                self.build_feature_risk_matrix()
        if not self.overfitting_check:
            self.run_overfitting_check()
        if not self.performance_diagnostics:
            self.run_performance_diagnostics()

        top_drivers = self.shap_importance.head(10).to_dict(orient="records")
        high_risk = self.feature_risk_matrix.to_dict(orient="records")
        vif_findings = self.vif_report.to_dict(orient="records")
        self.output = {
            "agent": "Varuna",
            "agent_name": "Agent 02: Varuna",
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config_version": self.config_version,
            "reference_model_type": self.reference_model_type,
            "explanation_method": self.explanation_method,
            "explainability_reliability": reliability,
            "top_model_drivers": top_drivers,
            "top_global_drivers": top_drivers,
            "high_risk_feature_matrix": high_risk,
            "multicollinearity_findings": vif_findings,
            "overfitting_check": self.overfitting_check,
            "performance_diagnostics": self.performance_diagnostics,
            "plots_generated": self.plots_generated,
            "warnings": list(dict.fromkeys(self.warnings)),
            "source_files": self._source_files(),
            "skipped": reliability["should_skip"],
        }
        return self.output

    def save_outputs(self) -> dict[str, Path]:
        """Run Varuna and save JSON, CSV, and PNG artifacts."""
        output = self.build_output()
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.reports_dir / "model_lens_output.json"
        varuna_path = self.reports_dir / "varuna_output.json"
        shap_path = self.reports_dir / "shap_global_importance.csv"
        vif_path = self.reports_dir / "vif_report.csv"
        diagnostics_path = self.reports_dir / "model_diagnostics.json"
        risk_matrix_path = self.reports_dir / "feature_risk_matrix.csv"
        calibration_path = self.reports_dir / "calibration_report.csv"
        score_decile_path = self.reports_dir / "score_decile_report.csv"
        lift_report_path = self.reports_dir / "lift_report.csv"
        segment_performance_path = self.reports_dir / "segment_performance_report.csv"

        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        varuna_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        diagnostics_path.write_text(
            json.dumps(
                {
                    "agent": "Varuna",
                    "run_id": self.run_id,
                    "timestamp": self.timestamp,
                    "config_version": self.config_version,
                    "explanation_method": self.explanation_method,
                    "explainability_reliability": self.explainability_reliability,
                    "overfitting_check": self.overfitting_check,
                    "performance_diagnostics": self.performance_diagnostics,
                    "warnings": output["warnings"],
                    "source_files": self._source_files(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.shap_importance.to_csv(shap_path, index=False)
        self.vif_report.to_csv(vif_path, index=False)
        self.feature_risk_matrix.to_csv(risk_matrix_path, index=False)
        self.calibration_report.to_csv(calibration_path, index=False)
        self.score_decile_report.to_csv(score_decile_path, index=False)
        self.lift_report.to_csv(lift_report_path, index=False)
        self.segment_performance_report.to_csv(segment_performance_path, index=False)
        store = EvidenceStore()
        store.save_section("varuna", output)
        store.save_section("model_lens", output)
        return {
            "json": output_path,
            "varuna_json": varuna_path,
            "shap_global_importance": shap_path,
            "vif_report": vif_path,
            "model_diagnostics": diagnostics_path,
            "feature_risk_matrix": risk_matrix_path,
            "calibration_report": calibration_path,
            "score_decile_report": score_decile_path,
            "lift_report": lift_report_path,
            "segment_performance_report": segment_performance_path,
            "shap_bar": self.figures_dir / "shap_bar.png",
            "shap_beeswarm": self.figures_dir / "shap_beeswarm.png",
            "calibration_curve": self.figures_dir / "calibration_curve.png",
            "lift_chart": self.figures_dir / "lift_chart.png",
            "segment_performance_heatmap": self.figures_dir / "segment_performance_heatmap.png",
        }

    def run(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run Varuna as a future LangGraph-compatible state transformation."""
        next_state = dict(state or {})
        output_paths = self.save_outputs()
        next_state["varuna"] = self.output
        next_state["varuna_output_paths"] = {name: str(path) for name, path in output_paths.items()}
        return next_state


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Agent 02: Varuna model diagnostics.")
    parser.add_argument("--use_case", choices=["fraud", "purchase"], help="Optional configured use-case artifact folder.")
    return parser.parse_args()


def main() -> None:
    """Run Agent 02: Varuna from the command line."""
    args = parse_args()
    output_paths = ModelLensAgent(paths=default_paths(args.use_case)).save_outputs()
    print("Saved Agent 02: Varuna outputs:")
    for label, path in output_paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
