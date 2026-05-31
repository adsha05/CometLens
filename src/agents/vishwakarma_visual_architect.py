"""Agent 05: Vishwakarma deterministic visual intelligence generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.graph.graph_builder import build_default_model_lineage_graph
from src.graph.svg_renderer import render_lineage_svg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def default_paths(use_case: str | None = None) -> dict[str, Path]:
    """Return configurable Vishwakarma input and output paths."""
    use_case_reports = REPORTS_DIR / use_case if use_case else REPORTS_DIR
    reports_dir = use_case_reports if use_case and use_case_reports.exists() else REPORTS_DIR
    return {
        "evidence_packet": reports_dir / "evidence_packet.json",
        "mitra_output": reports_dir / "mitra_output.json",
        "varuna_output": reports_dir / "model_lens_output.json",
        "drift_report": reports_dir / "drift_report.csv",
        "prediction_drift_report": reports_dir / "prediction_drift_report.json",
        "shap_importance": reports_dir / "shap_global_importance.csv",
        "vif_report": reports_dir / "vif_report.csv",
        "feature_risk_matrix": reports_dir / "feature_risk_matrix.csv",
        "train_predictions": DATA_DIR / "train_predictions_sample.csv",
        "current_predictions": DATA_DIR / "current_predictions_sample.csv",
        "model_metadata": MODELS_DIR / "model_metadata.json",
        "calibration_config": CONFIGS_DIR / "calibration_config_v1.json",
        "visuals_dir": reports_dir / "visuals",
    }


class VishwakarmaVisualArchitect:
    """Build report-ready visuals from verified deterministic artifacts."""

    def __init__(
        self,
        paths: dict[str, str | Path] | None = None,
        config_path: str | Path = CONFIGS_DIR / "calibration_config_v1.json",
    ) -> None:
        """Configure artifact paths without changing upstream metric artifacts."""
        configured = default_paths()
        if paths:
            configured.update({key: Path(value) for key, value in paths.items()})
        configured["calibration_config"] = Path(config_path) if config_path else configured["calibration_config"]
        self.paths = {key: Path(value) for key, value in configured.items()}
        self.visuals_dir = self.paths["visuals_dir"]
        self.inputs: dict[str, Any] = {}
        self.visuals_generated: dict[str, dict[str, str]] = {}
        self.warnings: list[str] = []
        self.manifest: dict[str, Any] = {}

    @staticmethod
    def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
        """Load a JSON object with a helpful missing-file error."""
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required Vishwakarma input not found: {path}")
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}")
        return payload

    @staticmethod
    def _load_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
        """Load a CSV table with a helpful missing-file error."""
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required Vishwakarma input not found: {path}")
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_inputs(self) -> dict[str, Any]:
        """Load verified outputs and optional prediction logs."""
        self.inputs = {
            "evidence_packet": self._load_json(self.paths["evidence_packet"]),
            "mitra_output": self._load_json(self.paths["mitra_output"]),
            "varuna_output": self._load_json(self.paths["varuna_output"]),
            "drift_report": self._load_csv(self.paths["drift_report"]),
            "prediction_drift_report": self._load_json(self.paths["prediction_drift_report"]),
            "shap_importance": self._load_csv(self.paths["shap_importance"]),
            "vif_report": self._load_csv(self.paths["vif_report"]),
            "feature_risk_matrix": self._load_csv(self.paths["feature_risk_matrix"]),
            "train_predictions": self._load_csv(self.paths["train_predictions"], required=False),
            "current_predictions": self._load_csv(self.paths["current_predictions"]),
            "model_metadata": self._load_json(self.paths["model_metadata"]),
            "calibration_config": self._load_json(self.paths["calibration_config"]),
        }
        return self.inputs

    def _save_plotly_outputs(self, figure: go.Figure, stem: str) -> dict[str, str]:
        """Save interactive Plotly outputs and attempt optional static PNG export."""
        self.visuals_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.visuals_dir / f"{stem}.json"
        html_path = self.visuals_dir / f"{stem}.html"
        png_path = self.visuals_dir / f"{stem}.png"
        json_path.write_text(figure.to_json(), encoding="utf-8")
        figure.write_html(html_path, include_plotlyjs="cdn")
        outputs = {"json": str(json_path), "html": str(html_path)}
        try:
            figure.write_image(png_path)
        except Exception as error:
            self.warnings.append(
                f"Skipped optional PNG export for {stem}: install Kaleido for static Plotly image export "
                f"({type(error).__name__})."
            )
        else:
            outputs["png"] = str(png_path)
        return outputs

    def _feature_risk_frame(self) -> pd.DataFrame:
        """Return a plot-ready risk frame, merging saved artifacts when needed."""
        frame = self.inputs["feature_risk_matrix"].copy()
        if frame.empty:
            frame = self.inputs["shap_importance"].copy()
        sources = [
            (self.inputs["shap_importance"], ["feature", "mean_abs_shap", "shap_rank"]),
            (self.inputs["drift_report"], ["feature", "psi", "drift_level"]),
            (self.inputs["vif_report"], ["feature", "vif", "vif_risk"]),
        ]
        for source, columns in sources:
            missing = [column for column in columns if column != "feature" and column not in frame.columns]
            if missing and not source.empty:
                frame = frame.merge(source[["feature", *missing]], on="feature", how="left")
        defaults = {
            "mean_abs_shap": 0.0,
            "shap_rank": 999,
            "psi": 0.0,
            "vif": 1.0,
            "final_risk": "Unknown",
            "drift_level": "Unknown",
            "vif_risk": "Unknown",
            "reason": "No saved risk explanation available.",
        }
        for column, default in defaults.items():
            if column not in frame.columns:
                frame[column] = default
        numeric_vif = pd.to_numeric(frame["vif"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        frame["_marker_size"] = numeric_vif.fillna(1.0).clip(lower=1.0, upper=25.0) + 6.0
        return frame

    def build_feature_risk_scatter(self) -> go.Figure:
        """Build a SHAP-versus-PSI feature risk scatter from saved metric artifacts."""
        if not self.inputs:
            self.load_inputs()
        frame = self._feature_risk_frame()
        figure = px.scatter(
            frame,
            x="mean_abs_shap",
            y="psi",
            color="final_risk",
            size="_marker_size",
            hover_name="feature",
            hover_data={
                "shap_rank": True if "shap_rank" in frame.columns else False,
                "drift_level": True,
                "vif": True,
                "vif_risk": True,
                "reason": True,
                "_marker_size": False,
            },
            color_discrete_map={"High": "#dc2626", "Medium": "#f97316", "Low": "#16a34a", "Unknown": "#6b7280"},
            title="Feature Risk Map: Model Importance vs. Distribution Drift",
            labels={"mean_abs_shap": "Mean absolute SHAP value", "psi": "PSI", "final_risk": "Feature risk"},
        )
        figure.add_annotation(x=0.98, y=0.96, xref="paper", yref="paper", text="Danger Zone", showarrow=False)
        figure.add_annotation(x=0.02, y=0.04, xref="paper", yref="paper", text="Stable Zone", showarrow=False)
        figure.update_layout(template="plotly_white")
        self.visuals_generated["feature_risk_scatter"] = self._save_plotly_outputs(figure, "feature_risk_scatter")
        return figure

    def build_prediction_distribution_overlay(self) -> go.Figure | None:
        """Build a reference-versus-current prediction score overlay when both logs exist."""
        if not self.inputs:
            self.load_inputs()
        train_predictions = self.inputs["train_predictions"]
        current_predictions = self.inputs["current_predictions"]
        prediction_column = str(self.inputs["model_metadata"].get("prediction_column", ""))
        if train_predictions.empty:
            self.warnings.append("Skipped prediction distribution overlay: reference prediction log is unavailable.")
            return None
        if not prediction_column:
            candidates = [column for column in current_predictions.columns if "score" in column or "proba" in column]
            prediction_column = candidates[0] if candidates else ""
        if prediction_column not in train_predictions or prediction_column not in current_predictions:
            self.warnings.append(
                f"Skipped prediction distribution overlay: prediction column `{prediction_column}` is unavailable."
            )
            return None
        train_scores = pd.to_numeric(train_predictions[prediction_column], errors="coerce").dropna()
        current_scores = pd.to_numeric(current_predictions[prediction_column], errors="coerce").dropna()
        if train_scores.empty or current_scores.empty:
            self.warnings.append("Skipped prediction distribution overlay: prediction scores are empty after cleaning.")
            return None
        figure = go.Figure()
        figure.add_trace(go.Histogram(x=train_scores, name="Reference scores", opacity=0.65, histnorm="probability"))
        figure.add_trace(go.Histogram(x=current_scores, name="Current scores", opacity=0.65, histnorm="probability"))
        prediction_drift = self.inputs["prediction_drift_report"]
        figure.add_annotation(
            x=0.98,
            y=0.97,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="top",
            align="left",
            showarrow=False,
            text=(
                f"Reference mean: {train_scores.mean():.3f}<br>"
                f"Current mean: {current_scores.mean():.3f}<br>"
                f"Saved drift level: {prediction_drift.get('prediction_drift_level', 'Unknown')}"
            ),
        )
        figure.update_layout(
            barmode="overlay",
            template="plotly_white",
            title="Prediction Distribution Overlay",
            xaxis_title=prediction_column,
            yaxis_title="Probability share",
        )
        self.visuals_generated["prediction_distribution_overlay"] = self._save_plotly_outputs(
            figure, "prediction_distribution_overlay"
        )
        return figure

    def build_lineage_graph(self) -> dict[str, Any]:
        """Build and save JSON and SVG lineage graph outputs."""
        if not self.inputs:
            self.load_inputs()
        graph_spec = build_default_model_lineage_graph(
            self.inputs["evidence_packet"],
            self.inputs["mitra_output"],
            self.inputs["varuna_output"],
        )
        self.visuals_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.visuals_dir / "lineage_graph.json"
        svg_path = self.visuals_dir / "lineage_graph.svg"
        json_path.write_text(json.dumps(graph_spec, indent=2), encoding="utf-8")
        render_lineage_svg(graph_spec, svg_path)
        self.visuals_generated["lineage_graph"] = {"json": str(json_path), "svg": str(svg_path)}
        return graph_spec

    def build_visual_manifest(self) -> dict[str, Any]:
        """Build an auditable visual manifest with sources and warnings."""
        if not self.inputs:
            self.load_inputs()
        evidence = self.inputs["evidence_packet"]
        model_metadata = self.inputs["model_metadata"]
        self.manifest = {
            "agent": "Vishwakarma",
            "run_id": evidence.get("run_id", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": model_metadata.get("model_name", "unknown"),
            "config_version": evidence.get("config_version", "unknown"),
            "visuals_generated": self.visuals_generated,
            "recommended_report_visuals": [
                visual
                for visual in ["feature_risk_scatter", "prediction_distribution_overlay", "lineage_graph"]
                if visual in self.visuals_generated
            ],
            "warnings": list(dict.fromkeys(self.warnings)),
            "source_files": {
                name: str(path)
                for name, path in self.paths.items()
                if name not in {"visuals_dir"} and path.exists()
            },
        }
        return self.manifest

    def save_outputs(self) -> Path:
        """Generate and save all Vishwakarma visual intelligence outputs."""
        self.load_inputs()
        self.build_feature_risk_scatter()
        self.build_prediction_distribution_overlay()
        self.build_lineage_graph()
        manifest = self.build_visual_manifest()
        manifest_path = self.visuals_dir / "vishwakarma_output.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def run(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run Vishwakarma as a future graph-orchestrator-compatible state transformation."""
        next_state = dict(state or {})
        manifest_path = self.save_outputs()
        next_state["vishwakarma"] = self.manifest
        next_state["vishwakarma_output_path"] = str(manifest_path)
        return next_state


def parse_args() -> argparse.Namespace:
    """Parse Vishwakarma CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Agent 05: Vishwakarma visual architect.")
    parser.add_argument("--use_case", choices=["fraud", "purchase"], help="Optional configured report folder.")
    return parser.parse_args()


def main() -> None:
    """Run Vishwakarma from the command line."""
    args = parse_args()
    manifest_path = VishwakarmaVisualArchitect(paths=default_paths(args.use_case)).save_outputs()
    print(f"Saved Vishwakarma visual manifest to {manifest_path}")


if __name__ == "__main__":
    main()
