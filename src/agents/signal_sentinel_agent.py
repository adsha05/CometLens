"""Agent 01: Mitra for deterministic drift and data-health monitoring."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cometlens-matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.evidence_store import EvidenceStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RANDOM_SEED = 42


class SignalSentinelAgent:
    """Detect deterministic signal, prediction, missingness, and cluster shifts."""

    def __init__(
        self,
        train_features_path: Path = DATA_DIR / "train_features_sample.csv",
        current_features_path: Path = DATA_DIR / "current_features_sample.csv",
        predictions_path: Path = DATA_DIR / "current_predictions_sample.csv",
        model_metadata_path: Path = MODELS_DIR / "model_metadata.json",
        feature_metadata_path: Path = MODELS_DIR / "feature_metadata.json",
    ) -> None:
        """Configure input and output artifacts."""
        self.train_features_path = Path(train_features_path)
        self.current_features_path = Path(current_features_path)
        self.predictions_path = Path(predictions_path)
        self.model_metadata_path = Path(model_metadata_path)
        self.feature_metadata_path = Path(feature_metadata_path)
        self.train_features: pd.DataFrame | None = None
        self.current_features: pd.DataFrame | None = None
        self.predictions: pd.DataFrame | None = None
        self.model_metadata: dict[str, Any] = {}
        self.feature_metadata: dict[str, Any] = {}
        self.target_col: str | None = None
        self.entity_id_col: str | None = None
        self.prediction_col: str = "propensity_score"
        self.feature_cols: list[str] = []
        self.missing_value_summary: pd.DataFrame = pd.DataFrame()
        self.drift_report: pd.DataFrame = pd.DataFrame()
        self.prediction_drift_summary: dict[str, Any] = {}
        self.cluster_shift_report: pd.DataFrame = pd.DataFrame()
        self.output: dict[str, Any] = {}

    def load_inputs(self) -> None:
        """Load feature, prediction, and feature metadata artifacts."""
        required_paths = [
            self.train_features_path,
            self.current_features_path,
            self.predictions_path,
            self.model_metadata_path,
            self.feature_metadata_path,
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing Agent 01: Mitra input artifacts. Run "
                "`python src/generate_sample_artifacts.py` first. Missing: "
                + ", ".join(missing)
            )

        self.train_features = pd.read_csv(self.train_features_path)
        self.current_features = pd.read_csv(self.current_features_path)
        self.predictions = pd.read_csv(self.predictions_path)
        with self.model_metadata_path.open("r", encoding="utf-8") as input_file:
            self.model_metadata = json.load(input_file)
        with self.feature_metadata_path.open("r", encoding="utf-8") as input_file:
            self.feature_metadata = json.load(input_file)

        self.target_col = self.model_metadata.get("target") or self.feature_metadata.get("target")
        self.entity_id_col = self.model_metadata.get("entity_id") or self.feature_metadata.get("entity_id")
        self.prediction_col = self.model_metadata.get("prediction_column", self.prediction_col)

        metadata_features = self.model_metadata.get("feature_columns") or [
            feature["name"] for feature in self.feature_metadata.get("features", [])
        ]
        numeric_features = self.train_features.select_dtypes(include=np.number).columns.tolist()
        excluded_columns = {column for column in [self.target_col, self.entity_id_col] if column}
        self.feature_cols = [
            feature
            for feature in metadata_features
            if feature in numeric_features and feature in self.current_features.columns
        ]
        if not self.feature_cols:
            self.feature_cols = [
                column
                for column in numeric_features
                if column not in excluded_columns and column in self.current_features.columns
            ]

    @staticmethod
    def _clean_numeric(values: pd.Series | np.ndarray) -> np.ndarray:
        """Return finite numeric values for statistical checks."""
        numeric = pd.to_numeric(pd.Series(values), errors="coerce")
        return numeric.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)

    @staticmethod
    def calculate_psi(expected: pd.Series | np.ndarray, actual: pd.Series | np.ndarray, buckets: int = 10) -> float:
        """Calculate Population Stability Index between expected and actual arrays."""
        expected_values = SignalSentinelAgent._clean_numeric(expected)
        actual_values = SignalSentinelAgent._clean_numeric(actual)
        if len(expected_values) == 0 or len(actual_values) == 0:
            return 0.0
        if np.nanstd(expected_values) == 0 and np.nanstd(actual_values) == 0:
            return 0.0 if np.nanmean(expected_values) == np.nanmean(actual_values) else 1.0

        quantiles = np.linspace(0, 1, buckets + 1)
        breakpoints = np.unique(np.quantile(expected_values, quantiles))
        if len(breakpoints) <= 2:
            low = min(expected_values.min(), actual_values.min())
            high = max(expected_values.max(), actual_values.max())
            if low == high:
                return 0.0
            breakpoints = np.linspace(low, high, buckets + 1)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_counts, _ = np.histogram(expected_values, bins=breakpoints)
        actual_counts, _ = np.histogram(actual_values, bins=breakpoints)
        expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
        actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
        return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

    @staticmethod
    def calculate_ks(expected: pd.Series | np.ndarray, actual: pd.Series | np.ndarray) -> tuple[float, float]:
        """Calculate Kolmogorov-Smirnov statistic and p-value."""
        expected_values = SignalSentinelAgent._clean_numeric(expected)
        actual_values = SignalSentinelAgent._clean_numeric(actual)
        if len(expected_values) == 0 or len(actual_values) == 0:
            return 0.0, 1.0
        if np.nanstd(expected_values) == 0 and np.nanstd(actual_values) == 0:
            return (0.0, 1.0) if np.nanmean(expected_values) == np.nanmean(actual_values) else (1.0, 0.0)
        result = ks_2samp(expected_values, actual_values, method="asymp")
        return float(result.statistic), float(result.pvalue)

    @staticmethod
    def calculate_wasserstein(expected: pd.Series | np.ndarray, actual: pd.Series | np.ndarray) -> float:
        """Calculate Wasserstein distance between two numeric distributions."""
        expected_values = SignalSentinelAgent._clean_numeric(expected)
        actual_values = SignalSentinelAgent._clean_numeric(actual)
        if len(expected_values) == 0 or len(actual_values) == 0:
            return 0.0
        return float(wasserstein_distance(expected_values, actual_values))

    @staticmethod
    def _drift_level(psi: float, ks_pvalue: float) -> str:
        """Classify drift from PSI and KS thresholds."""
        if psi >= 0.25 or ks_pvalue < 0.01:
            return "High"
        if psi >= 0.10:
            return "Medium"
        return "Low"

    @staticmethod
    def _mean_change_pct(train_mean: float, current_mean: float) -> float:
        """Calculate percentage mean change while handling zero baselines."""
        if pd.isna(train_mean) or pd.isna(current_mean):
            return 0.0
        if abs(train_mean) < 1e-12:
            return 0.0 if abs(current_mean) < 1e-12 else float(np.sign(current_mean) * 100.0)
        return float(((current_mean - train_mean) / abs(train_mean)) * 100)

    def run_missing_value_checks(self) -> pd.DataFrame:
        """Compare missing rates between train and current feature tables."""
        if self.train_features is None or self.current_features is None:
            self.load_inputs()
        assert self.train_features is not None
        assert self.current_features is not None

        rows = []
        for feature in self.feature_cols:
            train_missing = float(self.train_features[feature].isna().mean())
            current_missing = float(self.current_features[feature].isna().mean())
            rows.append(
                {
                    "feature": feature,
                    "train_missing_rate": train_missing,
                    "current_missing_rate": current_missing,
                    "missing_rate_change_pct_points": (current_missing - train_missing) * 100,
                }
            )
        self.missing_value_summary = pd.DataFrame(rows)
        return self.missing_value_summary

    def run_feature_drift_checks(self) -> pd.DataFrame:
        """Run PSI, KS, Wasserstein, and mean-shift checks for every feature."""
        if self.train_features is None or self.current_features is None:
            self.load_inputs()
        assert self.train_features is not None
        assert self.current_features is not None

        rows = []
        for feature in self.feature_cols:
            train_values = self.train_features[feature]
            current_values = self.current_features[feature]
            train_mean = float(pd.to_numeric(train_values, errors="coerce").mean())
            current_mean = float(pd.to_numeric(current_values, errors="coerce").mean())
            psi = self.calculate_psi(train_values, current_values)
            ks_statistic, ks_pvalue = self.calculate_ks(train_values, current_values)
            wasserstein = self.calculate_wasserstein(train_values, current_values)
            rows.append(
                {
                    "feature": feature,
                    "train_mean": train_mean,
                    "current_mean": current_mean,
                    "mean_change_pct": self._mean_change_pct(train_mean, current_mean),
                    "psi": psi,
                    "ks_statistic": ks_statistic,
                    "ks_pvalue": ks_pvalue,
                    "wasserstein_distance": wasserstein,
                    "drift_level": self._drift_level(psi, ks_pvalue),
                }
            )
        self.drift_report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
        return self.drift_report

    def run_prediction_drift_check(self) -> dict[str, Any]:
        """Summarize current prediction distribution and label mix."""
        if self.predictions is None:
            self.load_inputs()
        assert self.predictions is not None
        if self.prediction_col not in self.predictions.columns:
            self.prediction_drift_summary = {
                "available": False,
                "message": f"Missing `{self.prediction_col}` in predictions file.",
            }
            return self.prediction_drift_summary

        scores = pd.to_numeric(self.predictions[self.prediction_col], errors="coerce")
        summary: dict[str, Any] = {
            "available": True,
            "row_count": int(len(self.predictions)),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "score_p10": float(scores.quantile(0.10)),
            "score_p50": float(scores.quantile(0.50)),
            "score_p90": float(scores.quantile(0.90)),
        }
        if "predicted_label" in self.predictions.columns:
            summary["predicted_positive_rate"] = float(self.predictions["predicted_label"].mean())
        actual_label_col = None
        if "actual_label" in self.predictions.columns:
            actual_label_col = "actual_label"
        elif self.target_col and self.target_col in self.predictions.columns:
            actual_label_col = self.target_col
        if actual_label_col:
            summary["actual_positive_rate"] = float(self.predictions[actual_label_col].mean())
        if "predicted_label" in self.predictions.columns and actual_label_col:
            summary["prediction_actual_rate_gap"] = float(
                self.predictions["predicted_label"].mean() - self.predictions[actual_label_col].mean()
            )
        self.prediction_drift_summary = summary
        return summary

    def run_cluster_shift_check(self) -> pd.DataFrame:
        """Fit train clusters and compare train/current cluster share."""
        if self.train_features is None or self.current_features is None:
            self.load_inputs()
        assert self.train_features is not None
        assert self.current_features is not None

        train_matrix = self.train_features[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        current_matrix = self.current_features[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        train_matrix = train_matrix.fillna(train_matrix.median(numeric_only=True))
        current_matrix = current_matrix.fillna(train_matrix.median(numeric_only=True))

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_matrix)
        current_scaled = scaler.transform(current_matrix)
        model = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=10)
        train_clusters = model.fit_predict(train_scaled)
        current_clusters = model.predict(current_scaled)

        train_counts = pd.Series(train_clusters).value_counts(normalize=True).sort_index() * 100
        current_counts = pd.Series(current_clusters).value_counts(normalize=True).sort_index() * 100
        rows = []
        for cluster in range(3):
            train_share = float(train_counts.get(cluster, 0.0))
            current_share = float(current_counts.get(cluster, 0.0))
            rows.append(
                {
                    "cluster": cluster,
                    "train_share_pct": train_share,
                    "current_share_pct": current_share,
                    "share_change_pct_points": current_share - train_share,
                }
            )
        self.cluster_shift_report = pd.DataFrame(rows)
        return self.cluster_shift_report

    def _save_drift_plot(self) -> Path:
        """Save a bar chart for the top PSI drift features."""
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "drift_top_features.png"
        if self.drift_report.empty:
            return output_path

        top_drift = self.drift_report.sort_values("psi", ascending=False).head(8)
        plt.figure(figsize=(10, 5))
        plt.barh(top_drift["feature"], top_drift["psi"], color="#3855ff")
        plt.axvline(0.10, color="#f59e0b", linestyle="--", linewidth=1, label="Medium PSI")
        plt.axvline(0.25, color="#dc2626", linestyle="--", linewidth=1, label="High PSI")
        plt.gca().invert_yaxis()
        plt.xlabel("PSI")
        plt.title("Top Feature Drift by PSI")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close()
        return output_path

    def build_output(self) -> dict[str, Any]:
        """Build the final Mitra JSON payload."""
        if self.missing_value_summary.empty:
            self.run_missing_value_checks()
        if self.drift_report.empty:
            self.run_feature_drift_checks()
        if not self.prediction_drift_summary:
            self.run_prediction_drift_check()
        if self.cluster_shift_report.empty:
            self.run_cluster_shift_check()

        high_drift = self.drift_report.loc[self.drift_report["drift_level"] == "High"]
        medium_drift = self.drift_report.loc[self.drift_report["drift_level"] == "Medium"]
        largest_cluster_shift = (
            float(self.cluster_shift_report["share_change_pct_points"].abs().max())
            if not self.cluster_shift_report.empty
            else 0.0
        )
        material_missing = self.missing_value_summary.loc[
            self.missing_value_summary["missing_rate_change_pct_points"].abs() >= 5
        ]

        if len(high_drift) >= 2 or largest_cluster_shift >= 10:
            overall_risk = "High"
        elif len(high_drift) == 1 or len(medium_drift) >= 2 or largest_cluster_shift >= 5:
            overall_risk = "Medium"
        else:
            overall_risk = "Low"

        recommended_checks = [
            "Review high-drift features for data pipeline or behavior-shift explanations.",
            "Compare prediction score distribution against the model training baseline when train predictions are available.",
            "Inspect cluster share movement for population-mix changes before high-impact business use.",
        ]
        if not material_missing.empty:
            recommended_checks.append("Investigate features with material missing-rate changes.")

        self.output = {
            "agent_name": "Agent 01: Mitra",
            "overall_risk_level": overall_risk,
            "artifact_contract": {
                "target": self.target_col,
                "entity_id": self.entity_id_col,
                "prediction_column": self.prediction_col,
                "feature_count": len(self.feature_cols),
            },
            "data_health_summary": {
                "train_rows": int(len(self.train_features)) if self.train_features is not None else 0,
                "current_rows": int(len(self.current_features)) if self.current_features is not None else 0,
                "feature_count": len(self.feature_cols),
                "missing_value_findings": self.missing_value_summary.to_dict(orient="records"),
            },
            "high_drift_features": high_drift.to_dict(orient="records"),
            "medium_drift_features": medium_drift.to_dict(orient="records"),
            "prediction_drift_summary": self.prediction_drift_summary,
            "cluster_findings": self.cluster_shift_report.to_dict(orient="records"),
            "recommended_checks": recommended_checks,
        }
        return self.output

    def save_outputs(self) -> dict[str, Path]:
        """Run checks and save JSON, CSV, and figure outputs."""
        self.load_inputs()
        self.run_missing_value_checks()
        self.run_feature_drift_checks()
        self.run_prediction_drift_check()
        self.run_cluster_shift_check()
        output = self.build_output()

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = REPORTS_DIR / "signal_sentinel_output.json"
        drift_path = REPORTS_DIR / "drift_report.csv"
        cluster_path = REPORTS_DIR / "cluster_shift_report.csv"
        figure_path = self._save_drift_plot()

        json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        self.drift_report.to_csv(drift_path, index=False)
        self.cluster_shift_report.to_csv(cluster_path, index=False)
        EvidenceStore().save_section("signal_sentinel", output)

        return {
            "json": json_path,
            "drift_report": drift_path,
            "cluster_shift_report": cluster_path,
            "figure": figure_path,
        }


def main() -> None:
    """Run Agent 01: Mitra from the command line."""
    output_paths = SignalSentinelAgent().save_outputs()
    print("Saved Agent 01: Mitra outputs:")
    for label, path in output_paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
