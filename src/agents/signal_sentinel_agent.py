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
CONFIGS_DIR = PROJECT_ROOT / "configs"
RANDOM_SEED = 42


class SignalSentinelAgent:
    """Detect deterministic signal, prediction, missingness, and cluster shifts."""

    def __init__(
        self,
        train_features_path: Path = DATA_DIR / "train_features_sample.csv",
        current_features_path: Path = DATA_DIR / "current_features_sample.csv",
        train_predictions_path: Path = DATA_DIR / "train_predictions_sample.csv",
        predictions_path: Path = DATA_DIR / "current_predictions_sample.csv",
        model_metadata_path: Path = MODELS_DIR / "model_metadata.json",
        feature_metadata_path: Path = MODELS_DIR / "feature_metadata.json",
        calibration_config_path: Path = CONFIGS_DIR / "calibration_config_v1.json",
    ) -> None:
        """Configure input and output artifacts."""
        self.train_features_path = Path(train_features_path)
        self.current_features_path = Path(current_features_path)
        self.train_predictions_path = Path(train_predictions_path)
        self.predictions_path = Path(predictions_path)
        self.model_metadata_path = Path(model_metadata_path)
        self.feature_metadata_path = Path(feature_metadata_path)
        self.calibration_config_path = Path(calibration_config_path)
        self.train_features: pd.DataFrame | None = None
        self.current_features: pd.DataFrame | None = None
        self.train_predictions: pd.DataFrame | None = None
        self.predictions: pd.DataFrame | None = None
        self.model_metadata: dict[str, Any] = {}
        self.feature_metadata: dict[str, Any] = {}
        self.calibration_config: dict[str, Any] = {}
        self.config_version: str = "unconfigured"
        self.target_col: str | None = None
        self.entity_id_col: str | None = None
        self.prediction_col: str = "propensity_score"
        self.metadata_feature_cols: list[str] = []
        self.feature_cols: list[str] = []
        self.data_quality_report: pd.DataFrame = pd.DataFrame()
        self.missing_value_summary: pd.DataFrame = pd.DataFrame()
        self.drift_report: pd.DataFrame = pd.DataFrame()
        self.prediction_drift_summary: dict[str, Any] = {}
        self.cluster_shift_report: pd.DataFrame = pd.DataFrame()
        self.output: dict[str, Any] = {}

    def load_calibration_config(self) -> None:
        """Load Mitra calibration thresholds from config."""
        if self.calibration_config_path.exists():
            self.calibration_config = json.loads(self.calibration_config_path.read_text(encoding="utf-8"))
        else:
            self.calibration_config = {
                "config_version": "fallback_defaults",
                "mitra": {
                    "psi_medium": 0.10,
                    "psi_high": 0.25,
                    "ks_pvalue_high": 0.01,
                    "ks_pvalue_medium": 0.05,
                    "missing_rate_change_pct_points": 5.0,
                    "missing_rate_high_pct_points": 20.0,
                    "boundary_violation_medium_rate": 0.0,
                    "boundary_violation_high_rate": 0.05,
                    "cluster_shift_medium_pct_points": 5.0,
                    "cluster_shift_high_pct_points": 10.0,
                },
            }
        self.config_version = str(self.calibration_config.get("config_version", "unknown"))

    def mitra_config(self) -> dict[str, Any]:
        """Return Mitra thresholds from loaded calibration config."""
        if not self.calibration_config:
            self.load_calibration_config()
        return self.calibration_config.get("mitra", {})

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
        self.train_predictions = pd.read_csv(self.train_predictions_path) if self.train_predictions_path.exists() else None
        self.predictions = pd.read_csv(self.predictions_path)
        with self.model_metadata_path.open("r", encoding="utf-8") as input_file:
            self.model_metadata = json.load(input_file)
        with self.feature_metadata_path.open("r", encoding="utf-8") as input_file:
            self.feature_metadata = json.load(input_file)
        self.load_calibration_config()

        self.target_col = self.model_metadata.get("target") or self.feature_metadata.get("target")
        self.entity_id_col = self.model_metadata.get("entity_id") or self.feature_metadata.get("entity_id")
        self.prediction_col = self.model_metadata.get("prediction_column", self.prediction_col)

        metadata_features = self.model_metadata.get("feature_columns") or [
            feature["name"] for feature in self.feature_metadata.get("features", [])
        ]
        self.metadata_feature_cols = list(metadata_features)
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

    def _drift_level(self, psi: float, ks_pvalue: float) -> str:
        """Classify drift from PSI and KS thresholds."""
        config = self.mitra_config()
        psi_high = float(config.get("psi_high", 0.25))
        psi_medium = float(config.get("psi_medium", 0.10))
        ks_high = float(config.get("ks_pvalue_high", 0.01))
        ks_medium = float(config.get("ks_pvalue_medium", 0.0))
        if psi >= psi_high or ks_pvalue < ks_high:
            return "High"
        if psi >= psi_medium or ks_pvalue < ks_medium:
            return "Medium"
        return "Low"

    def _drift_level_reason(self, psi: float, ks_pvalue: float) -> str:
        """Explain why a drift level was assigned."""
        config = self.mitra_config()
        psi_high = float(config.get("psi_high", 0.25))
        psi_medium = float(config.get("psi_medium", 0.10))
        ks_high = float(config.get("ks_pvalue_high", 0.01))
        ks_medium = float(config.get("ks_pvalue_medium", 0.0))
        if psi >= psi_high:
            return f"High because PSI {psi:.4f} is at or above configured high threshold {psi_high:.4f}."
        if ks_pvalue < ks_high:
            return f"High because KS p-value {ks_pvalue:.4g} is below configured high threshold {ks_high:.4g}."
        if psi >= psi_medium:
            return f"Medium because PSI {psi:.4f} is at or above configured medium threshold {psi_medium:.4f}."
        if ks_medium > 0 and ks_pvalue < ks_medium:
            return f"Medium because KS p-value {ks_pvalue:.4g} is below configured medium threshold {ks_medium:.4g}."
        return "Low because PSI and KS p-value did not cross configured drift thresholds."

    @staticmethod
    def _mean_change_pct(train_mean: float, current_mean: float) -> float:
        """Calculate percentage mean change while handling zero baselines."""
        if pd.isna(train_mean) or pd.isna(current_mean):
            return 0.0
        if abs(train_mean) < 1e-12:
            return 0.0 if abs(current_mean) < 1e-12 else float(np.sign(current_mean) * 100.0)
        return float(((current_mean - train_mean) / abs(train_mean)) * 100)

    def _feature_metadata_by_name(self) -> dict[str, dict[str, Any]]:
        """Return feature metadata keyed by feature name."""
        return {
            feature.get("name"): feature
            for feature in self.feature_metadata.get("features", [])
            if feature.get("name")
        }

    def _feature_type(self, feature: str) -> str:
        """Return the metadata-defined feature type when available."""
        return self._feature_metadata_by_name().get(feature, {}).get("type", "numeric")

    @staticmethod
    def _recommended_action(drift_level: str, missing_rate_change_pct_points: float) -> str:
        """Map drift and missingness evidence to a concise recommended action."""
        if abs(missing_rate_change_pct_points) >= 5:
            return "Investigate upstream data quality before interpreting drift."
        if drift_level == "High":
            return "Review feature lineage and business context before relying on current-window scoring."
        if drift_level == "Medium":
            return "Monitor in the next run and confirm whether the shift is expected."
        return "No immediate action; continue routine monitoring."

    def _quality_level(self, check_type: str, value: float) -> str:
        """Classify data-quality checks using simple MVP thresholds."""
        config = self.mitra_config()
        if check_type == "missing_rate_change_pct_points":
            if abs(value) >= float(config.get("missing_rate_high_pct_points", 20.0)):
                return "High"
            if abs(value) >= float(config.get("missing_rate_change_pct_points", 5.0)):
                return "Medium"
            return "Low"
        if check_type == "boundary_violation_rate":
            if value >= float(config.get("boundary_violation_high_rate", 0.05)):
                return "High"
            if value > float(config.get("boundary_violation_medium_rate", 0.0)):
                return "Medium"
            return "Low"
        if check_type in {"duplicate_entity_rate", "prediction_score_out_of_range_rate"}:
            return "High" if value > 0 else "Low"
        return "Low"

    def _quality_level_reason(self, check_type: str, value: float) -> str:
        """Explain why a data-quality level was assigned."""
        config = self.mitra_config()
        if check_type == "missing_rate_change_pct_points":
            high = float(config.get("missing_rate_high_pct_points", 20.0))
            medium = float(config.get("missing_rate_change_pct_points", 5.0))
            if abs(value) >= high:
                return f"High because missing-rate shift {value:+.2f} pp is at or above configured high threshold {high:.2f} pp."
            if abs(value) >= medium:
                return f"Medium because missing-rate shift {value:+.2f} pp is at or above configured medium threshold {medium:.2f} pp."
            return "Low because missing-rate shift did not cross configured thresholds."
        if check_type == "boundary_violation_rate":
            high = float(config.get("boundary_violation_high_rate", 0.05))
            medium = float(config.get("boundary_violation_medium_rate", 0.0))
            if value >= high:
                return f"High because boundary violation rate {value:.2%} is at or above configured high threshold {high:.2%}."
            if value > medium:
                return f"Medium because boundary violation rate {value:.2%} is above configured medium threshold {medium:.2%}."
            return "Low because no boundary violations crossed configured thresholds."
        if check_type in {"duplicate_entity_rate", "prediction_score_out_of_range_rate"}:
            return "High because non-zero rate was observed." if value > 0 else "Low because no issue was observed."
        return "Low because this check did not report a configured issue."

    def _source_files(self) -> dict[str, str | None]:
        """Return source file paths used by Mitra."""
        return {
            "train_features": str(self.train_features_path),
            "current_features": str(self.current_features_path),
            "train_predictions": str(self.train_predictions_path) if self.train_predictions_path.exists() else None,
            "current_predictions": str(self.predictions_path),
            "model_metadata": str(self.model_metadata_path),
            "feature_metadata": str(self.feature_metadata_path),
            "calibration_config": str(self.calibration_config_path) if self.calibration_config_path.exists() else None,
        }

    def run_data_quality_checks(self) -> pd.DataFrame:
        """Run schema, missingness, duplicate, boundary, and prediction range checks."""
        if self.train_features is None or self.current_features is None or self.predictions is None:
            self.load_inputs()
        assert self.train_features is not None
        assert self.current_features is not None
        assert self.predictions is not None

        rows: list[dict[str, Any]] = []
        feature_metadata = self._feature_metadata_by_name()
        required_train_columns = [self.entity_id_col, self.target_col, *self.metadata_feature_cols]
        required_current_columns = [self.entity_id_col, self.target_col, *self.metadata_feature_cols]
        required_prediction_columns = [self.entity_id_col, self.prediction_col]

        for dataset_name, dataframe, required_columns in [
            ("train_features", self.train_features, required_train_columns),
            ("current_features", self.current_features, required_current_columns),
            ("current_predictions", self.predictions, required_prediction_columns),
        ]:
            missing_columns = [column for column in required_columns if column and column not in dataframe.columns]
            rows.append(
                {
                    "config_version": self.config_version,
                    "source_train_features": str(self.train_features_path),
                    "source_current_features": str(self.current_features_path),
                    "source_current_predictions": str(self.predictions_path),
                    "check_type": "required_columns_present",
                    "dataset": dataset_name,
                    "feature": None,
                    "train_value": None,
                    "current_value": None,
                    "change_value": len(missing_columns),
                    "issue_level": "High" if missing_columns else "Low",
                    "message": (
                        "Missing required columns: " + ", ".join(missing_columns)
                        if missing_columns
                        else "All required columns are present."
                    ),
                    "issue_reason": (
                        "High because required columns are missing."
                        if missing_columns
                        else "Low because all required columns are present."
                    ),
                }
            )
        if self.train_predictions is not None:
            missing_columns = [
                column
                for column in required_prediction_columns
                if column and column not in self.train_predictions.columns
            ]
            rows.append(
                {
                    "config_version": self.config_version,
                    "source_train_features": str(self.train_features_path),
                    "source_current_features": str(self.current_features_path),
                    "source_current_predictions": str(self.predictions_path),
                    "check_type": "required_columns_present",
                    "dataset": "train_predictions",
                    "feature": None,
                    "train_value": None,
                    "current_value": None,
                    "change_value": len(missing_columns),
                    "issue_level": "High" if missing_columns else "Low",
                    "message": (
                        "Missing required columns: " + ", ".join(missing_columns)
                        if missing_columns
                        else "All required columns are present."
                    ),
                    "issue_reason": (
                        "High because required columns are missing."
                        if missing_columns
                        else "Low because all required columns are present."
                    ),
                }
            )

        if self.entity_id_col:
            for dataset_name, dataframe in [
                ("train_features", self.train_features),
                ("current_features", self.current_features),
                ("current_predictions", self.predictions),
            ]:
                if self.entity_id_col in dataframe.columns:
                    duplicate_rate = float(dataframe[self.entity_id_col].duplicated().mean())
                    issue_level = self._quality_level("duplicate_entity_rate", duplicate_rate)
                    rows.append(
                        {
                            "config_version": self.config_version,
                            "source_train_features": str(self.train_features_path),
                            "source_current_features": str(self.current_features_path),
                            "source_current_predictions": str(self.predictions_path),
                            "check_type": "duplicate_entity_rate",
                            "dataset": dataset_name,
                            "feature": self.entity_id_col,
                            "train_value": None,
                            "current_value": duplicate_rate,
                            "change_value": duplicate_rate,
                            "issue_level": issue_level,
                            "message": f"Duplicate entity rate is {duplicate_rate:.2%}.",
                            "issue_reason": self._quality_level_reason("duplicate_entity_rate", duplicate_rate),
                        }
                    )
            if self.train_predictions is not None and self.entity_id_col in self.train_predictions.columns:
                duplicate_rate = float(self.train_predictions[self.entity_id_col].duplicated().mean())
                issue_level = self._quality_level("duplicate_entity_rate", duplicate_rate)
                rows.append(
                    {
                        "config_version": self.config_version,
                        "source_train_features": str(self.train_features_path),
                        "source_current_features": str(self.current_features_path),
                        "source_current_predictions": str(self.predictions_path),
                        "check_type": "duplicate_entity_rate",
                        "dataset": "train_predictions",
                        "feature": self.entity_id_col,
                        "train_value": duplicate_rate,
                        "current_value": None,
                        "change_value": duplicate_rate,
                        "issue_level": issue_level,
                        "message": f"Duplicate entity rate is {duplicate_rate:.2%}.",
                        "issue_reason": self._quality_level_reason("duplicate_entity_rate", duplicate_rate),
                    }
                )

        for feature in self.feature_cols:
            train_missing = float(self.train_features[feature].isna().mean())
            current_missing = float(self.current_features[feature].isna().mean())
            missing_change = (current_missing - train_missing) * 100
            issue_level = self._quality_level("missing_rate_change_pct_points", missing_change)
            rows.append(
                {
                    "config_version": self.config_version,
                    "source_train_features": str(self.train_features_path),
                    "source_current_features": str(self.current_features_path),
                    "source_current_predictions": str(self.predictions_path),
                    "check_type": "missing_rate_change_pct_points",
                    "dataset": "current_features",
                    "feature": feature,
                    "train_value": train_missing,
                    "current_value": current_missing,
                    "change_value": missing_change,
                    "issue_level": issue_level,
                    "message": f"Missing rate changed by {missing_change:+.2f} percentage points.",
                    "issue_reason": self._quality_level_reason("missing_rate_change_pct_points", missing_change),
                }
            )

            metadata = feature_metadata.get(feature, {})
            train_numeric = pd.to_numeric(self.train_features[feature], errors="coerce")
            current_numeric = pd.to_numeric(self.current_features[feature], errors="coerce")
            lower_bound = metadata.get("expected_min", float(train_numeric.min()))
            upper_bound = metadata.get("expected_max", float(train_numeric.max()))
            if pd.notna(lower_bound) and pd.notna(upper_bound):
                violation_mask = (current_numeric < float(lower_bound)) | (current_numeric > float(upper_bound))
                violation_rate = float(violation_mask.fillna(False).mean())
                issue_level = self._quality_level("boundary_violation_rate", violation_rate)
                rows.append(
                    {
                        "config_version": self.config_version,
                        "source_train_features": str(self.train_features_path),
                        "source_current_features": str(self.current_features_path),
                        "source_current_predictions": str(self.predictions_path),
                        "check_type": "boundary_violation_rate",
                        "dataset": "current_features",
                        "feature": feature,
                        "train_value": f"[{float(lower_bound):.6g}, {float(upper_bound):.6g}]",
                        "current_value": violation_rate,
                        "change_value": violation_rate,
                        "issue_level": issue_level,
                        "message": f"Current values outside training/metadata bounds: {violation_rate:.2%}.",
                        "issue_reason": self._quality_level_reason("boundary_violation_rate", violation_rate),
                    }
                )

        if self.prediction_col in self.predictions.columns:
            scores = pd.to_numeric(self.predictions[self.prediction_col], errors="coerce")
            out_of_range_rate = float(((scores < 0) | (scores > 1)).fillna(False).mean())
            issue_level = self._quality_level("prediction_score_out_of_range_rate", out_of_range_rate)
            rows.append(
                {
                    "config_version": self.config_version,
                    "source_train_features": str(self.train_features_path),
                    "source_current_features": str(self.current_features_path),
                    "source_current_predictions": str(self.predictions_path),
                    "check_type": "prediction_score_out_of_range_rate",
                    "dataset": "current_predictions",
                    "feature": self.prediction_col,
                    "train_value": "[0, 1]",
                    "current_value": out_of_range_rate,
                    "change_value": out_of_range_rate,
                    "issue_level": issue_level,
                    "message": f"Prediction scores outside [0, 1]: {out_of_range_rate:.2%}.",
                    "issue_reason": self._quality_level_reason(
                        "prediction_score_out_of_range_rate", out_of_range_rate
                    ),
                }
            )
        if self.train_predictions is not None and self.prediction_col in self.train_predictions.columns:
            scores = pd.to_numeric(self.train_predictions[self.prediction_col], errors="coerce")
            out_of_range_rate = float(((scores < 0) | (scores > 1)).fillna(False).mean())
            issue_level = self._quality_level("prediction_score_out_of_range_rate", out_of_range_rate)
            rows.append(
                {
                    "config_version": self.config_version,
                    "source_train_features": str(self.train_features_path),
                    "source_current_features": str(self.current_features_path),
                    "source_current_predictions": str(self.predictions_path),
                    "check_type": "prediction_score_out_of_range_rate",
                    "dataset": "train_predictions",
                    "feature": self.prediction_col,
                    "train_value": out_of_range_rate,
                    "current_value": None,
                    "change_value": out_of_range_rate,
                    "issue_level": issue_level,
                    "message": f"Prediction scores outside [0, 1]: {out_of_range_rate:.2%}.",
                    "issue_reason": self._quality_level_reason(
                        "prediction_score_out_of_range_rate", out_of_range_rate
                    ),
                }
            )

        label_available = bool(
            ("actual_label" in self.predictions.columns)
            or (self.target_col and self.target_col in self.predictions.columns)
            or (self.target_col and self.target_col in self.current_features.columns)
        )
        rows.append(
            {
                "config_version": self.config_version,
                "source_train_features": str(self.train_features_path),
                "source_current_features": str(self.current_features_path),
                "source_current_predictions": str(self.predictions_path),
                "check_type": "label_availability",
                "dataset": "current_predictions",
                "feature": self.target_col,
                "train_value": None,
                "current_value": label_available,
                "change_value": None,
                "issue_level": "Low" if label_available else "Medium",
                "message": "Current labels are available." if label_available else "Current labels are unavailable.",
                "issue_reason": (
                    "Low because current labels are available."
                    if label_available
                    else "Medium because current labels are unavailable for outcome validation."
                ),
            }
        )

        self.data_quality_report = pd.DataFrame(rows)
        return self.data_quality_report

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
            train_missing = float(train_values.isna().mean())
            current_missing = float(current_values.isna().mean())
            missing_change = (current_missing - train_missing) * 100
            psi = self.calculate_psi(train_values, current_values)
            ks_statistic, ks_pvalue = self.calculate_ks(train_values, current_values)
            wasserstein = self.calculate_wasserstein(train_values, current_values)
            drift_level = self._drift_level(psi, ks_pvalue)
            rows.append(
                {
                    "config_version": self.config_version,
                    "source_train_features": str(self.train_features_path),
                    "source_current_features": str(self.current_features_path),
                    "feature": feature,
                    "feature_type": self._feature_type(feature),
                    "train_mean": train_mean,
                    "current_mean": current_mean,
                    "mean_change_pct": self._mean_change_pct(train_mean, current_mean),
                    "train_missing_rate": train_missing,
                    "current_missing_rate": current_missing,
                    "missing_rate_change_pct_points": missing_change,
                    "psi": psi,
                    "ks_statistic": ks_statistic,
                    "ks_pvalue": ks_pvalue,
                    "wasserstein_distance": wasserstein,
                    "drift_level": drift_level,
                    "drift_level_reason": self._drift_level_reason(psi, ks_pvalue),
                    "recommended_action": self._recommended_action(drift_level, missing_change),
                }
            )
        self.drift_report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
        return self.drift_report

    def run_prediction_drift_check(self) -> dict[str, Any]:
        """Compare reference/current prediction score distributions and label mix."""
        if self.predictions is None:
            self.load_inputs()
        assert self.predictions is not None
        if self.prediction_col not in self.predictions.columns:
            self.prediction_drift_summary = {
                "available": False,
                "config_version": self.config_version,
                "source_files": self._source_files(),
                "prediction_drift_level": "Unknown",
                "prediction_drift_reason": "Unknown because the configured prediction column is missing.",
                "message": f"Missing `{self.prediction_col}` in predictions file.",
            }
            return self.prediction_drift_summary

        current_scores = pd.to_numeric(self.predictions[self.prediction_col], errors="coerce")
        summary: dict[str, Any] = {
            "available": True,
            "config_version": self.config_version,
            "source_files": self._source_files(),
            "reference_available": bool(
                self.train_predictions is not None and self.prediction_col in self.train_predictions.columns
            ),
            "current_row_count": int(len(self.predictions)),
            "current_score_mean": float(current_scores.mean()),
            "current_score_std": float(current_scores.std()),
            "current_score_min": float(current_scores.min()),
            "current_score_max": float(current_scores.max()),
            "current_score_p10": float(current_scores.quantile(0.10)),
            "current_score_p50": float(current_scores.quantile(0.50)),
            "current_score_p90": float(current_scores.quantile(0.90)),
        }
        train_predicted_positive_rate = None
        current_predicted_positive_rate = None
        train_actual_positive_rate = None
        current_actual_positive_rate = None

        if self.train_predictions is not None and self.prediction_col in self.train_predictions.columns:
            train_scores = pd.to_numeric(self.train_predictions[self.prediction_col], errors="coerce")
            psi = self.calculate_psi(train_scores, current_scores)
            ks_statistic, ks_pvalue = self.calculate_ks(train_scores, current_scores)
            drift_level = self._drift_level(psi, ks_pvalue)
            train_score_mean = float(train_scores.mean())
            current_score_mean = float(current_scores.mean())
            summary.update(
                {
                    "reference_row_count": int(len(self.train_predictions)),
                    "reference_score_mean": train_score_mean,
                    "reference_score_std": float(train_scores.std()),
                    "reference_score_min": float(train_scores.min()),
                    "reference_score_max": float(train_scores.max()),
                    "reference_score_p10": float(train_scores.quantile(0.10)),
                    "reference_score_p50": float(train_scores.quantile(0.50)),
                    "reference_score_p90": float(train_scores.quantile(0.90)),
                    "score_mean_change_pct": self._mean_change_pct(train_score_mean, current_score_mean),
                    "score_psi": psi,
                    "score_ks_statistic": ks_statistic,
                    "score_ks_pvalue": ks_pvalue,
                    "score_wasserstein_distance": self.calculate_wasserstein(train_scores, current_scores),
                    "prediction_drift_level": drift_level,
                    "prediction_drift_reason": self._drift_level_reason(psi, ks_pvalue),
                }
            )
        else:
            summary.update(
                {
                    "prediction_drift_level": "Unknown",
                    "prediction_drift_reason": "Unknown because reference prediction scores are unavailable.",
                    "message": "Reference prediction scores are unavailable; only current prediction distribution was summarized.",
                }
            )

        if self.train_predictions is not None and "predicted_label" in self.train_predictions.columns:
            train_predicted_positive_rate = float(self.train_predictions["predicted_label"].mean())
            summary["reference_predicted_positive_rate"] = train_predicted_positive_rate
        if "predicted_label" in self.predictions.columns:
            current_predicted_positive_rate = float(self.predictions["predicted_label"].mean())
            summary["current_predicted_positive_rate"] = current_predicted_positive_rate
            summary["predicted_positive_rate"] = current_predicted_positive_rate
        if train_predicted_positive_rate is not None and current_predicted_positive_rate is not None:
            summary["predicted_positive_rate_change_pct_points"] = float(
                (current_predicted_positive_rate - train_predicted_positive_rate) * 100
            )

        train_actual_label_col = None
        if self.train_predictions is not None and "actual_label" in self.train_predictions.columns:
            train_actual_label_col = "actual_label"
        elif self.train_predictions is not None and self.target_col and self.target_col in self.train_predictions.columns:
            train_actual_label_col = self.target_col
        if self.train_predictions is not None and train_actual_label_col:
            train_actual_positive_rate = float(self.train_predictions[train_actual_label_col].mean())
            summary["reference_actual_positive_rate"] = train_actual_positive_rate

        actual_label_col = None
        if "actual_label" in self.predictions.columns:
            actual_label_col = "actual_label"
        elif self.target_col and self.target_col in self.predictions.columns:
            actual_label_col = self.target_col
        if actual_label_col:
            current_actual_positive_rate = float(self.predictions[actual_label_col].mean())
            summary["actual_positive_rate"] = current_actual_positive_rate
            summary["current_actual_positive_rate"] = current_actual_positive_rate
        if train_actual_positive_rate is not None and current_actual_positive_rate is not None:
            summary["actual_positive_rate_change_pct_points"] = float(
                (current_actual_positive_rate - train_actual_positive_rate) * 100
            )
        if "predicted_label" in self.predictions.columns and actual_label_col:
            summary["prediction_actual_rate_gap"] = float(
                self.predictions["predicted_label"].mean() - self.predictions[actual_label_col].mean()
            )
        if "score_psi" in summary:
            if summary["prediction_drift_level"] == "High":
                summary["recommended_action"] = (
                    "Review score distribution movement before using current-window model decisions."
                )
            elif summary["prediction_drift_level"] == "Medium":
                summary["recommended_action"] = "Monitor score distribution and confirm whether decision mix shift is expected."
            else:
                summary["recommended_action"] = "No immediate prediction drift action; continue routine monitoring."
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
                    "config_version": self.config_version,
                    "source_train_features": str(self.train_features_path),
                    "source_current_features": str(self.current_features_path),
                    "cluster": cluster,
                    "train_share_pct": train_share,
                    "current_share_pct": current_share,
                    "share_change_pct_points": current_share - train_share,
                }
            )
        self.cluster_shift_report = pd.DataFrame(rows)
        return self.cluster_shift_report

    def _overall_risk_decision(
        self,
        high_quality_issues: int,
        medium_quality_issues: int,
        high_drift_count: int,
        medium_drift_count: int,
        largest_cluster_shift: float,
        prediction_drift_level: str,
    ) -> dict[str, Any]:
        """Apply ordered Mitra risk rules and return the final decision with evidence."""
        config = self.mitra_config()
        cluster_high = float(config.get("cluster_shift_high_pct_points", 10.0))
        cluster_medium = float(config.get("cluster_shift_medium_pct_points", 5.0))
        rule_hierarchy = [
            "High if any high-severity data-quality issue exists.",
            "High if two or more high-drift features exist.",
            f"High if max absolute cluster share shift is at or above {cluster_high:.2f} percentage points.",
            "High if prediction drift level is High.",
            "Medium if any medium-severity data-quality issue exists.",
            "Medium if one high-drift feature exists.",
            "Medium if two or more medium-drift features exist.",
            f"Medium if max absolute cluster share shift is at or above {cluster_medium:.2f} percentage points.",
            "Medium if prediction drift level is Medium.",
            "Low if none of the configured High or Medium rules fire.",
        ]
        high_rules = [
            (
                high_quality_issues > 0,
                f"{high_quality_issues} high-severity data-quality issue(s) detected.",
            ),
            (high_drift_count >= 2, f"{high_drift_count} high-drift feature(s) detected."),
            (
                largest_cluster_shift >= cluster_high,
                f"Max cluster shift {largest_cluster_shift:.2f} pp crossed high threshold {cluster_high:.2f} pp.",
            ),
            (prediction_drift_level == "High", "Prediction drift level is High."),
        ]
        medium_rules = [
            (
                medium_quality_issues > 0,
                f"{medium_quality_issues} medium-severity data-quality issue(s) detected.",
            ),
            (high_drift_count == 1, "One high-drift feature detected."),
            (medium_drift_count >= 2, f"{medium_drift_count} medium-drift feature(s) detected."),
            (
                largest_cluster_shift >= cluster_medium,
                f"Max cluster shift {largest_cluster_shift:.2f} pp crossed medium threshold {cluster_medium:.2f} pp.",
            ),
            (prediction_drift_level == "Medium", "Prediction drift level is Medium."),
        ]

        triggered_high = [reason for condition, reason in high_rules if condition]
        if triggered_high:
            return {
                "overall_risk_level": "High",
                "primary_reason": triggered_high[0],
                "triggered_rules": triggered_high,
                "rule_hierarchy": rule_hierarchy,
            }

        triggered_medium = [reason for condition, reason in medium_rules if condition]
        if triggered_medium:
            return {
                "overall_risk_level": "Medium",
                "primary_reason": triggered_medium[0],
                "triggered_rules": triggered_medium,
                "rule_hierarchy": rule_hierarchy,
            }

        return {
            "overall_risk_level": "Low",
            "primary_reason": "No configured High or Medium Mitra risk rules fired.",
            "triggered_rules": [],
            "rule_hierarchy": rule_hierarchy,
        }

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
        if self.data_quality_report.empty:
            self.run_data_quality_checks()
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
        data_quality_issue_counts = (
            {
                str(level): int(count)
                for level, count in self.data_quality_report["issue_level"].value_counts().to_dict().items()
            }
            if not self.data_quality_report.empty
            else {}
        )
        high_quality_issues = int(data_quality_issue_counts.get("High", 0))
        medium_quality_issues = int(data_quality_issue_counts.get("Medium", 0))
        prediction_drift_level = self.prediction_drift_summary.get("prediction_drift_level", "Unknown")

        risk_assessment = self._overall_risk_decision(
            high_quality_issues=high_quality_issues,
            medium_quality_issues=medium_quality_issues,
            high_drift_count=len(high_drift),
            medium_drift_count=len(medium_drift),
            largest_cluster_shift=largest_cluster_shift,
            prediction_drift_level=prediction_drift_level,
        )
        overall_risk = risk_assessment["overall_risk_level"]
        risk_assessment["config_version"] = self.config_version

        recommended_checks = [
            "Review high-drift features for data pipeline or behavior-shift explanations.",
            "Compare prediction score distribution against the model training baseline when train predictions are available.",
            "Inspect cluster share movement for population-mix changes before high-impact business use.",
        ]
        if not material_missing.empty:
            recommended_checks.append("Investigate features with material missing-rate changes.")
        if high_quality_issues > 0:
            recommended_checks.insert(0, "Resolve high-severity data quality issues before interpreting model drift.")
        if prediction_drift_level in {"High", "Medium"}:
            recommended_checks.append("Review prediction score drift against decision thresholds and downstream actions.")

        self.output = {
            "agent_name": "Agent 01: Mitra",
            "config_version": self.config_version,
            "source_files": self._source_files(),
            "overall_risk_level": overall_risk,
            "overall_risk_explanation": risk_assessment["primary_reason"],
            "risk_assessment": risk_assessment,
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
                "required_feature_count": len(self.metadata_feature_cols),
                "data_quality_issue_counts": data_quality_issue_counts,
                "data_quality_findings": self.data_quality_report.to_dict(orient="records"),
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
        self.run_data_quality_checks()
        self.run_missing_value_checks()
        self.run_feature_drift_checks()
        self.run_prediction_drift_check()
        self.run_cluster_shift_check()
        output = self.build_output()

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = REPORTS_DIR / "mitra_output.json"
        legacy_json_path = REPORTS_DIR / "signal_sentinel_output.json"
        drift_path = REPORTS_DIR / "drift_report.csv"
        data_quality_path = REPORTS_DIR / "data_quality_report.csv"
        prediction_drift_path = REPORTS_DIR / "prediction_drift_report.json"
        cluster_path = REPORTS_DIR / "cluster_shift_report.csv"
        figure_path = self._save_drift_plot()

        json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        legacy_json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        self.data_quality_report.to_csv(data_quality_path, index=False)
        prediction_drift_path.write_text(json.dumps(self.prediction_drift_summary, indent=2), encoding="utf-8")
        self.drift_report.to_csv(drift_path, index=False)
        self.cluster_shift_report.to_csv(cluster_path, index=False)
        store = EvidenceStore()
        store.save_section("mitra", output)
        store.save_section("signal_sentinel", output)

        return {
            "json": json_path,
            "legacy_json": legacy_json_path,
            "data_quality_report": data_quality_path,
            "prediction_drift_report": prediction_drift_path,
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
