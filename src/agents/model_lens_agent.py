"""Agent 02: Varuna for explainability and model-level risk diagnostics."""

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
import shap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

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


class ModelLensAgent:
    """Explain model behavior and combine explainability with model risk signals."""

    def __init__(
        self,
        train_features_path: Path = DATA_DIR / "train_features_sample.csv",
        current_features_path: Path = DATA_DIR / "current_features_sample.csv",
        predictions_path: Path = DATA_DIR / "current_predictions_sample.csv",
        model_metadata_path: Path = MODELS_DIR / "model_metadata.json",
        feature_metadata_path: Path = MODELS_DIR / "feature_metadata.json",
        signal_sentinel_path: Path = REPORTS_DIR / "signal_sentinel_output.json",
    ) -> None:
        """Configure input and output artifact paths."""
        self.train_features_path = Path(train_features_path)
        self.current_features_path = Path(current_features_path)
        self.predictions_path = Path(predictions_path)
        self.model_metadata_path = Path(model_metadata_path)
        self.feature_metadata_path = Path(feature_metadata_path)
        self.signal_sentinel_path = Path(signal_sentinel_path)
        self.train_features: pd.DataFrame | None = None
        self.current_features: pd.DataFrame | None = None
        self.predictions: pd.DataFrame | None = None
        self.model_metadata: dict[str, Any] = {}
        self.feature_metadata: dict[str, Any] = {}
        self.signal_sentinel: dict[str, Any] = {}
        self.target_col: str | None = None
        self.entity_id_col: str | None = None
        self.prediction_col: str = "propensity_score"
        self.feature_cols: list[str] = []
        self.model: XGBClassifier | XGBRegressor | None = None
        self.shap_importance: pd.DataFrame = pd.DataFrame()
        self.vif_report: pd.DataFrame = pd.DataFrame()
        self.high_risk_feature_matrix: pd.DataFrame = pd.DataFrame()
        self.output: dict[str, Any] = {}

    def load_inputs(self) -> None:
        """Load feature, prediction, metadata, and Mitra artifacts."""
        required_paths = [
            self.train_features_path,
            self.current_features_path,
            self.predictions_path,
            self.model_metadata_path,
            self.feature_metadata_path,
            self.signal_sentinel_path,
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing Agent 02: Varuna inputs. Run sample generation and Agent 01: Mitra first. "
                f"Missing: {', '.join(missing)}"
            )

        self.train_features = pd.read_csv(self.train_features_path)
        self.current_features = pd.read_csv(self.current_features_path)
        self.predictions = pd.read_csv(self.predictions_path)
        self.model_metadata = json.loads(self.model_metadata_path.read_text(encoding="utf-8"))
        self.feature_metadata = json.loads(self.feature_metadata_path.read_text(encoding="utf-8"))
        self.signal_sentinel = json.loads(self.signal_sentinel_path.read_text(encoding="utf-8"))

        self.target_col = self.model_metadata.get("target") or self.feature_metadata.get("target")
        self.entity_id_col = self.model_metadata.get("entity_id") or self.feature_metadata.get("entity_id")
        self.prediction_col = self.model_metadata.get("prediction_column", self.prediction_col)
        metadata_features = self.model_metadata.get("feature_columns") or [
            feature["name"] for feature in self.feature_metadata.get("features", [])
        ]
        self.feature_cols = [
            feature
            for feature in metadata_features
            if feature in self.train_features.columns and feature in self.current_features.columns
        ]
        if not self.feature_cols:
            numeric_columns = self.train_features.select_dtypes(include=np.number).columns.tolist()
            excluded_columns = {column for column in [self.target_col, self.entity_id_col] if column}
            self.feature_cols = [
                column
                for column in numeric_columns
                if column not in excluded_columns and column in self.current_features.columns
            ]

    def _is_classification_review(self, y: pd.Series) -> bool:
        """Infer whether the local reviewer should use classification behavior."""
        model_type = str(self.model_metadata.get("model_type", "")).lower()
        if "class" in model_type:
            return True
        if "regress" in model_type:
            return False
        return y.dropna().nunique() <= 20

    def train_model(self) -> XGBClassifier | XGBRegressor:
        """Train a small XGBoost reviewer model for local explainability."""
        if self.train_features is None:
            self.load_inputs()
        assert self.train_features is not None
        if not self.target_col or self.target_col not in self.train_features.columns:
            raise ValueError(
                "Agent 02: Varuna needs a target column in model_metadata.json and train features. "
                f"Configured target: {self.target_col!r}."
            )
        if not self.feature_cols:
            raise ValueError("Agent 02: Varuna could not identify numeric feature columns to review.")

        X = self.train_features[self.feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = pd.to_numeric(self.train_features[self.target_col], errors="coerce").fillna(0)
        is_classification = self._is_classification_review(y)
        stratify = y.astype(int) if is_classification and y.nunique() > 1 else None
        X_train, _, y_train, _ = train_test_split(
            X,
            y.astype(int) if is_classification else y,
            test_size=0.25,
            random_state=RANDOM_SEED,
            stratify=stratify,
        )
        common_params = {
            "n_estimators": 80,
            "max_depth": 3,
            "learning_rate": 0.06,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": RANDOM_SEED,
            "n_jobs": 1,
        }
        if is_classification:
            self.model = XGBClassifier(
                **common_params,
                objective="binary:logistic",
                eval_metric="logloss",
            )
        else:
            self.model = XGBRegressor(
                **common_params,
                objective="reg:squarederror",
                eval_metric="rmse",
            )
        self.model.fit(X_train, y_train)
        return self.model

    def compute_shap_importance(self) -> pd.DataFrame:
        """Compute global SHAP importance on current features."""
        if self.current_features is None:
            self.load_inputs()
        if self.model is None:
            self.train_model()
        assert self.current_features is not None
        assert self.model is not None

        X_current = self.current_features[self.feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_current)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]

        self._shap_values = shap_values
        self._x_current = X_current
        self.shap_importance = (
            pd.DataFrame(
                {
                    "feature": self.feature_cols,
                    "mean_abs_shap_value": np.abs(shap_values).mean(axis=0),
                }
            )
            .sort_values("mean_abs_shap_value", ascending=False)
            .reset_index(drop=True)
        )
        self.shap_importance["shap_rank"] = np.arange(1, len(self.shap_importance) + 1)
        return self.shap_importance

    def generate_shap_plots(self) -> dict[str, Path]:
        """Generate SHAP global bar and beeswarm plots."""
        if self.shap_importance.empty:
            self.compute_shap_importance()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        bar_path = FIGURES_DIR / "shap_global_bar.png"
        beeswarm_path = FIGURES_DIR / "shap_beeswarm.png"

        top_features = self.shap_importance.head(10).sort_values("mean_abs_shap_value")
        plt.figure(figsize=(9, 5))
        plt.barh(top_features["feature"], top_features["mean_abs_shap_value"], color="#3855ff")
        plt.xlabel("Mean absolute SHAP value")
        plt.title("Global SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(bar_path, dpi=160, bbox_inches="tight")
        plt.close()

        shap.summary_plot(
            self._shap_values,
            self._x_current,
            feature_names=self.feature_cols,
            show=False,
            max_display=10,
        )
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
        plt.close()
        return {"shap_global_bar": bar_path, "shap_beeswarm": beeswarm_path}

    @staticmethod
    def _vif_level(vif: float) -> str:
        """Classify VIF severity."""
        if vif >= 10:
            return "High"
        if vif >= 5:
            return "Medium"
        return "Low"

    def calculate_vif(self) -> pd.DataFrame:
        """Calculate VIF for numeric model features using linear-regression R-squared."""
        if self.train_features is None:
            self.load_inputs()
        assert self.train_features is not None

        X = self.train_features[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))
        rows = []
        for feature in self.feature_cols:
            y = X[feature]
            others = X.drop(columns=[feature])
            if others.empty or float(y.var()) == 0.0:
                vif = 1.0
            else:
                model = LinearRegression()
                model.fit(others, y)
                r_squared = float(model.score(others, y))
                vif = float("inf") if r_squared >= 0.999999 else 1.0 / (1.0 - r_squared)
            rows.append({"feature": feature, "vif": vif, "vif_level": self._vif_level(vif)})
        self.vif_report = pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)
        return self.vif_report

    @staticmethod
    def _overfitting_level(delta: float) -> str:
        """Classify train-validation metric delta."""
        if delta >= 0.07:
            return "High"
        if delta >= 0.03:
            return "Medium"
        return "Low"

    def calculate_overfitting_check(self) -> dict[str, Any]:
        """Calculate train-validation metric delta from metadata when available."""
        train_metric = self.model_metadata.get("train_auc", self.model_metadata.get("train_metric", 0.0))
        validation_metric = self.model_metadata.get(
            "validation_auc",
            self.model_metadata.get("validation_metric", 0.0),
        )
        metric_name = "auc" if "train_auc" in self.model_metadata else self.model_metadata.get("metric_name", "metric")
        train_auc = float(train_metric)
        validation_auc = float(validation_metric)
        delta = train_auc - validation_auc
        return {
            "metric_name": metric_name,
            "train_auc": train_auc,
            "validation_auc": validation_auc,
            "delta": delta,
            "risk_level": self._overfitting_level(delta),
        }

    def build_high_risk_feature_matrix(self) -> pd.DataFrame:
        """Combine SHAP rank, drift level, and VIF warning by feature."""
        if self.shap_importance.empty:
            self.compute_shap_importance()
        if self.vif_report.empty:
            self.calculate_vif()

        drift_rows = (
            self.signal_sentinel.get("high_drift_features", [])
            + self.signal_sentinel.get("medium_drift_features", [])
        )
        drift_by_feature = {
            row["feature"]: row.get("drift_level", "Low")
            for row in drift_rows
        }
        matrix = self.shap_importance.merge(self.vif_report, on="feature", how="left")
        matrix["drift_level"] = matrix["feature"].map(drift_by_feature).fillna("Low")
        matrix["vif_warning"] = matrix["vif_level"].fillna("Low")

        def combined_risk(row: pd.Series) -> str:
            if row["drift_level"] == "High" and row["shap_rank"] <= 5:
                return "High"
            if row["vif_warning"] == "High" and row["shap_rank"] <= 5:
                return "High"
            if row["drift_level"] in {"High", "Medium"} or row["vif_warning"] == "Medium":
                return "Medium"
            return "Low"

        matrix["combined_risk"] = matrix.apply(combined_risk, axis=1)
        self.high_risk_feature_matrix = matrix[
            [
                "feature",
                "shap_rank",
                "mean_abs_shap_value",
                "drift_level",
                "vif",
                "vif_warning",
                "combined_risk",
            ]
        ].sort_values(["combined_risk", "shap_rank"], ascending=[True, True])
        risk_order = {"High": 0, "Medium": 1, "Low": 2}
        self.high_risk_feature_matrix = (
            self.high_risk_feature_matrix.assign(
                _risk_order=self.high_risk_feature_matrix["combined_risk"].map(risk_order)
            )
            .sort_values(["_risk_order", "shap_rank"])
            .drop(columns=["_risk_order"])
            .reset_index(drop=True)
        )
        return self.high_risk_feature_matrix

    def build_output(self) -> dict[str, Any]:
        """Build final Varuna JSON output."""
        if self.train_features is None:
            self.load_inputs()
        if self.model is None:
            self.train_model()
        if self.shap_importance.empty:
            self.compute_shap_importance()
        if self.vif_report.empty:
            self.calculate_vif()
        if self.high_risk_feature_matrix.empty:
            self.build_high_risk_feature_matrix()
        plots = self.generate_shap_plots()
        overfitting_check = self.calculate_overfitting_check()

        self.output = {
            "agent_name": "Agent 02: Varuna",
            "model_name": self.model_metadata.get("model_name"),
            "top_global_drivers": self.shap_importance.head(10).to_dict(orient="records"),
            "high_risk_feature_matrix": self.high_risk_feature_matrix.to_dict(orient="records"),
            "multicollinearity_findings": self.vif_report.to_dict(orient="records"),
            "overfitting_check": overfitting_check,
            "plots_generated": {name: str(path) for name, path in plots.items()},
        }
        return self.output

    def save_outputs(self) -> dict[str, Path]:
        """Run Varuna and save JSON, CSV, and figure artifacts."""
        output = self.build_output()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_DIR / "model_lens_output.json"
        shap_path = REPORTS_DIR / "shap_global_importance.csv"
        vif_path = REPORTS_DIR / "vif_report.csv"

        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        self.shap_importance.to_csv(shap_path, index=False)
        self.vif_report.to_csv(vif_path, index=False)
        EvidenceStore().save_section("model_lens", output)
        return {
            "json": output_path,
            "shap_global_importance": shap_path,
            "vif_report": vif_path,
            "shap_global_bar": FIGURES_DIR / "shap_global_bar.png",
            "shap_beeswarm": FIGURES_DIR / "shap_beeswarm.png",
        }


def main() -> None:
    """Run Agent 02: Varuna from the command line."""
    output_paths = ModelLensAgent().save_outputs()
    print("Saved Agent 02: Varuna outputs:")
    for label, path in output_paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
