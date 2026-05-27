"""SHAP-based model explainability for QSR purchase propensity predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "shap_plots"
RANDOM_SEED = 42
TOP_FEATURE_COUNT = 10


class ExplainabilityAgent:
    """Generate global and local SHAP explanations for a tree model."""

    def __init__(
        self,
        model: Any,
        X_reference: pd.DataFrame,
        X_current: pd.DataFrame,
        feature_cols: list[str],
    ) -> None:
        """Initialize the agent with model inputs in the trained feature order."""
        missing_reference = set(feature_cols) - set(X_reference.columns)
        missing_current = set(feature_cols) - set(X_current.columns)
        if missing_reference or missing_current:
            raise ValueError(
                "Feature columns must exist in both datasets. "
                f"Missing from reference: {sorted(missing_reference)}; "
                f"missing from current: {sorted(missing_current)}."
            )

        self.model = model
        self.feature_cols = feature_cols
        self.X_reference = X_reference.loc[:, feature_cols].copy()
        self.X_current = X_current.loc[:, feature_cols].copy()
        background = self.X_reference.sample(
            n=min(100, len(self.X_reference)), random_state=RANDOM_SEED
        )
        self.explainer = shap.TreeExplainer(model, data=background)

    def _shap_values(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate a two-dimensional SHAP matrix for binary classification."""
        values = self.explainer.shap_values(data.loc[:, self.feature_cols])
        if isinstance(values, list):
            values = values[-1]
        values_array = np.asarray(values)
        if values_array.ndim == 3:
            values_array = values_array[:, :, -1]
        if values_array.ndim != 2:
            raise ValueError(f"Unexpected SHAP value shape: {values_array.shape}")
        return values_array

    def _current_sample(self, sample_size: int) -> pd.DataFrame:
        """Select a reproducible sample for global explanation outputs."""
        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")
        sample_count = min(sample_size, len(self.X_current))
        return self.X_current.sample(n=sample_count, random_state=RANDOM_SEED)

    def _explanation(self, data: pd.DataFrame) -> shap.Explanation:
        """Build a SHAP Explanation object for plotting."""
        shap_values = self._shap_values(data)
        expected_value = np.asarray(self.explainer.expected_value)
        if expected_value.ndim:
            expected_value = expected_value[-1]
        return shap.Explanation(
            values=shap_values,
            base_values=np.full(len(data), float(expected_value)),
            data=data.loc[:, self.feature_cols].to_numpy(),
            feature_names=self.feature_cols,
        )

    def compute_global_importance(self, sample_size: int = 1000) -> pd.DataFrame:
        """Return features ranked by mean absolute SHAP value on current data."""
        sample = self._current_sample(sample_size)
        shap_values = self._shap_values(sample)
        importance = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "mean_abs_shap_value": np.abs(shap_values).mean(axis=0),
            }
        )
        return importance.sort_values("mean_abs_shap_value", ascending=False).reset_index(drop=True)

    def explain_user(self, user_row: pd.Series | pd.DataFrame | dict[str, Any]) -> dict[str, Any]:
        """Return the strongest positive and negative SHAP drivers for one user."""
        if isinstance(user_row, pd.Series):
            user_data = user_row.to_frame().T
        elif isinstance(user_row, dict):
            user_data = pd.DataFrame([user_row])
        else:
            user_data = user_row.copy()
        if len(user_data) != 1:
            raise ValueError("explain_user expects exactly one user row.")

        missing_columns = set(self.feature_cols) - set(user_data.columns)
        if missing_columns:
            raise ValueError(f"User row missing model features: {sorted(missing_columns)}")
        user_features = user_data.loc[:, self.feature_cols]
        shap_values = self._shap_values(user_features)[0]
        contributions = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "feature_value": user_features.iloc[0].to_numpy(),
                "shap_value": shap_values,
            }
        )
        positive = (
            contributions.loc[contributions["shap_value"] > 0]
            .sort_values("shap_value", ascending=False)
            .head(5)
        )
        negative = (
            contributions.loc[contributions["shap_value"] < 0]
            .sort_values("shap_value")
            .head(5)
        )
        return {
            "predicted_probability": float(self.model.predict_proba(user_features)[0, 1]),
            "top_positive_drivers": positive.to_dict(orient="records"),
            "top_negative_drivers": negative.to_dict(orient="records"),
        }

    @staticmethod
    def _save_current_plot(path: Path) -> None:
        """Save and close the active Matplotlib figure."""
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()

    def generate_plots(self, sample_size: int = 1000) -> dict[str, Any]:
        """Generate essential global and local SHAP visualizations as PNG files."""
        sample = self._current_sample(sample_size)
        explanation = self._explanation(sample)
        mean_importance = np.abs(explanation.values).mean(axis=0)
        top_feature = self.feature_cols[int(np.argmax(mean_importance))]

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        plt.figure()
        shap.plots.bar(explanation, max_display=TOP_FEATURE_COUNT, show=False)
        global_bar_path = PLOTS_DIR / "global_importance_bar.png"
        self._save_current_plot(global_bar_path)

        plt.figure()
        shap.plots.beeswarm(explanation, max_display=TOP_FEATURE_COUNT, show=False)
        beeswarm_path = PLOTS_DIR / "summary_beeswarm.png"
        self._save_current_plot(beeswarm_path)

        plt.figure()
        shap.plots.scatter(explanation[:, top_feature], color=explanation, show=False)
        dependence_path = PLOTS_DIR / "top_feature_dependence.png"
        self._save_current_plot(dependence_path)

        probabilities = self.model.predict_proba(self.X_current)[:, 1]
        high_propensity_position = int(np.argmax(probabilities))
        high_propensity_row = self.X_current.iloc[[high_propensity_position]]
        local_explanation = self._explanation(high_propensity_row)[0]
        plt.figure()
        shap.plots.waterfall(local_explanation, max_display=TOP_FEATURE_COUNT, show=False)
        waterfall_path = PLOTS_DIR / "high_propensity_user_waterfall.png"
        self._save_current_plot(waterfall_path)

        return {
            "global_importance_bar": str(global_bar_path.relative_to(PROJECT_ROOT)),
            "summary_beeswarm": str(beeswarm_path.relative_to(PROJECT_ROOT)),
            "top_feature_dependence": str(dependence_path.relative_to(PROJECT_ROOT)),
            "high_propensity_user_waterfall": str(waterfall_path.relative_to(PROJECT_ROOT)),
            "waterfall_current_row_position": high_propensity_position,
            "waterfall_predicted_probability": float(probabilities[high_propensity_position]),
            "dependence_feature": top_feature,
        }

    def run(self) -> pd.DataFrame:
        """Save global SHAP tables and plots, then return the ranked features."""
        global_importance = self.compute_global_importance()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        global_importance.to_csv(REPORTS_DIR / "shap_global_importance.csv", index=False)
        plot_outputs = self.generate_plots()

        top_features = {
            "sample_dataset": "current_features",
            "sample_size": min(1000, len(self.X_current)),
            "top_features": global_importance.head(TOP_FEATURE_COUNT).to_dict(orient="records"),
            "plots": plot_outputs,
        }
        with (REPORTS_DIR / "top_features.json").open("w", encoding="utf-8") as output_file:
            json.dump(top_features, output_file, indent=2)
        return global_importance


def main() -> None:
    """Load model artifacts, write SHAP reports, and print the top ten features."""
    model_path = PROJECT_ROOT / "models" / "qsr_xgb_model.joblib"
    metadata_path = PROJECT_ROOT / "models" / "model_metadata.json"
    train_path = PROJECT_ROOT / "data" / "train_features.csv"
    current_path = PROJECT_ROOT / "data" / "current_features.csv"
    required_paths = [model_path, metadata_path, train_path, current_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required model or feature artifacts: {missing_paths}")

    model = joblib.load(model_path)
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        feature_cols = json.load(metadata_file)["feature_list"]
    reference_df = pd.read_csv(train_path)
    current_df = pd.read_csv(current_path)

    agent = ExplainabilityAgent(model, reference_df, current_df, feature_cols)
    global_importance = agent.run()
    print("Top 10 features by mean absolute SHAP value:")
    print(global_importance.head(TOP_FEATURE_COUNT).to_string(index=False, float_format="%.6f"))
    print(f"\nSaved SHAP reports and plots to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
