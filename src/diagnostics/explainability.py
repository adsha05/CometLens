"""Deterministic explainability helpers for Agent 02: Varuna."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cometlens-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

RANDOM_SEED = 42
EXCLUDED_MODEL_COLUMNS = {
    "prediction",
    "prediction_score",
    "predicted_label",
    "actual_label",
    "propensity_score",
    "y_true",
    "y_pred",
    "y_pred_proba",
}


def get_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    entity_id_col: str,
    time_col: str | None = None,
) -> list[str]:
    """Return numeric model feature columns while excluding IDs, labels, and outputs."""
    excluded = {target_col, entity_id_col, time_col, *EXCLUDED_MODEL_COLUMNS}
    return [
        column
        for column in df.select_dtypes(include=np.number).columns
        if column not in excluded and not column.lower().endswith(("_label", "_score", "_prediction"))
    ]


def _numeric_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return a numeric feature frame with deterministic median imputation."""
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)


def train_reference_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[Any, str]:
    """Train an XGBoost classifier, falling back to RandomForest when unavailable."""
    X = _numeric_frame(train_df, feature_cols)
    y = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0).astype(int)
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
            n_jobs=1,
            objective="binary:logistic",
            eval_metric="logloss",
        )
        model.fit(X, y)
        return model, "XGBoostClassifier"
    except Exception as error:
        warnings.warn(
            f"XGBoost reference model failed; using RandomForestClassifier fallback: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        model.fit(X, y)
        return model, "RandomForestClassifier"


def _normalize_shap_values(values: Any, feature_count: int) -> np.ndarray:
    """Normalize common SHAP output shapes to rows x features."""
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 3:
        if array.shape[1] == feature_count:
            array = array[:, :, -1]
        else:
            array = array[-1]
    if array.ndim != 2 or array.shape[1] != feature_count:
        raise ValueError(f"Unexpected SHAP shape: {array.shape}")
    return array


def compute_shap_importance(
    model: Any,
    current_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute SHAP importance or deterministic permutation-importance fallback."""
    X_current = _numeric_frame(current_df, feature_cols)
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = _normalize_shap_values(explainer.shap_values(X_current), len(feature_cols))
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        method = "shap_tree_explainer"
        warning_message = None
    except Exception as error:
        pseudo_labels = model.predict(X_current)
        result = permutation_importance(
            model,
            X_current,
            pseudo_labels,
            n_repeats=5,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        mean_abs_shap = np.abs(result.importances_mean)
        shap_values = None
        method = "permutation_importance_fallback"
        warning_message = f"SHAP failed; used permutation importance fallback: {error}"

    importance = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance["shap_rank"] = np.arange(1, len(importance) + 1)
    importance["mean_abs_shap_value"] = importance["mean_abs_shap"]
    importance.attrs["explanation_method"] = method
    importance.attrs["warning"] = warning_message
    importance.attrs["shap_values"] = shap_values
    importance.attrs["x_current"] = X_current
    return importance


def save_shap_plots(
    model: Any,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str | Path,
    importance_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Save SHAP bar and optional beeswarm plots with clear warnings."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    importance = importance_df if importance_df is not None else compute_shap_importance(model, current_df, feature_cols)
    warnings_out: list[str] = []

    bar_path = output_path / "shap_bar.png"
    top_features = importance.head(10).sort_values("mean_abs_shap")
    plt.figure(figsize=(9, 5))
    plt.barh(top_features["feature"], top_features["mean_abs_shap"], color="#3855ff")
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Global SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close()

    plots = {"shap_bar": str(bar_path)}
    shap_values = importance.attrs.get("shap_values")
    x_current = importance.attrs.get("x_current")
    if shap_values is None or x_current is None:
        warnings_out.append("SHAP beeswarm plot was not generated because SHAP values are unavailable.")
        return {"plots_generated": plots, "warnings": warnings_out}

    beeswarm_path = output_path / "shap_beeswarm.png"
    try:
        import shap

        shap.summary_plot(
            shap_values,
            x_current,
            feature_names=feature_cols,
            show=False,
            max_display=10,
        )
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
        plt.close()
        plots["shap_beeswarm"] = str(beeswarm_path)
    except Exception as error:
        plt.close()
        warnings_out.append(f"SHAP beeswarm plot was not generated: {error}")
    return {"plots_generated": plots, "warnings": warnings_out}
