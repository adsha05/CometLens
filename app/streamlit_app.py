"""Streamlit dashboard for PurchaseIntel Lens monitoring artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load a JSON artifact, returning a user-facing issue when unavailable."""
    if not path.exists():
        return None, f"Missing `{path.relative_to(PROJECT_ROOT)}`. Generate its upstream artifact first."
    try:
        with path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file), None
    except (OSError, json.JSONDecodeError) as error:
        return None, f"Could not read `{path.relative_to(PROJECT_ROOT)}`: {error}"


def load_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    """Load a CSV artifact, returning a user-facing issue when unavailable."""
    if not path.exists():
        return None, f"Missing `{path.relative_to(PROJECT_ROOT)}`. Run the related agent first."
    try:
        return pd.read_csv(path), None
    except (OSError, pd.errors.ParserError) as error:
        return None, f"Could not read `{path.relative_to(PROJECT_ROOT)}`: {error}"


def load_markdown(path: Path) -> tuple[str | None, str | None]:
    """Load a generated Markdown report."""
    if not path.exists():
        return None, f"Missing `{path.relative_to(PROJECT_ROOT)}`. Run the report agent first."
    try:
        return path.read_text(encoding="utf-8"), None
    except OSError as error:
        return None, f"Could not read `{path.relative_to(PROJECT_ROOT)}`: {error}"


def calculate_risk_level(metadata: dict[str, Any], drift_report: pd.DataFrame) -> tuple[str, str]:
    """Calculate dashboard risk using the report-agent monitoring thresholds."""
    metrics = metadata["metrics"]
    validation_auc = float(metrics["validation"]["auc"])
    current_auc = float(metrics["current"]["auc"])
    auc_drop = validation_auc - current_auc
    high_drift_count = int((drift_report["drift_level"] == "High").sum())
    medium_drift_count = int((drift_report["drift_level"] == "Medium").sum())
    if high_drift_count >= 3 or auc_drop > 0.05:
        return "High", f"{high_drift_count} high-drift features; AUC change {current_auc - validation_auc:+.4f}."
    if medium_drift_count >= 2 or auc_drop > 0.02:
        return "Medium", f"{medium_drift_count} medium-drift features; AUC change {current_auc - validation_auc:+.4f}."
    return "Low", f"AUC change {current_auc - validation_auc:+.4f}; no risk threshold exceeded."


def render_missing(st: Any, message: str | None) -> None:
    """Render an artifact issue without interrupting other dashboard sections."""
    if message:
        st.info(message)


def main() -> None:
    """Render the PurchaseIntel Lens observability dashboard."""
    try:
        import plotly.express as px
        import streamlit as st
    except ModuleNotFoundError as error:
        print(
            f"Dashboard dependency `{error.name}` is not installed. "
            "Install requirements.txt to run the Streamlit dashboard."
        )
        return

    st.set_page_config(page_title="PurchaseIntel Lens", layout="wide")
    st.title("PurchaseIntel Lens")
    st.caption("Synthetic QSR purchase-propensity model observability dashboard")

    metadata, metadata_error = load_json(MODELS_DIR / "model_metadata.json")
    drift_report, drift_error = load_csv(REPORTS_DIR / "drift_report.csv")
    shap_importance, shap_error = load_csv(REPORTS_DIR / "shap_global_importance.csv")
    cluster_shift, cluster_error = load_csv(REPORTS_DIR / "cluster_shift_report.csv")
    feature_suggestions, suggestion_error = load_csv(REPORTS_DIR / "feature_suggestions.csv")
    review_report, review_error = load_markdown(REPORTS_DIR / "model_review_report.md")
    llm_review, llm_review_error = load_markdown(REPORTS_DIR / "llm_model_review.md")
    llm_artifact, llm_artifact_error = load_json(REPORTS_DIR / "llm_model_review.json")
    train_features, train_error = load_csv(DATA_DIR / "train_features.csv")
    current_features, current_error = load_csv(DATA_DIR / "current_features.csv")

    st.header("1. Project Overview")
    st.write(
        "PurchaseIntel Lens monitors an XGBoost purchase-propensity model using "
        "synthetic consumer behavior data. The dashboard summarizes predictive "
        "performance, explanations, feature drift, customer segments, and follow-up features."
    )
    if train_features is not None and current_features is not None:
        overview_columns = st.columns(3)
        overview_columns[0].metric("Training Rows", f"{len(train_features):,}")
        overview_columns[1].metric("Current Rows", f"{len(current_features):,}")
        overview_columns[2].metric("Model Features", str(len(metadata["feature_list"])) if metadata else "N/A")
    else:
        render_missing(st, train_error)
        render_missing(st, current_error)

    st.header("2. Model Health Summary")
    if metadata is not None and drift_report is not None:
        risk_level, risk_reason = calculate_risk_level(metadata, drift_report)
        high_drift_count = int((drift_report["drift_level"] == "High").sum())
        health_columns = st.columns(4)
        health_columns[0].metric("Risk Level", risk_level)
        health_columns[1].metric("High Drift Features", high_drift_count)
        health_columns[2].metric("Validation AUC", f"{metadata['metrics']['validation']['auc']:.4f}")
        health_columns[3].metric("Current AUC", f"{metadata['metrics']['current']['auc']:.4f}")
        if risk_level == "High":
            st.error(f"High model monitoring risk: {risk_reason}")
        elif risk_level == "Medium":
            st.warning(f"Medium model monitoring risk: {risk_reason}")
        else:
            st.success(f"Low model monitoring risk: {risk_reason}")
    else:
        render_missing(st, metadata_error)
        render_missing(st, drift_error)

    st.header("3. Model Performance Metrics")
    if metadata is not None:
        performance = pd.DataFrame(metadata["metrics"]).T.reset_index(names="dataset")
        performance["dataset"] = performance["dataset"].str.title()
        st.dataframe(performance.style.format({column: "{:.4f}" for column in performance.columns if column != "dataset"}), width="stretch")
    else:
        render_missing(st, metadata_error)

    st.header("4. Top SHAP Feature Drivers")
    if shap_importance is not None:
        top_shap = shap_importance.nlargest(10, "mean_abs_shap_value").sort_values("mean_abs_shap_value")
        shap_chart = px.bar(
            top_shap,
            x="mean_abs_shap_value",
            y="feature",
            orientation="h",
            title="Top 10 Features by Mean Absolute SHAP Value",
            labels={"mean_abs_shap_value": "Mean |SHAP Value|", "feature": "Feature"},
        )
        st.plotly_chart(shap_chart, width="stretch")
        st.dataframe(top_shap.sort_values("mean_abs_shap_value", ascending=False), width="stretch")
    else:
        render_missing(st, shap_error)

    st.header("5. Drift Report Table")
    if drift_report is not None:
        drift_display = drift_report.sort_values("psi", ascending=False)
        psi_chart = px.bar(
            drift_display.head(10).sort_values("psi"),
            x="psi",
            y="feature",
            color="drift_level",
            orientation="h",
            title="Top PSI Drift Features",
            labels={"psi": "Population Stability Index", "feature": "Feature"},
            color_discrete_map={"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"},
        )
        st.plotly_chart(psi_chart, width="stretch")
        st.dataframe(drift_display, width="stretch")
    else:
        render_missing(st, drift_error)

    st.header("6. Cluster Shift Report")
    if cluster_shift is not None:
        cluster_chart = px.bar(
            cluster_shift.sort_values("population_shift_pct_points"),
            x="population_shift_pct_points",
            y="cluster_name",
            orientation="h",
            title="Cluster Distribution Shift",
            labels={"population_shift_pct_points": "Share Change (percentage points)", "cluster_name": "Segment"},
            color="population_shift_pct_points",
            color_continuous_scale="RdBu",
        )
        st.plotly_chart(cluster_chart, width="stretch")
        st.dataframe(cluster_shift, width="stretch")
    else:
        render_missing(st, cluster_error)

    st.header("7. Feature Suggestions")
    if feature_suggestions is not None:
        st.dataframe(feature_suggestions, width="stretch")
    else:
        render_missing(st, suggestion_error)

    st.header("8. Model Review Report")
    if review_report is not None:
        st.markdown(review_report)
    else:
        render_missing(st, review_error)

    st.header("9. AI Narrative Review")
    st.caption(
        "Optional LLM interpretation over validated synthetic-data evidence. "
        "Metrics, drift, clustering, and risk thresholds remain deterministic."
    )
    if llm_review is not None and llm_artifact is not None:
        narrative_columns = st.columns(3)
        narrative_columns[0].metric("Provider", llm_artifact["provider"])
        narrative_columns[1].metric("LLM Model", llm_artifact["model"])
        narrative_columns[2].metric("Evidence Source", "Validated reports")
        st.markdown(llm_review)
    else:
        st.info(
            "No AI narrative has been generated yet. Start Ollama, pull the configured model, "
            "then run `python src/agents/narrative_agent.py`."
        )
        render_missing(st, llm_review_error)
        render_missing(st, llm_artifact_error)


if __name__ == "__main__":
    main()
