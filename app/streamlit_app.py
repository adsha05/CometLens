"""Streamlit dashboard for the AxionAI MVP."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import plotly.io as pio
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.samanvaya_calibration_agent import SamanvayaCalibrationAgent
from src.memory.feedback_store import (
    ALLOWED_FEEDBACK_TYPES,
    append_feedback_event,
    ensure_feedback_log,
    load_feedback_log,
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
VISUALS_DIR = REPORTS_DIR / "visuals"

RUN_ORDER = """python src/generate_sample_artifacts.py
python src/agents/signal_sentinel_agent.py
python src/agents/model_lens_agent.py
python src/agents/evidence_store.py
python src/agents/vishwakarma_visual_architect.py
python src/agents/evidence_store.py
python src/agents/executive_synthesis_agent.py
python src/agents/samanvaya_calibration_agent.py --demo
streamlit run app/streamlit_app.py"""


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON when available."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_with_fallback(path: Path, fallback_path: Path) -> dict[str, Any]:
    """Load a preferred JSON file, falling back to a legacy path."""
    if path.exists():
        return load_json(path)
    return load_json(fallback_path)


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert JSON records into a DataFrame for display."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def missing_file_message(path: Path, label: str) -> None:
    """Show a helpful message for missing dashboard artifacts."""
    st.info(f"{label} is missing: `{path}`. Run the pipeline commands below.")
    st.code(RUN_ORDER, language="bash")


def show_image_if_available(path: Path, caption: str) -> None:
    """Render an image if it exists, otherwise show a helpful notice."""
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Plot not available yet: `{path}`")


def overview_section() -> None:
    """Render the AxionAI overview and architecture summary."""
    st.header("1. Overview")
    st.write(
        "AxionAI is an agentic model intelligence MVP for reviewing synthetic tabular model artifacts. "
        "It does not deploy a production model or use real customer data. Instead, it reads "
        "feature tables, predictions, model metadata, and feature metadata, then produces "
        "auditable model-health evidence. The bundled QSR profile is only a demo; the agents "
        "use metadata-defined target, entity, prediction, and feature columns."
    )
    st.subheader("5-Agent Architecture")
    st.markdown(
        """
        - **Agent 01: Mitra** detects feature drift, prediction drift, missing-value shifts, and cluster/context movement.
        - **Agent 02: Varuna** explains model behavior with a small local XGBoost reviewer model, SHAP, VIF, and overfitting checks.
        - **Agent 03: Aryaman** converts verified evidence into a consulting-style model health brief.
        - **Agent 04: Samanvaya** reads feedback and proposes calibration changes for human review.
        - **Agent 05: Vishwakarma** renders report-ready risk visuals and a run-specific lineage graph.
        """
    )
    with st.expander("Run Order"):
        st.code(RUN_ORDER, language="bash")


def model_metadata_section() -> None:
    """Render model metadata."""
    st.header("2. Model Metadata")
    path = MODELS_DIR / "model_metadata.json"
    metadata = load_json(path)
    if not metadata:
        missing_file_message(path, "Model metadata")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", metadata.get("model_name", "unknown"))
    col2.metric("Type", metadata.get("model_type", "unknown"))
    col3.metric("Target", metadata.get("target", "unknown"))
    st.write(f"**Use case:** {metadata.get('business_use_case', 'Not provided')}")

    metrics = {
        "Train Metric": metadata.get("train_auc", metadata.get("train_metric")),
        "Validation Metric": metadata.get("validation_auc", metadata.get("validation_metric")),
        "Current Metric": metadata.get("current_auc", metadata.get("current_metric")),
        "Metric Name": metadata.get("metric_name", "auc" if "train_auc" in metadata else "metric"),
        "Precision": metadata.get("precision"),
        "Recall": metadata.get("recall"),
        "F1": metadata.get("f1"),
    }
    st.dataframe(pd.DataFrame([metrics]), width="stretch")
    with st.expander("Raw model metadata"):
        st.json(metadata)


def signal_sentinel_section() -> None:
    """Render Mitra outputs."""
    st.header("3. Agent 01: Mitra")
    path = REPORTS_DIR / "mitra_output.json"
    signal = load_json_with_fallback(path, REPORTS_DIR / "signal_sentinel_output.json")
    if not signal:
        missing_file_message(path, "Agent 01: Mitra output")
        return

    high_drift = signal.get("high_drift_features", [])
    cluster_findings = signal.get("cluster_findings", [])
    prediction_summary = signal.get("prediction_drift_summary", {})
    data_health = signal.get("data_health_summary", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Level", signal.get("overall_risk_level", "unknown"))
    col2.metric("High Drift Features", len(high_drift))
    col3.metric("Prediction Gap", f"{prediction_summary.get('prediction_actual_rate_gap', 0):+.3f}")
    quality_counts = data_health.get("data_quality_issue_counts", {})
    col4.metric(
        "Data Quality Issues",
        int(quality_counts.get("High", 0)) + int(quality_counts.get("Medium", 0)),
        help="Count of high and medium data-quality checks.",
    )
    if signal.get("overall_risk_explanation"):
        st.caption(f"Risk explanation: {signal['overall_risk_explanation']}")
    if signal.get("risk_assessment"):
        with st.expander("Mitra risk rule hierarchy", expanded=False):
            st.json(signal["risk_assessment"])

    st.subheader("Data Quality Gate")
    data_quality_path = REPORTS_DIR / "data_quality_report.csv"
    if data_quality_path.exists():
        data_quality_df = pd.read_csv(data_quality_path)
        issue_filter = st.multiselect(
            "Filter data quality issue level",
            options=["High", "Medium", "Low"],
            default=["High", "Medium"],
        )
        filtered_quality_df = data_quality_df.loc[data_quality_df["issue_level"].isin(issue_filter)]
        if filtered_quality_df.empty:
            st.success("No selected data quality issues.")
        else:
            st.dataframe(filtered_quality_df, width="stretch")
    else:
        st.info("Data quality report is missing. Run the pipeline to generate `reports/data_quality_report.csv`.")

    st.subheader("High Drift Features")
    high_drift_df = records_to_frame(high_drift)
    if high_drift_df.empty:
        st.success("No high-drift features reported.")
    else:
        st.dataframe(high_drift_df, width="stretch")

    st.subheader("Feature Drift Report")
    drift_path = REPORTS_DIR / "drift_report.csv"
    if drift_path.exists():
        drift_df = pd.read_csv(drift_path)
        st.dataframe(drift_df, width="stretch")
    else:
        st.info("Drift report is missing. Run the pipeline to generate `reports/drift_report.csv`.")

    st.subheader("Prediction Drift")
    prediction_drift_path = REPORTS_DIR / "prediction_drift_report.json"
    prediction_drift = load_json(prediction_drift_path) or prediction_summary
    if prediction_drift:
        pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
        pred_col1.metric("Score Drift", prediction_drift.get("prediction_drift_level", "Unknown"))
        pred_col2.metric("Score PSI", f"{prediction_drift.get('score_psi', 0):.3f}")
        pred_col3.metric(
            "Score Mean Change",
            f"{prediction_drift.get('score_mean_change_pct', 0):+.1f}%",
        )
        pred_col4.metric(
            "Predicted Positive Shift",
            f"{prediction_drift.get('predicted_positive_rate_change_pct_points', 0):+.1f} pp",
        )
        with st.expander("Prediction drift details", expanded=False):
            st.json(prediction_drift)
    else:
        st.info("Prediction drift report is missing. Run the pipeline to generate `reports/prediction_drift_report.json`.")

    st.subheader("Cluster Findings")
    cluster_df = records_to_frame(cluster_findings)
    if cluster_df.empty:
        st.info("No cluster findings available.")
    else:
        st.dataframe(cluster_df, width="stretch")

    st.subheader("Drift Plot")
    show_image_if_available(FIGURES_DIR / "drift_top_features.png", "Top drift features by PSI")


def model_lens_section() -> None:
    """Render Varuna outputs."""
    st.header("4. Agent 02: Varuna — Model Auditor")
    path = REPORTS_DIR / "varuna_output.json"
    lens = load_json_with_fallback(path, REPORTS_DIR / "model_lens_output.json")
    if not lens:
        missing_file_message(path, "Agent 02: Varuna output")
        return

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Reference Model", lens.get("reference_model_type", "unknown"))
    metric2.metric("Explanation Method", lens.get("explanation_method", "unknown"))
    metric3.metric("Config Version", lens.get("config_version", "unknown"))
    for warning in lens.get("warnings", []):
        st.warning(warning)

    st.subheader("Top Global Drivers")
    reliability = lens.get("explainability_reliability", {})
    if reliability.get("status") == "unreliable":
        st.warning(
            "Varuna reliability warning: SHAP outputs are directional only because Mitra found severe drift."
        )
        with st.expander("Reliability gate details"):
            st.json(reliability)
    elif reliability:
        st.success("Varuna reliability gate passed.")

    top_drivers_df = records_to_frame(lens.get("top_global_drivers", []))
    if top_drivers_df.empty:
        st.info("Top global drivers are missing.")
    else:
        st.dataframe(top_drivers_df, width="stretch")

    st.subheader("Feature Risk Matrix")
    feature_risk_path = REPORTS_DIR / "feature_risk_matrix.csv"
    risk_matrix_df = (
        pd.read_csv(feature_risk_path)
        if feature_risk_path.exists()
        else records_to_frame(lens.get("high_risk_feature_matrix", []))
    )
    if risk_matrix_df.empty:
        st.info("Feature risk matrix is missing.")
    else:
        st.dataframe(risk_matrix_df, width="stretch")

    st.subheader("VIF And Overfitting Findings")
    left, right = st.columns(2)
    with left:
        vif_df = records_to_frame(lens.get("multicollinearity_findings", []))
        if vif_df.empty:
            st.info("VIF report is missing.")
        else:
            st.dataframe(vif_df, width="stretch")
    with right:
        overfitting = lens.get("overfitting_check", {})
        if overfitting:
            st.json(overfitting)
        else:
            st.info("Overfitting check is missing.")

    diagnostics_path = REPORTS_DIR / "model_diagnostics.json"
    diagnostics = load_json(diagnostics_path)
    if diagnostics:
        with st.expander("Varuna diagnostics JSON", expanded=False):
            st.json(diagnostics)

    st.subheader("SHAP Plots")
    col1, col2 = st.columns(2)
    with col1:
        show_image_if_available(FIGURES_DIR / "shap_bar.png", "SHAP global bar plot")
    with col2:
        show_image_if_available(FIGURES_DIR / "shap_beeswarm.png", "SHAP beeswarm plot")


def executive_synthesis_section() -> None:
    """Render Aryaman markdown output."""
    st.header("7. Agent 03: Aryaman — Executive Synthesis")
    output_path = REPORTS_DIR / "aryaman_output.json"
    output = load_json(output_path)
    markdown_path = REPORTS_DIR / "executive_model_report.md"
    if not output:
        missing_file_message(output_path, "Agent 03: Aryaman output")
        return
    st.metric("Model Health Status", output.get("model_health_status", "unknown"))
    st.subheader("Executive Summary")
    st.markdown(output.get("executive_summary", "No executive summary available."))
    st.subheader("Key Findings")
    for finding in output.get("key_findings_used", []):
        st.markdown(f"- {finding}")
    st.subheader("Recommended Actions")
    for action in output.get("recommended_actions", []):
        st.markdown(f"- {action}")
    if markdown_path.exists():
        with st.expander("Render full executive report", expanded=False):
            st.markdown(markdown_path.read_text(encoding="utf-8"))
    else:
        st.info(f"Executive Markdown report is missing: `{markdown_path}`")


def evidence_packet_section() -> None:
    """Render evidence packet in an expandable JSON viewer."""
    st.header("5. Evidence Packet")
    path = REPORTS_DIR / "evidence_packet.json"
    evidence = load_json(path)
    if not evidence:
        missing_file_message(path, "Evidence packet")
        return
    with st.expander("View evidence packet JSON", expanded=False):
        st.json(evidence)


def vishwakarma_section() -> None:
    """Render Vishwakarma visual intelligence outputs."""
    st.header("6. Agent 05: Vishwakarma — Visual Intelligence")
    manifest_path = VISUALS_DIR / "vishwakarma_output.json"
    manifest = load_json(manifest_path)
    if not manifest:
        missing_file_message(manifest_path, "Agent 05: Vishwakarma output")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Visual Agent", manifest.get("agent", "unknown"))
    col2.metric("Visuals Generated", len(manifest.get("visuals_generated", {})))
    col3.metric("Config Version", manifest.get("config_version", "unknown"))
    for warning in manifest.get("warnings", []):
        st.warning(warning)

    st.subheader("Feature Risk Map")
    scatter_path = VISUALS_DIR / "feature_risk_scatter.json"
    if scatter_path.exists():
        st.plotly_chart(pio.from_json(scatter_path.read_text(encoding="utf-8")), width="stretch")
    else:
        st.info(f"Feature risk visual is missing: `{scatter_path}`")

    st.subheader("Prediction Distribution Overlay")
    overlay_path = VISUALS_DIR / "prediction_distribution_overlay.json"
    if overlay_path.exists():
        st.plotly_chart(pio.from_json(overlay_path.read_text(encoding="utf-8")), width="stretch")
    else:
        st.info(f"Prediction overlay is not available: `{overlay_path}`")

    st.subheader("Run-Specific Lineage Graph")
    lineage_path = VISUALS_DIR / "lineage_graph.svg"
    if lineage_path.exists():
        st.html(lineage_path.read_text(encoding="utf-8"))
    else:
        st.info(f"Lineage SVG is missing: `{lineage_path}`")

    with st.expander("Vishwakarma visual manifest", expanded=False):
        st.json(manifest)


def samanvaya_section() -> None:
    """Render governed feedback capture and Samanvaya calibration review."""
    st.header("8. Agent 04: Samanvaya — Feedback Calibration")
    st.caption(
        "Samanvaya proposes auditable calibration updates for human approval. "
        "It never changes active agent behavior automatically."
    )
    feedback_path = ensure_feedback_log(REPORTS_DIR / "feedback_log.csv")
    evidence = load_json(REPORTS_DIR / "evidence_packet.json")

    st.subheader("Submit Feedback Signal")
    with st.form("samanvaya_feedback_form"):
        user_role = st.selectbox("User role", ["model_analyst", "data_scientist", "executive", "client_safe"])
        finding_id = st.text_input("Finding ID", placeholder="Example: MITRA_DRIFT_001")
        feature = st.text_input("Feature", placeholder="Example: merchant_novelty_rate")
        feedback_type = st.selectbox("Feedback type", sorted(ALLOWED_FEEDBACK_TYPES))
        severity = st.selectbox("Severity", ["Low", "Medium", "High", "Not Set"])
        related_agent = st.selectbox("Related agent", ["Mitra", "Varuna", "Aryaman", "Vishwakarma"])
        comment = st.text_area("Comment", placeholder="Explain why this signal is useful, noisy, or needs follow-up.")
        submitted = st.form_submit_button("Submit feedback")
    if submitted:
        path = append_feedback_event(
            feedback_path,
            {
                "run_id": evidence.get("run_id", "unknown"),
                "user_role": user_role,
                "finding_id": finding_id,
                "feature": feature,
                "feedback_type": feedback_type,
                "severity": severity,
                "comment": comment,
                "related_agent": related_agent,
                "action_taken": "submitted_for_calibration_review",
            },
        )
        st.success(f"Saved feedback to {path}. Run the calibration review to refresh recommendations.")

    feedback = load_feedback_log(feedback_path)
    st.subheader("Feedback Log")
    if feedback.empty:
        st.info("No feedback events are available yet.")
    else:
        st.dataframe(feedback, width="stretch")
        st.subheader("Feedback Type Counts")
        st.dataframe(
            feedback["feedback_type"].value_counts().rename_axis("feedback_type").reset_index(name="count"),
            width="stretch",
        )

    if st.button("Run Samanvaya Calibration Review"):
        try:
            output_paths = SamanvayaCalibrationAgent().save_outputs()
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Samanvaya could not run: {error}")
        else:
            st.success(f"Samanvaya review completed. Saved {len(output_paths)} governed artifacts.")

    output_path = REPORTS_DIR / "samanvaya_output.json"
    recommendations_path = REPORTS_DIR / "calibration_recommendations.json"
    config_path = PROJECT_ROOT / "configs" / "calibration_config_v2_recommended.json"
    output = load_json(output_path)
    recommendations = load_json(recommendations_path)
    if not output:
        missing_file_message(output_path, "Agent 04: Samanvaya output")
        return

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Recommendations", output.get("recommendations_count", 0))
    metric2.metric("Pending Human Approval", output.get("pending_human_approval_count", 0))
    metric3.metric("High Confidence", output.get("high_confidence_recommendations", 0))
    recommendation_rows = recommendations.get("recommendations", [])
    st.subheader("Calibration Recommendations")
    if recommendation_rows:
        recommendations_df = pd.DataFrame(recommendation_rows)
        for column in ["current_value", "recommended_value", "evidence_used"]:
            if column in recommendations_df:
                recommendations_df[column] = recommendations_df[column].map(
                    lambda value: json.dumps(value) if isinstance(value, (dict, list)) else value
                )
        st.dataframe(recommendations_df, width="stretch")
    else:
        st.info("No calibration changes are recommended at this time.")

    with st.expander("Samanvaya output JSON", expanded=False):
        st.json(output)
    with st.expander("Calibration recommendations JSON", expanded=False):
        st.json(recommendations)
    if config_path.exists():
        st.download_button(
            "Download recommended calibration config",
            data=config_path.read_text(encoding="utf-8"),
            file_name=config_path.name,
            mime="application/json",
        )
    else:
        st.info(f"Recommended calibration config is missing: `{config_path}`")


def main() -> None:
    """Render the complete AxionAI dashboard."""
    st.set_page_config(page_title="AxionAI MVP", layout="wide")
    st.title("AxionAI MVP")
    st.caption("Executive model intelligence for generic tabular model artifact review")

    overview_section()
    model_metadata_section()
    signal_sentinel_section()
    model_lens_section()
    evidence_packet_section()
    vishwakarma_section()
    executive_synthesis_section()
    samanvaya_section()


if __name__ == "__main__":
    main()
