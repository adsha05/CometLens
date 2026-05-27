"""Validate artifact handoffs across the PurchaseIntel Lens pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TARGET_COLUMN = "purchase_qsr_next_30d"
ID_COLUMN = "user_id"


def require_file(path: Path) -> None:
    """Raise a clear error when a required pipeline artifact is absent."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path.relative_to(PROJECT_ROOT)}")


def assert_columns(data: pd.DataFrame, expected: list[str], artifact_name: str) -> None:
    """Validate an artifact's exact public column contract."""
    if data.columns.tolist() != expected:
        raise AssertionError(
            f"{artifact_name} columns do not match contract. "
            f"Expected {expected}; received {data.columns.tolist()}."
        )


def validate() -> list[str]:
    """Validate generated artifacts and return completed check descriptions."""
    required_paths = [
        DATA_DIR / "train_features.csv",
        DATA_DIR / "current_features.csv",
        DATA_DIR / "train_predictions.csv",
        DATA_DIR / "current_predictions.csv",
        MODELS_DIR / "qsr_xgb_model.joblib",
        MODELS_DIR / "model_metadata.json",
        REPORTS_DIR / "drift_report.csv",
        REPORTS_DIR / "shap_global_importance.csv",
        REPORTS_DIR / "cluster_shift_report.csv",
        REPORTS_DIR / "reference_cluster_profile.csv",
        REPORTS_DIR / "current_cluster_profile.csv",
        REPORTS_DIR / "feature_suggestions.csv",
        REPORTS_DIR / "model_review_report.md",
    ]
    for path in required_paths:
        require_file(path)

    checks: list[str] = ["All required pipeline artifacts exist."]
    with (MODELS_DIR / "model_metadata.json").open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    feature_cols = metadata["feature_list"]
    if len(feature_cols) != 15:
        raise AssertionError(f"Expected 15 model features; received {len(feature_cols)}.")

    train_features = pd.read_csv(DATA_DIR / "train_features.csv")
    current_features = pd.read_csv(DATA_DIR / "current_features.csv")
    feature_schema = [ID_COLUMN, *feature_cols, TARGET_COLUMN]
    assert_columns(train_features, feature_schema, "train_features.csv")
    assert_columns(current_features, feature_schema, "current_features.csv")
    if train_features.shape != (10_000, 17) or current_features.shape != (10_000, 17):
        raise AssertionError("Feature snapshots must each contain 10,000 rows and 17 columns.")
    checks.append("Feature snapshots expose the trained 15-feature schema for 10,000 users each.")

    for artifact_name in ("train_predictions.csv", "current_predictions.csv"):
        predictions = pd.read_csv(DATA_DIR / artifact_name)
        assert_columns(predictions, [ID_COLUMN, "y_true", "y_pred_proba"], artifact_name)
        if len(predictions) != 10_000 or not predictions["y_pred_proba"].between(0, 1).all():
            raise AssertionError(f"{artifact_name} row count or probability range is invalid.")
    checks.append("Prediction artifacts align to users and contain valid probabilities.")

    drift_report = pd.read_csv(REPORTS_DIR / "drift_report.csv")
    assert_columns(
        drift_report,
        [
            "feature",
            "psi",
            "ks_statistic",
            "ks_pvalue",
            "reference_mean",
            "current_mean",
            "mean_change_pct",
            "drift_level",
        ],
        "drift_report.csv",
    )
    if set(drift_report["feature"]) != set(feature_cols):
        raise AssertionError("Drift report does not cover the complete model feature list.")
    checks.append("Drift report covers every trained feature.")

    shap_importance = pd.read_csv(REPORTS_DIR / "shap_global_importance.csv")
    assert_columns(shap_importance, ["feature", "mean_abs_shap_value"], "shap_global_importance.csv")
    if set(shap_importance["feature"]) != set(feature_cols):
        raise AssertionError("SHAP report does not cover the complete model feature list.")
    checks.append("SHAP importance covers every trained feature.")

    shift_report = pd.read_csv(REPORTS_DIR / "cluster_shift_report.csv")
    reference_profile = pd.read_csv(REPORTS_DIR / "reference_cluster_profile.csv")
    current_profile = pd.read_csv(REPORTS_DIR / "current_cluster_profile.csv")
    if len(shift_report) != 4 or reference_profile["size"].sum() != 10_000 or current_profile["size"].sum() != 10_000:
        raise AssertionError("Cluster outputs do not reconcile to four clusters and both populations.")
    if abs(float(shift_report["population_shift_pct_points"].sum())) > 1e-9:
        raise AssertionError("Cluster population shifts must net to zero percentage points.")
    checks.append("Cluster reports reconcile population counts and distribution shifts.")

    feature_suggestions = pd.read_csv(REPORTS_DIR / "feature_suggestions.csv")
    assert_columns(
        feature_suggestions,
        ["suggested_feature", "reason", "linked_evidence", "priority"],
        "feature_suggestions.csv",
    )
    checks.append("Feature recommendations satisfy their reporting schema.")

    report_text = (REPORTS_DIR / "model_review_report.md").read_text(encoding="utf-8")
    required_headings = [
        "## 1. Executive Summary",
        "## 2. Model Objective",
        "## 3. Model Performance",
        "## 4. Top Feature Drivers",
        "## 5. Drift Findings",
        "## 6. Segment/Cluster Findings",
        "## 7. Feature Recommendations",
        "## 8. Recommended Actions",
    ]
    if not all(heading in report_text for heading in required_headings):
        raise AssertionError("Model review report is missing one or more required sections.")
    checks.append("Model review Markdown contains all expected sections.")

    high_drift_count = int((drift_report["drift_level"] == "High").sum())
    risk_level = "High" if high_drift_count >= 3 else "Not High"
    if risk_level == "High" and "**Model risk level: High.**" not in report_text:
        raise AssertionError("Report risk level is inconsistent with high drift count.")
    checks.append(f"Model risk output is consistent with monitoring evidence ({risk_level}).")

    evidence_path = REPORTS_DIR / "llm_evidence_context.json"
    if evidence_path.exists():
        with evidence_path.open("r", encoding="utf-8") as evidence_file:
            evidence = json.load(evidence_file)
        if evidence["model"]["risk_level"] != risk_level:
            raise AssertionError("LLM evidence package risk does not match deterministic report evidence.")
        if len(evidence["high_drift_features"]) != high_drift_count:
            raise AssertionError("LLM evidence package does not contain all high-drift findings.")
        checks.append("Optional LLM evidence context matches deterministic monitoring results.")

    llm_json_path = REPORTS_DIR / "llm_model_review.json"
    llm_markdown_path = REPORTS_DIR / "llm_model_review.md"
    if llm_json_path.exists() or llm_markdown_path.exists():
        require_file(llm_json_path)
        require_file(llm_markdown_path)
        require_file(evidence_path)
        from src.llm.schemas import NarrativeArtifact
        from src.agents.narrative_agent import NarrativeAgent

        artifact = NarrativeArtifact.model_validate_json(llm_json_path.read_text(encoding="utf-8"))
        NarrativeAgent.validate_grounding(evidence, artifact)
        narrative_text = llm_markdown_path.read_text(encoding="utf-8")
        if artifact.model not in narrative_text or "AI Narrative Review" not in narrative_text:
            raise AssertionError("AI narrative Markdown does not match saved structured output.")
        checks.append(f"Optional AI narrative output is valid and attributed to `{artifact.provider}`.")
    return checks


def main() -> None:
    """Run integration checks and print a short audit result."""
    checks = validate()
    print("PurchaseIntel Lens integration validation passed:")
    for check in checks:
        print(f"- {check}")


if __name__ == "__main__":
    main()
