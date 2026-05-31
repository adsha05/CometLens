"""Auditable CSV feedback storage for Agent 04: Samanvaya."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEEDBACK_PATH = PROJECT_ROOT / "reports" / "feedback_log.csv"
FEEDBACK_COLUMNS = [
    "timestamp",
    "run_id",
    "user_role",
    "finding_id",
    "feature",
    "feedback_type",
    "severity",
    "comment",
    "related_agent",
    "action_taken",
]
ALLOWED_FEEDBACK_TYPES = {
    "helpful",
    "not_helpful",
    "false_alarm",
    "accepted_recommendation",
    "rejected_recommendation",
    "too_technical",
    "too_vague",
    "needs_more_detail",
    "wrong_audience",
    "requires_follow_up",
}


def _sample_feedback_rows() -> list[dict[str, str]]:
    """Return explicitly synthetic demo feedback events."""
    timestamp = datetime.now(timezone.utc).isoformat()
    return [
        {
            "timestamp": timestamp,
            "run_id": "synthetic_demo",
            "user_role": "model_analyst",
            "finding_id": "MITRA_DRIFT_001",
            "feature": "fuel_spend_30d",
            "feedback_type": "false_alarm",
            "severity": "Medium",
            "comment": "Synthetic demo feedback: expected seasonal fuel movement.",
            "related_agent": "Mitra",
            "action_taken": "review_requested",
        },
        {
            "timestamp": timestamp,
            "run_id": "synthetic_demo",
            "user_role": "model_analyst",
            "finding_id": "MITRA_DRIFT_001",
            "feature": "fuel_spend_30d",
            "feedback_type": "false_alarm",
            "severity": "Medium",
            "comment": "Synthetic demo feedback: repeat seasonal review signal.",
            "related_agent": "Mitra",
            "action_taken": "review_requested",
        },
        {
            "timestamp": timestamp,
            "run_id": "synthetic_demo",
            "user_role": "executive",
            "finding_id": "ARYAMAN_REPORT_STYLE",
            "feature": "",
            "feedback_type": "too_technical",
            "severity": "Low",
            "comment": "Synthetic demo feedback: simplify executive language.",
            "related_agent": "Aryaman",
            "action_taken": "report_review_requested",
        },
        {
            "timestamp": timestamp,
            "run_id": "synthetic_demo",
            "user_role": "client_safe",
            "finding_id": "ARYAMAN_REPORT_STYLE",
            "feature": "",
            "feedback_type": "too_technical",
            "severity": "Low",
            "comment": "Synthetic demo feedback: hide technical appendix by default.",
            "related_agent": "Aryaman",
            "action_taken": "report_review_requested",
        },
    ]


def ensure_feedback_log(
    path: str | Path = DEFAULT_FEEDBACK_PATH,
    *,
    demo_mode: bool = False,
) -> Path:
    """Create a feedback CSV with headers and optional synthetic demo rows."""
    feedback_path = Path(path)
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    if not feedback_path.exists():
        frame = pd.DataFrame(_sample_feedback_rows() if demo_mode else [], columns=FEEDBACK_COLUMNS)
        frame.to_csv(feedback_path, index=False)
    return feedback_path


def append_feedback_event(path: str | Path, event: dict[str, Any]) -> Path:
    """Validate and append one feedback event."""
    feedback_path = ensure_feedback_log(path)
    feedback_type = str(event.get("feedback_type", "")).strip()
    if feedback_type not in ALLOWED_FEEDBACK_TYPES:
        allowed = ", ".join(sorted(ALLOWED_FEEDBACK_TYPES))
        raise ValueError(f"Unsupported feedback_type `{feedback_type}`. Allowed values: {allowed}")
    row = {
        "timestamp": str(event.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        "run_id": str(event.get("run_id", "unknown")),
        "user_role": str(event.get("user_role", "unknown")),
        "finding_id": str(event.get("finding_id", "")),
        "feature": str(event.get("feature", "")),
        "feedback_type": feedback_type,
        "severity": str(event.get("severity", "Not Set")),
        "comment": str(event.get("comment", "")).strip(),
        "related_agent": str(event.get("related_agent", "")),
        "action_taken": str(event.get("action_taken", "")),
    }
    pd.DataFrame([row], columns=FEEDBACK_COLUMNS).to_csv(feedback_path, mode="a", header=False, index=False)
    return feedback_path


def load_feedback_log(path: str | Path = DEFAULT_FEEDBACK_PATH) -> pd.DataFrame:
    """Load feedback records with the governed column contract."""
    feedback_path = ensure_feedback_log(path)
    frame = pd.read_csv(feedback_path).fillna("")
    for column in FEEDBACK_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[FEEDBACK_COLUMNS]


def summarize_feedback(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize feedback counts without changing calibration behavior."""
    if df.empty:
        return {
            "total_events": 0,
            "feedback_type_counts": {},
            "related_agent_counts": {},
            "feature_counts": {},
            "user_role_counts": {},
        }
    non_empty_features = df.loc[df["feature"].astype(str).str.strip() != "", "feature"]
    return {
        "total_events": int(len(df)),
        "feedback_type_counts": df["feedback_type"].value_counts().to_dict(),
        "related_agent_counts": df["related_agent"].value_counts().to_dict(),
        "feature_counts": non_empty_features.value_counts().to_dict(),
        "user_role_counts": df["user_role"].value_counts().to_dict(),
    }
