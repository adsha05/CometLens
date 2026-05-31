"""Backward-compatible feedback helpers backed by the governed memory store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.memory.feedback_store import (
    DEFAULT_FEEDBACK_PATH,
    append_feedback_event,
    ensure_feedback_log,
    load_feedback_log,
)

FEEDBACK_PATH = DEFAULT_FEEDBACK_PATH


def append_feedback(agent: str, signal: str, comment: str = "", path: Path = FEEDBACK_PATH) -> Path:
    """Append one legacy dashboard feedback event using the governed schema."""
    feedback_type = {
        "useful": "helpful",
        "not useful": "not_helpful",
        "false positive": "false_alarm",
    }.get(signal, signal)
    return append_feedback_event(
        path,
        {
            "user_role": "analyst",
            "finding_id": "",
            "feature": "",
            "feedback_type": feedback_type,
            "severity": "Not Set",
            "comment": comment,
            "related_agent": agent,
            "action_taken": "",
        },
    )


def load_feedback(path: Path = FEEDBACK_PATH) -> pd.DataFrame:
    """Load feedback in the governed schema."""
    ensure_feedback_log(path)
    return load_feedback_log(path)
