"""Simple feedback logging for future organizational intelligence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
FEEDBACK_PATH = REPORTS_DIR / "feedback_log.csv"


def append_feedback(agent: str, signal: str, comment: str = "", path: Path = FEEDBACK_PATH) -> Path:
    """Append one dashboard feedback event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "agent": agent,
                "signal": signal,
                "comment": comment.strip(),
            }
        ]
    )
    if path.exists():
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, index=False)
    return path


def load_feedback(path: Path = FEEDBACK_PATH) -> pd.DataFrame:
    """Load feedback events when available."""
    if not path.exists():
        return pd.DataFrame(columns=["timestamp_utc", "agent", "signal", "comment"])
    return pd.read_csv(path)
