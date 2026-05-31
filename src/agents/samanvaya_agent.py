"""Backward-compatible adapter for Agent 04: Samanvaya."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.samanvaya_calibration_agent import SamanvayaCalibrationAgent, main
from src.memory.feedback_store import DEFAULT_FEEDBACK_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SamanvayaAgent(SamanvayaCalibrationAgent):
    """Preserve the earlier Samanvaya import while using governed calibration."""

    def __init__(
        self,
        feedback_path: Path = DEFAULT_FEEDBACK_PATH,
        calibration_config_path: Path = PROJECT_ROOT / "configs" / "calibration_config_v1.json",
        **kwargs: Any,
    ) -> None:
        """Translate legacy path arguments into the governed path contract."""
        paths = dict(kwargs.pop("paths", {}) or {})
        paths.update({"feedback_log": feedback_path, "calibration_config": calibration_config_path})
        super().__init__(paths=paths, **kwargs)

    def build_recommendations(self) -> dict[str, Any]:
        """Return recommendations using the earlier method name."""
        return self.generate_calibration_recommendations()


if __name__ == "__main__":
    main()
