"""Timestamped run archiving utilities for AxionAI."""

from __future__ import annotations

from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def archive_run(run_id: str, artifacts: list[Path], archive_root: Path = REPORTS_DIR / "runs") -> Path:
    """Copy generated artifacts into a timestamped run folder."""
    run_dir = archive_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    for artifact in artifacts:
        artifact = Path(artifact)
        if not artifact.exists():
            continue
        relative_path = artifact.relative_to(PROJECT_ROOT) if artifact.is_absolute() else artifact
        destination = run_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if artifact.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(artifact, destination)
        else:
            shutil.copy2(artifact, destination)
    return run_dir
