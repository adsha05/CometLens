"""Run the full VyaAI MVP pipeline in order."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cometlens-matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.evidence_store import EvidencePacketBuilder
from src.agents.executive_synthesis_agent import ExecutiveSynthesisAgent
from src.agents.model_lens_agent import ModelLensAgent
from src.agents.signal_sentinel_agent import SignalSentinelAgent
from src.generate_sample_artifacts import main as generate_sample_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def print_paths(paths: dict[str, Path] | list[Path] | tuple[Path, ...]) -> None:
    """Print output paths in a consistent format."""
    if isinstance(paths, dict):
        for label, path in paths.items():
            print(f"  - {label}: {path}")
        return
    for path in paths:
        print(f"  - {path}")


def run_step(name: str, action) -> object:
    """Run one pipeline step with clear status messages."""
    print(f"Starting step: {name}", flush=True)
    started_at = perf_counter()
    result = action()
    print(f"Completed step: {name} ({perf_counter() - started_at:.1f}s)", flush=True)
    if result is not None:
        print("Output file paths:", flush=True)
        print_paths(result)  # type: ignore[arg-type]
    print("", flush=True)
    return result


def run_generate_sample_artifacts() -> list[Path]:
    """Generate sample artifacts and return expected output paths."""
    generate_sample_artifacts()
    return [
        DATA_DIR / "train_features_sample.csv",
        DATA_DIR / "current_features_sample.csv",
        DATA_DIR / "current_predictions_sample.csv",
        MODELS_DIR / "model_metadata.json",
        MODELS_DIR / "feature_metadata.json",
    ]


def main() -> None:
    """Run the VyaAI MVP pipeline from the command line."""
    print("Running VyaAI MVP pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    pipeline_started_at = perf_counter()

    run_step("1. Generate sample artifacts", run_generate_sample_artifacts)
    run_step("2. Run Agent 01: Mitra", lambda: SignalSentinelAgent().save_outputs())
    run_step("3. Run Agent 02: Varuna", lambda: ModelLensAgent().save_outputs())
    run_step("4. Run Evidence Store builder", lambda: [EvidencePacketBuilder().save()])
    run_step("5. Run Agent 03: Aryaman", lambda: ExecutiveSynthesisAgent().save_outputs())

    print(f"VyaAI MVP pipeline completed successfully in {perf_counter() - pipeline_started_at:.1f}s.")


if __name__ == "__main__":
    main()
