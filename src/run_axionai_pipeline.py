"""Run the full AxionAI MVP pipeline in order."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
from src.agents.samanvaya_agent import SamanvayaAgent
from src.agents.signal_sentinel_agent import SignalSentinelAgent
from src.generate_sample_artifacts import main as generate_sample_artifacts
from src.utils.artifact_validation import ArtifactValidationError, validate_input_contract
from src.utils.run_archive import archive_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIGS_DIR = PROJECT_ROOT / "configs"


class PipelineExecutionError(RuntimeError):
    """Raised when a pipeline step fails with a user-facing message."""


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
    try:
        result = action()
    except (ArtifactValidationError, FileNotFoundError, ValueError) as error:
        raise PipelineExecutionError(f"{name} failed: {error}") from error
    except Exception as error:
        raise PipelineExecutionError(
            f"{name} failed unexpectedly. Check artifact schemas and upstream outputs. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error
    else:
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
        DATA_DIR / "train_predictions_sample.csv",
        DATA_DIR / "current_predictions_sample.csv",
        MODELS_DIR / "model_metadata.json",
        MODELS_DIR / "feature_metadata.json",
    ]


def run_validate_artifacts() -> list[Path]:
    """Validate the input artifact contract before model intelligence agents run."""
    summary = validate_input_contract(
        train_features_path=DATA_DIR / "train_features_sample.csv",
        current_features_path=DATA_DIR / "current_features_sample.csv",
        predictions_path=DATA_DIR / "current_predictions_sample.csv",
        model_metadata_path=MODELS_DIR / "model_metadata.json",
        feature_metadata_path=MODELS_DIR / "feature_metadata.json",
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    validation_path = REPORTS_DIR / "artifact_validation.json"
    config_path = CONFIGS_DIR / "calibration_config_v1.json"
    config_version = "unknown"
    if config_path.exists():
        config_version = json.loads(config_path.read_text(encoding="utf-8")).get("config_version", "unknown")
    validation_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "config_version": config_version,
                "source_files": {
                    "train_features": str(DATA_DIR / "train_features_sample.csv"),
                    "current_features": str(DATA_DIR / "current_features_sample.csv"),
                    "current_predictions": str(DATA_DIR / "current_predictions_sample.csv"),
                    "model_metadata": str(MODELS_DIR / "model_metadata.json"),
                    "feature_metadata": str(MODELS_DIR / "feature_metadata.json"),
                    "calibration_config": str(config_path) if config_path.exists() else None,
                },
                **summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return [validation_path]


def write_pipeline_error(error: PipelineExecutionError) -> Path:
    """Persist a graceful pipeline failure message."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    error_path = REPORTS_DIR / "pipeline_error.json"
    error_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "message": str(error),
                "next_step": "Fix the artifact schema or upstream output listed in the message, then rerun python src/run_axionai_pipeline.py.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return error_path


def archive_current_run(run_id: str) -> list[Path]:
    """Archive current generated artifacts under reports/runs/<run_id>/."""
    artifacts = [
        DATA_DIR / "train_features_sample.csv",
        DATA_DIR / "current_features_sample.csv",
        DATA_DIR / "train_predictions_sample.csv",
        DATA_DIR / "current_predictions_sample.csv",
        MODELS_DIR / "model_metadata.json",
        MODELS_DIR / "feature_metadata.json",
        REPORTS_DIR / "artifact_validation.json",
        REPORTS_DIR / "mitra_output.json",
        REPORTS_DIR / "signal_sentinel_output.json",
        REPORTS_DIR / "data_quality_report.csv",
        REPORTS_DIR / "prediction_drift_report.json",
        REPORTS_DIR / "drift_report.csv",
        REPORTS_DIR / "cluster_shift_report.csv",
        REPORTS_DIR / "varuna_output.json",
        REPORTS_DIR / "model_lens_output.json",
        REPORTS_DIR / "shap_global_importance.csv",
        REPORTS_DIR / "vif_report.csv",
        REPORTS_DIR / "evidence_packet.json",
        REPORTS_DIR / "executive_model_report.json",
        REPORTS_DIR / "executive_model_report.md",
        REPORTS_DIR / "samanvaya_recommendations.json",
        REPORTS_DIR / "config_change_log.json",
        REPORTS_DIR / "figures",
        CONFIGS_DIR / "calibration_config_v1.json",
        CONFIGS_DIR / "calibration_config_v2.json",
    ]
    return [archive_run(run_id, artifacts)]


def parse_args() -> argparse.Namespace:
    """Parse pipeline CLI flags."""
    parser = argparse.ArgumentParser(description="Run the AxionAI MVP pipeline.")
    parser.add_argument(
        "--use-existing-artifacts",
        action="store_true",
        help="Skip synthetic sample generation and validate/run against existing data/ and models/ artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the AxionAI MVP pipeline from the command line."""
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stale_error_path = REPORTS_DIR / "pipeline_error.json"
    if stale_error_path.exists():
        stale_error_path.unlink()
    print("Running AxionAI MVP pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Run ID: {run_id}")
    print("")
    pipeline_started_at = perf_counter()

    try:
        step_number = 1
        if not args.use_existing_artifacts:
            run_step(f"{step_number}. Generate sample artifacts", run_generate_sample_artifacts)
            step_number += 1
        else:
            print("Skipping sample generation; using existing data/ and models/ artifacts.")

        run_step(f"{step_number}. Validate artifact contract", run_validate_artifacts)
        step_number += 1
        run_step(f"{step_number}. Run Agent 01: Mitra", lambda: SignalSentinelAgent().save_outputs())
        step_number += 1
        run_step(f"{step_number}. Run Agent 02: Varuna", lambda: ModelLensAgent().save_outputs())
        step_number += 1
        run_step(f"{step_number}. Run Evidence Store builder", lambda: [EvidencePacketBuilder().save()])
        step_number += 1
        run_step(f"{step_number}. Run Agent 03: Aryaman", lambda: ExecutiveSynthesisAgent().save_outputs())
        step_number += 1
        run_step(f"{step_number}. Run Agent 04: Samanvaya", lambda: SamanvayaAgent().save_outputs())
        step_number += 1
        run_step(f"{step_number}. Archive timestamped run", lambda: archive_current_run(run_id))
    except PipelineExecutionError as error:
        error_path = write_pipeline_error(error)
        print(f"Pipeline failed gracefully: {error}", file=sys.stderr)
        print(f"Failure details saved to {error_path}", file=sys.stderr)
        raise SystemExit(1) from error

    print(f"AxionAI MVP pipeline completed successfully in {perf_counter() - pipeline_started_at:.1f}s.")


if __name__ == "__main__":
    main()
