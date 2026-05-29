"""Run the VyaAI MVP pipeline end to end."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineStep:
    """One command in the VyaAI pipeline."""

    name: str
    command: list[str]


class PipelineRunner:
    """Orchestrate the deterministic VyaAI MVP workflow."""

    def __init__(self, include_narrative: bool = False) -> None:
        """Configure optional deterministic narrative generation."""
        self.include_narrative = include_narrative
        self.python = sys.executable

    def steps(self) -> list[PipelineStep]:
        """Return the ordered pipeline steps."""
        steps = [
            PipelineStep("Generate synthetic sample artifacts", [self.python, "src/generate_sample_artifacts.py"]),
            PipelineStep("Run Agent 01: Mitra", [self.python, "src/agents/signal_sentinel_agent.py"]),
            PipelineStep("Run Agent 02: Varuna", [self.python, "src/agents/model_lens_agent.py"]),
            PipelineStep("Build evidence packet", [self.python, "src/agents/evidence_store.py"]),
            PipelineStep("Run Agent 03: Aryaman", [self.python, "src/agents/executive_synthesis_agent.py"]),
        ]
        if self.include_narrative:
            steps.append(PipelineStep("Preview optional LLM prompt", [self.python, "src/llm/narrative_writer.py"]))
        return steps

    @staticmethod
    def environment() -> dict[str, str]:
        """Return stable local execution settings for plotting and sklearn."""
        env = os.environ.copy()
        env.setdefault("LOKY_MAX_CPU_COUNT", "4")
        env.setdefault("MPLCONFIGDIR", "/private/tmp/cometlens-matplotlib")
        return env

    def run(self) -> None:
        """Execute all configured pipeline steps."""
        started_at = perf_counter()
        steps = self.steps()
        print("Running VyaAI MVP pipeline", flush=True)
        print(f"Project root: {PROJECT_ROOT}", flush=True)
        print(f"Python: {self.python}", flush=True)
        print("", flush=True)
        for index, step in enumerate(steps, start=1):
            step_started_at = perf_counter()
            print(f"[{index}/{len(steps)}] {step.name}", flush=True)
            subprocess.run(step.command, cwd=PROJECT_ROOT, env=self.environment(), check=True)
            print(f"Completed in {perf_counter() - step_started_at:.1f}s", flush=True)
            print("", flush=True)
        print("VyaAI pipeline completed successfully.", flush=True)
        print(f"Total runtime: {perf_counter() - started_at:.1f}s", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the VyaAI MVP pipeline.")
    parser.add_argument(
        "--with-narrative",
        action="store_true",
        help="Also print the optional future-LLM prompt preview.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CLI pipeline."""
    args = parse_args()
    PipelineRunner(include_narrative=args.with_narrative).run()


if __name__ == "__main__":
    main()
