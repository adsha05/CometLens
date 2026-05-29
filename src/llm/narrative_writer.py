"""Prompt utilities for optional future VyaAI LLM narrative generation.

This module intentionally does not call OpenAI, Ollama, or any other external
LLM API. The deterministic executive report remains implemented in
`src/agents/executive_synthesis_agent.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.schemas import ExecutiveModelReport

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

SYSTEM_PROMPT = """You are an executive model intelligence analyst.
Use only supplied evidence.
Do not invent metrics, features, or causes.
Clearly separate facts from hypotheses.
Write for data science leaders and business stakeholders."""


def _compact_json(value: Any) -> str:
    """Serialize evidence compactly while preserving deterministic values."""
    return json.dumps(value, indent=2, sort_keys=True)


def build_llm_prompt(evidence: dict) -> str:
    """Convert an evidence packet into a compact prompt for a future LLM call."""
    model_metadata = evidence.get("model_metadata", {})
    business_context = evidence.get("business_context", {})
    signal_summary = evidence.get("signal_sentinel_summary", {})
    model_lens_summary = evidence.get("model_lens_summary", {})

    prompt_sections = {
        "task": (
            "Create an ExecutiveModelReport JSON object. Use the schema fields exactly. "
            "Do not calculate new metrics. Do not add unsupported causes."
        ),
        "model_context": {
            "model_name": model_metadata.get("model_name"),
            "model_type": model_metadata.get("model_type"),
            "target": model_metadata.get("target"),
            "business_use_case": business_context.get("business_use_case"),
            "training_window": business_context.get("training_window"),
            "current_window": business_context.get("current_window"),
        },
        "deterministic_key_findings": evidence.get("key_findings", []),
        "signal_sentinel_summary": {
            "overall_risk_level": signal_summary.get("overall_risk_level"),
            "high_drift_feature_count": signal_summary.get("high_drift_feature_count"),
            "medium_drift_feature_count": signal_summary.get("medium_drift_feature_count"),
            "prediction_drift_summary": signal_summary.get("prediction_drift_summary", {}),
            "cluster_findings": signal_summary.get("cluster_findings", []),
        },
        "model_lens_summary": {
            "top_global_drivers": model_lens_summary.get("top_global_drivers", [])[:5],
            "high_risk_feature_matrix": model_lens_summary.get("high_risk_feature_matrix", []),
            "overfitting_check": model_lens_summary.get("overfitting_check", {}),
            "multicollinearity_findings": model_lens_summary.get("multicollinearity_findings", []),
        },
        "available_plots": evidence.get("available_plots", {}),
        "limitations": evidence.get("limitations", []),
        "output_schema": ExecutiveModelReport.model_json_schema(),
    }
    return f"{SYSTEM_PROMPT}\n\nEvidence packet:\n{_compact_json(prompt_sections)}"


def validate_llm_report_dict(report_dict: dict) -> ExecutiveModelReport:
    """Validate a future LLM response against the ExecutiveModelReport schema."""
    return ExecutiveModelReport.model_validate(report_dict)


def load_evidence_packet(path: Path = REPORTS_DIR / "evidence_packet.json") -> dict:
    """Load the deterministic evidence packet for prompt preparation."""
    if not path.exists():
        raise FileNotFoundError(f"Evidence packet not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Print the prompt that would be sent to a future LLM provider."""
    evidence = load_evidence_packet()
    print(build_llm_prompt(evidence))


# Future provider integration notes:
# - OpenAI: pass SYSTEM_PROMPT as the system/developer instruction and send the
#   compact evidence prompt through the Responses API with structured output
#   constrained to ExecutiveModelReport.model_json_schema().
# - Ollama: call a local chat endpoint with SYSTEM_PROMPT and build_llm_prompt(),
#   request JSON output where supported, then pass the parsed dict through
#   validate_llm_report_dict().
# - In both cases, persist provider name, model name, timestamp, raw evidence
#   packet path, and schema validation status. Never let the provider calculate
#   metrics or override deterministic risk.


if __name__ == "__main__":
    main()
