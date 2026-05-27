"""Optional LLM narrative agent grounded in deterministic monitoring artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.llm_context_builder import LLMContextBuilder
from src.llm.ollama_provider import OllamaProvider
from src.llm.schemas import NarrativeArtifact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


class NarrativeAgent:
    """Generate an AI narrative while preserving deterministic source reports."""

    def __init__(self, provider: OllamaProvider) -> None:
        """Configure the narrative provider."""
        self.provider = provider
        self.context_builder = LLMContextBuilder()

    @staticmethod
    def validate_grounding(evidence: dict, artifact: NarrativeArtifact) -> None:
        """Reject obvious narrative claims not supported by the deterministic evidence."""
        text = artifact.narrative.model_dump_json().lower()
        if "synthetic" not in text:
            raise ValueError("AI narrative must explicitly disclose that all data is synthetic.")
        if "market condition" in text:
            raise ValueError("AI narrative introduced market conditions without supporting evidence.")

        high_drift_features = {
            row["feature"].lower() for row in evidence["high_drift_features"]
        }
        if not all(feature in text for feature in high_drift_features):
            raise ValueError("AI narrative omitted one or more high-drift evidence features.")
        if evidence["model"]["risk_level"].lower() not in text:
            raise ValueError("AI narrative omitted the deterministic model risk level.")

        validation_auc = evidence["performance"]["validation"]["auc"]
        current_auc = evidence["performance"]["current"]["auc"]
        if current_auc >= validation_auc and "auc declined" in text:
            raise ValueError("AI narrative contradicts the supplied AUC direction.")

    @staticmethod
    def render_markdown(artifact: NarrativeArtifact) -> str:
        """Render validated narrative output for the dashboard."""
        narrative = artifact.narrative
        lines = [
            "# AI Narrative Review",
            "",
            f"**Provider:** `{artifact.provider}`  ",
            f"**Model:** `{artifact.model}`  ",
            f"**Generated UTC:** `{artifact.generated_at_utc}`",
            "",
            "> AI-generated interpretation grounded in deterministic synthetic-data monitoring artifacts.",
            "",
            "## Executive Summary",
            "",
            narrative.executive_summary,
            "",
            "## Main Risk Drivers",
            "",
        ]
        lines.extend(f"- {driver}" for driver in narrative.main_risk_drivers)
        lines.extend(
            [
                "",
                "## Performance Interpretation",
                "",
                narrative.performance_interpretation,
                "",
                "## Drift Interpretation",
                "",
                narrative.drift_interpretation,
                "",
                "## Segment Interpretation",
                "",
                narrative.segment_interpretation,
                "",
                "## Recommended Actions",
                "",
            ]
        )
        lines.extend(f"1. {action}" for action in narrative.recommended_actions)
        lines.extend(["", "## Questions For Analyst", ""])
        lines.extend(f"- {question}" for question in narrative.questions_for_analyst)
        lines.extend(["", "## Evidence Used", ""])
        lines.extend(f"- {item}" for item in narrative.evidence_used)
        lines.extend(["", "## Data Disclosure", "", narrative.synthetic_data_disclosure, ""])
        return "\n".join(lines)

    def run(self) -> tuple[Path, Path]:
        """Create and save JSON and Markdown narrative outputs."""
        evidence_path = self.context_builder.save()
        with evidence_path.open("r", encoding="utf-8") as evidence_file:
            evidence = json.load(evidence_file)
        narrative = self.provider.generate_report(evidence)
        artifact = NarrativeArtifact.create(
            self.provider.provider_name,
            self.provider.model_name,
            narrative,
        )
        self.validate_grounding(evidence, artifact)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = REPORTS_DIR / "llm_model_review.json"
        markdown_path = REPORTS_DIR / "llm_model_review.md"
        json_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        markdown_path.write_text(self.render_markdown(artifact), encoding="utf-8")
        return json_path, markdown_path


def build_provider() -> OllamaProvider:
    """Build the configured local provider for the first demo phase."""
    load_dotenv(PROJECT_ROOT / ".env")
    provider_name = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider_name != "ollama":
        raise ValueError(
            "Only the local `ollama` provider is implemented in this phase. "
            "Set LLM_PROVIDER=ollama."
        )
    model_name = os.getenv("LLM_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    return OllamaProvider(model_name=model_name, base_url=base_url)


def main() -> None:
    """Run the locally configured LLM narrative generator."""
    provider = build_provider()
    json_path, markdown_path = NarrativeAgent(provider).run()
    print(f"Generated AI narrative JSON: {json_path}")
    print(f"Generated AI narrative Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
