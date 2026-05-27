"""Local Ollama provider for grounded narrative reporting."""

from __future__ import annotations

import json
from typing import Any

import httpx

from src.llm.schemas import NarrativeReport

SYSTEM_PROMPT = """You are the narrative review layer for an ML observability demo.
Use only the supplied deterministic evidence.
Do not invent metrics, features, segments, causes, or external business facts.
Do not claim model performance declined if the supplied AUC did not decline.
Identify possible interpretations as hypotheses, not proven causes.
Describe changes only as synthetic behavior shifts; do not mention market conditions.
State the provided risk level and cite exact supplied AUC and drift evidence in the narrative.
In evidence_used, list concrete evidence facts with values, not guardrail instructions.
Always state that all data is synthetic.
Return valid JSON conforming exactly to the supplied schema."""


class OllamaProvider:
    """Generate structured narrative output through a local Ollama server."""

    provider_name = "ollama"

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 180.0,
    ) -> None:
        """Configure a local Ollama chat endpoint and model."""
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate_report(self, evidence: dict[str, Any]) -> NarrativeReport:
        """Call Ollama with structured output and validate the resulting report."""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Create the narrative review from this evidence:\n"
                    + json.dumps(evidence, indent=2),
                },
            ],
            "format": NarrativeReport.model_json_schema(),
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise RuntimeError(
                "Could not reach the local Ollama model. Start Ollama and pull the configured "
                f"model `{self.model_name}` before running the narrative agent. Details: {error}"
            ) from error

        content = response.json().get("message", {}).get("content")
        if not content:
            raise RuntimeError("Ollama returned no narrative content.")
        return NarrativeReport.model_validate_json(content)
