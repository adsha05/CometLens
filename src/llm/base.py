"""Provider protocol for structured narrative generation."""

from __future__ import annotations

from typing import Any, Protocol

from src.llm.schemas import NarrativeReport


class LLMProvider(Protocol):
    """Contract for a provider that generates a validated narrative report."""

    provider_name: str
    model_name: str

    def generate_report(self, evidence: dict[str, Any]) -> NarrativeReport:
        """Generate a structured report using only supplied evidence."""
        ...
