"""Validated schemas for LLM-generated narrative review output."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class NarrativeReport(BaseModel):
    """Grounded narrative output rendered in the demo dashboard."""

    executive_summary: str = Field(description="Short executive summary grounded in supplied evidence.")
    main_risk_drivers: list[str] = Field(description="Most important monitored risks from evidence.")
    performance_interpretation: str = Field(description="Interpretation of provided model metrics only.")
    drift_interpretation: str = Field(description="Interpretation of supplied feature drift only.")
    segment_interpretation: str = Field(description="Interpretation of supplied cluster shifts only.")
    recommended_actions: list[str] = Field(description="Evidence-based actions for model monitoring.")
    questions_for_analyst: list[str] = Field(description="Questions that need analyst review or follow-up.")
    evidence_used: list[str] = Field(description="Evidence items used for the narrative.")
    synthetic_data_disclosure: str = Field(description="Statement that all observations are synthetic.")


class NarrativeArtifact(BaseModel):
    """Saved narrative review with provider provenance."""

    provider: str
    model: str
    generated_at_utc: str
    evidence_source: str
    narrative: NarrativeReport

    @classmethod
    def create(cls, provider: str, model: str, narrative: NarrativeReport) -> "NarrativeArtifact":
        """Create an artifact with generation metadata."""
        return cls(
            provider=provider,
            model=model,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            evidence_source="deterministic PurchaseIntel Lens monitoring artifacts",
            narrative=narrative,
        )
