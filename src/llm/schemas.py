"""Pydantic schemas for AxionAI executive narrative outputs."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ExecutiveModelReport(BaseModel):
    """Structured consulting-style model intelligence brief."""

    config_version: str = Field(default="unknown", description="Calibration config version used for source evidence.")
    source_files: dict[str, object] = Field(default_factory=dict, description="Source artifact paths used for the report.")
    report_title: str = Field(description="Title of the model intelligence report.")
    model_health_status: str = Field(description="Overall model health or risk status.")
    executive_summary: str = Field(description="Concise executive summary.")
    what_changed: list[str] = Field(description="Evidence-backed changes observed in the artifacts.")
    why_it_matters: str = Field(description="Business implication of the observed changes.")
    top_model_drivers: list[str] = Field(description="Most important model drivers from deterministic evidence.")
    high_risk_features: list[str] = Field(description="Features with high drift, high VIF, or combined model risk.")
    business_risks: list[str] = Field(description="Business risks implied by the evidence.")
    recommended_actions: list[str] = Field(description="Recommended next actions for business and data science teams.")
    plots_to_include: list[str] = Field(description="Relevant plot artifact paths or plot names.")
    questions_for_team: list[str] = Field(description="Follow-up questions for model, data, or business owners.")
    client_safe_summary: str = Field(description="Client-safe summary without unsupported claims.")
    limitations: list[str] = Field(description="Limitations and data-use boundaries.")


def _bullet_section(title: str, items: list[str]) -> list[str]:
    """Render a Markdown bullet section."""
    lines = [f"## {title}", ""]
    lines.extend(f"- {item}" for item in items) if items else lines.append("- None flagged")
    lines.append("")
    return lines


def executive_report_to_markdown(report: ExecutiveModelReport) -> str:
    """Render an ExecutiveModelReport as a concise consulting-style brief."""
    lines = [
        f"# {report.report_title}",
        "",
        f"**Model health status:** {report.model_health_status}",
        f"**Config version:** {report.config_version}",
        "",
        "## Executive Summary",
        "",
        report.executive_summary,
        "",
    ]
    lines.extend(_bullet_section("What Changed", report.what_changed))
    lines.extend(["## Why It Matters", "", report.why_it_matters, ""])
    lines.extend(_bullet_section("Top Model Drivers", report.top_model_drivers))
    lines.extend(_bullet_section("High-Risk Features", report.high_risk_features))
    lines.extend(_bullet_section("Business Risks", report.business_risks))
    lines.extend(_bullet_section("Recommended Actions", report.recommended_actions))
    lines.extend(_bullet_section("Plots To Include", report.plots_to_include))
    lines.extend(_bullet_section("Questions For The Team", report.questions_for_team))
    lines.extend(["## Client-Safe Summary", "", report.client_safe_summary, ""])
    lines.extend(_bullet_section("Limitations", report.limitations))
    if report.source_files:
        lines.extend(["## Source Files", ""])
        for name, path in report.source_files.items():
            lines.append(f"- `{name}`: `{path}`")
        lines.append("")
    return "\n".join(lines)


# Backward-compatible aliases for earlier local provider scaffolding.
NarrativeReport = ExecutiveModelReport
narrative_report_to_markdown = executive_report_to_markdown


class NarrativeArtifact(BaseModel):
    """Executive report plus provider provenance for future LLM integrations."""

    provider: str
    model: str
    generated_at_utc: str
    evidence_source: str
    narrative: ExecutiveModelReport

    @classmethod
    def create(
        cls,
        provider: str,
        model: str,
        narrative: ExecutiveModelReport,
    ) -> "NarrativeArtifact":
        """Create a narrative artifact with generation metadata."""
        return cls(
            provider=provider,
            model=model,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            evidence_source="verified AxionAI evidence artifacts",
            narrative=narrative,
        )
