"""Business-ready Markdown rendering for Agent 03: Aryaman."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _bullets(items: list[str]) -> str:
    """Render list items as Markdown bullets."""
    return "\n".join(f"- {item}" for item in items) if items else "- None flagged"


def render_executive_report_markdown(evidence_packet: dict[str, Any], aryaman_output: dict[str, Any]) -> str:
    """Render a concise deterministic executive model-health report."""
    metadata = evidence_packet.get("model_metadata", {})
    source_files = evidence_packet.get("source_files", {})
    appendix = [
        f"`{name}`: `{path}`"
        for name, path in source_files.items()
    ]
    recommended_visuals = aryaman_output.get("recommended_report_visuals", [])
    return "\n".join(
        [
            "# Agent 03: Aryaman Executive Model Health Brief",
            "",
            "## 1. Executive Summary",
            "",
            aryaman_output["executive_summary"],
            "",
            "## 2. Model Health Status",
            "",
            f"**{aryaman_output['model_health_status']}**",
            "",
            "## 3. What Changed",
            "",
            _bullets(aryaman_output["key_findings_used"]),
            "",
            "## 4. Why It Matters",
            "",
            aryaman_output["why_it_matters"],
            "",
            "## 5. Top Model Drivers",
            "",
            _bullets(aryaman_output["top_model_drivers"]),
            "",
            "## 6. High-Risk Features",
            "",
            _bullets(aryaman_output["high_risk_features"]),
            "",
            "## 7. Business Risks",
            "",
            _bullets(aryaman_output["business_risks"]),
            "",
            "## 8. Recommended Actions",
            "",
            _bullets(aryaman_output["recommended_actions"]),
            "",
            "## 9. Evidence Appendix",
            "",
            f"- Model: `{metadata.get('model_name', 'unknown')}`",
            f"- Config version: `{evidence_packet.get('config_version', 'unknown')}`",
            f"- Run ID: `{evidence_packet.get('run_id', 'unknown')}`",
            *_bullets(appendix).splitlines(),
            "",
            "### Recommended Visuals",
            "",
            _bullets(recommended_visuals),
            "",
            "## 10. Limitations",
            "",
            _bullets(evidence_packet.get("limitations", [])),
            "",
        ]
    )


def save_markdown_report(markdown: str, path: str | Path) -> Path:
    """Save Markdown report text and return the output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path
