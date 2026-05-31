"""Render AxionAI graph specifications as dependency-free SVG files."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

NODE_WIDTH = 170
NODE_HEIGHT = 58
POSITIONS = {
    "raw_data": (30, 100),
    "data_quality": (235, 100),
    "feature_pipeline": (440, 100),
    "feature_store": (645, 100),
    "model_scores": (850, 100),
    "prediction_logs": (1055, 100),
    "mitra": (235, 280),
    "varuna": (440, 280),
    "evidence_store": (645, 280),
    "vishwakarma": (850, 240),
    "aryaman": (850, 350),
    "executive_report": (1055, 350),
    "feedback": (1055, 465),
}
RISK_COLORS = {
    "High": ("#fee2e2", "#dc2626"),
    "Medium": ("#ffedd5", "#f97316"),
    "Low": ("#dcfce7", "#16a34a"),
    "Unknown": ("#f3f4f6", "#6b7280"),
}


def _node_colors(node: dict[str, Any]) -> tuple[str, str]:
    """Return fill and border colors for a node risk state."""
    risk_level = str(node.get("risk_level", "Unknown"))
    return RISK_COLORS.get(risk_level, RISK_COLORS["Unknown"])


def _center(node_id: str) -> tuple[float, float]:
    """Return the center point for a positioned node."""
    x, y = POSITIONS[node_id]
    return x + NODE_WIDTH / 2, y + NODE_HEIGHT / 2


def render_lineage_svg(graph_spec: dict[str, Any], output_path: str | Path) -> Path:
    """Render a readable lineage SVG and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = [node for node in graph_spec.get("nodes", []) if node.get("id") in POSITIONS]
    node_ids = {str(node["id"]) for node in nodes}
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="570" viewBox="0 0 1280 570">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>',
        "</marker>",
        "</defs>",
        '<rect width="1280" height="570" fill="#f8fafc"/>',
        '<text x="30" y="40" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">'
        f"{escape(str(graph_spec.get('graph_name', 'Model Lineage')))}</text>",
        '<text x="30" y="67" font-family="Arial" font-size="14" fill="#475569">'
        f"Run {escape(str(graph_spec.get('run_id', 'unknown')))} | "
        f"Config {escape(str(graph_spec.get('config_version', 'unknown')))}</text>",
    ]
    for edge in graph_spec.get("edges", []):
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source not in node_ids or target not in node_ids:
            continue
        x1, y1 = _center(source)
        x2, y2 = _center(target)
        lines.append(
            f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
            'stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>'
        )
    for node in nodes:
        node_id = str(node["id"])
        x, y = POSITIONS[node_id]
        fill, border = _node_colors(node)
        label = escape(str(node.get("label", node_id)))
        risk = escape(str(node.get("risk_level", "Unknown")))
        status = escape(str(node.get("status", "Unknown")))
        lines.extend(
            [
                f'<rect x="{x}" y="{y}" width="{NODE_WIDTH}" height="{NODE_HEIGHT}" rx="8" '
                f'fill="{fill}" stroke="{border}" stroke-width="2"/>',
                f'<text x="{x + 10}" y="{y + 23}" font-family="Arial" font-size="13" '
                f'font-weight="bold" fill="#0f172a">{label}</text>',
                f'<text x="{x + 10}" y="{y + 44}" font-family="Arial" font-size="11" '
                f'fill="#475569">Risk: {risk} | {status}</text>',
            ]
        )
    lines.extend(
        [
            '<text x="30" y="545" font-family="Arial" font-size="12" fill="#64748b">'
            "Generated deterministically from saved AxionAI evidence. Red nodes require review.</text>",
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
