"""Build deterministic AxionAI lineage graph specifications from saved evidence."""

from __future__ import annotations

from typing import Any


def _node(
    node_id: str,
    label: str,
    node_type: str,
    *,
    status: str = "Completed",
    risk_level: str = "Low",
    description: str = "",
) -> dict[str, str]:
    """Return one normalized lineage node."""
    return {
        "id": node_id,
        "label": label,
        "type": node_type,
        "status": status,
        "risk_level": risk_level,
        "description": description,
    }


def _edge(source: str, target: str, relationship: str = "flows_to") -> dict[str, str]:
    """Return one normalized directed edge."""
    return {"source": source, "target": target, "relationship": relationship}


def build_default_model_lineage_graph(
    evidence_packet: dict[str, Any],
    mitra_output: dict[str, Any],
    varuna_output: dict[str, Any],
) -> dict[str, Any]:
    """Build a run-specific model intelligence graph using only saved outputs."""
    metadata = evidence_packet.get("model_metadata", {})
    mitra_risk = str(mitra_output.get("overall_risk_level", "Unknown"))
    prediction_risk = str(
        evidence_packet.get("prediction_drift_summary", {}).get("prediction_drift_level", "Unknown")
    )
    high_drift_count = len(mitra_output.get("high_drift_features", []))
    feature_pipeline_risk = "High" if high_drift_count else "Low"
    high_risk_count = len(evidence_packet.get("high_risk_features", []))
    varuna_risk = "High" if high_risk_count else "Low"

    nodes = [
        _node("raw_data", "Synthetic Feature Tables", "data_source", description="Reference and current windows"),
        _node("data_quality", "Data Quality Gate", "processing", risk_level=mitra_risk),
        _node("feature_pipeline", "Feature Pipeline", "processing", risk_level=feature_pipeline_risk),
        _node("feature_store", "Feature Store", "storage", risk_level=feature_pipeline_risk),
        _node("model_scores", str(metadata.get("model_name", "Reviewed Model")), "model"),
        _node("prediction_logs", "Prediction Logs", "data_source", risk_level=prediction_risk),
        _node("mitra", "Agent 01: Mitra", "agent", risk_level=mitra_risk, description="Signal monitoring"),
        _node("varuna", "Agent 02: Varuna", "agent", risk_level=varuna_risk, description="Model diagnostics"),
        _node("evidence_store", "Verified Evidence Store", "storage"),
        _node("vishwakarma", "Agent 05: Vishwakarma", "agent", description="Visual intelligence"),
        _node("aryaman", "Agent 03: Aryaman", "agent", description="Executive synthesis"),
        _node("executive_report", "Executive Model Brief", "output"),
        _node("feedback", "Stakeholder Feedback", "output", status="Available"),
    ]
    edges = [
        _edge("raw_data", "data_quality"),
        _edge("data_quality", "feature_pipeline"),
        _edge("feature_pipeline", "feature_store"),
        _edge("feature_store", "model_scores"),
        _edge("model_scores", "prediction_logs"),
        _edge("feature_store", "mitra"),
        _edge("prediction_logs", "mitra"),
        _edge("feature_store", "varuna"),
        _edge("mitra", "varuna", "reliability_gate"),
        _edge("mitra", "evidence_store"),
        _edge("varuna", "evidence_store"),
        _edge("evidence_store", "vishwakarma"),
        _edge("evidence_store", "aryaman"),
        _edge("vishwakarma", "aryaman", "visual_manifest"),
        _edge("aryaman", "executive_report"),
        _edge("executive_report", "feedback"),
    ]
    return {
        "graph_name": "AxionAI Model Intelligence Lineage",
        "run_id": evidence_packet.get("run_id", "unknown"),
        "config_version": evidence_packet.get("config_version", "unknown"),
        "model_name": metadata.get("model_name", "unknown"),
        "nodes": nodes,
        "edges": edges,
        "source_files": evidence_packet.get("source_files", {}),
    }
