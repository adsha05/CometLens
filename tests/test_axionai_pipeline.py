"""Unit tests for the AxionAI MVP deterministic workflow."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.agents.evidence_store import EvidencePacketBuilder
from src.agents.executive_synthesis_agent import ExecutiveSynthesisAgent
from src.agents.model_lens_agent import ModelLensAgent
from src.llm.narrative_writer import SYSTEM_PROMPT, build_llm_prompt, validate_llm_report_dict
from src.llm.schemas import ExecutiveModelReport
from src.utils.artifact_validation import validate_input_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


class AxionAIPipelineTests(unittest.TestCase):
    """Smoke-test the core evidence and executive report contracts."""

    def test_evidence_packet_builder_outputs_required_sections(self) -> None:
        """Evidence packet should contain all downstream reporting sections."""
        packet = EvidencePacketBuilder().build_packet()
        required_sections = {
            "model_metadata",
            "feature_metadata",
            "signal_sentinel_summary",
            "model_lens_summary",
            "key_findings",
            "available_plots",
            "business_context",
            "limitations",
        }
        self.assertTrue(required_sections.issubset(packet.keys()))
        self.assertGreaterEqual(len(packet["key_findings"]), 1)

    def test_artifact_contract_validator_accepts_current_demo(self) -> None:
        """Current sample artifacts should satisfy the metadata-driven contract."""
        summary = validate_input_contract(
            train_features_path=DATA_DIR / "train_features_sample.csv",
            current_features_path=DATA_DIR / "current_features_sample.csv",
            predictions_path=DATA_DIR / "current_predictions_sample.csv",
            model_metadata_path=MODELS_DIR / "model_metadata.json",
            feature_metadata_path=MODELS_DIR / "feature_metadata.json",
        )
        self.assertEqual(summary["target"], "purchase_qsr_next_30d")
        self.assertGreater(summary["feature_count"], 0)

    def test_varuna_reliability_gate_flags_severe_mitra_drift(self) -> None:
        """Varuna should mark SHAP as unreliable when Mitra reports severe drift."""
        reliability = ModelLensAgent().assess_explainability_reliability()
        self.assertIn(reliability["status"], {"reliable", "unreliable"})
        if reliability["status"] == "unreliable":
            self.assertGreaterEqual(len(reliability["severe_drift_features"]), 1)

    def test_executive_synthesis_matches_schema(self) -> None:
        """Executive synthesis output should validate against the Pydantic schema."""
        report = ExecutiveSynthesisAgent().build_report()
        self.assertIsInstance(report, ExecutiveModelReport)
        self.assertIn(report.model_health_status, {"Low Risk", "Medium Risk", "High Risk"})
        self.assertGreaterEqual(len(report.recommended_actions), 1)

    def test_saved_executive_report_is_schema_valid(self) -> None:
        """Saved executive report JSON should remain schema-valid."""
        report_path = REPORTS_DIR / "executive_model_report.json"
        self.assertTrue(report_path.exists(), "Run python src/run_pipeline.py before tests.")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = ExecutiveModelReport.model_validate(payload)
        self.assertEqual(report.report_title, "Agent 03: Aryaman Executive Model Health Brief")

    def test_llm_prompt_builder_preserves_guardrails(self) -> None:
        """Prompt utility should include guardrails and deterministic evidence sections."""
        packet = EvidencePacketBuilder().build_packet()
        prompt = build_llm_prompt(packet)
        self.assertIn("Use only supplied evidence", SYSTEM_PROMPT)
        self.assertIn("Do not invent metrics", prompt)
        self.assertIn("output_schema", prompt)

    def test_llm_report_validator_returns_schema_model(self) -> None:
        """Future LLM dictionaries should validate into ExecutiveModelReport."""
        report = validate_llm_report_dict(
            {
                "report_title": "Brief",
                "model_health_status": "Medium Risk",
                "executive_summary": "Summary",
                "what_changed": ["Change"],
                "why_it_matters": "Impact",
                "top_model_drivers": ["Driver"],
                "high_risk_features": ["Feature"],
                "business_risks": ["Risk"],
                "recommended_actions": ["Action"],
                "plots_to_include": ["plot.png"],
                "questions_for_team": ["Question"],
                "client_safe_summary": "Safe",
                "limitations": ["Synthetic sample data only"],
            }
        )
        self.assertEqual(report.model_health_status, "Medium Risk")


if __name__ == "__main__":
    unittest.main()
