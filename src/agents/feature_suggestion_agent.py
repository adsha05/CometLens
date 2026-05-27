"""Rule-based feature recommendations from monitoring and explanation reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
MEDIUM_HIGH_DRIFT = {"Medium", "High"}
IMPORTANT_FEATURE_COUNT = 10
CLUSTER_SHIFT_THRESHOLD_PCT_POINTS = 1.0


class FeatureSuggestionAgent:
    """Propose candidate features based on detected model monitoring signals."""

    def __init__(
        self,
        drift_report: pd.DataFrame,
        shap_importance: pd.DataFrame,
        cluster_shift_report: pd.DataFrame,
    ) -> None:
        """Initialize recommendation evidence from generated report tables."""
        self.drift_report = drift_report.copy()
        self.shap_importance = shap_importance.copy()
        self.cluster_shift_report = cluster_shift_report.copy()
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate report schemas needed by the recommendation rules."""
        requirements = {
            "drift_report": (self.drift_report, {"feature", "drift_level", "psi", "mean_change_pct"}),
            "shap_importance": (self.shap_importance, {"feature", "mean_abs_shap_value"}),
            "cluster_shift_report": (
                self.cluster_shift_report,
                {"cluster_name", "population_shift_pct_points"},
            ),
        }
        for name, (report, required_columns) in requirements.items():
            missing = required_columns - set(report.columns)
            if missing:
                raise ValueError(f"{name} missing required columns: {sorted(missing)}")

    def _drift_record(self, feature: str) -> pd.Series | None:
        """Return drift evidence for one feature when available."""
        rows = self.drift_report.loc[self.drift_report["feature"] == feature]
        return None if rows.empty else rows.iloc[0]

    def _is_drifted(self, feature: str) -> bool:
        """Return whether a feature has medium or high drift severity."""
        record = self._drift_record(feature)
        return record is not None and record["drift_level"] in MEDIUM_HIGH_DRIFT

    def _important_features(self) -> set[str]:
        """Select important features as the top ten global SHAP drivers."""
        ranked = self.shap_importance.sort_values("mean_abs_shap_value", ascending=False)
        return set(ranked.head(IMPORTANT_FEATURE_COUNT)["feature"])

    def _drift_evidence(self, feature: str) -> str:
        """Format concise supporting drift evidence for a recommendation."""
        record = self._drift_record(feature)
        if record is None:
            return f"{feature}: no drift record"
        return (
            f"{feature}: {record['drift_level']} drift "
            f"(PSI={record['psi']:.4f}, mean change={record['mean_change_pct']:.2f}%)"
        )

    def _shap_evidence(self, feature: str) -> str:
        """Format concise global-importance evidence for a recommendation."""
        rows = self.shap_importance.loc[self.shap_importance["feature"] == feature]
        if rows.empty:
            return f"{feature}: no SHAP record"
        value = float(rows.iloc[0]["mean_abs_shap_value"])
        return f"{feature}: top-{IMPORTANT_FEATURE_COUNT} SHAP feature (mean |SHAP|={value:.4f})"

    @staticmethod
    def _add_recommendation(
        records: list[dict[str, str]],
        suggested_feature: str,
        reason: str,
        linked_evidence: str,
        priority: str,
    ) -> None:
        """Add one unique recommendation row."""
        if any(record["suggested_feature"] == suggested_feature for record in records):
            return
        records.append(
            {
                "suggested_feature": suggested_feature,
                "reason": reason,
                "linked_evidence": linked_evidence,
                "priority": priority,
            }
        )

    def run(self) -> pd.DataFrame:
        """Generate, save, and return rule-based feature recommendations."""
        suggestions: list[dict[str, str]] = []
        important_features = self._important_features()

        if self._is_drifted("merchant_novelty_rate"):
            evidence = self._drift_evidence("merchant_novelty_rate")
            for suggested_feature in (
                "merchant_descriptor_novelty_rate",
                "merchant_confidence_score",
            ):
                self._add_recommendation(
                    suggestions,
                    suggested_feature,
                    "Capture changing merchant behavior and distinguish genuine novelty from uncertain categorization.",
                    evidence,
                    "High",
                )

        competitor_feature = "competitor_qsr_share_90d"
        if self._is_drifted(competitor_feature) or competitor_feature in important_features:
            evidence_parts = [self._drift_evidence(competitor_feature)]
            if competitor_feature in important_features:
                evidence_parts.append(self._shap_evidence(competitor_feature))
            for suggested_feature in (
                "competitor_share_rolling_60d",
                "competitor_switching_velocity",
            ):
                self._add_recommendation(
                    suggestions,
                    suggested_feature,
                    "Measure recent movement toward competing QSR merchants with shorter and directional signals.",
                    "; ".join(evidence_parts),
                    "High",
                )

        weekend_feature = "weekend_dining_frequency"
        if self._is_drifted(weekend_feature) or weekend_feature in important_features:
            evidence_parts = [self._drift_evidence(weekend_feature)]
            if weekend_feature in important_features:
                evidence_parts.append(self._shap_evidence(weekend_feature))
            self._add_recommendation(
                suggestions,
                "weekend_dining_recovery_index",
                "Track whether reduced weekend dining is recovering or persisting for purchase propensity.",
                "; ".join(evidence_parts),
                "High",
            )

        if self._is_drifted("fuel_spend_30d") and self._is_drifted("grocery_spend_30d"):
            self._add_recommendation(
                suggestions,
                "fuel_grocery_recovery_interaction",
                "Model coordinated essential-spend shifts that may alter discretionary QSR purchasing.",
                f"{self._drift_evidence('fuel_spend_30d')}; {self._drift_evidence('grocery_spend_30d')}",
                "Medium",
            )

        shifted_clusters = self.cluster_shift_report.loc[
            self.cluster_shift_report["population_shift_pct_points"].abs()
            >= CLUSTER_SHIFT_THRESHOLD_PCT_POINTS
        ]
        if not shifted_clusters.empty:
            evidence = "; ".join(
                f"{row.cluster_name}: {row.population_shift_pct_points:+.2f} percentage points"
                for row in shifted_clusters.itertuples()
            )
            for suggested_feature in ("cluster_stability_score", "segment_specific_calibration"):
                self._add_recommendation(
                    suggestions,
                    suggested_feature,
                    "Account for material movement in customer segment composition between periods.",
                    evidence,
                    "Medium",
                )

        entropy_feature = "dining_category_entropy"
        if entropy_feature in important_features:
            self._add_recommendation(
                suggestions,
                "category_sequence_density",
                "Add ordering and density information where category diversity is influential to predictions.",
                self._shap_evidence(entropy_feature),
                "Medium",
            )

        suggestions_df = pd.DataFrame(
            suggestions,
            columns=["suggested_feature", "reason", "linked_evidence", "priority"],
        )
        priority_order = pd.CategoricalDtype(["High", "Medium", "Low"], ordered=True)
        if not suggestions_df.empty:
            suggestions_df["priority"] = suggestions_df["priority"].astype(priority_order)
            suggestions_df = suggestions_df.sort_values(["priority", "suggested_feature"]).reset_index(drop=True)
            suggestions_df["priority"] = suggestions_df["priority"].astype(str)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        suggestions_df.to_csv(REPORTS_DIR / "feature_suggestions.csv", index=False)
        return suggestions_df


def main() -> None:
    """Load monitoring reports, generate feature suggestions, and write CSV output."""
    report_paths = {
        "drift_report": REPORTS_DIR / "drift_report.csv",
        "shap_importance": REPORTS_DIR / "shap_global_importance.csv",
        "cluster_shift_report": REPORTS_DIR / "cluster_shift_report.csv",
    }
    missing_paths = [str(path) for path in report_paths.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Required input reports not found: {missing_paths}")

    agent = FeatureSuggestionAgent(
        pd.read_csv(report_paths["drift_report"]),
        pd.read_csv(report_paths["shap_importance"]),
        pd.read_csv(report_paths["cluster_shift_report"]),
    )
    suggestions = agent.run()
    print(suggestions.to_string(index=False))
    print(f"\nSaved feature suggestions to {REPORTS_DIR / 'feature_suggestions.csv'}")


if __name__ == "__main__":
    main()
