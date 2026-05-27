"""Feature drift analysis using PSI and Kolmogorov-Smirnov statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

TARGET_COLUMN = "purchase_qsr_next_30d"
ID_COLUMN = "user_id"
EPSILON = 1e-6


class DriftAgent:
    """Measure distribution and mean shifts between feature snapshots."""

    def __init__(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: list[str]
    ) -> None:
        """Initialize drift comparisons for a list of shared feature columns."""
        missing_reference = set(feature_cols) - set(reference_df.columns)
        missing_current = set(feature_cols) - set(current_df.columns)
        if missing_reference or missing_current:
            raise ValueError(
                "Feature columns must exist in both datasets. "
                f"Missing from reference: {sorted(missing_reference)}; "
                f"missing from current: {sorted(missing_current)}."
            )
        self.reference_df = reference_df
        self.current_df = current_df
        self.feature_cols = feature_cols

    @staticmethod
    def _clean_numeric(values: pd.Series | np.ndarray) -> np.ndarray:
        """Coerce input to finite numeric observations and remove missing values."""
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        return numeric[np.isfinite(numeric)]

    def calculate_psi(
        self, expected: pd.Series | np.ndarray, actual: pd.Series | np.ndarray, buckets: int = 10
    ) -> float:
        """Calculate population stability index using expected quantile bins."""
        if buckets < 2:
            raise ValueError("PSI requires at least two buckets.")

        expected_values = self._clean_numeric(expected)
        actual_values = self._clean_numeric(actual)
        if expected_values.size == 0 or actual_values.size == 0:
            return np.nan

        if np.all(expected_values == expected_values[0]):
            return 0.0 if np.all(actual_values == expected_values[0]) else float("inf")

        quantiles = np.linspace(0, 1, buckets + 1)
        bin_edges = np.unique(np.quantile(expected_values, quantiles))
        if bin_edges.size < 2:
            return 0.0

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        expected_counts = np.histogram(expected_values, bins=bin_edges)[0]
        actual_counts = np.histogram(actual_values, bins=bin_edges)[0]
        expected_pct = np.clip(expected_counts / expected_values.size, EPSILON, None)
        actual_pct = np.clip(actual_counts / actual_values.size, EPSILON, None)
        return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

    def calculate_ks(
        self, expected: pd.Series | np.ndarray, actual: pd.Series | np.ndarray
    ) -> tuple[float, float]:
        """Calculate the two-sample KS statistic and p-value."""
        expected_values = self._clean_numeric(expected)
        actual_values = self._clean_numeric(actual)
        if expected_values.size == 0 or actual_values.size == 0:
            return np.nan, np.nan
        if np.all(expected_values == expected_values[0]) and np.all(actual_values == actual_values[0]):
            if expected_values[0] == actual_values[0]:
                return 0.0, 1.0
            return 1.0, 0.0

        statistic, pvalue = ks_2samp(expected_values, actual_values, method="asymp")
        return float(statistic), float(pvalue)

    @staticmethod
    def _calculate_mean_change_pct(reference_mean: float, current_mean: float) -> float:
        """Calculate percent mean change, defining zero-baseline behavior explicitly."""
        if pd.isna(reference_mean) or pd.isna(current_mean):
            return np.nan
        if np.isclose(reference_mean, 0.0):
            return 0.0 if np.isclose(current_mean, 0.0) else np.nan
        return float(((current_mean - reference_mean) / abs(reference_mean)) * 100)

    @staticmethod
    def _drift_level(psi: float, ks_pvalue: float) -> str:
        """Assign drift severity from PSI and KS thresholds."""
        if (not pd.isna(psi) and psi >= 0.25) or (
            not pd.isna(ks_pvalue) and ks_pvalue < 0.01
        ):
            return "High"
        if not pd.isna(psi) and psi >= 0.10:
            return "Medium"
        return "Low"

    def run(self) -> pd.DataFrame:
        """Run drift checks and return one result row per feature."""
        records: list[dict[str, float | str]] = []
        for feature in self.feature_cols:
            reference_values = self._clean_numeric(self.reference_df[feature])
            current_values = self._clean_numeric(self.current_df[feature])
            reference_mean = float(np.mean(reference_values)) if reference_values.size else np.nan
            current_mean = float(np.mean(current_values)) if current_values.size else np.nan
            psi = self.calculate_psi(reference_values, current_values)
            ks_statistic, ks_pvalue = self.calculate_ks(reference_values, current_values)
            records.append(
                {
                    "feature": feature,
                    "psi": psi,
                    "ks_statistic": ks_statistic,
                    "ks_pvalue": ks_pvalue,
                    "reference_mean": reference_mean,
                    "current_mean": current_mean,
                    "mean_change_pct": self._calculate_mean_change_pct(
                        reference_mean, current_mean
                    ),
                    "drift_level": self._drift_level(psi, ks_pvalue),
                }
            )
        return pd.DataFrame(records)


def main() -> None:
    """Generate the drift report for the synthetic training and current snapshots."""
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    reports_dir = project_root / "reports"
    train_path = data_dir / "train_features.csv"
    current_path = data_dir / "current_features.csv"
    if not train_path.exists() or not current_path.exists():
        raise FileNotFoundError(
            "Feature CSVs not found. Run `python src/generate_synthetic_data.py` first."
        )

    reference_df = pd.read_csv(train_path)
    current_df = pd.read_csv(current_path)
    feature_cols = [
        column
        for column in reference_df.select_dtypes(include="number").columns
        if column not in {ID_COLUMN, TARGET_COLUMN} and column in current_df.columns
    ]
    drift_report = DriftAgent(reference_df, current_df, feature_cols).run()

    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "drift_report.csv"
    drift_report.to_csv(output_path, index=False)
    print(drift_report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved drift report to {output_path}")


if __name__ == "__main__":
    main()
