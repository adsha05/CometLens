"""Customer segment clustering and distribution shift reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
TARGET_COLUMN = "purchase_qsr_next_30d"
ID_COLUMN = "user_id"
PREDICTION_COLUMN = "y_pred_proba"
RANDOM_SEED = 42


class ClusterAgent:
    """Cluster reference customers and compare segment movement over time."""

    def __init__(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: list[str],
        n_clusters: int = 4,
    ) -> None:
        """Initialize segment analysis against shared numeric feature columns."""
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        missing_reference = set(feature_cols) - set(reference_df.columns)
        missing_current = set(feature_cols) - set(current_df.columns)
        if missing_reference or missing_current:
            raise ValueError(
                "Feature columns must exist in both datasets. "
                f"Missing from reference: {sorted(missing_reference)}; "
                f"missing from current: {sorted(missing_current)}."
            )
        if len(reference_df) < n_clusters:
            raise ValueError("Reference dataset must have at least n_clusters rows.")

        self.reference_df = reference_df.copy()
        self.current_df = current_df.copy()
        self.feature_cols = feature_cols
        self.n_clusters = n_clusters
        self.scaler: StandardScaler | None = None
        self.model: KMeans | None = None
        self.cluster_names: dict[int, str] = {}

    @staticmethod
    def _prepare_features(data: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Convert clustering predictors to numeric values with median imputation."""
        features = data.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
        medians = features.median().fillna(0.0)
        return features.fillna(medians)

    def fit_clusters(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit clusters on the reference population and label both snapshots."""
        reference_features = self._prepare_features(self.reference_df, self.feature_cols)
        current_features = self._prepare_features(self.current_df, self.feature_cols)

        self.scaler = StandardScaler()
        reference_scaled = self.scaler.fit_transform(reference_features)
        current_scaled = self.scaler.transform(current_features)
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=RANDOM_SEED,
            n_init=20,
        )
        self.reference_df["cluster"] = self.model.fit_predict(reference_scaled)
        self.current_df["cluster"] = self.model.predict(current_scaled)
        self.cluster_names = self._name_clusters(reference_features)
        self.reference_df["cluster_name"] = self.reference_df["cluster"].map(self.cluster_names)
        self.current_df["cluster_name"] = self.current_df["cluster"].map(self.cluster_names)
        return self.reference_df, self.current_df

    def _name_clusters(self, reference_features: pd.DataFrame) -> dict[int, str]:
        """Assign one readable label to each cluster from characteristic behavior groups."""
        means = reference_features.assign(cluster=self.reference_df["cluster"]).groupby("cluster").mean()
        scaled_means = (means - reference_features.mean()) / reference_features.std(ddof=0).replace(0, 1)
        rules = {
            "Loyal QSR Buyers": ["qsr_txn_count_30d"],
            "Competitor Switchers": ["competitor_qsr_share_90d"],
            "Value-Seeking Routine Shoppers": ["grocery_spend_30d", "fuel_spend_30d"],
            "Unstable/New Behavior Segment": ["merchant_novelty_rate", "dining_category_entropy"],
        }
        scores = pd.DataFrame(
            {
                name: scaled_means[[column for column in columns if column in scaled_means]].mean(axis=1)
                for name, columns in rules.items()
            }
        )

        # Assign each business label once by repeatedly selecting its strongest remaining match.
        assigned: dict[int, str] = {}
        remaining_clusters = set(scores.index.tolist())
        remaining_names = set(scores.columns.tolist())
        while remaining_clusters and remaining_names:
            candidates = [
                (float(scores.loc[cluster, name]), int(cluster), name)
                for cluster in remaining_clusters
                for name in remaining_names
            ]
            _, cluster, name = max(candidates, key=lambda item: item[0])
            assigned[cluster] = name
            remaining_clusters.remove(cluster)
            remaining_names.remove(name)
        return assigned

    @staticmethod
    def _merge_prediction_scores(data: pd.DataFrame, predictions: pd.DataFrame | None) -> pd.DataFrame:
        """Merge optional model probabilities onto cluster-assigned rows."""
        if predictions is None or ID_COLUMN not in predictions or PREDICTION_COLUMN not in predictions:
            return data.copy()
        score_data = predictions[[ID_COLUMN, PREDICTION_COLUMN]].drop_duplicates(ID_COLUMN)
        return data.merge(score_data, on=ID_COLUMN, how="left")

    def _cluster_profile(
        self, data: pd.DataFrame, predictions: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Build per-cluster behavior, target, and optional prediction summaries."""
        scored_data = self._merge_prediction_scores(data, predictions)
        features = self._prepare_features(scored_data, self.feature_cols)
        overall_mean = features.mean()
        overall_std = features.std(ddof=0).replace(0, 1)
        records: list[dict[str, float | int | str]] = []
        for cluster in sorted(scored_data["cluster"].unique()):
            members = scored_data.loc[scored_data["cluster"] == cluster]
            member_features = features.loc[members.index]
            mean_values = member_features.mean()
            distinguishing = ((mean_values - overall_mean) / overall_std).abs().nlargest(5).index
            feature_summary = "; ".join(
                f"{feature}={mean_values[feature]:.4f}" for feature in distinguishing
            )
            average_score = (
                float(members[PREDICTION_COLUMN].mean())
                if PREDICTION_COLUMN in members.columns
                else np.nan
            )
            records.append(
                {
                    "cluster": int(cluster),
                    "cluster_name": self.cluster_names[int(cluster)],
                    "size": int(len(members)),
                    "size_pct": float(len(members) / len(scored_data) * 100),
                    "target_rate": float(members[TARGET_COLUMN].mean())
                    if TARGET_COLUMN in members
                    else np.nan,
                    "avg_prediction_score": average_score,
                    "top_distinguishing_features": feature_summary,
                }
            )
        return pd.DataFrame(records)

    def profile_clusters(
        self,
        reference_predictions: pd.DataFrame | None = None,
        current_predictions: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return profiles for reference and current assigned clusters."""
        if "cluster" not in self.reference_df or "cluster" not in self.current_df:
            self.fit_clusters()
        return (
            self._cluster_profile(self.reference_df, reference_predictions),
            self._cluster_profile(self.current_df, current_predictions),
        )

    def compare_cluster_distribution(self) -> pd.DataFrame:
        """Compare cluster population shares between reference and current snapshots."""
        if "cluster" not in self.reference_df or "cluster" not in self.current_df:
            self.fit_clusters()
        reference_counts = self.reference_df["cluster"].value_counts().sort_index()
        current_counts = self.current_df["cluster"].value_counts().sort_index()
        records: list[dict[str, float | int | str]] = []
        for cluster in range(self.n_clusters):
            reference_size = int(reference_counts.get(cluster, 0))
            current_size = int(current_counts.get(cluster, 0))
            reference_pct = reference_size / len(self.reference_df) * 100
            current_pct = current_size / len(self.current_df) * 100
            records.append(
                {
                    "cluster": cluster,
                    "cluster_name": self.cluster_names[cluster],
                    "reference_size": reference_size,
                    "current_size": current_size,
                    "reference_pct": reference_pct,
                    "current_pct": current_pct,
                    "population_shift_pct_points": current_pct - reference_pct,
                }
            )
        return pd.DataFrame(records)

    def run(
        self,
        reference_predictions: pd.DataFrame | None = None,
        current_predictions: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run clustering analysis and save reference, current, and shift reports."""
        self.fit_clusters()
        reference_profile, current_profile = self.profile_clusters(
            reference_predictions, current_predictions
        )
        cluster_shift = self.compare_cluster_distribution()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        reference_profile.to_csv(REPORTS_DIR / "reference_cluster_profile.csv", index=False)
        current_profile.to_csv(REPORTS_DIR / "current_cluster_profile.csv", index=False)
        cluster_shift.to_csv(REPORTS_DIR / "cluster_shift_report.csv", index=False)
        return reference_profile, current_profile, cluster_shift


def _load_optional_predictions(path: Path) -> pd.DataFrame | None:
    """Read a prediction artifact when model training has already been run."""
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    """Cluster synthetic customers and write segment profile reports."""
    train_path = DATA_DIR / "train_features.csv"
    current_path = DATA_DIR / "current_features.csv"
    metadata_path = PROJECT_ROOT / "models" / "model_metadata.json"
    if not train_path.exists() or not current_path.exists():
        raise FileNotFoundError(
            "Feature CSVs not found. Run `python src/generate_synthetic_data.py` first."
        )

    reference_df = pd.read_csv(train_path)
    current_df = pd.read_csv(current_path)
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            feature_cols = json.load(metadata_file)["feature_list"]
    else:
        feature_cols = [
            column
            for column in reference_df.select_dtypes(include="number").columns
            if column not in {ID_COLUMN, TARGET_COLUMN}
        ]

    agent = ClusterAgent(reference_df, current_df, feature_cols)
    reference_profile, current_profile, cluster_shift = agent.run(
        _load_optional_predictions(DATA_DIR / "train_predictions.csv"),
        _load_optional_predictions(DATA_DIR / "current_predictions.csv"),
    )
    print("Reference cluster profiles:")
    print(reference_profile.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nCurrent cluster profiles:")
    print(current_profile.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nCluster distribution shift:")
    print(cluster_shift.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved cluster reports to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
