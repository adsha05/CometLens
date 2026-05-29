"""Render README screenshot assets from deterministic VyaAI artifacts."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cometlens-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

COLORS = {
    "dark": "#111827",
    "muted": "#6b7280",
    "green": "#16a34a",
    "amber": "#f59e0b",
    "red": "#dc2626",
    "panel": "#f8fafc",
}


def load_json(path: Path) -> dict:
    """Load a JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_card(path: Path, title: str, lines: list[str], footer: str) -> None:
    """Save a simple screenshot-style card."""
    fig, ax = plt.subplots(figsize=(12, 7.4))
    fig.patch.set_facecolor("white")
    ax.set_axis_off()
    ax.add_patch(
        plt.Rectangle(
            (0.03, 0.06),
            0.94,
            0.88,
            transform=ax.transAxes,
            color=COLORS["panel"],
            ec="#e5e7eb",
            lw=1.5,
        )
    )
    ax.text(
        0.07,
        0.88,
        title,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        color=COLORS["dark"],
        va="top",
    )
    y_position = 0.76
    for line in lines:
        for index, part in enumerate(textwrap.wrap(str(line).replace("**", ""), width=78)):
            prefix = "- " if index == 0 else "  "
            ax.text(
                0.08,
                y_position,
                prefix + part,
                transform=ax.transAxes,
                fontsize=12.2,
                color=COLORS["dark"],
                va="top",
            )
            y_position -= 0.055
        y_position -= 0.01

    ax.text(
        0.07,
        0.10,
        footer,
        transform=ax.transAxes,
        fontsize=11.5,
        color=COLORS["muted"],
        va="bottom",
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_model_health_summary() -> Path:
    """Render a model health summary card."""
    signal = load_json(REPORTS_DIR / "signal_sentinel_output.json")
    executive_report = load_json(REPORTS_DIR / "executive_model_report.json")
    metadata = load_json(MODELS_DIR / "model_metadata.json")
    prediction_summary = signal.get("prediction_drift_summary", {})
    lines = [
        f"Model: {metadata.get('model_name', 'unknown')}",
        f"Health status: {executive_report.get('model_health_status', 'unknown')}",
        f"Mitra overall risk: {signal.get('overall_risk_level', 'unknown')}",
        f"High-drift features: {len(signal.get('high_drift_features', []))}",
        "Prediction-positive vs actual-positive gap: "
        f"{float(prediction_summary.get('prediction_actual_rate_gap', 0.0)):+.3f}",
    ]
    path = ASSETS_DIR / "model_health_summary.png"
    save_card(
        path,
        "Model Health Summary",
        lines,
        "Generated from reports/signal_sentinel_output.json and reports/executive_model_report.json",
    )
    return path


def render_drift_report() -> Path:
    """Render a drift report bar chart."""
    drift = pd.read_csv(REPORTS_DIR / "drift_report.csv").sort_values("psi", ascending=True).tail(8)
    fig, ax = plt.subplots(figsize=(12, 6.8))
    colors = [
        COLORS["red"] if level == "High" else COLORS["amber"] if level == "Medium" else COLORS["green"]
        for level in drift["drift_level"]
    ]
    ax.barh(drift["feature"], drift["psi"], color=colors)
    ax.axvline(0.10, color=COLORS["amber"], linestyle="--", lw=1.5, label="Medium PSI")
    ax.axvline(0.25, color=COLORS["red"], linestyle="--", lw=1.5, label="High PSI")
    ax.set_title("Drift Report: Top Features by PSI", fontsize=18, fontweight="bold")
    ax.set_xlabel("Population Stability Index")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = ASSETS_DIR / "drift_report.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def render_shap_feature_importance() -> Path:
    """Copy generated SHAP feature importance chart into README assets."""
    output_path = ASSETS_DIR / "shap_feature_importance.png"
    shutil.copyfile(FIGURES_DIR / "shap_global_bar.png", output_path)
    return output_path


def render_high_risk_feature_matrix() -> Path:
    """Render the high-risk feature matrix as a table image."""
    lens = load_json(REPORTS_DIR / "model_lens_output.json")
    risk_df = pd.DataFrame(lens.get("high_risk_feature_matrix", []))
    risk_df = risk_df.loc[
        risk_df["combined_risk"].isin(["High", "Medium"]),
        ["feature", "shap_rank", "drift_level", "vif_warning", "combined_risk"],
    ].head(8)

    fig, ax = plt.subplots(figsize=(12, 3.3))
    ax.set_axis_off()
    ax.set_title("High-Risk Feature Matrix", loc="left", pad=18, fontsize=18, fontweight="bold")
    table = ax.table(
        cellText=risk_df.values,
        colLabels=risk_df.columns,
        bbox=[0, 0.06, 1, 0.68],
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#e5e7eb")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(COLORS["dark"])
        elif risk_df.iloc[row - 1]["combined_risk"] == "High":
            cell.set_facecolor("#fee2e2")
        else:
            cell.set_facecolor("#fef3c7")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.08)
    path = ASSETS_DIR / "high_risk_feature_matrix.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def render_executive_report() -> Path:
    """Render the executive report summary as a screenshot-style card."""
    executive_report = load_json(REPORTS_DIR / "executive_model_report.json")
    lines = [
        executive_report.get("executive_summary", ""),
        "Recommended actions:",
        *executive_report.get("recommended_actions", [])[:3],
    ]
    path = ASSETS_DIR / "executive_report.png"
    save_card(
        path,
        "Agent 03: Aryaman Executive Report",
        lines,
        "Generated from reports/executive_model_report.json",
    )
    return path


def main() -> None:
    """Render all README assets."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        render_model_health_summary(),
        render_drift_report(),
        render_shap_feature_importance(),
        render_high_risk_feature_matrix(),
        render_executive_report(),
    ]
    print("Created README screenshot assets:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
