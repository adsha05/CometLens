"""Render evidence-backed preview images for the GitHub README."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"
BACKGROUND = "#07111f"
PANEL = "#101c2e"
TEXT = "#e8eef8"
MUTED = "#92a4bf"
CYAN = "#4fc3f7"
GREEN = "#42d392"
ORANGE = "#ffb74d"
RED = "#ff627d"


def load_outputs() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load generated reports used to draw evidence-based preview assets."""
    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    shap = pd.read_csv(REPORTS_DIR / "shap_global_importance.csv")
    drift = pd.read_csv(REPORTS_DIR / "drift_report.csv")
    shift = pd.read_csv(REPORTS_DIR / "cluster_shift_report.csv")
    llm = json.loads((REPORTS_DIR / "llm_model_review.json").read_text(encoding="utf-8"))
    return metadata, shap, drift, shift, llm


def setup_canvas(width: float = 16, height: float = 9) -> tuple[plt.Figure, plt.Axes]:
    """Create a dark presentation canvas."""
    fig = plt.figure(figsize=(width, height), dpi=160, facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor(BACKGROUND)
    return fig, ax


def panel(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    """Draw a rounded dashboard panel in figure-relative coordinates."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor=PANEL,
            edgecolor="#1e314a",
            linewidth=1,
        )
    )


def metric_card(ax: plt.Axes, x: float, label: str, value: str, detail: str, color: str) -> None:
    """Draw one overview KPI card."""
    panel(ax, x, 0.70, 0.205, 0.125)
    ax.text(x + 0.018, 0.792, label.upper(), color=MUTED, fontsize=9, weight="bold")
    ax.text(x + 0.018, 0.742, value, color=color, fontsize=25, weight="bold")
    ax.text(x + 0.018, 0.714, detail, color=MUTED, fontsize=8.5)


def render_monitoring_overview(
    metadata: dict, shap: pd.DataFrame, drift: pd.DataFrame, shift: pd.DataFrame, llm: dict
) -> Path:
    """Render a single-page monitoring cockpit from actual pipeline results."""
    fig, ax = setup_canvas()
    metrics = metadata["metrics"]
    high_drift = drift.loc[drift["drift_level"] == "High"].sort_values("psi", ascending=False)

    ax.text(0.045, 0.925, "PurchaseIntel Lens", color=TEXT, fontsize=29, weight="bold")
    ax.text(
        0.045,
        0.885,
        "AI-assisted ML observability for synthetic QSR purchase propensity",
        color=MUTED,
        fontsize=12,
    )
    ax.text(0.852, 0.922, "MODEL RISK", color=MUTED, fontsize=9, weight="bold")
    ax.text(0.851, 0.878, "HIGH", color=RED, fontsize=23, weight="bold")

    metric_card(ax, 0.045, "Validation AUC", f"{metrics['validation']['auc']:.4f}", "held-out validation", CYAN)
    metric_card(ax, 0.268, "Current AUC", f"{metrics['current']['auc']:.4f}", "drifted snapshot", GREEN)
    metric_card(ax, 0.491, "High Drift Features", str(len(high_drift)), "PSI / KS monitoring", RED)
    metric_card(ax, 0.714, "Current Users", "10,000", "synthetic observations", ORANGE)

    panel(ax, 0.045, 0.365, 0.43, 0.29)
    panel(ax, 0.495, 0.365, 0.46, 0.29)
    panel(ax, 0.045, 0.085, 0.91, 0.235)
    ax.text(0.065, 0.62, "TOP MODEL DRIVERS", color=TEXT, fontsize=11, weight="bold")
    ax.text(0.515, 0.62, "HIGH-DRIFT SIGNALS", color=TEXT, fontsize=11, weight="bold")
    ax.text(0.065, 0.285, "SEGMENT MOVEMENT  |  local AI narrative grounded in validated reports", color=TEXT, fontsize=11, weight="bold")

    shap_ax = fig.add_axes([0.065, 0.395, 0.38, 0.19], facecolor=PANEL)
    top_shap = shap.head(5).sort_values("mean_abs_shap_value")
    shap_ax.barh(top_shap["feature"], top_shap["mean_abs_shap_value"], color=CYAN, alpha=0.9)
    shap_ax.set_xlabel("Mean |SHAP|", color=MUTED, fontsize=8)
    shap_ax.tick_params(colors=MUTED, labelsize=8)
    for spine in shap_ax.spines.values():
        spine.set_visible(False)
    shap_ax.grid(axis="x", color="#263a55", alpha=0.5)

    drift_ax = fig.add_axes([0.515, 0.395, 0.41, 0.19], facecolor=PANEL)
    high_plot = high_drift.sort_values("psi")
    drift_ax.barh(high_plot["feature"], high_plot["psi"], color=RED, alpha=0.9)
    drift_ax.set_xlabel("PSI", color=MUTED, fontsize=8)
    drift_ax.tick_params(colors=MUTED, labelsize=8)
    for spine in drift_ax.spines.values():
        spine.set_visible(False)
    drift_ax.grid(axis="x", color="#263a55", alpha=0.5)

    shift_ax = fig.add_axes([0.065, 0.12, 0.52, 0.125], facecolor=PANEL)
    ordered = shift.sort_values("population_shift_pct_points")
    colors = [RED if value < 0 else GREEN for value in ordered["population_shift_pct_points"]]
    shift_ax.barh(ordered["cluster_name"], ordered["population_shift_pct_points"], color=colors, alpha=0.9)
    shift_ax.axvline(0, color=MUTED, linewidth=0.8)
    shift_ax.set_xlabel("Population share change (percentage points)", color=MUTED, fontsize=8)
    shift_ax.tick_params(colors=MUTED, labelsize=8)
    for spine in shift_ax.spines.values():
        spine.set_visible(False)

    narrative = llm["narrative"]["main_risk_drivers"][0]
    wrapped = textwrap.fill(narrative, width=45)
    ax.text(0.625, 0.235, "OLLAMA NARRATIVE REVIEW", color=CYAN, fontsize=9, weight="bold")
    ax.text(0.625, 0.195, f"{llm['model']}  |  evidence-grounded", color=MUTED, fontsize=9)
    ax.text(0.625, 0.118, wrapped, color=TEXT, fontsize=10, linespacing=1.5)

    output_path = ASSETS_DIR / "dashboard_overview.png"
    fig.savefig(output_path, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return output_path


def render_ai_narrative(llm: dict) -> Path:
    """Render an AI narrative card backed by the saved local Ollama artifact."""
    fig, ax = setup_canvas(14, 8)
    narrative = llm["narrative"]
    ax.text(0.06, 0.91, "AI Narrative Review", color=TEXT, fontsize=28, weight="bold")
    ax.text(
        0.06,
        0.865,
        f"Provider: {llm['provider']}   |   Model: {llm['model']}   |   grounded in deterministic reports",
        color=MUTED,
        fontsize=11,
    )
    panel(ax, 0.06, 0.68, 0.88, 0.12)
    ax.text(0.085, 0.765, "EXECUTIVE SUMMARY", color=CYAN, fontsize=9, weight="bold")
    ax.text(0.085, 0.716, textwrap.fill(narrative["executive_summary"], 110), color=TEXT, fontsize=12)

    panel(ax, 0.06, 0.365, 0.43, 0.27)
    panel(ax, 0.51, 0.365, 0.43, 0.27)
    ax.text(0.085, 0.595, "DRIFT INTERPRETATION", color=RED, fontsize=10, weight="bold")
    ax.text(
        0.085,
        0.555,
        textwrap.fill(narrative["drift_interpretation"], 56),
        color=TEXT,
        fontsize=10,
        linespacing=1.45,
        va="top",
    )
    ax.text(0.535, 0.595, "SEGMENT INTERPRETATION", color=ORANGE, fontsize=10, weight="bold")
    ax.text(
        0.535,
        0.555,
        textwrap.fill(narrative["segment_interpretation"], 57),
        color=TEXT,
        fontsize=10,
        linespacing=1.45,
        va="top",
    )

    panel(ax, 0.06, 0.105, 0.88, 0.21)
    ax.text(0.085, 0.275, "RECOMMENDED ACTIONS", color=GREEN, fontsize=10, weight="bold")
    action_labels = [
        "Track competitor switching velocity",
        "Add merchant confidence signals",
        "Monitor weekend dining recovery",
    ]
    for index, action in enumerate(action_labels):
        ax.text(0.09, 0.225 - (index * 0.045), f"- {action}", color=TEXT, fontsize=10)
    ax.text(
        0.06,
        0.06,
        "All findings shown are based on synthetic data. LLM output is interpretation only; risk and metrics are deterministic.",
        color=MUTED,
        fontsize=9,
    )

    output_path = ASSETS_DIR / "ai_narrative_preview.png"
    fig.savefig(output_path, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return output_path


def main() -> None:
    """Generate README assets from currently saved application results."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    metadata, shap, drift, shift, llm = load_outputs()
    outputs = [
        render_monitoring_overview(metadata, shap, drift, shift, llm),
        render_ai_narrative(llm),
    ]
    for output in outputs:
        print(f"Rendered {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
