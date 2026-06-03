from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures" / "paper_main"
TABLE_DIR = ROOT / "tables" / "paper_main"


TASK_ORDER = [
    "P1_thiram_presence",
    "P2_mg_presence",
    "P3_mba_presence",
    "G1_thiram_molar_grade",
    "G2_mg_molar_grade",
    "G3_mba_molar_grade",
]

TASK_LABEL = {
    "P1_thiram_presence": "P1 Thiram",
    "P2_mg_presence": "P2 MG",
    "P3_mba_presence": "P3 4-MBA",
    "G1_thiram_molar_grade": "G1 Thiram grade",
    "G2_mg_molar_grade": "G2 MG grade",
    "G3_mba_molar_grade": "G3 4-MBA grade",
}

ANALYTE_COLORS = {
    "Thiram": "#0072B2",
    "MG": "#D55E00",
    "MBA": "#009E73",
    "4-MBA": "#009E73",
    "Blank": "#999999",
}

TASK_COLORS = {
    "P1_thiram_presence": ANALYTE_COLORS["Thiram"],
    "G1_thiram_molar_grade": ANALYTE_COLORS["Thiram"],
    "P2_mg_presence": ANALYTE_COLORS["MG"],
    "G2_mg_molar_grade": ANALYTE_COLORS["MG"],
    "P3_mba_presence": ANALYTE_COLORS["MBA"],
    "G3_mba_molar_grade": ANALYTE_COLORS["MBA"],
}

CONDITION_COLORS = {
    "Pure MG": "#333333",
    "MG+MBA": ANALYTE_COLORS["MBA"],
    "MG+Thiram": ANALYTE_COLORS["Thiram"],
    "Ternary": ANALYTE_COLORS["MG"],
}

MODEL_ORDER = [
    "PLS-DA",
    "LDA",
    "SVM",
    "KNN",
    "RF",
    "ExtraTrees",
    "XGBoost",
    "HistGradientBoosting",
    "1D-CNN",
    "1D-ResNet",
    "Spectrum-KAN",
    "KAN-CNN",
    "RamanNet-Lite",
]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax, label: str, x: float = -0.14, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=8,
    )


def save_figure(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("pdf", 600), ("png", 300)):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=dpi)
    plt.close(fig)


def cm_label() -> str:
    return "Raman shift (cm$^{-1}$)"


def add_overlap_regions(ax, alpha: float = 0.10) -> None:
    ax.axvspan(1162, 1188, color=ANALYTE_COLORS["MG"], alpha=alpha, lw=0)
    ax.axvspan(1370, 1405, color=ANALYTE_COLORS["MG"], alpha=alpha, lw=0)


def normalize(y):
    y = y.copy()
    y = y - y.min()
    denom = y.max()
    if denom == 0:
        return y
    return y / denom
