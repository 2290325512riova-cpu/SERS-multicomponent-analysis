from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image

from paper_style import (
    ANALYTE_COLORS,
    CONDITION_COLORS,
    FIG_DIR,
    MODEL_ORDER,
    ROOT,
    TASK_COLORS,
    TASK_LABEL,
    TASK_ORDER,
    add_overlap_regions,
    apply_style,
    cm_label,
    despine,
    normalize,
    panel_label,
    save_figure,
)


PROC = ROOT / "data" / "pure63_mainline" / "processed"
SPLIT = ROOT / "data" / "pure63_mainline" / "splits" / "cv_split_random_5fold.csv"
PAPER = ROOT / "data" / "pure63_mainline" / "models" / "paper_main"
ORIGIN_FIG_DIR = ROOT / "figures" / "paper_main_origin"

PEAKS = {
    "Thiram": [(560, "S-S"), (930, "C-S"), (1145, "C-N"), (1380, "CH3"), (1510, "C=N")],
    "MG": [(908, "ring"), (1172, "C-H*"), (1220, "C-H"), (1394, "C-C*"), (1616, "C=C")],
    "MBA": [(1078, "ring"), (1180, "C-H"), (1490, "C-C"), (1590, "C=C")],
}


def load_processed(variant: str = "p4"):
    x = np.load(PROC / f"X_{variant}.npy")
    wn = np.load(PROC / "wavenumber.npy")
    meta = pd.read_csv(SPLIT)
    if len(meta) != x.shape[0]:
        raise ValueError(f"Row mismatch: X_{variant}={x.shape[0]}, split={len(meta)}")
    return x, wn, meta


def pure_mask(meta: pd.DataFrame, analyte: str, level: int | None = None):
    if analyte == "Thiram":
        mask = (meta.has_thiram == 1) & (meta.has_mg == 0) & (meta.has_mba == 0)
        if level is not None:
            mask &= meta.c_thiram == level
    elif analyte == "MG":
        mask = (meta.has_mg == 1) & (meta.has_thiram == 0) & (meta.has_mba == 0)
        if level is not None:
            mask &= meta.c_mg == level
    elif analyte == "MBA":
        mask = (meta.has_mba == 1) & (meta.has_thiram == 0) & (meta.has_mg == 0)
        if level is not None:
            mask &= meta.c_mba == level
    else:
        raise ValueError(analyte)
    return mask.values


def mean_sd(x: np.ndarray, mask) -> tuple[np.ndarray, np.ndarray]:
    sub = x[mask]
    return sub.mean(axis=0), sub.std(axis=0)


def trimmed_image(path: Path, threshold: int = 248, pad: int = 18) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    mask = np.any(arr < threshold, axis=2)
    if not mask.any():
        return arr
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad, arr.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, arr.shape[1])
    return arr[y0:y1, x0:x1]


def crop_fraction(arr: np.ndarray, crop: tuple[float, float, float, float]) -> np.ndarray:
    top, bottom, left, right = crop
    h, w = arr.shape[:2]
    y0 = int(round(top * h))
    y1 = int(round((1.0 - bottom) * h))
    x0 = int(round(left * w))
    x1 = int(round((1.0 - right) * w))
    if y1 <= y0 or x1 <= x0:
        return arr
    return arr[y0:y1, x0:x1]


def origin_x_fraction(
    value: float,
    x_min: float,
    x_max: float,
    plot_left: float,
    plot_right: float,
    crop: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> float:
    raw = plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)
    _, _, crop_left, crop_right = crop
    return (raw - crop_left) / (1.0 - crop_left - crop_right)


def show_origin_image(ax, filename: str, crop: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> bool:
    path = ORIGIN_FIG_DIR / filename
    if not path.exists():
        return False
    ax.clear()
    ax.imshow(crop_fraction(trimmed_image(path), crop), aspect="auto")
    ax.set_axis_off()
    return True


def peak_height(x: np.ndarray, wn: np.ndarray, mask, center: float, half: float = 6.0):
    band = (wn >= center - half) & (wn <= center + half)
    return x[np.ix_(np.asarray(mask, dtype=bool), band)].max(axis=1)


def mg_ratio_table(x: np.ndarray, wn: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    peaks = [1172, 1220, 1394, 1616]
    refs: dict[int, dict[int, float]] = {}
    for lvl in [4, 5, 6]:
        mask = (meta.has_mg == 1) & (meta.has_thiram == 0) & (meta.has_mba == 0) & (meta.c_mg == lvl)
        refs[lvl] = {p: float(np.median(peak_height(x, wn, mask.values, p))) for p in peaks}

    conditions = {
        "MG+MBA": (meta.has_mg == 1) & (meta.has_thiram == 0) & (meta.has_mba == 1),
        "MG+Thiram": (meta.has_mg == 1) & (meta.has_thiram == 1) & (meta.has_mba == 0),
        "Ternary": (meta.has_mg == 1) & (meta.has_thiram == 1) & (meta.has_mba == 1),
    }
    rows = []
    for cond, base_mask in conditions.items():
        for lvl in [4, 5, 6]:
            mask = base_mask & (meta.c_mg == lvl)
            if not mask.any():
                continue
            for p in peaks:
                values = peak_height(x, wn, mask.values, p) / refs[lvl][p]
                for value in values:
                    rows.append({"condition": cond, "c_mg": lvl, "peak": p, "ratio": float(value)})
    return pd.DataFrame(rows)


def plot_fig2_quality() -> None:
    x, wn, meta = load_processed("p1")
    fig = plt.figure(figsize=(7.2, 3.0), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    offset = 0.0
    for analyte, level, color in [
        ("Thiram", 4, ANALYTE_COLORS["Thiram"]),
        ("MG", 4, ANALYTE_COLORS["MG"]),
        ("MBA", 4, ANALYTE_COLORS["MBA"]),
    ]:
        mask = pure_mask(meta, analyte, level)
        idx = np.where(mask)[0][:10]
        for i in idx:
            ax0.plot(wn, normalize(x[i]) + offset, color=color, lw=0.55, alpha=0.55)
            offset += 0.12
        mean = normalize(x[mask].mean(axis=0)) + offset
        ax0.plot(wn, mean, color=color, lw=1.4, label=analyte)
        offset += 0.35
    ax0.set_xlim(400, 1800)
    ax0.set_yticks([])
    ax0.set_xlabel(cm_label())
    ax0.set_ylabel("Replicate spectra (offset)")
    ax0.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.00, 1.03), borderaxespad=0.0)
    panel_label(ax0, "a")
    despine(ax0)

    # Main-text reproducibility uses the same concentration (10^-4 M) and a
    # marker-band area metric. This avoids an unnecessary concentration mismatch
    # while matching common SERS practice of reporting representative peak/band
    # repeatability rather than exhaustive all-condition QC.
    marker_defs = {
        "Thiram 1382": ("Thiram", 1382, 4),
        "MG 1616": ("MG", 1616, 4),
        "4-MBA 1080": ("MBA", 1080, 4),
    }
    rows = []
    for label, (analyte, peak_wn, level) in marker_defs.items():
        mask = pure_mask(meta, analyte, level)
        idx = np.where(mask)[0]
        band = (wn >= peak_wn - 8) & (wn <= peak_wn + 8)
        areas = np.trapezoid(x[idx][:, band], wn[band], axis=1)
        rsd_percent = np.std(areas, ddof=1) / abs(np.mean(areas)) * 100
        rows.append(
            {
                "label": f"{label}\n10$^{{-{level}}}$ M, n={len(idx)}",
                "rsd": float(rsd_percent),
                "color": ANALYTE_COLORS[analyte],
            }
        )
    y = np.arange(len(rows))
    values = np.array([r["rsd"] for r in rows])
    ax1.barh(y, values, color=[r["color"] for r in rows], alpha=0.85)
    ax1.axvline(15, color="#555555", ls="--", lw=0.9)
    ax1.set_yticks(y, [r["label"] for r in rows])
    ax1.invert_yaxis()
    for yy, value in zip(y, values):
        ax1.text(value + 0.45, yy, f"{value:.1f}%", va="center", fontsize=6)
    ax1.set_xlabel("Marker-band area RSD (%)")
    ax1.set_xlim(0, 16.5)
    ax1.set_title("Pure-standard repeatability")
    panel_label(ax1, "b", x=-0.13, y=1.10)
    despine(ax1)
    save_figure(fig, "fig2_quality_panels")


def plot_fig3_spectra() -> None:
    x, wn, meta = load_processed("p4")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), layout="constrained")
    analytes = ["Thiram", "MG", "MBA"]
    for ax, analyte, letter in zip(axes.flat[:3], analytes, ["a", "b", "c"]):
        color = ANALYTE_COLORS[analyte]
        mask = pure_mask(meta, analyte, 4)
        m, s = mean_sd(x, mask)
        y = normalize(m)
        band = s / (np.max(m) - np.min(m) + 1e-12)
        ax.plot(wn, y, color=color, lw=1.4)
        ax.fill_between(wn, y - band, y + band, color=color, alpha=0.15, lw=0)
        for peak, mode in PEAKS[analyte]:
            ax.axvline(peak, color=color, lw=0.7, alpha=0.45)
            yy = np.interp(peak, wn, y)
            ax.text(peak, min(1.05, yy + 0.10), f"{peak}\n{mode}", ha="center", va="bottom", fontsize=5)
        if analyte == "MG":
            add_overlap_regions(ax, alpha=0.10)
            ax.axvline(1616, color=color, lw=1.2)
            ax.text(1616, 0.08, "1616 C=C", rotation=90, ha="right", va="bottom", fontsize=6)
        ax.set_xlim(400, 1800)
        ax.set_ylim(-0.08, 1.18)
        ax.set_xlabel(cm_label())
        ax.set_ylabel("Normalized intensity")
        ax.set_title(analyte if analyte != "MBA" else "4-MBA")
        panel_label(ax, letter)
        despine(ax)

    ax = axes.flat[3]
    for analyte in analytes:
        mask = pure_mask(meta, analyte, 4)
        m = normalize(x[mask].mean(axis=0))
        ax.plot(wn, m, color=ANALYTE_COLORS[analyte], lw=1.1, label=analyte if analyte != "MBA" else "4-MBA")
    add_overlap_regions(ax, alpha=0.12)
    ax.axvline(1616, color=ANALYTE_COLORS["MG"], lw=1.0, ls="-")
    ax.set_xlim(400, 1800)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel(cm_label())
    ax.set_ylabel("Normalized intensity")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Reference spectra and overlap regions")
    panel_label(ax, "d")
    despine(ax)
    save_figure(fig, "fig3_spectra")


def plot_fig4_benchmark() -> None:
    df = pd.read_csv(PAPER / "benchmark_random_cv" / "benchmark_full_matrix.csv")
    best = pd.read_csv(PAPER / "benchmark_random_cv" / "benchmark_best_by_task.csv")
    task_order = [t for t in TASK_ORDER if t in df.Task.unique()]
    models = [m for m in MODEL_ORDER if m in df.Model.unique()]
    pivot = (
        df.groupby(["Model", "Task"], as_index=False)["F1_mean"]
        .max()
        .pivot(index="Model", columns="Task", values="F1_mean")
        .reindex(index=models, columns=task_order)
    )
    fig = plt.figure(figsize=(7.45, 4.2), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.05])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    im = ax0.imshow(pivot.values, vmin=0.45, vmax=1.0, cmap="YlGnBu", aspect="auto")
    ax0.set_xticks(np.arange(len(task_order)), [TASK_LABEL[t].replace(" ", "\n") for t in task_order], rotation=0)
    ax0.set_yticks(np.arange(len(models)), models)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            weight = "bold" if np.isclose(val, np.nanmax(pivot.iloc[:, j])) else "normal"
            color = "white" if val >= 0.93 else "black"
            ax0.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5, color=color, fontweight=weight)
    for j in range(pivot.shape[1]):
        best_i = int(np.nanargmax(pivot.iloc[:, j].to_numpy()))
        ax0.add_patch(Rectangle((j - 0.5, best_i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.1))
    ax0.set_title("Model-task performance landscape")
    panel_label(ax0, "a", x=-0.18)
    cb = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
    cb.set_label("Macro-F1")

    b = best.set_index("Task").reindex(task_order).reset_index()
    y = np.arange(len(b))
    colors = [TASK_COLORS[t] for t in b.Task]
    short_labels = {
        "P1_thiram_presence": "P1 Thiram",
        "P2_mg_presence": "P2 MG",
        "P3_mba_presence": "P3 4-MBA",
        "G1_thiram_molar_grade": "G1 Thiram",
        "G2_mg_molar_grade": "G2 MG",
        "G3_mba_molar_grade": "G3 4-MBA",
    }
    rng = np.random.default_rng(20260602)
    for yy, task, color in zip(y, b.Task, colors):
        values = pivot[task].dropna().to_numpy()
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        vmed = float(np.median(values))
        ax1.hlines(yy, vmin, vmax, color=color, lw=6, alpha=0.18, zorder=1)
        ax1.hlines(yy, vmin, vmax, color=color, lw=1.1, alpha=0.80, zorder=2)
        jitter = rng.normal(0, 0.045, len(values))
        ax1.scatter(values, yy + jitter, s=18, color=color, alpha=0.62, edgecolors="white", linewidths=0.35, zorder=3)
        ax1.vlines(vmed, yy - 0.20, yy + 0.20, color="black", lw=1.1, zorder=4)
        best_row = b.loc[b.Task == task].iloc[0]
        ax1.scatter(best_row.F1_mean, yy, s=42, color=color, edgecolors="black", linewidths=0.9, zorder=5)
        ax1.text(
            1.03,
            yy,
            f"{best_row.Model}/{best_row.Preprocess}",
            transform=ax1.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=6.0,
            color="black",
            clip_on=False,
            zorder=6,
        )
    ax1.text(
        1.03,
        1.02,
        "Best model",
        transform=ax1.transAxes,
        va="bottom",
        ha="left",
        fontsize=6.2,
        fontweight="bold",
        clip_on=False,
    )
    ax1.set_yticks(y, [short_labels[t] for t in b.Task])
    ax1.set_xlim(0.55, 1.005)
    ax1.invert_yaxis()
    ax1.set_xlabel("Macro-F1 across 13 models")
    g2_y = list(b.Task).index("G2_mg_molar_grade")
    ax1.text(
        0.585,
        g2_y + 0.52,
        "largest model dependence",
        fontsize=5.8,
        va="center",
        color="#333333",
    )
    ax1.set_title("Task-dependent model performance")
    panel_label(ax1, "b", x=-0.25)
    despine(ax1)
    save_figure(fig, "fig4_benchmark_heatmap")


def plot_fig5_mg_suppression() -> None:
    x, wn, meta = load_processed("p4")
    ratios = mg_ratio_table(x, wn, meta)
    fig = plt.figure(figsize=(7.2, 5.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    masks = {
        "Pure MG": (meta.has_mg == 1) & (meta.has_thiram == 0) & (meta.has_mba == 0) & (meta.c_mg == 4),
        "MG+MBA": (meta.has_mg == 1) & (meta.has_thiram == 0) & (meta.has_mba == 1) & (meta.c_mg == 4),
        "MG+Thiram": (meta.has_mg == 1) & (meta.has_thiram == 1) & (meta.has_mba == 0) & (meta.c_mg == 4),
        "Ternary": (meta.has_mg == 1) & (meta.has_thiram == 1) & (meta.has_mba == 1) & (meta.c_mg == 4),
    }
    means = {}
    for label, mask in masks.items():
        m = x[mask.values].mean(axis=0)
        means[label] = m
    base = means["Pure MG"]
    base_shift = float(np.percentile(base, 5))
    base_scale = float(np.max(base - base_shift))
    if base_scale == 0:
        base_scale = 1.0
    scaled_means = {label: (m - base_shift) / base_scale for label, m in means.items()}
    line_styles = {
        "Pure MG": ("-", 1.4),
        "MG+MBA": ("-", 1.5),
        "MG+Thiram": ("-", 1.5),
        "Ternary": ("-", 1.9),
    }
    for label, m in means.items():
        y = scaled_means[label]
        ls, lw = line_styles[label]
        ax0.plot(wn, y, color=CONDITION_COLORS[label], lw=lw, ls=ls, label=label)
    ax0.axvspan(1608, 1624, color=ANALYTE_COLORS["MG"], alpha=0.08, lw=0)
    ax0.axvline(1616, color=ANALYTE_COLORS["MG"], lw=1.1)
    med_1616 = ratios[ratios.peak == 1616].groupby("condition")["ratio"].median()
    ratio_text = "\n".join(
        [
            "1616 ratio vs pure MG:",
            f"MG+MBA {med_1616['MG+MBA']:.3f}",
            f"MG+Thiram {med_1616['MG+Thiram']:.3f}",
            f"Ternary {med_1616['Ternary']:.3f}",
        ]
    )
    ax0.text(0.03, 0.96, ratio_text, transform=ax0.transAxes, ha="left", va="top", fontsize=7)
    y_max = max(1.05, max(float(np.nanmax(v[(wn >= 1500) & (wn <= 1700)])) for v in scaled_means.values()) * 1.08)
    ax0.set_xlim(1500, 1700)
    ax0.set_ylim(-0.03, y_max)
    ax0.text(1616, y_max * 0.94, "MG 1616", rotation=90, ha="right", va="top", fontsize=6)
    ax0.set_xlabel(cm_label())
    ax0.set_ylabel("Common-scaled intensity")
    ax0.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.55, 1.04), ncol=2, borderaxespad=0)
    use_origin_fig5_panel = False
    fig5_origin_crop = (0.0, 0.0, 0.006, 0.002)
    if use_origin_fig5_panel and show_origin_image(ax0, "origin_clean_fig5_mg_suppression.png", crop=fig5_origin_crop):
        ax0.text(
            0.14,
            0.90,
            ratio_text,
            transform=ax0.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )
        x_1616 = origin_x_fraction(1616, 1500, 1700, 0.094, 0.960, fig5_origin_crop)
        x_1586 = origin_x_fraction(1586, 1500, 1700, 0.094, 0.960, fig5_origin_crop)
        peak_label_color = "#303030"
        secondary_peak_color = "#5A5A5A"
        ax0.plot([x_1616, x_1616], [0.54, 0.745], transform=ax0.transAxes, color=peak_label_color, lw=0.95)
        ax0.plot([x_1616 - 0.010, x_1616 + 0.010], [0.745, 0.745], transform=ax0.transAxes, color=peak_label_color, lw=0.8)
        ax0.text(
            x_1616 + 0.012,
            0.750,
            "1616",
            transform=ax0.transAxes,
            rotation=90,
            ha="left",
            va="top",
            fontsize=6,
            color=peak_label_color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.58, "pad": 0.8},
        )
        ax0.plot([x_1586, x_1586], [0.485, 0.600], transform=ax0.transAxes, color=secondary_peak_color, lw=0.75)
        ax0.text(
            x_1586 - 0.010,
            0.610,
            "1586",
            transform=ax0.transAxes,
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=5.7,
            color=secondary_peak_color,
        )
        ax0.text(0.52, -0.08, cm_label(), transform=ax0.transAxes, ha="center", va="top", fontsize=7)
        ax0.text(-0.07, 0.50, "Common-scaled intensity", transform=ax0.transAxes, ha="center", va="center", rotation=90, fontsize=7)
        proxy_lines = [
            plt.Line2D([0], [0], color=CONDITION_COLORS["Pure MG"], lw=1.4, label="Pure MG"),
            plt.Line2D([0], [0], color=CONDITION_COLORS["MG+MBA"], lw=1.5, label="MG+MBA"),
            plt.Line2D([0], [0], color=CONDITION_COLORS["MG+Thiram"], lw=1.5, label="MG+Thiram"),
            plt.Line2D([0], [0], color=CONDITION_COLORS["Ternary"], lw=1.9, label="Ternary"),
        ]
        ax0.legend(handles=proxy_lines, frameon=False, loc="lower center", bbox_to_anchor=(0.56, 1.000), ncol=2, borderaxespad=0)
        panel_label(ax0, "a", x=-0.06)
    else:
        panel_label(ax0, "a")
        despine(ax0)

    conds = ["MG+MBA", "MG+Thiram", "Ternary"]
    data = [ratios[(ratios.condition == c) & (ratios.peak == 1616)]["ratio"].to_numpy() for c in conds]
    vio = ax1.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for body, cond in zip(vio["bodies"], conds):
        body.set_facecolor(CONDITION_COLORS[cond])
        body.set_edgecolor("none")
        body.set_alpha(0.35)
    rng = np.random.default_rng(7)
    for i, (cond, arr) in enumerate(zip(conds, data), start=1):
        ax1.scatter(i + rng.normal(0, 0.035, len(arr)), arr, s=8, color=CONDITION_COLORS[cond], alpha=0.45, lw=0)
        med = np.median(arr)
        ax1.plot([i - 0.22, i + 0.22], [med, med], color="black", lw=1.2)
        ax1.text(i, min(1.05, med + 0.08), f"{med:.3f}\nn={len(arr)}", ha="center", fontsize=6)
    ternary = data[-1]
    ax1.axhline(1.0, color="#777777", lw=0.8, ls="--")
    ax1.axhline(0.5, color="#777777", lw=0.8, ls=":")
    ax1.text(
        3.42,
        0.5,
        "0.5 x pure MG",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#555555",
    )
    ax1.set_xticks(np.arange(1, 4), conds, rotation=20, ha="right")
    ax1.set_ylabel("MG 1616 peak ratio\nvs same-level pure MG")
    ax1.set_ylim(-0.02, 1.22)
    ax1.text(
        3.34,
        0.64,
        f"{int((ternary < 0.5).sum())}/{len(ternary)}\nternary <0.5x",
        ha="right",
        va="center",
        fontsize=6.4,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.74},
    )
    try:
        from scipy.stats import kruskal

        p = kruskal(*data).pvalue
        ptext = "Kruskal-Wallis p<0.001" if p < 0.001 else f"Kruskal-Wallis p={p:.3f}"
        ax1.text(1.1, 1.12, ptext, fontsize=6)
    except Exception:
        pass
    panel_label(ax1, "b", y=1.12)
    despine(ax1)

    heat = (
        ratios.groupby(["peak", "condition"])["ratio"].median().unstack("condition").reindex(index=[1616, 1220, 1172, 1394], columns=conds)
    )
    im = ax2.imshow(heat.values, cmap="magma", vmin=0, vmax=1.0, aspect="auto")
    ax2.set_xticks(np.arange(len(conds)), conds, rotation=20, ha="right")
    ax2.set_yticks(np.arange(len(heat.index)), [f"{p}" + ("*" if p in [1172, 1394] else "") for p in heat.index])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax2.text(j, i, f"{heat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if heat.iloc[i, j] < 0.45 else "black")
    ax2.set_title("Median peak ratio")
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.02)
    cb.set_label("Peak ratio\nlower = stronger suppression")
    panel_label(ax2, "c")

    ax3.set_axis_off()
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.text(0.50, 0.91, "Competitive adsorption at an AgNP hot spot", ha="center", fontsize=8, fontweight="bold")
    ag_colors = {"facecolor": "#E2E2E2", "edgecolor": "#777777", "lw": 1.0}
    ax3.add_patch(Circle((0.27, 0.52), 0.18, **ag_colors))
    ax3.add_patch(Circle((0.42, 0.52), 0.18, **ag_colors))
    ax3.add_patch(Rectangle((0.335, 0.38), 0.052, 0.28, facecolor="#FFF2CC", edgecolor="#C7B25A", lw=0.6, alpha=0.85))
    ads_sites = [
        (0.34, 0.43, "Thiram", ANALYTE_COLORS["Thiram"]),
        (0.37, 0.49, "4-MBA", ANALYTE_COLORS["MBA"]),
        (0.35, 0.56, "MG", ANALYTE_COLORS["MG"]),
        (0.38, 0.61, "Thiram", ANALYTE_COLORS["Thiram"]),
        (0.36, 0.66, "4-MBA", ANALYTE_COLORS["MBA"]),
    ]
    for x0, y0, _, color in ads_sites:
        ax3.add_patch(Circle((x0, y0), 0.018, facecolor=color, edgecolor="white", lw=0.4, alpha=0.95))
    ax3.add_patch(FancyArrowPatch((0.52, 0.52), (0.61, 0.52), arrowstyle="-|>", mutation_scale=10, lw=0.9, color="#555555"))
    ax3.text(0.77, 0.75, "MG 1616 response", ha="center", fontsize=7)
    bar_base, bar_scale = 0.29, 0.32
    bars = [("Pure MG", 1.00, "#333333", 0.70), ("Ternary", 0.13, ANALYTE_COLORS["MG"], 0.84)]
    ax3.plot([0.64, 0.92], [bar_base, bar_base], color="#777777", lw=0.8)
    ax3.plot([0.64, 0.64], [bar_base, bar_base + bar_scale], color="#777777", lw=0.8)
    for label, value, color, x0 in bars:
        ax3.add_patch(Rectangle((x0 - 0.035, bar_base), 0.07, value * bar_scale, facecolor=color, edgecolor="none", alpha=0.88))
        ax3.text(x0, bar_base + value * bar_scale + 0.030, f"{value:.2f}", ha="center", fontsize=6.5)
        ax3.text(x0, 0.20, label, ha="center", va="top", fontsize=6.2, rotation=15)
    ax3.text(0.50, 0.05, "Ternary adsorption attenuates the MG marker band", ha="center", fontsize=7.5)
    panel_label(ax3, "d")
    save_figure(fig, "fig5_mg_suppression")


def aggregate_signed_shap(npz_path: Path):
    shap = np.load(npz_path)
    signed = []
    for task in TASK_ORDER:
        arr = shap[task]
        signed.append(arr[..., -1].mean(axis=0))
    return np.mean(np.vstack(signed), axis=0)


def plot_shap_overlay(ax, wn, spectrum, mean_abs, signed, title=None):
    y = normalize(spectrum)
    ax.plot(wn, y, color="#555555", lw=0.9)
    add_overlap_regions(ax, alpha=0.08)
    thresh = np.quantile(mean_abs, 0.94)
    idx = np.where(mean_abs >= thresh)[0]
    draw_idx = np.r_[idx[signed[idx] >= 0], idx[signed[idx] < 0]]
    sizes = 12 + 80 * (mean_abs[draw_idx] / (mean_abs[draw_idx].max() + 1e-12))
    max_abs = float(np.nanmax(np.abs(signed[draw_idx])))
    if max_abs == 0:
        max_abs = 1.0
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    sc = ax.scatter(
        wn[draw_idx],
        y[draw_idx],
        c=signed[draw_idx],
        s=sizes,
        cmap="RdBu_r",
        norm=norm,
        alpha=0.78,
        lw=0,
    )
    ax.set_xlim(400, 1800)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel(cm_label())
    ax.set_ylabel("Normalized intensity")
    if title:
        ax.set_title(title)
    despine(ax)
    return sc


def plot_fig6_shap() -> None:
    x, wn, meta = load_processed("p4")
    spectrum = x.mean(axis=0)
    shap_dir = PAPER / "shap_peak_cluster"
    audit = pd.read_csv(shap_dir / "shap_assignment_audit.csv")
    mean_abs = pd.read_csv(shap_dir / "shap_mean_abs_by_wavenumber.csv")
    shap_npz = np.load(shap_dir / "shap_values_full.npz")
    fig = plt.figure(figsize=(7.2, 6.4), layout="constrained")
    outer = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.25], width_ratios=[1.05, 1.15])
    ax0 = fig.add_subplot(outer[0, 0])
    ax1 = fig.add_subplot(outer[0, 1])
    sub = outer[1, 0].subgridspec(2, 3, wspace=0.10, hspace=0.18)
    ax3 = fig.add_subplot(outer[1, 1])

    top = audit[audit.assignment_level != "unassigned"].sort_values("total_shap", ascending=False).head(18).iloc[::-1]
    colors = top.task.map(TASK_COLORS)
    analyte_short = {
        "P1_thiram_presence": "Thiram",
        "G1_thiram_molar_grade": "Thiram",
        "P2_mg_presence": "MG",
        "G2_mg_molar_grade": "MG",
        "P3_mba_presence": "4-MBA",
        "G3_mba_molar_grade": "4-MBA",
    }
    labels = [f"{analyte_short.get(r.task, r.task.split('_')[0])} {r.center_wavenumber:.0f}" for r in top.itertuples()]
    ax0.barh(np.arange(len(top)), top.total_shap, color=colors, alpha=0.90)
    ax0.set_yticks(np.arange(len(top)), labels)
    ax0.set_xlabel("Cluster SHAP mass")
    ax0.set_xlim(0, float(top.total_shap.max()) * 1.10)
    assigned_count = int((audit.assignment_level != "unassigned").sum())
    total_count = int(len(audit))
    assigned_mass = float(audit.loc[audit.assignment_level != "unassigned", "total_shap"].sum())
    total_mass = float(audit["total_shap"].sum())
    assigned_pct = 100 * assigned_mass / total_mass if total_mass else np.nan
    ax0.text(
        1.02,
        0.98,
        f"{assigned_count}/{total_count} clusters\n{assigned_pct:.1f}% assigned\nSHAP mass",
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        fontweight="bold",
        clip_on=False,
    )
    panel_label(ax0, "a", x=-0.28)
    despine(ax0)

    focus_task = "G2_mg_molar_grade"
    focus_abs = (
        mean_abs[mean_abs.task == focus_task]
        .set_index("feature_index")["mean_abs_shap"]
        .reindex(np.arange(len(wn)))
        .fillna(0)
        .to_numpy()
    )
    focus_signed = shap_npz[focus_task][..., -1].mean(axis=0)
    sc = plot_shap_overlay(ax1, wn, spectrum, focus_abs, focus_signed, "Characteristic and interference-sensitive bands")
    ax1.set_xlim(800, 1700)
    ax1.axvline(1616, color=ANALYTE_COLORS["MG"], lw=1.0)
    ax1.text(1616, 0.08, "1616", rotation=90, ha="right", va="bottom", fontsize=6)
    panel_label(ax1, "b")
    cb = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.02)
    cb.set_label("Signed SHAP")

    mini_axes = []
    for i, task in enumerate(TASK_ORDER):
        ax = fig.add_subplot(sub[i // 3, i % 3])
        task_abs = mean_abs[mean_abs.task == task].set_index("feature_index")["mean_abs_shap"].reindex(np.arange(len(wn))).fillna(0).to_numpy()
        task_signed = shap_npz[task][..., -1].mean(axis=0)
        plot_shap_overlay(ax, wn, spectrum, task_abs, task_signed, TASK_LABEL[task])
        ax.set_ylabel("")
        if i // 3 == 0:
            ax.set_xlabel("")
        mini_axes.append(ax)
    panel_label(mini_axes[0], "c", x=-0.34)

    mg_regions = [
        ("MG 1616 marker band", [(1600, 1630)], ANALYTE_COLORS["MG"]),
        ("1172/1220 MG bands", [(1162, 1230)], "#E99445"),
        ("1394 overlap band", [(1370, 1405)], "#C85A3E"),
        ("Coexisting-analyte bands", [(850, 880), (1490, 1515)], "#777777"),
    ]
    region_values = []
    for label, windows, color in mg_regions:
        mask = np.zeros_like(wn, dtype=bool)
        for lo, hi in windows:
            mask |= (wn >= lo) & (wn <= hi)
        region_values.append((label, float(focus_abs[mask].sum()), color))
    region_values = sorted(region_values, key=lambda item: item[1], reverse=True)
    max_region = max(v for _, v, _ in region_values) or 1.0
    y_pos = np.arange(len(region_values))[::-1]
    for y_i, (label, value, color) in zip(y_pos, region_values):
        ax3.barh(y_i, value / max_region, color=color, alpha=0.88, height=0.52)
        ax3.text(value / max_region + 0.035, y_i, f"{value / max_region:.2f}", va="center", fontsize=6)
    ax3.set_yticks(y_pos, [label for label, _, _ in region_values])
    ax3.set_xlim(0, 1.12)
    ax3.set_xlabel("Relative G2 SHAP mass")
    ax3.set_title("G2/MG attribution by spectral region", loc="left", pad=4)
    panel_label(ax3, "d", x=-0.20)
    despine(ax3)
    save_figure(fig, "fig6_shap_attribution")


def plot_fig7_feature_selection() -> None:
    x, wn, meta = load_processed("p4")
    fs_dir = PAPER / "shap_feature_selection"
    curve = pd.read_csv(fs_dir / "shap_feature_elimination_curve.csv")
    mean_abs = pd.read_csv(PAPER / "shap_peak_cluster" / "shap_mean_abs_by_wavenumber.csv")
    fig = plt.figure(figsize=(7.2, 4.7), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 0.9])
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    for task in TASK_ORDER:
        sub = curve[curve.task == task].sort_values("retention_pct")
        ax0.plot(sub.retention_pct, sub.macro_f1_mean, marker="o", ms=3, lw=1.0, color=TASK_COLORS[task], label=TASK_LABEL[task])
        ax0.fill_between(sub.retention_pct, sub.macro_f1_mean - sub.macro_f1_std, sub.macro_f1_mean + sub.macro_f1_std, color=TASK_COLORS[task], alpha=0.10, lw=0)
    ax0.axvline(10, color="#333333", lw=0.9, ls="--")
    ax0.axhline(0.95, color="#555555", lw=0.8, ls=":")
    ax0.text(11.5, 0.953, "10% = 141/1401\n6/6 tasks >0.95", fontsize=8, fontweight="bold")
    ax0.set_xlim(8, 102)
    ax0.set_ylim(0.93, 1.005)
    ax0.set_xlabel("Retained wavenumbers (%)")
    ax0.set_ylabel("Macro-F1")
    ax0.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.50, -0.13), borderaxespad=0.0, fontsize=5.7)
    sec = ax0.secondary_xaxis("top", functions=(lambda p: p * 14.01, lambda n: n / 14.01))
    sec.set_xlabel("Number of retained points")
    panel_label(ax0, "a", x=-0.10)
    despine(ax0)

    spectrum = normalize(x.mean(axis=0))
    ax1.plot(wn, spectrum, color="#666666", lw=0.9)
    for task in TASK_ORDER:
        task_abs = mean_abs[mean_abs.task == task].sort_values("mean_abs_shap", ascending=False).head(141)
        heights = np.interp(task_abs.wavenumber, wn, spectrum)
        ax1.vlines(task_abs.wavenumber, 0, heights, color=TASK_COLORS[task], lw=0.35, alpha=0.35)
    add_overlap_regions(ax1, alpha=0.08)
    ax1.axvline(1616, color=ANALYTE_COLORS["MG"], lw=0.9)
    ax1.set_xlim(400, 1800)
    ax1.set_ylim(-0.03, 1.08)
    ax1.set_xlabel(cm_label())
    ax1.set_ylabel("Retained regions")
    panel_label(ax1, "b", x=-0.20)
    despine(ax1)

    bar_rows = []
    for task in TASK_ORDER:
        sub = curve[curve.task == task].set_index("retention_pct")
        bar_rows.append(
            {
                "task": task,
                "full": sub.loc[100, "macro_f1_mean"],
                "full_std": sub.loc[100, "macro_f1_std"],
                "ten": sub.loc[10, "macro_f1_mean"],
                "ten_std": sub.loc[10, "macro_f1_std"],
            }
        )
    bdf = pd.DataFrame(bar_rows)
    bdf["delta"] = bdf["ten"] - bdf["full"]
    xloc = np.arange(len(bdf))
    ax2.axhline(0, color="#555555", lw=0.8)
    ax2.bar(xloc, bdf.delta, width=0.58, color=[TASK_COLORS[t] for t in bdf.task], alpha=0.88)
    for x_i, delta in zip(xloc, bdf.delta):
        va = "bottom" if delta >= 0 else "top"
        y_text = delta + (0.0018 if delta >= 0 else -0.0018)
        ax2.text(x_i, y_text, f"{delta:+.3f}", ha="center", va=va, fontsize=5.8)
    ax2.set_xticks(xloc, [t.split("_")[0] for t in bdf.task], rotation=0)
    ylim = max(0.018, float(np.nanmax(np.abs(bdf.delta))) * 1.35)
    ax2.set_ylim(-ylim, ylim)
    ax2.set_ylabel(r"$\Delta$macro-F1 (10% - full)")
    ax2.text(
        0.02,
        0.96,
        "all |delta| <= 0.014",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#333333",
    )
    panel_label(ax2, "c", x=-0.20)
    despine(ax2)
    save_figure(fig, "fig7_feature_elimination")


def auc_score(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score)
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, score))
    except Exception:
        pos = y_true == 1
        neg = y_true == 0
        if pos.sum() == 0 or neg.sum() == 0:
            return np.nan
        order = np.argsort(score)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(score) + 1)
        return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def permutation_null(pred: pd.DataFrame, out_path: Path, n_perm: int = 10000) -> pd.DataFrame:
    if out_path.exists():
        return pd.read_csv(out_path)
    rng = np.random.default_rng(20260601)
    rows = []
    for task, sub in pred.groupby("task"):
        sub = sub.copy()
        folds = sorted(sub.fold.unique())
        scores = sub.prob_positive.to_numpy()
        for i in range(n_perm):
            y_perm = sub.y_true.to_numpy().copy()
            for fold in folds:
                idx = np.where(sub.fold.to_numpy() == fold)[0]
                y_perm[idx] = rng.permutation(y_perm[idx])
            rows.append({"task": task, "perm_id": i, "auc": auc_score(y_perm, scores)})
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def plot_fig8_soil() -> None:
    soil_dir = PAPER / "soil_validation"
    summary = pd.read_csv(soil_dir / "soil_cv_summary.csv")
    roc = pd.read_csv(soil_dir / "soil_cv_roc_curve.csv")
    pred = pd.read_csv(soil_dir / "soil_cv_predictions.csv")
    blank = pd.read_csv(soil_dir / "soil_blank_predictions.csv")
    x_soil = np.load(soil_dir / "X_soil_p1.npy")
    x_blank = np.load(soil_dir / "X_blanks_p1.npy")
    wn = np.load(soil_dir / "wavenumber.npy")
    meta = pd.read_csv(soil_dir / "soil_metadata.csv")

    fig = plt.figure(figsize=(7.3, 5.35), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.96, 1.08])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    grid = np.linspace(0, 1, 200)
    for task in ["P1_thiram_presence", "P2_mg_presence", "P3_mba_presence"]:
        sub = roc[roc.task == task]
        fold_curves = []
        for _, g in sub.groupby("fold"):
            ax0.plot(g.fpr, g.tpr, color=TASK_COLORS[task], alpha=0.18, lw=0.7)
            fold_curves.append(np.interp(grid, g.fpr, g.tpr))
        mean = np.mean(fold_curves, axis=0)
        auc = summary.loc[summary.task == task, "AUC_mean"].iloc[0]
        ax0.plot(grid, mean, color=TASK_COLORS[task], lw=1.6, label=f"{TASK_LABEL[task]} AUC={auc:.3f}")
    ax0.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1.02)
    ax0.set_xlabel("False positive rate")
    ax0.set_ylabel("True positive rate")
    ax0.legend(frameon=False, loc="lower right")
    panel_label(ax0, "a")
    despine(ax0)

    tasks = ["P1_thiram_presence", "P2_mg_presence", "P3_mba_presence"]
    rng = np.random.default_rng(11)
    for i, task in enumerate(tasks):
        sub = pred[pred.task == task]
        pos = sub[sub.y_true == 1]
        bl = blank[blank.task == task]
        ax1.scatter(i - 0.08 + rng.normal(0, 0.026, len(pos)), pos.prob_positive, color=TASK_COLORS[task], s=15, alpha=0.72, lw=0)
        ax1.scatter(i + 0.16 + rng.normal(0, 0.018, len(bl)), bl.prob_positive, marker="D", facecolors="none", edgecolors="#555555", s=20, lw=0.75)
    ax1.set_xticks(np.arange(len(tasks)), [TASK_LABEL[t] for t in tasks], rotation=15, ha="right")
    ax1.axhline(0.5, color="#777777", lw=0.8, ls=":")
    ax1.text(
        2.35,
        0.515,
        "0.5 threshold",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#555555",
    )
    ax1.set_ylabel("Predicted positive probability")
    ax1.set_xlabel("")
    ax1.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", linestyle="none", color="#555555", markerfacecolor="#555555", markersize=4.0, label="Spiked soil"),
            plt.Line2D([0], [0], marker="D", linestyle="none", color="#555555", markerfacecolor="none", markersize=4.0, label="Soil blank"),
        ],
        frameon=False,
        loc="upper left",
        fontsize=6,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    ax1.set_ylim(-0.03, 1.12)
    panel_label(ax1, "b")
    despine(ax1)

    auc_values = []
    auc_std_values = []
    for task in tasks:
        auc_values.append(summary.loc[summary.task == task, "AUC_mean"].iloc[0])
        auc_std_values.append(summary.loc[summary.task == task, "AUC_std"].iloc[0])
    y = np.arange(len(tasks))
    ax2.barh(
        y,
        auc_values,
        xerr=auc_std_values,
        height=0.42,
        color=[TASK_COLORS[t] for t in tasks],
        alpha=0.88,
        capsize=2,
    )
    for yi, auc, sd in zip(y, auc_values, auc_std_values):
        ax2.text(1.012, yi, f"{auc:.3f} ± {sd:.3f}", va="center", fontsize=6)
    ax2.set_yticks(y, [TASK_LABEL[t] for t in tasks])
    ax2.set_xlim(0.90, 1.04)
    ax2.set_xlabel("Five-fold AUC")
    ax2.set_title("Soil matrix screening")
    panel_label(ax2, "c")
    despine(ax2)

    ternary_mask = (meta.has_thiram == 1) & (meta.has_mg == 1) & (meta.has_mba == 1)
    soil_mean = normalize(x_soil[ternary_mask.values].mean(axis=0))
    blank_mean = normalize(x_blank.mean(axis=0))
    ax3.plot(wn, blank_mean, color="#777777", lw=1.0, label="Soil blank")
    ax3.plot(wn, soil_mean + 1.05, color=ANALYTE_COLORS["MG"], lw=1.0, label="Spiked ternary soil")
    peak_labels = [
        (560, "560", ANALYTE_COLORS["Thiram"]),
        (1078, "1078", ANALYTE_COLORS["MBA"]),
        (1380, "1380", ANALYTE_COLORS["Thiram"]),
        (1586, "1586", ANALYTE_COLORS["MBA"]),
        (1616, "1616", ANALYTE_COLORS["MG"]),
    ]
    neutral_peak_color = "#4A4A4A"
    for peak, label, color in peak_labels:
        ax3.vlines(peak, -0.04, 0.12, color=neutral_peak_color, lw=0.9, alpha=0.92)
        ax3.text(peak, 2.18, label, color=neutral_peak_color, rotation=90, ha="center", va="top", fontsize=5.5)
    add_overlap_regions(ax3, alpha=0.08)
    ax3.set_xlim(400, 1800)
    ax3.set_ylim(-0.05, 2.25)
    ax3.set_yticks([])
    ax3.set_xlabel(cm_label())
    ax3.set_ylabel("Mean p1 spectra (offset)")
    ax3.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig8_origin_crop = (0.0, 0.0, 0.006, 0.002)
    if show_origin_image(ax3, "origin_clean_fig8_soil_spectra.png", crop=fig8_origin_crop):
        neutral_peak_color = "#4A4A4A"
        label_offsets = {
            1586: (-0.008, 0.005, "right"),
            1616: (0.008, 0.030, "left"),
        }
        for peak, label, color in peak_labels:
            x_peak = origin_x_fraction(peak, 400, 1800, 0.098, 0.990, fig8_origin_crop)
            is_mg_marker = peak == 1616
            line_top = 0.150 if is_mg_marker else 0.135
            ax3.plot(
                [x_peak, x_peak],
                [0.070, line_top],
                transform=ax3.transAxes,
                color=neutral_peak_color,
                lw=0.9 if is_mg_marker else 0.75,
                alpha=0.95,
            )
            dx, dy, ha = label_offsets.get(peak, (0.0, 0.0, "center"))
            ax3.text(
                x_peak + dx,
                line_top + 0.010 + dy,
                label,
                transform=ax3.transAxes,
                color=neutral_peak_color,
                rotation=90,
                ha=ha,
                va="bottom",
                fontsize=5.4,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.58, "pad": 0.5},
            )
        ax3.text(0.52, -0.08, cm_label(), transform=ax3.transAxes, ha="center", va="top", fontsize=7)
        ax3.text(-0.07, 0.50, "Mean p1 spectra (offset)", transform=ax3.transAxes, ha="center", va="center", rotation=90, fontsize=7)
        proxy_lines = [
            plt.Line2D([0], [0], color="#777777", lw=1.0, label="Soil blank"),
            plt.Line2D([0], [0], color=ANALYTE_COLORS["MG"], lw=1.2, label="Spiked ternary soil"),
        ]
        ax3.legend(
            handles=proxy_lines,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.82,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.985),
            ncol=2,
            fontsize=5.8,
            handlelength=1.6,
            columnspacing=1.0,
            borderaxespad=0.2,
        )
        panel_label(ax3, "d", x=-0.06)
    else:
        panel_label(ax3, "d")
        despine(ax3)
    save_figure(fig, "fig8_soil_screening")


def main() -> None:
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_fig2_quality()
    plot_fig3_spectra()
    plot_fig4_benchmark()
    plot_fig5_mg_suppression()
    plot_fig6_shap()
    plot_fig7_feature_selection()
    plot_fig8_soil()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
