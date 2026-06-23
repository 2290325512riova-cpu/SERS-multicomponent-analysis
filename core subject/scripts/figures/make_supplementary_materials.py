from __future__ import annotations

from pathlib import Path
import math
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

from paper_style import (  # noqa: E402
    ANALYTE_COLORS,
    MODEL_ORDER,
    TASK_COLORS,
    TASK_LABEL,
    TASK_ORDER,
    apply_style,
    cm_label,
    despine,
    panel_label,
)

PROC = ROOT / "data" / "pure63_mainline" / "processed"
SPLIT = ROOT / "data" / "pure63_mainline" / "splits" / "cv_split_random_5fold.csv"
PAPER = ROOT / "data" / "pure63_mainline" / "models" / "paper_main"
TABLE_DIR = ROOT / "tables" / "paper_main"
FIG_SUPP = ROOT / "figures" / "paper_supplement"
LATEX_DIR = WORKSPACE / "latex_manuscript"
LATEX_FIG = LATEX_DIR / "figures"
SOURCE_DIR = FIG_SUPP / "source_data"

PREPROCESS_ORDER = ["raw", "p1", "p2", "p3", "p4", "p5"]
PREPROCESS_LABEL = {
    "raw": "raw",
    "p1": "p1",
    "p2": "p2",
    "p3": "p3",
    "p4": "p4",
    "p5": "p5",
}
DL_MODELS = ["1D-CNN", "1D-ResNet", "KAN-CNN", "RamanNet-Lite", "Spectrum-KAN"]


def save_supp_figure(fig, stem: str) -> None:
    FIG_SUPP.mkdir(parents=True, exist_ok=True)
    LATEX_FIG.mkdir(parents=True, exist_ok=True)
    for target in (FIG_SUPP, LATEX_FIG):
        fig.savefig(target / f"{stem}.pdf", dpi=600)
        fig.savefig(target / f"{stem}.png", dpi=300)
    plt.close(fig)


def normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = y - np.nanmin(y)
    denom = np.nanmax(y)
    if denom == 0 or not np.isfinite(denom):
        return y
    return y / denom


def plot_s1_full_benchmark() -> None:
    df = pd.read_csv(PAPER / "benchmark_random_cv" / "benchmark_full_matrix.csv")
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SOURCE_DIR / "figS1_full_benchmark_source.csv", index=False)

    tasks = [t for t in TASK_ORDER if t in df.Task.unique()]
    models = [m for m in MODEL_ORDER if m in df.Model.unique()]
    preps = [p for p in PREPROCESS_ORDER if p in df.Preprocess.unique()]

    fig, axes = plt.subplots(3, 2, figsize=(7.35, 8.8), layout="constrained")
    im = None
    for ax, task in zip(axes.flat, tasks):
        pivot = (
            df[df.Task == task]
            .pivot_table(index="Model", columns="Preprocess", values="F1_mean", aggfunc="max")
            .reindex(index=models, columns=preps)
        )
        im = ax.imshow(pivot.values, vmin=0.55, vmax=1.0, cmap="YlGnBu", aspect="auto")
        ax.set_title(TASK_LABEL.get(task, task), loc="left", pad=3)
        ax.set_xticks(np.arange(len(preps)), [PREPROCESS_LABEL[p] for p in preps], rotation=0)
        ax.set_yticks(np.arange(len(models)), models if ax in axes[:, 0] else [""] * len(models))
        best = np.nanmax(pivot.values)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if np.isfinite(val) and np.isclose(val, best):
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.0))
        ax.tick_params(length=2, width=0.6)
    for i, ax in enumerate(axes.flat):
        panel_label(ax, chr(ord("a") + i), x=-0.12 if i % 2 == 0 else -0.08, y=1.03)
    if im is not None:
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.62, pad=0.012)
        cb.set_label("Macro-F1")
    save_supp_figure(fig, "figS1_full_benchmark_matrix")


def plot_s2_augmentation_ablation() -> None:
    modes = [
        ("no_aug", "No augmentation"),
        ("aug_no_mixup", "Augmented, no mixup"),
        ("composition_mixup", "Composition mixup"),
    ]
    rows = []
    for mode, label in modes:
        path = PAPER / "augmentation_ablation" / mode / "benchmark_full_matrix.csv"
        df = pd.read_csv(path)
        df["augmentation"] = label
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(SOURCE_DIR / "figS2_augmentation_ablation_source.csv", index=False)

    best_task = (
        all_df.groupby(["augmentation", "Task"], as_index=False)["F1_mean"]
        .max()
        .assign(mode_order=lambda d: d["augmentation"].map({label: i for i, (_, label) in enumerate(modes)}))
        .sort_values(["Task", "mode_order"])
    )
    model_mean = (
        all_df.groupby(["augmentation", "Model", "Task"], as_index=False)["F1_mean"]
        .max()
        .groupby(["augmentation", "Model"], as_index=False)["F1_mean"]
        .mean()
        .assign(mode_order=lambda d: d["augmentation"].map({label: i for i, (_, label) in enumerate(modes)}))
    )

    fig = plt.figure(figsize=(7.2, 3.35), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(modes))
    for task in TASK_ORDER:
        sub = best_task[best_task.Task == task].sort_values("mode_order")
        ax0.plot(x, sub.F1_mean, marker="o", ms=3.2, lw=1.0, color=TASK_COLORS[task], label=TASK_LABEL[task])
    ax0.set_xticks(x, [label.replace(", ", "\n") for _, label in modes])
    ax0.set_ylim(0.74, 1.005)
    ax0.set_ylabel("Best DL macro-F1")
    ax0.set_title("Task-level response to augmentation", loc="left")
    ax0.legend(frameon=False, ncol=2, fontsize=5.5, loc="lower right")
    panel_label(ax0, "a", x=-0.10)
    despine(ax0)

    width = 0.22
    offsets = np.linspace(-width, width, len(modes))
    model_order = [m for m in DL_MODELS if m in model_mean.Model.unique()]
    y_base = np.arange(len(model_order))
    palette = ["#A7BBC7", "#6F98B5", "#D28E5D"]
    for offset, (_, label), color in zip(offsets, modes, palette):
        sub = model_mean[model_mean.augmentation == label].set_index("Model").reindex(model_order)
        ax1.barh(y_base + offset, sub.F1_mean, height=width * 0.9, color=color, label=label)
    ax1.set_yticks(y_base, model_order)
    ax1.set_xlim(0.72, 1.005)
    ax1.set_xlabel("Mean best macro-F1 across tasks")
    ax1.set_title("Model-level summary", loc="left")
    ax1.legend(frameon=False, fontsize=5.6, loc="lower right")
    panel_label(ax1, "b", x=-0.17)
    despine(ax1)
    save_supp_figure(fig, "figS2_augmentation_ablation")


def plot_s3_preprocessing_effects() -> None:
    wn = np.load(PROC / "wavenumber.npy")
    meta = pd.read_csv(SPLIT)
    mask = (
        (meta.has_thiram == 1)
        & (meta.has_mg == 1)
        & (meta.has_mba == 1)
        & (meta.c_thiram == 4)
        & (meta.c_mg == 4)
        & (meta.c_mba == 4)
    )
    sample_idx = int(np.where(mask.to_numpy())[0][0])
    rows = []
    fig, ax = plt.subplots(figsize=(7.25, 4.3), layout="constrained")
    offsets = np.arange(len(PREPROCESS_ORDER))[::-1] * 1.12
    colors = ["#5B5B5B", "#2F6FAE", "#8E6BBE", "#2A9D8F", "#C76A2A", "#7D8B44"]
    for offset, tag, color in zip(offsets, PREPROCESS_ORDER, colors):
        x = np.load(PROC / f"X_{tag}.npy")[sample_idx]
        y = normalize(x) + offset
        ax.plot(wn, y, color=color, lw=0.95)
        ax.text(1810, offset + 0.52, tag, va="center", ha="left", fontsize=7, color=color)
        for w, val in zip(wn, x):
            rows.append({"sample_index": sample_idx, "preprocess": tag, "wavenumber": w, "value": val})
    ax.set_xlim(400, 1865)
    ax.set_yticks([])
    ax.set_xlabel(cm_label())
    ax.set_ylabel("Normalized spectra (offset)")
    ax.set_title("Representative raw and preprocessed spectra", loc="left")
    for peak in [560, 1078, 1380, 1590, 1616]:
        ax.axvline(peak, color="#B7B7B7", lw=0.55, ls=":")
    despine(ax)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SOURCE_DIR / "figS3_preprocessing_effects_source.csv", index=False)
    save_supp_figure(fig, "figS3_preprocessing_effects")


def plot_s4_embedding() -> None:
    import umap  # type: ignore[import-not-found]

    wn = np.load(PROC / "wavenumber.npy")
    _ = wn  # retained for source traceability
    x = np.load(PROC / "X_p1.npy")
    meta = pd.read_csv(SPLIT)
    z = StandardScaler().fit_transform(x)
    pca50 = PCA(n_components=50, random_state=42).fit_transform(z)
    pca2 = PCA(n_components=2, random_state=42).fit_transform(z)
    tsne2 = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto", max_iter=1000).fit_transform(pca50)
    umap2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(pca50)

    source = pd.DataFrame(
        {
            "sample_id": meta.sample_id,
            "folder_name": meta.folder_name,
            "mixture_order": meta.mixture_order,
            "family": meta.family,
            "PCA1": pca2[:, 0],
            "PCA2": pca2[:, 1],
            "tSNE1": tsne2[:, 0],
            "tSNE2": tsne2[:, 1],
            "UMAP1": umap2[:, 0],
            "UMAP2": umap2[:, 1],
        }
    )
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    source.to_csv(SOURCE_DIR / "figS4_embedding_source.csv", index=False)

    panels = [("PCA", pca2), ("t-SNE", tsne2), ("UMAP", umap2)]
    order_label = {1: "Single", 2: "Binary", 3: "Ternary"}
    order_color = {1: "#6C8EBF", 2: "#D79A52", 3: "#8C6BB1"}
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.75), layout="constrained")
    for ax, (name, coords) in zip(axes, panels):
        for order in [1, 2, 3]:
            m = meta.mixture_order.to_numpy() == order
            ax.scatter(coords[m, 0], coords[m, 1], s=7, alpha=0.58, color=order_color[order], label=order_label[order], edgecolors="none")
        ax.set_title(name, loc="left")
        ax.set_xlabel(f"{name}-1")
        ax.set_ylabel(f"{name}-2")
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)
    axes[0].legend(frameon=False, loc="best", fontsize=6)
    for i, ax in enumerate(axes):
        panel_label(ax, chr(ord("a") + i), x=-0.12, y=1.03)
    save_supp_figure(fig, "figS4_embedding_overview")


def plot_s5_extended_rsd() -> None:
    rsd = pd.read_csv(PAPER / "data_quality" / "peak_rsd_by_folder.csv")
    meta = pd.read_csv(SPLIT)
    folder_flags = (
        meta.groupby("folder_name")[["has_thiram", "has_mg", "has_mba"]]
        .max()
        .reset_index()
    )
    rsd = rsd.merge(folder_flags, on="folder_name", how="left")
    valid = (
        ((rsd.peak == "Thiram_1382") & (rsd.has_thiram == 1))
        | ((rsd.peak == "MG_1616") & (rsd.has_mg == 1))
        | ((rsd.peak == "MBA_1080") & (rsd.has_mba == 1))
    )
    rsd = rsd[valid & rsd.variant.isin(["raw", "p1", "p3", "p5"])].copy()
    rsd["peak_label"] = rsd.peak.map({"Thiram_1382": "Thiram 1382", "MG_1616": "MG 1616", "MBA_1080": "4-MBA 1080"})
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rsd.to_csv(SOURCE_DIR / "figS5_extended_rsd_source.csv", index=False)

    variants = ["raw", "p1", "p3", "p5"]
    peaks = ["Thiram 1382", "MG 1616", "4-MBA 1080"]
    fig = plt.figure(figsize=(7.25, 3.65), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    positions = np.arange(len(variants))
    peak_offsets = [-0.22, 0.0, 0.22]
    peak_colors = [ANALYTE_COLORS["Thiram"], ANALYTE_COLORS["MG"], ANALYTE_COLORS["4-MBA"]]
    rng = np.random.default_rng(20260623)
    for peak, offset, color in zip(peaks, peak_offsets, peak_colors):
        for i, variant in enumerate(variants):
            values = rsd[(rsd.variant == variant) & (rsd.peak_label == peak)].rsd_percent.to_numpy()
            if len(values) == 0:
                continue
            xj = positions[i] + offset + rng.normal(0, 0.025, size=len(values))
            ax0.scatter(xj, values, s=10, alpha=0.42, color=color, edgecolors="none")
            ax0.plot([positions[i] + offset - 0.07, positions[i] + offset + 0.07], [np.median(values)] * 2, color=color, lw=1.4)
    ax0.axhline(15, color="#555555", lw=0.8, ls="--")
    ax0.set_yscale("log")
    ax0.set_xticks(positions, variants)
    ax0.set_ylabel("Marker-peak RSD (%)")
    ax0.set_title("Valid marker-containing folders", loc="left")
    ax0.set_ylim(0.2, max(300, float(rsd.rsd_percent.quantile(0.98)) * 1.2))
    panel_label(ax0, "a", x=-0.10)
    despine(ax0)

    heat = rsd.groupby(["peak_label", "variant"])["rsd_percent"].median().unstack().reindex(index=peaks, columns=variants)
    im = ax1.imshow(heat.values, cmap="YlOrRd", vmin=0, vmax=min(60, np.nanmax(heat.values)))
    ax1.set_xticks(np.arange(len(variants)), variants)
    ax1.set_yticks(np.arange(len(peaks)), peaks)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            ax1.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6, color="white" if val > 35 else "black")
    ax1.set_title("Median RSD (%)", loc="left")
    panel_label(ax1, "b", x=-0.20)
    cb = fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.02)
    cb.set_label("RSD (%)")
    save_supp_figure(fig, "figS5_extended_rsd")


def tex_escape(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    text = str(value)
    text = text.replace("±", r"$\pm$")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt_float(x, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def fmt_metric(mean, std, digits: int = 3) -> str:
    return f"{float(mean):.{digits}f} $\\pm$ {float(std):.{digits}f}"


def combo(*parts) -> str:
    return "; ".join(tex_escape(part) for part in parts if str(part).strip() and str(part).lower() != "nan")


def peak_text(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    text = str(value).replace("_", " ")
    text = text.replace(",", ", ")
    text = re.sub(r"\s+", " ", text)
    return tex_escape(text)


def longtable(caption: str, label: str, columns: list[str], rows: list[list[str]], widths: list[str], font: str = r"\scriptsize") -> str:
    spec = "".join(f"L{{{w}}}" for w in widths)
    header = " & ".join(tex_escape(col) for col in columns) + r" \\"
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return rf"""
{{{font}
\setlength{{\tabcolsep}}{{2.5pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{longtable}}{{{spec}}}
\caption{{{caption}}}\label{{{label}}}\\
\toprule
{header}
\midrule
\endfirsthead
\toprule
{header}
\midrule
\endhead
\endfoot
\bottomrule
\endlastfoot
{body}
\end{{longtable}}
}}
"""


def write_supplementary_tex() -> None:
    out = LATEX_DIR / "supplementary_materials.tex"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []

    chunks.append(r"""
\clearpage
\section*{补充材料}
\setcounter{figure}{0}
\renewcommand{\thefigure}{S\arabic{figure}}
\setcounter{table}{0}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\figurename}{补充图}
\renewcommand{\tablename}{补充表}

\subsection*{补充图}
""")

    figures = [
        ("figS1_full_benchmark_matrix.pdf", "完整模型-预处理-任务 benchmark 矩阵。六个面板分别对应 P1--P3 存在性任务和 G1--G3 离散浓度档判别任务；每个面板显示 13 类模型在 raw 和 p1--p5 六种光谱输入下的 macro-F1，黑框标出该任务中的最高值。", "fig:s1_full_benchmark"),
        ("figS2_augmentation_ablation.pdf", "深度学习候选模型的数据增强消融结果。a，三种增强策略下各任务的最佳深度学习 macro-F1；b，各深度学习模型跨任务平均最佳 macro-F1。", "fig:s2_augmentation"),
        ("figS3_preprocessing_effects.pdf", "同一样品在 raw 和 p1--p5 输入矩阵中的光谱形态变化示例。各曲线经归一化并纵向平移，仅用于展示预处理对峰形、归一化和导数输入的影响。", "fig:s3_preprocessing"),
        ("figS4_embedding_overview.pdf", "p1 光谱输入的无监督嵌入概览。PCA、t-SNE 和 UMAP 均按单组分、二元和三元混合复杂度着色，用于观察混合复杂度对全谱分布的影响。", "fig:s4_embedding"),
        ("figS5_extended_rsd.pdf", r"扩展重复性与 RSD 质控结果。a，仅保留含对应目标物的文件夹后，不同预处理输入下三个代表峰的 RSD 分布；虚线为 15\% 参考水平。b，各代表峰和预处理输入的中位 RSD。", "fig:s5_rsd"),
    ]
    for i, (filename, caption, label) in enumerate(figures):
        page_break = "" if i == 0 else r"\clearpage"
        chunks.append(rf"""
{page_break}
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.98\textwidth]{{{filename}}}
  \caption{{{caption}}}
  \label{{{label}}}
\end{{figure}}
""")

    chunks.append(r"""
\clearpage
\subsection*{补充表}
""")

    s1_rows = [
        ["raw", "Original spectra interpolated to the 400--1800 cm$^{-1}$ fingerprint region."],
        ["p1", "Spike removal, Savitzky--Golay smoothing, ALS baseline correction and SNV."],
        ["p2", "Savitzky--Golay smoothing, ALS baseline correction, first derivative and SNV."],
        ["p3", "ALS baseline correction and vector normalization."],
        ["p4", "Spike removal, Savitzky--Golay smoothing and ALS baseline correction, without normalization."],
        ["p5", "Spike removal, Savitzky--Golay smoothing, ALS baseline correction, second derivative and SNV."],
    ]
    chunks.append(longtable("Raw and p1--p5 spectral input matrices and preprocessing steps.", "tab:s1_preprocessing_inputs", ["Input", "Preprocessing steps"], s1_rows, ["1.5cm", "13.2cm"], r"\small"))

    s2 = pd.read_csv(TABLE_DIR / "T1_peak_assignment.csv")
    s2_rows = [
        [tex_escape(r["Analyte"]), str(int(r["Raman shift (cm^-1)"])), tex_escape(r["Assignment"]), tex_escape(r["Manuscript role"])]
        for _, r in s2.iterrows()
    ]
    chunks.append(longtable("Major SERS peak assignments and manuscript roles.", "tab:s2_peak_assignment", ["Analyte", "Shift", "Assignment", "Role"], s2_rows, ["1.7cm", "1.3cm", "7.3cm", "4.2cm"], r"\scriptsize"))

    bench = pd.read_csv(TABLE_DIR / "ST2_full_benchmark_matrix.csv")
    bench["task_order"] = bench["Task"].map({task: i for i, task in enumerate(TASK_ORDER)})
    bench["model_order"] = bench["Model"].map({model: i for i, model in enumerate(MODEL_ORDER)})
    bench["prep_order"] = bench["Preprocess"].map({prep: i for i, prep in enumerate(PREPROCESS_ORDER)})
    bench = bench.sort_values(["task_order", "model_order", "prep_order"])
    s3_rows = []
    for _, r in bench.iterrows():
        s3_rows.append([
            tex_escape(TASK_LABEL.get(r["Task"], r["Task"])),
            tex_escape(r["Model"]),
            tex_escape(r["Preprocess"]),
            fmt_metric(r["F1_mean"], r["F1_std"]),
            fmt_metric(r["BA_mean"], r["BA_std"]),
            fmt_metric(r["Acc_mean"], r["Acc_std"]),
            fmt_float(r["FitSeconds_mean"], 2),
        ])
    chunks.append(longtable("Full model-by-preprocessing benchmark matrix.", "tab:s3_full_benchmark", ["Task", "Model", "Input", "Macro-F1", "BA", "Accuracy", "Fit s/fold"], s3_rows, ["2.35cm", "2.4cm", "1.0cm", "2.35cm", "2.25cm", "2.25cm", "1.45cm"], r"\tiny"))

    shap = pd.read_csv(TABLE_DIR / "ST1_shap_assignment_audit.csv")
    s4_rows = []
    for _, r in shap.iterrows():
        s4_rows.append([
            tex_escape(TASK_LABEL.get(r["task"], r["task"])),
            str(int(r["cluster_id"])),
            fmt_float(r["center_wavenumber"], 1),
            fmt_float(r["width"], 1),
            fmt_float(r["shap_mass_pct"], 2),
            tex_escape(r["assignment_level"]),
            peak_text(r["matched_literature_peak"]),
            peak_text(r["assignment_note"]),
        ])
    chunks.append(longtable("Full SHAP peak-cluster assignment audit.", "tab:s4_shap_assignment", ["Task", "Cluster", "Center", "Width", "SHAP %", "Level", "Matched peak", "Assignment note"], s4_rows, ["2.1cm", "1.0cm", "1.25cm", "1.0cm", "1.1cm", "2.0cm", "2.2cm", "3.8cm"], r"\tiny"))

    fs = pd.read_csv(TABLE_DIR / "ST3_shap_feature_selection_curve.csv")
    fs = fs.sort_values(["task", "retention_pct"])
    s5_rows = []
    for _, r in fs.iterrows():
        s5_rows.append([
            tex_escape(TASK_LABEL.get(r["task"], r["task"])),
            str(int(r["retention_pct"])),
            str(int(r["n_features"])),
            fmt_metric(r["macro_f1_mean"], r["macro_f1_std"]),
        ])
    chunks.append(longtable("Complete SHAP-guided feature-retention scan.", "tab:s5_feature_selection_curve", ["Task", "Retained %", "Features", "Macro-F1"], s5_rows, ["4.0cm", "2.4cm", "2.4cm", "4.2cm"], r"\scriptsize"))

    soil = pd.read_csv(TABLE_DIR / "T3_soil_screening_metrics.csv")
    s6_rows = [[tex_escape(v) for v in row] for row in soil.astype(str).values.tolist()]
    chunks.append(longtable("Performance of three-target presence screening in the spiked soil matrix.", "tab:s6_soil_screening_metrics", list(soil.columns), s6_rows, ["2.0cm", "1.35cm", "1.35cm", "2.1cm", "2.1cm", "2.4cm", "1.7cm"], r"\footnotesize"))

    perm = pd.read_csv(TABLE_DIR / "ST4_soil_permutation_summary.csv")
    s7_rows = []
    for _, r in perm.iterrows():
        pval = "$<10^{-4}$" if float(r["p_value"]) < 1e-4 else fmt_float(r["p_value"], 4)
        s7_rows.append([
            tex_escape(TASK_LABEL.get(r["task"], r["task"])),
            tex_escape(r["metric"]),
            fmt_float(r["observed_metric"], 3),
            fmt_float(r["null_mean"], 3),
            fmt_float(r["null_std"], 3),
            str(int(r["n_permutations"])),
            pval,
        ])
    chunks.append(longtable("Soil presence-screening permutation test summary.", "tab:s7_soil_permutation", ["Task", "Metric", "Observed", "Null mean", "Null SD", "Permutations", "p"], s7_rows, ["3.0cm", "1.7cm", "1.5cm", "1.7cm", "1.6cm", "2.0cm", "1.6cm"], r"\footnotesize"))

    rsd = pd.read_csv(TABLE_DIR / "ST5_representative_peak_rsd.csv")
    s8_rows = []
    for _, r in rsd.iterrows():
        s8_rows.append([
            tex_escape(r["Analyte"]),
            str(int(r["Marker peak (cm^-1)"])),
            combo(r["Concentration"], r["Preprocess"], f"n={int(r['n spectra'])}"),
            tex_escape(r["Metric"]),
            combo(f"mean={r['Mean band area']}", f"RSD={r['RSD (%)']}%"),
            tex_escape(r["Figure role"]),
        ])
    chunks.append(longtable("Representative same-concentration marker-band repeatability used for Fig. 2.", "tab:s8_representative_rsd", ["Analyte", "Marker", "Sample", "Metric", "Value", "Role"], s8_rows, ["1.4cm", "1.25cm", "2.3cm", "3.3cm", "2.2cm", "4.1cm"], r"\scriptsize"))

    lit = pd.read_csv(TABLE_DIR / "ST6_representative_literature_comparison.csv")
    s9_rows = []
    for _, r in lit.iterrows():
        s9_rows.append([
            tex_escape(r["Reference"]),
            combo(r["Research system"], r["Sample / matrix"]),
            combo(r["Main method"], r["Competition / mixture interference"]),
            combo(f"Multicomponent: {r['Multicomponent']}", f"XAI/SHAP: {r['XAI / SHAP']}", f"Soil: {r['Soil validation']}"),
            tex_escape(r["Main difference from this work"]),
        ])
    chunks.append(longtable("Representative literature comparison for SERS mixture analysis, soil SERS detection and spectral XAI.", "tab:s9_literature_comparison", ["Reference", "System / matrix", "Method / focus", "Scope", "Difference from this work"], s9_rows, ["2.55cm", "3.0cm", "3.4cm", "2.45cm", "4.05cm"], r"\tiny"))

    content = "\n".join(chunks)
    content = content.replace(r"cm\textasciicircum{}-1", r"cm$^{-1}$")
    content = content.replace(r"(+/-8 cm$^{-1}$)", r"($\pm$8 cm$^{-1}$)")
    out.write_text(content, encoding="utf-8")


def main() -> None:
    apply_style()
    plot_s1_full_benchmark()
    plot_s2_augmentation_ablation()
    plot_s3_preprocessing_effects()
    plot_s4_embedding()
    plot_s5_extended_rsd()
    write_supplementary_tex()
    print(f"Wrote supplementary figures to {LATEX_FIG}")
    print(f"Wrote supplementary LaTeX to {LATEX_DIR / 'supplementary_materials.tex'}")


if __name__ == "__main__":
    main()
