"""
Visualization: EDA plots, preprocessing comparison, model evaluation figures.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from src.config import (
    FIG_EDA, FIG_PREPROCESS, FIG_MODELS, SEED, TASKS, N_FOLDS,
    PREPROCESS_SHORT_LABELS, PREPROCESS_TAGS,
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Color palette for families
FAM_COLORS = {
    'single': '#2196F3',
    'binary_Thiram_MG': '#9C27B0',
    'binary_Thiram_MBA': '#FF9800',
    'binary_MG_MBA': '#4CAF50',
    'ternary': '#F44336',
}

# Color palette for substances
SUBST_COLORS = {'Thiram': '#FF5722', 'MG': '#4CAF50', 'MBA': '#2196F3'}


def plot_eda(meta, wn, X, X_p1):
    """Generate EDA figures."""
    FIG_EDA.mkdir(parents=True, exist_ok=True)

    # 1. Mean spectra by single-substance type
    fig, ax = plt.subplots(figsize=(14, 6))
    single = meta[meta['family'] == 'single']
    for name, col, color in [
        ('Thiram', 'has_thiram', SUBST_COLORS['Thiram']),
        ('MG', 'has_mg', SUBST_COLORS['MG']),
        ('MBA', 'has_mba', SUBST_COLORS['MBA']),
    ]:
        idx = single[single[col] == True].index.values
        if len(idx) == 0:
            continue
        m = X_p1[idx].mean(0)
        s = X_p1[idx].std(0)
        ax.plot(wn, m, label=f'{name} (n={len(idx)})', color=color, lw=1.5)
        ax.fill_between(wn, m - s, m + s, alpha=0.15, color=color)
    ax.set_xlabel('Raman Shift (cm-1)')
    ax.set_ylabel('P1 Intensity (SNV)')
    ax.set_title('Single-Component Mean Spectra ± SD')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_EDA / '1_single_mean_spectra.png', dpi=200)
    plt.close()

    # 2. Global PCA colored by family
    pca = PCA(n_components=5, random_state=SEED)
    scores = pca.fit_transform(X_p1)
    ve = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    for fam, color in FAM_COLORS.items():
        mask = meta['family'] == fam
        if mask.sum() == 0:
            continue
        ax.scatter(scores[mask, 0], scores[mask, 1], c=color, label=fam,
                   alpha=0.6, s=15, edgecolors='none')
    ax.set_xlabel(f'PC1 ({ve[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({ve[1]*100:.1f}%)')
    ax.set_title('Global PCA by Mixture Family')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_EDA / '2_global_pca_by_family.png', dpi=200)
    plt.close()

    # 3. PCA variance bar
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, 6), ve * 100, color='#2196F3', alpha=0.8)
    for i, v in enumerate(ve):
        ax.text(i + 1, v * 100 + 0.5, f'{v*100:.1f}%', ha='center')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('PCA Explained Variance (Top 5)')
    ax.set_xticks(range(1, 6))
    plt.tight_layout()
    plt.savefig(FIG_EDA / '3_pca_variance_bar.png', dpi=200)
    plt.close()

    # 4. Folder-level mean PCA (removes pseudo-replication)
    fmeans, ffams = [], []
    for fn in meta['folder_name'].unique():
        idx = meta[meta['folder_name'] == fn].index
        fmeans.append(scores[idx].mean(0)[:2])
        ffams.append(meta.loc[idx[0], 'family'])
    fmeans = np.array(fmeans)
    ffams = np.array(ffams)

    fig, ax = plt.subplots(figsize=(9, 7))
    for fam, color in FAM_COLORS.items():
        mask = ffams == fam
        if mask.sum() > 0:
            ax.scatter(fmeans[mask, 0], fmeans[mask, 1], c=color,
                       label=f'{fam} ({mask.sum()} folders)',
                       s=60, edgecolors='k', linewidths=0.5)
    ax.set_xlabel(f'PC1 ({ve[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({ve[1]*100:.1f}%)')
    ax.set_title('Folder-Level Mean PCA (1 point per folder)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_EDA / '4_folder_level_mean_pca.png', dpi=200)
    plt.close()

    # 5. Concentration-dependent spectra per substance
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (subst, col) in zip(axes, [('Thiram', 'c_thiram'), ('MG', 'c_mg'), ('MBA', 'c_mba')]):
        for conc in [4, 5, 6]:
            idx = meta[(meta[col] == conc) & (meta['mixture_order'] == 1)].index.values
            if len(idx) == 0:
                # Try all samples with this concentration
                idx = meta[meta[col] == conc].index.values
            if len(idx) == 0:
                continue
            m = X_p1[idx].mean(0)
            ax.plot(wn, m, label=f'10^-{conc} M (n={len(idx)})', lw=1.2)
        ax.set_title(f'{subst} by Concentration')
        ax.set_xlabel('Raman Shift (cm-1)')
        ax.set_ylabel('P1 Intensity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_EDA / '5_concentration_dependent_spectra.png', dpi=200)
    plt.close()

    # 6. Sample count per folder (bar chart)
    folder_counts = meta.groupby('folder_name').size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = [FAM_COLORS.get(meta[meta['folder_name'] == fn].iloc[0]['family'], '#999')
              for fn in folder_counts.index]
    ax.bar(range(len(folder_counts)), folder_counts.values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(folder_counts)))
    ax.set_xticklabels(folder_counts.index, rotation=90, fontsize=6)
    ax.set_ylabel('Number of Spectra')
    ax.set_title('Spectra per Folder')
    plt.tight_layout()
    plt.savefig(FIG_EDA / '6_spectra_per_folder.png', dpi=200)
    plt.close()

    print(f"  EDA figures saved to {FIG_EDA}")
    return pca, scores


def plot_preprocessing_comparison(wn, X_raw, X_p1=None, X_p2=None, X_p3=None, X_p4=None, sample_idx=0):
    """Compare raw vs. preprocessing pipelines for one sample."""
    FIG_PREPROCESS.mkdir(parents=True, exist_ok=True)
    if isinstance(X_raw, dict):
        X_dict = X_raw
        titles = [f"{tag}: {PREPROCESS_SHORT_LABELS.get(tag, tag)}" for tag in PREPROCESS_TAGS if tag in X_dict]
        data = [X_dict[tag] for tag in PREPROCESS_TAGS if tag in X_dict]
    else:
        titles = ['Raw (interpolated)', 'P1: Cosmic+SG+ALS+SNV',
                  'P2: SG+ALS+Deriv+SNV', 'P3: ALS+VecNorm']
        data = [X_raw, X_p1, X_p2, X_p3]
        if X_p4 is not None:
            titles.append('P4: Cosmic+SG+ALS (no norm)')
            data.append(X_p4)
    n = len(data)
    ncols = 3 if n > 4 else 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]
    for i, (ax, title, d) in enumerate(zip(axes_flat, titles, data)):
        ax.plot(wn, d[sample_idx], lw=0.8)
        ax.set_title(title)
        ax.set_xlabel('Raman Shift (cm-1)')
        ax.grid(True, alpha=0.3)
    # Hide unused subplots
    for j in range(n, nrows * ncols):
        axes_flat[j].set_visible(False)
    plt.suptitle(f'Preprocessing Comparison (Sample #{sample_idx})', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_PREPROCESS / 'preprocessing_comparison.png', dpi=200)
    plt.close()
    print(f"  Preprocessing comparison saved to {FIG_PREPROCESS}")


def plot_confusion_matrices(agg_df, predictions, meta):
    """Plot confusion matrix for the best model per task."""
    FIG_MODELS.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in TASKS if t['id'] in agg_df['Task'].values]

    for tk in tasks:
        sub = agg_df[agg_df['Task'] == tk['id']]
        if sub.empty:
            continue
        best_row = sub.loc[sub['F1_mean'].idxmax()]
        best_model = best_row['Model']
        key = (tk['id'], best_model)
        if key not in predictions:
            continue
        yt, yp = predictions[key]
        cm = confusion_matrix(yt, yp, labels=tk['classes'])

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=tk['classes'], yticklabels=tk['classes'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f"{tk['name']}\n(Best: {best_model}, "
                     f"F1={best_row['F1_mean']:.3f}±{best_row['F1_std']:.3f})")
        plt.tight_layout()
        plt.savefig(FIG_MODELS / f'cm_{tk["id"]}.png', dpi=150)
        plt.close()

    print(f"  Confusion matrices saved to {FIG_MODELS}")


def plot_model_comparison(agg_df):
    """Bar chart comparing models across all tasks (F1 score)."""
    FIG_MODELS.mkdir(parents=True, exist_ok=True)

    tasks = agg_df['Task'].unique()
    models = agg_df['Model'].unique()
    n_tasks = len(tasks)
    n_models = len(models)
    x = np.arange(n_tasks)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(12, n_tasks * 2), 6))
    for i, model in enumerate(models):
        sub = agg_df[agg_df['Model'] == model].set_index('Task')
        vals = [sub.loc[t, 'F1_mean'] if t in sub.index else 0 for t in tasks]
        errs = [sub.loc[t, 'F1_std'] if t in sub.index else 0 for t in tasks]
        ax.bar(x + i * width, vals, width, yerr=errs, label=model, alpha=0.85, capsize=3)

    task_names = []
    for t in tasks:
        row = agg_df[agg_df['Task'] == t].iloc[0]
        task_names.append(row['TaskName'])

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(task_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('Model Comparison Across Tasks (5-Fold CV)')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_MODELS / 'model_comparison_bar.png', dpi=200)
    plt.close()
    print(f"  Model comparison bar chart saved")


def plot_split_distribution(meta):
    """Visualize fold distribution of labels."""
    FIG_EDA.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, title) in zip(axes.flat, [
        ('c_thiram', 'Thiram Conc.'), ('c_mg', 'MG Conc.'),
        ('c_mba', 'MBA Conc.'), ('family', 'Mixture Family'),
    ]):
        ct = pd.crosstab(meta['fold_id'], meta[col])
        ct.plot(kind='bar', ax=ax, rot=0)
        ax.set_title(f'{title} by Fold')
        ax.set_xlabel('Fold')
        ax.legend(fontsize=7, title=col)

    plt.suptitle(f'{N_FOLDS}-Fold Split Label Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_EDA / '7_split_distribution.png', dpi=200)
    plt.close()
    print(f"  Split distribution saved")
