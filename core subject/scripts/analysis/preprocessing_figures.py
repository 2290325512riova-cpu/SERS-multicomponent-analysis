"""
Preprocessing heatmap + embedding visualizations for supplementary figures.

Run locally:
    python scripts/analysis/preprocessing_figures.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIG_EDA, FIG_MODELS, MODELS_DIR, PROCESSED_DIR, PREPROCESS_SHORT_LABELS, PREPROCESS_TAGS
from src.result_index import get_active_screening_dir

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def make_preprocessing_heatmap():
    print('=' * 60)
    print('Preprocessing Sensitivity Heatmap')
    print('=' * 60)

    tags = list(PREPROCESS_TAGS)
    dfs = []
    screening_dir = get_active_screening_dir()
    benchmark_matrix = screening_dir / 'benchmark_full_matrix.csv' if screening_dir is not None else None
    if benchmark_matrix is not None and benchmark_matrix.exists():
        df = pd.read_csv(benchmark_matrix)
        dfs.append(df)
        print(f'  Loaded {benchmark_matrix.name}: {len(df)} full-screening rows')
    else:
        for tag in tags:
            base_dir = screening_dir if screening_dir is not None else MODELS_DIR
            path = base_dir / f'cv_results_summary_{tag}.csv'
            if path.exists():
                df = pd.read_csv(path)
                dfs.append(df)
                print(f'  Loaded {path.name}: {len(df)} rows')
            else:
                print(f'  WARNING: {path.name} not found, skipping')

    if not dfs:
        print('  No summary files found. Abort.')
        return

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df[~all_df['Model'].str.startswith('MT-')]

    pivot = all_df.groupby(['Model', 'Preprocess'])['F1_mean'].mean().reset_index()
    heatmap_data = pivot.pivot(index='Model', columns='Preprocess', values='F1_mean')
    col_order = [col for col in PREPROCESS_TAGS if col in heatmap_data.columns]
    heatmap_data = heatmap_data[col_order]

    heatmap_data.columns = [PREPROCESS_SHORT_LABELS.get(col, col) for col in heatmap_data.columns]

    heatmap_data['mean'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values('mean', ascending=True).drop(columns='mean')

    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', vmin=0.3, vmax=0.9)

    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, fontsize=9)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=8)

    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.iloc[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.7 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7.5, color=color)

    ax.set_xlabel('Preprocessing Method')
    ax.set_ylabel('Model')
    ax.set_title('Mean F1-macro across All Tasks\n(Model × Preprocessing)', fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('F1-macro')

    for i in range(len(heatmap_data.index)):
        row = heatmap_data.iloc[i]
        best_j = row.values.argmax()
        ax.add_patch(
            plt.Rectangle((best_j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='blue', linewidth=2)
        )

    FIG_MODELS.mkdir(parents=True, exist_ok=True)
    out_path = FIG_MODELS / 'fig_s5_preprocessing_heatmap.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f'\n  Saved: {out_path}')

    print('\n  Per-model best preprocessing:')
    for model in heatmap_data.index:
        row = heatmap_data.loc[model]
        best_col = row.idxmax()
        print(f'    {model:20s} -> {best_col} (F1={row[best_col]:.3f})')


def make_embedding_visualizations():
    print('\n' + '=' * 60)
    print('t-SNE / UMAP Visualization')
    print('=' * 60)

    from src.dataset import build_metadata, create_cv_splits

    meta = build_metadata()
    meta = create_cv_splits(meta)
    X_raw = np.load(PROCESSED_DIR / 'X_raw.npy')
    print(f'  Loaded {len(meta)} spectra, {X_raw.shape[1]} features')

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=50, random_state=42)
    X_pca50 = pca.fit_transform(X_scaled)
    print(f'  PCA 50-dim: explained variance = {pca.explained_variance_ratio_.sum():.3f}')

    order_labels = meta['mixture_order'].values
    order_names = {1: 'Single', 2: 'Binary', 3: 'Ternary'}
    colors_order = {1: '#2196F3', 2: '#FF9800', 3: '#E91E63'}

    families = meta['family'].values
    unique_families = sorted(meta['family'].unique())
    family_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_families)))
    family_cmap = {family_name: family_colors[i] for i, family_name in enumerate(unique_families)}
    family_map = {
        'single': 'Single',
        'binary_Thiram_MG': 'Thiram+MG',
        'binary_Thiram_MBA': 'Thiram+MBA',
        'binary_MG_MBA': 'MG+MBA',
        'ternary': 'Ternary',
    }

    print('  Computing t-SNE (perplexity=30)...')
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca')
    X_tsne = tsne.fit_transform(X_pca50)
    print(f'  t-SNE done. KL divergence = {tsne.kl_divergence_:.4f}')

    print('  Computing UMAP (n_neighbors=15)...')
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError:
        raise ModuleNotFoundError(
            'umap-learn is required for embedding figures. Install it from requirements.txt first.'
        )

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_pca50)
    print('  UMAP done.')

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    for col_idx, (X_embed, title) in enumerate([(X_tsne, 't-SNE'), (X_umap, 'UMAP')]):
        ax = axes[0, col_idx]
        for order in sorted(order_names.keys()):
            mask = order_labels == order
            ax.scatter(
                X_embed[mask, 0],
                X_embed[mask, 1],
                c=colors_order[order],
                label=order_names[order],
                s=12,
                alpha=0.6,
                edgecolors='none',
            )
        ax.set_title(f'{title} — by Mixture Complexity', fontweight='bold')
        ax.set_xlabel(f'{title}-1')
        ax.set_ylabel(f'{title}-2')
        ax.legend(fontsize=8, markerscale=2, loc='best')

    for col_idx, (X_embed, title) in enumerate([(X_tsne, 't-SNE'), (X_umap, 'UMAP')]):
        ax = axes[1, col_idx]
        for family_name in unique_families:
            mask = families == family_name
            label = family_map.get(family_name, family_name)
            ax.scatter(
                X_embed[mask, 0],
                X_embed[mask, 1],
                c=[family_cmap[family_name]],
                label=label,
                s=12,
                alpha=0.6,
                edgecolors='none',
            )
        ax.set_title(f'{title} — by Component Family', fontweight='bold')
        ax.set_xlabel(f'{title}-1')
        ax.set_ylabel(f'{title}-2')
        ax.legend(fontsize=6.5, markerscale=2, loc='best', ncol=2)

    plt.tight_layout(h_pad=2.0)

    FIG_EDA.mkdir(parents=True, exist_ok=True)
    out_path = FIG_EDA / 'fig_s7_tsne_umap.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f'\n  Saved: {out_path}')


if __name__ == '__main__':
    make_preprocessing_heatmap()
    make_embedding_visualizations()
    print('\nDone!')
