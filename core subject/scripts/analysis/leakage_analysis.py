"""
Leakage benchmark + reporting entry.
Compare Random Split vs StratifiedGroupKFold to quantify split leakage effects.
Runs the representative leakage benchmark and, by default, also refreshes the
comparison figure and summary table.

Outputs:
  leakage_summary_{tag}.csv
  leakage_detail_{tag}.csv
  leakage_analysis_tab6.csv
  figures/.../fig_8_leakage_comparison.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import (
    ACTIVE_SPLIT_FILE,
    DATA_VERSION,
    FIG_MODELS,
    MAINLINE_LEAKAGE_DIR,
    PREPROCESS_TAGS,
    PROCESSED_DIR,
    SPLITS_DIR,
    TASK_PROFILE,
    TASKS,
    TASKS_SUPPLEMENTARY,
)
from src.result_index import get_active_grouped_result_dir
from src.train_eval import run_cv_evaluation, aggregate_results
from src.dataset import load_preprocessed_variants

# Leakage analysis intentionally stays on a smaller representative pool.
LEAKAGE_MODELS = ['RF', 'SVM', 'PLS-DA', 'Spectrum-KAN', '1D-CNN']
VARIANTS = list(PREPROCESS_TAGS)

ALL_TASKS = TASKS + TASKS_SUPPLEMENTARY
TASK_ORDER = [task['id'] for task in ALL_TASKS]

TASK_SHORT = {
    'P1_thiram_presence': 'Thiram\nPres',
    'P2_mg_presence': 'MG\nPres',
    'P3_mba_presence': 'MBA\nPres',
    'G1_thiram_molar_grade': 'Thiram\nGrade',
    'G2_mg_molar_grade': 'MG\nGrade',
    'G3_mba_molar_grade': 'MBA\nGrade',
    'S1_mixture_order': 'Mixture\nOrder',
}

LOG_PATH = MAINLINE_LEAKAGE_DIR / 'leakage_log.txt'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run leakage benchmark and/or regenerate its report artifacts.')
    parser.add_argument(
        '--skip-benchmark',
        action='store_true',
        help='Only regenerate the leakage figure/table from existing leakage_summary_*.csv files.',
    )
    parser.add_argument(
        '--skip-report',
        action='store_true',
        help='Only run the leakage benchmark and skip figure/table regeneration.',
    )
    return parser.parse_args()


def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    MAINLINE_LEAKAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def load_data_from_cache():
    """Load preprocessed data directly from .npy and splits CSV (no raw data needed)."""
    split_path = SPLITS_DIR / ACTIVE_SPLIT_FILE
    meta = pd.read_csv(split_path)
    
    wn, X_by_variant = load_preprocessed_variants(VARIANTS)
    return meta, wn, X_by_variant


def run_benchmark() -> None:
    t0 = time.time()
    MAINLINE_LEAKAGE_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("Leakage Experiment — Random Split vs GroupKFold")
    log(f"Data version: {DATA_VERSION}")
    log(f"Task profile: {TASK_PROFILE}")
    log(f"Split file: {ACTIVE_SPLIT_FILE}")
    log(f"Models: {LEAKAGE_MODELS}")
    log(f"Tasks: {len(ALL_TASKS)} ({len(TASKS)} main + {len(TASKS_SUPPLEMENTARY)} supplementary)")
    log(f"Preprocessing: {', '.join(VARIANTS)}")
    log(f"Total experiments: {len(LEAKAGE_MODELS)} × {len(ALL_TASKS)} × 5 = {len(LEAKAGE_MODELS) * len(ALL_TASKS) * 5}")
    log("=" * 60)

    log("Loading preprocessed data from .npy files...")
    meta, wn, X_dict = load_data_from_cache()
    log(f"  Loaded {len(meta)} spectra, {next(iter(X_dict.values())).shape[1]} wavenumber points")

    for tag, X in X_dict.items():
        t1 = time.time()
        log(f"\n--- Preprocessing: {tag} ---")

        res_df = run_cv_evaluation(
            meta, X,
            model_names=LEAKAGE_MODELS,
            preprocess_tag=tag,
            wn=wn if tag == 'raw' else None,
            tasks_override=ALL_TASKS,
            use_random_split=True,
        )
        agg_df = aggregate_results(res_df)

        detail_path = MAINLINE_LEAKAGE_DIR / f'leakage_detail_{tag}.csv'
        res_df.to_csv(detail_path, index=False, encoding='utf-8-sig')

        summary_path = MAINLINE_LEAKAGE_DIR / f'leakage_summary_{tag}.csv'
        agg_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

        elapsed = time.time() - t1
        log(f"  {tag} done in {elapsed:.1f}s — {len(agg_df)} summary rows saved")

        for _, row in agg_df.iterrows():
            log(f"    {row['Model']:15s} | {row['TaskName']:25s} | F1={row['F1_mean']:.4f}±{row['F1_std']:.4f}")

    total = time.time() - t0
    log(f"\n{'=' * 60}")
    log(f"LEAKAGE COMPLETE — Total time: {total:.1f}s ({total/60:.1f} min)")
    log(f"{'=' * 60}")


def load_report_data() -> pd.DataFrame:
    leakage_dfs = []
    for variant in VARIANTS:
        path = MAINLINE_LEAKAGE_DIR / f'leakage_summary_{variant}.csv'
        if path.exists():
            leakage_dfs.append(pd.read_csv(path))
    if not leakage_dfs:
        raise FileNotFoundError(
            f'No leakage_summary_*.csv files found under {MAINLINE_LEAKAGE_DIR}. Run this script without --skip-benchmark first.'
        )

    leakage = pd.concat(leakage_dfs, ignore_index=True)
    result_dir = get_active_grouped_result_dir()
    if result_dir is None:
        raise FileNotFoundError('No grouped mainline result directory found. Run grouped candidate screening or locked grouped results first.')

    benchmark_matrix = result_dir / 'benchmark_full_matrix.csv'
    if benchmark_matrix.exists():
        grouped = pd.read_csv(benchmark_matrix)
    else:
        grouped_dfs = []
        for variant in VARIANTS:
            path = result_dir / f'cv_results_summary_{variant}.csv'
            if path.exists():
                grouped_dfs.append(pd.read_csv(path))
        if not grouped_dfs:
            raise FileNotFoundError(
                f'No grouped summary files found under {result_dir}.'
            )
        grouped = pd.concat(grouped_dfs, ignore_index=True)

    leakage = leakage[leakage['Model'].isin(LEAKAGE_MODELS)]
    grouped = grouped[grouped['Model'].isin(LEAKAGE_MODELS)]

    leakage_best = leakage.groupby(['Model', 'Task'])['F1_mean'].max().reset_index()
    leakage_best.rename(columns={'F1_mean': 'F1_Random'}, inplace=True)

    grouped_best = grouped.groupby(['Model', 'Task'])['F1_mean'].max().reset_index()
    grouped_best.rename(columns={'F1_mean': 'F1_Group'}, inplace=True)

    merged = pd.merge(grouped_best, leakage_best, on=['Model', 'Task'])
    merged['Delta'] = merged['F1_Random'] - merged['F1_Group']
    return merged


def make_leakage_figure(merged: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})

    ax = axes[0]
    model_avg = merged.groupby('Model')[['F1_Group', 'F1_Random', 'Delta']].mean()
    x = np.arange(len(LEAKAGE_MODELS))
    width = 0.35

    ax.bar(
        x - width / 2,
        [model_avg.loc[m, 'F1_Group'] for m in LEAKAGE_MODELS],
        width,
        label='StratifiedGroupKFold',
        color='#4472C4',
        edgecolor='black',
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        [model_avg.loc[m, 'F1_Random'] for m in LEAKAGE_MODELS],
        width,
        label='Random Split',
        color='#E86452',
        edgecolor='black',
        linewidth=0.5,
    )

    for i, model_name in enumerate(LEAKAGE_MODELS):
        delta = model_avg.loc[model_name, 'Delta']
        y_max = model_avg.loc[model_name, 'F1_Random']
        ax.annotate(
            f'+{delta:.1%}',
            xy=(x[i] + width / 2, y_max),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='red',
        )

    ax.set_ylabel(f'Macro F1 (avg across {len(TASK_ORDER)} tasks)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(LEAKAGE_MODELS, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_title('(a) Average F1: grouped vs random split', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    pivot = merged.pivot(index='Task', columns='Model', values='Delta')
    pivot = pivot.reindex(index=TASK_ORDER, columns=LEAKAGE_MODELS)
    im = ax2.imshow(pivot.values, cmap='Reds', aspect='auto', vmin=0, vmax=0.5)
    ax2.set_xticks(range(len(LEAKAGE_MODELS)))
    ax2.set_xticklabels(LEAKAGE_MODELS, fontsize=9, rotation=45, ha='right')
    ax2.set_yticks(range(len(TASK_ORDER)))
    ax2.set_yticklabels([TASK_SHORT.get(task_id, task_id) for task_id in TASK_ORDER], fontsize=8)

    for i in range(len(TASK_ORDER)):
        for j in range(len(LEAKAGE_MODELS)):
            value = pivot.values[i, j]
            color = 'white' if value > 0.3 else 'black'
            ax2.text(j, i, f'{value:+.2f}', ha='center', va='center', fontsize=7, color=color)

    ax2.set_title('(b) F1 inflation map', fontsize=13, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('ΔF1', fontsize=10)

    plt.tight_layout()
    FIG_MODELS.mkdir(parents=True, exist_ok=True)
    out_path = FIG_MODELS / 'fig_8_leakage_comparison.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def make_leakage_table(merged: pd.DataFrame) -> None:
    summary = merged.groupby('Model').agg(
        F1_Group_mean=('F1_Group', 'mean'),
        F1_Random_mean=('F1_Random', 'mean'),
        Delta_mean=('Delta', 'mean'),
        Delta_max=('Delta', 'max'),
        N_gt5pct=('Delta', lambda x: (x > 0.05).sum()),
        N_gt10pct=('Delta', lambda x: (x > 0.10).sum()),
        N_gt15pct=('Delta', lambda x: (x > 0.15).sum()),
    ).reindex(LEAKAGE_MODELS)

    print('\n=== Leakage Summary Table ===')
    print(summary.to_string())

    MAINLINE_LEAKAGE_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = MAINLINE_LEAKAGE_DIR / 'leakage_analysis_tab6.csv'
    merged.to_csv(out_csv, index=False)
    print(f'\nSaved: {out_csv}')
    print(f'\nOVERALL: avg_delta={merged["Delta"].mean():.3f}, max_delta={merged["Delta"].max():.3f}')
    print(f'Cases >5%: {(merged["Delta"] > 0.05).sum()}/{len(merged)}')
    print(f'Cases >10%: {(merged["Delta"] > 0.10).sum()}/{len(merged)}')
    print(f'Cases >15%: {(merged["Delta"] > 0.15).sum()}/{len(merged)}')


def run_report() -> None:
    merged_df = load_report_data()
    make_leakage_figure(merged_df)
    make_leakage_table(merged_df)


def main() -> None:
    args = parse_args()
    if not args.skip_benchmark:
        run_benchmark()
    if not args.skip_report:
        print('\n' + '=' * 70)
        print('Leakage Post-Processing')
        print('=' * 70)
        run_report()
        print('\nDone!')


if __name__ == '__main__':
    main()
