#!/usr/bin/env python3
"""Run paper-main TreeSHAP analysis for the locked ExtraTrees model."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    PAPER_MAIN_SHAP_DIR,
    PROCESSED_DIR,
    RANDOM_SPLIT_FILE,
    SEED,
    SPLITS_DIR,
    TASKS,
)
from src.dataset import load_preprocessed_variants  # noqa: E402
from src.models import MODEL_REGISTRY  # noqa: E402


DEFAULT_TASK_IDS = [
    'P1_thiram_presence',
    'P2_mg_presence',
    'P3_mba_presence',
    'G2_mg_molar_grade',
    'G3_mba_molar_grade',
]

LITERATURE_PEAKS = {
    'Thiram': [1380.0, 1145.0, 930.0, 1510.0, 560.0],
    'MG': [1616.0, 1172.0, 1394.0, 1220.0],
    'MBA': [1078.0, 1590.0, 1490.0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run ExtraTrees/p1 TreeSHAP peak-cluster analysis.')
    parser.add_argument('--model', default='ExtraTrees')
    parser.add_argument('--variant', default='p1')
    parser.add_argument('--tasks', nargs='+', default=DEFAULT_TASK_IDS)
    parser.add_argument('--top-n', type=int, default=20)
    parser.add_argument('--cluster-window', type=float, default=15.0)
    parser.add_argument('--match-window', type=float, default=15.0)
    parser.add_argument('--output-dir', type=Path, default=PAPER_MAIN_SHAP_DIR)
    parser.add_argument('--save-full', action='store_true', help='Save compressed per-task SHAP arrays.')
    parser.add_argument('--match-only', action='store_true', help='Refresh peak matching from existing top-N table.')
    return parser.parse_args()


def normalize_bool_columns(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    for col in ['has_thiram', 'has_mg', 'has_mba']:
        if col in meta.columns:
            meta[col] = meta[col].astype(str).str.strip().str.lower().map(
                {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
            ).fillna(0).astype(int)
    return meta


def task_subset_mask(meta: pd.DataFrame, task: dict) -> np.ndarray:
    mask = np.ones(len(meta), dtype=bool)
    filter_col = task.get('filter_col')
    if filter_col is not None:
        filter_value = task.get('filter_value', 1)
        mask &= meta[filter_col].astype(int).values == int(filter_value)
    filter_query = task.get('filter_query')
    if filter_query:
        mask &= meta.eval(filter_query).astype(bool).values
    return mask


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return 'unknown'


def package_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, '__version__', 'unknown'))
    except Exception:
        return 'not-installed'


def normalize_shap_array(values, n_samples: int, n_features: int) -> np.ndarray:
    """Return SHAP values as (n_samples, n_features, n_classes_or_outputs)."""
    if isinstance(values, list):
        arr = np.stack([np.asarray(v) for v in values], axis=-1)
    else:
        arr = np.asarray(values)

    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim == 3:
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            pass
        elif arr.shape[1] == n_samples and arr.shape[2] == n_features:
            arr = np.transpose(arr, (1, 2, 0))
        else:
            raise ValueError(f'Unexpected SHAP array shape: {arr.shape}')
    else:
        raise ValueError(f'Unexpected SHAP array ndim: {arr.ndim}, shape={arr.shape}')
    return arr


def match_literature_peaks(center: float, match_window: float) -> str:
    matches = []
    for substance, peaks in LITERATURE_PEAKS.items():
        for peak in peaks:
            delta = center - peak
            if abs(delta) <= match_window:
                matches.append(f'{substance}_{peak:.0f}({delta:+.1f})')
    return '; '.join(matches)


def match_cluster_literature_peaks(center: float, wavenumbers: np.ndarray, match_window: float) -> str:
    """Match a cluster if either its center or any member feature falls within the peak window."""
    matches = []
    candidates = np.concatenate([[center], np.asarray(wavenumbers, dtype=float)])
    for substance, peaks in LITERATURE_PEAKS.items():
        for peak in peaks:
            deltas = candidates - peak
            best_idx = int(np.argmin(np.abs(deltas)))
            best_delta = float(deltas[best_idx])
            if abs(best_delta) <= match_window:
                center_delta = center - peak
                matches.append(
                    f'{substance}_{peak:.0f}(center={center_delta:+.1f}, nearest={best_delta:+.1f})'
                )
    return '; '.join(matches)


def cluster_top_features(top_df: pd.DataFrame, cluster_window: float, match_window: float) -> pd.DataFrame:
    rows = []
    for task_id, task_df in top_df.groupby('task', sort=False):
        task_df = task_df.sort_values('wavenumber').reset_index(drop=True)
        current = []
        cluster_id = 1
        for _, row in task_df.iterrows():
            if not current:
                current = [row]
                continue
            if abs(float(row['wavenumber']) - float(current[-1]['wavenumber'])) <= cluster_window:
                current.append(row)
            else:
                rows.append(summarize_cluster(task_id, cluster_id, current, match_window))
                cluster_id += 1
                current = [row]
        if current:
            rows.append(summarize_cluster(task_id, cluster_id, current, match_window))
    return pd.DataFrame(rows)


def summarize_cluster(task_id: str, cluster_id: int, cluster_rows: list[pd.Series], match_window: float) -> dict:
    frame = pd.DataFrame(cluster_rows)
    weights = frame['mean_abs_shap'].abs().values
    if np.all(weights == 0):
        center = float(frame['wavenumber'].mean())
    else:
        center = float(np.average(frame['wavenumber'], weights=weights))
    width = float(frame['wavenumber'].max() - frame['wavenumber'].min())
    return {
        'task': task_id,
        'task_name': frame['task_name'].iloc[0],
        'cluster_id': cluster_id,
        'center_wavenumber': center,
        'min_wavenumber': float(frame['wavenumber'].min()),
        'max_wavenumber': float(frame['wavenumber'].max()),
        'width': width,
        'n_features': int(len(frame)),
        'rank_min': int(frame['rank'].min()),
        'rank_list': ';'.join(str(int(v)) for v in sorted(frame['rank'].values)),
        'feature_wavenumbers': ';'.join(f'{float(v):.1f}' for v in sorted(frame['wavenumber'].values)),
        'total_shap': float(frame['mean_abs_shap'].sum()),
        'matched_literature_peak': match_cluster_literature_peaks(
            center,
            frame['wavenumber'].values,
            match_window,
        ),
    }


def main() -> None:
    args = parse_args()
    run_prefix = 'paper_shap_match_only' if args.match_only else f'paper_shap_{args.model.lower()}'
    run_id = f"{run_prefix}_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'run_metadata').mkdir(parents=True, exist_ok=True)

    if args.match_only:
        top_path = out_dir / 'shap_top20_per_task.csv'
        if not top_path.exists():
            raise FileNotFoundError(f'Missing existing SHAP top-N table: {top_path}')
        top_df = pd.read_csv(top_path)
        top_df['matched_literature_peak'] = [
            match_literature_peaks(float(wave), args.match_window)
            for wave in top_df['wavenumber'].values
        ]
        top_df.to_csv(top_path, index=False, encoding='utf-8-sig')
        cluster_df = cluster_top_features(top_df, args.cluster_window, args.match_window)
        cluster_df.to_csv(out_dir / 'shap_cluster_summary.csv', index=False, encoding='utf-8-sig')
        metadata = {
            'run_id': run_id,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'git_commit': get_git_commit(),
            'command': ' '.join(sys.argv),
            'python': sys.version,
            'platform': platform.platform(),
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'model': args.model,
            'variant': args.variant,
            'tasks': args.tasks,
            'top_n': args.top_n,
            'cluster_window': args.cluster_window,
            'match_window': args.match_window,
            'mode': 'match_only',
            'literature_peaks': LITERATURE_PEAKS,
            'seed': SEED,
        }
        with open(out_dir / 'run_metadata' / f'{run_id}.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f'RUN_ID={run_id}')
        print('Match-only cluster output:', out_dir / 'shap_cluster_summary.csv')
        return

    try:
        import shap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'shap' package is required for TreeSHAP. Install requirements.txt before running this script."
        ) from exc

    if args.model not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{args.model}'. Available: {sorted(MODEL_REGISTRY)}")

    split_path = SPLITS_DIR / RANDOM_SPLIT_FILE
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    meta = normalize_bool_columns(pd.read_csv(split_path))
    wn, X_by_variant = load_preprocessed_variants([args.variant])
    X = X_by_variant[args.variant]
    if len(meta) != len(X):
        raise ValueError(f'Metadata/X length mismatch: {len(meta)} vs {len(X)}')

    selected_tasks = [task for task in TASKS if task['id'] in set(args.tasks)]
    missing = [task_id for task_id in args.tasks if task_id not in {task['id'] for task in selected_tasks}]
    if missing:
        raise KeyError(f'Unknown task ids: {missing}')

    print(f'RUN_ID={run_id}')
    print(f'Output: {out_dir}')
    print(f'Model: {args.model}')
    print(f'Variant: {args.variant}')
    print(f'Tasks: {[task["id"] for task in selected_tasks]}')
    print(f'ExtraTrees recipe: {MODEL_REGISTRY[args.model]()}')

    top_rows = []
    full_mean_rows = []
    full_arrays = {}
    timings = []

    for task in selected_tasks:
        task_start = perf_counter()
        mask = task_subset_mask(meta, task)
        task_meta = meta.loc[mask].reset_index(drop=True)
        task_X = X[mask]
        y = task_meta[task['col']].values.astype(int)
        if len(np.unique(y)) < 2:
            print(f"Skipping {task['id']}: only one class present after filtering.")
            continue

        clf = MODEL_REGISTRY[args.model]()
        fit_start = perf_counter()
        clf.fit(task_X, y)
        fit_seconds = perf_counter() - fit_start

        shap_start = perf_counter()
        explainer = shap.TreeExplainer(clf)
        raw_values = explainer.shap_values(task_X, check_additivity=False)
        shap_values = normalize_shap_array(raw_values, n_samples=task_X.shape[0], n_features=task_X.shape[1])
        shap_seconds = perf_counter() - shap_start

        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        order = np.argsort(mean_abs)[::-1]
        for rank, feature_idx in enumerate(order[:args.top_n], start=1):
            wave = float(wn[feature_idx])
            top_rows.append({
                'task': task['id'],
                'task_name': task['name'],
                'rank': rank,
                'feature_index': int(feature_idx),
                'wavenumber': wave,
                'mean_abs_shap': float(mean_abs[feature_idx]),
                'matched_literature_peak': match_literature_peaks(wave, args.match_window),
            })

        for feature_idx in order:
            full_mean_rows.append({
                'task': task['id'],
                'task_name': task['name'],
                'feature_index': int(feature_idx),
                'wavenumber': float(wn[feature_idx]),
                'mean_abs_shap': float(mean_abs[feature_idx]),
            })

        if args.save_full:
            full_arrays[task['id']] = shap_values.astype(np.float32)

        total_seconds = perf_counter() - task_start
        timings.append({
            'run_id': run_id,
            'task': task['id'],
            'n_samples': int(task_X.shape[0]),
            'n_features': int(task_X.shape[1]),
            'classes': ';'.join(str(v) for v in sorted(np.unique(y).tolist())),
            'fit_seconds': fit_seconds,
            'shap_seconds': shap_seconds,
            'total_seconds': total_seconds,
        })
        print(
            f"  {task['id']}: samples={task_X.shape[0]}, "
            f"fit={fit_seconds:.2f}s, shap={shap_seconds:.2f}s"
        )

    top_df = pd.DataFrame(top_rows).sort_values(['task', 'rank'])
    top_df.to_csv(out_dir / 'shap_top20_per_task.csv', index=False, encoding='utf-8-sig')

    cluster_df = cluster_top_features(top_df, args.cluster_window, args.match_window)
    cluster_df.to_csv(out_dir / 'shap_cluster_summary.csv', index=False, encoding='utf-8-sig')

    full_mean_df = pd.DataFrame(full_mean_rows).sort_values(['task', 'mean_abs_shap'], ascending=[True, False])
    full_mean_df.to_csv(out_dir / 'shap_mean_abs_by_wavenumber.csv', index=False, encoding='utf-8-sig')

    pd.DataFrame(timings).to_csv(out_dir / 'shap_run_timings.csv', index=False, encoding='utf-8-sig')

    if args.save_full and full_arrays:
        np.savez_compressed(out_dir / 'shap_values_full.npz', **full_arrays)

    metadata = {
        'run_id': run_id,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'git_commit': get_git_commit(),
        'command': ' '.join(sys.argv),
        'python': sys.version,
        'platform': platform.platform(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'sklearn': package_version('sklearn'),
        'shap': package_version('shap'),
        'model': args.model,
        'model_recipe': repr(MODEL_REGISTRY[args.model]()),
        'variant': args.variant,
        'tasks': [task['id'] for task in selected_tasks],
        'top_n': args.top_n,
        'cluster_window': args.cluster_window,
        'match_window': args.match_window,
        'wavenumber_path': str(PROCESSED_DIR / 'wavenumber.npy'),
        'seed': SEED,
    }
    with open(out_dir / 'run_metadata' / f'{run_id}.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print('\nTop feature output:', out_dir / 'shap_top20_per_task.csv')
    print('Cluster output:', out_dir / 'shap_cluster_summary.csv')
    print('Mean full output:', out_dir / 'shap_mean_abs_by_wavenumber.csv')


if __name__ == '__main__':
    main()
