#!/usr/bin/env python3
"""SHAP-guided spectral masking and progressive elimination for ExtraTrees/p1."""
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
from sklearn.metrics import f1_score

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
from src.models import MODEL_REGISTRY  # noqa: E402


DEFAULT_TASK_IDS = [
    'P1_thiram_presence',
    'P2_mg_presence',
    'P3_mba_presence',
    'G1_thiram_molar_grade',
    'G2_mg_molar_grade',
    'G3_mba_molar_grade',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run SHAP-guided spectral feature masking.')
    parser.add_argument('--model', default='ExtraTrees')
    parser.add_argument('--variant', default='p1')
    parser.add_argument('--tasks', nargs='+', default=DEFAULT_TASK_IDS)
    parser.add_argument(
        '--retention-pct',
        nargs='+',
        type=int,
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    )
    parser.add_argument('--mask-report-pct', type=int, default=30)
    parser.add_argument('--output-dir', type=Path, default=None)
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


def selected_tasks(task_ids: list[str]) -> list[dict]:
    task_map = {task['id']: task for task in TASKS}
    missing = [task_id for task_id in task_ids if task_id not in task_map]
    if missing:
        raise KeyError(f'Unknown task ids: {missing}')
    return [task_map[task_id] for task_id in task_ids]


def load_inputs(variant: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    wn = np.load(PROCESSED_DIR / 'wavenumber.npy')
    X = np.load(PROCESSED_DIR / f'X_{variant}.npy')
    meta = normalize_bool_columns(pd.read_csv(SPLITS_DIR / RANDOM_SPLIT_FILE))
    shap_df = pd.read_csv(PAPER_MAIN_SHAP_DIR / 'shap_mean_abs_by_wavenumber.csv')
    if len(meta) != len(X):
        raise ValueError(f'Metadata/X length mismatch: {len(meta)} vs {len(X)}')
    if len(wn) != X.shape[1]:
        raise ValueError(f'Wavenumber/X feature mismatch: {len(wn)} vs {X.shape[1]}')
    return wn, X, meta, shap_df


def feature_count(n_features: int, retention_pct: int) -> int:
    return int(np.ceil(n_features * retention_pct / 100.0))


def task_shap_order(shap_df: pd.DataFrame, task_id: str, n_features: int) -> pd.DataFrame:
    task_df = shap_df[shap_df['task'] == task_id].copy()
    if len(task_df) != n_features:
        raise ValueError(f'{task_id} SHAP rows={len(task_df)}, expected {n_features}')
    return task_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)


def run_masked_cv(
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    selected_feature_idx: np.ndarray,
    model_name: str,
) -> tuple[float, float, list[dict]]:
    feature_mask = np.zeros(X.shape[1], dtype=bool)
    feature_mask[selected_feature_idx] = True
    X_masked = X.copy()
    X_masked[:, ~feature_mask] = 0.0

    fold_rows = []
    scores = []
    for fold_id in sorted(np.unique(fold_ids)):
        train_idx = np.where(fold_ids != fold_id)[0]
        test_idx = np.where(fold_ids == fold_id)[0]
        clf = MODEL_REGISTRY[model_name]()
        clf.fit(X_masked[train_idx], y[train_idx])
        pred = clf.predict(X_masked[test_idx]).astype(int)
        score = float(f1_score(y[test_idx], pred, average='macro', zero_division=0))
        scores.append(score)
        fold_rows.append({
            'fold_id': int(fold_id),
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
            'macro_f1': score,
        })
    return float(np.mean(scores)), float(np.std(scores, ddof=1)), fold_rows


def main() -> None:
    args = parse_args()
    if args.model not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{args.model}'. Available: {sorted(MODEL_REGISTRY)}")
    if args.variant != 'p1':
        raise ValueError('This paper-main feature-selection script is locked to p1.')
    if any(pct <= 0 or pct > 100 for pct in args.retention_pct):
        raise ValueError(f'Invalid retention percentages: {args.retention_pct}')

    out_dir = args.output_dir or (PAPER_MAIN_SHAP_DIR.parent / 'shap_feature_selection')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'run_metadata').mkdir(parents=True, exist_ok=True)
    run_id = f"shap_feature_selection_{args.model.lower()}_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start = perf_counter()

    wn, X, meta, shap_df = load_inputs(args.variant)
    tasks = selected_tasks(args.tasks)
    retention_values = sorted(dict.fromkeys(args.retention_pct))

    print(f'RUN_ID={run_id}')
    print(f'Output: {out_dir}')
    print(f'Model: {args.model}')
    print(f'Variant: {args.variant}')
    print(f'Retention: {retention_values}')
    print(f'Model recipe: {MODEL_REGISTRY[args.model]()}')

    curve_rows = []
    fold_rows_all = []
    retained_rows = []
    mask_rows = []
    mask_arrays = {}

    for task in tasks:
        task_start = perf_counter()
        subset = task_subset_mask(meta, task)
        task_meta = meta.loc[subset].reset_index(drop=True)
        task_X = X[subset]
        y = task_meta[task['col']].values.astype(int)
        fold_ids = task_meta['fold_id'].values.astype(int)
        order_df = task_shap_order(shap_df, task['id'], X.shape[1])

        for retention_pct in retention_values:
            n_keep = feature_count(X.shape[1], retention_pct)
            selected_idx = order_df.head(n_keep)['feature_index'].values.astype(int)
            mean_f1, std_f1, fold_rows = run_masked_cv(task_X, y, fold_ids, selected_idx, args.model)
            curve_rows.append({
                'run_id': run_id,
                'task': task['id'],
                'task_name': task['name'],
                'retention_pct': int(retention_pct),
                'n_features': int(n_keep),
                'macro_f1_mean': mean_f1,
                'macro_f1_std': std_f1,
            })
            for row in fold_rows:
                row.update({
                    'run_id': run_id,
                    'task': task['id'],
                    'task_name': task['name'],
                    'retention_pct': int(retention_pct),
                    'n_features': int(n_keep),
                })
                fold_rows_all.append(row)
            print(f"  {task['id']} retain={retention_pct}% n={n_keep}: F1={mean_f1:.4f}±{std_f1:.4f}")

            if retention_pct == args.mask_report_pct:
                mask = np.zeros(X.shape[1], dtype=np.int8)
                mask[selected_idx] = 1
                mask_arrays[task['id']] = mask
                selected_table = order_df.head(n_keep).copy()
                selected_table['run_id'] = run_id
                selected_table['retention_pct'] = int(retention_pct)
                selected_table['selection_rank'] = np.arange(1, n_keep + 1)
                retained_rows.extend(selected_table[[
                    'run_id',
                    'task',
                    'task_name',
                    'retention_pct',
                    'selection_rank',
                    'feature_index',
                    'wavenumber',
                    'mean_abs_shap',
                ]].to_dict('records'))
                for feature_index, wave in enumerate(wn):
                    mask_rows.append({
                        'run_id': run_id,
                        'task': task['id'],
                        'task_name': task['name'],
                        'retention_pct': int(retention_pct),
                        'feature_index': int(feature_index),
                        'wavenumber': float(wave),
                        'retained': int(mask[feature_index]),
                    })

        print(f"  {task['id']} done in {perf_counter() - task_start:.1f}s")

    pd.DataFrame(curve_rows).to_csv(out_dir / 'shap_feature_elimination_curve.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(fold_rows_all).to_csv(out_dir / 'shap_feature_elimination_folds.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(retained_rows).to_csv(out_dir / 'shap_retained_wavenumbers.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(mask_rows).to_csv(out_dir / 'shap_retention30_masks.csv', index=False, encoding='utf-8-sig')
    if mask_arrays:
        np.savez_compressed(out_dir / 'shap_retention30_masks.npz', **mask_arrays)

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
        'model': args.model,
        'model_recipe': repr(MODEL_REGISTRY[args.model]()),
        'variant': args.variant,
        'tasks': [task['id'] for task in tasks],
        'retention_pct': retention_values,
        'mask_report_pct': int(args.mask_report_pct),
        'shap_source': str(PAPER_MAIN_SHAP_DIR / 'shap_mean_abs_by_wavenumber.csv'),
        'split_source': str(SPLITS_DIR / RANDOM_SPLIT_FILE),
        'seed': SEED,
        'elapsed_seconds': perf_counter() - start,
    }
    with open(out_dir / 'run_metadata' / f'{run_id}.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print('Curve output:', out_dir / 'shap_feature_elimination_curve.csv')
    print('Retained wavenumbers:', out_dir / 'shap_retained_wavenumbers.csv')
    print('Mask output:', out_dir / 'shap_retention30_masks.npz')


if __name__ == '__main__':
    main()
