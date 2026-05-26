#!/usr/bin/env python3
"""Run formal soil-only CV with blank negatives and OOF permutation testing."""
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
from scipy.interpolate import interp1d
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    PAPER_MAIN_SOIL_DIR,
    PREPROCESS_TAGS,
    SEED,
    TASKS,
    WN_MAX,
    WN_MIN,
    WN_STEP,
)
from src.dataset import (  # noqa: E402
    _compute_preprocess_variant,
    build_soil_metadata,
    read_bwram_spectrum,
)
from src.models import MODEL_REGISTRY  # noqa: E402


DEFAULT_PRESENCE_TASK_IDS = [
    'P1_thiram_presence',
    'P2_mg_presence',
    'P3_mba_presence',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run soil-only 5-fold screening CV with blank negatives.')
    parser.add_argument('--model', default='ExtraTrees')
    parser.add_argument('--variant', default='p1')
    parser.add_argument('--tasks', nargs='+', default=DEFAULT_PRESENCE_TASK_IDS)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--n-permutations', type=int, default=10000)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--include-blanks', action='store_true', default=True)
    parser.add_argument('--no-include-blanks', action='store_false', dest='include_blanks')
    parser.add_argument('--rebuild-blanks', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=PAPER_MAIN_SOIL_DIR)
    return parser.parse_args()


def normalize_bool_columns(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    for col in ['has_thiram', 'has_mg', 'has_mba', 'is_blank', 'is_design_sample', 'is_background']:
        if col in meta.columns:
            meta[col] = meta[col].astype(str).str.strip().str.lower().map(
                {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
            ).fillna(0).astype(int)
    return meta


def selected_presence_tasks(task_ids: list[str]) -> list[dict]:
    task_map = {task['id']: task for task in TASKS}
    missing = [task_id for task_id in task_ids if task_id not in task_map]
    if missing:
        raise KeyError(f'Unknown task ids: {missing}')
    tasks = [task_map[task_id] for task_id in task_ids]
    non_presence = [task['id'] for task in tasks if task.get('filter_col') or task['id'].startswith('G')]
    if non_presence:
        raise ValueError(f'Soil-only CV is restricted to presence tasks, got: {non_presence}')
    return tasks


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


def build_raw_matrix(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    wn_common = np.arange(WN_MIN, WN_MAX + WN_STEP, WN_STEP)
    X_raw = np.zeros((len(meta), len(wn_common)))
    unreadable = []
    for i, (_, row) in enumerate(meta.iterrows()):
        wn, intensity = read_bwram_spectrum(row['file_path'])
        if wn is None or len(wn) < 50:
            unreadable.append(row['file_path'])
            continue
        order = np.argsort(wn)
        wn, intensity = wn[order], intensity[order]
        mask = (wn >= WN_MIN - 20) & (wn <= WN_MAX + 20)
        wn, intensity = wn[mask], intensity[mask]
        if len(wn) < 50:
            unreadable.append(row['file_path'])
            continue
        X_raw[i] = interp1d(wn, intensity, kind='linear', bounds_error=False, fill_value=0)(wn_common)
    if unreadable:
        raise ValueError(f'Unreadable spectra: {unreadable}')
    return wn_common, X_raw


def load_soil_design_data(variant: str) -> tuple[pd.DataFrame, np.ndarray]:
    metadata_path = PAPER_MAIN_SOIL_DIR / 'soil_metadata.csv'
    if not metadata_path.exists():
        meta = build_soil_metadata(include_auxiliary=False)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(metadata_path, index=False, encoding='utf-8-sig')
    else:
        meta = pd.read_csv(metadata_path)
    meta = normalize_bool_columns(meta)
    if 'sample_role' in meta.columns:
        meta = meta[meta['sample_role'].astype(str).str.lower().eq('design')].reset_index(drop=True)

    x_path = PAPER_MAIN_SOIL_DIR / f'X_soil_{variant}.npy'
    if not x_path.exists():
        raise FileNotFoundError(f'Missing soil cache for {variant}: {x_path}')
    X = np.load(x_path)
    if len(meta) != len(X):
        raise ValueError(f'Soil metadata/X length mismatch: {len(meta)} vs {len(X)}')
    return meta, X


def load_blank_data(variant: str, rebuild: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    try:
        meta_all = build_soil_metadata(include_auxiliary=True)
        blank_meta = normalize_bool_columns(meta_all[meta_all['is_blank'].astype(bool)].reset_index(drop=True))
        blank_meta.to_csv(PAPER_MAIN_SOIL_DIR / 'blank_metadata.csv', index=False, encoding='utf-8-sig')
    except FileNotFoundError:
        blank_path = PAPER_MAIN_SOIL_DIR / 'blank_metadata.csv'
        if not blank_path.exists():
            raise
        blank_meta = normalize_bool_columns(pd.read_csv(blank_path))

    raw_path = PAPER_MAIN_SOIL_DIR / 'X_blanks_raw.npy'
    variant_path = PAPER_MAIN_SOIL_DIR / f'X_blanks_{variant}.npy'
    if not rebuild and variant != 'raw' and variant_path.exists():
        X_variant = np.load(variant_path)
        if len(blank_meta) != len(X_variant):
            raise ValueError(f'Blank metadata/X length mismatch: {len(blank_meta)} vs {len(X_variant)}')
        return blank_meta, X_variant

    if rebuild or not raw_path.exists():
        _, X_raw = build_raw_matrix(blank_meta)
        np.save(raw_path, X_raw)
    else:
        X_raw = np.load(raw_path)

    if len(blank_meta) != len(X_raw):
        raise ValueError(f'Blank metadata/X length mismatch: {len(blank_meta)} vs {len(X_raw)}')

    if variant == 'raw':
        X_variant = X_raw
    elif rebuild or not variant_path.exists():
        X_variant = _compute_preprocess_variant(variant, X_raw)
        np.save(variant_path, X_variant)
    else:
        X_variant = np.load(variant_path)
    return blank_meta, X_variant


def force_blank_negative_labels(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    blank_mask = meta['is_blank'].astype(int).values == 1
    for col in ['has_thiram', 'has_mg', 'has_mba', 'c_thiram', 'c_mg', 'c_mba', 'mixture_order']:
        if col in meta.columns:
            meta.loc[blank_mask, col] = 0
    if 'family' in meta.columns:
        meta.loc[blank_mask, 'family'] = 'blank'
    return meta


def assemble_soil_dataset(
    variant: str,
    include_blanks: bool,
    rebuild_blanks: bool,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    design_meta, X_design = load_soil_design_data(variant)
    blank_meta, X_blank = load_blank_data(variant, rebuild=rebuild_blanks)
    blank_meta = force_blank_negative_labels(blank_meta)
    if include_blanks:
        meta = pd.concat([design_meta, blank_meta], ignore_index=True)
        X = np.vstack([X_design, X_blank])
    else:
        meta = design_meta.copy()
        X = X_design.copy()
    meta = normalize_bool_columns(meta)
    return meta, X, blank_meta, X_blank


def make_presence_stratification_key(meta: pd.DataFrame) -> pd.Series:
    return (
        meta['has_thiram'].astype(int).astype(str)
        + meta['has_mg'].astype(int).astype(str)
        + meta['has_mba'].astype(int).astype(str)
    )


def make_stratified_folds(meta: pd.DataFrame, n_splits: int) -> np.ndarray:
    strat_key = make_presence_stratification_key(meta)
    counts = strat_key.value_counts()
    if int(counts.min()) < n_splits:
        raise ValueError(
            f'Min stratification count is {int(counts.min())}, smaller than n_splits={n_splits}. '
            f'Counts: {counts.to_dict()}'
        )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_ids = np.full(len(meta), -1, dtype=int)
    for fold_id, (_, test_idx) in enumerate(skf.split(np.zeros(len(meta)), strat_key), start=1):
        fold_ids[test_idx] = fold_id
    if np.any(fold_ids < 0):
        raise RuntimeError('Fold assignment failed.')
    return fold_ids


def positive_probability(clf, X: np.ndarray) -> np.ndarray:
    if not hasattr(clf, 'predict_proba'):
        return clf.predict(X).astype(float)
    proba = clf.predict_proba(X)
    class_to_col = {int(cls): idx for idx, cls in enumerate(clf.classes_)}
    if 1 not in class_to_col:
        return np.zeros(len(X), dtype=float)
    return proba[:, class_to_col[1]]


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_score))


def safe_roc_rows(run_id: str, task: dict, fold_id: int, y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    if len(np.unique(y_true)) < 2:
        return []
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return [
        {
            'run_id': run_id,
            'task': task['id'],
            'task_name': task['name'],
            'fold': int(fold_id),
            'point_index': int(i),
            'fpr': float(fp),
            'tpr': float(tp),
            'threshold': float(th),
        }
        for i, (fp, tp, th) in enumerate(zip(fpr, tpr, thresholds))
    ]


def run_soil_cv(
    meta: pd.DataFrame,
    X: np.ndarray,
    fold_ids: np.ndarray,
    tasks: list[dict],
    args: argparse.Namespace,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows = []
    prediction_rows = []
    roc_rows = []

    base_cols = [
        'sample_id',
        'file_name',
        'folder_name',
        'sample_role',
        'is_blank',
        'c_thiram',
        'c_mg',
        'c_mba',
        'has_thiram',
        'has_mg',
        'has_mba',
    ]

    for task in tasks:
        y = meta[task['col']].values.astype(int)
        for fold_id in sorted(np.unique(fold_ids)):
            train_idx = np.where(fold_ids != fold_id)[0]
            test_idx = np.where(fold_ids == fold_id)[0]
            clf = MODEL_REGISTRY[args.model]()
            clf.fit(X[train_idx], y[train_idx])
            y_score = positive_probability(clf, X[test_idx])
            y_pred = (y_score >= args.threshold).astype(int)

            auc = safe_auc(y[test_idx], y_score)
            macro_f1 = float(f1_score(y[test_idx], y_pred, average='macro', zero_division=0))
            balanced_acc = float(balanced_accuracy_score(y[test_idx], y_pred))
            result_rows.append({
                'run_id': run_id,
                'task': task['id'],
                'task_name': task['name'],
                'fold': int(fold_id),
                'n_test': int(len(test_idx)),
                'n_positive': int(y[test_idx].sum()),
                'n_negative': int(len(test_idx) - y[test_idx].sum()),
                'n_blank_test': int(meta.iloc[test_idx]['is_blank'].astype(int).sum()),
                'AUC': auc,
                'macro_f1': macro_f1,
                'balanced_acc': balanced_acc,
            })
            roc_rows.extend(safe_roc_rows(run_id, task, int(fold_id), y[test_idx], y_score))

            for local_pos, sample_idx in enumerate(test_idx):
                row = {col: meta.iloc[sample_idx][col] for col in base_cols if col in meta.columns}
                row.update({
                    'run_id': run_id,
                    'task': task['id'],
                    'task_name': task['name'],
                    'fold': int(fold_id),
                    'sample_index': int(sample_idx),
                    'y_true': int(y[sample_idx]),
                    'prob_positive': float(y_score[local_pos]),
                    'y_pred': int(y_pred[local_pos]),
                })
                prediction_rows.append(row)

            print(f"  {task['id']} fold {fold_id}: AUC={auc:.3f}, F1={macro_f1:.3f}, BAcc={balanced_acc:.3f}")

    results = pd.DataFrame(result_rows)
    summary_rows = []
    for (task_id, task_name), group in results.groupby(['task', 'task_name'], sort=False):
        summary_rows.append({
            'run_id': run_id,
            'task': task_id,
            'task_name': task_name,
            'n_folds': int(group['fold'].nunique()),
            'n_test_total': int(group['n_test'].sum()),
            'n_blank_test_total': int(group['n_blank_test'].sum()),
            'AUC_mean': float(group['AUC'].mean()),
            'AUC_std': float(group['AUC'].std(ddof=1)),
            'F1_mean': float(group['macro_f1'].mean()),
            'F1_std': float(group['macro_f1'].std(ddof=1)),
            'balanced_acc_mean': float(group['balanced_acc'].mean()),
            'balanced_acc_std': float(group['balanced_acc'].std(ddof=1)),
        })
    return results, pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows), pd.DataFrame(roc_rows)


def run_oof_permutation(
    prediction_df: pd.DataFrame,
    tasks: list[dict],
    n_permutations: int,
    run_id: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for task in tasks:
        task_df = prediction_df[prediction_df['task'] == task['id']].copy()
        y = task_df['y_true'].values.astype(int)
        score = task_df['prob_positive'].values.astype(float)
        folds = task_df['fold'].values.astype(int)
        observed = safe_auc(y, score)
        null_auc = []
        for _ in range(n_permutations):
            y_perm = y.copy()
            for fold_id in np.unique(folds):
                idx = np.where(folds == fold_id)[0]
                y_perm[idx] = rng.permutation(y_perm[idx])
            null_auc.append(safe_auc(y_perm, score))
        null_arr = np.asarray(null_auc, dtype=float)
        valid = null_arr[~np.isnan(null_arr)]
        p_value = float((1 + np.sum(valid >= observed)) / (len(valid) + 1))
        rows.append({
            'run_id': run_id,
            'task': task['id'],
            'task_name': task['name'],
            'observed_metric': observed,
            'metric': 'OOF_AUC',
            'p_value': p_value,
            'n_permutations': int(n_permutations),
            'null_mean': float(valid.mean()),
            'null_std': float(valid.std(ddof=1)),
            'permutation_mode': 'OOF_score_label_permutation_within_fold',
        })
        print(f"  permutation {task['id']}: observed AUC={observed:.4f}, p={p_value:.6f}")
    return pd.DataFrame(rows)


def run_blank_specificity(
    meta: pd.DataFrame,
    X: np.ndarray,
    blank_meta: pd.DataFrame,
    X_blank: np.ndarray,
    tasks: list[dict],
    args: argparse.Namespace,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    prediction_rows = []
    base_cols = ['file_name', 'folder_name', 'sample_role', 'is_blank']

    for task in tasks:
        y = meta[task['col']].values.astype(int)
        clf = MODEL_REGISTRY[args.model]()
        clf.fit(X, y)
        score = positive_probability(clf, X_blank)
        pred = (score >= args.threshold).astype(int)

        for i in range(len(blank_meta)):
            row = {col: blank_meta.iloc[i][col] for col in base_cols if col in blank_meta.columns}
            row.update({
                'run_id': run_id,
                'task': task['id'],
                'task_name': task['name'],
                'blank_index': int(i),
                'expected_label': 0,
                'prob_positive': float(score[i]),
                'y_pred': int(pred[i]),
            })
            prediction_rows.append(row)

        n_pred_positive = int(pred.sum())
        summary_rows.append({
            'run_id': run_id,
            'task': task['id'],
            'task_name': task['name'],
            'n_blanks': int(len(blank_meta)),
            'n_predicted_negative': int(len(blank_meta) - n_pred_positive),
            'n_predicted_positive': n_pred_positive,
            'specificity': float((len(blank_meta) - n_pred_positive) / len(blank_meta)),
            'max_prob_positive': float(np.max(score)),
            'mean_prob_positive': float(np.mean(score)),
            'blank_training_mode': 'included_in_full_soil_training' if args.include_blanks else 'not_included',
        })
        print(
            f"  blank {task['id']}: "
            f"{len(blank_meta) - n_pred_positive}/{len(blank_meta)} negative, "
            f"specificity={(len(blank_meta) - n_pred_positive) / len(blank_meta):.3f}"
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def write_metadata(
    args: argparse.Namespace,
    run_id: str,
    out_dir: Path,
    tasks: list[dict],
    meta: pd.DataFrame,
    blank_meta: pd.DataFrame,
    elapsed: float,
) -> None:
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
        'n_total_soil_cv_samples': int(len(meta)),
        'n_design_samples': int((meta['is_blank'].astype(int) == 0).sum()),
        'n_blank_samples': int(len(blank_meta)),
        'include_blanks': bool(args.include_blanks),
        'n_splits': int(args.n_splits),
        'threshold': float(args.threshold),
        'n_permutations': int(args.n_permutations),
        'seed': SEED,
        'elapsed_seconds': elapsed,
    }
    with open(out_dir / 'run_metadata' / f'{run_id}.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if args.model not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{args.model}'. Available: {sorted(MODEL_REGISTRY)}")
    if args.variant not in PREPROCESS_TAGS:
        raise KeyError(f"Unknown preprocessing variant '{args.variant}'")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'run_metadata').mkdir(parents=True, exist_ok=True)
    run_id = f"soil_only_cv_{args.model.lower()}_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start = perf_counter()

    tasks = selected_presence_tasks(args.tasks)
    meta, X, blank_meta, X_blank = assemble_soil_dataset(args.variant, args.include_blanks, args.rebuild_blanks)
    fold_ids = make_stratified_folds(meta, args.n_splits)
    split_df = meta.copy()
    split_df['fold'] = fold_ids
    split_df['stratification_key'] = make_presence_stratification_key(meta)
    split_df.to_csv(out_dir / 'soil_cv_split_assignments.csv', index=False, encoding='utf-8-sig')
    (
        split_df.groupby(['fold', 'stratification_key', 'sample_role'])
        .size()
        .reset_index(name='n_spectra')
        .to_csv(out_dir / 'soil_cv_split_qc.csv', index=False, encoding='utf-8-sig')
    )

    print(f'RUN_ID={run_id}')
    print(f'Output: {out_dir}')
    print(f'Model: {args.model}')
    print(f'Variant: {args.variant}')
    print(f'Include blanks: {args.include_blanks}')
    print(f'Model recipe: {MODEL_REGISTRY[args.model]()}')
    print(f'Total CV spectra: {len(meta)}; blanks detected: {len(blank_meta)}')
    print(f'Stratification counts: {make_presence_stratification_key(meta).value_counts().to_dict()}')

    cv_results, cv_summary, predictions, roc_df = run_soil_cv(meta, X, fold_ids, tasks, args, run_id)
    permutation = run_oof_permutation(predictions, tasks, args.n_permutations, run_id)
    blank_summary, blank_predictions = run_blank_specificity(meta, X, blank_meta, X_blank, tasks, args, run_id)

    cv_results.to_csv(out_dir / 'soil_cv_results.csv', index=False, encoding='utf-8-sig')
    cv_summary.to_csv(out_dir / 'soil_cv_summary.csv', index=False, encoding='utf-8-sig')
    predictions.to_csv(out_dir / 'soil_cv_predictions.csv', index=False, encoding='utf-8-sig')
    roc_df.to_csv(out_dir / 'soil_cv_roc_curve.csv', index=False, encoding='utf-8-sig')
    permutation.to_csv(out_dir / 'soil_permutation_test.csv', index=False, encoding='utf-8-sig')
    blank_summary.to_csv(out_dir / 'soil_blank_specificity.csv', index=False, encoding='utf-8-sig')
    blank_predictions.to_csv(out_dir / 'soil_blank_predictions.csv', index=False, encoding='utf-8-sig')

    write_metadata(args, run_id, out_dir, tasks, meta, blank_meta, perf_counter() - start)

    print('\nSoil CV summary:')
    print(cv_summary[['task', 'AUC_mean', 'AUC_std', 'F1_mean', 'F1_std', 'balanced_acc_mean']].to_string(index=False))
    print('\nPermutation:')
    print(permutation[['task', 'observed_metric', 'p_value', 'n_permutations']].to_string(index=False))
    print('\nBlank specificity:')
    print(blank_summary[['task', 'n_blanks', 'n_predicted_negative', 'specificity']].to_string(index=False))


if __name__ == '__main__':
    main()
