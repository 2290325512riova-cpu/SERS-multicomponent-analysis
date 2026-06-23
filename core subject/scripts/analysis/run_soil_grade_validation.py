#!/usr/bin/env python3
"""Run ternary spiked-soil grade validation for the manuscript.

Soil-G is deliberately defined on ternary spiked-soil spectra only.  The grade is
the total molar loading of the three coexisting analytes, not an independent
low/middle/high level for every analyte.  This keeps the experiment aligned with
the manuscript's competition-adsorption storyline: single-peak MG readout is
compared with multi-band and model-derived spectral inputs under ternary
coexistence.
"""
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import PAPER_MAIN_DIR, PAPER_MAIN_SOIL_DIR, PREPROCESS_TAGS, SEED  # noqa: E402
from src.models import MODEL_REGISTRY  # noqa: E402


TASK_ORDER = [
    'P1_thiram_presence',
    'P2_mg_presence',
    'P3_mba_presence',
    'G1_thiram_molar_grade',
    'G2_mg_molar_grade',
    'G3_mba_molar_grade',
]

MANUAL_MARKER_PEAKS = [
    560,
    860,
    930,
    1145,
    1380,
    1444,
    1510,
    908,
    1172,
    1220,
    1394,
    1616,
    1078,
    1180,
    1490,
    1590,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run Soil-G ternary spiked-soil grade validation with input-set comparison.'
    )
    parser.add_argument('--model', default='ExtraTrees')
    parser.add_argument('--variant', default='p1')
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--n-permutations', type=int, default=10000)
    parser.add_argument('--band-half-width', type=float, default=5.0)
    parser.add_argument('--shap-retention-pct', type=float, default=10.0)
    parser.add_argument('--output-dir', type=Path, default=PAPER_MAIN_DIR / 'soil_grade_validation')
    return parser.parse_args()


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


def normalize_bool_columns(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    for col in ['has_thiram', 'has_mg', 'has_mba', 'is_blank', 'is_design_sample', 'is_background']:
        if col in meta.columns:
            meta[col] = meta[col].astype(str).str.strip().str.lower().map(
                {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
            ).fillna(0).astype(int)
    return meta


def load_ternary_soil(variant: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta_path = PAPER_MAIN_SOIL_DIR / 'soil_metadata.csv'
    x_path = PAPER_MAIN_SOIL_DIR / f'X_soil_{variant}.npy'
    wn_path = PAPER_MAIN_SOIL_DIR / 'wavenumber.npy'
    if not meta_path.exists():
        raise FileNotFoundError(f'Missing soil metadata: {meta_path}')
    if not x_path.exists():
        raise FileNotFoundError(f'Missing soil spectrum cache: {x_path}')
    if not wn_path.exists():
        raise FileNotFoundError(f'Missing wavenumber cache: {wn_path}')

    meta_all = normalize_bool_columns(pd.read_csv(meta_path))
    X_all = np.load(x_path)
    wn = np.load(wn_path)
    if len(meta_all) != len(X_all):
        raise ValueError(f'Soil metadata/X length mismatch: {len(meta_all)} vs {len(X_all)}')

    ternary_mask = (
        (meta_all['sample_role'].astype(str).str.lower() == 'design')
        & (meta_all['has_thiram'] == 1)
        & (meta_all['has_mg'] == 1)
        & (meta_all['has_mba'] == 1)
    )
    meta = meta_all.loc[ternary_mask].reset_index(drop=True).copy()
    X = X_all[np.where(ternary_mask.to_numpy())[0]]
    if len(meta) == 0:
        raise ValueError('No ternary spiked-soil spectra were found.')

    for col in ['c_thiram_molar', 'c_mg_molar', 'c_mba_molar']:
        if col not in meta.columns:
            raise KeyError(f'Missing molar concentration column: {col}')
    total = meta[['c_thiram_molar', 'c_mg_molar', 'c_mba_molar']].sum(axis=1).astype(float)
    meta['soil_g_total_molar'] = total

    unique_totals = np.array(sorted(total.unique()), dtype=float)
    if len(unique_totals) != 3:
        raise ValueError(
            'Soil-G expects exactly three total molar loading levels; '
            f'found {len(unique_totals)}: {unique_totals.tolist()}'
        )
    total_to_grade = {
        unique_totals[0]: 'low',
        unique_totals[1]: 'middle',
        unique_totals[2]: 'high',
    }
    meta['soil_g_grade'] = total.map(total_to_grade)
    meta['soil_g_grade_order'] = meta['soil_g_grade'].map({'low': 0, 'middle': 1, 'high': 2}).astype(int)
    return meta, X, wn


def nearest_index(wn: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(wn - target)))


def band_area_features(
    X: np.ndarray,
    wn: np.ndarray,
    centers: list[float],
    half_width: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    features = []
    rows = []
    for center in centers:
        mask = (wn >= center - half_width) & (wn <= center + half_width)
        if int(mask.sum()) < 2:
            idx = nearest_index(wn, center)
            area = X[:, idx]
            used_indices = [idx]
        else:
            used_indices = np.where(mask)[0].tolist()
            area = np.trapezoid(X[:, used_indices], wn[used_indices], axis=1)
        features.append(area)
        rows.append(
            {
                'feature_set': 'manual_marker_band_areas',
                'peak_center_cm-1': float(center),
                'half_width_cm-1': float(half_width),
                'n_wavenumbers': int(len(used_indices)),
                'nearest_wavenumber_cm-1': float(wn[nearest_index(wn, center)]),
                'feature_indices': ';'.join(str(i) for i in used_indices),
            }
        )
    return np.vstack(features).T, pd.DataFrame(rows)


def load_global_shap_top_indices(wn: np.ndarray, retention_pct: float) -> tuple[np.ndarray, pd.DataFrame]:
    shap_path = PAPER_MAIN_DIR / 'shap_peak_cluster' / 'shap_mean_abs_by_wavenumber.csv'
    if not shap_path.exists():
        raise FileNotFoundError(f'Missing SHAP wavenumber importance file: {shap_path}')
    shap = pd.read_csv(shap_path)
    missing_tasks = [task for task in TASK_ORDER if task not in set(shap['task'])]
    if missing_tasks:
        raise ValueError(f'SHAP file is missing expected tasks: {missing_tasks}')

    agg = (
        shap.groupby(['feature_index', 'wavenumber'], as_index=False)['mean_abs_shap']
        .mean()
        .rename(columns={'mean_abs_shap': 'mean_abs_shap_across_tasks'})
        .sort_values('mean_abs_shap_across_tasks', ascending=False)
        .reset_index(drop=True)
    )
    n_features = max(1, int(np.ceil(len(wn) * retention_pct / 100.0)))
    selected = agg.head(n_features).copy()
    selected['feature_set'] = f'shap_global_top{retention_pct:g}pct'
    selected['selection_rank'] = np.arange(1, len(selected) + 1)
    selected['nearest_cache_wavenumber_cm-1'] = selected['feature_index'].map(lambda i: float(wn[int(i)]))
    selected_indices = selected['feature_index'].to_numpy(dtype=int)
    return selected_indices, selected


def make_feature_sets(
    X: np.ndarray,
    wn: np.ndarray,
    band_half_width: float,
    shap_retention_pct: float,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    mg_area, mg_rows = band_area_features(X, wn, [1616], band_half_width)
    mg_rows['feature_set'] = 'mg1616_single_band_area'

    manual_area, manual_rows = band_area_features(X, wn, MANUAL_MARKER_PEAKS, band_half_width)
    shap_idx, shap_rows = load_global_shap_top_indices(wn, shap_retention_pct)

    feature_sets = {
        'mg1616_single_band_area': mg_area,
        'manual_marker_band_areas': manual_area,
        'shap_global_top10pct': X[:, shap_idx],
        'full_p1_spectrum': X.copy(),
    }

    full_rows = pd.DataFrame(
        [
            {
                'feature_set': 'full_p1_spectrum',
                'feature_index': 'all',
                'wavenumber': f'{float(wn.min()):.1f}-{float(wn.max()):.1f}',
                'n_features': int(X.shape[1]),
                'description': 'All p1-preprocessed wavenumbers from 400 to 1800 cm^-1.',
            },
            {
                'feature_set': 'mg1616_single_band_area',
                'feature_index': 'band_area',
                'wavenumber': 1616,
                'n_features': 1,
                'description': f'Integrated p1 intensity within +/-{band_half_width:g} cm^-1 around MG 1616 cm^-1.',
            },
            {
                'feature_set': 'manual_marker_band_areas',
                'feature_index': 'band_area_vector',
                'wavenumber': ';'.join(str(x) for x in MANUAL_MARKER_PEAKS),
                'n_features': len(MANUAL_MARKER_PEAKS),
                'description': f'Integrated p1 intensities within +/-{band_half_width:g} cm^-1 around manually assigned marker bands.',
            },
            {
                'feature_set': 'shap_global_top10pct',
                'feature_index': 'SHAP top-ranked indices',
                'wavenumber': f'top {shap_retention_pct:g}% by across-task mean |SHAP|',
                'n_features': int(len(shap_idx)),
                'description': 'Global top SHAP wavenumbers derived from the six pure-mixture tasks, then applied to Soil-G without re-ranking on soil data.',
            },
        ]
    )
    feature_details = pd.concat(
        [
            mg_rows,
            manual_rows,
            shap_rows,
            full_rows,
        ],
        ignore_index=True,
        sort=False,
    )
    return feature_sets, feature_details


def make_folds(y: np.ndarray, n_splits: int) -> np.ndarray:
    counts = pd.Series(y).value_counts()
    if int(counts.min()) < n_splits:
        raise ValueError(f'Min class count {int(counts.min())} is smaller than n_splits={n_splits}.')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_ids = np.full(len(y), -1, dtype=int)
    for fold_id, (_, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        fold_ids[test_idx] = fold_id
    return fold_ids


def run_cv_for_feature_set(
    feature_set: str,
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    fold_ids: np.ndarray,
    model_name: str,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    pred_rows = []
    base_cols = [
        'sample_id',
        'file_name',
        'folder_name',
        'c_thiram',
        'c_mg',
        'c_mba',
        'c_thiram_molar',
        'c_mg_molar',
        'c_mba_molar',
        'soil_g_total_molar',
        'soil_g_grade',
    ]

    for fold_id in sorted(np.unique(fold_ids)):
        train_idx = np.where(fold_ids != fold_id)[0]
        test_idx = np.where(fold_ids == fold_id)[0]
        clf = MODEL_REGISTRY[model_name]()
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])

        macro_f1 = float(f1_score(y[test_idx], y_pred, average='macro', zero_division=0))
        bal_acc = float(balanced_accuracy_score(y[test_idx], y_pred))
        acc = float(accuracy_score(y[test_idx], y_pred))
        result_rows.append(
            {
                'run_id': run_id,
                'feature_set': feature_set,
                'fold': int(fold_id),
                'n_train': int(len(train_idx)),
                'n_test': int(len(test_idx)),
                'macro_f1': macro_f1,
                'balanced_accuracy': bal_acc,
                'accuracy': acc,
            }
        )
        for local_pos, sample_idx in enumerate(test_idx):
            row = {col: meta.iloc[sample_idx][col] for col in base_cols if col in meta.columns}
            row.update(
                {
                    'run_id': run_id,
                    'feature_set': feature_set,
                    'fold': int(fold_id),
                    'sample_index': int(sample_idx),
                    'y_true': int(y[sample_idx]),
                    'grade_true': meta.iloc[sample_idx]['soil_g_grade'],
                    'y_pred': int(y_pred[local_pos]),
                    'grade_pred': {0: 'low', 1: 'middle', 2: 'high'}[int(y_pred[local_pos])],
                }
            )
            pred_rows.append(row)
        print(
            f"  {feature_set} fold {fold_id}: "
            f"F1={macro_f1:.3f}, BAcc={bal_acc:.3f}, Acc={acc:.3f}"
        )
    return pd.DataFrame(result_rows), pd.DataFrame(pred_rows)


def summarize_folds(results: pd.DataFrame, predictions: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows = []
    for feature_set, group in results.groupby('feature_set', sort=False):
        pred = predictions[predictions['feature_set'] == feature_set]
        y = pred['y_true'].to_numpy(dtype=int)
        y_pred = pred['y_pred'].to_numpy(dtype=int)
        rows.append(
            {
                'run_id': run_id,
                'feature_set': feature_set,
                'n_spectra': int(len(pred)),
                'n_features': int(group['n_features'].iloc[0]) if 'n_features' in group.columns else np.nan,
                'macro_f1_mean': float(group['macro_f1'].mean()),
                'macro_f1_std': float(group['macro_f1'].std(ddof=1)),
                'balanced_accuracy_mean': float(group['balanced_accuracy'].mean()),
                'balanced_accuracy_std': float(group['balanced_accuracy'].std(ddof=1)),
                'accuracy_mean': float(group['accuracy'].mean()),
                'accuracy_std': float(group['accuracy'].std(ddof=1)),
                'oof_macro_f1': float(f1_score(y, y_pred, average='macro', zero_division=0)),
                'oof_balanced_accuracy': float(balanced_accuracy_score(y, y_pred)),
                'oof_accuracy': float(accuracy_score(y, y_pred)),
            }
        )
    return pd.DataFrame(rows)


def confusion_and_recall(predictions: pd.DataFrame, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = [0, 1, 2]
    label_names = {0: 'low', 1: 'middle', 2: 'high'}
    cm_rows = []
    recall_rows = []
    for feature_set, group in predictions.groupby('feature_set', sort=False):
        y = group['y_true'].to_numpy(dtype=int)
        yp = group['y_pred'].to_numpy(dtype=int)
        cm = confusion_matrix(y, yp, labels=labels)
        recalls = recall_score(y, yp, labels=labels, average=None, zero_division=0)
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                cm_rows.append(
                    {
                        'run_id': run_id,
                        'feature_set': feature_set,
                        'true_label': int(true_label),
                        'true_grade': label_names[true_label],
                        'pred_label': int(pred_label),
                        'pred_grade': label_names[pred_label],
                        'n': int(cm[i, j]),
                    }
                )
            recall_rows.append(
                {
                    'run_id': run_id,
                    'feature_set': feature_set,
                    'grade_label': int(true_label),
                    'grade': label_names[true_label],
                    'recall': float(recalls[i]),
                    'support': int((y == true_label).sum()),
                }
            )
    return pd.DataFrame(cm_rows), pd.DataFrame(recall_rows)


def permutation_test(
    predictions: pd.DataFrame,
    n_permutations: int,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    summary_rows = []
    null_rows = []
    for feature_set, group in predictions.groupby('feature_set', sort=False):
        y = group['y_true'].to_numpy(dtype=int)
        y_pred = group['y_pred'].to_numpy(dtype=int)
        folds = group['fold'].to_numpy(dtype=int)
        observed = float(f1_score(y, y_pred, average='macro', zero_division=0))
        null_scores = np.zeros(n_permutations, dtype=float)
        for i in range(n_permutations):
            y_perm = y.copy()
            for fold_id in np.unique(folds):
                idx = np.where(folds == fold_id)[0]
                y_perm[idx] = rng.permutation(y_perm[idx])
            null_scores[i] = f1_score(y_perm, y_pred, average='macro', zero_division=0)
            null_rows.append(
                {
                    'run_id': run_id,
                    'feature_set': feature_set,
                    'permutation_index': int(i + 1),
                    'null_macro_f1': float(null_scores[i]),
                }
            )
        p_value = float((1 + np.sum(null_scores >= observed)) / (n_permutations + 1))
        summary_rows.append(
            {
                'run_id': run_id,
                'feature_set': feature_set,
                'metric': 'OOF_macro_F1',
                'observed_metric': observed,
                'p_value': p_value,
                'n_permutations': int(n_permutations),
                'null_mean': float(null_scores.mean()),
                'null_std': float(null_scores.std(ddof=1)),
                'permutation_mode': 'OOF_prediction_label_permutation_within_fold',
            }
        )
        print(f"  permutation {feature_set}: observed macro-F1={observed:.4f}, p={p_value:.6f}")
    return pd.DataFrame(summary_rows), pd.DataFrame(null_rows)


def write_report(
    out_dir: Path,
    run_id: str,
    meta: pd.DataFrame,
    summary: pd.DataFrame,
    recall_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    feature_details: pd.DataFrame,
) -> None:
    class_counts = (
        meta.groupby(['soil_g_grade', 'soil_g_total_molar'])
        .size()
        .reset_index(name='n_spectra')
        .sort_values('soil_g_total_molar')
    )
    feature_counts = (
        feature_details.groupby('feature_set')
        .size()
        .reset_index(name='n_rows_in_definition_file')
        .sort_values('feature_set')
    )
    lines = [
        '# Soil-G validation report',
        '',
        f'Run id: `{run_id}`',
        '',
        '## Soil-G definition',
        '',
        'Soil-G used ternary spiked-soil spectra only. Grades were assigned from the total molar loading of Thiram, MG, and 4-MBA.',
        '',
        class_counts.to_markdown(index=False),
        '',
        '## Input-set comparison',
        '',
        summary.to_markdown(index=False),
        '',
        '## Per-grade recall',
        '',
        recall_df.to_markdown(index=False),
        '',
        '## Permutation test',
        '',
        permutation_df.to_markdown(index=False),
        '',
        '## Feature definitions',
        '',
        feature_counts.to_markdown(index=False),
        '',
        'Interpretation note: this is a matrix-matched fivefold assessment within spiked soil spectra, not an external unknown-soil validation or continuous concentration calibration.',
    ]
    (out_dir / 'soil_g_validation_report.md').write_text('\n'.join(lines), encoding='utf-8')


def write_metadata(args: argparse.Namespace, run_id: str, out_dir: Path, meta: pd.DataFrame, elapsed: float) -> None:
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
        'n_splits': int(args.n_splits),
        'n_permutations': int(args.n_permutations),
        'band_half_width_cm-1': float(args.band_half_width),
        'shap_retention_pct': float(args.shap_retention_pct),
        'n_ternary_soil_spectra': int(len(meta)),
        'soil_g_counts': meta['soil_g_grade'].value_counts().to_dict(),
        'seed': int(SEED),
        'elapsed_seconds': float(elapsed),
    }
    with open(out_dir / 'run_metadata' / f'{run_id}.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if args.model not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{args.model}'. Available: {sorted(MODEL_REGISTRY)}")
    if args.variant not in PREPROCESS_TAGS:
        raise KeyError(f"Unknown preprocessing variant '{args.variant}'.")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'run_metadata').mkdir(parents=True, exist_ok=True)
    run_id = f"soil_g_{args.model.lower()}_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start = perf_counter()

    meta, X, wn = load_ternary_soil(args.variant)
    y = meta['soil_g_grade_order'].to_numpy(dtype=int)
    fold_ids = make_folds(y, args.n_splits)
    meta_with_folds = meta.copy()
    meta_with_folds['fold'] = fold_ids

    feature_sets, feature_details = make_feature_sets(
        X,
        wn,
        band_half_width=args.band_half_width,
        shap_retention_pct=args.shap_retention_pct,
    )

    print(f'RUN_ID={run_id}')
    print(f'Output: {out_dir}')
    print(f'Model: {args.model}; variant: {args.variant}')
    print(f'Model recipe: {MODEL_REGISTRY[args.model]()}')
    print('Soil-G grade counts:')
    print(meta[['soil_g_grade', 'soil_g_total_molar']].value_counts().sort_index().to_string())
    print('Feature sets:')
    for name, x_feat in feature_sets.items():
        print(f'  {name}: {x_feat.shape[1]} features')

    all_results = []
    all_predictions = []
    for feature_set, x_feat in feature_sets.items():
        cv_results, preds = run_cv_for_feature_set(
            feature_set,
            x_feat,
            y,
            meta_with_folds,
            fold_ids,
            args.model,
            run_id,
        )
        cv_results['n_features'] = int(x_feat.shape[1])
        all_results.append(cv_results)
        all_predictions.append(preds)

    results_df = pd.concat(all_results, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    summary_df = summarize_folds(results_df, predictions_df, run_id)
    cm_df, recall_df = confusion_and_recall(predictions_df, run_id)
    permutation_df, null_df = permutation_test(predictions_df, args.n_permutations, run_id)

    meta_with_folds.to_csv(out_dir / 'soil_g_metadata.csv', index=False, encoding='utf-8-sig')
    results_df.to_csv(out_dir / 'soil_g_fold_metrics.csv', index=False, encoding='utf-8-sig')
    summary_df.to_csv(out_dir / 'soil_g_input_comparison_summary.csv', index=False, encoding='utf-8-sig')
    predictions_df.to_csv(out_dir / 'soil_g_oof_predictions.csv', index=False, encoding='utf-8-sig')
    cm_df.to_csv(out_dir / 'soil_g_confusion_matrix.csv', index=False, encoding='utf-8-sig')
    recall_df.to_csv(out_dir / 'soil_g_per_class_recall.csv', index=False, encoding='utf-8-sig')
    permutation_df.to_csv(out_dir / 'soil_g_permutation_test.csv', index=False, encoding='utf-8-sig')
    null_df.to_csv(out_dir / 'soil_g_permutation_null_macro_f1.csv', index=False, encoding='utf-8-sig')
    feature_details.to_csv(out_dir / 'soil_g_feature_sets.csv', index=False, encoding='utf-8-sig')

    elapsed = perf_counter() - start
    write_metadata(args, run_id, out_dir, meta, elapsed)
    write_report(out_dir, run_id, meta, summary_df, recall_df, permutation_df, feature_details)

    print('\nSoil-G input comparison:')
    print(
        summary_df[
            [
                'feature_set',
                'n_spectra',
                'n_features',
                'macro_f1_mean',
                'macro_f1_std',
                'balanced_accuracy_mean',
                'balanced_accuracy_std',
                'accuracy_mean',
                'accuracy_std',
                'oof_macro_f1',
            ]
        ].to_string(index=False)
    )
    print('\nPermutation:')
    print(permutation_df[['feature_set', 'observed_metric', 'p_value', 'n_permutations']].to_string(index=False))
    print(f'\nElapsed: {elapsed:.1f} s')


if __name__ == '__main__':
    main()
