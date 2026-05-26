#!/usr/bin/env python3
"""Prepare soil metadata/caches and run pure-trained soil presence validation."""
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
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    PAPER_MAIN_SOIL_DIR,
    PREPROCESS_TAGS,
    RANDOM_SPLIT_FILE,
    SEED,
    SPLITS_DIR,
    TASKS,
    WN_MAX,
    WN_MIN,
    WN_STEP,
)
from src.dataset import (  # noqa: E402
    _compute_preprocess_variant,
    build_soil_metadata,
    load_preprocessed_variants,
    read_bwram_spectrum,
)
from src.models import MODEL_REGISTRY  # noqa: E402


DEFAULT_PRESENCE_TASK_IDS = [
    'P1_thiram_presence',
    'P2_mg_presence',
    'P3_mba_presence',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare soil external-validation artifacts.')
    parser.add_argument('--variants', nargs='+', default=['raw', 'p1', 'p5'])
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--run-validation', action='store_true', help='Train on all pure spectra and predict soil presence labels.')
    parser.add_argument('--model', default='ExtraTrees')
    parser.add_argument('--variant', default='p1', help='Preprocessing variant to use for soil validation.')
    parser.add_argument('--tasks', nargs='+', default=DEFAULT_PRESENCE_TASK_IDS)
    return parser.parse_args()


def build_soil_raw(meta: pd.DataFrame, rebuild: bool = False) -> tuple[np.ndarray, np.ndarray]:
    PAPER_MAIN_SOIL_DIR.mkdir(parents=True, exist_ok=True)
    wn_path = PAPER_MAIN_SOIL_DIR / 'wavenumber.npy'
    raw_path = PAPER_MAIN_SOIL_DIR / 'X_soil_raw.npy'
    if not rebuild and wn_path.exists() and raw_path.exists():
        return np.load(wn_path), np.load(raw_path)

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

    np.save(wn_path, wn_common)
    np.save(raw_path, X_raw)
    if unreadable:
        pd.Series(unreadable, name='file_path').to_csv(PAPER_MAIN_SOIL_DIR / 'unreadable_files.csv', index=False)
    return wn_common, X_raw


def normalize_bool_columns(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    for col in ['has_thiram', 'has_mg', 'has_mba']:
        if col in meta.columns:
            meta[col] = meta[col].astype(str).str.strip().str.lower().map(
                {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
            ).fillna(0).astype(int)
    return meta


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


def selected_presence_tasks(task_ids: list[str]) -> list[dict]:
    task_map = {task['id']: task for task in TASKS}
    missing = [task_id for task_id in task_ids if task_id not in task_map]
    if missing:
        raise KeyError(f'Unknown task ids: {missing}')
    tasks = [task_map[task_id] for task_id in task_ids]
    non_presence = [task['id'] for task in tasks if task.get('filter_col') or task['id'].startswith('G')]
    if non_presence:
        raise ValueError(f'Soil validation is restricted to presence tasks, got: {non_presence}')
    return tasks


def run_presence_validation(meta_soil: pd.DataFrame, args: argparse.Namespace) -> None:
    if args.model not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{args.model}'. Available: {sorted(MODEL_REGISTRY)}")
    if args.variant not in PREPROCESS_TAGS:
        raise KeyError(f"Unknown preprocessing variant '{args.variant}'")

    run_id = f"soil_validation_{args.model.lower()}_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (PAPER_MAIN_SOIL_DIR / 'run_metadata').mkdir(parents=True, exist_ok=True)

    split_path = SPLITS_DIR / RANDOM_SPLIT_FILE
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    meta_pure = normalize_bool_columns(pd.read_csv(split_path))
    meta_soil = normalize_bool_columns(meta_soil)
    tasks = selected_presence_tasks(args.tasks)

    _, X_by_variant = load_preprocessed_variants([args.variant])
    X_pure = X_by_variant[args.variant]
    soil_path = PAPER_MAIN_SOIL_DIR / f'X_soil_{args.variant}.npy'
    if not soil_path.exists():
        raise FileNotFoundError(f'Missing soil cache for {args.variant}: {soil_path}')
    X_soil = np.load(soil_path)

    if len(meta_pure) != len(X_pure):
        raise ValueError(f'Pure metadata/X length mismatch: {len(meta_pure)} vs {len(X_pure)}')
    if len(meta_soil) != len(X_soil):
        raise ValueError(f'Soil metadata/X length mismatch: {len(meta_soil)} vs {len(X_soil)}')

    print(f'RUN_ID={run_id}')
    print(f'Output: {PAPER_MAIN_SOIL_DIR}')
    print(f'Model: {args.model}')
    print(f'Variant: {args.variant}')
    print(f'Tasks: {[task["id"] for task in tasks]}')
    print(f'Model recipe: {MODEL_REGISTRY[args.model]()}')
    print(f'Pure train spectra: {len(meta_pure)}')
    print(f'Soil design spectra: {len(meta_soil)}')

    result_rows = []
    prediction_df = meta_soil[[
        'file_path',
        'file_name',
        'folder_name',
        'c_thiram',
        'c_mg',
        'c_mba',
        'has_thiram',
        'has_mg',
        'has_mba',
    ]].copy()
    timing_rows = []

    for task in tasks:
        task_start = perf_counter()
        y_train = meta_pure[task['col']].values.astype(int)
        y_true = meta_soil[task['col']].values.astype(int)

        clf = MODEL_REGISTRY[args.model]()
        fit_start = perf_counter()
        clf.fit(X_pure, y_train)
        fit_seconds = perf_counter() - fit_start

        pred_start = perf_counter()
        y_pred = clf.predict(X_soil).astype(int)
        pred_seconds = perf_counter() - pred_start

        labels = [0, 1]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f'true_{v}' for v in labels], columns=[f'pred_{v}' for v in labels])
        cm_df.to_csv(PAPER_MAIN_SOIL_DIR / f'confusion_matrix_{task["id"]}.csv', encoding='utf-8-sig')

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).T.to_csv(
            PAPER_MAIN_SOIL_DIR / f'classification_report_{task["id"]}.csv',
            encoding='utf-8-sig',
        )

        result_rows.append({
            'run_id': run_id,
            'task': task['id'],
            'task_name': task['name'],
            'model': args.model,
            'preprocess': args.variant,
            'n_train_pure': int(len(y_train)),
            'n_soil': int(len(y_true)),
            'n_soil_positive': int(y_true.sum()),
            'n_soil_negative': int(len(y_true) - y_true.sum()),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'precision_positive': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            'recall_positive': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            'f1_positive': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        })

        prediction_df[f'{task["id"]}_true'] = y_true
        prediction_df[f'{task["id"]}_pred'] = y_pred
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(X_soil)
            class_to_col = {int(cls): i for i, cls in enumerate(clf.classes_)}
            for cls in labels:
                if cls in class_to_col:
                    prediction_df[f'{task["id"]}_prob_{cls}'] = proba[:, class_to_col[cls]]
                else:
                    prediction_df[f'{task["id"]}_prob_{cls}'] = 0.0

        timing_rows.append({
            'run_id': run_id,
            'task': task['id'],
            'fit_seconds': fit_seconds,
            'predict_seconds': pred_seconds,
            'total_seconds': perf_counter() - task_start,
        })
        print(
            f"  {task['id']}: acc={accuracy_score(y_true, y_pred):.3f}, "
            f"macro-F1={f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}, "
            f"positive recall={recall_score(y_true, y_pred, pos_label=1, zero_division=0):.3f}"
        )

    pd.DataFrame(result_rows).to_csv(PAPER_MAIN_SOIL_DIR / 'soil_validation_results.csv', index=False, encoding='utf-8-sig')
    prediction_df.to_csv(PAPER_MAIN_SOIL_DIR / 'soil_predictions_presence.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(timing_rows).to_csv(PAPER_MAIN_SOIL_DIR / 'soil_validation_timings.csv', index=False, encoding='utf-8-sig')

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
        'n_train_pure': int(len(meta_pure)),
        'n_soil_design': int(len(meta_soil)),
        'seed': SEED,
    }
    with open(PAPER_MAIN_SOIL_DIR / 'run_metadata' / f'{run_id}.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print('Summary output:', PAPER_MAIN_SOIL_DIR / 'soil_validation_results.csv')
    print('Prediction output:', PAPER_MAIN_SOIL_DIR / 'soil_predictions_presence.csv')


def main() -> None:
    args = parse_args()
    unknown = [variant for variant in args.variants if variant not in PREPROCESS_TAGS]
    if unknown:
        raise KeyError(f'Unknown variants: {unknown}')
    if args.variant not in args.variants:
        args.variants = list(dict.fromkeys([*args.variants, args.variant]))

    PAPER_MAIN_SOIL_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = PAPER_MAIN_SOIL_DIR / 'soil_metadata.csv'
    try:
        meta = build_soil_metadata()
        meta.to_csv(metadata_path, index=False, encoding='utf-8-sig')
    except FileNotFoundError:
        if not metadata_path.exists():
            raise
        print(f'WARNING: raw soil source directory is unavailable; using cached metadata: {metadata_path}')
        meta = pd.read_csv(metadata_path)
    label_summary = meta.groupby(['folder_name', 'c_thiram', 'c_mg', 'c_mba']).size().reset_index(name='n_spectra')
    label_summary.to_csv(PAPER_MAIN_SOIL_DIR / 'soil_label_summary.csv', index=False, encoding='utf-8-sig')

    _, X_raw = build_soil_raw(meta, rebuild=args.rebuild)
    for variant in args.variants:
        out_path = PAPER_MAIN_SOIL_DIR / f'X_soil_{variant}.npy'
        if not args.rebuild and out_path.exists():
            continue
        X_variant = _compute_preprocess_variant(variant, X_raw)
        np.save(out_path, X_variant)

    print(f'Soil spectra: {len(meta)}')
    print(f'Folders: {meta["folder_name"].nunique()}')
    print(f'Output: {PAPER_MAIN_SOIL_DIR}')
    if args.run_validation:
        run_presence_validation(meta, args)


if __name__ == '__main__':
    main()
