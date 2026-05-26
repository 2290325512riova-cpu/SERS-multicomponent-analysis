#!/usr/bin/env python3
"""Run the paper-main spectrum-level random-CV benchmark."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    PAPER_MAIN_BENCHMARK_DIR,
    PREPROCESS_TAGS,
    RANDOM_SPLIT_FILE,
    SPLITS_DIR,
    TASKS,
    TASKS_SUPPLEMENTARY,
)
from src.dataset import load_preprocessed_variants  # noqa: E402
from src.result_index import write_result_index  # noqa: E402
from src.train_eval import aggregate_results, run_cv_evaluation, save_results  # noqa: E402

PAPER_MAIN_MODELS = [
    'RF',
    'ExtraTrees',
    'HistGradientBoosting',
    'XGBoost',
    'SVM',
    'KNN',
    'PLS-DA',
    'LDA',
    '1D-CNN',
    '1D-ResNet',
    'RamanNet-Lite',
    'Spectrum-KAN',
    'KAN-CNN',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run paper-main random 5-fold benchmark.')
    parser.add_argument('--models', nargs='+', default=PAPER_MAIN_MODELS)
    parser.add_argument('--variants', nargs='+', default=list(PREPROCESS_TAGS))
    parser.add_argument('--task-group', choices=['main', 'supplementary', 'all'], default='main')
    parser.add_argument('--strict-models', action='store_true', help='Fail instead of skipping unavailable optional models.')
    return parser.parse_args()


def normalize_bool_columns(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    for col in ['has_thiram', 'has_mg', 'has_mba']:
        if col in meta.columns:
            meta[col] = meta[col].astype(str).str.strip().str.lower().map(
                {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
            ).fillna(0).astype(int)
    return meta


def resolve_models(model_names: list[str], strict: bool) -> list[str]:
    models = list(model_names)
    if 'XGBoost' in models and importlib.util.find_spec('xgboost') is None:
        msg = 'XGBoost is not installed; install requirements.txt before running the full paper-main benchmark.'
        if strict:
            raise ModuleNotFoundError(msg)
        print(f'WARNING: {msg} Skipping XGBoost for this run.')
        models = [model for model in models if model != 'XGBoost']
    return models


def resolve_tasks(task_group: str) -> list[dict]:
    if task_group == 'main':
        return TASKS
    if task_group == 'supplementary':
        return TASKS_SUPPLEMENTARY
    return TASKS + TASKS_SUPPLEMENTARY


def main() -> None:
    args = parse_args()
    models = resolve_models(args.models, strict=args.strict_models)
    tasks = resolve_tasks(args.task_group)

    split_path = SPLITS_DIR / RANDOM_SPLIT_FILE
    if not split_path.exists():
        raise FileNotFoundError(f'Missing random split: {split_path}')
    meta = normalize_bool_columns(pd.read_csv(split_path))
    wn, X_by_variant = load_preprocessed_variants(args.variants)

    PAPER_MAIN_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Output: {PAPER_MAIN_BENCHMARK_DIR}')
    print(f'Models: {models}')
    print(f'Variants: {args.variants}')
    print(f'Tasks: {[task["id"] for task in tasks]}')

    for variant in args.variants:
        X = X_by_variant[variant]
        print(f'\n{"=" * 70}\nRandom-CV benchmark on {variant}\n{"=" * 70}')
        res_df = run_cv_evaluation(
            meta,
            X,
            model_names=models,
            preprocess_tag=variant,
            wn=wn,
            tasks_override=tasks,
            use_random_split=False,
        )
        agg_df = aggregate_results(res_df)
        save_results(res_df, agg_df, tag=variant, output_dir=PAPER_MAIN_BENCHMARK_DIR)

    matrix, best = write_result_index(
        PAPER_MAIN_BENCHMARK_DIR,
        source_stage='paper_main_random_cv',
        description='Paper-main spectrum-level random 5-fold CV benchmark.',
        models=None,
        variants=None,
        source_priority=10,
    )
    print(f'Indexed {len(matrix)} rows; best-by-task rows: {len(best)}')


if __name__ == '__main__':
    main()
