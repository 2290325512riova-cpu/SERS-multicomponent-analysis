#!/usr/bin/env python3
"""Smoke-test lightweight DL candidates on concentration-grade tasks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import PAPER_MAIN_DIR, RANDOM_SPLIT_FILE, SPLITS_DIR, TASKS  # noqa: E402
from src.dataset import load_preprocessed_variants  # noqa: E402
from src.models import MODEL_REGISTRY  # noqa: E402
from src.result_index import write_result_index  # noqa: E402
from src.train_eval import aggregate_results, run_cv_evaluation, save_results  # noqa: E402


DEFAULT_MODELS = ['RamanNet-Lite', 'PatchTransformer']
DEFAULT_TASKS = ['G2_mg_molar_grade', 'G3_mba_molar_grade']
TRAIN_CFGS = {
    'no_aug': {'aug_enabled': False, 'mixup_enabled': False},
    'composition_mixup': {'aug_enabled': True, 'aug_n': 4, 'mixup_enabled': True, 'mixup_alpha': 0.3},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run DL candidate smoke tests for G2/G3.')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--variants', nargs='+', default=['p1', 'raw'])
    parser.add_argument('--tasks', nargs='+', default=DEFAULT_TASKS)
    parser.add_argument('--mode', choices=sorted(TRAIN_CFGS), default='composition_mixup')
    parser.add_argument('--output-dir', default=str(PAPER_MAIN_DIR / 'dl_candidate_smoke'))
    return parser.parse_args()


def normalize_bool_columns(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    for col in ['has_thiram', 'has_mg', 'has_mba']:
        if col in meta.columns:
            meta[col] = meta[col].astype(str).str.strip().str.lower().map(
                {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
            ).fillna(0).astype(int)
    return meta


def main() -> None:
    args = parse_args()
    unknown_models = [model for model in args.models if model not in MODEL_REGISTRY]
    if unknown_models:
        raise ValueError(f'Unknown model(s): {unknown_models}. Available models: {sorted(MODEL_REGISTRY)}')

    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = SPLITS_DIR / RANDOM_SPLIT_FILE
    meta = normalize_bool_columns(pd.read_csv(split_path))
    wn, X_by_variant = load_preprocessed_variants(args.variants)
    tasks = [task for task in TASKS if task['id'] in set(args.tasks)]
    if not tasks:
        raise ValueError(f'No tasks selected from: {args.tasks}')

    model_kwargs = {
        model: {'train_cfg': dict(TRAIN_CFGS[args.mode])}
        for model in args.models
    }
    for variant in args.variants:
        print(f'\n{"=" * 70}\nDL candidate smoke mode={args.mode}, variant={variant}\n{"=" * 70}')
        res_df = run_cv_evaluation(
            meta,
            X_by_variant[variant],
            model_names=args.models,
            preprocess_tag=variant,
            wn=wn,
            tasks_override=tasks,
            model_kwargs=model_kwargs,
            use_random_split=False,
        )
        agg_df = aggregate_results(res_df)
        save_results(res_df, agg_df, tag=variant, output_dir=output_dir)

    write_result_index(
        output_dir,
        source_stage=f'dl_candidate_smoke_{args.mode}',
        description=f'DL candidate smoke test for {args.mode}.',
        models=args.models,
        variants=args.variants,
        source_priority=6,
    )


if __name__ == '__main__':
    main()
