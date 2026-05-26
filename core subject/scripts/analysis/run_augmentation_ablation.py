#!/usr/bin/env python3
"""Run DL augmentation ablations under the paper-main random split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    PAPER_MAIN_AUGMENTATION_DIR,
    RANDOM_SPLIT_FILE,
    SPLITS_DIR,
    TASKS,
)
from src.dataset import load_preprocessed_variants  # noqa: E402
from src.result_index import write_result_index  # noqa: E402
from src.train_eval import aggregate_results, run_cv_evaluation, save_results  # noqa: E402

DEFAULT_DL_MODELS = ['1D-CNN', '1D-ResNet', 'Spectrum-KAN', 'KAN-CNN']
ABLATION_MODES = {
    'no_aug': {'aug_enabled': False, 'mixup_enabled': False},
    'aug_no_mixup': {'aug_enabled': True, 'aug_n': 4, 'mixup_enabled': False},
    'composition_mixup': {'aug_enabled': True, 'aug_n': 4, 'mixup_enabled': True, 'mixup_alpha': 0.3},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run augmentation ablation for DL models.')
    parser.add_argument('--models', nargs='+', default=DEFAULT_DL_MODELS)
    parser.add_argument('--variants', nargs='+', default=['p5'])
    parser.add_argument('--modes', nargs='+', default=list(ABLATION_MODES))
    return parser.parse_args()


def normalize_bool_columns(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    for col in ['has_thiram', 'has_mg', 'has_mba']:
        if col in meta.columns:
            meta[col] = meta[col].astype(str).str.strip().str.lower().map(
                {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
            ).fillna(0).astype(int)
    return meta


def model_kwargs_for(mode: str, models: list[str]) -> dict[str, dict]:
    if mode not in ABLATION_MODES:
        raise KeyError(f'Unknown ablation mode: {mode}')
    train_cfg = ABLATION_MODES[mode]
    return {model: {'train_cfg': dict(train_cfg)} for model in models}


def main() -> None:
    args = parse_args()
    split_path = SPLITS_DIR / RANDOM_SPLIT_FILE
    meta = normalize_bool_columns(pd.read_csv(split_path))
    wn, X_by_variant = load_preprocessed_variants(args.variants)

    for mode in args.modes:
        output_dir = PAPER_MAIN_AUGMENTATION_DIR / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        kwargs = model_kwargs_for(mode, args.models)
        for variant in args.variants:
            print(f'\n{"=" * 70}\nAugmentation mode={mode}, variant={variant}\n{"=" * 70}')
            res_df = run_cv_evaluation(
                meta,
                X_by_variant[variant],
                model_names=args.models,
                preprocess_tag=variant,
                wn=wn,
                tasks_override=TASKS,
                model_kwargs=kwargs,
                use_random_split=False,
            )
            agg_df = aggregate_results(res_df)
            save_results(res_df, agg_df, tag=variant, output_dir=output_dir)
        write_result_index(
            output_dir,
            source_stage=f'augmentation_{mode}',
            description=f'Paper-main augmentation ablation: {mode}.',
            models=args.models,
            variants=args.variants,
            source_priority=5,
        )


if __name__ == '__main__':
    main()
