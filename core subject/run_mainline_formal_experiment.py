#!/usr/bin/env python3
"""Mainline formal experiment runner.

Stages:
    - candidate_screening
    - representative_model_optimization
    - locked_grouped_main_results
    - random_control
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.config import (  # noqa: E402
    ACTIVE_SPLIT_FILE,
    DATA_VERSION,
    MAINLINE_LOCKED_DIR,
    MAINLINE_OPTIMIZATION_DIR,
    MAINLINE_RANDOM_DIR,
    MAINLINE_SCREENING_DIR,
    PROCESSED_DIR,
    PREPROCESS_TAGS,
    SPLITS_DIR,
    TASKS,
    TASKS_SUPPLEMENTARY,
    TASK_PROFILE,
)
from src.result_index import write_result_index  # noqa: E402
from src.train_eval import aggregate_results, run_cv_evaluation, save_results  # noqa: E402


CANDIDATE_MODELS = [
    'PLS-DA',
    'SVM',
    'RF',
    'KNN',
    'HistGradientBoosting',
    '1D-CNN',
    'Spectrum-KAN',
    'Chem-KAN',
]
DEFAULT_VARIANTS = list(PREPROCESS_TAGS)
TRAIN_CFG_KEYS = {
    'lr',
    'epochs',
    'patience',
    'batch_size',
    'weight_decay',
    'val_frac',
    'scheduler',
    'optimizer',
    'step_size',
    'step_gamma',
    'aug_enabled',
    'aug_n',
    'mixup_enabled',
    'mixup_alpha',
    'label_smoothing',
    'focal_gamma',
    'use_class_weight',
}
OPTIMIZATION_PROFILE = {
    'RF': {
        'n_estimators': 800,
        'max_depth': 28,
        'min_samples_leaf': 6,
        'max_features': 0.25,
        'class_weight': 'balanced_subsample',
    },
    'SVM': {
        'C': 8.0,
        'gamma': 'scale',
    },
    'PLS-DA': {
        'n_components': 8,
    },
    'KNN': {
        'n_neighbors': 5,
        'weights': 'distance',
    },
    'HistGradientBoosting': {
        'max_iter': 450,
        'learning_rate': 0.03,
        'min_samples_leaf': 5,
        'l2_regularization': 1e-2,
    },
    '1D-CNN': {
        'train_cfg': {
            'optimizer': 'adamw',
            'scheduler': 'step',
            'lr': 2e-4,
            'weight_decay': 3e-4,
            'batch_size': 96,
            'patience': 30,
            'step_size': 40,
            'step_gamma': 0.7,
            'aug_enabled': True,
            'aug_n': 2,
            'mixup_enabled': False,
            'label_smoothing': 0.05,
            'focal_gamma': 1.5,
        },
    },
    'Spectrum-KAN': {
        'train_cfg': {
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'lr': 3e-4,
            'epochs': 200,
            'patience': 20,
            'batch_size': 128,
            'weight_decay': 1e-4,
            'val_frac': 0.15,
            'aug_enabled': True,
            'aug_n': 4,
            'mixup_enabled': True,
            'mixup_alpha': 0.3,
            'label_smoothing': 0.1,
            'focal_gamma': 2.0,
            'use_class_weight': True,
        },
    },
    'Chem-KAN': {
        'train_cfg': {
            'optimizer': 'adamw',
            'scheduler': 'step',
            'lr': 2.5e-4,
            'weight_decay': 4e-4,
            'batch_size': 96,
            'patience': 30,
            'step_size': 40,
            'step_gamma': 0.7,
            'aug_enabled': True,
            'aug_n': 2,
            'mixup_enabled': False,
            'label_smoothing': 0.05,
            'focal_gamma': 1.5,
        },
    },
}
TRAINING_PROFILES = {
    'screening': {},
    'optimization': OPTIMIZATION_PROFILE,
    'locked': OPTIMIZATION_PROFILE,
}
STAGE_CONFIG = {
    'candidate_screening': {
        'output_dir': MAINLINE_SCREENING_DIR,
        'description': 'Mainline candidate screening under grouped CV.',
        'source_priority': 1,
        'default_models': CANDIDATE_MODELS,
        'default_profile': 'screening',
        'default_split_mode': 'grouped',
    },
    'representative_model_optimization': {
        'output_dir': MAINLINE_OPTIMIZATION_DIR,
        'description': 'Fair grouped optimization on selected representative models.',
        'source_priority': 2,
        'default_models': None,
        'default_profile': 'optimization',
        'default_split_mode': 'grouped',
    },
    'locked_grouped_main_results': {
        'output_dir': MAINLINE_LOCKED_DIR,
        'description': 'Locked grouped main results for the final mainline comparison pool.',
        'source_priority': 3,
        'default_models': None,
        'default_profile': 'locked',
        'default_split_mode': 'grouped',
    },
    'random_control': {
        'output_dir': MAINLINE_RANDOM_DIR,
        'description': 'Random-split control for leakage comparison after grouped results are locked.',
        'source_priority': 1,
        'default_models': CANDIDATE_MODELS,
        'default_profile': 'screening',
        'default_split_mode': 'random',
    },
}


def _clone_model_kwargs(model_kwargs: dict[str, dict]) -> dict[str, dict]:
    cloned = {}
    for model_name, kwargs in model_kwargs.items():
        cloned[model_name] = {}
        for key, value in kwargs.items():
            cloned[model_name][key] = dict(value) if isinstance(value, dict) else value
    return cloned


def _parse_scalar(value: str):
    raw = value.strip()
    lowered = raw.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    if lowered in {'none', 'null'}:
        return None
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _build_model_kwargs(profile_name: str, raw_overrides: list[str]) -> dict[str, dict]:
    model_kwargs = _clone_model_kwargs(TRAINING_PROFILES[profile_name])
    for item in raw_overrides:
        if '=' not in item or '.' not in item:
            raise ValueError(f"Invalid --model-param '{item}'. Expected MODEL.KEY=VALUE.")

        lhs, raw_value = item.split('=', 1)
        model_name, key = lhs.rsplit('.', 1)
        value = _parse_scalar(raw_value)

        if key in TRAIN_CFG_KEYS:
            model_kwargs.setdefault(model_name, {}).setdefault('train_cfg', {})[key] = value
        else:
            model_kwargs.setdefault(model_name, {})[key] = value
    return model_kwargs


def _load_meta() -> pd.DataFrame:
    split_path = SPLITS_DIR / ACTIVE_SPLIT_FILE
    meta = pd.read_csv(split_path)
    for col in ['has_thiram', 'has_mg', 'has_mba']:
        if col in meta.columns:
            meta[col] = meta[col].astype(str).str.strip().str.lower().map(
                {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
            ).fillna(0).astype(int)
    return meta


def _run_task_group(meta, X, wn, preprocess_tag, model_names, tasks_override,
                    output_dir, use_random_split, model_kwargs):
    res_df = run_cv_evaluation(
        meta,
        X,
        model_names=model_names,
        preprocess_tag=preprocess_tag,
        wn=wn,
        tasks_override=tasks_override,
        model_kwargs=model_kwargs,
        use_random_split=use_random_split,
    )
    agg_df = aggregate_results(res_df)
    save_results(res_df, agg_df, tag=preprocess_tag, output_dir=output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the current mainline formal experiment.')
    parser.add_argument(
        '--stage',
        choices=sorted(STAGE_CONFIG.keys()),
        default='candidate_screening',
        help='Mainline stage to run.',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Explicit model names to evaluate. Required for stages after candidate screening unless you accept the stage default.',
    )
    parser.add_argument(
        '--variants',
        nargs='+',
        default=DEFAULT_VARIANTS,
        help='Preprocessing variants to evaluate.',
    )
    parser.add_argument(
        '--include-supplementary',
        action='store_true',
        help='Also evaluate supplementary tasks such as mixture order.',
    )
    parser.add_argument(
        '--training-profile',
        choices=sorted(TRAINING_PROFILES.keys()),
        default=None,
        help='Named profile for stage-specific model overrides.',
    )
    parser.add_argument(
        '--model-param',
        action='append',
        default=[],
        metavar='MODEL.KEY=VALUE',
        help='Override a model parameter, e.g. RF.n_estimators=500 or Chem-KAN.lr=2e-4.',
    )
    parser.add_argument(
        '--split-mode',
        choices=['grouped', 'random'],
        default=None,
        help='Override the default evaluation protocol for the current stage.',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stage_cfg = STAGE_CONFIG[args.stage]
    models = args.models or stage_cfg['default_models']
    if models is None:
        raise ValueError(
            f"Stage '{args.stage}' requires an explicit --models list because the representative pool must be chosen from prior screening evidence."
        )

    split_mode = args.split_mode or stage_cfg['default_split_mode']
    profile_name = args.training_profile or stage_cfg['default_profile']
    use_random_split = split_mode == 'random'
    model_kwargs = _build_model_kwargs(profile_name, args.model_param)
    output_dir = stage_cfg['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_meta()
    wn = np.load(PROCESSED_DIR / 'wavenumber.npy')

    print(f"Data version: {DATA_VERSION}", flush=True)
    print(f"Task profile: {TASK_PROFILE}", flush=True)
    print(f"Stage: {args.stage}", flush=True)
    print(f"Split mode: {split_mode}", flush=True)
    print(f"Models: {models}", flush=True)
    print(f"Training profile: {profile_name}", flush=True)
    if model_kwargs:
        print(f"Model overrides: {model_kwargs}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)

    total_start = time.time()
    for tag in args.variants:
        path = PROCESSED_DIR / f'X_{tag}.npy'
        if not path.exists():
            print(f"Skip {tag}: {path} not found", flush=True)
            continue

        X = np.load(path)
        print(f"\n{'=' * 60}\nMainline stage {args.stage} on {tag}\n{'=' * 60}", flush=True)
        t0 = time.time()

        _run_task_group(
            meta,
            X,
            wn,
            tag,
            models,
            tasks_override=TASKS,
            output_dir=output_dir,
            use_random_split=use_random_split,
            model_kwargs=model_kwargs,
        )

        if args.include_supplementary and TASKS_SUPPLEMENTARY:
            _run_task_group(
                meta,
                X,
                wn,
                tag,
                models,
                tasks_override=TASKS_SUPPLEMENTARY,
                output_dir=output_dir,
                use_random_split=use_random_split,
                model_kwargs=model_kwargs,
            )

        print(f"Completed {tag} in {time.time() - t0:.0f}s", flush=True)

    matrix, best = write_result_index(
        output_dir,
        source_stage=args.stage,
        description=stage_cfg['description'],
        models=models,
        variants=args.variants,
        source_priority=stage_cfg['source_priority'],
    )
    print(f"Indexed {len(matrix)} rows and {len(best)} best-task rows in {output_dir}", flush=True)
    print(f"\nAll mainline runs complete in {(time.time() - total_start) / 60:.1f} minutes", flush=True)


if __name__ == '__main__':
    main()
