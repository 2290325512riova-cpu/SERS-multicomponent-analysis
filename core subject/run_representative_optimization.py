#!/usr/bin/env python3
"""Run bounded single-model optimization trials for stage 02.

Each trial is isolated under 02_representative_model_optimization/trials/<trial_id>
so trial-level search does not pollute the active stage root. The stage root only
stores machine-readable search summaries such as trial_registry.csv and
selected_config_manifest.csv.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_mainline_formal_experiment import _load_meta
from src.config import (
    DATA_VERSION,
    MAINLINE_OPTIMIZATION_DIR,
    PROCESSED_DIR,
    TASK_PROFILE,
    TASKS,
)
from src.result_index import write_result_index
from src.train_eval import aggregate_results, run_cv_evaluation, save_results

TASK_SCOPE = 'P1-P3+G1-G3'
G2_TASK_ID = 'G2_mg_molar_grade'
TRIALS_DIR = MAINLINE_OPTIMIZATION_DIR / 'trials'
TRIAL_REGISTRY_PATH = MAINLINE_OPTIMIZATION_DIR / 'trial_registry.csv'
SELECTED_CONFIG_MANIFEST_PATH = MAINLINE_OPTIMIZATION_DIR / 'selected_config_manifest.csv'

TRIAL_REGISTRY_COLUMNS = [
    'trial_id',
    'trial_order',
    'phase',
    'model_family',
    'preprocess',
    'task_scope',
    'params_json',
    'mean_f1_all',
    'mean_f1_g2',
    'fold_stats',
    'status',
    'notes',
    'duration_sec',
    'relative_dir',
]

SELECTED_CONFIG_COLUMNS = [
    'selection_stage',
    'model_family',
    'selected_trial_id',
    'selected_phase',
    'selection_metric_primary',
    'selection_metric_secondary',
    'preprocess',
    'params_json',
    'mean_f1_all',
    'mean_f1_g2',
    'ready_for_stage03',
    'ready_reason',
    'relative_dir',
    'notes',
]


def _rf_trial_specs() -> list[dict]:
    return [
        {
            'trial_id': 'rf_coarse_01',
            'phase': 'coarse',
            'model_family': 'RF',
            'preprocess': 'p2',
            'params': {
                'n_estimators': 400,
                'max_depth': None,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
            },
            'notes': 'RF coarse anchor on p2 using optimization-profile defaults.',
        },
        {
            'trial_id': 'rf_coarse_02',
            'phase': 'coarse',
            'model_family': 'RF',
            'preprocess': 'p4',
            'params': {
                'n_estimators': 400,
                'max_depth': None,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
            },
            'notes': 'RF coarse anchor on p4 using optimization-profile defaults.',
        },
        {
            'trial_id': 'rf_coarse_03',
            'phase': 'coarse',
            'model_family': 'RF',
            'preprocess': 'p2',
            'params': {
                'n_estimators': 600,
                'max_depth': 20,
                'min_samples_leaf': 1,
                'max_features': 0.5,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
            },
            'notes': 'RF coarse high-capacity p2 trial.',
        },
        {
            'trial_id': 'rf_coarse_04',
            'phase': 'coarse',
            'model_family': 'RF',
            'preprocess': 'p4',
            'params': {
                'n_estimators': 600,
                'max_depth': 20,
                'min_samples_leaf': 1,
                'max_features': 0.5,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
            },
            'notes': 'RF coarse high-capacity p4 trial.',
        },
        {
            'trial_id': 'rf_coarse_05',
            'phase': 'coarse',
            'model_family': 'RF',
            'preprocess': 'p2',
            'params': {
                'n_estimators': 300,
                'max_depth': 12,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'class_weight': 'balanced_subsample',
                'random_state': 42,
                'n_jobs': -1,
            },
            'notes': 'RF coarse conservative p2 trial.',
        },
        {
            'trial_id': 'rf_coarse_06',
            'phase': 'coarse',
            'model_family': 'RF',
            'preprocess': 'p4',
            'params': {
                'n_estimators': 300,
                'max_depth': 12,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'class_weight': 'balanced_subsample',
                'random_state': 42,
                'n_jobs': -1,
            },
            'notes': 'RF coarse conservative p4 trial.',
        },
        {
            'trial_id': 'rf_coarse_07',
            'phase': 'coarse',
            'model_family': 'RF',
            'preprocess': 'p2',
            'params': {
                'n_estimators': 800,
                'max_depth': 28,
                'min_samples_leaf': 4,
                'max_features': 0.25,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
            },
            'notes': 'RF coarse deep p2 trial with lower feature fraction.',
        },
        {
            'trial_id': 'rf_coarse_08',
            'phase': 'coarse',
            'model_family': 'RF',
            'preprocess': 'p4',
            'params': {
                'n_estimators': 800,
                'max_depth': 28,
                'min_samples_leaf': 6,
                'max_features': 0.25,
                'class_weight': 'balanced_subsample',
                'random_state': 42,
                'n_jobs': -1,
            },
            'notes': 'RF coarse deep p4 trial with stronger regularization.',
        },
    ]


def _spectrum_kan_screening_default_params() -> dict:
    return {
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
    }


def _spectrum_kan_anchor_repro_specs() -> list[dict]:
    anchor = _spectrum_kan_screening_default_params()
    return [
        {
            'trial_id': 'skan_anchor_repro_01',
            'phase': 'anchor_repro',
            'model_family': 'Spectrum-KAN',
            'preprocess': 'p4',
            'params': dict(anchor),
            'notes': 'Spectrum-KAN exact p4 reproduction of the stage-01 screening-default strongest G2 anchor.',
        },
        {
            'trial_id': 'skan_anchor_repro_02',
            'phase': 'anchor_repro',
            'model_family': 'Spectrum-KAN',
            'preprocess': 'p1',
            'params': dict(anchor),
            'notes': 'Spectrum-KAN exact p1 reproduction of the stage-01 screening-default reference branch.',
        },
    ]


def _build_trial_plan(phase: str, model_names: list[str]) -> list[dict]:
    if phase == 'local_refine':
        raise NotImplementedError('Only coarse phase is implemented in this runner.')

    if phase == 'coarse':
        plans = {
            'RF': _rf_trial_specs(),
        }
        trial_order = 1
    elif phase == 'anchor_repro':
        plans = {
            'Spectrum-KAN': _spectrum_kan_anchor_repro_specs(),
        }
        trial_order = 101
    else:
        raise ValueError(f'Unsupported phase: {phase}')

    trial_plan = []
    for model_name in model_names:
        if model_name not in plans:
            raise ValueError(
                f"Phase '{phase}' does not define an active trial plan for model '{model_name}'."
            )
        for spec in plans[model_name]:
            row = dict(spec)
            row['trial_order'] = trial_order
            trial_plan.append(row)
            trial_order += 1
    return trial_plan


def _load_registry() -> pd.DataFrame:
    if TRIAL_REGISTRY_PATH.exists():
        return pd.read_csv(TRIAL_REGISTRY_PATH)
    return pd.DataFrame(columns=TRIAL_REGISTRY_COLUMNS)


def _save_registry(registry: pd.DataFrame) -> None:
    MAINLINE_OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)
    ordered = registry.copy()
    for col in TRIAL_REGISTRY_COLUMNS:
        if col not in ordered.columns:
            ordered[col] = ''
    ordered = ordered[TRIAL_REGISTRY_COLUMNS].sort_values('trial_order', kind='stable')
    ordered.to_csv(TRIAL_REGISTRY_PATH, index=False, encoding='utf-8-sig')


def _save_selected_manifest(registry: pd.DataFrame) -> None:
    rows = []
    completed = registry[registry['status'] == 'completed'].copy()
    for model_name in ['RF', 'Spectrum-KAN']:
        sub = completed[completed['model_family'] == model_name].copy()
        if sub.empty:
            rows.append({
                'selection_stage': '02_representative_model_optimization',
                'model_family': model_name,
                'selected_trial_id': '',
                'selected_phase': '',
                'selection_metric_primary': 'mean_f1_all' if model_name == 'RF' else 'mean_f1_g2',
                'selection_metric_secondary': 'mean_f1_g2' if model_name == 'RF' else 'mean_f1_all',
                'preprocess': '',
                'params_json': '',
                'mean_f1_all': np.nan,
                'mean_f1_g2': np.nan,
                'ready_for_stage03': False,
                'ready_reason': 'no_completed_trial',
                'relative_dir': '',
                'notes': 'No completed trial yet.',
            })
            continue

        primary = 'mean_f1_all' if model_name == 'RF' else 'mean_f1_g2'
        secondary = 'mean_f1_g2' if model_name == 'RF' else 'mean_f1_all'
        sub = sub.sort_values([primary, secondary, 'trial_order'], ascending=[False, False, True])
        best = sub.iloc[0]
        ready, ready_reason = _is_ready_for_stage03(sub)
        rows.append({
            'selection_stage': '02_representative_model_optimization',
            'model_family': model_name,
            'selected_trial_id': best['trial_id'],
            'selected_phase': best['phase'],
            'selection_metric_primary': primary,
            'selection_metric_secondary': secondary,
            'preprocess': best['preprocess'],
            'params_json': best['params_json'],
            'mean_f1_all': best['mean_f1_all'],
            'mean_f1_g2': best['mean_f1_g2'],
            'ready_for_stage03': ready,
            'ready_reason': ready_reason,
            'relative_dir': best['relative_dir'],
            'notes': best['notes'],
        })

    manifest = pd.DataFrame(rows)
    for col in SELECTED_CONFIG_COLUMNS:
        if col not in manifest.columns:
            manifest[col] = ''
    manifest = manifest[SELECTED_CONFIG_COLUMNS]
    manifest.to_csv(SELECTED_CONFIG_MANIFEST_PATH, index=False, encoding='utf-8-sig')


def _is_ready_for_stage03(model_trials: pd.DataFrame) -> tuple[bool, str]:
    ordered = model_trials.sort_values('trial_order', kind='stable')
    local_refine = ordered[ordered['phase'] == 'local_refine']
    if local_refine.empty:
        if (ordered['phase'] == 'anchor_repro').any():
            return False, 'anchor_repro_completed_pending_refine'
        return False, 'coarse_only'

    best_g2 = float('-inf')
    best_all = float('-inf')
    no_improve_streak = 0
    for _, row in ordered.iterrows():
        improved = (
            row['mean_f1_g2'] >= best_g2 + 0.02 or
            row['mean_f1_all'] >= best_all + 0.01
        )
        best_g2 = max(best_g2, float(row['mean_f1_g2']))
        best_all = max(best_all, float(row['mean_f1_all']))

        if row['phase'] != 'local_refine':
            continue
        if improved:
            no_improve_streak = 0
        else:
            no_improve_streak += 1
            if no_improve_streak >= 4:
                return True, 'local_refine_stop_rule_triggered'
    return False, 'local_refine_not_exhausted'


def _trial_output_dir(trial_id: str) -> Path:
    return TRIALS_DIR / trial_id


def _trial_model_kwargs(model_name: str, params: dict) -> dict[str, dict]:
    if model_name == 'Spectrum-KAN':
        return {model_name: {'train_cfg': dict(params)}}
    return {model_name: dict(params)}


def _fold_stats_json(summary_df: pd.DataFrame) -> str:
    payload = {}
    for _, row in summary_df.iterrows():
        payload[str(row['Task'])] = {
            'task_name': str(row['TaskName']),
            'f1_mean': float(row['F1_mean']),
            'f1_std': float(row['F1_std']),
        }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _run_single_trial(spec: dict, meta: pd.DataFrame, wn: np.ndarray) -> dict:
    trial_dir = _trial_output_dir(spec['trial_id'])
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    preprocess = spec['preprocess']
    matrix = None
    duration_sec = 0.0
    try:
        X = np.load(PROCESSED_DIR / f"X_{preprocess}.npy")
        start = time.time()
        res_df = run_cv_evaluation(
            meta,
            X,
            model_names=[spec['model_family']],
            preprocess_tag=preprocess,
            wn=wn,
            tasks_override=TASKS,
            model_kwargs=_trial_model_kwargs(spec['model_family'], spec['params']),
            use_random_split=False,
        )
        agg_df = aggregate_results(res_df)
        save_results(res_df, agg_df, tag=preprocess, output_dir=trial_dir)
        matrix, _ = write_result_index(
            trial_dir,
            source_stage=spec['trial_id'],
            description=f"Representative optimization trial {spec['trial_id']}",
            models=[spec['model_family']],
            variants=[preprocess],
            source_priority=1,
        )
        duration_sec = time.time() - start
        mean_f1_all = float(matrix['F1_mean'].mean())
        g2_rows = matrix[matrix['Task'] == G2_TASK_ID]
        mean_f1_g2 = float(g2_rows.iloc[0]['F1_mean']) if not g2_rows.empty else np.nan
        return {
            'trial_id': spec['trial_id'],
            'trial_order': spec['trial_order'],
            'phase': spec['phase'],
            'model_family': spec['model_family'],
            'preprocess': preprocess,
            'task_scope': TASK_SCOPE,
            'params_json': json.dumps(spec['params'], ensure_ascii=False, sort_keys=True),
            'mean_f1_all': mean_f1_all,
            'mean_f1_g2': mean_f1_g2,
            'fold_stats': _fold_stats_json(matrix),
            'status': 'completed',
            'notes': spec['notes'],
            'duration_sec': round(duration_sec, 2),
            'relative_dir': str(trial_dir.relative_to(MAINLINE_OPTIMIZATION_DIR)).replace('\\', '/'),
        }
    except Exception as exc:
        return {
            'trial_id': spec['trial_id'],
            'trial_order': spec['trial_order'],
            'phase': spec['phase'],
            'model_family': spec['model_family'],
            'preprocess': preprocess,
            'task_scope': TASK_SCOPE,
            'params_json': json.dumps(spec['params'], ensure_ascii=False, sort_keys=True),
            'mean_f1_all': np.nan,
            'mean_f1_g2': np.nan,
            'fold_stats': '',
            'status': 'failed',
            'notes': f"{spec['notes']} | error: {exc}",
            'duration_sec': round(duration_sec, 2),
            'relative_dir': str(trial_dir.relative_to(MAINLINE_OPTIMIZATION_DIR)).replace('\\', '/'),
        }


def _upsert_trial_row(registry: pd.DataFrame, row: dict) -> pd.DataFrame:
    registry = registry.copy()
    registry = registry[registry['trial_id'] != row['trial_id']]
    registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    return registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run bounded stage-02 optimization trials.')
    parser.add_argument(
        '--phase',
        choices=['coarse', 'anchor_repro', 'local_refine'],
        default='coarse',
        help='Optimization phase to execute.',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['RF', 'Spectrum-KAN'],
        help='Representative models to optimize.',
    )
    parser.add_argument(
        '--trial-id',
        nargs='*',
        default=None,
        help='Only run the specified trial ids.',
    )
    parser.add_argument(
        '--limit-trials',
        type=int,
        default=None,
        help='Run only the first N trials from the selected plan.',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip trials already marked completed in trial_registry.csv.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the selected trial plan without running evaluation.',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    MAINLINE_OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)
    TRIALS_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Data version: {DATA_VERSION}', flush=True)
    print(f'Task profile: {TASK_PROFILE}', flush=True)
    print(f'Optimization phase: {args.phase}', flush=True)
    print(f'Models: {args.models}', flush=True)
    print(f'Stage root: {MAINLINE_OPTIMIZATION_DIR}', flush=True)

    registry = _load_registry()
    trial_plan = _build_trial_plan(args.phase, args.models)
    if args.trial_id:
        chosen_ids = set(args.trial_id)
        trial_plan = [spec for spec in trial_plan if spec['trial_id'] in chosen_ids]
    if args.limit_trials is not None:
        trial_plan = trial_plan[: args.limit_trials]

    if args.skip_existing and not registry.empty:
        completed_ids = set(registry.loc[registry['status'] == 'completed', 'trial_id'])
        trial_plan = [spec for spec in trial_plan if spec['trial_id'] not in completed_ids]

    if not trial_plan:
        print('No trials selected.', flush=True)
        _save_selected_manifest(registry)
        return

    print('Selected trials:', flush=True)
    for spec in trial_plan:
        print(
            f"  - {spec['trial_id']}: {spec['model_family']} on {spec['preprocess']} | {spec['notes']}",
            flush=True,
        )

    if args.dry_run:
        return

    meta = _load_meta()
    wn = np.load(PROCESSED_DIR / 'wavenumber.npy')
    total_start = time.time()
    for spec in trial_plan:
        print(f"\n{'=' * 60}", flush=True)
        print(f"Trial {spec['trial_id']} | {spec['model_family']} | {spec['preprocess']}", flush=True)
        print(f"Params: {spec['params']}", flush=True)
        print(f"{'=' * 60}", flush=True)
        row = _run_single_trial(spec, meta, wn)
        registry = _upsert_trial_row(registry, row)
        _save_registry(registry)
        _save_selected_manifest(registry)
        print(
            f"Trial {row['trial_id']} finished with status={row['status']} | "
            f"mean_f1_all={row['mean_f1_all']} | mean_f1_g2={row['mean_f1_g2']}",
            flush=True,
        )

    print(f"\nAll selected trials complete in {(time.time() - total_start) / 60:.1f} minutes", flush=True)
    print(f'Trial registry: {TRIAL_REGISTRY_PATH}', flush=True)
    print(f'Selected config manifest: {SELECTED_CONFIG_MANIFEST_PATH}', flush=True)


if __name__ == '__main__':
    main()