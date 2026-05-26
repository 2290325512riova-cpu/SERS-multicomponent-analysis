#!/usr/bin/env python3
"""NSGA-II Multi-Objective Optimization for stage 02 closure.

Uses Optuna with NSGAIISampler to simultaneously optimize:
  - objective_1: mean_f1_g2 (hardest slice)
  - objective_2: mean_f1_all (overall performance)

Supports both RF and Spectrum-KAN with ordinal loss variants.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import NSGAIISampler

from run_mainline_formal_experiment import _load_meta
from run_representative_optimization import (
    G2_TASK_ID,
    MAINLINE_OPTIMIZATION_DIR,
    SELECTED_CONFIG_MANIFEST_PATH,
    TASK_SCOPE,
    TRIAL_REGISTRY_COLUMNS,
    TRIAL_REGISTRY_PATH,
    TRIALS_DIR,
    _fold_stats_json,
    _load_registry,
    _save_registry,
    _save_selected_manifest,
    _trial_output_dir,
    _upsert_trial_row,
)
from src.config import PROCESSED_DIR, TASKS
from src.result_index import write_result_index
from src.train_eval import aggregate_results, run_cv_evaluation, save_results

STUDY_DIR = MAINLINE_OPTIMIZATION_DIR / 'optuna_studies'


def _kan_objective(trial: optuna.Trial, meta: pd.DataFrame, X: np.ndarray,
                   wn: np.ndarray) -> tuple[float, float]:
    """Spectrum-KAN objective: returns (mean_f1_g2, mean_f1_all)."""
    params = {
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'lr': trial.suggest_float('lr', 1e-4, 5e-4, log=True),
        'epochs': 200,
        'patience': 20,
        'batch_size': 128,
        'weight_decay': trial.suggest_float('weight_decay', 5e-5, 3e-3, log=True),
        'val_frac': 0.15,
        'aug_enabled': True,
        'aug_n': trial.suggest_categorical('aug_n', [2, 4]),
        'mixup_enabled': True,
        'mixup_alpha': trial.suggest_float('mixup_alpha', 0.1, 0.4),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.15),
        'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0),
        'use_class_weight': True,
        'loss_type': 'focal',
        'class_balance': trial.suggest_categorical('class_balance', ['simple', 'effective_number']),
    }

    model_kwargs = {'Spectrum-KAN': {'train_cfg': params}}
    res_df = run_cv_evaluation(
        meta, X,
        model_names=['Spectrum-KAN'],
        preprocess_tag='p4',
        wn=wn,
        tasks_override=TASKS,
        model_kwargs=model_kwargs,
        use_random_split=False,
    )
    agg_df = aggregate_results(res_df)
    mean_f1_all = float(agg_df['F1_mean'].mean())
    g2_rows = agg_df[agg_df['Task'] == G2_TASK_ID]
    mean_f1_g2 = float(g2_rows.iloc[0]['F1_mean']) if not g2_rows.empty else 0.0
    return mean_f1_g2, mean_f1_all


def _rf_objective(trial: optuna.Trial, meta: pd.DataFrame, X: np.ndarray,
                  wn: np.ndarray) -> tuple[float, float]:
    """RF objective: returns (mean_f1_g2, mean_f1_all)."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1200, step=100),
        'max_depth': trial.suggest_int('max_depth', 8, 35),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'max_features': trial.suggest_float('max_features', 0.1, 0.5),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
        'random_state': 42,
        'n_jobs': -1,
    }
    model_kwargs = {'RF': params}
    res_df = run_cv_evaluation(
        meta, X,
        model_names=['RF'],
        preprocess_tag='p4',
        wn=wn,
        tasks_override=TASKS,
        model_kwargs=model_kwargs,
        use_random_split=False,
    )
    agg_df = aggregate_results(res_df)
    mean_f1_all = float(agg_df['F1_mean'].mean())
    g2_rows = agg_df[agg_df['Task'] == G2_TASK_ID]
    mean_f1_g2 = float(g2_rows.iloc[0]['F1_mean']) if not g2_rows.empty else 0.0
    return mean_f1_g2, mean_f1_all


def _run_optimization(model: str, n_trials: int, meta: pd.DataFrame,
                      X: np.ndarray, wn: np.ndarray) -> optuna.Study:
    """Run NSGA-II optimization for a given model."""
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    storage_path = STUDY_DIR / f'nsga_{model.lower().replace("-", "_")}.db'
    storage_url = f'sqlite:///{storage_path}'

    study = optuna.create_study(
        study_name=f'nsga_{model}',
        directions=['maximize', 'maximize'],
        sampler=NSGAIISampler(population_size=20, seed=42),
        storage=storage_url,
        load_if_exists=True,
    )

    # Seed with anchor config so search starts from known-best
    if len(study.trials) == 0:
        if model == 'Spectrum-KAN':
            study.enqueue_trial({
                'lr': 3e-4, 'weight_decay': 1e-4, 'aug_n': 4,
                'mixup_alpha': 0.3, 'label_smoothing': 0.1,
                'focal_gamma': 2.0, 'class_balance': 'simple',
            })
        elif model == 'RF':
            study.enqueue_trial({
                'n_estimators': 800, 'max_depth': 28,
                'min_samples_leaf': 6, 'max_features': 0.25,
                'class_weight': 'balanced_subsample',
            })

    if model == 'Spectrum-KAN':
        objective_fn = lambda trial: _kan_objective(trial, meta, X, wn)
    elif model == 'RF':
        objective_fn = lambda trial: _rf_objective(trial, meta, X, wn)
    else:
        raise ValueError(f'Unsupported model: {model}')

    remaining = max(0, n_trials - len(study.trials))
    if remaining > 0:
        print(f'Running {remaining} trials for {model} (already have {len(study.trials)})')
        study.optimize(objective_fn, n_trials=remaining, show_progress_bar=True)
    else:
        print(f'{model} already has {len(study.trials)} trials, skipping.')

    return study


def _save_pareto_results(study: optuna.Study, model: str, meta: pd.DataFrame,
                         X: np.ndarray, wn: np.ndarray) -> None:
    """Save Pareto-optimal trials to trial_registry and trial directories."""
    pareto_trials = study.best_trials
    if not pareto_trials:
        print(f'No Pareto-optimal trials found for {model}.')
        return

    registry = _load_registry()
    print(f'\n{model}: {len(pareto_trials)} Pareto-optimal trials found.')

    for i, trial in enumerate(pareto_trials):
        trial_id = f'{model.lower().replace("-", "_")}_nsga_best_{i+1:02d}'
        g2_val, all_val = trial.values
        print(f'  {trial_id}: G2={g2_val:.4f}, All={all_val:.4f}, params={trial.params}')

        trial_dir = _trial_output_dir(trial_id)
        if trial_dir.exists():
            import shutil
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        if model == 'Spectrum-KAN':
            params = dict(trial.params)
            params.update({'optimizer': 'adam', 'scheduler': 'cosine', 'epochs': 200,
                          'patience': 20, 'batch_size': 128, 'val_frac': 0.15,
                          'use_class_weight': True, 'aug_enabled': True, 'mixup_enabled': True})
            model_kwargs = {'Spectrum-KAN': {'train_cfg': params}}
        else:
            params = dict(trial.params)
            params['random_state'] = 42
            params['n_jobs'] = -1
            model_kwargs = {'RF': params}

        res_df = run_cv_evaluation(
            meta, X,
            model_names=[model],
            preprocess_tag='p4',
            wn=wn,
            tasks_override=TASKS,
            model_kwargs=model_kwargs,
            use_random_split=False,
        )
        agg_df = aggregate_results(res_df)
        save_results(res_df, agg_df, tag='p4', output_dir=trial_dir)
        matrix, _ = write_result_index(
            trial_dir,
            source_stage=trial_id,
            description=f'NSGA-II Pareto trial {trial_id}',
            models=[model],
            variants=['p4'],
            source_priority=1,
        )

        mean_f1_all = float(matrix['F1_mean'].mean())
        g2_rows = matrix[matrix['Task'] == G2_TASK_ID]
        mean_f1_g2 = float(g2_rows.iloc[0]['F1_mean']) if not g2_rows.empty else np.nan

        row = {
            'trial_id': trial_id,
            'trial_order': 200 + i,
            'phase': 'nsga_refine',
            'model_family': model,
            'preprocess': 'p4',
            'task_scope': TASK_SCOPE,
            'params_json': json.dumps(params, ensure_ascii=False, sort_keys=True, default=str),
            'mean_f1_all': mean_f1_all,
            'mean_f1_g2': mean_f1_g2,
            'fold_stats': _fold_stats_json(matrix),
            'status': 'completed',
            'notes': f'NSGA-II Pareto trial #{i+1}',
            'duration_sec': trial.duration.total_seconds() if trial.duration else 0,
            'relative_dir': str(trial_dir.relative_to(MAINLINE_OPTIMIZATION_DIR)).replace('\\', '/'),
        }
        registry = _upsert_trial_row(registry, row)

    _save_registry(registry)
    _save_selected_manifest(registry)


def _plot_pareto_front(study: optuna.Study, model: str) -> None:
    """Generate Pareto front plot (F6b)."""
    try:
        from optuna.visualization.matplotlib import plot_pareto_front as plot_pf
        import matplotlib.pyplot as plt

        fig = plot_pf(study, target_names=['mean_f1_g2', 'mean_f1_all'])
        out_path = MAINLINE_OPTIMIZATION_DIR / f'pareto_front_{model.lower().replace("-","_")}.png'
        fig.figure.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Pareto front plot saved: {out_path}')
    except Exception as e:
        print(f'Could not generate Pareto plot: {e}')

    try:
        from optuna.visualization.matplotlib import plot_param_importances
        import matplotlib.pyplot as plt

        fig = plot_param_importances(study, target=lambda t: t.values[0],
                                     target_name='mean_f1_g2')
        out_path = MAINLINE_OPTIMIZATION_DIR / f'importance_g2_{model.lower().replace("-","_")}.png'
        fig.figure.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Importance plot saved: {out_path}')
    except Exception as e:
        print(f'Could not generate importance plot: {e}')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='NSGA-II multi-objective optimization for stage 02.')
    parser.add_argument('--model', choices=['RF', 'Spectrum-KAN', 'both'], default='both')
    parser.add_argument('--n-trials-kan', type=int, default=50)
    parser.add_argument('--n-trials-rf', type=int, default=100)
    parser.add_argument('--save-pareto', action='store_true', default=True,
                        help='Re-run and save Pareto-optimal trials with full CV output.')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate plots from existing study, no new trials.')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    TRIALS_DIR.mkdir(parents=True, exist_ok=True)

    meta = _load_meta()
    X = np.load(PROCESSED_DIR / 'X_p4.npy')
    wn = np.load(PROCESSED_DIR / 'wavenumber.npy')

    models_to_run = []
    if args.model in ('RF', 'both'):
        models_to_run.append(('RF', args.n_trials_rf))
    if args.model in ('Spectrum-KAN', 'both'):
        models_to_run.append(('Spectrum-KAN', args.n_trials_kan))

    for model, n_trials in models_to_run:
        print(f'\n{"="*60}')
        print(f'  NSGA-II Optimization: {model} ({n_trials} trials)')
        print(f'{"="*60}')

        if args.plot_only:
            storage_path = STUDY_DIR / f'nsga_{model.lower().replace("-", "_")}.db'
            if storage_path.exists():
                study = optuna.load_study(
                    study_name=f'nsga_{model}',
                    storage=f'sqlite:///{storage_path}',
                )
                _plot_pareto_front(study, model)
            else:
                print(f'No existing study found for {model}.')
            continue

        study = _run_optimization(model, n_trials, meta, X, wn)

        if args.save_pareto:
            _save_pareto_results(study, model, meta, X, wn)

        _plot_pareto_front(study, model)

    print('\nOptimization complete.')


if __name__ == '__main__':
    main()
