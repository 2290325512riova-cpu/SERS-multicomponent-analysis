"""
Two-Stage Context-Aware Prediction Experiment (E5b).

Stage 1: Predict mixture context from presence predictions (P1+P2+P3).
Stage 2: Train context-specific concentration models for grade tasks.

Hypothesis: MG concentration grading (G2) is harder in multi-component
mixtures due to competitive adsorption. Context-aware models should
improve by learning context-specific spectral patterns.

Design:
  - Within each CV fold, Stage 1 uses ground-truth or predicted presence
    to determine mixture context (oracle vs realistic).
  - Stage 2 trains separate G2 models per context group.
  - Compare: global model vs context-specific vs oracle-context.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.config import (
    MAINLINE_FORMAL_DIR, SPLITS_DIR, PROCESSED_DIR,
    PREPROCESS_NAMES, N_FOLDS, TASKS, SEED,
)
from src.models import MODEL_REGISTRY

OUTPUT_DIR = MAINLINE_FORMAL_DIR / "03_two_stage_context_aware"

GRADE_TASKS_ALL = {
    'G1_thiram_molar_grade': {'filter_col': 'has_thiram', 'target_col': 'c_thiram'},
    'G2_mg_molar_grade': {'filter_col': 'has_mg', 'target_col': 'c_mg'},
    'G3_mba_molar_grade': {'filter_col': 'has_mba', 'target_col': 'c_mba'},
}
# Run all grade tasks for complete comparison
GRADE_TASKS = GRADE_TASKS_ALL

CONTEXT_MODELS = ['RF']
PREPROCESS_LIST = ['p4']


def infer_mixture_context(has_thiram, has_mg, has_mba):
    """Infer mixture context from presence labels/predictions."""
    n_present = int(has_thiram) + int(has_mg) + int(has_mba)
    if n_present == 1:
        return 'single'
    elif n_present == 2:
        if has_thiram and has_mg:
            return 'binary_Thiram_MG'
        elif has_mg and has_mba:
            return 'binary_MG_MBA'
        else:
            return 'binary_Thiram_MBA'
    else:
        return 'ternary'


def compute_metrics(y_true, y_pred):
    """Compute standard + ordinal metrics."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return {}
    return {
        'n_samples': n,
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'balanced_acc': balanced_accuracy_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'within_1_acc': float(np.mean(np.abs(y_true - y_pred) <= 1)),
        'mae_levels': float(np.mean(np.abs(y_true - y_pred))),
    }


def load_data(preprocess: str):
    """Load spectra and metadata."""
    meta = pd.read_csv(SPLITS_DIR / 'cv_split_pure63_main.csv')
    X = np.load(PROCESSED_DIR / f'X_{preprocess}.npy')
    return meta, X


def run_global_baseline(meta, X, task_id, task_info, model_name, preprocess):
    """Run standard global model as baseline (no context awareness)."""
    filter_mask = meta[task_info['filter_col']].astype(bool).values
    task_idx = np.where(filter_mask)[0]
    task_X = X[task_idx]
    task_y = meta.iloc[task_idx][task_info['target_col']].values.astype(int)
    fold_ids = meta.iloc[task_idx]['fold_id'].values

    yt_all, yp_all = [], []
    for fold in range(N_FOLDS):
        tri = np.where(fold_ids != fold)[0]
        vai = np.where(fold_ids == fold)[0]
        if len(tri) == 0 or len(vai) == 0:
            continue
        clf = MODEL_REGISTRY[model_name]()
        clf.fit(task_X[tri], task_y[tri])
        pred = clf.predict(task_X[vai])
        yt_all.extend(task_y[vai])
        yp_all.extend(pred)

    metrics = compute_metrics(yt_all, yp_all)
    metrics.update({'task_id': task_id, 'model': model_name,
                    'preprocess': preprocess, 'method': 'global_baseline'})
    return metrics, np.array(yt_all), np.array(yp_all)


def run_oracle_context(meta, X, task_id, task_info, model_name, preprocess):
    """Two-stage with oracle (ground-truth) mixture context.
    Uses binary split: single vs multi-component (avoids small-group problem).
    """
    filter_mask = meta[task_info['filter_col']].astype(bool).values
    task_idx = np.where(filter_mask)[0]
    task_meta = meta.iloc[task_idx].reset_index(drop=True)
    task_X = X[task_idx]
    task_y = task_meta[task_info['target_col']].values.astype(int)
    fold_ids = task_meta['fold_id'].values

    # Binary context: single vs multi-component
    contexts = np.where(task_meta['mixture_order'].values == 1, 'single', 'multi')

    yt_all, yp_all = [], []
    for fold in range(N_FOLDS):
        tri = np.where(fold_ids != fold)[0]
        vai = np.where(fold_ids == fold)[0]
        if len(tri) == 0 or len(vai) == 0:
            continue

        context_models = {}
        for ctx in np.unique(contexts[tri]):
            ctx_tri = tri[contexts[tri] == ctx]
            if len(np.unique(task_y[ctx_tri])) < 2:
                continue
            clf = MODEL_REGISTRY[model_name]()
            clf.fit(task_X[ctx_tri], task_y[ctx_tri])
            context_models[ctx] = clf

        # Fallback: global model for missing contexts
        clf_global = MODEL_REGISTRY[model_name]()
        clf_global.fit(task_X[tri], task_y[tri])

        for i in vai:
            ctx = contexts[i]
            if ctx in context_models:
                pred = context_models[ctx].predict(task_X[i:i+1])[0]
            else:
                pred = clf_global.predict(task_X[i:i+1])[0]
            yt_all.append(task_y[i])
            yp_all.append(pred)

    metrics = compute_metrics(yt_all, yp_all)
    metrics.update({'task_id': task_id, 'model': model_name,
                    'preprocess': preprocess, 'method': 'oracle_context'})
    return metrics, np.array(yt_all), np.array(yp_all)


def run_realistic_context(meta, X, task_id, task_info, model_name, preprocess):
    """Two-stage with predicted mixture context (Stage 1 → Stage 2).
    Binary split: single vs multi, predicted from presence models.
    """
    filter_mask = meta[task_info['filter_col']].astype(bool).values
    task_idx = np.where(filter_mask)[0]
    task_meta = meta.iloc[task_idx].reset_index(drop=True)
    task_X = X[task_idx]
    task_y = task_meta[task_info['target_col']].values.astype(int)
    fold_ids = task_meta['fold_id'].values

    gt_contexts = np.where(
        task_meta['mixture_order'].values == 1, 'single', 'multi')

    all_fold_ids = meta['fold_id'].values

    yt_all, yp_all = [], []
    for fold in range(N_FOLDS):
        tri = np.where(fold_ids != fold)[0]
        vai = np.where(fold_ids == fold)[0]
        if len(tri) == 0 or len(vai) == 0:
            continue

        # Stage 1: predict presence for val samples
        all_tri = np.where(all_fold_ids != fold)[0]
        presence_preds = {}
        for sub in ['has_thiram', 'has_mg', 'has_mba']:
            y_pres = meta[sub].astype(int).values
            clf_pres = MODEL_REGISTRY[model_name]()
            clf_pres.fit(X[all_tri], y_pres[all_tri])
            global_vai = task_idx[vai]
            presence_preds[sub] = clf_pres.predict(X[global_vai])

        # Infer predicted context (binary: single vs multi)
        n_present = (presence_preds['has_thiram'].astype(int)
                     + presence_preds['has_mg'].astype(int)
                     + presence_preds['has_mba'].astype(int))
        pred_contexts = np.where(n_present == 1, 'single', 'multi')

        # Stage 2: context-specific models (trained on GT context)
        context_models = {}
        for ctx in np.unique(gt_contexts[tri]):
            ctx_tri = tri[gt_contexts[tri] == ctx]
            if len(np.unique(task_y[ctx_tri])) < 2:
                continue
            clf = MODEL_REGISTRY[model_name]()
            clf.fit(task_X[ctx_tri], task_y[ctx_tri])
            context_models[ctx] = clf

        clf_global = MODEL_REGISTRY[model_name]()
        clf_global.fit(task_X[tri], task_y[tri])

        for i, vi in enumerate(vai):
            ctx = pred_contexts[i]
            if ctx in context_models:
                pred = context_models[ctx].predict(task_X[vi:vi+1])[0]
            else:
                pred = clf_global.predict(task_X[vi:vi+1])[0]
            yt_all.append(task_y[vi])
            yp_all.append(pred)

    metrics = compute_metrics(yt_all, yp_all)
    metrics.update({'task_id': task_id, 'model': model_name,
                    'preprocess': preprocess, 'method': 'realistic_context'})
    return metrics, np.array(yt_all), np.array(yp_all)


def run_context_as_feature(meta, X, task_id, task_info, model_name, preprocess):
    """Append mixture_order as additional feature to spectrum."""
    filter_mask = meta[task_info['filter_col']].astype(bool).values
    task_idx = np.where(filter_mask)[0]
    task_meta = meta.iloc[task_idx].reset_index(drop=True)
    task_X = X[task_idx]
    task_y = task_meta[task_info['target_col']].values.astype(int)
    fold_ids = task_meta['fold_id'].values
    mixture_order = task_meta['mixture_order'].values.reshape(-1, 1).astype(float)

    # Augment X with mixture_order feature
    task_X_aug = np.hstack([task_X, mixture_order])

    yt_all, yp_all = [], []
    for fold in range(N_FOLDS):
        tri = np.where(fold_ids != fold)[0]
        vai = np.where(fold_ids == fold)[0]
        if len(tri) == 0 or len(vai) == 0:
            continue
        clf = MODEL_REGISTRY[model_name]()
        clf.fit(task_X_aug[tri], task_y[tri])
        pred = clf.predict(task_X_aug[vai])
        yt_all.extend(task_y[vai])
        yp_all.extend(pred)

    metrics = compute_metrics(yt_all, yp_all)
    metrics.update({'task_id': task_id, 'model': model_name,
                    'preprocess': preprocess, 'method': 'context_as_feature'})
    return metrics, np.array(yt_all), np.array(yp_all)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Two-Stage Context-Aware Prediction Experiment")
    print("=" * 70)

    all_results = []

    for preprocess in PREPROCESS_LIST:
        print(f"\n{'='*70}")
        print(f"Preprocess: {preprocess} ({PREPROCESS_NAMES.get(preprocess, '')})")
        print(f"{'='*70}")

        meta, X = load_data(preprocess)

        for task_id, task_info in GRADE_TASKS.items():
            print(f"\n--- {task_id} ---")

            for model_name in CONTEXT_MODELS:
                print(f"  Model: {model_name}")

                # 1. Global baseline
                m1, yt1, yp1 = run_global_baseline(
                    meta, X, task_id, task_info, model_name, preprocess)
                print(f"    Global:    acc={m1['accuracy']:.3f}, "
                      f"w1={m1['within_1_acc']:.3f}, "
                      f"f1={m1['macro_f1']:.3f}")
                all_results.append(m1)

                # 2. Oracle context (binary: single vs multi)
                m2, yt2, yp2 = run_oracle_context(
                    meta, X, task_id, task_info, model_name, preprocess)
                print(f"    Oracle:    acc={m2['accuracy']:.3f}, "
                      f"w1={m2['within_1_acc']:.3f}, "
                      f"f1={m2['macro_f1']:.3f}")
                all_results.append(m2)

                # 3. Realistic context
                m3, yt3, yp3 = run_realistic_context(
                    meta, X, task_id, task_info, model_name, preprocess)
                print(f"    Realistic: acc={m3['accuracy']:.3f}, "
                      f"w1={m3['within_1_acc']:.3f}, "
                      f"f1={m3['macro_f1']:.3f}")
                all_results.append(m3)

                # 4. Context as feature
                m4, yt4, yp4 = run_context_as_feature(
                    meta, X, task_id, task_info, model_name, preprocess)
                print(f"    Ctx-feat:  acc={m4['accuracy']:.3f}, "
                      f"w1={m4['within_1_acc']:.3f}, "
                      f"f1={m4['macro_f1']:.3f}")
                all_results.append(m4)

                # Delta
                print(f"    Delta(oracle):    {m2['accuracy']-m1['accuracy']:+.3f}")
                print(f"    Delta(realistic): {m3['accuracy']-m1['accuracy']:+.3f}")
                print(f"    Delta(ctx-feat):  {m4['accuracy']-m1['accuracy']:+.3f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'two_stage_results.csv',
                      index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {OUTPUT_DIR / 'two_stage_results.csv'}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    pivot = results_df.pivot_table(
        index=['task_id', 'model', 'preprocess'],
        columns='method',
        values=['accuracy', 'within_1_acc', 'macro_f1'],
    )
    print(pivot.to_string())


if __name__ == '__main__':
    main()
