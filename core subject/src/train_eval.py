"""Training and evaluation for the clean molar mainline benchmark."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report,
)

from src.config import TASKS, N_FOLDS, MODELS_DIR
from src.models import MODEL_REGISTRY


def _task_subset_mask(meta: pd.DataFrame, task: dict) -> np.ndarray:
    mask = np.ones(len(meta), dtype=bool)

    filter_col = task.get('filter_col')
    if filter_col is not None:
        filter_value = task.get('filter_value', 1)
        mask &= meta[filter_col].astype(int).values == int(filter_value)

    filter_query = task.get('filter_query')
    if filter_query:
        mask &= meta.eval(filter_query).astype(bool).values

    return mask


def _instantiate_model(model_name: str, model_kwargs: dict[str, dict] | None = None):
    factory = MODEL_REGISTRY[model_name]
    kwargs = {}
    if model_kwargs is not None:
        kwargs = dict(model_kwargs.get(model_name, {}))
    try:
        return factory(**kwargs) if kwargs else factory()
    except TypeError as exc:
        raise TypeError(
            f"Model '{model_name}' does not accept overrides: {sorted(kwargs.keys())}"
        ) from exc


def run_cv_evaluation(meta: pd.DataFrame, X: np.ndarray,
                      model_names: list[str] = None,
                      task_ids: list[str] = None,
                      preprocess_tag: str = 'p1',
                      wn: np.ndarray = None,
                      tasks_override: list = None,
                      model_kwargs: dict[str, dict] | None = None,
                      use_random_split: bool = False) -> pd.DataFrame:
    """Run cross-validated evaluation for specified models on specified tasks.

    If use_random_split=True, uses StratifiedKFold (no group awareness)
    as a control experiment to quantify leakage effects.

    Single-task models (RF, SVM, PLS-DA, 1D-CNN, 1D-ResNet):
        for task → for model → for fold → fit(X, y) / predict(X)

    Multi-task models (MT-CNN, MT-ResNet, MT-KAN-CNN):
        for model → for fold → fit(X, y_dict) / predict(X) → score all tasks
        One fit covers all tasks simultaneously.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    task_source = tasks_override if tasks_override is not None else TASKS

    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())
    if task_ids is None:
        task_ids = [t['id'] for t in task_source]

    tasks = [t for t in task_source if t['id'] in task_ids]
    results = []
    all_preds = {}

    st_names = [m for m in model_names if not m.startswith('MT-')]
    mt_names = [m for m in model_names if m.startswith('MT-')]

    # Build fold assignment: random split vs pre-computed group split
    if use_random_split:
        from sklearn.model_selection import StratifiedKFold
        # Use first task's labels for stratification
        first_y = meta[tasks[0]['col']].values.astype(int) if tasks else np.zeros(len(meta), dtype=int)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        fold_ids = np.full(len(meta), -1, dtype=int)
        for fold_i, (_, val_idx) in enumerate(skf.split(np.zeros(len(meta)), first_y)):
            fold_ids[val_idx] = fold_i
        print(f"  [Random Split mode — no group awareness]")
    else:
        fold_ids = meta['fold_id'].values

    if st_names:
        for tk in tasks:
            task_mask = _task_subset_mask(meta, tk)
            task_idx = np.where(task_mask)[0]
            if len(task_idx) == 0:
                print(f"\n  Task: {tk['name']} skipped (no samples after filtering)")
                continue

            task_meta = meta.iloc[task_idx].reset_index(drop=True)
            task_X = X[task_idx]
            task_fold_ids = fold_ids[task_idx]
            col_data = task_meta[tk['col']]
            if col_data.dtype == object:
                col_data = col_data.map({'True': 1, 'False': 0}).fillna(col_data)
            y = col_data.values.astype(int)

            print(
                f"\n  Task: {tk['name']} ({tk['col']}), classes={tk['classes']}, "
                f"samples={len(task_meta)}"
            )

            for mname in st_names:
                if mname not in MODEL_REGISTRY:
                    print(f"    WARNING: model '{mname}' not in registry, skipping.")
                    continue

                yt_all, yp_all = [], []
                fold_f1s, fold_bas, fold_accs = [], [], []

                for fold in range(N_FOLDS):
                    tri = np.where(task_fold_ids != fold)[0]
                    vai = np.where(task_fold_ids == fold)[0]
                    if len(tri) == 0 or len(vai) == 0:
                        continue

                    clf = _instantiate_model(mname, model_kwargs=model_kwargs)
                    fit_start = perf_counter()
                    if hasattr(clf, 'is_feature_model') and clf.is_feature_model:
                        groups_train = task_meta.loc[tri, 'group_id'].values if 'group_id' in task_meta.columns else task_meta.loc[tri, 'folder_name'].values
                        clf.fit(task_X[tri], y[tri], groups=groups_train, wn=wn)
                    elif hasattr(clf, 'device'):
                        groups_train = task_meta.loc[tri, 'group_id'].values if 'group_id' in task_meta.columns else task_meta.loc[tri, 'folder_name'].values
                        clf.fit(task_X[tri], y[tri], groups=groups_train)
                    else:
                        clf.fit(task_X[tri], y[tri])
                    fit_seconds = perf_counter() - fit_start
                    predict_start = perf_counter()
                    pred = clf.predict(task_X[vai])
                    predict_seconds = perf_counter() - predict_start

                    f1 = f1_score(y[vai], pred, average='macro', zero_division=0)
                    ba = balanced_accuracy_score(y[vai], pred)
                    acc = accuracy_score(y[vai], pred)
                    fold_f1s.append(f1)
                    fold_bas.append(ba)
                    fold_accs.append(acc)
                    yt_all.extend(y[vai])
                    yp_all.extend(pred)

                    results.append({
                        'Task': tk['id'], 'TaskName': tk['name'],
                        'Model': mname, 'Preprocess': preprocess_tag,
                        'Fold': fold, 'MacroF1': f1, 'BalancedAcc': ba, 'Accuracy': acc,
                        'FitSeconds': fit_seconds, 'PredictSeconds': predict_seconds,
                    })

                model_fit_seconds = sum(
                    row['FitSeconds']
                    for row in results
                    if row['Task'] == tk['id']
                    and row['Model'] == mname
                    and row['Preprocess'] == preprocess_tag
                )
                print(f"    {mname}: F1={np.mean(fold_f1s):.3f}±{np.std(fold_f1s):.3f}, "
                      f"BA={np.mean(fold_bas):.3f}±{np.std(fold_bas):.3f}, "
                      f"Acc={np.mean(fold_accs):.3f}±{np.std(fold_accs):.3f}, "
                      f"fit={model_fit_seconds:.1f}s")
                all_preds[(tk['id'], mname)] = (np.array(yt_all), np.array(yp_all))

    if mt_names and any(tk.get('filter_col') or tk.get('filter_query') for tk in tasks):
        print("\n  WARNING: filtered task sets are not supported by MT models yet; skipping MT models.")
        mt_names = []

    for mname in mt_names:
        if mname not in MODEL_REGISTRY:
            print(f"\n  WARNING: model '{mname}' not in registry, skipping.")
            continue

        y_arrays = {}
        for tk in tasks:
            col_data = meta[tk['col']]
            if col_data.dtype == object:
                col_data = col_data.map({'True': 1, 'False': 0}).fillna(col_data)
            y_arrays[tk['id']] = col_data.values.astype(int)

        print(f"\n  Multi-Task Model: {mname}")
        task_yt  = {tk['id']: [] for tk in tasks}
        task_yp  = {tk['id']: [] for tk in tasks}
        task_f1s = {tk['id']: [] for tk in tasks}
        task_bas = {tk['id']: [] for tk in tasks}
        task_accs = {tk['id']: [] for tk in tasks}

        for fold in range(N_FOLDS):
            tri = np.where(fold_ids != fold)[0]
            vai = np.where(fold_ids == fold)[0]

            y_dict_train = {tid: y_arrays[tid][tri] for tid in y_arrays}
            y_dict_val   = {tid: y_arrays[tid][vai] for tid in y_arrays}

            clf = _instantiate_model(mname, model_kwargs=model_kwargs)
            groups_train = meta.loc[tri, 'folder_name'].values
            fit_start = perf_counter()
            if hasattr(clf, 'is_feature_model') and clf.is_feature_model:
                clf.fit(X[tri], y_dict_train, groups=groups_train, wn=wn)
            else:
                clf.fit(X[tri], y_dict_train, groups=groups_train)
            fit_seconds = perf_counter() - fit_start
            predict_start = perf_counter()
            pred_dict = clf.predict(X[vai])
            predict_seconds = perf_counter() - predict_start

            for tk in tasks:
                tid = tk['id']
                f1 = f1_score(y_dict_val[tid], pred_dict[tid],
                              average='macro', zero_division=0)
                ba = balanced_accuracy_score(y_dict_val[tid], pred_dict[tid])
                acc = accuracy_score(y_dict_val[tid], pred_dict[tid])
                task_f1s[tid].append(f1)
                task_bas[tid].append(ba)
                task_accs[tid].append(acc)
                task_yt[tid].extend(y_dict_val[tid])
                task_yp[tid].extend(pred_dict[tid])
                results.append({
                    'Task': tid, 'TaskName': tk['name'],
                    'Model': mname, 'Preprocess': preprocess_tag,
                    'Fold': fold, 'MacroF1': f1, 'BalancedAcc': ba, 'Accuracy': acc,
                    'FitSeconds': fit_seconds, 'PredictSeconds': predict_seconds,
                })

        for tk in tasks:
            tid = tk['id']
            print(f"    {tk['name']}: F1={np.mean(task_f1s[tid]):.3f}±{np.std(task_f1s[tid]):.3f}, "
                  f"BA={np.mean(task_bas[tid]):.3f}±{np.std(task_bas[tid]):.3f}, "
                  f"Acc={np.mean(task_accs[tid]):.3f}±{np.std(task_accs[tid]):.3f}")
            all_preds[(tid, mname)] = (np.array(task_yt[tid]), np.array(task_yp[tid]))

    res_df = pd.DataFrame(results)
    res_df.attrs['predictions'] = all_preds
    return res_df


def aggregate_results(res_df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = {
        'F1_mean': ('MacroF1', 'mean'),
        'F1_std': ('MacroF1', 'std'),
        'BA_mean': ('BalancedAcc', 'mean'),
        'BA_std': ('BalancedAcc', 'std'),
    }
    if 'Accuracy' in res_df.columns:
        agg_cols['Acc_mean'] = ('Accuracy', 'mean')
        agg_cols['Acc_std'] = ('Accuracy', 'std')
    if 'FitSeconds' in res_df.columns:
        agg_cols['FitSeconds_total'] = ('FitSeconds', 'sum')
        agg_cols['FitSeconds_mean'] = ('FitSeconds', 'mean')
    if 'PredictSeconds' in res_df.columns:
        agg_cols['PredictSeconds_total'] = ('PredictSeconds', 'sum')
    return res_df.groupby(['Task', 'TaskName', 'Model', 'Preprocess']).agg(**agg_cols).reset_index()


def get_best_models(agg_df: pd.DataFrame) -> pd.DataFrame:
    """For each task, find the model with highest mean F1."""
    idx = agg_df.groupby('Task')['F1_mean'].idxmax()
    return agg_df.loc[idx].reset_index(drop=True)


def _merge_replace(existing: pd.DataFrame, new: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if existing.empty:
        return new.copy()

    existing = existing.copy()
    new = new.copy()

    def _make_key(df, cols):
        return df[cols[0]].astype(str).str.cat([df[c].astype(str) for c in cols[1:]], sep='||')

    existing['__merge_key__'] = _make_key(existing, key_cols)
    new['__merge_key__'] = _make_key(new, key_cols)
    existing = existing[~existing['__merge_key__'].isin(set(new['__merge_key__']))]
    existing = existing.drop(columns='__merge_key__')
    new = new.drop(columns='__merge_key__')
    return pd.concat([existing, new], ignore_index=True)


def save_results(
    res_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    tag: str = '',
    output_dir: Path | None = None,
):
    result_dir = output_dir if output_dir is not None else MODELS_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'_{tag}' if tag else ''

    detail_path = result_dir / f'cv_results_detail{suffix}.csv'
    if detail_path.exists():
        old = pd.read_csv(detail_path)
        merged = _merge_replace(old, res_df, ['Task', 'TaskName', 'Model', 'Preprocess', 'Fold'])
    else:
        merged = res_df.copy()
    merged.to_csv(detail_path, index=False, encoding='utf-8-sig')

    summary_path = result_dir / f'cv_results_summary{suffix}.csv'
    if summary_path.exists():
        old_agg = pd.read_csv(summary_path)
        merged_agg = _merge_replace(old_agg, agg_df, ['Task', 'TaskName', 'Model', 'Preprocess'])
    else:
        merged_agg = agg_df.copy()
    merged_agg.to_csv(summary_path, index=False, encoding='utf-8-sig')

    predictions = res_df.attrs.get('predictions', {})
    if predictions:
        pred_records = []
        for (task_id, model_name), (yt, yp) in predictions.items():
            for t, p in zip(yt, yp):
                pred_records.append({
                    'task_id': task_id, 'model': model_name,
                    'y_true': int(t), 'y_pred': int(p),
                })
        new_pred = pd.DataFrame(pred_records)
        pred_path = result_dir / f'cv_predictions{suffix}.csv'
        if pred_path.exists():
            old_pred = pd.read_csv(pred_path)
            new_pred = _merge_replace(old_pred, new_pred, ['task_id', 'model', 'y_true', 'y_pred'])
        new_pred.to_csv(pred_path, index=False, encoding='utf-8-sig')

    print(f"  Results saved to {result_dir}")


def load_predictions(tag: str = '', output_dir: Path | None = None) -> dict:
    """Load predictions from CSV for report generation."""
    result_dir = output_dir if output_dir is not None else MODELS_DIR
    suffix = f'_{tag}' if tag else ''
    pred_path = result_dir / f'cv_predictions{suffix}.csv'
    if not pred_path.exists():
        return {}
    df = pd.read_csv(pred_path)
    predictions = {}
    for (task_id, model), grp in df.groupby(['task_id', 'model']):
        predictions[(task_id, model)] = (grp['y_true'].values, grp['y_pred'].values)
    return predictions
