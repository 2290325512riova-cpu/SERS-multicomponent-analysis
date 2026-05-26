#!/usr/bin/env python3
"""
Model diagnostics script.
Runs on GPU server when needed.

Part A: Loss curves — retrain DL models with epoch-level loss recording
Part B: Integrated Gradients — wavenumber attribution for Spectrum-KAN
Part C: Augmentation ablation — with vs without data augmentation
Part D: Per-class F1 + Bootstrap CI (offline, from existing predictions)
"""
from __future__ import annotations
import sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
from pathlib import Path

# Add project to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import (
    PROCESSED_DIR, SPLITS_DIR, MODELS_DIR, SEED,
    DL_LR, DL_EPOCHS, DL_PATIENCE, DL_BATCH, DL_WEIGHT_DECAY,
    N_FOLDS,
    AUG_ENABLED, AUG_N, MIXUP_ENABLED, MIXUP_ALPHA,
    TASKS, TASKS_SUPPLEMENTARY, FIGURES_DIR, ACTIVE_SPLIT_FILE,
    PREPROCESS_TAGS,
)
from src.models import (
    _CNN1DNet, _SpectrumKANNet, _ChemKANNet, _make_loss,
)
from src.dataset import augment_spectra, spectral_mixup
from src.result_index import get_active_grouped_result_dir

# Output directories
DIAGNOSTICS_DIR = MODELS_DIR / "diagnostics"
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIAGNOSTICS = FIGURES_DIR / "diagnostics"
FIG_DIAGNOSTICS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ======================================================================
#                         HELPER: Load Data
# ======================================================================
def load_data(preprocess_tag: str):
    """Load preprocessed data and metadata."""
    X = np.load(PROCESSED_DIR / f'X_{preprocess_tag}.npy')
    wn = np.load(PROCESSED_DIR / 'wavenumber.npy')
    meta = pd.read_csv(SPLITS_DIR / ACTIVE_SPLIT_FILE)
    for col in ['has_thiram', 'has_mg', 'has_mba']:
        if col in meta.columns:
            meta[col] = meta[col].astype(str).str.strip().str.lower().map(
                {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
            ).fillna(0).astype(int)
    return X, wn, meta


def _task_by_id(task_id: str) -> dict:
    for task in TASKS + TASKS_SUPPLEMENTARY:
        if task['id'] == task_id:
            return task
    raise KeyError(f"Unknown active task id: {task_id}")


def _task_subset(meta: pd.DataFrame, X: np.ndarray, task: dict):
    """Apply the active task filter, especially positive-only molar grading."""
    mask = np.ones(len(meta), dtype=bool)
    if 'filter_col' in task:
        mask &= meta[task['filter_col']].values == task.get('filter_value', 1)

    task_meta = meta.loc[mask].reset_index(drop=True)
    X_task = X[mask]
    y = task_meta[task['col']].values.astype(int)

    valid = np.isin(y, task['classes'])
    if not valid.all():
        dropped = int((~valid).sum())
        print(f"  Warning: dropping {dropped} rows outside active classes for {task['id']}")
        task_meta = task_meta.loc[valid].reset_index(drop=True)
        X_task = X_task[valid]
        y = y[valid]

    return X_task, task_meta, y


def _best_preprocess_for_model_task(model_name: str, task_id: str, fallback: str) -> str:
    """Pick the best known benchmark preprocessing for a model/task pair."""
    result_dir = get_active_grouped_result_dir()
    if result_dir is None:
        return fallback

    matrix_path = result_dir / 'benchmark_full_matrix.csv'
    if not matrix_path.exists():
        return fallback

    df = pd.read_csv(matrix_path)
    sub = df[(df['Model'] == model_name) & (df['Task'] == task_id)]
    if sub.empty:
        return fallback
    return str(sub.loc[sub['F1_mean'].idxmax(), 'Preprocess'])


# ======================================================================
#           PART A: Loss Curves (DL training with loss recording)
# ======================================================================
def train_with_loss_recording(net_cls, X_train, y_train, X_val, y_val,
                              num_classes, groups_train=None,
                              aug_enabled=True):
    """Train a DL model and return epoch-level train/val loss."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True

    label_map = {label: idx for idx, label in enumerate(sorted(set(y_train)))}
    yi_train = np.array([label_map[v] for v in y_train])
    yi_val = np.array([label_map[v] for v in y_val])

    # Data augmentation
    if aug_enabled and AUG_ENABLED:
        composition_keys = np.asarray(groups_train) if groups_train is not None else None
        X_tr_aug, y_tr_aug = augment_spectra(X_train, yi_train, n_aug=AUG_N)
        if composition_keys is not None:
            composition_keys = np.tile(composition_keys, AUG_N + 1)
        if MIXUP_ENABLED and composition_keys is not None:
            X_mix, y_mix = spectral_mixup(X_tr_aug, y_tr_aug,
                                          n_mix=len(X_train), alpha=MIXUP_ALPHA,
                                          composition_keys=composition_keys)
            X_tr_aug = np.concatenate([X_tr_aug, X_mix])
            y_tr_aug = np.concatenate([y_tr_aug, y_mix])
    else:
        X_tr_aug, y_tr_aug = X_train.copy(), yi_train.copy()

    Xt_train = torch.tensor(X_tr_aug, dtype=torch.float32).unsqueeze(1)
    yt_train = torch.tensor(y_tr_aug, dtype=torch.long)
    Xt_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    yt_val = torch.tensor(yi_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xt_train, yt_train),
                              batch_size=DL_BATCH, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xt_val, yt_val),
                            batch_size=256, shuffle=False)

    model = net_cls(X_train.shape[1], num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=DL_LR,
                                 weight_decay=DL_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=DL_EPOCHS, eta_min=1e-6)

    criterion_train = _make_loss(y_tr_aug, num_classes)
    criterion_val = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_state = None
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(DL_EPOCHS):
        # Train
        model.train()
        epoch_train_loss = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion_train(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(yb)
            train_count += len(yb)

        epoch_train_loss /= max(train_count, 1)
        train_losses.append(epoch_train_loss)

        # Validate
        model.eval()
        epoch_val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                epoch_val_loss += criterion_val(model(xb), yb).item() * len(yb)
                val_count += len(yb)

        epoch_val_loss /= max(val_count, 1)
        val_losses.append(epoch_val_loss)
        scheduler.step()

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= DL_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Get predictions with best model
    model.eval()
    with torch.no_grad():
        preds = model(Xt_val.to(DEVICE)).argmax(1).cpu().numpy()
    inv_map = {idx: label for label, idx in label_map.items()}
    preds_orig = np.array([inv_map[p] for p in preds])

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': len(train_losses) - wait,
        'model': model,
        'label_map': label_map,
        'preds': preds_orig,
    }


def run_part_a():
    """Part A: Loss curves for 5 DL models on best preprocessing + representative task."""
    print("\n" + "=" * 70)
    print("PART A: Loss Curves")
    print("=" * 70)

    # Model configs: (name, net_cls, best_preprocess)
    model_configs = [
        ('1D-CNN',       _CNN1DNet,       'p2'),
        ('1D-ResNet',    _ResNet1DNet,    'p2'),
        ('Spectrum-KAN', _SpectrumKANNet, 'p1'),
        ('KAN-CNN',      _KANCNN1DNet,    'p2'),
    ]

    # Representative task: P1_thiram_presence (binary, clearest signal)
    task = {'id': 'P1_thiram_presence', 'col': 'has_thiram', 'classes': [0, 1]}
    all_results = []

    for model_name, net_cls, pp_tag in model_configs:
        print(f"\n  {model_name} on {pp_tag}, task={task['id']}")
        X, wn, meta = load_data(pp_tag)
        y = meta[task['col']].values.astype(int)
        fold_ids = meta['fold_id'].values

        for fold in range(N_FOLDS):
            tri = np.where(fold_ids != fold)[0]
            vai = np.where(fold_ids == fold)[0]
            groups_train = meta.loc[tri, 'folder_name'].values

            t0 = time.time()
            result = train_with_loss_recording(
                net_cls, X[tri], y[tri], X[vai], y[vai],
                num_classes=len(task['classes']),
                groups_train=groups_train,
                aug_enabled=True,
            )
            elapsed = time.time() - t0

            f1 = f1_score(y[vai], result['preds'], average='macro', zero_division=0)
            print(f"    Fold {fold}: F1={f1:.3f}, epochs={len(result['train_losses'])}, "
                  f"best@{result['best_epoch']}, {elapsed:.1f}s")

            all_results.append({
                'model': model_name,
                'preprocess': pp_tag,
                'task': task['id'],
                'fold': fold,
                'train_losses': result['train_losses'],
                'val_losses': result['val_losses'],
                'best_epoch': result['best_epoch'],
                'final_f1': f1,
            })

    # Save loss histories as CSV
    rows = []
    for r in all_results:
        for ep, (tl, vl) in enumerate(zip(r['train_losses'], r['val_losses'])):
            rows.append({
                'Model': r['model'], 'Preprocess': r['preprocess'],
                'Task': r['task'], 'Fold': r['fold'],
                'Epoch': ep, 'TrainLoss': tl, 'ValLoss': vl,
            })
    df = pd.DataFrame(rows)
    out_path = DIAGNOSTICS_DIR / 'loss_curves.csv'
    df.to_csv(out_path, index=False)
    print(f"\n  Loss curves saved to {out_path}")

    # Plot loss curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for i, (model_name, _, _) in enumerate(model_configs):
            ax = axes[i]
            model_df = df[df['Model'] == model_name]
            for fold in range(N_FOLDS):
                fold_df = model_df[model_df['Fold'] == fold]
                ax.plot(fold_df['Epoch'], fold_df['TrainLoss'],
                        alpha=0.3, color='blue', linewidth=0.8)
                ax.plot(fold_df['Epoch'], fold_df['ValLoss'],
                        alpha=0.3, color='red', linewidth=0.8)
            # Mean curves
            mean_train = model_df.groupby('Epoch')['TrainLoss'].mean()
            mean_val = model_df.groupby('Epoch')['ValLoss'].mean()
            ax.plot(mean_train.index, mean_train.values, 'b-', linewidth=2, label='Train (mean)')
            ax.plot(mean_val.index, mean_val.values, 'r-', linewidth=2, label='Val (mean)')
            # Mark best epoch
            best_ep = model_df.groupby('Fold').first()['Epoch'].iloc[0]  # approximate
            ax.set_title(model_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = FIG_DIAGNOSTICS / 'fig_s_loss_curves.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Loss curve figure saved to {fig_path}")
    except Exception as e:
        print(f"  Warning: Could not plot loss curves: {e}")

    return all_results


# ======================================================================
#           PART B: Integrated Gradients for Spectrum-KAN
# ======================================================================
def run_part_b():
    """Part B: Integrated Gradients for Spectrum-KAN on 3 presence tasks."""
    print("\n" + "=" * 70)
    print("PART B: Integrated Gradients (Spectrum-KAN)")
    print("=" * 70)

    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        raise ModuleNotFoundError(
            'captum is required for Integrated Gradients. Install it from requirements.txt first.'
        )

    presence_tasks = [
        {'id': 'T4_thiram_pres', 'col': 'has_thiram', 'classes': [0, 1], 'name': 'Thiram'},
        {'id': 'T5_mg_pres',     'col': 'has_mg',     'classes': [0, 1], 'name': 'MG'},
        {'id': 'T6_mba_pres',    'col': 'has_mba',    'classes': [0, 1], 'name': 'MBA'},
    ]

    pp_tag = 'p1'  # Best preprocessing for Spectrum-KAN
    X, wn, meta = load_data(pp_tag)
    fold_ids = meta['fold_id'].values

    all_ig_results = {}

    for task in presence_tasks:
        print(f"\n  Task: {task['name']} ({task['id']})")
        y = meta[task['col']].values.astype(int)
        num_classes = len(task['classes'])

        # Accumulate attributions across folds
        attr_positive = []  # attributions for positive class samples
        attr_all = []       # all attributions

        for fold in range(N_FOLDS):
            tri = np.where(fold_ids != fold)[0]
            vai = np.where(fold_ids == fold)[0]
            groups_train = meta.loc[tri, 'folder_name'].values

            # Train model
            result = train_with_loss_recording(
                _SpectrumKANNet, X[tri], y[tri], X[vai], y[vai],
                num_classes=num_classes,
                groups_train=groups_train,
            )
            model = result['model']
            model.eval()

            # Run Integrated Gradients on validation set
            ig = IntegratedGradients(model)

            X_val_t = torch.tensor(X[vai], dtype=torch.float32).unsqueeze(1).to(DEVICE)
            y_val = y[vai]

            # For each sample, compute IG w.r.t. predicted class
            # Process in batches
            batch_attrs = []
            batch_size = 64
            for start in range(0, len(X_val_t), batch_size):
                end = min(start + batch_size, len(X_val_t))
                batch = X_val_t[start:end]
                batch.requires_grad_(True)

                # Target: positive class (1)
                target = torch.ones(end - start, dtype=torch.long).to(DEVICE)
                attr = ig.attribute(batch, target=target, n_steps=50)
                batch_attrs.append(attr.squeeze(1).cpu().detach().numpy())

            attributions = np.concatenate(batch_attrs, axis=0)  # (n_val, 1401)

            # Store attributions for positive samples only
            pos_mask = y_val == 1
            if pos_mask.any():
                attr_positive.append(attributions[pos_mask])
            attr_all.append(attributions)

            f1 = f1_score(y_val, result['preds'], average='macro', zero_division=0)
            print(f"    Fold {fold}: F1={f1:.3f}, "
                  f"n_positive={pos_mask.sum()}, n_val={len(y_val)}")

        # Aggregate attributions
        attr_positive = np.concatenate(attr_positive, axis=0)
        attr_all_concat = np.concatenate(attr_all, axis=0)

        # Mean absolute attribution across positive samples
        mean_abs_attr = np.mean(np.abs(attr_positive), axis=0)
        mean_abs_attr_all = np.mean(np.abs(attr_all_concat), axis=0)

        all_ig_results[task['id']] = {
            'mean_abs_attr_positive': mean_abs_attr,
            'mean_abs_attr_all': mean_abs_attr_all,
            'task_name': task['name'],
        }

    # Save IG results as CSV
    ig_rows = []
    for task_id, res in all_ig_results.items():
        for i, w in enumerate(wn):
            ig_rows.append({
                'Task': task_id,
                'TaskName': res['task_name'],
                'Wavenumber': w,
                'MeanAbsAttr_Positive': res['mean_abs_attr_positive'][i],
                'MeanAbsAttr_All': res['mean_abs_attr_all'][i],
            })
    ig_df = pd.DataFrame(ig_rows)
    ig_path = DIAGNOSTICS_DIR / 'integrated_gradients.csv'
    ig_df.to_csv(ig_path, index=False)
    print(f"\n  IG results saved to {ig_path}")

    # Plot IG wavenumber attribution
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        for i, task in enumerate(presence_tasks):
            ax = axes[i]
            res = all_ig_results[task['id']]
            attr = res['mean_abs_attr_positive']
            # Normalize
            attr_norm = attr / attr.max() if attr.max() > 0 else attr

            ax.plot(wn, attr_norm, 'b-', linewidth=0.8, alpha=0.8)
            ax.fill_between(wn, 0, attr_norm, alpha=0.2)

            # Mark top-20 wavenumbers
            top20_idx = np.argsort(attr_norm)[-20:]
            top20_wn = wn[top20_idx]
            for w in top20_wn:
                ax.axvline(w, color='red', alpha=0.15, linewidth=0.5)

            ax.set_title(f'{task["name"]} — Integrated Gradients (Spectrum-KAN, {pp_tag})',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Normalized |Attribution|')
            ax.invert_xaxis()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = FIG_DIAGNOSTICS / 'fig_s_integrated_gradients.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  IG figure saved to {fig_path}")
    except Exception as e:
        print(f"  Warning: Could not plot IG: {e}")

    return all_ig_results


# ======================================================================
#           PART C: Augmentation Ablation
# ======================================================================
def run_part_c():
    """Part C: Compare with vs without augmentation for Spectrum-KAN and Feature-KAN."""
    print("\n" + "=" * 70)
    print("PART C: Augmentation Ablation")
    print("=" * 70)

    tasks_to_test = [
        {'id': 'T4_thiram_pres',  'col': 'has_thiram',      'classes': [0, 1], 'name': 'Thiram Pres'},
        {'id': 'T1_thiram_conc',  'col': 'c_thiram',        'classes': [0, 4, 5, 6], 'name': 'Thiram Conc'},
        {'id': 'T7_mixture_order','col': 'mixture_order',   'classes': [1, 2, 3], 'name': 'Mixture Order'},
    ]

    # Spectrum-KAN on p1 (best overall)
    # Feature-KAN needs feature extraction — use _FeatureKANNet on pre-extracted features
    configs = [
        ('Spectrum-KAN', _SpectrumKANNet, 'p1', False),  # full spectrum
    ]

    results = []

    for model_name, net_cls, pp_tag, is_feature in configs:
        X, wn, meta = load_data(pp_tag)
        fold_ids = meta['fold_id'].values

        if is_feature:
            from src.feature_engineering import extract_all_features
            X, _ = extract_all_features(X, wn)

        for task in tasks_to_test:
            y = meta[task['col']].values.astype(int)
            num_classes = len(task['classes'])

            for aug_label, aug_flag in [('with_aug', True), ('no_aug', False)]:
                print(f"\n  {model_name} | {task['name']} | {aug_label}")
                fold_f1s = []

                for fold in range(N_FOLDS):
                    tri = np.where(fold_ids != fold)[0]
                    vai = np.where(fold_ids == fold)[0]
                    groups_train = meta.loc[tri, 'folder_name'].values

                    result = train_with_loss_recording(
                        net_cls, X[tri], y[tri], X[vai], y[vai],
                        num_classes=num_classes,
                        groups_train=groups_train,
                        aug_enabled=aug_flag,
                    )
                    f1 = f1_score(y[vai], result['preds'], average='macro', zero_division=0)
                    fold_f1s.append(f1)
                    print(f"    Fold {fold}: F1={f1:.3f}")

                mean_f1 = np.mean(fold_f1s)
                std_f1 = np.std(fold_f1s)
                results.append({
                    'Model': model_name, 'Task': task['id'], 'TaskName': task['name'],
                    'Preprocess': pp_tag, 'Augmentation': aug_label,
                    'F1_mean': mean_f1, 'F1_std': std_f1,
                    'Fold_F1s': fold_f1s,
                })
                print(f"    Mean: F1={mean_f1:.3f}±{std_f1:.3f}")

    # Feature-KAN ablation (separate because it uses feature extraction)
    print(f"\n  Feature-KAN ablation (feature-space augmentation)")
    from src.feature_engineering import extract_all_features
    pp_tag = 'raw'  # Best for Feature-KAN
    X_raw, wn, meta = load_data(pp_tag)
    X_feat, _ = extract_all_features(X_raw, wn)
    fold_ids = meta['fold_id'].values

    for task in tasks_to_test:
        y = meta[task['col']].values.astype(int)
        num_classes = len(task['classes'])

        for aug_label, aug_flag in [('with_aug', True), ('no_aug', False)]:
            print(f"\n  Feature-KAN | {task['name']} | {aug_label}")
            fold_f1s = []

            for fold in range(N_FOLDS):
                tri = np.where(fold_ids != fold)[0]
                vai = np.where(fold_ids == fold)[0]

                # Feature-KAN uses _FeatureKANNet directly on features
                # Need custom training that mimics _FeatureKANWrapper
                torch.manual_seed(SEED)
                np.random.seed(SEED)

                label_map = {label: idx for idx, label in enumerate(sorted(set(y[tri])))}
                inv_map = {idx: label for label, idx in label_map.items()}
                yi_tr = np.array([label_map[v] for v in y[tri]])
                yi_val = np.array([label_map[v] for v in y[vai]])

                if aug_flag:
                    rng = np.random.RandomState(SEED)
                    F_tr = X_feat[tri].copy()
                    y_tr = yi_tr.copy()
                    aug_feats = [F_tr]
                    aug_labels = [y_tr]
                    for _ in range(AUG_N):
                        noisy = F_tr + rng.normal(0, 0.02, F_tr.shape) * np.std(F_tr, axis=0, keepdims=True)
                        noisy *= rng.uniform(0.95, 1.05, (len(noisy), 1))
                        aug_feats.append(noisy)
                        aug_labels.append(y_tr.copy())
                    F_aug = np.concatenate(aug_feats)
                    y_aug = np.concatenate(aug_labels)
                else:
                    F_aug, y_aug = X_feat[tri].copy(), yi_tr.copy()

                train_loader = DataLoader(
                    TensorDataset(torch.tensor(F_aug, dtype=torch.float32),
                                  torch.tensor(y_aug, dtype=torch.long)),
                    batch_size=DL_BATCH, shuffle=True)
                val_loader = DataLoader(
                    TensorDataset(torch.tensor(X_feat[vai], dtype=torch.float32),
                                  torch.tensor(yi_val, dtype=torch.long)),
                    batch_size=256, shuffle=False)

                model = _FeatureKANNet(X_feat.shape[1], num_classes).to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=DL_LR,
                                             weight_decay=DL_WEIGHT_DECAY)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=DL_EPOCHS, eta_min=1e-6)
                train_loss = _make_loss(y_aug, num_classes)
                val_loss = nn.CrossEntropyLoss()
                best_loss, best_state, wait = float('inf'), None, 0

                for _ in range(DL_EPOCHS):
                    model.train()
                    for xb, yb in train_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        optimizer.zero_grad()
                        loss = train_loss(model(xb), yb)
                        loss.backward()
                        optimizer.step()
                    model.eval()
                    ev_loss, vc = 0.0, 0
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                            ev_loss += val_loss(model(xb), yb).item() * len(yb)
                            vc += len(yb)
                    ev_loss /= max(vc, 1)
                    scheduler.step()
                    if ev_loss < best_loss:
                        best_loss = ev_loss
                        wait = 0
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    else:
                        wait += 1
                        if wait >= DL_PATIENCE:
                            break
                if best_state:
                    model.load_state_dict(best_state)

                model.eval()
                with torch.no_grad():
                    preds_idx = model(torch.tensor(X_feat[vai], dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy()
                preds = np.array([inv_map[p] for p in preds_idx])
                f1 = f1_score(y[vai], preds, average='macro', zero_division=0)
                fold_f1s.append(f1)
                print(f"    Fold {fold}: F1={f1:.3f}")

            mean_f1 = np.mean(fold_f1s)
            std_f1 = np.std(fold_f1s)
            results.append({
                'Model': 'Feature-KAN', 'Task': task['id'], 'TaskName': task['name'],
                'Preprocess': pp_tag, 'Augmentation': aug_label,
                'F1_mean': mean_f1, 'F1_std': std_f1,
                'Fold_F1s': fold_f1s,
            })
            print(f"    Mean: F1={mean_f1:.3f}±{std_f1:.3f}")

    # Save ablation results
    abl_rows = []
    for r in results:
        abl_rows.append({k: v for k, v in r.items() if k != 'Fold_F1s'})
        for i, f1 in enumerate(r['Fold_F1s']):
            abl_rows[-1][f'Fold{i}_F1'] = f1
    abl_df = pd.DataFrame(abl_rows)
    abl_path = DIAGNOSTICS_DIR / 'augmentation_ablation.csv'
    abl_df.to_csv(abl_path, index=False)
    print(f"\n  Ablation results saved to {abl_path}")
    return results


# ======================================================================
#           PART D: Per-class F1 + Bootstrap CI (offline)
# ======================================================================
def run_part_d():
    """Part D: Per-class F1 and Bootstrap 95% CI from existing predictions."""
    print("\n" + "=" * 70)
    print("PART D: Per-class F1 + Bootstrap CI")
    print("=" * 70)

    perclass_results = []
    bootstrap_results = []

    for tag in PREPROCESS_TAGS:
        pred_path = MODELS_DIR / f'cv_predictions_{tag}.csv'
        if not pred_path.exists():
            print(f"  Skipping {tag}: {pred_path} not found")
            continue

        df = pd.read_csv(pred_path)
        print(f"\n  Processing {tag}: {len(df)} predictions")

        for (task_id, model_name), grp in df.groupby(['task_id', 'model']):
            y_true = grp['y_true'].values
            y_pred = grp['y_pred'].values

            # Per-class F1
            report = classification_report(y_true, y_pred, output_dict=True,
                                           zero_division=0)
            for cls_label, metrics in report.items():
                if cls_label in ('accuracy', 'macro avg', 'weighted avg'):
                    if cls_label == 'macro avg':
                        perclass_results.append({
                            'Task': task_id, 'Model': model_name,
                            'Preprocess': tag, 'Class': 'macro_avg',
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall'],
                            'F1': metrics['f1-score'],
                            'Support': metrics['support'],
                        })
                    continue
                perclass_results.append({
                    'Task': task_id, 'Model': model_name,
                    'Preprocess': tag, 'Class': cls_label,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1': metrics['f1-score'],
                    'Support': metrics['support'],
                })

            # Bootstrap 95% CI for macro F1
            rng = np.random.RandomState(SEED)
            n_bootstrap = 1000
            n_samples = len(y_true)
            boot_f1s = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n_samples, size=n_samples, replace=True)
                bf1 = f1_score(y_true[idx], y_pred[idx], average='macro',
                               zero_division=0)
                boot_f1s.append(bf1)
            boot_f1s = np.array(boot_f1s)
            ci_low = np.percentile(boot_f1s, 2.5)
            ci_high = np.percentile(boot_f1s, 97.5)
            mean_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            bootstrap_results.append({
                'Task': task_id, 'Model': model_name,
                'Preprocess': tag, 'F1': mean_f1,
                'CI_low': ci_low, 'CI_high': ci_high,
                'CI_width': ci_high - ci_low,
            })

    # Save per-class F1
    pc_df = pd.DataFrame(perclass_results)
    pc_path = DIAGNOSTICS_DIR / 'perclass_f1.csv'
    pc_df.to_csv(pc_path, index=False)
    print(f"\n  Per-class F1 saved to {pc_path} ({len(pc_df)} rows)")

    # Save bootstrap CI
    bs_df = pd.DataFrame(bootstrap_results)
    bs_path = DIAGNOSTICS_DIR / 'bootstrap_ci.csv'
    bs_df.to_csv(bs_path, index=False)
    print(f"  Bootstrap CI saved to {bs_path} ({len(bs_df)} rows)")

    return perclass_results, bootstrap_results


# ======================================================================
#                         MAIN
# ======================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--parts', nargs='+', default=['A', 'B', 'C', 'D'],
                        help='Parts to run: A (loss), B (IG), C (ablation), D (perclass+CI)')
    args = parser.parse_args()

    t_start = time.time()
    parts_to_run = [p.upper() for p in args.parts]

    if 'D' in parts_to_run:
        run_part_d()

    if 'A' in parts_to_run:
        run_part_a()

    if 'B' in parts_to_run:
        run_part_b()

    if 'C' in parts_to_run:
        run_part_c()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"All done! Total time: {elapsed / 60:.1f} minutes")
    print(f"Results in: {DIAGNOSTICS_DIR}")
    print(f"Figures in: {FIG_DIAGNOSTICS}")
    print(f"{'=' * 70}")
