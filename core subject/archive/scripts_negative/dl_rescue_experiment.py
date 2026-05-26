"""
DL Rescue Experiment: diagnose and fix deep learning model collapse.

Strategy:
1. Fix training recipe (batch=32, warmup, grad clip, AdamW)
2. Test lightweight Attention-ResNet (fewer params, spectral attention)
3. Compare against baseline (current config) on all 6 tasks

Key insight from literature:
- Small spectral datasets (< 1000) need: small batch, strong regularization,
  warmup, and architectures with < 100K params
- Spectral Attention Module (SAM) helps model focus on informative peaks
  and provides interpretability (attention weights ≈ peak importance)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    ARTIFACTS_DIR, PROCESSED_DIR, SPLITS_DIR, ACTIVE_SPLIT_FILE,
    TASKS, SEED, N_FOLDS,
)

# ============ Spectral Attention Module ============
class SpectralAttention(nn.Module):
    """Channel + spatial attention for 1D spectra."""
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ca = self.channel_att(x).unsqueeze(-1)
        x = x * ca
        sa = self.spatial_att(x)
        return x * sa


class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, drop_path=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.att = SpectralAttention(channels)
        self.drop_path = drop_path
        self.act = nn.GELU()

    def forward(self, x):
        residual = self.block(x)
        residual = self.att(residual)
        if self.training and self.drop_path > 0:
            mask = (torch.rand(x.size(0), 1, 1, device=x.device) > self.drop_path).float()
            residual = residual * mask
        return self.act(residual + x)


class AttentionResNet1D(nn.Module):
    """Lightweight ResNet with Spectral Attention. ~45K params."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.GELU(), nn.MaxPool1d(2),
        )
        self.stage1 = nn.Sequential(
            ResBlock1D(16, drop_path=0.1),
            ResBlock1D(16, drop_path=0.1),
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
        )
        self.stage2 = nn.Sequential(
            ResBlock1D(32, drop_path=0.15),
            ResBlock1D(32, drop_path=0.15),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(48),
        )
        self.stage3 = ResBlock1D(48, drop_path=0.2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x)


# ============ Fixed Training Recipe ============
def train_model(model, X_train, y_train, X_val, y_val, cfg):
    """Train with fixed recipe: small batch, warmup, grad clip, AdamW."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    Xt = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    yv = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=256, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    warmup_epochs = cfg.get('warmup_epochs', 10)
    total_epochs = cfg['epochs']

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Class-weighted focal loss
    classes, counts = np.unique(y_train, return_counts=True)
    weights = torch.tensor(len(y_train) / (len(classes) * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_f1 = -1
    best_state = None
    wait = 0

    for epoch in range(total_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(yb.numpy())
        val_f1 = f1_score(all_true, all_preds, average='macro')

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg['patience']:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


# ============ Experiment Configs ============
CONFIGS = {
    'baseline': {
        'lr': 3e-4, 'epochs': 200, 'patience': 20,
        'batch_size': 128, 'weight_decay': 1e-4, 'warmup_epochs': 0,
    },
    'fixed_recipe': {
        'lr': 1e-3, 'epochs': 300, 'patience': 40,
        'batch_size': 32, 'weight_decay': 5e-4, 'warmup_epochs': 15,
    },
    'conservative': {
        'lr': 5e-4, 'epochs': 300, 'patience': 40,
        'batch_size': 48, 'weight_decay': 3e-4, 'warmup_epochs': 10,
    },
}

MODELS = {
    'ResNet_baseline': lambda dim, nc: _ResNet1DNet_Baseline(dim, nc),
    'AttResNet': lambda dim, nc: AttentionResNet1D(dim, nc),
}


class _ResNet1DNet_Baseline(nn.Module):
    """Current ResNet from models.py for comparison."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.stage1 = self._make_stage(16)
        self.down1 = nn.Sequential(nn.Conv1d(16, 32, 1, stride=2), nn.BatchNorm1d(32))
        self.stage2 = self._make_stage(32)
        self.down2 = nn.Sequential(nn.Conv1d(32, 64, 1, stride=2), nn.BatchNorm1d(64))
        self.stage3 = self._make_stage(64)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, num_classes))

    def _make_stage(self, ch):
        return nn.Sequential(self._res_block(ch), self._res_block(ch))

    def _res_block(self, ch):
        return _SimpleResBlock(ch)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


class _SimpleResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch),
        )

    def forward(self, x):
        return F.relu(self.block(x) + x)


# ============ Main Experiment ============
def run_experiment():
    print("=" * 70)
    print("DL RESCUE EXPERIMENT: Fixing model collapse")
    print("=" * 70)

    # Load data
    split_df = pd.read_csv(SPLITS_DIR / ACTIVE_SPLIT_FILE)
    wn = np.load(PROCESSED_DIR / "wavenumber.npy")

    # Test on p1 (best for SHAP) and p4 (best for MG)
    preprocs = ['p1', 'p4']
    results = []

    for pp in preprocs:
        X = np.load(PROCESSED_DIR / f"X_{pp}.npy")
        print(f"\n{'='*60}")
        print(f"Preprocessing: {pp} | X shape: {X.shape}")
        print(f"{'='*60}")

        for task in TASKS:
            task_id = task['id']
            task_name = task['name']

            # Prepare labels
            y_col = task['col']
            if 'filter_col' in task:
                mask = split_df[task['filter_col']] == task['filter_value']
                df_task = split_df[mask].reset_index(drop=True)
                X_task = X[mask.values]
            else:
                df_task = split_df
                X_task = X

            y_raw = df_task[y_col].values
            classes = sorted(set(y_raw))
            label_map = {c: i for i, c in enumerate(classes)}
            y = np.array([label_map[v] for v in y_raw])
            num_classes = len(classes)
            folds = df_task['fold_id'].values

            print(f"\n  Task: {task_name} | n={len(y)} | classes={num_classes}")

            for model_name, model_fn in MODELS.items():
                for cfg_name, cfg in CONFIGS.items():
                    fold_f1s = []
                    collapsed = False

                    for fold_idx in range(N_FOLDS):
                        train_mask = folds != fold_idx
                        test_mask = folds == fold_idx
                        X_tr, y_tr = X_task[train_mask], y[train_mask]
                        X_te, y_te = X_task[test_mask], y[test_mask]

                        # Internal val split (15%)
                        n_val = max(int(len(X_tr) * 0.15), 1)
                        perm = np.random.RandomState(SEED + fold_idx).permutation(len(X_tr))
                        val_idx = perm[:n_val]
                        tr_idx = perm[n_val:]

                        torch.manual_seed(SEED + fold_idx)
                        np.random.seed(SEED + fold_idx)

                        model = model_fn(X_task.shape[1], num_classes)
                        model, _ = train_model(
                            model, X_tr[tr_idx], y_tr[tr_idx],
                            X_tr[val_idx], y_tr[val_idx], cfg
                        )

                        # Evaluate
                        model.eval()
                        device = next(model.parameters()).device
                        with torch.no_grad():
                            Xt = torch.tensor(X_te, dtype=torch.float32).unsqueeze(1).to(device)
                            preds = model(Xt).argmax(1).cpu().numpy()
                        f1 = f1_score(y_te, preds, average='macro')
                        fold_f1s.append(f1)

                        if f1 < 0.35:
                            collapsed = True

                    mean_f1 = np.mean(fold_f1s)
                    std_f1 = np.std(fold_f1s)
                    min_f1 = np.min(fold_f1s)
                    status = "COLLAPSED" if collapsed else ("UNSTABLE" if std_f1 > 0.15 else "OK")

                    results.append({
                        'preprocess': pp,
                        'task': task_id,
                        'model': model_name,
                        'config': cfg_name,
                        'mean_f1': mean_f1,
                        'std_f1': std_f1,
                        'min_f1': min_f1,
                        'status': status,
                    })
                    print(f"    {model_name:18s} | {cfg_name:14s} | "
                          f"F1={mean_f1:.3f}±{std_f1:.3f} | min={min_f1:.3f} | {status}")

    # Summary
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("SUMMARY: Best config per model")
    print("=" * 70)

    for model_name in MODELS:
        sub = df[df['model'] == model_name]
        ok = sub[sub['status'] == 'OK']
        n_ok = len(ok)
        n_total = len(sub)
        if n_ok > 0:
            best = ok.loc[ok['mean_f1'].idxmax()]
            print(f"  {model_name}: {n_ok}/{n_total} stable | "
                  f"best={best['mean_f1']:.3f} ({best['config']}, {best['preprocess']}, {best['task']})")
        else:
            print(f"  {model_name}: {n_ok}/{n_total} stable | ALL COLLAPSED/UNSTABLE")

    # Collapse rate comparison
    print("\n  Collapse rate by config:")
    for cfg_name in CONFIGS:
        sub = df[df['config'] == cfg_name]
        n_collapsed = len(sub[sub['status'] == 'COLLAPSED'])
        print(f"    {cfg_name:14s}: {n_collapsed}/{len(sub)} collapsed")

    # Save results
    out_path = ARTIFACTS_DIR / "models" / "dl_rescue_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # Parameter count
    print("\n  Model parameter counts:")
    for model_name, model_fn in MODELS.items():
        m = model_fn(1401, 2)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"    {model_name}: {n_params:,} params")


if __name__ == '__main__':
    run_experiment()
