"""Quick DL rescue test: minimal comparison to verify training fix works."""
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import PROCESSED_DIR, SPLITS_DIR, ACTIVE_SPLIT_FILE, SEED

# ============ Models ============
class SpectralAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch, ch // 4), nn.ReLU(),
            nn.Linear(ch // 4, ch), nn.Sigmoid(),
        )
        self.sa = nn.Sequential(nn.Conv1d(ch, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x):
        x = x * self.ca(x).unsqueeze(-1)
        return x * self.sa(x)


class ResBlock(nn.Module):
    def __init__(self, ch, drop_path=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch),
        )
        self.att = SpectralAttention(ch)
        self.dp = drop_path

    def forward(self, x):
        r = self.att(self.block(x))
        if self.training and self.dp > 0:
            r = r * (torch.rand(x.size(0), 1, 1, device=x.device) > self.dp).float()
        return F.gelu(r + x)


class AttResNet(nn.Module):
    """Lightweight Attention-ResNet. ~35K params."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.GELU(), nn.MaxPool1d(2),
            ResBlock(16), ResBlock(16),
            nn.Conv1d(16, 32, 3, stride=2, padding=1), nn.BatchNorm1d(32),
            ResBlock(32), ResBlock(32),
            nn.Conv1d(32, 48, 3, stride=2, padding=1), nn.BatchNorm1d(48),
            ResBlock(48),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(48, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class BaselineResNet(nn.Module):
    """Current project ResNet (simplified). ~60K params."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            self._block(16), self._block(16),
            nn.Conv1d(16, 32, 1, stride=2), nn.BatchNorm1d(32),
            self._block(32), self._block(32),
            nn.Conv1d(32, 64, 1, stride=2), nn.BatchNorm1d(64),
            self._block(64), self._block(64),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64, num_classes),
        )

    @staticmethod
    def _block(ch):
        return _SimpleRes(ch)

    def forward(self, x):
        return self.net(x)


class _SimpleRes(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch))

    def forward(self, x):
        return F.relu(self.b(x) + x)


# ============ Training ============
def train_eval(model, X_tr, y_tr, X_te, y_te, cfg, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    torch.manual_seed(seed)

    n_val = max(int(len(X_tr) * 0.15), 8)
    perm = np.random.RandomState(seed).permutation(len(X_tr))
    vi, ti = perm[:n_val], perm[n_val:]

    Xt = torch.tensor(X_tr[ti], dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor(y_tr[ti], dtype=torch.long)
    Xv = torch.tensor(X_tr[vi], dtype=torch.float32).unsqueeze(1)
    yv = torch.tensor(y_tr[vi], dtype=torch.long)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=cfg['batch'], shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    warmup = cfg.get('warmup', 0)
    total = cfg['epochs']

    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / max(warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(total - warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    classes, counts = np.unique(y_tr[ti], return_counts=True)
    w = torch.tensor(len(y_tr[ti]) / (len(classes) * counts), dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)

    best_f1, best_state, wait = -1, None, 0
    for ep in range(total):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vp = model(Xv.to(device)).argmax(1).cpu().numpy()
        vf1 = f1_score(yv.numpy(), vp, average='macro', zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg['patience']:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Xtest = torch.tensor(X_te, dtype=torch.float32).unsqueeze(1).to(device)
        preds = model(Xtest).argmax(1).cpu().numpy()
    return f1_score(y_te, preds, average='macro', zero_division=0)


# ============ Main ============
def main():
    print("DL Rescue Quick Test")
    print("=" * 60)

    split_df = pd.read_csv(SPLITS_DIR / ACTIVE_SPLIT_FILE)
    X = np.load(PROCESSED_DIR / "X_p1.npy")

    # Test tasks: P2 (MG presence, hardest) and G2 (MG grade, most collapse)
    test_tasks = [
        ('P2_mg_presence', 'has_mg', None, None, [0, 1]),
        ('G2_mg_grade', 'c_mg', 'has_mg', 1, [4, 5, 6]),
        ('P1_thiram_presence', 'has_thiram', None, None, [0, 1]),
    ]

    configs = {
        'old(b128,lr3e-4)': {'lr': 3e-4, 'epochs': 150, 'patience': 20, 'batch': 128, 'wd': 1e-4, 'warmup': 0},
        'fix(b32,warmup)':  {'lr': 1e-3, 'epochs': 150, 'patience': 35, 'batch': 32,  'wd': 5e-4, 'warmup': 15},
    }

    models_dict = {
        'BaselineResNet': BaselineResNet,
        'AttResNet': AttResNet,
    }

    # Print param counts
    for mn, mc in models_dict.items():
        m = mc(1401, 2)
        n = sum(p.numel() for p in m.parameters())
        print(f"  {mn}: {n:,} params")
    print()

    n_folds_test = 3  # quick test with 3 folds

    for task_name, y_col, filt_col, filt_val, classes in test_tasks:
        if filt_col:
            mask = split_df[filt_col] == filt_val
            df_t = split_df[mask].reset_index(drop=True)
            X_t = X[mask.values]
        else:
            df_t = split_df
            X_t = X

        y_raw = df_t[y_col].values
        lmap = {c: i for i, c in enumerate(classes)}
        y = np.array([lmap[v] for v in y_raw])
        folds = df_t['fold_id'].values

        print(f"Task: {task_name} (n={len(y)}, classes={len(classes)})")
        print(f"  {'Model':<16} | {'Config':<18} | F1 per fold | Mean±Std")
        print(f"  {'-'*70}")

        for mn, mc in models_dict.items():
            for cn, cfg in configs.items():
                fold_f1s = []
                for fi in range(n_folds_test):
                    tr_mask = folds != fi
                    te_mask = folds == fi
                    model = mc(X_t.shape[1], len(classes))
                    f1 = train_eval(model, X_t[tr_mask], y[tr_mask],
                                    X_t[te_mask], y[te_mask], cfg, seed=SEED+fi)
                    fold_f1s.append(f1)
                    print(f"    fold{fi}: {f1:.3f}", end="", flush=True)

                mean_f1 = np.mean(fold_f1s)
                std_f1 = np.std(fold_f1s)
                status = "COLLAPSE" if min(fold_f1s) < 0.35 else "OK"
                print(f"\n  {mn:<16} | {cn:<18} | {[f'{x:.3f}' for x in fold_f1s]} | "
                      f"{mean_f1:.3f}±{std_f1:.3f} [{status}]")
        print()


if __name__ == '__main__':
    main()
