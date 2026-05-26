"""Quick test: moderate config + p4 preprocessing for MG tasks."""
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


class SpectralAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch, ch // 4), nn.ReLU(),
            nn.Linear(ch // 4, ch), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv1d(ch, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x):
        x = x * self.ca(x).unsqueeze(-1)
        return x * self.sa(x)


class ResBlock(nn.Module):
    def __init__(self, ch, dp=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch))
        self.att = SpectralAttention(ch)
        self.dp = dp

    def forward(self, x):
        r = self.att(self.block(x))
        if self.training and self.dp > 0:
            r = r * (torch.rand(x.size(0), 1, 1, device=x.device) > self.dp).float()
        return F.gelu(r + x)


class AttResNet(nn.Module):
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
            nn.Dropout(0.4), nn.Linear(48, num_classes))

    def forward(self, x):
        return self.net(x)


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


def main():
    print("DL Rescue Round 2: Moderate config + p4 for MG")
    print("=" * 60)

    split_df = pd.read_csv(SPLITS_DIR / ACTIVE_SPLIT_FILE)

    # Configs to test
    configs = {
        'moderate(b32,lr5e-4)': {'lr': 5e-4, 'epochs': 200, 'patience': 30, 'batch': 32, 'wd': 3e-4, 'warmup': 10},
        'gentle(b48,lr3e-4)':   {'lr': 3e-4, 'epochs': 200, 'patience': 30, 'batch': 48, 'wd': 2e-4, 'warmup': 10},
        'best_p1(b32,lr1e-3)':  {'lr': 1e-3, 'epochs': 150, 'patience': 35, 'batch': 32, 'wd': 5e-4, 'warmup': 15},
    }

    # All 6 tasks, p1 and p4
    tasks = [
        ('P1_thiram', 'has_thiram', None, None, [0, 1]),
        ('P2_mg', 'has_mg', None, None, [0, 1]),
        ('P3_mba', 'has_mba', None, None, [0, 1]),
        ('G1_thiram', 'c_thiram', 'has_thiram', 1, [4, 5, 6]),
        ('G2_mg', 'c_mg', 'has_mg', 1, [4, 5, 6]),
        ('G3_mba', 'c_mba', 'has_mba', 1, [4, 5, 6]),
    ]

    results = []
    for pp in ['p1', 'p4']:
        X = np.load(PROCESSED_DIR / f"X_{pp}.npy")
        print(f"\n{'='*60}")
        print(f"Preprocessing: {pp}")
        print(f"{'='*60}")

        for task_name, y_col, filt_col, filt_val, classes in tasks:
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
            nc = len(classes)

            for cn, cfg in configs.items():
                fold_f1s = []
                for fi in range(5):
                    tr_mask = folds != fi
                    te_mask = folds == fi
                    torch.manual_seed(SEED + fi)
                    np.random.seed(SEED + fi)
                    model = AttResNet(X_t.shape[1], nc)
                    f1 = train_eval(model, X_t[tr_mask], y[tr_mask],
                                    X_t[te_mask], y[te_mask], cfg, seed=SEED+fi)
                    fold_f1s.append(f1)

                mean_f1 = np.mean(fold_f1s)
                std_f1 = np.std(fold_f1s)
                min_f1 = np.min(fold_f1s)
                status = "COLLAPSE" if min_f1 < 0.35 else ("UNSTABLE" if std_f1 > 0.15 else "OK")
                results.append({
                    'pp': pp, 'task': task_name, 'config': cn,
                    'mean': mean_f1, 'std': std_f1, 'min': min_f1, 'status': status
                })
                print(f"  {task_name:10s} | {cn:22s} | F1={mean_f1:.3f}±{std_f1:.3f} min={min_f1:.3f} [{status}]")

    # Best per task
    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("BEST CONFIG PER TASK (AttResNet):")
    print(f"{'='*60}")
    for task_name in [t[0] for t in tasks]:
        sub = df[df['task'] == task_name]
        best = sub.loc[sub['mean'].idxmax()]
        print(f"  {task_name:10s}: F1={best['mean']:.3f}±{best['std']:.3f} | {best['pp']} + {best['config']} [{best['status']}]")

    # Compare with RF reference (from previous experiments)
    print(f"\n  Reference: RF on p1 typically gets P1~0.93, P2~0.72, P3~0.90, G1~0.65, G2~0.45, G3~0.55")


if __name__ == '__main__':
    main()
