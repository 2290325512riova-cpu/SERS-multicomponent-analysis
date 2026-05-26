"""Final focused test: AttResNet best config, all 6 tasks, 5 folds, p1 only."""
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


class SA(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch, ch//4), nn.ReLU(), nn.Linear(ch//4, ch), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv1d(ch, 1, 7, padding=3), nn.Sigmoid())
    def forward(self, x):
        x = x * self.ca(x).unsqueeze(-1)
        return x * self.sa(x)

class RB(nn.Module):
    def __init__(self, ch, dp=0.1):
        super().__init__()
        self.b = nn.Sequential(nn.Conv1d(ch,ch,3,padding=1), nn.BatchNorm1d(ch),
            nn.GELU(), nn.Conv1d(ch,ch,3,padding=1), nn.BatchNorm1d(ch))
        self.a = SA(ch); self.dp = dp
    def forward(self, x):
        r = self.a(self.b(x))
        if self.training and self.dp > 0:
            r = r * (torch.rand(x.size(0),1,1,device=x.device) > self.dp).float()
        return F.gelu(r + x)

class AttResNet(nn.Module):
    def __init__(self, dim, nc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,16,7,padding=3), nn.BatchNorm1d(16), nn.GELU(), nn.MaxPool1d(2),
            RB(16), RB(16),
            nn.Conv1d(16,32,3,stride=2,padding=1), nn.BatchNorm1d(32),
            RB(32), RB(32),
            nn.Conv1d(32,48,3,stride=2,padding=1), nn.BatchNorm1d(48),
            RB(48),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(48,nc))
    def forward(self, x): return self.net(x)


def train_eval(model, X_tr, y_tr, X_te, y_te, seed=42):
    cfg = {'lr': 1e-3, 'epochs': 150, 'patience': 35, 'batch': 32, 'wd': 5e-4, 'warmup': 15}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_val = max(int(len(X_tr) * 0.15), 8)
    perm = np.random.RandomState(seed).permutation(len(X_tr))
    vi, ti = perm[:n_val], perm[n_val:]

    Xt = torch.tensor(X_tr[ti], dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor(y_tr[ti], dtype=torch.long)
    Xv = torch.tensor(X_tr[vi], dtype=torch.float32).unsqueeze(1)
    yv = torch.tensor(y_tr[vi], dtype=torch.long)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=cfg['batch'], shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    total = cfg['epochs']
    warmup = cfg['warmup']

    def lr_fn(ep):
        if ep < warmup: return (ep+1)/warmup
        return 0.5*(1+np.cos(np.pi*(ep-warmup)/max(total-warmup,1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    classes, counts = np.unique(y_tr[ti], return_counts=True)
    w = torch.tensor(len(y_tr[ti])/(len(classes)*counts), dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)

    best_f1, best_state, wait = -1, None, 0
    for ep in range(total):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xv.to(device)).argmax(1).cpu().numpy()
        vf1 = f1_score(yv.numpy(), vp, average='macro', zero_division=0)
        if vf1 > best_f1:
            best_f1, wait = vf1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg['patience']: break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(X_te, dtype=torch.float32).unsqueeze(1).to(device)).argmax(1).cpu().numpy()
    return f1_score(y_te, p, average='macro', zero_division=0)


def main():
    print("AttResNet Final 5-Fold Test (best config: b32, lr1e-3, warmup15)", flush=True)
    print("="*60, flush=True)

    split_df = pd.read_csv(SPLITS_DIR / ACTIVE_SPLIT_FILE)
    X = np.load(PROCESSED_DIR / "X_p1.npy")
    folds_all = split_df['fold_id'].values

    tasks = [
        ('P1_thiram_presence', 'has_thiram', None, None, [0,1]),
        ('P2_mg_presence', 'has_mg', None, None, [0,1]),
        ('P3_mba_presence', 'has_mba', None, None, [0,1]),
        ('G1_thiram_grade', 'c_thiram', 'has_thiram', 1, [4,5,6]),
        ('G2_mg_grade', 'c_mg', 'has_mg', 1, [4,5,6]),
        ('G3_mba_grade', 'c_mba', 'has_mba', 1, [4,5,6]),
    ]

    print(f"\n{'Task':<20} | {'Fold F1s':<40} | {'Mean±Std':<12} | Status", flush=True)
    print("-"*90, flush=True)

    for task_name, y_col, filt_col, filt_val, classes in tasks:
        if filt_col:
            mask = split_df[filt_col] == filt_val
            df_t = split_df[mask].reset_index(drop=True)
            X_t = X[mask.values]
            folds = df_t['fold_id'].values
        else:
            df_t = split_df
            X_t = X
            folds = folds_all

        y_raw = df_t[y_col].values
        lmap = {c: i for i, c in enumerate(classes)}
        y = np.array([lmap[v] for v in y_raw])
        nc = len(classes)

        fold_f1s = []
        for fi in range(5):
            tr_mask = folds != fi
            te_mask = folds == fi
            model = AttResNet(X_t.shape[1], nc)
            f1 = train_eval(model, X_t[tr_mask], y[tr_mask], X_t[te_mask], y[te_mask], seed=SEED+fi)
            fold_f1s.append(f1)
            print(f"  {task_name} fold{fi}: {f1:.3f}", flush=True)

        m, s = np.mean(fold_f1s), np.std(fold_f1s)
        mn = np.min(fold_f1s)
        st = "COLLAPSE" if mn < 0.35 else ("UNSTABLE" if s > 0.15 else "OK")
        print(f"{task_name:<20} | {str([f'{x:.3f}' for x in fold_f1s]):<40} | {m:.3f}±{s:.3f} | {st}", flush=True)
        print(flush=True)

    n = sum(p.numel() for p in AttResNet(1401, 2).parameters())
    print(f"\nAttResNet params: {n:,}", flush=True)
    print("\nRF reference: P1~0.93, P2~0.72, P3~0.90, G1~0.65, G2~0.45, G3~0.55", flush=True)


if __name__ == '__main__':
    main()
