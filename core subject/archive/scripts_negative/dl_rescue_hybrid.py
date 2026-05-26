"""CNN feature extractor + RF: use DL for representation, RF for classification."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
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
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch))
        self.att = SpectralAttention(ch)

    def forward(self, x):
        return F.gelu(self.att(self.block(x)) + x)


class AttResNetEncoder(nn.Module):
    """Feature extractor: outputs 48-dim embedding."""
    def __init__(self, input_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.GELU(), nn.MaxPool1d(2),
            ResBlock(16), ResBlock(16),
            nn.Conv1d(16, 32, 3, stride=2, padding=1), nn.BatchNorm1d(32),
            ResBlock(32), ResBlock(32),
            nn.Conv1d(32, 48, 3, stride=2, padding=1), nn.BatchNorm1d(48),
            ResBlock(48),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )

    def forward(self, x):
        return self.backbone(x)


def pretrain_encoder(encoder, X_all, epochs=100, lr=5e-4, batch=64, seed=42):
    """Self-supervised pretraining: reconstruct masked spectrum regions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)

    decoder = nn.Sequential(
        nn.Linear(48, 128), nn.GELU(),
        nn.Linear(128, X_all.shape[1]),
    ).to(device)
    encoder = encoder.to(device)

    Xt = torch.tensor(X_all, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(Xt), batch_size=batch, shuffle=True)

    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    for ep in range(epochs):
        encoder.train()
        decoder.train()
        for (xb,) in loader:
            xb = xb.to(device)
            # Random masking: zero out 30% of wavenumber positions
            mask = (torch.rand_like(xb) > 0.3).float()
            xb_masked = xb * mask
            features = encoder(xb_masked)
            recon = decoder(features)
            # Only compute loss on masked positions
            loss = F.mse_loss(recon * (1 - mask.squeeze(1)), xb.squeeze(1) * (1 - mask.squeeze(1)))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()

    return encoder


def extract_features(encoder, X, batch=256):
    """Extract features using pretrained encoder."""
    device = next(encoder.parameters()).device
    encoder.eval()
    feats = []
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(Xt), batch_size=batch, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            f = encoder(xb.to(device)).cpu().numpy()
            feats.append(f)
    return np.concatenate(feats, axis=0)


def main():
    print("DL Hybrid: Self-supervised CNN encoder + RF classifier")
    print("=" * 60)

    split_df = pd.read_csv(SPLITS_DIR / ACTIVE_SPLIT_FILE)

    tasks = [
        ('P1_thiram', 'has_thiram', None, None, [0, 1]),
        ('P2_mg', 'has_mg', None, None, [0, 1]),
        ('P3_mba', 'has_mba', None, None, [0, 1]),
        ('G1_thiram', 'c_thiram', 'has_thiram', 1, [4, 5, 6]),
        ('G2_mg', 'c_mg', 'has_mg', 1, [4, 5, 6]),
        ('G3_mba', 'c_mba', 'has_mba', 1, [4, 5, 6]),
    ]

    for pp in ['p1', 'p4']:
        X = np.load(PROCESSED_DIR / f"X_{pp}.npy")
        print(f"\nPreprocessing: {pp} | X shape: {X.shape}")
        print("-" * 60)

        # Pretrain encoder on ALL data (self-supervised, no labels)
        print("  Pretraining encoder (masked reconstruction)...", flush=True)
        encoder = AttResNetEncoder(X.shape[1])
        encoder = pretrain_encoder(encoder, X, epochs=80, seed=SEED)
        print("  Done. Extracting features...")

        # Extract features for all samples
        feats_all = extract_features(encoder, X)
        print(f"  Feature shape: {feats_all.shape}")

        folds = split_df['fold_id'].values

        for task_name, y_col, filt_col, filt_val, classes in tasks:
            if filt_col:
                mask = split_df[filt_col] == filt_val
                df_t = split_df[mask].reset_index(drop=True)
                feats_t = feats_all[mask.values]
                folds_t = df_t['fold_id'].values
            else:
                feats_t = feats_all
                folds_t = folds
                df_t = split_df

            y_raw = df_t[y_col].values
            lmap = {c: i for i, c in enumerate(classes)}
            y = np.array([lmap[v] for v in y_raw])

            fold_f1s = []
            for fi in range(5):
                tr_mask = folds_t != fi
                te_mask = folds_t == fi
                rf = RandomForestClassifier(
                    n_estimators=200, max_depth=None, min_samples_leaf=2,
                    class_weight='balanced', random_state=SEED, n_jobs=-1)
                rf.fit(feats_t[tr_mask], y[tr_mask])
                preds = rf.predict(feats_t[te_mask])
                fold_f1s.append(f1_score(y[te_mask], preds, average='macro', zero_division=0))

            mean_f1 = np.mean(fold_f1s)
            std_f1 = np.std(fold_f1s)
            min_f1 = np.min(fold_f1s)
            status = "COLLAPSE" if min_f1 < 0.35 else ("UNSTABLE" if std_f1 > 0.15 else "OK")
            print(f"  {task_name:10s} | F1={mean_f1:.3f}±{std_f1:.3f} min={min_f1:.3f} [{status}]")

    print("\n  Reference (RF on raw features):")
    print("    P1~0.93, P2~0.72, P3~0.90, G1~0.65, G2~0.45, G3~0.55")


if __name__ == '__main__':
    main()
