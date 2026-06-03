"""Read-only check of GPT's MG peak-suppression claim. Recomputes peak-height
ratios of MG bands in mixtures vs pure MG at the SAME MG level. No writes."""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "pure63_mainline" / "processed"
SPLIT = ROOT / "data" / "pure63_mainline" / "splits" / "cv_split_random_5fold.csv"

X = np.load(PROC / "X_p4.npy")          # no-norm spectra
wn = np.load(PROC / "wavenumber.npy")
df = pd.read_csv(SPLIT)
print(f"X_p4 shape={X.shape}  wn={wn.shape}  csv rows={len(df)}")
assert X.shape[0] == len(df), "row mismatch — alignment assumption broken"
assert X.shape[1] == wn.shape[0]

def peak_h(level_mask, wn_center, half=6):
    """max intensity in +/-half cm^-1 window, per spectrum."""
    band = (wn >= wn_center - half) & (wn <= wn_center + half)
    return X[np.ix_(level_mask, band)].max(axis=1)

MG_PEAKS = [1172, 1220, 1394, 1616]
print("\nrows by group:")
for tag, m in {
    "pure MG":   (df.has_mg==1)&(df.has_thiram==0)&(df.has_mba==0),
    "MG+MBA":    (df.has_mg==1)&(df.has_thiram==0)&(df.has_mba==1),
    "MG+Thiram": (df.has_mg==1)&(df.has_thiram==1)&(df.has_mba==0),
    "ternary":   (df.has_mg==1)&(df.has_thiram==1)&(df.has_mba==1),
}.items():
    print(f"  {tag:11s}: {int(m.sum())} spectra")

# pure-MG reference median peak height per MG level
ref = {}
for lvl in [4,5,6]:
    m = (df.has_mg==1)&(df.has_thiram==0)&(df.has_mba==0)&(df.c_mg==lvl)
    ref[lvl] = {p: np.median(peak_h(m.values, p)) for p in MG_PEAKS}

def ratios(cond_mask):
    """ratio of each spectrum's peak height to pure-MG ref at its own MG level."""
    out = {p: [] for p in MG_PEAKS}
    for lvl in [4,5,6]:
        m = (cond_mask & (df.c_mg==lvl)).values
        if not m.any(): continue
        for p in MG_PEAKS:
            h = peak_h(m, p)
            out[p].extend((h / ref[lvl][p]).tolist())
    return out

print("\nMedian peak-height ratio vs pure MG (same MG level):")
print(f"{'peak':>6} {'MG+MBA':>10} {'MG+Thiram':>10} {'ternary':>10}")
conds = {
    "MG+MBA":    (df.has_mg==1)&(df.has_thiram==0)&(df.has_mba==1),
    "MG+Thiram": (df.has_mg==1)&(df.has_thiram==1)&(df.has_mba==0),
    "ternary":   (df.has_mg==1)&(df.has_thiram==1)&(df.has_mba==1),
}
R = {k: ratios(v) for k,v in conds.items()}
for p in MG_PEAKS:
    print(f"{p:>6} {np.median(R['MG+MBA'][p]):>10.3f} "
          f"{np.median(R['MG+Thiram'][p]):>10.3f} {np.median(R['ternary'][p]):>10.3f}")

print("\nMG 1616 — fraction of spectra below threshold:")
for k in conds:
    arr = np.array(R[k][1616])
    print(f"  {k:11s} n={len(arr):3d}  <0.3x: {int((arr<0.3).sum())}/{len(arr)}  "
          f"<0.5x: {int((arr<0.5).sum())}/{len(arr)}  <0.6x: {int((arr<0.6).sum())}/{len(arr)}")
