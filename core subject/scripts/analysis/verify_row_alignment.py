"""Read-only DECISIVE verification that X_p4.npy row i == the spectrum named in
split.csv row i. For sampled rows we read the original CSV by its file_path,
re-run the EXACT p4 pipeline (reusing src.dataset functions), and compare to
X_p4[i]. If a row mismatches we search which X_p4 row it DOES match, to diagnose
any offset/shuffle. No writes."""
import numpy as np, pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.dataset import read_bwram_spectrum, _compute_preprocess_variant
from src.config import WN_MIN, WN_MAX, WN_STEP
from scipy.interpolate import interp1d

PROC = ROOT / "data" / "pure63_mainline" / "processed"
SPLIT = ROOT / "data" / "pure63_mainline" / "splits" / "cv_split_random_5fold.csv"

X = np.load(PROC / "X_p4.npy")
df = pd.read_csv(SPLIT)
wn_common = np.arange(WN_MIN, WN_MAX + WN_STEP, WN_STEP)
print(f"X_p4 {X.shape}  csv rows {len(df)}  wn_common {wn_common.shape}")

def build_raw_one(fp):
    wn, inten = read_bwram_spectrum(fp)
    if wn is None or len(wn) < 50:
        return None
    order = np.argsort(wn); wn, inten = wn[order], inten[order]
    mask = (wn >= WN_MIN - 20) & (wn <= WN_MAX + 20)
    wn, inten = wn[mask], inten[mask]
    if len(wn) < 50:
        return None
    f = interp1d(wn, inten, kind='linear', bounds_error=False, fill_value=0)
    return f(wn_common)

# sample indices spread across the array + a few MG rows
idx = sorted(set(np.linspace(0, len(df) - 1, 12).astype(int).tolist()
                 + df.index[df.has_mg == 1].tolist()[:4]))
print(f"\nchecking {len(idx)} rows: {idx}")
print(f"{'row':>4} {'folder':>10} {'file':>10} {'allclose':>9} {'corr':>7} {'maxdiff':>10}")
n_ok = 0
for i in idx:
    raw = build_raw_one(df.loc[i, 'file_path'])
    if raw is None:
        print(f"{i:>4}  <unreadable file>"); continue
    p4 = _compute_preprocess_variant('p4', raw[None, :])[0]
    ok = np.allclose(p4, X[i], rtol=1e-4, atol=1e-6)
    corr = np.corrcoef(p4, X[i])[0, 1]
    md = np.max(np.abs(p4 - X[i]))
    n_ok += ok
    flag = "" if ok else "  <-- MISMATCH"
    print(f"{i:>4} {df.loc[i,'folder_name']:>10} {df.loc[i,'file_name']:>10} "
          f"{str(ok):>9} {corr:>7.4f} {md:>10.3e}{flag}")
    if not ok:  # diagnose: which row does this file actually match?
        corrs = np.corrcoef(p4, X)[0, 1:]
        best = int(np.argmax(corrs))
        print(f"      best-matching X_p4 row = {best} (corr {corrs[best]:.4f})")

print(f"\nRESULT: {n_ok}/{len(idx)} rows aligned. "
      f"{'ALIGNED — X_p4[i] == split.csv[i]' if n_ok == len(idx) else 'ALIGNMENT BROKEN'}")
