"""
RSD vs Preprocessing Study
Goal: Find preprocessing that gives reportable RSD (<15-20%) at characteristic peaks.

Logic:
- SERS hotspot variation is primarily multiplicative (intensity scaling)
- Normalization (SNV, vector norm, max norm) removes this
- Papers report RSD at specific peaks, not full-spectrum
- Peak area (integrated) is more robust than peak height (point)

Characteristic peaks (cm-1):
- Thiram: 1380 (CS stretch), 1145 (CN stretch), 560 (SS stretch)
- MG: 1617 (ring stretch), 1175 (C-H bend), 800 (ring)
- MBA: 1590 (ring), 1080 (CS stretch), 1180 (CH bend)
"""
import sys
sys.path.insert(0, "d:/通过拉曼光谱预测物及其浓度/core subject/src")
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("d:/通过拉曼光谱预测物及其浓度/core subject/data/pure63_mainline/processed")
SPLIT_FILE = Path("d:/通过拉曼光谱预测物及其浓度/core subject/data/pure63_mainline/splits/cv_split_pure63_main.csv")

wn = np.load(DATA_DIR / "wavenumber.npy")
meta = pd.read_csv(SPLIT_FILE)

PEAKS = {
    'Thiram_1380': 1380, 'Thiram_1145': 1145, 'Thiram_560': 560,
    'MG_1617': 1617, 'MG_1175': 1175, 'MG_800': 800,
    'MBA_1590': 1590, 'MBA_1080': 1080, 'MBA_1180': 1180,
}
HALF_WIDTH = 10  # +/- 10 cm-1 for peak area integration

def find_peak_idx(target_wn):
    return np.argmin(np.abs(wn - target_wn))

def peak_height(spectra, target_wn):
    idx = find_peak_idx(target_wn)
    return spectra[:, idx]

def peak_area(spectra, target_wn, half_w=HALF_WIDTH):
    lo = find_peak_idx(target_wn - half_w)
    hi = find_peak_idx(target_wn + half_w)
    if lo > hi:
        lo, hi = hi, lo
    return np.trapezoid(spectra[:, lo:hi+1], wn[lo:hi+1], axis=1)

def snv(spectra):
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    std[std == 0] = 1
    return (spectra - mean) / std

def vector_norm(spectra):
    norms = np.linalg.norm(spectra, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return spectra / norms

def max_norm(spectra):
    mx = spectra.max(axis=1, keepdims=True)
    mx[mx == 0] = 1
    return spectra / mx

def area_norm(spectra):
    areas = np.abs(spectra).sum(axis=1, keepdims=True)
    areas[areas == 0] = 1
    return spectra / areas

def calc_rsd(values):
    values = values[values > 0]
    if len(values) < 3:
        return np.nan
    return (np.std(values, ddof=1) / np.mean(values)) * 100


# ============ Main analysis ============
PREPS = {
    'raw': DATA_DIR / "X_raw.npy",
    'p1_SNV': DATA_DIR / "X_p1.npy",
    'p2_D1_SNV': DATA_DIR / "X_p2.npy",
    'p3_VecNorm': DATA_DIR / "X_p3.npy",
    'p4_noNorm': DATA_DIR / "X_p4.npy",
}

# Additional normalizations applied on top of raw
EXTRA_NORMS = {
    'raw+SNV': ('raw', snv),
    'raw+VecNorm': ('raw', vector_norm),
    'raw+MaxNorm': ('raw', max_norm),
    'raw+AreaNorm': ('raw', area_norm),
    'p4+SNV': ('p4_noNorm', snv),
    'p4+MaxNorm': ('p4_noNorm', max_norm),
}

# Single-component folders for RSD (pure substances, no mixture)
SINGLE_FOLDERS = ['K4', 'K5', 'K6', '美4', '美5', '美6', 'm4', 'm5', 'm6']
FOLDER_PEAKS = {
    'K4': ['MG_1617', 'MG_1175'], 'K5': ['MG_1617', 'MG_1175'],
    'K6': ['MG_1617', 'MG_1175'],
    '美4': ['Thiram_1380', 'Thiram_1145'], '美5': ['Thiram_1380', 'Thiram_1145'],
    '美6': ['Thiram_1380', 'Thiram_1145'],
    'm4': ['MBA_1080', 'MBA_1590'], 'm5': ['MBA_1080', 'MBA_1590'],
    'm6': ['MBA_1080', 'MBA_1590'],
}

print("=" * 80)
print("RSD ANALYSIS: Preprocessing × Peak × Folder")
print("=" * 80)

# Load all data
loaded = {}
for name, path in PREPS.items():
    loaded[name] = np.load(path)
for name, (base, func) in EXTRA_NORMS.items():
    loaded[name] = func(loaded[base].copy())

all_prep_names = list(PREPS.keys()) + list(EXTRA_NORMS.keys())

results = []
for folder in SINGLE_FOLDERS:
    mask = meta['folder_name'] == folder
    indices = meta.index[mask].values
    if len(indices) < 3:
        continue
    peaks_to_check = FOLDER_PEAKS[folder]
    for prep_name in all_prep_names:
        X = loaded[prep_name][indices]
        for peak_name in peaks_to_check:
            target_wn = PEAKS[peak_name]
            h = peak_height(X, target_wn)
            a = peak_area(X, target_wn)
            rsd_h = calc_rsd(np.abs(h))
            rsd_a = calc_rsd(np.abs(a))
            results.append({
                'folder': folder, 'prep': prep_name,
                'peak': peak_name, 'rsd_height%': rsd_h,
                'rsd_area%': rsd_a, 'n_spectra': len(indices)
            })

df = pd.DataFrame(results)

# Summary: best preprocessing per folder
print("\n\n--- BEST PREPROCESSING PER FOLDER (peak height RSD) ---")
print(f"{'Folder':<8} {'Peak':<15} {'Best Prep':<15} {'RSD%':<8} {'n'}")
print("-" * 60)
for folder in SINGLE_FOLDERS:
    sub = df[df['folder'] == folder]
    if sub.empty:
        continue
    for peak in FOLDER_PEAKS[folder]:
        sub2 = sub[sub['peak'] == peak]
        best = sub2.loc[sub2['rsd_height%'].idxmin()]
        print(f"{folder:<8} {peak:<15} {best['prep']:<15} "
              f"{best['rsd_height%']:.1f}    {best['n_spectra']}")

print("\n\n--- BEST PREPROCESSING PER FOLDER (peak area RSD) ---")
print(f"{'Folder':<8} {'Peak':<15} {'Best Prep':<15} {'RSD%':<8} {'n'}")
print("-" * 60)
for folder in SINGLE_FOLDERS:
    sub = df[df['folder'] == folder]
    if sub.empty:
        continue
    for peak in FOLDER_PEAKS[folder]:
        sub2 = sub[sub['peak'] == peak]
        best = sub2.loc[sub2['rsd_area%'].idxmin()]
        print(f"{folder:<8} {peak:<15} {best['prep']:<15} "
              f"{best['rsd_area%']:.1f}    {best['n_spectra']}")

# Full table for the most promising preps
print("\n\n--- FULL COMPARISON: Normalized preps ---")
norm_preps = ['p1_SNV', 'p3_VecNorm', 'raw+SNV', 'raw+MaxNorm', 'p4+SNV']
print(f"{'Folder':<6} {'Peak':<14}", end="")
for p in norm_preps:
    print(f" {p:<12}", end="")
print()
print("-" * (20 + 12 * len(norm_preps)))
for folder in SINGLE_FOLDERS:
    for peak in FOLDER_PEAKS[folder]:
        print(f"{folder:<6} {peak:<14}", end="")
        for p in norm_preps:
            row = df[(df['folder']==folder) & (df['peak']==peak) & (df['prep']==p)]
            if not row.empty:
                v = row.iloc[0]['rsd_height%']
                print(f" {v:>6.1f}%     ", end="")
            else:
                print(f" {'N/A':<12}", end="")
        print()

print("\n\nDone.", flush=True)
