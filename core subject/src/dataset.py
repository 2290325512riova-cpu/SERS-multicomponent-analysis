"""
Dataset module: folder scanning, metadata parsing, spectrum reading,
preprocessing, and cross-validation splitting.

Current active layout:
        总混合光谱/
                纯净物混合光谱/                -> main pure benchmark source
                真实样品（土壤）混合光谱/      -> external validation source

Folder naming convention for design folders:
        美X = Thiram at molar label code X    (X ∈ {4, 5, 6})
        KX  = MG at molar label code X
        mX  = MBA at molar label code X
    Combinations concatenated: 美4K5m6 = Thiram 4 + MG 5 + MBA 6

Strategy 3: MBA is a regular component, NOT internal standard.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    CONC_LEVEL_TO_MOLAR,
    DATA_DIR, PURE_DATA_DIR, SOIL_DATA_DIR, PROCESSED_DIR, SPLITS_DIR, OUTPUT_DIR,
    WN_MIN, WN_MAX, WN_STEP,
    SG_WINDOW, SG_POLY, ALS_LAM, ALS_P, ALS_NITER,
    COSMIC_THRESHOLD, COSMIC_WINDOW,
    N_FOLDS, SEED, ACTIVE_SPLIT_FILE,
    PREPROCESS_CACHE_FILES, PREPROCESS_NAMES, PREPROCESS_TAGS,
    RANDOM_SPLIT_FILE, RANDOM_SPLIT_QC_BY_FOLD_FILE,
    RANDOM_SPLIT_QC_BY_STRATUM_FILE, RANDOM_STRATIFY_COLS,
)


# ====================== Folder Name Parsing ======================

def parse_folder_name(folder_name: str) -> dict:
    """Parse folder name into substance composition and concentrations.

    Returns dict with keys:
        c_thiram, c_mg, c_mba (int: historical label code 0/4/5/6),
        has_thiram, has_mg, has_mba (bool),
        mixture_order (int: 1/2/3),
        family (str: single/binary_XX_YY/ternary)
    """
    fn = folder_name.strip()
    c_thiram, c_mg, c_mba = 0, 0, 0

    # Ternary: 美XKYmZ
    m = re.match(r'^美(\d)K(\d)m(\d)$', fn)
    if m:
        c_thiram, c_mg, c_mba = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return _build_result(c_thiram, c_mg, c_mba)

    # Binary: 美XKY (Thiram + MG)
    m = re.match(r'^美(\d)K(\d)$', fn)
    if m:
        c_thiram, c_mg = int(m.group(1)), int(m.group(2))
        return _build_result(c_thiram, c_mg, c_mba)

    # Binary: 美XmY (Thiram + MBA)
    m = re.match(r'^美(\d)m(\d)$', fn)
    if m:
        c_thiram = int(m.group(1))
        c_mba = int(m.group(2))
        return _build_result(c_thiram, c_mg, c_mba)

    # Binary: KXmY (MG + MBA)
    m = re.match(r'^K(\d)m(\d)$', fn)
    if m:
        c_mg, c_mba = int(m.group(1)), int(m.group(2))
        return _build_result(c_thiram, c_mg, c_mba)

    # Single: 美X (Thiram only)
    m = re.match(r'^美(\d)$', fn)
    if m:
        c_thiram = int(m.group(1))
        return _build_result(c_thiram, c_mg, c_mba)

    # Single: KX (MG only)
    m = re.match(r'^K(\d)$', fn)
    if m:
        c_mg = int(m.group(1))
        return _build_result(c_thiram, c_mg, c_mba)

    # Single: mX (MBA only)
    m = re.match(r'^m(\d)$', fn)
    if m:
        c_mba = int(m.group(1))
        return _build_result(c_thiram, c_mg, c_mba)

    raise ValueError(f"Cannot parse folder name: {folder_name}")


def _build_result(c_thiram: int, c_mg: int, c_mba: int) -> dict:
    has_t = c_thiram > 0
    has_m = c_mg > 0
    has_b = c_mba > 0
    order = sum([has_t, has_m, has_b])

    if order == 1:
        family = 'single'
    elif order == 2:
        parts = []
        if has_t: parts.append('Thiram')
        if has_m: parts.append('MG')
        if has_b: parts.append('MBA')
        family = f'binary_{"_".join(parts)}'
    else:
        family = 'ternary'

    return {
        'c_thiram': c_thiram, 'c_mg': c_mg, 'c_mba': c_mba,
        'has_thiram': has_t, 'has_mg': has_m, 'has_mba': has_b,
        'mixture_order': order, 'family': family,
    }


def _empty_design_result(sample_role: str) -> dict:
    return {
        'c_thiram': 0,
        'c_mg': 0,
        'c_mba': 0,
        'has_thiram': False,
        'has_mg': False,
        'has_mba': False,
        'mixture_order': 0,
        'family': sample_role,
    }


def _classify_folder_role(folder_name: str) -> str:
    if '背景' in folder_name:
        return 'background'
    if '空白' in folder_name:
        return 'blank'
    return 'design'


def _list_spectrum_files(folder: Path) -> list[Path]:
    files = []
    for fp in sorted(folder.glob('*.csv')):
        if fp.stem in {'3333', 'SP_0'}:
            continue
        files.append(fp)
    return files


def _append_common_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df['sample_id'] = [f'S{i + 1:04d}' for i in range(len(df))]
    df['group_id'] = df['source_type'].astype(str) + ':' + df['folder_name'].astype(str)
    for src_col, dst_col in [
        ('c_thiram', 'c_thiram_molar'),
        ('c_mg', 'c_mg_molar'),
        ('c_mba', 'c_mba_molar'),
    ]:
        df[dst_col] = df[src_col].map(CONC_LEVEL_TO_MOLAR).astype(float)
    return df


def _scan_source_tree(data_dir: Path, source_type: str, matrix_type: str,
                      include_auxiliary: bool = False) -> pd.DataFrame:
    records = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue

        sample_role = _classify_folder_role(folder.name)
        if sample_role != 'design' and not include_auxiliary:
            continue

        parsed = parse_folder_name(folder.name) if sample_role == 'design' else _empty_design_result(sample_role)
        csv_files = _list_spectrum_files(folder)

        for fp in csv_files:
            records.append({
                'file_path': str(fp),
                'file_name': fp.name,
                'folder_name': folder.name,
                'source_type': source_type,
                'matrix_type': matrix_type,
                'sample_role': sample_role,
                'is_design_sample': sample_role == 'design',
                'is_blank': sample_role == 'blank',
                'is_background': sample_role == 'background',
                **parsed,
            })

    return _append_common_metadata(pd.DataFrame(records))


# ====================== Spectrum I/O ======================

def read_bwram_spectrum(filepath: str | Path):
    """Read a BWRam CSV file, return (wavenumber, intensity) arrays or (None, None)."""
    filepath = Path(filepath)
    header_line = None
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped.startswith('Pixel,Wavelength') or stripped.startswith('Pixel, Wavelength'):
                header_line = i
                break
    if header_line is None:
        return None, None

    df = pd.read_csv(filepath, skiprows=header_line, header=0,
                     skipinitialspace=True, on_bad_lines='skip')
    cols = df.columns.tolist()

    rs_col = int_col = None
    for c in cols:
        cl = c.strip().lower()
        if 'raman shift' in cl:
            rs_col = c
        if 'dark subtracted' in cl:
            int_col = c

    if rs_col is None or int_col is None:
        if len(cols) >= 8:
            rs_col, int_col = cols[3], cols[7]
        else:
            return None, None

    wn = pd.to_numeric(df[rs_col], errors='coerce').values
    intensity = pd.to_numeric(df[int_col], errors='coerce').values
    valid = ~np.isnan(wn) & ~np.isnan(intensity)
    return wn[valid], intensity[valid]


# ====================== Preprocessing Functions ======================

def baseline_als(y, lam=ALS_LAM, p=ALS_P, niter=ALS_NITER):
    L = len(y)
    D = diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = csc_matrix(W + D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def remove_cosmic_spikes(y, threshold=COSMIC_THRESHOLD, window=COSMIC_WINDOW):
    y_out = y.copy()
    dy = np.diff(y_out)
    med = np.median(dy)
    mad = np.median(np.abs(dy - med))
    if mad == 0:
        return y_out
    mz = 0.6745 * (dy - med) / mad
    spikes = np.where(np.abs(mz) > threshold)[0]
    for idx in spikes:
        lo = max(0, idx - window)
        hi = min(len(y_out) - 1, idx + window)
        nbrs = [n for n in range(lo, hi + 1) if n != idx and n not in spikes]
        if nbrs:
            y_out[idx] = np.mean(y_out[nbrs])
    return y_out


def snv(X):
    mu = np.mean(X, axis=1, keepdims=True)
    sd = np.std(X, axis=1, keepdims=True)
    sd[sd == 0] = 1
    return (X - mu) / sd


def vector_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def augment_spectra(X, y, n_aug=5, seed=SEED):
    """Online-style spectral augmentation for training set only.

    Augmentations (applied randomly per sample):
      1. Gaussian noise: std = 0.01 * max(|x|)
      2. Intensity scaling: uniform [0.9, 1.1]
      3. Wavenumber shift: roll ±2 points
      4. Baseline tilt: add small linear slope

    Returns augmented (X_aug, y_aug) appended to originals.
    """
    rng = np.random.RandomState(seed)
    X_aug_list = [X]
    y_aug_list = [y]

    for _ in range(n_aug):
        X_new = X.copy()
        n = len(X_new)

        # 1. Gaussian noise
        noise_mask = rng.rand(n) < 0.8
        for i in np.where(noise_mask)[0]:
            scale = 0.01 * np.max(np.abs(X_new[i]))
            X_new[i] += rng.normal(0, scale, X_new[i].shape)

        # 2. Intensity scaling
        scale_mask = rng.rand(n) < 0.5
        scales = rng.uniform(0.9, 1.1, n)
        X_new[scale_mask] *= scales[scale_mask, None]

        # 3. Wavenumber shift (roll)
        shift_mask = rng.rand(n) < 0.5
        for i in np.where(shift_mask)[0]:
            shift = rng.randint(-2, 3)
            X_new[i] = np.roll(X_new[i], shift)

        # 4. Small baseline tilt
        tilt_mask = rng.rand(n) < 0.3
        for i in np.where(tilt_mask)[0]:
            slope = rng.uniform(-0.001, 0.001)
            X_new[i] += slope * np.arange(X_new[i].shape[0])

        X_aug_list.append(X_new)
        y_aug_list.append(y.copy())

    return np.concatenate(X_aug_list, axis=0), np.concatenate(y_aug_list, axis=0)


def spectral_mixup(
    X,
    y,
    n_mix=None,
    alpha=0.3,
    seed=SEED,
    composition_keys=None,
):
    """Composition-constrained spectral mixup.

    Linear spectral mixing is only used between spectra that share the same
    full composition key and the same task label. This keeps the augmented
    label unambiguous and avoids mixing different co-analyte contexts.

    Args:
        X: (n, d) spectra
        y: (n,) integer labels
        n_mix: number of mixup samples to generate (default: len(X))
        alpha: Beta distribution parameter (smaller = closer to originals)
        seed: random state
        composition_keys: (n,) full-composition identifiers for same-composition pairing

    Returns:
        X_mixed, y_mixed (only the new samples, NOT including originals)
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    if n_mix is None:
        n_mix = n

    if composition_keys is None:
        raise ValueError("composition_keys is required for composition-constrained spectral_mixup.")
    composition_keys = np.asarray(composition_keys)
    if len(composition_keys) != n:
        raise ValueError("composition_keys must have the same length as X and y.")

    X_list, y_list = [], []
    pair_keys = np.array([f"{composition_keys[i]}||{y[i]}" for i in range(n)], dtype=object)
    unique_keys = np.unique(pair_keys)
    key_idx = {key: np.where(pair_keys == key)[0] for key in unique_keys}
    eligible_keys = [key for key, idx in key_idx.items() if len(idx) >= 2]
    if not eligible_keys:
        return np.empty((0, X.shape[1])), np.empty(0, dtype=y.dtype)

    generated = 0
    max_attempts = n_mix * 3
    attempts = 0
    while generated < n_mix and attempts < max_attempts:
        attempts += 1
        key = eligible_keys[rng.randint(len(eligible_keys))]
        idx_c = key_idx[key]
        i, j = rng.choice(idx_c, size=2, replace=False)
        lam = rng.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # ensure lam >= 0.5
        x_new = lam * X[i] + (1 - lam) * X[j]
        X_list.append(x_new)
        y_list.append(y[i])
        generated += 1

    if X_list:
        return np.array(X_list), np.array(y_list)
    return np.empty((0, X.shape[1])), np.empty(0, dtype=y.dtype)


def spectral_mixup_mt(X, y_dict, n_mix=None, alpha=0.3, seed=SEED, composition_keys=None):
    """Multi-task composition-constrained mixup."""
    rng = np.random.RandomState(seed)
    n = len(X)
    if n_mix is None:
        n_mix = n

    tids = sorted(y_dict.keys())
    if composition_keys is None:
        # Compatibility fallback for old MT callers. Active code should pass
        # folder/composition keys so the constraint is independent of task subset.
        keys = np.array([tuple(y_dict[t][i] for t in tids) for i in range(n)])
    else:
        composition_keys = np.asarray(composition_keys)
        if len(composition_keys) != n:
            raise ValueError("composition_keys must have the same length as X.")
        keys = np.array([(composition_keys[i], *(y_dict[t][i] for t in tids)) for i in range(n)])
    unique_keys = list(set(map(tuple, keys)))

    key_to_idx = {}
    for uk in unique_keys:
        mask = np.all(keys == np.array(uk), axis=1)
        key_to_idx[uk] = np.where(mask)[0]

    X_list, y_lists = [], {t: [] for t in tids}
    generated = 0
    max_attempts = n_mix * 3
    attempts = 0

    while generated < n_mix and attempts < max_attempts:
        attempts += 1
        uk = unique_keys[rng.randint(len(unique_keys))]
        indices = key_to_idx[uk]
        if len(indices) < 2:
            continue
        i, j = rng.choice(indices, size=2, replace=False)
        lam = rng.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
        X_list.append(lam * X[i] + (1 - lam) * X[j])
        for t in tids:
            y_lists[t].append(y_dict[t][i])
        generated += 1

    if X_list:
        X_mix = np.array(X_list)
        y_mix = {t: np.array(v) for t, v in y_lists.items()}
    else:
        X_mix = np.empty((0, X.shape[1]))
        y_mix = {t: np.empty(0, dtype=y_dict[t].dtype) for t in tids}

    return X_mix, y_mix


def augment_spectra_mt(X, y_dict, n_aug=5, seed=SEED):
    """Multi-task version: augments X and replicates all task labels in sync."""
    rng = np.random.RandomState(seed)
    # Augment X using the single-task function with a dummy y
    dummy_y = np.zeros(len(X), dtype=int)
    X_aug, _ = augment_spectra(X, dummy_y, n_aug=n_aug, seed=seed)

    # Replicate each task's labels (n_aug+1) times
    y_dict_aug = {}
    for tid, y in y_dict.items():
        y_dict_aug[tid] = np.tile(y, n_aug + 1)

    return X_aug, y_dict_aug


# ====================== Build Metadata ======================

def build_pure_metadata(include_auxiliary: bool = False) -> pd.DataFrame:
    """Build metadata for the main pure benchmark branch."""
    return _scan_source_tree(
        PURE_DATA_DIR,
        source_type='pure',
        matrix_type='pure',
        include_auxiliary=include_auxiliary,
    )


def build_soil_metadata(include_auxiliary: bool = False) -> pd.DataFrame:
    """Build metadata for the soil external-validation branch."""
    return _scan_source_tree(
        SOIL_DATA_DIR,
        source_type='soil',
        matrix_type='soil',
        include_auxiliary=include_auxiliary,
    )


def build_metadata(data_dir: Path = DATA_DIR, include_auxiliary: bool = False) -> pd.DataFrame:
    """Build metadata from the active source tree.

    Default behavior is pure-only mainline metadata. Soil must be requested
    explicitly so it does not leak into the benchmark pipeline.
    """
    resolved = Path(data_dir)
    if resolved == PURE_DATA_DIR:
        return build_pure_metadata(include_auxiliary=include_auxiliary)
    if resolved == SOIL_DATA_DIR:
        return build_soil_metadata(include_auxiliary=include_auxiliary)
    return _scan_source_tree(
        resolved,
        source_type='custom',
        matrix_type=resolved.name,
        include_auxiliary=include_auxiliary,
    )


# ====================== Preprocessing Pipeline ======================

def preprocess_cache_files(variants: list[str] | tuple[str, ...] | None = None) -> dict[str, Path]:
    """Return cache paths for the requested preprocessing variants."""
    tags = tuple(variants or PREPROCESS_TAGS)
    unknown = [tag for tag in tags if tag not in PREPROCESS_CACHE_FILES]
    if unknown:
        raise KeyError(f"Unknown preprocessing variants: {unknown}")
    return {tag: PROCESSED_DIR / PREPROCESS_CACHE_FILES[tag] for tag in tags}


def load_preprocessed_variants(
    variants: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load cached wavenumber and preprocessing arrays as a registry-keyed dict."""
    tags = tuple(variants or PREPROCESS_TAGS)
    wn_path = PROCESSED_DIR / 'wavenumber.npy'
    if not wn_path.exists():
        raise FileNotFoundError(f"Missing wavenumber cache: {wn_path}")

    X_by_variant = {}
    for tag, path in preprocess_cache_files(tags).items():
        if not path.exists():
            raise FileNotFoundError(f"Missing preprocessing cache for {tag}: {path}")
        X_by_variant[tag] = np.load(path)
    return np.load(wn_path), X_by_variant


def _compute_preprocess_variant(tag: str, X_raw: np.ndarray) -> np.ndarray:
    """Compute one registered preprocessing variant from interpolated raw spectra."""
    n = len(X_raw)
    if tag == 'raw':
        return X_raw

    if tag == 'p1':
        X = np.zeros_like(X_raw)
        for i in range(n):
            y = remove_cosmic_spikes(X_raw[i])
            y = savgol_filter(y, SG_WINDOW, SG_POLY)
            bl = baseline_als(y)
            X[i] = y - bl
        return snv(X)

    if tag == 'p2':
        X = np.zeros_like(X_raw)
        for i in range(n):
            y = savgol_filter(X_raw[i], SG_WINDOW, SG_POLY)
            bl = baseline_als(y)
            y = y - bl
            X[i] = savgol_filter(y, SG_WINDOW, SG_POLY, deriv=1)
        return snv(X)

    if tag == 'p3':
        X = np.zeros_like(X_raw)
        for i in range(n):
            bl = baseline_als(X_raw[i])
            X[i] = X_raw[i] - bl
        return vector_normalize(X)

    if tag == 'p4':
        X = np.zeros_like(X_raw)
        for i in range(n):
            y = remove_cosmic_spikes(X_raw[i])
            y = savgol_filter(y, SG_WINDOW, SG_POLY)
            bl = baseline_als(y)
            X[i] = y - bl
        return X

    if tag == 'p5':
        X = np.zeros_like(X_raw)
        for i in range(n):
            y = remove_cosmic_spikes(X_raw[i])
            y = savgol_filter(y, SG_WINDOW, SG_POLY)
            bl = baseline_als(y)
            y = y - bl
            X[i] = savgol_filter(y, SG_WINDOW, SG_POLY, deriv=2, delta=WN_STEP)
        return snv(X)

    raise KeyError(f"Unknown preprocessing variant: {tag}")


def _format_preprocess_return(
    wn: np.ndarray,
    X_by_variant: dict[str, np.ndarray],
    return_dict: bool,
):
    if return_dict:
        return wn, X_by_variant
    return (wn, *(X_by_variant[tag] for tag in PREPROCESS_TAGS))


def load_and_preprocess(
    meta: pd.DataFrame,
    rebuild: bool = False,
    variants: list[str] | tuple[str, ...] | None = None,
    return_dict: bool = False,
):
    """Load spectra, interpolate to common axis, and apply registered preprocessing.

    By default this preserves the historical tuple return for old callers.
    New code should pass ``return_dict=True`` and consume the registry-keyed dict.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tags = tuple(variants or PREPROCESS_TAGS)

    cache_files = {
        'wn': PROCESSED_DIR / 'wavenumber.npy',
        **preprocess_cache_files(tags),
    }

    if not rebuild and all(f.exists() for f in cache_files.values()):
        print("[Dataset] Loading cached preprocessed data...")
        wn, X_by_variant = load_preprocessed_variants(tags)
        return _format_preprocess_return(wn, X_by_variant, return_dict=return_dict)

    raw_cache = PROCESSED_DIR / PREPROCESS_CACHE_FILES['raw']
    if not rebuild and cache_files['wn'].exists() and raw_cache.exists():
        print("[Dataset] Computing missing preprocessing variants from cached raw spectra...")
        wn = np.load(cache_files['wn'])
        X_raw = np.load(raw_cache)
        missing = [tag for tag in tags if not cache_files[tag].exists()]
        for tag in missing:
            print(f"  {tag}: {PREPROCESS_NAMES[tag]}")
            X_variant = _compute_preprocess_variant(tag, X_raw)
            np.save(cache_files[tag], X_variant)
        wn, X_by_variant = load_preprocessed_variants(tags)
        return _format_preprocess_return(wn, X_by_variant, return_dict=return_dict)

    print("[Dataset] Reading and preprocessing spectra from scratch...")
    wn_common = np.arange(WN_MIN, WN_MAX + WN_STEP, WN_STEP)
    n_pts = len(wn_common)
    n = len(meta)
    X_raw = np.zeros((n, n_pts))
    errors = []

    for i, (_, row) in enumerate(meta.iterrows()):
        wn, intensity = read_bwram_spectrum(row['file_path'])
        if wn is None or len(wn) < 50:
            errors.append((i, row['file_path']))
            continue
        # Sort by wavenumber
        order = np.argsort(wn)
        wn, intensity = wn[order], intensity[order]
        # Clip to valid range
        mask = (wn >= WN_MIN - 20) & (wn <= WN_MAX + 20)
        wn, intensity = wn[mask], intensity[mask]
        if len(wn) < 50:
            errors.append((i, row['file_path']))
            continue
        # Interpolate to common axis
        f_interp = interp1d(wn, intensity, kind='linear', bounds_error=False, fill_value=0)
        X_raw[i] = f_interp(wn_common)

    if errors:
        print(f"  WARNING: {len(errors)} files could not be read/interpolated.")

    X_by_variant = {}
    for tag in tags:
        print(f"  {tag}: {PREPROCESS_NAMES[tag]}")
        X_by_variant[tag] = _compute_preprocess_variant(tag, X_raw)

    # Save caches
    np.save(cache_files['wn'], wn_common)
    for tag, X_variant in X_by_variant.items():
        np.save(cache_files[tag], X_variant)
    print(f"  Saved: {n} spectra × {n_pts} points, {len(tags)} preprocessing versions")
    return _format_preprocess_return(wn_common, X_by_variant, return_dict=return_dict)


# ====================== Splitting ======================

def create_cv_splits(meta: pd.DataFrame, rebuild=False, split_filename: str | None = None) -> pd.DataFrame:
    """5-fold StratifiedGroupKFold, group=folder_name, stratify=family.

    Returns meta with 'fold_id' column appended.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_name = split_filename or ACTIVE_SPLIT_FILE
    split_path = SPLITS_DIR / split_name

    if not rebuild and split_path.exists():
        print(f"[Split] Loading cached split: {split_path.name}")
        return pd.read_csv(split_path)

    if 'is_design_sample' in meta.columns:
        meta = meta[meta['is_design_sample'].astype(bool)].reset_index(drop=True)

    print(f"[Split] Creating {N_FOLDS}-fold StratifiedGroupKFold -> {split_path.name}")
    group_col = 'group_id' if 'group_id' in meta.columns else 'folder_name'
    groups = meta[group_col].values
    # Stratify by 7-class presence pattern (finer than 5-class family)
    y_strat = ('T' + meta['has_thiram'].astype(int).astype(str)
               + '_M' + meta['has_mg'].astype(int).astype(str)
               + '_B' + meta['has_mba'].astype(int).astype(str)).values

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_ids = np.zeros(len(meta), dtype=int)
    for fold, (_, val_idx) in enumerate(sgkf.split(meta, y_strat, groups)):
        fold_ids[val_idx] = fold

    meta_split = meta.copy()
    meta_split['fold_id'] = fold_ids
    meta_split.to_csv(split_path, index=False, encoding='utf-8-sig')

    # Print split summary
    for fold in range(N_FOLDS):
        val = meta_split[meta_split['fold_id'] == fold]
        print(f"  Fold {fold}: {val['folder_name'].nunique()} folders, {len(val)} spectra")

    return meta_split


def make_joint_stratify_key(
    meta: pd.DataFrame,
    cols: tuple[str, ...] = RANDOM_STRATIFY_COLS,
) -> pd.Series:
    """Build the six-label composition key used for random-CV stratification."""
    key_df = meta.loc[:, list(cols)].copy()
    for col in cols:
        if col.startswith('has_'):
            key_df[col] = key_df[col].astype(str).str.strip().str.lower().map(
                {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
            ).fillna(key_df[col]).astype(int)
        else:
            key_df[col] = key_df[col].astype(int)
    return key_df.astype(str).agg('|'.join, axis=1)


def _write_random_split_qc(meta_split: pd.DataFrame) -> None:
    """Write fold-level and joint-stratum-level QC tables for random CV."""
    by_fold = []
    for fold, sub in meta_split.groupby('fold_id'):
        row = {
            'fold_id': int(fold),
            'n_spectra': int(len(sub)),
            'n_folders': int(sub['folder_name'].nunique()),
            'n_joint_strata': int(sub['random_stratify_key'].nunique()),
            'has_thiram_rate': float(sub['has_thiram'].astype(int).mean()),
            'has_mg_rate': float(sub['has_mg'].astype(int).mean()),
            'has_mba_rate': float(sub['has_mba'].astype(int).mean()),
        }
        for col in ['c_thiram', 'c_mg', 'c_mba', 'mixture_order']:
            counts = sub[col].value_counts().sort_index()
            for value, count in counts.items():
                row[f'{col}_{value}'] = int(count)
        by_fold.append(row)

    by_fold_df = pd.DataFrame(by_fold).sort_values('fold_id')
    by_fold_df.to_csv(SPLITS_DIR / RANDOM_SPLIT_QC_BY_FOLD_FILE, index=False, encoding='utf-8-sig')

    fold_counts = pd.crosstab(meta_split['random_stratify_key'], meta_split['fold_id'])
    for fold in range(N_FOLDS):
        if fold not in fold_counts.columns:
            fold_counts[fold] = 0
    fold_counts = fold_counts[[fold for fold in range(N_FOLDS)]]
    fold_counts.columns = [f'fold_{fold}' for fold in range(N_FOLDS)]
    fold_counts['total'] = fold_counts.sum(axis=1)
    fold_cols = [f'fold_{fold}' for fold in range(N_FOLDS)]
    fold_counts['folds_present'] = (fold_counts[fold_cols] > 0).sum(axis=1)
    fold_counts['min_fold_count'] = fold_counts[fold_cols].min(axis=1)
    fold_counts['max_fold_count'] = fold_counts[fold_cols].max(axis=1)

    key_parts = fold_counts.index.to_series().str.split('|', expand=True)
    key_parts.columns = list(RANDOM_STRATIFY_COLS)
    by_stratum_df = pd.concat(
        [key_parts.reset_index(drop=True), fold_counts.reset_index(drop=True)],
        axis=1,
    )
    by_stratum_df.to_csv(
        SPLITS_DIR / RANDOM_SPLIT_QC_BY_STRATUM_FILE,
        index=False,
        encoding='utf-8-sig',
    )


def create_random_stratified_splits(
    meta: pd.DataFrame,
    rebuild: bool = False,
    split_filename: str = RANDOM_SPLIT_FILE,
) -> pd.DataFrame:
    """Create a spectrum-level random 5-fold split using a six-label joint key.

    The pure63 dataset has two rare exact-composition strata with fewer than
    five spectra, so strict sklearn StratifiedKFold is mathematically
    impossible. This routine assigns spectra round-robin within each joint key,
    balancing fold size and preserving the full composition key as closely as
    the available replicate count permits. The QC files record those rare strata.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_path = SPLITS_DIR / split_filename

    if not rebuild and split_path.exists():
        print(f"[Split] Loading cached random split: {split_path.name}")
        return pd.read_csv(split_path)

    if 'is_design_sample' in meta.columns:
        meta = meta[meta['is_design_sample'].astype(bool)].reset_index(drop=True)

    print(f"[Split] Creating random {N_FOLDS}-fold split -> {split_path.name}")
    rng = np.random.RandomState(SEED)
    meta_split = meta.copy()
    meta_split['random_stratify_key'] = make_joint_stratify_key(meta_split)

    fold_ids = np.full(len(meta_split), -1, dtype=int)
    fold_counts = np.zeros(N_FOLDS, dtype=int)
    grouped_indices = [
        (key, np.array(indices, dtype=int))
        for key, indices in meta_split.groupby('random_stratify_key').groups.items()
    ]
    grouped_indices.sort(key=lambda item: (-len(item[1]), item[0]))

    for _, indices in grouped_indices:
        indices = indices.copy()
        rng.shuffle(indices)
        local_counts = np.zeros(N_FOLDS, dtype=int)
        for idx in indices:
            fold_order = np.lexsort((np.arange(N_FOLDS), fold_counts, local_counts))
            fold = int(fold_order[0])
            fold_ids[idx] = fold
            local_counts[fold] += 1
            fold_counts[fold] += 1

    if np.any(fold_ids < 0):
        raise RuntimeError("Random split assignment failed for some spectra.")

    meta_split['fold_id'] = fold_ids
    meta_split['split_mode'] = 'random_spectrum_level'
    meta_split.to_csv(split_path, index=False, encoding='utf-8-sig')
    _write_random_split_qc(meta_split)

    for fold in range(N_FOLDS):
        val = meta_split[meta_split['fold_id'] == fold]
        print(
            f"  Fold {fold}: {len(val)} spectra, "
            f"{val['folder_name'].nunique()} folders, "
            f"{val['random_stratify_key'].nunique()} joint strata"
        )

    rare = (
        meta_split['random_stratify_key']
        .value_counts()
        .loc[lambda s: s < N_FOLDS]
    )
    if not rare.empty:
        print(f"  WARNING: {len(rare)} joint strata have fewer than {N_FOLDS} spectra; see QC table.")

    return meta_split
