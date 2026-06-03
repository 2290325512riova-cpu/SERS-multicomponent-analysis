#!/usr/bin/env python3
"""Prepare soil external-validation metadata and preprocessing caches."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    PAPER_MAIN_SOIL_DIR,
    PREPROCESS_TAGS,
    WN_MAX,
    WN_MIN,
    WN_STEP,
)
from src.dataset import (  # noqa: E402
    _compute_preprocess_variant,
    build_soil_metadata,
    read_bwram_spectrum,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare soil external-validation artifacts.')
    parser.add_argument('--variants', nargs='+', default=['raw', 'p1', 'p5'])
    parser.add_argument('--rebuild', action='store_true')
    return parser.parse_args()


def build_soil_raw(meta: pd.DataFrame, rebuild: bool = False) -> tuple[np.ndarray, np.ndarray]:
    PAPER_MAIN_SOIL_DIR.mkdir(parents=True, exist_ok=True)
    wn_path = PAPER_MAIN_SOIL_DIR / 'wavenumber.npy'
    raw_path = PAPER_MAIN_SOIL_DIR / 'X_soil_raw.npy'
    if not rebuild and wn_path.exists() and raw_path.exists():
        return np.load(wn_path), np.load(raw_path)

    wn_common = np.arange(WN_MIN, WN_MAX + WN_STEP, WN_STEP)
    X_raw = np.zeros((len(meta), len(wn_common)))
    unreadable = []
    for i, (_, row) in enumerate(meta.iterrows()):
        wn, intensity = read_bwram_spectrum(row['file_path'])
        if wn is None or len(wn) < 50:
            unreadable.append(row['file_path'])
            continue
        order = np.argsort(wn)
        wn, intensity = wn[order], intensity[order]
        mask = (wn >= WN_MIN - 20) & (wn <= WN_MAX + 20)
        wn, intensity = wn[mask], intensity[mask]
        if len(wn) < 50:
            unreadable.append(row['file_path'])
            continue
        X_raw[i] = interp1d(wn, intensity, kind='linear', bounds_error=False, fill_value=0)(wn_common)

    np.save(wn_path, wn_common)
    np.save(raw_path, X_raw)
    if unreadable:
        pd.Series(unreadable, name='file_path').to_csv(PAPER_MAIN_SOIL_DIR / 'unreadable_files.csv', index=False)
    return wn_common, X_raw


def main() -> None:
    args = parse_args()
    unknown = [variant for variant in args.variants if variant not in PREPROCESS_TAGS]
    if unknown:
        raise KeyError(f'Unknown variants: {unknown}')

    PAPER_MAIN_SOIL_DIR.mkdir(parents=True, exist_ok=True)
    meta = build_soil_metadata()
    meta.to_csv(PAPER_MAIN_SOIL_DIR / 'soil_metadata.csv', index=False, encoding='utf-8-sig')
    label_summary = meta.groupby(['folder_name', 'c_thiram', 'c_mg', 'c_mba']).size().reset_index(name='n_spectra')
    label_summary.to_csv(PAPER_MAIN_SOIL_DIR / 'soil_label_summary.csv', index=False, encoding='utf-8-sig')

    _, X_raw = build_soil_raw(meta, rebuild=args.rebuild)
    for variant in args.variants:
        out_path = PAPER_MAIN_SOIL_DIR / f'X_soil_{variant}.npy'
        if not args.rebuild and out_path.exists():
            continue
        X_variant = _compute_preprocess_variant(variant, X_raw)
        np.save(out_path, X_variant)

    print(f'Soil spectra: {len(meta)}')
    print(f'Folders: {meta["folder_name"].nunique()}')
    print(f'Output: {PAPER_MAIN_SOIL_DIR}')


if __name__ == '__main__':
    main()
