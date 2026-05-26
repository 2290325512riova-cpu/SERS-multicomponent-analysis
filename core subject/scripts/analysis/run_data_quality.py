#!/usr/bin/env python3
"""Generate paper-main data-quality tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import PAPER_MAIN_DATA_QUALITY_DIR, PREPROCESS_TAGS  # noqa: E402
from src.dataset import build_metadata, load_preprocessed_variants  # noqa: E402

PEAKS = {
    'Thiram_1382': 1382,
    'MG_1616': 1616,
    'MBA_1080': 1080,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate RSD data-quality tables.')
    parser.add_argument('--variants', nargs='+', default=['raw', 'p1', 'p3', 'p5'])
    parser.add_argument('--window', type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [variant for variant in args.variants if variant not in PREPROCESS_TAGS]
    if unknown:
        raise KeyError(f'Unknown variants: {unknown}')

    PAPER_MAIN_DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    meta = build_metadata()
    wn, X_by_variant = load_preprocessed_variants(args.variants)
    rows = []
    for variant, X in X_by_variant.items():
        for folder, sub in meta.groupby('folder_name'):
            idx = sub.index.values
            if len(idx) < 2:
                continue
            for peak_name, peak_wn in PEAKS.items():
                center = int(np.argmin(np.abs(wn - peak_wn)))
                lo = max(0, center - args.window)
                hi = min(X.shape[1], center + args.window + 1)
                values = np.max(X[idx, lo:hi], axis=1)
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                rsd = float(std / abs(mean) * 100.0) if abs(mean) > 1e-12 else np.nan
                rows.append({
                    'variant': variant,
                    'folder_name': folder,
                    'peak': peak_name,
                    'peak_wn': peak_wn,
                    'n_spectra': int(len(idx)),
                    'mean_peak': mean,
                    'std_peak': std,
                    'rsd_percent': rsd,
                })

    by_folder = pd.DataFrame(rows)
    by_folder.to_csv(PAPER_MAIN_DATA_QUALITY_DIR / 'peak_rsd_by_folder.csv', index=False, encoding='utf-8-sig')
    summary = (
        by_folder.groupby(['variant', 'peak'])['rsd_percent']
        .agg(['count', 'median', 'mean', 'max'])
        .reset_index()
    )
    summary.to_csv(PAPER_MAIN_DATA_QUALITY_DIR / 'peak_rsd_summary.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
    print(f'Output: {PAPER_MAIN_DATA_QUALITY_DIR}')


if __name__ == '__main__':
    main()
