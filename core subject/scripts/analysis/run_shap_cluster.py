#!/usr/bin/env python3
"""Cluster existing SHAP top features into peak windows."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import MAINLINE_SHAP_DIR, PAPER_MAIN_SHAP_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build peak-window SHAP cluster table from top features.')
    parser.add_argument('--input', type=Path, default=MAINLINE_SHAP_DIR / 'shap_top20_features.csv')
    parser.add_argument('--window', type=float, default=15.0)
    return parser.parse_args()


def cluster_task(df: pd.DataFrame, window: float) -> list[dict]:
    rows = []
    df = df.sort_values('Wavenumber').reset_index(drop=True)
    current = []
    for _, row in df.iterrows():
        if not current or abs(float(row['Wavenumber']) - float(current[-1]['Wavenumber'])) <= window:
            current.append(row)
        else:
            rows.append(summarize_cluster(current))
            current = [row]
    if current:
        rows.append(summarize_cluster(current))
    return rows


def summarize_cluster(cluster_rows: list[pd.Series]) -> dict:
    frame = pd.DataFrame(cluster_rows)
    weights = frame['Mean_SHAP'].abs().values
    if np.all(weights == 0):
        center = float(frame['Wavenumber'].mean())
    else:
        center = float(np.average(frame['Wavenumber'], weights=weights))
    assignments = sorted(set(str(v) for v in frame.get('Assignment', []) if str(v) not in {'', 'nan'}))
    substances = sorted(set(str(v) for v in frame.get('Substance', []) if str(v) not in {'', 'nan'}))
    matches = frame.get('Match', pd.Series(dtype=object)).astype(str)
    return {
        'Task': frame['Task'].iloc[0],
        'cluster_min_wn': float(frame['Wavenumber'].min()),
        'cluster_max_wn': float(frame['Wavenumber'].max()),
        'cluster_center_wn': center,
        'n_features': int(len(frame)),
        'rank_min': int(frame['Rank'].min()),
        'mean_abs_shap_sum': float(frame['Mean_SHAP'].abs().sum()),
        'assignments': '; '.join(assignments),
        'substances': '; '.join(substances),
        'has_peak_match': bool(matches.str.contains('✓|鉁', regex=True).any() or len(assignments) > 0),
    }


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    PAPER_MAIN_SHAP_DIR.mkdir(parents=True, exist_ok=True)
    top = pd.read_csv(args.input)
    clusters = []
    for _, task_df in top.groupby('Task'):
        clusters.extend(cluster_task(task_df, window=args.window))
    out = pd.DataFrame(clusters).sort_values(['Task', 'rank_min'])
    out_path = PAPER_MAIN_SHAP_DIR / 'shap_peak_clusters.csv'
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    summary = out.groupby('Task').agg(
        n_clusters=('n_features', 'count'),
        n_matched_clusters=('has_peak_match', 'sum'),
    ).reset_index()
    summary['cluster_match_rate'] = summary['n_matched_clusters'] / summary['n_clusters']
    summary.to_csv(PAPER_MAIN_SHAP_DIR / 'shap_peak_cluster_summary.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
    print(f'Output: {PAPER_MAIN_SHAP_DIR}')


if __name__ == '__main__':
    main()
