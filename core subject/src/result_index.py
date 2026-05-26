"""Helpers for mainline result directories and index files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    MAINLINE_FORMAL_DIR,
    MAINLINE_OPTIMIZATION_DIR,
    MAINLINE_SCREENING_DIR,
)

SUMMARY_PREFIX = 'cv_results_summary_'
SUMMARY_SUFFIX = '.csv'


def list_summary_files(result_dir: Path) -> list[Path]:
    return sorted(result_dir.glob(f'{SUMMARY_PREFIX}*{SUMMARY_SUFFIX}'))


def stage_has_results(result_dir: Path) -> bool:
    return any(result_dir.glob(f'{SUMMARY_PREFIX}*{SUMMARY_SUFFIX}')) or (result_dir / 'benchmark_full_matrix.csv').exists()


def list_variants(result_dir: Path) -> list[str]:
    variants = []
    for path in list_summary_files(result_dir):
        variants.append(path.stem.replace(SUMMARY_PREFIX, '', 1))
    return variants


def get_active_grouped_result_dir() -> Path | None:
    candidates = [
        MAINLINE_OPTIMIZATION_DIR,
        MAINLINE_SCREENING_DIR,
    ]
    for result_dir in candidates:
        if result_dir.exists() and stage_has_results(result_dir):
            return result_dir
    return None


def get_active_screening_dir() -> Path | None:
    if MAINLINE_SCREENING_DIR.exists() and stage_has_results(MAINLINE_SCREENING_DIR):
        return MAINLINE_SCREENING_DIR
    return get_active_grouped_result_dir()


def get_active_source_root() -> Path | None:
    if MAINLINE_FORMAL_DIR.exists():
        stage_dirs = [MAINLINE_SCREENING_DIR, MAINLINE_OPTIMIZATION_DIR]
        if any(d.exists() and stage_has_results(d) for d in stage_dirs):
            return MAINLINE_FORMAL_DIR
    return None


def build_result_matrix(
    result_dir: Path,
    source_stage: str | None = None,
    source_priority: int = 1,
) -> pd.DataFrame:
    source_name = source_stage or result_dir.name
    frames = []
    for summary_path in list_summary_files(result_dir):
        df = pd.read_csv(summary_path)
        variant = summary_path.stem.replace(SUMMARY_PREFIX, '', 1)
        df['source_stage'] = source_name
        df['source_priority'] = source_priority
        df['source_variant_file'] = summary_path.name
        df['variant_scope'] = variant
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    matrix = pd.concat(frames, ignore_index=True)
    sort_cols = ['Task', 'F1_mean', 'BA_mean', 'Acc_mean', 'Model', 'Preprocess']
    ascending = [True, False, False, False, True, True]
    return matrix.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def write_result_index(
    result_dir: Path,
    source_stage: str | None = None,
    description: str | None = None,
    models: list[str] | None = None,
    variants: list[str] | None = None,
    source_priority: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_dir.mkdir(parents=True, exist_ok=True)
    source_name = source_stage or result_dir.name
    matrix = build_result_matrix(result_dir, source_stage=source_name, source_priority=source_priority)
    if matrix.empty:
        raise FileNotFoundError(f'No summary files found under {result_dir}')
    matrix_path = result_dir / 'benchmark_full_matrix.csv'
    matrix.to_csv(matrix_path, index=False, encoding='utf-8-sig')
    best = (
        matrix.sort_values(['Task', 'F1_mean', 'source_priority'], ascending=[True, False, False])
        .groupby('Task', as_index=False).head(1).reset_index(drop=True)
    )
    best.to_csv(result_dir / 'benchmark_best_by_task.csv', index=False, encoding='utf-8-sig')
    metric_cols = [c for c in ['F1_mean', 'BA_mean', 'Acc_mean'] if c in matrix.columns]
    if metric_cols:
        model_mean = matrix.groupby(['source_stage', 'Model'])[metric_cols].mean().reset_index()
        model_mean.to_csv(result_dir / 'benchmark_source_model_mean.csv', index=False, encoding='utf-8-sig')
        preprocess_mean = matrix.groupby(['source_stage', 'Preprocess'])[metric_cols].mean().reset_index()
        preprocess_mean.to_csv(result_dir / 'benchmark_source_preprocess_mean.csv', index=False, encoding='utf-8-sig')
    manifest = pd.DataFrame([{
        'source_stage': source_name,
        'relative_dir': result_dir.name,
        'description': description or source_name,
        'models': '/'.join(models) if models else '/'.join(sorted(matrix['Model'].unique())),
        'variants': '/'.join(variants) if variants else '/'.join(list_variants(result_dir)),
    }])
    manifest.to_csv(result_dir / 'benchmark_source_manifest.csv', index=False, encoding='utf-8-sig')
    return matrix, best
