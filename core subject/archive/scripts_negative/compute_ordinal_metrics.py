"""
Compute ordinal classification metrics for grade tasks (G1/G2/G3).
Outputs: exact accuracy, ±1 level accuracy, MAE in levels, adjacent error rate.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import MAINLINE_SCREENING_DIR, MAINLINE_OPTIMIZATION_DIR, PREPROCESS_NAMES

GRADE_TASKS = ['G1_thiram_molar_grade', 'G2_mg_molar_grade', 'G3_mba_molar_grade']


def ordinal_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return {}
    exact_acc = np.mean(y_true == y_pred)
    within_1 = np.mean(np.abs(y_true - y_pred) <= 1)
    mae = np.mean(np.abs(y_true - y_pred))
    errors = y_true != y_pred
    if errors.sum() > 0:
        adjacent_errors = np.abs(y_true[errors] - y_pred[errors]) == 1
        adjacent_rate = adjacent_errors.mean()
    else:
        adjacent_rate = 0.0
    return {
        'n_samples': n,
        'exact_acc': exact_acc,
        'within_1_acc': within_1,
        'mae_levels': mae,
        'adjacent_error_rate': adjacent_rate,
    }


def load_all_predictions(stage_dir: Path) -> pd.DataFrame:
    """Load and concatenate all cv_predictions_*.csv from a stage directory."""
    frames = []
    for f in sorted(stage_dir.glob('cv_predictions_*.csv')):
        preprocess = f.stem.replace('cv_predictions_', '')
        df = pd.read_csv(f)
        df['preprocess'] = preprocess
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_stage_metrics(stage_dir: Path, stage_name: str) -> pd.DataFrame:
    """Compute ordinal metrics for all grade tasks in a stage."""
    df = load_all_predictions(stage_dir)
    if df.empty:
        print(f"  [SKIP] No predictions found in {stage_dir}")
        return pd.DataFrame()

    grade_df = df[df['task_id'].isin(GRADE_TASKS)]
    if grade_df.empty:
        print(f"  [SKIP] No grade task predictions in {stage_dir}")
        return pd.DataFrame()

    rows = []
    for (task_id, model, preprocess), grp in grade_df.groupby(
        ['task_id', 'model', 'preprocess']
    ):
        m = ordinal_metrics(grp['y_true'].values, grp['y_pred'].values)
        if m:
            rows.append({
                'stage': stage_name,
                'task_id': task_id,
                'model': model,
                'preprocess': preprocess,
                **m,
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("Ordinal Metrics for Grade Tasks (G1/G2/G3)")
    print("=" * 70)

    results = []

    # Stage 01: candidate screening
    print("\n[Stage 01] Candidate Screening")
    r1 = compute_stage_metrics(MAINLINE_SCREENING_DIR, '01_screening')
    if not r1.empty:
        results.append(r1)

    # Stage 02: NSGA-II best trials (RF)
    print("\n[Stage 02] NSGA-II Best Trials")
    trials_dir = MAINLINE_OPTIMIZATION_DIR / 'trials'
    if trials_dir.exists():
        for trial_dir in sorted(trials_dir.glob('rf_nsga_best_*')):
            r2 = compute_stage_metrics(trial_dir, f'02_{trial_dir.name}')
            if not r2.empty:
                results.append(r2)

    if not results:
        print("\nNo grade task predictions found. Exiting.")
        return

    all_results = pd.concat(results, ignore_index=True)

    # Summary table: best ±1 accuracy per task across all models/preprocess
    print("\n" + "=" * 70)
    print("SUMMARY: Best Ordinal Metrics per Task (Stage 01)")
    print("=" * 70)
    s01 = all_results[all_results['stage'] == '01_screening']
    if not s01.empty:
        for task_id in GRADE_TASKS:
            task_data = s01[s01['task_id'] == task_id]
            if task_data.empty:
                continue
            print(f"\n--- {task_id} ---")
            # Best by exact accuracy
            best_exact = task_data.loc[task_data['exact_acc'].idxmax()]
            print(f"  Best exact_acc:    {best_exact['exact_acc']:.3f} "
                  f"({best_exact['model']}, {best_exact['preprocess']})")
            # Best by within_1 accuracy
            best_w1 = task_data.loc[task_data['within_1_acc'].idxmax()]
            print(f"  Best within_1_acc: {best_w1['within_1_acc']:.3f} "
                  f"({best_w1['model']}, {best_w1['preprocess']})")
            # Best by MAE
            best_mae = task_data.loc[task_data['mae_levels'].idxmin()]
            print(f"  Best MAE_levels:   {best_mae['mae_levels']:.3f} "
                  f"({best_mae['model']}, {best_mae['preprocess']})")
            # Average across all models
            print(f"  Avg exact_acc:     {task_data['exact_acc'].mean():.3f}")
            print(f"  Avg within_1_acc:  {task_data['within_1_acc'].mean():.3f}")

    # Full detail table
    print("\n" + "=" * 70)
    print("DETAIL: All Model × Preprocess × Task (sorted by within_1_acc)")
    print("=" * 70)
    cols = ['stage', 'task_id', 'model', 'preprocess',
            'n_samples', 'exact_acc', 'within_1_acc', 'mae_levels',
            'adjacent_error_rate']
    detail = all_results[cols].sort_values(
        ['task_id', 'within_1_acc'], ascending=[True, False]
    )
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 140)
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(detail.to_string(index=False))

    # Save to CSV
    out_path = MAINLINE_SCREENING_DIR / 'ordinal_metrics_summary.csv'
    all_results.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
