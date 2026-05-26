"""
Benchmark SHAP Analysis — Presence + Grade cross-task comparison.

Generates:
  Fig.9:  SHAP beeswarm for P1, P2, P3, G2 (4 subplots)
  Fig.10: Mean |SHAP| vs wavenumber overlaid with literature peaks
  Fig.11b: P2 vs G2 cross-task SHAP comparison (competitive adsorption evidence)
  Tab.7:  Top-20 SHAP features per task + chemical assignment

Model: RF with 02-locked config (TreeExplainer, exact & fast)
Key hypothesis: If G2's top features include MBA peaks (1080, 1590 cm⁻¹),
  the model implicitly learned the competitive adsorption mechanism.

References:
  - Lundberg & Lee 2017, NeurIPS
  - Lundberg et al. 2020, Nat Machine Intelligence
  - Peak assignments: Linh 2024; Oliveira 2021; Qin 2021; Yang 2020; Mhlanga 2021
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import (
    TASKS, FIGURES_DIR, PROCESSED_DIR, FIG_MODELS, MAINLINE_SHAP_DIR,
    N_FOLDS, SEED, WN_MIN, WN_MAX, SPLITS_DIR, ACTIVE_SPLIT_FILE,
)
from src.dataset import build_metadata, create_cv_splits
from src.result_index import get_active_grouped_result_dir

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Literature-backed peak ranges (cm⁻¹) ──
KNOWN_PEAKS = [
    (556,  20, 'ν(S–S)+ν(C–S–S)', 'Thiram'),
    (928,  20, 'ν(C–S)', 'Thiram'),
    (1150, 20, 'ν(N–C)+ρ(CH₃)', 'Thiram'),
    (1382, 20, 'ν(N–C)+δ(CH₃)', 'Thiram'),
    (1510, 20, 'ν(C=N)', 'Thiram'),
    (908,  20, 'Ring(o.p.)', 'MG'),
    (1170, 20, 'δ(C–H)i.p.', 'MG'),
    (1394, 20, 'δ(C–H)i.p.', 'MG'),
    (1616, 20, 'ν(C=C)ring', 'MG'),
    (1080, 20, 'Ring+ν(C–S)', 'MBA'),
    (1180, 20, 'δ(C–H)i.p.', 'MBA'),
    (1490, 20, 'ν(COO⁻)+ν(C=C)', 'MBA'),
    (1590, 20, 'ν(C=C)ring', 'MBA'),
]

SUBST_COLORS = {'Thiram': '#E53935', 'MG': '#43A047', 'MBA': '#1E88E5'}

# RF config from 02 NSGA-II locked results
RF_LOCKED_PARAMS = dict(
    n_estimators=800, max_depth=28, min_samples_leaf=6,
    max_features=0.25, class_weight='balanced_subsample',
    random_state=SEED, n_jobs=-1,
)

SHAP_TASKS = [
    {'id': 'P1_thiram_presence', 'name': 'Thiram Presence',
     'col': 'has_thiram', 'substance': 'Thiram', 'filter_col': None},
    {'id': 'P2_mg_presence', 'name': 'MG Presence',
     'col': 'has_mg', 'substance': 'MG', 'filter_col': None},
    {'id': 'P3_mba_presence', 'name': 'MBA Presence',
     'col': 'has_mba', 'substance': 'MBA', 'filter_col': None},
    {'id': 'G2_mg_molar_grade', 'name': 'MG Molar Grade',
     'col': 'c_mg', 'substance': 'MG', 'filter_col': 'has_mg'},
]


def find_best_preprocess(task_id, model_name='RF'):
    result_dir = get_active_grouped_result_dir()
    full_matrix = result_dir / 'benchmark_full_matrix.csv' if result_dir else None
    if full_matrix is not None and full_matrix.exists():
        df = pd.read_csv(full_matrix)
        rows = df[(df['Task'] == task_id) & (df['Model'] == model_name)]
        if not rows.empty:
            best = rows.loc[rows['F1_mean'].idxmax()]
            return best['Preprocess'], float(best['F1_mean'])
    return 'p4', -1.0


def assign_chemical_peak(wn_val):
    for center, half_w, label, subst in KNOWN_PEAKS:
        if abs(wn_val - center) <= half_w:
            return label, subst
    return None, None


def compute_shap_cv(X, y, fold_ids, sample_mask=None):
    """Train RF per fold and collect SHAP values across all CV folds."""
    shap_list, x_list, idx_list = [], [], []
    for fold in range(N_FOLDS):
        if sample_mask is not None:
            tri = np.where((fold_ids != fold) & sample_mask)[0]
            vai = np.where((fold_ids == fold) & sample_mask)[0]
        else:
            tri = np.where(fold_ids != fold)[0]
            vai = np.where(fold_ids == fold)[0]
        if len(vai) == 0 or len(np.unique(y[tri])) < 2:
            continue
        rf = RandomForestClassifier(**RF_LOCKED_PARAMS)
        rf.fit(X[tri], y[tri])
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X[vai])
        # For binary: use class=1; for multiclass: average absolute across classes
        if isinstance(sv, list):
            sv = np.mean([np.abs(s) for s in sv], axis=0)
        elif sv.ndim == 3:
            if sv.shape[2] == 2:
                sv = sv[:, :, 1]
            else:
                sv = np.mean(np.abs(sv), axis=2)
        shap_list.append(sv)
        x_list.append(X[vai])
        idx_list.extend(vai.tolist())
    return np.vstack(shap_list), np.vstack(x_list), idx_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=['P1', 'P2', 'P3', 'G2'])
    args = parser.parse_args()

    task_filter = set(args.tasks)
    tasks_to_run = [t for t in SHAP_TASKS
                    if any(t['id'].startswith(f) for f in task_filter)]

    print("=" * 60)
    print("SHAP Analysis — RF (02-locked config)")
    print(f"Tasks: {[t['name'] for t in tasks_to_run]}")
    print("=" * 60)

    MAINLINE_SHAP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_MODELS.mkdir(parents=True, exist_ok=True)

    meta = build_metadata()
    meta = create_cv_splits(meta)
    wn = np.load(PROCESSED_DIR / 'wavenumber.npy')
    fold_ids = meta['fold_id'].values
    feature_names = [f'{wn[i]:.0f}' for i in range(len(wn))]

    all_shap = {}
    all_X = {}

    for tk in tasks_to_run:
        best_pp, best_f1 = find_best_preprocess(tk['id'], 'RF')
        print(f"\n[{tk['name']}] preprocess={best_pp}, F1={best_f1:.3f}")
        X = np.load(PROCESSED_DIR / f'X_{best_pp}.npy')
        y = meta[tk['col']].values.astype(int)
        mask = None
        if tk['filter_col']:
            mask = meta[tk['filter_col']].values.astype(int) == 1
        shap_vals, X_explain, _ = compute_shap_cv(X, y, fold_ids, mask)
        all_shap[tk['id']] = shap_vals
        all_X[tk['id']] = X_explain
        print(f"  SHAP shape: {shap_vals.shape}")

    # ── Fig.9: Beeswarm per task ──
    print("\n[Fig.9] SHAP beeswarm plots...")
    n_tasks = len(tasks_to_run)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(16, 5 * n_tasks))
    if n_tasks == 1:
        axes = [axes]
    for ax, tk in zip(axes, tasks_to_run):
        sv = all_shap[tk['id']]
        Xv = all_X[tk['id']]
        mean_abs = np.mean(np.abs(sv), axis=0)
        top_idx = np.argsort(mean_abs)[-20:][::-1]
        plt.sca(ax)
        shap.summary_plot(
            sv[:, top_idx], Xv[:, top_idx],
            feature_names=[feature_names[i] for i in top_idx],
            show=False, plot_size=None, max_display=20,
        )
        ax.set_title(f'{tk["name"]} – Top 20 SHAP (RF locked)', fontsize=11)
    plt.tight_layout()
    fig.savefig(FIG_MODELS / 'fig9_shap_beeswarm.png')
    plt.close()
    print(f"  Saved: {FIG_MODELS / 'fig9_shap_beeswarm.png'}")

    # ── Fig.10: Mean |SHAP| vs wavenumber with literature peaks ──
    print("\n[Fig.10] SHAP importance vs wavenumber...")
    fig, axes = plt.subplots(n_tasks, 1, figsize=(16, 4 * n_tasks), sharex=True)
    if n_tasks == 1:
        axes = [axes]
    for ax, tk in zip(axes, tasks_to_run):
        sv = all_shap[tk['id']]
        mean_abs = np.mean(np.abs(sv), axis=0)
        mean_abs_norm = mean_abs / mean_abs.max() if mean_abs.max() > 0 else mean_abs
        color = SUBST_COLORS[tk['substance']]
        ax.fill_between(wn, 0, mean_abs_norm, alpha=0.3, color=color)
        ax.plot(wn, mean_abs_norm, color=color, lw=1.2)
        for center, half_w, label, subst in KNOWN_PEAKS:
            if subst == tk['substance']:
                ax.axvline(x=center, color=SUBST_COLORS[subst], ls='--', alpha=0.7, lw=1.5)
                cidx = np.argmin(np.abs(wn - center))
                local_v = mean_abs_norm[max(0, cidx-10):min(len(wn), cidx+10)].max()
                ax.annotate(f'{center}\n{label}', xy=(center, local_v),
                            xytext=(center, min(1.0, local_v + 0.12)),
                            fontsize=7, ha='center', fontweight='bold',
                            color=SUBST_COLORS[subst],
                            arrowprops=dict(arrowstyle='->', color=SUBST_COLORS[subst]),
                            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                      ec=SUBST_COLORS[subst], alpha=0.85))
            else:
                ax.axvline(x=center, color='gray', ls=':', alpha=0.3, lw=0.5)
        ax.set_ylabel('Normalized |SHAP|')
        ax.set_title(f'{tk["name"]}', fontsize=11)
        ax.set_ylim(0, 1.15)
    axes[-1].set_xlabel('Raman Shift (cm⁻¹)')
    axes[-1].set_xlim(WN_MIN, WN_MAX)
    plt.tight_layout()
    fig.savefig(FIG_MODELS / 'fig10_shap_wavenumber.png')
    plt.close()
    print(f"  Saved: {FIG_MODELS / 'fig10_shap_wavenumber.png'}")

    # ── Fig.11b: P2 vs G2 cross-task comparison ──
    if 'P2_mg_presence' in all_shap and 'G2_mg_molar_grade' in all_shap:
        print("\n[Fig.11b] P2 vs G2 cross-task SHAP comparison...")
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        sv_p2 = np.mean(np.abs(all_shap['P2_mg_presence']), axis=0)
        sv_g2 = np.mean(np.abs(all_shap['G2_mg_molar_grade']), axis=0)
        sv_p2_n = sv_p2 / sv_p2.max() if sv_p2.max() > 0 else sv_p2
        sv_g2_n = sv_g2 / sv_g2.max() if sv_g2.max() > 0 else sv_g2
        ax.plot(wn, sv_p2_n, color='#43A047', lw=1.5, label='P2 (MG Presence)', alpha=0.8)
        ax.plot(wn, sv_g2_n, color='#FF6F00', lw=1.5, label='G2 (MG Grade)', alpha=0.8)
        # Highlight MBA peak regions
        for center, half_w, label, subst in KNOWN_PEAKS:
            if subst == 'MBA':
                ax.axvspan(center - half_w, center + half_w,
                           alpha=0.15, color='#1E88E5')
                ax.text(center, 1.05, f'MBA\n{center}', ha='center',
                        fontsize=8, color='#1E88E5', fontweight='bold')
        ax.set_xlabel('Raman Shift (cm⁻¹)')
        ax.set_ylabel('Normalized mean |SHAP|')
        ax.set_title('P2 vs G2: Cross-task SHAP — MBA peaks highlight competitive adsorption')
        ax.set_xlim(WN_MIN, WN_MAX)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        fig.savefig(FIG_MODELS / 'fig11b_shap_p2_vs_g2.png')
        plt.close()
        print(f"  Saved: {FIG_MODELS / 'fig11b_shap_p2_vs_g2.png'}")

    # ── Tab.7: Top-20 features with chemical assignments ──
    print("\n[Tab.7] Feature importance table...")
    rows = []
    for tk in tasks_to_run:
        sv = all_shap[tk['id']]
        mean_abs = np.mean(np.abs(sv), axis=0)
        top_idx = np.argsort(mean_abs)[-20:][::-1]
        for rank, idx in enumerate(top_idx, 1):
            wn_val = wn[idx]
            peak_label, peak_subst = assign_chemical_peak(wn_val)
            rows.append({
                'Task': tk['name'], 'Rank': rank,
                'Wavenumber': f'{wn_val:.0f}',
                'Mean_SHAP': f'{mean_abs[idx]:.4f}',
                'Assignment': peak_label or '',
                'Substance': peak_subst or '',
                'Match': '✓' if peak_subst == tk['substance'] else
                         ('other' if peak_subst else ''),
            })
    tab7 = pd.DataFrame(rows)
    tab7.to_csv(MAINLINE_SHAP_DIR / 'shap_top20_features.csv', index=False)
    print(f"  Saved: {MAINLINE_SHAP_DIR / 'shap_top20_features.csv'}")

    # Print cross-task summary
    print("\n=== Cross-task peak attribution summary ===")
    for tk in tasks_to_run:
        sub = tab7[tab7['Task'] == tk['name']]
        n_self = (sub['Match'] == '✓').sum()
        n_other = (sub['Match'] == 'other').sum()
        other_substs = sub[sub['Match'] == 'other']['Substance'].value_counts()
        print(f"  {tk['name']}: {n_self}/20 own peaks, {n_other}/20 other peaks")
        if not other_substs.empty:
            print(f"    Cross-substance: {dict(other_substs)}")

    print("\n" + "=" * 60)
    print("SHAP analysis complete!")


if __name__ == '__main__':
    main()
