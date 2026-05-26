"""
Benchmark summary tasks for paper-ready tables and figures
Generates all figures (→ figures/) and tables (→ figures/tables_*.xlsx)

Tasks:
  C20 – Formal result tables (Excel)
  C12 – Wilcoxon signed-rank tests between model families
  C19 – Model family boxplot (Fig.7)
  C18 – Confusion matrix montage (Fig.6)
  C17 – Peak-annotated mean spectra (Fig.1)
  E5  – Intra-folder RSD of peak intensities

Literature references for peak assignments:
  Thiram: Linh 2024 RSC Adv; Oliveira 2021 J Raman Spectrosc (59 cit.)
  MG:     Qin 2021 Spectroscopy; Yang 2020 Microchimica Acta (133 cit.)
  MBA:    Mhlanga 2021 Mat Today Comm (25 cit.); Marques 2022 PCCP (24 cit.)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix
from itertools import combinations

# ── project imports ──
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import (
    TASKS, TASKS_SUPPLEMENTARY, MODELS_DIR, FIGURES_DIR, PROCESSED_DIR,
    FIG_EDA, FIG_MODELS, N_FOLDS, WN_MIN, WN_MAX, WN_STEP,
    PREPROCESS_TAGS,
)
from src.dataset import build_metadata
from src.result_index import get_active_grouped_result_dir, get_active_screening_dir, get_active_source_root
from src.train_eval import load_predictions

warnings.filterwarnings('ignore', category=UserWarning)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# =====================================================================
# Shared Data Loading
# =====================================================================
def load_all_summaries():
    """Load the current benchmark summary.

    Prefer the active mainline_formal_experiment stage-local summary.
    If no mainline grouped results exist yet, fall back to the archived
    pilot benchmark bundle before trying legacy flat summary files.
    """
    result_dir = get_active_grouped_result_dir()
    if result_dir is not None:
        full_matrix = result_dir / 'benchmark_full_matrix.csv'
        if full_matrix.exists():
            return pd.read_csv(full_matrix)

        frames = []
        for path in sorted(result_dir.glob('cv_results_summary_*.csv')):
            frames.append(pd.read_csv(path))
        if frames:
            return pd.concat(frames, ignore_index=True)

    frames = []
    for tag in PREPROCESS_TAGS:
        p = MODELS_DIR / f'cv_results_summary_{tag}.csv'
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True)


def load_all_details():
    """Load per-fold detail rows from the active grouped evidence layer.

    Family-level tests should use the broad screening layer to avoid
    overweighting models repeated in later optimization or historical review layers.
    """
    official_dir = get_active_screening_dir()
    if official_dir is not None and official_dir.exists():
        frames = []
        for tag in PREPROCESS_TAGS:
            p = official_dir / f'cv_results_detail_{tag}.csv'
            if p.exists():
                df = pd.read_csv(p)
                df['source_stage'] = official_dir.name
                frames.append(df)
        if frames:
            return pd.concat(frames, ignore_index=True)

    frames = []
    for tag in PREPROCESS_TAGS:
        p = MODELS_DIR / f'cv_results_detail_{tag}.csv'
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True)


def load_all_predictions():
    """Load predictions from benchmark source layers.

    Keys include source_stage so confusion matrices match the exact evidence
    row selected from benchmark_full_matrix.csv.
    """
    all_preds = {}
    source_root = get_active_source_root()
    source_dirs = []
    if source_root is not None and source_root.exists():
        source_dirs = sorted(
            path for path in source_root.iterdir()
            if path.is_dir() and any(path.glob('cv_predictions_*.csv'))
        )
    if source_dirs:
        for source_dir in source_dirs:
            source_stage = source_dir.name
            for tag in PREPROCESS_TAGS:
                preds = load_predictions(tag, output_dir=source_dir)
                for k, v in preds.items():
                    all_preds[(k[0], k[1], tag, source_stage)] = v
        return all_preds

    for tag in PREPROCESS_TAGS:
        preds = load_predictions(tag)
        for k, v in preds.items():
            all_preds[(k[0], k[1], tag)] = v
    return all_preds


# =====================================================================
# C20: Formal Result Tables → Excel
# =====================================================================
def task_c20(summary_df):
    """
    Generate paper-ready result tables as Excel.
    Tab.1: Best model+preprocessing per task (overview)
    Tab.2: Presence tasks – all models at best preprocessing per model
    Tab.3: Concentration tasks – all models at best preprocessing per model
    Tab.4: Preprocessing comparison for best model per task
    """
    print("\n[C20] Generating formal result tables...")
    out_path = FIGURES_DIR / 'tables_results.xlsx'

    # --- Tab.1: Best overall per task ---
    task_ids_main = [t['id'] for t in TASKS]
    task_ids_supp = [t['id'] for t in TASKS_SUPPLEMENTARY]
    all_task_ids = task_ids_main + task_ids_supp

    rows = []
    for tid in all_task_ids:
        sub = summary_df[summary_df['Task'] == tid]
        if sub.empty:
            continue
        best_idx = sub['F1_mean'].idxmax()
        r = sub.loc[best_idx]
        rows.append({
            'Task': r['Task'],
            'Task Name': r['TaskName'],
            'Best Model': r['Model'],
            'Best Preprocess': r['Preprocess'],
            'F1 (mean±std)': f"{r['F1_mean']:.3f}±{r['F1_std']:.3f}",
            'BA (mean±std)': f"{r['BA_mean']:.3f}±{r['BA_std']:.3f}",
            'Acc (mean±std)': f"{r['Acc_mean']:.3f}±{r['Acc_std']:.3f}",
            'F1_mean': r['F1_mean'],
        })
    tab1 = pd.DataFrame(rows).sort_values('F1_mean', ascending=False).drop(columns='F1_mean')

    # --- Tab.2: Presence tasks, each model at its best preprocessing ---
    pres_ids = ['P1_thiram_presence', 'P2_mg_presence', 'P3_mba_presence']
    tab2 = _best_preprocess_per_model_task(summary_df, pres_ids)

    # --- Tab.3: Positive-only molar grading tasks ---
    conc_ids = ['G1_thiram_molar_grade', 'G2_mg_molar_grade', 'G3_mba_molar_grade']
    tab3 = _best_preprocess_per_model_task(summary_df, conc_ids)

    # --- Tab.4: Preprocessing comparison for best model per task ---
    rows4 = []
    for tid in all_task_ids:
        sub = summary_df[summary_df['Task'] == tid]
        if sub.empty:
            continue
        best_model = sub.loc[sub['F1_mean'].idxmax(), 'Model']
        for pp in PREPROCESS_TAGS:
            row = sub[(sub['Model'] == best_model) & (sub['Preprocess'] == pp)]
            if row.empty:
                continue
            r = row.loc[row['F1_mean'].idxmax()]
            rows4.append({
                'Task': tid,
                'Task Name': r['TaskName'],
                'Model': best_model,
                'Preprocess': pp,
                'F1_mean': r['F1_mean'],
                'F1_std': r['F1_std'],
                'BA_mean': r['BA_mean'],
                'Acc_mean': r['Acc_mean'],
            })
    tab4 = pd.DataFrame(rows4)

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        tab1.to_excel(writer, sheet_name='Tab1_Overview', index=False)
        tab2.to_excel(writer, sheet_name='Tab2_Presence', index=False)
        tab3.to_excel(writer, sheet_name='Tab3_PositiveGrade', index=False)
        tab4.to_excel(writer, sheet_name='Tab4_Preprocessing', index=False)

    print(f"  Saved: {out_path}")
    print(f"  Tab.1: {len(tab1)} rows (best per task)")
    print(f"  Tab.2: {len(tab2)} rows (presence tasks)")
    print(f"  Tab.3: {len(tab3)} rows (concentration tasks)")
    print(f"  Tab.4: {len(tab4)} rows (preprocessing comparison)")
    return tab1


def _best_preprocess_per_model_task(summary_df, task_ids):
    """For each (model, task), pick the best preprocessing."""
    sub = summary_df[summary_df['Task'].isin(task_ids)]
    rows = []
    for (tid, model), grp in sub.groupby(['Task', 'Model']):
        best_idx = grp['F1_mean'].idxmax()
        r = grp.loc[best_idx]
        rows.append({
            'Task': tid,
            'Task Name': r['TaskName'],
            'Model': model,
            'Preprocess': r['Preprocess'],
            'F1 (mean±std)': f"{r['F1_mean']:.3f}±{r['F1_std']:.3f}",
            'BA (mean±std)': f"{r['BA_mean']:.3f}±{r['BA_std']:.3f}",
            'Acc (mean±std)': f"{r['Acc_mean']:.3f}±{r['Acc_std']:.3f}",
        })
    return pd.DataFrame(rows)


SCREENING_SELECTION_RULES = {
    'RF': {
        'selection_role': 'Global baseline',
        'selection_anchor_type': 'full_matrix_mean_f1',
        'recommended_variant_scope': 'p2/p4',
        'selection_reason': (
            'Highest grouped full-matrix mean F1 across the 8 candidate families; '
            'retained as the strongest global classical baseline for fair optimization.'
        ),
    },
    'Spectrum-KAN': {
        'selection_role': 'Hardest-slice specialist',
        'selection_anchor_type': 'hardest_slice_task_win',
        'recommended_variant_scope': 'p1/p4',
        'selection_reason': (
            'Best current grouped result on G2 MG Positive Molar Grade, the hardest '
            'mainline slice; retained as the KAN-side representative model.'
        ),
    },
}


def task_f4_f5_screening_closure(summary_df):
    """Generate E3 closure artifacts for candidate screening.

    Outputs:
    - stage-local screening_summary_table.csv
    - stage-local screening_best_by_task.csv
    - stage-local selected_model_manifest.csv
    - figures/tables_screening_closure.xlsx
    """
    print("\n[F4/F5] Generating screening closure artifacts...")

    screening_dir = get_active_screening_dir(include_legacy=False)
    if screening_dir is None or not screening_dir.exists():
        screening_dir = get_active_screening_dir()
    if screening_dir is None or not screening_dir.exists():
        print('  Skipped: no screening directory found.')
        return None, None, None

    matrix_path = screening_dir / 'benchmark_full_matrix.csv'
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path)
    else:
        matrix = summary_df.copy()

    best_path = screening_dir / 'benchmark_best_by_task.csv'
    if best_path.exists():
        best_by_task = pd.read_csv(best_path)
    else:
        best_by_task = (
            matrix.sort_values(['Task', 'F1_mean'], ascending=[True, False])
            .groupby('Task', as_index=False)
            .head(1)
            .reset_index(drop=True)
        )

    task_order = {task['id']: idx for idx, task in enumerate(TASKS + TASKS_SUPPLEMENTARY)}
    task_wins = best_by_task['Model'].value_counts()
    best_per_model = (
        matrix.sort_values(['Model', 'F1_mean'], ascending=[True, False])
        .groupby('Model', as_index=False)
        .head(1)
        .reset_index(drop=True)
        .rename(columns={
            'Task': 'BestTask',
            'TaskName': 'BestTaskName',
            'Preprocess': 'BestTaskPreprocess',
            'F1_mean': 'BestTaskF1',
            'F1_std': 'BestTaskF1Std',
        })
    )

    closure_df = (
        matrix.groupby('Model', as_index=False)[['F1_mean', 'BA_mean', 'Acc_mean']]
        .mean()
        .rename(columns={
            'F1_mean': 'MeanF1',
            'BA_mean': 'MeanBA',
            'Acc_mean': 'MeanAcc',
        })
        .merge(
            best_per_model[
                ['Model', 'BestTask', 'BestTaskName', 'BestTaskPreprocess', 'BestTaskF1', 'BestTaskF1Std']
            ],
            on='Model',
            how='left',
        )
    )
    closure_df['TaskWins'] = closure_df['Model'].map(task_wins).fillna(0).astype(int)
    closure_df['SelectedForE4'] = closure_df['Model'].isin(SCREENING_SELECTION_RULES)
    closure_df['SelectionRole'] = closure_df['Model'].map(
        lambda model: SCREENING_SELECTION_RULES.get(model, {}).get('selection_role', 'Screening reference only')
    )
    closure_df['RecommendedVariantScope'] = closure_df['Model'].map(
        lambda model: SCREENING_SELECTION_RULES.get(model, {}).get('recommended_variant_scope', '')
    )
    closure_df['SelectionReason'] = closure_df['Model'].map(
        lambda model: SCREENING_SELECTION_RULES.get(model, {}).get(
            'selection_reason',
            'Not retained for the representative-model optimization pool.',
        )
    )
    closure_df = closure_df.sort_values(['MeanF1', 'MeanBA', 'MeanAcc'], ascending=False).reset_index(drop=True)
    closure_df.insert(0, 'Rank', np.arange(1, len(closure_df) + 1))

    screening_best_df = best_by_task.copy().rename(columns={
        'Model': 'BestModel',
        'Preprocess': 'BestPreprocess',
    })
    screening_best_df['SelectedForE4'] = screening_best_df['BestModel'].isin(SCREENING_SELECTION_RULES)
    screening_best_df['SelectionRole'] = screening_best_df['BestModel'].map(
        lambda model: SCREENING_SELECTION_RULES.get(model, {}).get('selection_role', 'Task winner only')
    )
    screening_best_df['SelectionReason'] = screening_best_df['BestModel'].map(
        lambda model: SCREENING_SELECTION_RULES.get(model, {}).get(
            'selection_reason',
            'Task-level winner in candidate screening, but not retained for the representative pool.',
        )
    )
    screening_best_df['TaskOrder'] = screening_best_df['Task'].map(lambda tid: task_order.get(tid, 999))
    screening_best_df = screening_best_df.sort_values('TaskOrder').drop(columns='TaskOrder').reset_index(drop=True)

    selection_anchor_df = (
        best_by_task[best_by_task['Model'].isin(SCREENING_SELECTION_RULES)]
        .sort_values(['Model', 'F1_mean'], ascending=[True, False])
        .groupby('Model', as_index=False)
        .head(1)
        .reset_index(drop=True)
        .rename(columns={
            'Task': 'SelectionAnchorTask',
            'TaskName': 'SelectionAnchorTaskName',
            'Preprocess': 'SelectionAnchorPreprocess',
            'F1_mean': 'SelectionAnchorF1',
            'F1_std': 'SelectionAnchorF1Std',
        })
    )

    selected_manifest = closure_df[closure_df['SelectedForE4']].copy()
    selected_manifest['SelectionStage'] = screening_dir.name
    selected_manifest['SelectionAnchorType'] = selected_manifest['Model'].map(
        lambda model: SCREENING_SELECTION_RULES.get(model, {}).get('selection_anchor_type', '')
    )
    selected_manifest = selected_manifest.merge(
        selection_anchor_df[
            [
                'Model',
                'SelectionAnchorTask',
                'SelectionAnchorTaskName',
                'SelectionAnchorPreprocess',
                'SelectionAnchorF1',
                'SelectionAnchorF1Std',
            ]
        ],
        on='Model',
        how='left',
    )
    selected_manifest = selected_manifest[
        [
            'SelectionStage',
            'Model',
            'SelectionRole',
            'SelectionAnchorType',
            'SelectionAnchorTask',
            'SelectionAnchorTaskName',
            'SelectionAnchorPreprocess',
            'SelectionAnchorF1',
            'SelectionAnchorF1Std',
            'MeanF1',
            'MeanBA',
            'MeanAcc',
            'RecommendedVariantScope',
            'SelectionReason',
        ]
    ].reset_index(drop=True)

    summary_out = screening_dir / 'screening_summary_table.csv'
    best_out = screening_dir / 'screening_best_by_task.csv'
    manifest_out = screening_dir / 'selected_model_manifest.csv'
    excel_out = FIGURES_DIR / 'tables_screening_closure.xlsx'

    closure_df.to_csv(summary_out, index=False, encoding='utf-8-sig')
    screening_best_df.to_csv(best_out, index=False, encoding='utf-8-sig')
    selected_manifest.to_csv(manifest_out, index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
        closure_df.to_excel(writer, sheet_name='F4_ScreeningSummary', index=False)
        screening_best_df.to_excel(writer, sheet_name='F5_BestByTask', index=False)
        selected_manifest.to_excel(writer, sheet_name='SelectedModelManifest', index=False)

    print(f'  Saved: {summary_out}')
    print(f'  Saved: {best_out}')
    print(f'  Saved: {manifest_out}')
    print(f'  Saved: {excel_out}')
    return closure_df, screening_best_df, selected_manifest


# =====================================================================
# C12: Wilcoxon Signed-Rank Tests Between Model Families
# =====================================================================
MODEL_FAMILY_BY_MODEL = {
    'PLS-DA': 'Classical-ML',
    'SVM': 'Classical-ML',
    'RF': 'Classical-ML',
    'ExtraTrees': 'Classical-ML',
    'KNN': 'Classical-ML',
    'LDA': 'Classical-ML',
    'HistGradientBoosting': 'Classical-ML',
    'XGBoost': 'Classical-ML',
    '1D-CNN': '1D-DL',
    '1D-ResNet': '1D-DL',
    'Spectrum-KAN': 'KAN',
    'Chem-KAN': 'KAN',
    'KAN-CNN': 'KAN',
    'PLS-DA-feat': 'Feature-ML',
    'RF-feat': 'Feature-ML',
    'SVM-feat': 'Feature-ML',
    '1D-CNN-feat': 'Feature-DL',
    '1D-ResNet-feat': 'Feature-DL',
    'Feature-KAN': 'Feature-KAN',
    'KAN-CNN-feat': 'Feature-KAN',
    'Ensemble': 'Ensemble',
}

FAMILY_COLORS = {
    'Classical-ML': '#1f77b4',
    '1D-DL': '#ff7f0e',
    'KAN': '#2ca02c',
    'Feature-ML': '#d62728',
    'Feature-DL': '#9467bd',
    'Feature-KAN': '#8c564b',
    'Ensemble': '#e377c2',
    'Other': '#7f7f7f',
}


def _annotate_model_family(detail_df):
    annotated = detail_df.copy()
    annotated['Family'] = annotated['Model'].map(
        lambda model: MODEL_FAMILY_BY_MODEL.get(model, 'Other')
    )
    return annotated


def _build_family_scores(detail_best):
    detail_with_family = _annotate_model_family(detail_best)
    family_models = (
        detail_with_family.groupby('Family')['Model']
        .apply(lambda values: sorted(set(values)))
        .to_dict()
    )

    fam_scores = {}
    for fam_name, fam_df in detail_with_family.groupby('Family'):
        fam_mean = fam_df.groupby(['Task', 'Fold'])['MacroF1'].mean().reset_index()
        fam_mean = fam_mean.sort_values(['Task', 'Fold'])
        fam_scores[fam_name] = fam_mean['MacroF1'].values

    return detail_with_family, fam_scores, family_models

def task_c12(detail_df):
    """
    Wilcoxon signed-rank test between model families.

    Methodology (following Demšar 2006, JMLR 7:1-30):
    For each family, compute mean F1 per (task, fold) across models in that family.
    Then pairwise Wilcoxon across all (task × fold) observations.
    """
    print("\n[C12] Running Wilcoxon signed-rank tests...")

    # Use best preprocessing per (model, task) from detail data
    # First, find best preprocessing per (model, task) from summary-level
    best_pp = detail_df.groupby(['Task', 'Model', 'Preprocess'])['MacroF1'].mean()
    best_pp = best_pp.reset_index()
    idx = best_pp.groupby(['Task', 'Model'])['MacroF1'].idxmax()
    best_pp = best_pp.loc[idx][['Task', 'Model', 'Preprocess']].copy()

    # Merge to get per-fold F1 at best preprocessing
    detail_best = detail_df.merge(best_pp, on=['Task', 'Model', 'Preprocess'])

    detail_best, fam_scores, family_models = _build_family_scores(detail_best)
    fam_names = [name for name, values in fam_scores.items() if len(values) > 0]
    if len(fam_names) < 2:
        print('  Skipped: fewer than 2 model families present in active results.')
        return pd.DataFrame()

    # Pairwise Wilcoxon
    results = []
    for a, b in combinations(fam_names, 2):
        va, vb = fam_scores[a], fam_scores[b]
        min_len = min(len(va), len(vb))
        va, vb = va[:min_len], vb[:min_len]

        diff = va - vb
        if np.all(diff == 0):
            stat, pval = np.nan, 1.0
        else:
            try:
                stat, pval = wilcoxon(va, vb, alternative='two-sided')
            except ValueError:
                stat, pval = np.nan, 1.0

        results.append({
            'Family A': a,
            'Family B': b,
            'Mean F1 (A)': f"{np.mean(va):.3f}",
            'Mean F1 (B)': f"{np.mean(vb):.3f}",
            'Wilcoxon Stat': f"{stat:.1f}" if not np.isnan(stat) else 'N/A',
            'p-value': f"{pval:.4f}",
            'Significant (α=0.05)': '✓' if pval < 0.05 else '',
        })

    tab5 = pd.DataFrame(results)

    out_path = FIGURES_DIR / 'tables_wilcoxon.xlsx'
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        tab5.to_excel(writer, sheet_name='Tab5_Wilcoxon', index=False)

        # Also add family summary
        fam_summary = []
        for fam_name in fam_names:
            vals = fam_scores[fam_name]
            fam_summary.append({
                'Family': fam_name,
                'Models': ', '.join(family_models[fam_name]),
                'Mean F1': f"{np.mean(vals):.4f}",
                'Std F1': f"{np.std(vals):.4f}",
                'N observations': len(vals),
            })
        pd.DataFrame(fam_summary).to_excel(writer, sheet_name='Family_Summary', index=False)

    print(f"  Saved: {out_path}")
    n_sig = sum(1 for r in results if r['Significant (α=0.05)'] == '✓')
    print(f"  {n_sig}/{len(results)} pairwise comparisons significant at α=0.05")
    return tab5


# =====================================================================
# C19: Model Family Boxplot (Fig.7)
# =====================================================================
def task_c19(detail_df):
    """
    Boxplot of per-fold F1 scores grouped by model family.
    Following Kalatzis 2025 (benchmark paper visualization convention).
    """
    print("\n[C19] Generating model family boxplot (Fig.7)...")

    # Use best preprocessing per (model, task)
    best_pp = detail_df.groupby(['Task', 'Model', 'Preprocess'])['MacroF1'].mean()
    best_pp = best_pp.reset_index()
    idx = best_pp.groupby(['Task', 'Model'])['MacroF1'].idxmax()
    best_pp = best_pp.loc[idx][['Task', 'Model', 'Preprocess']].copy()
    detail_best = detail_df.merge(best_pp, on=['Task', 'Model', 'Preprocess'])

    detail_best = _annotate_model_family(detail_best)
    detail_best = detail_best.dropna(subset=['Family']).copy()

    # Family order by median F1
    fam_medians = detail_best.groupby('Family')['MacroF1'].median().sort_values(ascending=False)
    fam_order = fam_medians.index.tolist()
    if not fam_order:
        print('  Skipped: no model families present in active results.')
        return

    # Color palette
    fig, ax = plt.subplots(figsize=(12, 6))
    bp_data = [detail_best[detail_best['Family'] == fam]['MacroF1'].values for fam in fam_order]
    bp = ax.boxplot(bp_data, labels=fam_order, patch_artist=True, widths=0.6,
                    showfliers=True, flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5})

    for patch, fam in zip(bp['boxes'], fam_order):
        patch.set_facecolor(FAMILY_COLORS.get(fam, '#999'))
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, fam in enumerate(fam_order):
        vals = detail_best[detail_best['Family'] == fam]['MacroF1'].values
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   c=FAMILY_COLORS.get(fam, '#999'), alpha=0.3, s=8, zorder=3)

    ax.set_ylabel('Macro F1 Score', fontsize=12)
    ax.set_title('Model Family Performance Comparison (All Tasks × All Folds)', fontsize=13)
    ax.set_xticklabels(fam_order, rotation=25, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Annotate medians
    for i, fam in enumerate(fam_order):
        med = np.median(bp_data[i])
        ax.annotate(f'{med:.3f}', xy=(i + 1, med), xytext=(i + 1.3, med + 0.02),
                    fontsize=8, color='black', ha='center')

    plt.tight_layout()
    out_path = FIG_MODELS / 'fig7_family_boxplot.png'
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# =====================================================================
# C18: Confusion Matrix Montage (Fig.6)
# =====================================================================
def task_c18(summary_df, all_preds):
    """
    Re-generate confusion matrices for best model per task,
    arranged as a 2×4 montage figure for paper.
    Uses predictions from the best preprocessing.
    """
    print("\n[C18] Generating confusion matrix montage (Fig.6)...")

    all_tasks = TASKS + TASKS_SUPPLEMENTARY
    task_ids_main = [t['id'] for t in TASKS]

    # Find best (model, preprocess) per task
    best_configs = []
    for tk in all_tasks:
        sub = summary_df[summary_df['Task'] == tk['id']]
        if sub.empty:
            continue
        best_idx = sub['F1_mean'].idxmax()
        r = sub.loc[best_idx]
        source_stage = r['source_stage'] if 'source_stage' in r.index else ''
        key = (tk['id'], r['Model'], r['Preprocess'], source_stage)
        legacy_key = (tk['id'], r['Model'], r['Preprocess'])
        if key in all_preds:
            best_configs.append((tk, r['Model'], r['Preprocess'], source_stage, r['F1_mean'], r['F1_std']))
        elif legacy_key in all_preds:
            best_configs.append((tk, r['Model'], r['Preprocess'], '', r['F1_mean'], r['F1_std']))

    if not best_configs:
        print("  No predictions found, skipping.")
        return

    # Select the main tasks for the core figure
    configs_main = [c for c in best_configs if c[0]['id'] in task_ids_main]
    _plot_cm_montage(configs_main, all_preds, FIG_MODELS / 'fig6_cm_montage.png',
                     title='Confusion Matrices – Best Model per Task (5-Fold CV)')

    # Also generate for supplementary tasks
    configs_supp = [c for c in best_configs if c[0]['id'] not in task_ids_main]
    if configs_supp:
        _plot_cm_montage(configs_supp, all_preds, FIG_MODELS / 'fig6b_cm_montage_supplementary.png',
                         title='Confusion Matrices – Supplementary Tasks')


def _plot_cm_montage(configs, all_preds, out_path, title=''):
    n = len(configs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (tk, model, pp, source_stage, f1m, f1s) in enumerate(configs):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        key = (tk['id'], model, pp, source_stage)
        legacy_key = (tk['id'], model, pp)
        yt, yp = all_preds[key] if key in all_preds else all_preds[legacy_key]
        cm = confusion_matrix(yt, yp, labels=tk['classes'])

        # Normalize for color (row-wise), display counts
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=tk['classes'], yticklabels=tk['classes'],
                    ax=ax, cbar=False, vmin=0, vmax=1)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True', fontsize=9)
        ax.set_title(f"{tk['name']}\n{model} ({pp}) F1={f1m:.3f}", fontsize=9)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# =====================================================================
# C17: Peak-Annotated Mean Spectra (Fig.1)
# =====================================================================
# Literature-backed peak assignments
# Thiram: Linh 2024 RSC Adv (DOI:10.1039/D4RA00048J); Oliveira 2021 J Raman Spectrosc
# MG:     Qin 2021 Spectroscopy 36(4); Yang 2020 Microchimica Acta (133 cit.)
# MBA:    Mhlanga 2021 Mat Today Comm; Marques 2022 PCCP
PEAK_ANNOTATIONS = {
    'Thiram': [
        (556,  'ν(S–S)\n+ν(C–S–S)'),
        (928,  'ν(C–S)'),
        (1150, 'ν(N–C)\n+ρ(CH₃)'),
        (1382, 'ν(N–C)\n+δ(CH₃)'),
        (1510, 'ν(C=N)'),
    ],
    'MG': [
        (908,  'Ring\n(o.p.)'),
        (1170, 'δ(C–H)\ni.p.'),
        (1394, 'δ(C–H)\ni.p.'),
        (1616, 'ν(C=C)\nring'),
    ],
    'MBA': [
        (1080, 'Ring br.\n+ν(C–S)'),
        (1180, 'δ(C–H)\ni.p.'),
        (1490, 'ν(COO⁻)\n+ν(C=C)'),
        (1590, 'ν(C=C)\nring'),
    ],
}

SUBST_COLORS = {'Thiram': '#E53935', 'MG': '#43A047', 'MBA': '#1E88E5'}


def task_c17(meta, wn, X_p1):
    """
    Publication-quality mean spectra with literature-backed peak annotations.
    Uses P1 preprocessing (SNV-normalized) for clearest peaks.
    """
    print("\n[C17] Generating peak-annotated mean spectra (Fig.1)...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    for ax, (subst, col, has_col) in zip(axes, [
        ('Thiram', 'c_thiram', 'has_thiram'),
        ('MG', 'c_mg', 'has_mg'),
        ('MBA', 'c_mba', 'has_mba'),
    ]):
        color = SUBST_COLORS[subst]

        # Use single-substance spectra only for cleanest signal
        mask_single = (meta['mixture_order'] == 1) & (meta[has_col] == True)
        idx = meta[mask_single].index.values

        if len(idx) == 0:
            idx = meta[meta[has_col] == True].index.values

        mean_spec = X_p1[idx].mean(axis=0)
        std_spec = X_p1[idx].std(axis=0)

        ax.plot(wn, mean_spec, color=color, lw=1.5, label=f'{subst} (n={len(idx)})')
        ax.fill_between(wn, mean_spec - std_spec, mean_spec + std_spec,
                        alpha=0.15, color=color)

        # Annotate peaks
        peaks = PEAK_ANNOTATIONS[subst]
        for peak_wn, label in peaks:
            # Find actual intensity at peak position
            peak_idx = np.argmin(np.abs(wn - peak_wn))
            peak_y = mean_spec[peak_idx]

            # Draw vertical line at peak
            ax.axvline(x=peak_wn, color='gray', linestyle=':', alpha=0.4, lw=0.8)

            # Annotate with arrow
            y_offset = (mean_spec.max() - mean_spec.min()) * 0.15
            ax.annotate(
                f'{peak_wn}\n{label}',
                xy=(peak_wn, peak_y),
                xytext=(peak_wn, peak_y + y_offset),
                fontsize=7.5,
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8),
            )

        ax.set_ylabel('P1 Intensity (SNV)', fontsize=11)
        ax.set_title(f'{subst} – Single-Component Mean SERS Spectrum', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Raman Shift (cm⁻¹)', fontsize=11)
    axes[-1].set_xlim(WN_MIN, WN_MAX)

    plt.tight_layout()
    out_path = FIG_EDA / 'fig1_peak_annotated_spectra.png'
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# =====================================================================
# E5: Intra-Folder RSD of Peak Intensities
# =====================================================================
def task_e5(meta, wn, X_p1):
    """
    Calculate Relative Standard Deviation (RSD) within each folder
    at characteristic peak positions.

    RSD < 20% is generally acceptable for SERS quantitative analysis
    (following Chen 2018 Anal Chem; Linh 2024 RSC Adv).
    """
    print("\n[E5] Calculating intra-folder RSD...")

    peak_positions = {
        'T_560': 560, 'T_1382': 1382,
        'MG_1170': 1170, 'MG_1616': 1616,
        'MBA_1080': 1080, 'MBA_1590': 1590,
    }
    window = 5  # ±5 indices around peak

    rows = []
    for folder_name in meta['folder_name'].unique():
        folder_idx = meta[meta['folder_name'] == folder_name].index.values
        n_spec = len(folder_idx)
        if n_spec < 3:
            continue

        row = {'Folder': folder_name, 'N_spectra': n_spec}
        for peak_name, peak_wn in peak_positions.items():
            cidx = np.argmin(np.abs(wn - peak_wn))
            lo = max(0, cidx - window)
            hi = min(X_p1.shape[1], cidx + window + 1)

            # Peak intensity = max in window
            intensities = np.max(X_p1[folder_idx, lo:hi], axis=1)
            mean_val = np.mean(intensities)
            std_val = np.std(intensities)
            rsd = (std_val / abs(mean_val) * 100) if abs(mean_val) > 1e-10 else np.nan
            row[f'{peak_name}_RSD%'] = rsd
            row[f'{peak_name}_mean'] = mean_val
        rows.append(row)

    rsd_df = pd.DataFrame(rows)

    # Summary statistics
    rsd_cols = [c for c in rsd_df.columns if c.endswith('_RSD%')]
    summary_rows = []
    for col in rsd_cols:
        vals = rsd_df[col].dropna()
        summary_rows.append({
            'Peak': col.replace('_RSD%', ''),
            'Median RSD%': f"{vals.median():.1f}",
            'Mean RSD%': f"{vals.mean():.1f}",
            'Max RSD%': f"{vals.max():.1f}",
            'N_folders (<20%)': f"{(vals < 20).sum()}/{len(vals)}",
        })
    summary_df = pd.DataFrame(summary_rows)

    out_path = FIGURES_DIR / 'tables_rsd.xlsx'
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        rsd_df.to_excel(writer, sheet_name='RSD_by_folder', index=False)
        summary_df.to_excel(writer, sheet_name='RSD_summary', index=False)

    print(f"  Saved: {out_path}")
    print(f"  RSD Summary:")
    print(summary_df.to_string(index=False))
    return rsd_df


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 60)
    print("Benchmark Summary Tasks")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_EDA.mkdir(parents=True, exist_ok=True)
    FIG_MODELS.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[Loading data...]")
    summary_df = load_all_summaries()
    detail_df = load_all_details()
    all_preds = load_all_predictions()
    print(f"  Summary: {len(summary_df)} rows")
    print(f"  Detail:  {len(detail_df)} rows")
    print(f"  Predictions: {len(all_preds)} (task, model, preprocess) combos")

    # Load spectral data for C17 and E5
    wn = np.load(PROCESSED_DIR / 'wavenumber.npy')
    X_p1 = np.load(PROCESSED_DIR / 'X_p1.npy')
    meta = build_metadata()
    print(f"  Spectra: {X_p1.shape}, Wavenumber: {wn.shape}")

    # Execute tasks
    task_c20(summary_df)
    task_f4_f5_screening_closure(summary_df)
    task_c12(detail_df)
    task_c19(detail_df)
    task_c18(summary_df, all_preds)
    task_c17(meta, wn, X_p1)
    task_e5(meta, wn, X_p1)

    print("\n" + "=" * 60)
    print("Benchmark summary complete (run scripts/analysis/benchmark_shap.py separately for SHAP)")
    print("=" * 60)


if __name__ == '__main__':
    main()
