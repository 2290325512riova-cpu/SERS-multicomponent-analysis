# Project Architecture

Last updated: 2026-05-25

This document is the current source of truth for the active repository architecture. Historical grouped-CV work is preserved, but the active paper pipeline now lives under `paper_main/`.

## 1. Paper Position

Working title direction:

> Systematic Benchmarking of Machine Learning Models for Multi-Pesticide Screening via SERS: Preprocessing Effects, Model Selection, and Chemical Interpretability

Core contributions:

1. 13模型×6预处理的系统benchmark，揭示预处理×模型交互效应（RF/ExtraTrees在SNV+SG下最优且最稳定）
2. TreeSHAP峰簇分析证明模型决策基于真实化学振动模式而非噪声
3. SHAP-guided spectral region selection证明模型性能可在仅保留化学可解释区域时维持（actionable interpretability）
4. 土壤基质筛查可行性验证（soil-only CV），证明方法在复杂基质中的适用性

Core narrative:

- AgNPs-SERS spectra are used for simultaneous ternary pesticide-mixture screening.
- Random stratified spectrum-level 5-fold CV is used as the main within-dataset benchmark because it matches common applied SERS-ML reporting practice.
- Grouped/folder-level CV is retained as a stricter transfer stress test in supplementary information.
- Soil-only CV (train and test within soil spectra) demonstrates screening feasibility in complex matrices. The earlier pure→soil transfer experiment failed due to domain shift and serves as motivation for the soil-only approach.
- The augmentation method is limited to composition-constrained spectral mixing.
- SHAP is reported at peak-cluster/window level and cross-checked against known Raman/SERS peaks. Literature peak list pending update (add Thiram 930/1510, MG 1220).
- Formal Optuna is skipped in the manuscript mainline; downstream interpretation and soil screening use the fixed ExtraTrees/p1 recipe.

## 2. Repository Boundaries

The larger workspace contains a parent Git repository and this child repository.

Active repository:

```text
D:\通过拉曼光谱预测物及其浓度\core subject
```

Rules:

- Work inside `core subject/` unless the user explicitly requests parent-repo cleanup.
- Do not run broad staging commands in the parent repository.
- Keep cleanup, archiving, feature changes, and documentation in separate commits where possible.

## 3. Data Facts

| Item | Value |
| --- | --- |
| Data version | `pure63_mainline` |
| Pure spectra | 901 |
| Pure composition folders | 63 |
| Wavenumber grid | 400-1800 cm^-1, 1 cm^-1 step, 1401 points |
| Current soil spectra detected | 90 total (79 design + 11 blanks) across 13 folders |
| Main tasks | P1/P2/P3 presence and G1/G2/G3 positive concentration grade |
| Supplementary task | S1 mixture order |

Important caution:

- Soil spectra total 90: 79 design samples (10 folders with analytes) + 11 blanks/controls (3 folders). The parser correctly reports 79 design spectra. Blanks should be included in specificity testing.

## 4. Split Protocols

| Split | File | Role |
| --- | --- | --- |
| Grouped 5-fold CV | `data/pure63_mainline/splits/cv_split_pure63_main.csv` | Supplementary folder-level transfer stress test |
| Random stratified 5-fold CV | `data/pure63_mainline/splits/cv_split_random_5fold.csv` | Main within-dataset benchmark |

Random split QC:

- `cv_split_random_5fold_qc_by_fold.csv`
- `cv_split_random_5fold_qc_by_stratum.csv`

The random split uses the joint key:

```text
has_thiram, has_mg, has_mba, c_thiram, c_mg, c_mba
```

Two exact-composition strata have fewer than five spectra, so the split is best-effort joint stratified rather than perfectly balanced at every exact composition.

## 5. Preprocessing Registry

Preprocessing variants are centralized in `src/config.py` as `PREPROCESS_VARIANTS`.

| Tag | Definition | Cache |
| --- | --- | --- |
| `raw` | interpolated raw spectra | `X_raw.npy` |
| `p1` | cosmic removal + SG smoothing + ALS baseline + SNV | `X_p1.npy` |
| `p2` | SG smoothing + ALS baseline + first derivative + SNV | `X_p2.npy` |
| `p3` | ALS baseline + vector normalization | `X_p3.npy` |
| `p4` | cosmic removal + SG smoothing + ALS baseline, no normalization | `X_p4.npy` |
| `p5` | cosmic removal + SG smoothing + ALS baseline + second derivative + SNV | `X_p5.npy` |

Generated p5 QC:

- `data/pure63_mainline/processed/X_p5_qc.csv`
- Shape: 901 spectra by 1401 points.
- Finite fraction: 1.0.
- Lightweight peak-window RSD QC does not make p5 the most repeatable representation; use p5 as a derivative-feature candidate, not as a data-quality headline.

## 6. Model Families

Main benchmark registry includes:

- RF
- ExtraTrees
- HistGradientBoosting
- XGBoost
- SVM
- KNN
- PLS-DA
- LDA
- 1D-CNN
- 1D-ResNet
- Spectrum-KAN
- KAN-CNN
- RamanNet-Lite

Operational note from the first RTX 3090 server run:

- The first fast classical screen used RF, ExtraTrees, HistGradientBoosting, XGBoost, SVM, KNN, and PLS-DA across raw/p1/p2/p3/p4/p5.
- The completion run `benchmark_completion_3090_20260522` added the missing DL variants, 1D-ResNet, KAN-CNN, and LDA for all six preprocessing variants.
- The complete 13-model x 6-preprocessing benchmark has been returned and verified locally under `data/pure63_mainline/models/paper_main/benchmark_random_cv/` after appending RamanNet-Lite.
- Current complete-benchmark best-by-task macro-F1 values are P1 0.996, P2 0.985, P3 0.994, G1 1.000, G2 0.966, and G3 0.981. These are conventional spectrum-level random-CV results, not external validation.
- Overall mean macro-F1 across the complete benchmark is led by ExtraTrees, XGBoost, and RF. The added DL/KAN models are useful benchmark evidence but do not displace the classical leaders.
- Note: an earlier LDA fold-level profile was slow, but the final completion wrapper finished LDA quickly and the LDA baseline is included in the complete matrix.

XGBoost is optional at runtime. If `xgboost` is not installed, `run_random_benchmark.py` skips it unless `--strict-models` is used.

Multi-task experimental models remain in code/history where present, but they are not part of the active paper-main benchmark list.

## 7. Active Result Layout

```text
data/pure63_mainline/models/
  mainline_formal_experiment/
    01_candidate_screening/
    02_representative_model_optimization/
    03b_locked_main_results/
    04_leakage_analysis/
    05_shap_explainability/
  paper_main/
    benchmark_random_cv/
    augmentation_ablation/
    optuna_final/        # reserved legacy placeholder; formal Optuna skipped
    robustness_check/    # optional SI-only ExtraTrees sensitivity if needed
    shap_peak_cluster/
    soil_validation/
    data_quality/
```

Historical and supplementary roles:

- `01_candidate_screening/`: grouped-CV baseline landscape for SI.
- `02_representative_model_optimization/`: optimization history and hyperparameter reference.
- `03b_locked_main_results/`: locked grouped-CV strict-validation table for SI. Do not move.
- `04_leakage_analysis/`: random-vs-grouped protocol comparison for SI/discussion.
- `05_shap_explainability/`: point-level SHAP baseline to upgrade into cluster-level analysis.

Archived negative work:

```text
archive/
  MANIFEST.md
  scripts_negative/
  03_two_stage_context_aware/
```

Archived means preserved, not erased.

## 8. Active Scripts

| Script | Purpose | Typical command |
| --- | --- | --- |
| `scripts/analysis/run_data_quality.py` | RSD and peak-intensity QC tables | `python scripts/analysis/run_data_quality.py` |
| `scripts/analysis/run_random_benchmark.py` | Main random-CV benchmark | `python scripts/analysis/run_random_benchmark.py --strict-models` |
| `scripts/analysis/run_augmentation_ablation.py` | DL augmentation comparison | `python scripts/analysis/run_augmentation_ablation.py --models 1D-CNN 1D-ResNet Spectrum-KAN KAN-CNN --variants p1 raw --modes no_aug aug_no_mixup composition_mixup` |
| `scripts/analysis/run_paper_shap.py` | Final ExtraTrees/p1 TreeSHAP and peak-cluster tables | `python scripts/analysis/run_paper_shap.py --model ExtraTrees --variant p1 --tasks P1_thiram_presence P2_mg_presence P3_mba_presence G2_mg_molar_grade G3_mba_molar_grade --save-full` |
| `scripts/analysis/run_shap_cluster.py` | Legacy converter for existing top-20 SHAP points | `python scripts/analysis/run_shap_cluster.py` |
| `scripts/analysis/run_soil_validation.py` | Prepare soil metadata/cache and run train-pure -> predict-soil presence validation | `python scripts/analysis/run_soil_validation.py --variants raw p1 p5 --variant p1 --run-validation --model ExtraTrees --tasks P1_thiram_presence P2_mg_presence P3_mba_presence` |
| `scripts/analysis/leakage_analysis.py` | Historical random-vs-grouped analysis | `python scripts/analysis/leakage_analysis.py` |
| `scripts/analysis/benchmark_summary.py` | Summarize historical benchmark outputs | `python scripts/analysis/benchmark_summary.py` |

## 9. Execution Order From Here

Recommended order:

1. Run `run_data_quality.py` and inspect RSD/peak statistics.
2. Return and verify the first fast classical random-CV benchmark. Done on 2026-05-22.
3. Return and verify the complete 12-model x 6-preprocessing random-CV benchmark. Done on 2026-05-22.
4. Append RamanNet-Lite and verify the complete 13-model x 6-preprocessing random-CV benchmark. Done on 2026-05-24.
5. Generate the benchmark heatmap/table from `benchmark_full_matrix.csv` and `benchmark_best_by_task.csv`.
6. Use the fixed benchmark result to lock ExtraTrees/p1 as the primary practical model; do not run formal Optuna.
7. Treat RamanNet-Lite as the strongest DL-family representative, not as a second star model.
8. Run ExtraTrees/p1 TreeSHAP peak-cluster analysis for P1/P2/P3/G2/G3. Done on 2026-05-25. Pending: update LITERATURE_PEAKS (add Thiram 930/1510, MG 1220) and rerun matching.
9. Soil screening: (a) Pure→soil transfer failed (E019, domain shift) — serves as motivation. (b) Soil-only 5-fold CV pending formal run — preliminary results show AUC>0.93 for P1/P2/P3 with p1. (c) Blank specificity test pending. (d) Permutation test pending.
10. Generate benchmark, augmentation, SHAP, and soil figures.
11. Update `project_memory/CLAIM_EVIDENCE_MATRIX.md` before writing paper claims.

## 10. Server Artifact Return Protocol

Remote servers are compute workers, not the source of truth. A remote experiment is accepted only after its artifacts land in the local stage directory below and pass a local completeness check.

### 10.1 Before A Remote Run

Record these items in the server log bundle before a long command starts:

- approved experiment scope and whether it is smoke, screening, final, or ablation work
- local Git commit and remote checkout commit
- command line, model list, preprocessing variants, task scope, seed/fold protocol, and output directory
- Python, PyTorch, CUDA/GPU, XGBoost when used, CPU/RAM, and disk information
- remote log path and target local stage directory

Do not store remote credentials in the repository.

Within each returned local stage directory, keep generated experiment files in their normal stage paths and use these support subdirectories when the runner does not already provide a stricter layout:

- `logs/`: stdout, stderr, progress logs, and failure traces
- `timings/`: per-run or per-unit timing tables
- `run_metadata/`: environment snapshots, Git commit records, command manifests, and sync notes

### 10.2 Local Landing Map

| Remote work | Local artifact directory | Required return payload | Docs updated after local verification |
| --- | --- | --- | --- |
| Data-quality tables or figures | `data/pure63_mainline/models/paper_main/data_quality/` | generated tables/figures, logs, environment snapshot, command metadata | `EXPERIMENT_TRACKER.md`, `reports/实验结果摘要.md`, `reports/图表与表格清单.md` when evidence changes |
| Random-CV benchmark | `data/pure63_mainline/models/paper_main/benchmark_random_cv/` | `cv_results_detail_*.csv`, `cv_results_summary_*.csv`, predictions when generated, benchmark index files, logs, timings, environment snapshot, command metadata | `EXPERIMENT_TRACKER.md`, `CLAIM_EVIDENCE_MATRIX.md`, `reports/全局进度看板.md`, `reports/实验结果摘要.md`, `reports/图表与表格清单.md` |
| Augmentation ablation | `data/pure63_mainline/models/paper_main/augmentation_ablation/` | mode-specific result directories, benchmark index files, logs, timings, environment snapshot, command metadata | same as benchmark, scoped to the augmentation claim; Spectrum-KAN targeted audit and final 5-DL ablation returned and verified |
| Optional ExtraTrees robustness check | `data/pure63_mainline/models/paper_main/robustness_check/` | small sensitivity table, logs, timings, environment snapshot, command metadata; not a formal Optuna study | `EXPERIMENT_TRACKER.md`, `CLAIM_EVIDENCE_MATRIX.md`, `reports/实验结果摘要.md`, `reports/论文写作路线图.md`, `project_architecture.md` |
| Final SHAP peak-cluster work | `data/pure63_mainline/models/paper_main/shap_peak_cluster/` | attribution tables, plots, peak-window mapping inputs/outputs, logs, environment snapshot, command metadata | `EXPERIMENT_TRACKER.md`, `CLAIM_EVIDENCE_MATRIX.md`, `reports/实验结果摘要.md`, `reports/图表与表格清单.md` |
| Soil screening validation | `data/pure63_mainline/models/paper_main/soil_validation/` | metadata/QC, soil-only CV results, predictions, metrics, plots/tables, logs, environment snapshot, command metadata | `EXPERIMENT_TRACKER.md`, `CLAIM_EVIDENCE_MATRIX.md`, `reports/全局进度看板.md`, `reports/实验结果摘要.md`, `reports/图表与表格清单.md` |

Historical grouped-CV directories under `mainline_formal_experiment/` are supplementary evidence. Do not write new paper-main remote outputs into them unless the user approves a supplementary rerun.

### 10.3 Local Verification And Closeout

After return:

1. Confirm the expected model, variant, task, fold, and ablation-mode coverage from the returned detail/summary files.
2. Confirm logs and environment metadata match the returned code commit and command scope.
3. Refresh or regenerate stage-local result indexes from local summary files when needed.
4. Update reports and project memory only after the artifact set is locally readable and complete enough for its declared scope.
5. Remove transient sync archives from the repo root after verified extraction; keep the stage files and their log/metadata bundle.

## 11. Claim Guardrails

Use:

- "conventional spectrum-level validation"
- "within-dataset screening benchmark"
- "folder-level transfer stress test"
- "soil-matrix screening feasibility" (for soil-only CV results)
- "same-matrix CV" (for soil-only CV)
- "composition-constrained spectral mixing"
- "semi-quantitative MG grade screening"

Do not use without new evidence:

- "external validation" for random CV
- "leakage-free generalization" for spectrum-level random CV
- "first-ever Beer-Lambert augmentation"
- "SHAP proves competitive adsorption"
- "17/20 SHAP peak matching"
- "precise MG quantification"
- "successful cross-matrix transfer" (pure→soil transfer failed)

## 12. Memory Update Rules

At the end of any substantial phase:

1. Update `reports/全局进度看板.md`.
2. Update `project_memory/EXPERIMENT_TRACKER.md`.
3. Update `project_memory/CLAIM_EVIDENCE_MATRIX.md` if evidence status changed.
4. Commit the code/data/docs change with a focused message.
