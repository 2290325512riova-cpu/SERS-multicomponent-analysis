# Project Architecture

Last updated: 2026-06-03

This document is the current source of truth for the active repository architecture. Historical grouped-CV work is preserved, but the active paper pipeline now lives under `paper_main/`.

## 1. Paper Position

Main-line (locked 2026-05-29, strengthened 2026-06-01 by D021/D022):

> 三组分 SERS 混合体系中存在竞争吸附诱导的 MG 1616 cm⁻¹ 谱峰压制；可解释 ML 将该混合物谱干扰定位到化学峰区，并转化为紧凑谱区筛查流程。

Contribution ordering (headline → supporting):

1. **[Headline] Competitive-adsorption-induced MG suppression**: MG 1616 cm⁻¹ is strongly suppressed under co-existence (median ratio 0.355 MG+MBA / 0.206 MG+Thiram / 0.129 ternary; 283/358 ternary spectra <0.5× same-level pure MG). This is written as a discovered mixture-interference phenomenon, not a timid caveat.
2. **[Supporting] TreeSHAP peak attribution**: model decisions fall back onto real chemical bands and overlap-sensitive interference windows. The formal audited output is 22/29 clusters on known chemical bands, accounting for 91.2% of total SHAP attribution mass; 2 cross-component clusters are framed as captured mixture-interference chemistry. This shows the model captures chemical information rather than random noise.
3. **[Supporting] SHAP-guided compact screening**: keep 10% wavenumbers (141/1401), retrain, all six tasks remain >0.95 macro-F1 → chemically auditable compact workflow.
4. **[Supporting/methods] 13×6 benchmark + model selection**: rigor framing; classical ML outperforms DL at this data scale stated objectively (cite Grinsztajn 2022), NOT as "DL failed".
5. **[Application payoff] Spiked-soil matrix screening validation**: soil-matrix 5-fold CV (AUC 0.956–0.996) + blank specificity + permutation test; `same-matrix CV` is methods-level wording only.

Core narrative:

- AgNPs-SERS spectra of a ternary pesticide mixture; co-existence suppresses MG 1616 and reshapes overlap-sensitive MG-associated bands (headline), interpretable ML localizes the interference to chemical bands, a compact band set suffices for screening, and the method is demonstrated in a spiked soil matrix.
- Random stratified spectrum-level 5-fold CV is the within-dataset benchmark (matches applied SERS-ML practice). Grouped/folder-level CV is NOT in the paper (D017).
- Soil-only CV supports spiked-soil matrix screening validation; the failed pure→soil transfer is motivation only.
- Augmentation limited to composition-constrained spectral mixing; reframed as a spectral-additivity probe supporting the interference story.
- SHAP reported at peak-cluster/window level, cross-checked vs known peaks, and quantified with both raw cluster count and SHAP-mass weighted assignment consistency.
- Publication-style phrasing: main text says the work reveals competitive-adsorption-induced MG 1616 suppression and mixture interference. 1172/1394 are overlap-sensitive MG-associated regions that support peak-envelope reshaping, not discarded flaws. Avoid only unsupported extras such as adsorption constants, Langmuir fits, or external-validation labels.

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
| Current soil spectra detected | 89 total (79 design + 10 blanks) |
| Main tasks | P1/P2/P3 presence and G1/G2/G3 positive concentration grade |
| Supplementary task | S1 mixture order |

Important caution:

- Formal soil arm uses 79 design samples plus 10 blank/control spectra. Blank specificity is reported as 10/10 blanks rejected.

## 4. Split Protocols

| Split | File | Role |
| --- | --- | --- |
| Grouped 5-fold CV | `data/pure63_mainline/splits/cv_split_pure63_main.csv` | Data preserved for future work, NOT in current paper |
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
    shap_feature_selection/
    soil_validation/
    data_quality/
```

Historical data preserved for future work (NOT in current paper per D017):

- `01_candidate_screening/`: grouped-CV baseline landscape.
- `02_representative_model_optimization/`: optimization history and hyperparameter reference.
- `03b_locked_main_results/`: locked grouped-CV strict-validation table. Do not move.
- `04_leakage_analysis/`: random-vs-grouped protocol comparison.
- `05_shap_explainability/`: point-level SHAP baseline to upgrade into cluster-level analysis.

Archived negative work:

```text
archive/
  MANIFEST.md
  scripts_negative/
  03_two_stage_context_aware/
```

Archived means preserved, not erased.

Figure-generation cleanup note (2026-06-03):

- Superseded v2 figures, early Origin iterations, temporary style-reference downloads, and Python caches were removed after the current figure set was generated.
- Active figure outputs live in `figures/paper_main/`.
- The current reproducible figure scripts live in `scripts/figures/`.
- `make_paper_figures.py` intentionally embeds two retained Origin-clean PNGs from `figures/paper_main_origin/` for the final Fig5/Fig8 spectral panels; do not delete `origin_clean_fig5_mg_suppression.png`, `origin_clean_fig8_soil_spectra.png`, or `sers_ml_origin_clean.opju` unless those panels are redrawn from scratch.

## 8. Active Scripts

| Script | Purpose | Typical command |
| --- | --- | --- |
| `scripts/analysis/run_data_quality.py` | RSD and peak-intensity QC tables | `python scripts/analysis/run_data_quality.py` |
| `scripts/analysis/run_random_benchmark.py` | Main random-CV benchmark | `python scripts/analysis/run_random_benchmark.py --strict-models` |
| `scripts/analysis/run_augmentation_ablation.py` | DL augmentation comparison | `python scripts/analysis/run_augmentation_ablation.py --models 1D-CNN 1D-ResNet Spectrum-KAN KAN-CNN --variants p1 raw --modes no_aug aug_no_mixup composition_mixup` |
| `scripts/analysis/run_paper_shap.py` | Final ExtraTrees/p1 TreeSHAP and peak-cluster tables | `python scripts/analysis/run_paper_shap.py --model ExtraTrees --variant p1 --save-full` |
| `scripts/analysis/run_shap_feature_selection.py` | SHAP-guided spectral masking / compact screening | `python scripts/analysis/run_shap_feature_selection.py` |
| `scripts/analysis/run_shap_cluster.py` | Legacy converter for existing top-20 SHAP points | `python scripts/analysis/run_shap_cluster.py` |
| `scripts/analysis/run_soil_validation.py` | Prepare soil metadata/cache and run train-pure -> predict-soil presence validation | `python scripts/analysis/run_soil_validation.py --variants raw p1 p5 --variant p1 --run-validation --model ExtraTrees --tasks P1_thiram_presence P2_mg_presence P3_mba_presence` |
| `scripts/analysis/run_soil_only_cv.py` | Formal spiked-soil matrix screening validation | `python scripts/analysis/run_soil_only_cv.py` |
| `scripts/analysis/verify_mg_suppression.py` | MG 1616 suppression analysis for Fig5 headline | `python scripts/analysis/verify_mg_suppression.py` |
| `scripts/analysis/verify_row_alignment.py` | Verify `X_p4[i]` aligns with split metadata rows | `python scripts/analysis/verify_row_alignment.py` |
| `scripts/analysis/leakage_analysis.py` | Historical random-vs-grouped analysis | `python scripts/analysis/leakage_analysis.py` |
| `scripts/analysis/benchmark_summary.py` | Summarize historical benchmark outputs | `python scripts/analysis/benchmark_summary.py` |
| `scripts/figures/make_paper_figures.py` | Generate current Fig2-Fig8 paper-main figure drafts | `python scripts/figures/make_paper_figures.py` |
| `scripts/figures/make_paper_tables.py` | Generate T1-T3 and ST1-ST5 manuscript tables | `python scripts/figures/make_paper_tables.py` |
| `scripts/figures/build_origin_adjusted_project.py` | Rebuild retained Origin-clean spectral panels used by Fig5/Fig8 | `python scripts/figures/build_origin_adjusted_project.py` |

## 9. Execution Order From Here

Recommended order:

1. Main experiments are complete and locally verified: 13-model benchmark, 5-DL ablation, six-task TreeSHAP, SHAP-guided feature selection, MG 1616 suppression analysis, soil-only CV, permutation test summary, and blank specificity.
2. Use D020/D021/D022 as the manuscript spine: Fig5 MG 1616 suppression is the headline; Fig4 benchmark, Fig6 TreeSHAP assignment consistency, Fig7 compact screening, and Fig8 soil matrix validation support it.
3. Current Fig2-Fig8 and T1-T3/ST1-ST5 drafts are generated; Fig1 and Fig2(a)(b) await user-provided artwork/TEM/UV-Vis assets.
4. Write the Chinese manuscript draft using publication-forward Chinese-first wording. Do not reopen grouped CV, formal Optuna, or DL-rescue routes unless the user explicitly changes the frozen decisions.
5. After the Chinese draft stabilizes, perform final figure polishing without changing the locked data story.
6. Update `CLAIM_EVIDENCE_MATRIX.md`, reports, and this architecture file only when evidence status changes, not when only a figure is redrawn.

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

Historical grouped-CV directories under `mainline_formal_experiment/` are preserved for future work (NOT in current paper). Do not write new paper-main remote outputs into them unless the user approves a future rerun.

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
- "soil-matrix screening validation" / "spiked-soil screening validation" (title/caption-level)
- "same-matrix CV" (methods-level detail for soil-only CV)
- "composition-constrained spectral mixing"
- "semi-quantitative MG grade screening"

Do not use without new evidence:

- "external validation" for random CV
- "leakage-free generalization" for spectrum-level random CV
- "first-ever Beer-Lambert augmentation" (use "composition-constrained spectral mixing")
- "SHAP proves competitive adsorption"
- any SHAP peak-matching number higher than the documented audited output. Current formal output is 22/29 plus 91.2% SHAP-mass consistency; ~24/29 is SI-level extended co-adsorbed/reference-region assignment unless additional own-reference evidence upgrades it.
- "precise MG quantification"
- "successful cross-matrix transfer" (pure→soil transfer failed)
- any mention of grouped CV as paper content (not in paper per D017)

## 12. Memory Update Rules

At the end of any substantial phase:

1. Update `reports/全局进度看板.md`.
2. Update `project_memory/EXPERIMENT_TRACKER.md`.
3. Update `project_memory/CLAIM_EVIDENCE_MATRIX.md` if evidence status changed.
4. Commit the code/data/docs change with a focused message.
