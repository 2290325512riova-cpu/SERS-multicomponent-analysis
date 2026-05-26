# Experiment Tracker

## E001: Grouped Candidate Screening

Status: done
Result path: `data/pure63_mainline/models/mainline_formal_experiment/01_candidate_screening/`
Paper role: supplementary strict baseline

Notes:

- Presence tasks are usable.
- G2 MG grade is weak under grouped validation.

## E002: Representative Optimization

Status: done
Result path: `data/pure63_mainline/models/mainline_formal_experiment/02_representative_model_optimization/`
Paper role: historical optimization evidence / hyperparameter reference

Notes:

- RF remains a strong classical baseline.
- KAN/DL stability depends heavily on recipe.
- NSGA-II did not solve the core G2 limitation.

## E003: Locked Grouped Main Results

Status: done
Result path: `data/pure63_mainline/models/mainline_formal_experiment/03b_locked_main_results/`
Paper role: supplementary folder-level transfer stress test

Notes:

- Do not move this directory.

## E004: Two-Stage Context-Aware Experiment

Status: negative
Current path: `archive/03_two_stage_context_aware/`
Archived script: `archive/scripts_negative/run_two_stage_experiment.py`
Paper role: internal negative result / possible rebuttal evidence

## E005: Random-vs-Grouped Analysis

Status: done
Result path: `data/pure63_mainline/models/mainline_formal_experiment/04_leakage_analysis/`
Paper role: supplementary validation-protocol analysis

Notes:

- Avoid using the word "leakage" in paper main text.
- Use "spectrum-level validation vs folder-level transfer stress test".

## E006: SHAP Point-Level Baseline

Status: partial
Result path: `data/pure63_mainline/models/mainline_formal_experiment/05_shap_explainability/`
Paper role: basis for upgraded peak-cluster analysis

Notes:

- Current point-level hit rates do not support a strong 17/20 peak-matching claim.
- Upgrade to peak-window / cluster-level attribution.

## E007: paper_main Random-CV Pipeline

Status: complete 13-model remote benchmark returned and verified
Result path: `data/pure63_mainline/models/paper_main/`
Paper role: main result pipeline

Notes:

- Directory structure and runner scripts exist.
- The RTX 3090 remote smoke/profile run has been returned locally for `p1` on RF, XGBoost, 1D-CNN, and Spectrum-KAN.
- The returned smoke bundle includes p1 detail/summary/prediction CSVs, stage indexes, logs, and run metadata under `benchmark_random_cv/`.
- Remote run `classical_fast_no_lda_3090_20260522` has been returned and verified locally for RF, ExtraTrees, HistGradientBoosting, XGBoost, SVM, KNN, and PLS-DA across `raw,p1,p2,p3,p4,p5`.
- Local verification: `raw,p2,p3,p4,p5` each have 210 detail rows and 42 summary rows; `p1` has 270 detail rows and 54 summary rows because the earlier 1D-CNN and Spectrum-KAN smoke rows are also present.
- Final first-round index files exist under `benchmark_random_cv/`: `benchmark_best_by_task.csv`, `benchmark_full_matrix.csv`, `benchmark_source_manifest.csv`, `benchmark_source_model_mean.csv`, and `benchmark_source_preprocess_mean.csv`.
- Best first-round random-CV macro-F1 values: P1 KNN/p1 0.996, P2 RF/p1 0.985, P3 ExtraTrees/p4 0.994, G1 ExtraTrees/p2 1.000, G2 ExtraTrees/p1 0.966, G3 SVM/p1 0.981.
- Remote run `benchmark_completion_3090_20260522` has been returned and verified locally. It completed the missing 1D-CNN/Spectrum-KAN variants, added 1D-ResNet/KAN-CNN for all variants, added LDA for all variants, and rebuilt the full 12-model index.
- Local verification for the complete benchmark: each of `raw,p1,p2,p3,p4,p5` has 360 detail rows, 72 summary rows, 56,184 prediction rows, 12 models, 6 tasks, and 5 folds.
- `benchmark_full_matrix.csv` now contains 432 summary rows; `benchmark_best_by_task.csv` contains the six current best task rows.
- Best complete random-CV macro-F1 values remain: P1 KNN/p1 0.996, P2 RF/p1 0.985, P3 ExtraTrees/p4 0.994, G1 ExtraTrees/p2 1.000, G2 ExtraTrees/p1 0.966, G3 SVM/p1 0.981.
- Model-family takeaway: ExtraTrees, XGBoost, and RF have the best overall means; the added DL/KAN models provide negative/benchmark evidence rather than replacing the classical leaders.
- Note: an earlier LDA timing profile was slow, but the final completion run finished LDA quickly under the current wrapper and included it in the 12-model matrix.
- Remote run `ramannet_lite_benchmark_3090_20260524_141543` has been returned and verified locally. It appended RamanNet-Lite for all six preprocessing variants and rebuilt the full 13-model index.
- Local verification after RamanNet-Lite extension: each of `raw,p1,p2,p3,p4,p5` has 390 detail rows, 78 summary rows, 13 models, 6 tasks, and 5 folds.
- `benchmark_full_matrix.csv` now contains 468 summary rows. Best-by-task winners remain unchanged: P1 KNN/p1 0.996, P2 RF/p1 0.985, P3 ExtraTrees/p4 0.994, G1 ExtraTrees/p2 1.000, G2 ExtraTrees/p1 0.966, G3 SVM/p1 0.981.
- RamanNet-Lite has the best DL-family overall mean macro-F1 (0.877), but does not displace the classical leaders.
- **IMPORTANT (2026-05-23)**: All 4 DL models in this benchmark ran with AUG_ENABLED=True and MIXUP_ENABLED=True (config.py defaults). The runner script does not pass model_kwargs to override. Therefore current DL results are NOT a no-augmentation baseline; they already include generic augmentation + composition-constrained mixup. Augmentation ablation (E010) is required to isolate the augmentation contribution.
- **Spectrum-KAN G2 working diagnosis (2026-05-23)**: F1=0.480 is reproducible and not a data download error. Current evidence points to an interaction among weak/unstable MG signal under p1/SNV (QC diagnostic: 11/48 MG-positive folders have negative mean MG_1616 peak values), KAN tanh-basis compression, and an overparameterized first KAN layer (~1.6M parameters) with ~520 train samples per fold. Training time is ~10s/fold vs ~40s/fold for G1, suggesting early stopping into a poor solution. Treat this as a working diagnosis, not a confirmed root cause.

## E008: p5 Preprocessing Variant

Status: ready
Result path: `data/pure63_mainline/processed/X_p5.npy`
Paper role: preprocessing candidate for benchmark and SHAP peak-window analysis

Notes:

- p5 is cosmic removal + SG smooth + ALS baseline + second derivative + SNV.
- QC file: `data/pure63_mainline/processed/X_p5_qc.csv`.

## E009: Random Joint-Stratified Split

Status: ready
Result path: `data/pure63_mainline/splits/cv_split_random_5fold.csv`
Paper role: main spectrum-level validation split

Notes:

- Split uses six-label joint stratification.
- Two exact-composition strata have fewer than five spectra, so exact per-fold balance is best-effort.

## E010: Composition-Constrained Spectral Mixing

Status: full 5-DL ablation returned and verified locally
Result path: `src/dataset.py`, `src/models.py`, `data/pure63_mainline/models/paper_main/augmentation_ablation/`
Paper role: chemistry-grounded DL augmentation method

Notes:

- `spectral_mixup` requires composition keys.
- DL wrappers pass train-fold `group_id` or `folder_name` keys into mixup.
- Full 4-DL ablation returned from run `full_dl_ablation_3090_20260523_171250`.
- RamanNet-Lite add-on ablation returned from run `ramannet_lite_ablation_3090_20260524_182624`.
- Local verification after RamanNet-Lite add-on: all three modes (`no_aug`, `aug_no_mixup`, `composition_mixup`) and both variants (`p1`, `raw`) have complete summary/detail/prediction CSVs. Each summary has 30 rows, each detail has 150 rows, each prediction file has 23,410 rows; RamanNet-Lite contributes 6 summary rows, 30 detail rows, and 4,682 prediction rows per mode/variant.
- **IMPORTANT (2026-05-23)**: The current benchmark DL results ALREADY include this augmentation (config.py defaults AUG_ENABLED=True, MIXUP_ENABLED=True). The ablation must compare no_aug / aug_no_mixup / composition_mixup to isolate the contribution.
- Ablation script ready: `scripts/analysis/run_augmentation_ablation.py` (verified correct: override mechanism works, same fold_id as benchmark).
- Full ablation scope after the add-on: 5 DL models x p1 + raw x 3 modes x 6 tasks x 5 random folds.
- Adding `raw` variant is critical: p1/SNV can alter the selected MG_1616 peak sign/scale in a subset of MG-positive folders, which may confound the augmentation effect on G2.
- Targeted audit returned locally: `spectrum_kan_audit_3090_20260523_005715`, path `data/pure63_mainline/models/paper_main/augmentation_ablation/`.
- Audit scope: Spectrum-KAN only, p1/raw, no_aug/aug_no_mixup/composition_mixup, six tasks, five random folds.
- Local verification: every mode/variant has summary/detail/prediction CSVs; summaries have 7 rows, details 31 rows, predictions 4683 rows; logs and run metadata are present.
- Spectrum-KAN G2 audit macro-F1: no_aug p1 0.467, no_aug raw 0.514; aug_no_mixup p1 0.472, aug_no_mixup raw 0.472; composition_mixup p1 0.480, composition_mixup raw 0.588.
- `composition_mixup/p1` exactly reproduces original benchmark Spectrum-KAN p1 results for all six tasks, including G2=0.480.
- Limited audit takeaway: augmentation/mixup is not the sole cause of the Spectrum-KAN G2 anomaly; raw+composition improves G2 but remains far below classical ML.
- Full 4-DL G2 p1 macro-F1:
  - no_aug: 1D-CNN 0.649, 1D-ResNet 0.746, Spectrum-KAN 0.467, KAN-CNN 0.563.
  - aug_no_mixup: 1D-CNN 0.677, 1D-ResNet 0.769, Spectrum-KAN 0.472, KAN-CNN 0.647.
  - composition_mixup: 1D-CNN 0.710, 1D-ResNet 0.751, Spectrum-KAN 0.480, KAN-CNN 0.678.
- Full 4-DL G3 p1 macro-F1:
  - no_aug: 1D-CNN 0.167, 1D-ResNet 0.223, Spectrum-KAN 0.614, KAN-CNN 0.166.
  - aug_no_mixup: 1D-CNN 0.627, 1D-ResNet 0.787, Spectrum-KAN 0.866, KAN-CNN 0.677.
  - composition_mixup: 1D-CNN 0.745, 1D-ResNet 0.779, Spectrum-KAN 0.891, KAN-CNN 0.710.
- RamanNet-Lite p1 G2/G3 macro-F1:
  - no_aug: G2 0.580, G3 0.871.
  - aug_no_mixup: G2 0.767, G3 0.888.
  - composition_mixup: G2 0.767, G3 0.910.
- RamanNet-Lite raw G2/G3 macro-F1:
  - no_aug: G2 0.518, G3 0.779.
  - aug_no_mixup: G2 0.564, G3 0.756.
  - composition_mixup: G2 0.565, G3 0.739.
- Interpretation: the augmentation benefit is strongly task-dependent. It is large and cross-model for G3 in the original 4-DL set. For RamanNet-Lite, p1 is the effective operating point; ordinary augmentation rescues G2 from 0.580 to 0.767, while composition-constrained mixup mainly improves G3 and stabilizes G2 variance. Raw remains weak for RamanNet-Lite concentration grading.

## E014: DL Candidate Smoke Test For G2/G3

Status: returned and verified locally
Result path: `data/pure63_mainline/models/paper_main/dl_candidate_smoke/`
Paper role: DL-family representative screening

Notes:

- Run id: `candidate_smoke_after_full_3090_20260523_213809`.
- Scope: `RamanNet-Lite` and `PatchTransformer`, variants `p1` and `raw`, tasks `G2_mg_molar_grade` and `G3_mba_molar_grade`, mode `composition_mixup`.
- Local verification: `cv_results_summary_p1.csv` and `cv_results_summary_raw.csv` each have 5 rows; detail and prediction files are present.
- p1 macro-F1:
  - G2: RamanNet-Lite 0.767, PatchTransformer 0.653.
  - G3: RamanNet-Lite 0.910, PatchTransformer 0.830.
- raw macro-F1:
  - G2: RamanNet-Lite 0.565, PatchTransformer 0.220.
  - G3: RamanNet-Lite 0.739, PatchTransformer 0.168.
- Current interpretation: RamanNet-Lite is the best DL candidate found so far for balanced G2/G3 performance under p1+composition_mixup. PatchTransformer is not stable enough to prioritize.
- Candidate provenance: RamanNet-Lite is a lightweight local adaptation of the fixed-wavenumber RamanNet-style idea; PatchTransformer follows the spectral-patch attention idea but underperforms here.

## E015: RamanNet-Lite Full 13-Model Benchmark Extension

Status: returned and verified locally
Result path: `data/pure63_mainline/models/paper_main/benchmark_random_cv/`
Paper role: adds RamanNet-Lite into the full random-CV benchmark matrix as the strongest DL-family representative

Notes:

- Run id: `ramannet_lite_benchmark_3090_20260524_141543`.
- Scope: `RamanNet-Lite`, variants `raw`, `p1`, `p2`, `p3`, `p4`, `p5`, all six main tasks, five random folds.
- Local verification: every `cv_results_summary_*.csv` now has 78 rows with 6 RamanNet-Lite rows; every `cv_results_detail_*.csv` now has 390 rows with 30 RamanNet-Lite rows.
- `benchmark_full_matrix.csv` now has 468 rows = 13 models x 6 preprocessing variants x 6 tasks.
- Cross-task/cross-preprocess mean macro-F1: RamanNet-Lite 0.877, ranking above PLS-DA, Spectrum-KAN, 1D-ResNet, 1D-CNN, LDA, and KAN-CNN, but below SVM and tree/boosting models.
- RamanNet-Lite G2/G3 macro-F1 by preprocessing:
  - raw: G2 0.565, G3 0.739.
  - p1: G2 0.767, G3 0.910.
  - p2: G2 0.766, G3 0.893.
  - p3: G2 0.561, G3 0.888.
  - p4: G2 0.709, G3 0.765.
  - p5: G2 0.759, G3 0.836.
- Current interpretation: p1 is the best balanced RamanNet-Lite preprocessing choice; p2 is close for G2 but lower for G3. RamanNet-Lite is the strongest DL option for balanced G2/G3, but classical ML remains the accuracy leader.

## E016: RamanNet-Lite Augmentation Ablation Add-On

Status: returned and verified locally
Result path: `data/pure63_mainline/models/paper_main/augmentation_ablation/`
Paper role: completes the fair 5-DL augmentation ablation after RamanNet-Lite was added to the full benchmark

Notes:

- Run id: `ramannet_lite_ablation_3090_20260524_182624`.
- Scope: `RamanNet-Lite`, variants `p1` and `raw`, modes `no_aug`, `aug_no_mixup`, `composition_mixup`, all six main tasks, five random folds.
- Local verification: each mode/variant summary has 30 rows with 6 RamanNet-Lite rows; each detail file has 150 rows with 30 RamanNet-Lite rows; each prediction file has 23,410 rows with 4,682 RamanNet-Lite rows.
- p1 G2/G3 macro-F1:
  - no_aug: G2 0.580, G3 0.871.
  - aug_no_mixup: G2 0.767, G3 0.888.
  - composition_mixup: G2 0.767, G3 0.910.
- raw G2/G3 macro-F1:
  - no_aug: G2 0.518, G3 0.779.
  - aug_no_mixup: G2 0.564, G3 0.756.
  - composition_mixup: G2 0.565, G3 0.739.
- Current interpretation: RamanNet-Lite supports the reviewer-safe DL-representative role. It is the strongest balanced DL-family comparator under p1+composition_mixup, but it remains well below the classical ML leaders, especially ExtraTrees on G2. It should not be framed as the practical accuracy winner.

## E017: Optuna Mainline Skipped And ExtraTrees/p1 Locked

Status: decision recorded; downstream experiments pending
Result path: `data/pure63_mainline/models/paper_main/benchmark_random_cv/` as model-selection evidence
Paper role: final model-locking decision before SHAP and soil-only CV

Notes:

- Formal Optuna is skipped for the manuscript mainline.
- ExtraTrees/p1 is locked as the primary practical model for TreeSHAP and soil presence validation.
- The ExtraTrees implementation is a fixed project recipe (`n_estimators=400`, `min_samples_leaf=2`, `max_features='sqrt'`, `class_weight='balanced'`), not sklearn default and not Optuna-tuned.
- RamanNet-Lite remains the strongest DL-family representative, not a second star model.
- Optional future tuning-related work is limited to an ExtraTrees robustness/sensitivity check for SI if needed; it must not be used to relabel the headline benchmark result.

## E018: ExtraTrees/p1 Paper TreeSHAP

Status: returned and verified locally
Result path: `data/pure63_mainline/models/paper_main/shap_peak_cluster/`
Paper role: final peak-cluster interpretability evidence for the locked practical model

Notes:

- Remote run id: `paper_shap_extratrees_p1_20260525_010444`; match-only refresh run id: `paper_shap_match_only_p1_20260527_001443`.
- Scope: ExtraTrees/p1, all 901 pure spectra for training, tasks P1/P2/P3/G2/G3, G1 skipped.
- Script: `scripts/analysis/run_paper_shap.py`.
- Local verification: `shap_top20_per_task.csv` has 100 rows; `shap_cluster_summary.csv` has 24 clusters; `shap_mean_abs_by_wavenumber.csv` has 7005 rows; `shap_values_full.npz` is present.
- Matched cluster counts after the 2026-05-27 literature-peak refresh:
  - P1 Thiram presence: 4/7 clusters matched, partial but improved by Thiram 930/1510 windows.
  - P2 MG presence: 4/4 clusters matched, strong MG peak alignment at 1172/1220/1394/1616 windows.
  - P3 MBA presence: 2/2 clusters matched, strong MBA peak alignment at 1078/1590 windows.
  - G2 MG grade: 4/7 clusters matched, partial MG peak alignment including MG 1220; important unmatched clusters remain.
  - G3 MBA grade: 3/4 clusters matched, useful MBA peak support plus a mixed 1380/1394 region.
- Interpretation guardrail: write "partial peak-cluster consistency," not "SHAP proves competitive adsorption" or "all tasks are peak-driven."

## E019: Pure→Soil Transfer Diagnostic (FAILED — retained as motivation)

Status: returned and verified locally; result is DOMAIN SHIFT FAILURE
Result path: `data/pure63_mainline/models/paper_main/soil_validation/`
Paper role: motivation for soil-only CV approach (Discussion/Introduction), NOT the paper's soil result

Notes:

- Remote run id: `soil_validation_extratrees_p1_20260525_010159`.
- Scope: train ExtraTrees/p1 on all 901 pure spectra; predict 79 soil design spectra; tasks P1/P2/P3 presence only.
- Script: `scripts/analysis/run_soil_validation.py`.
- Soil metrics: P1 F1=0.161, P2/P3 balanced acc=0.5 (all-positive prediction).
- Conclusion: pure→soil transfer fails due to domain shift. This motivates the soil-only CV approach.
- This experiment is NOT the paper's soil result. The paper uses soil-only CV (E020, pending).

## E020: Soil-Only 5-Fold CV

Status: returned and verified locally
Result path: `data/pure63_mainline/models/paper_main/soil_validation/`
Paper role: §3.6 soil-matrix screening feasibility — the paper's main soil result

Notes:

- Run id: `soil_only_cv_extratrees_p1_20260527_002017`.
- Script: `scripts/analysis/run_soil_only_cv.py`.
- Strategy: 5-fold same-matrix CV with 79 soil design spectra plus 10 blank/control spectra included as negative samples.
- Model: ExtraTrees/p1, full spectrum 400-1800 cm⁻¹, fixed project recipe (`n_estimators=400`, `min_samples_leaf=2`, `max_features='sqrt'`, `class_weight='balanced'`).
- Primary metric: AUC. F1 and balanced accuracy use the fixed 0.5 threshold as auxiliary metrics; no Youden threshold is used in the main result.
- Formal CV results: P1 AUC 0.996±0.008, F1 0.887±0.087; P2 AUC 0.956±0.052, F1 0.811±0.127; P3 AUC 0.976±0.034, F1 0.919±0.085.
- OOF label-permutation test: 10000 permutations; P1/P2/P3 all p=0.0001.
- Blank specificity after full soil+blank training: P1/P2/P3 all 10/10 blanks predicted negative; max blank positive probabilities are 0.065, 0.061, and 0.085.
- Framing: "soil-matrix screening feasibility" / "same-matrix CV"; do not call this external validation or cross-matrix transfer success.

## E021: SHAP-Guided Spectral Masking

Status: returned and verified locally
Result path: `data/pure63_mainline/models/paper_main/shap_feature_selection/`
Paper role: §3.5 actionable interpretability — core contribution #4

Notes:

- Run id: `shap_feature_selection_extratrees_p1_20260527_001506`.
- Script: `scripts/analysis/run_shap_feature_selection.py`.
- Method: rank wavenumbers by ExtraTrees/p1 mean |SHAP| per task, retain top-k% regions, zero out remaining features, and retrain ExtraTrees on the locked random 5-fold split.
- Retention tested: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, and 100%.
- At 30% retention, macro-F1 remains within the success criterion for all five SHAP tasks: P1 0.994, P2 0.977, P3 0.983, G2 0.961, G3 0.979.
- Compared with 100% retention, the 30% feature masks show no practical degradation for P1/P2/G2/G3 and a small P3 decrease (0.988→0.983), all below the 5% threshold.
- Outputs include `shap_feature_elimination_curve.csv`, `shap_feature_elimination_folds.csv`, `shap_retained_wavenumbers.csv`, `shap_retention30_masks.csv`, and `shap_retention30_masks.npz`.

## E022: Progressive Feature Elimination Curve

Status: returned and verified locally
Result path: `data/pure63_mainline/models/paper_main/shap_feature_selection/`
Paper role: §3.5 Figure — performance-parsimony curve

Notes:

- Completed by the same run as E021: `shap_feature_selection_extratrees_p1_20260527_001506`.
- `shap_feature_elimination_curve.csv` contains 50 rows = 5 tasks × 10 retention levels.
- 30-40% retention is sufficient for the paper claim: P1 0.994/0.996, P2 0.977/0.980, P3 0.983/0.988, G2 0.961/0.966, G3 0.979/0.979.
- This supports an "actionable interpretability" figure, not an "optimal feature subset" claim.

## E023: Cross-Model SHAP Consensus (OPTIONAL)

Status: pending (lower priority than E021/E022)
Result path: `data/pure63_mainline/models/paper_main/shap_feature_selection/`
Paper role: §3.5 supplementary evidence — model-agnostic chemical signatures

Notes:

- Method: compute permutation importance (or TreeSHAP where available) for top 3 models (ExtraTrees, RF, XGBoost). Compare top-20% important wavenumber sets via Jaccard index.
- If Jaccard > 0.6 AND top regions align with known peaks → strong evidence of chemistry-driven decisions.
- Lower priority because ExtraTrees SHAP alone may be sufficient for the narrative.
- Can be deferred to revision if reviewer asks "is this model-specific?"

## E011: Paper-Main Data Quality Tables

Status: generated
Result path: `data/pure63_mainline/models/paper_main/data_quality/`
Paper role: RSD and peak-intensity QC

Notes:

- `peak_rsd_by_folder.csv` and `peak_rsd_summary.csv` exist.
- p1 and p3 have more defensible median RSD values than p5 for the selected peak-height windows.
- p5 should be treated as a derivative-feature candidate, not as the most repeatable raw intensity representation.

## E012: SHAP Peak-Cluster Preview

Status: generated from historical SHAP
Result path: `data/pure63_mainline/models/paper_main/shap_peak_cluster/`
Paper role: preview of cluster/window-level interpretability

Notes:

- Current preview is based on historical `05_shap_explainability/shap_top20_features.csv`.
- Cluster match rates remain modest: Thiram 0.43, MBA 0.40, MG presence 0.20, MG grade 0.17.
- Final SHAP should be regenerated after the paper-main model is locked.

## E013: Soil Metadata And Cache Preparation

Status: generated / needs QC
Result path: `data/pure63_mainline/models/paper_main/soil_validation/`
Paper role: soil data preparation for soil-only CV (E020)

Notes:

- Current parser detected 79 soil spectra across 10 folders.
- This differs from the approximate expected count of 90 and must be reconciled before final soil claims.
