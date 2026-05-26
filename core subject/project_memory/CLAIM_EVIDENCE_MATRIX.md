# Claim Evidence Matrix

## C001: Spectrum-Level Random CV Provides Strong Within-Dataset Screening Performance

Status: supported for the complete 13-model benchmark; figure/table rendering still needed

Evidence needed:

- `data/pure63_mainline/splits/cv_split_random_5fold.csv`
- `data/pure63_mainline/splits/cv_split_random_5fold_qc_by_fold.csv`
- `data/pure63_mainline/splits/cv_split_random_5fold_qc_by_stratum.csv`
- `data/pure63_mainline/models/paper_main/benchmark_random_cv/benchmark_best_by_task.csv`
- 5-fold random-CV mean and standard deviation

Evidence available:

- random split CSV and split QC files exist.
- a returned RTX 3090 smoke/profile subset exists under `paper_main/benchmark_random_cv/` for `p1` on RF, XGBoost, 1D-CNN, and Spectrum-KAN.
- first-round fast classical benchmark files have been returned and verified locally under `paper_main/benchmark_random_cv/`.
- verified coverage: RF, ExtraTrees, HistGradientBoosting, XGBoost, SVM, KNN, and PLS-DA across raw/p1/p2/p3/p4/p5, plus p1 smoke rows for 1D-CNN and Spectrum-KAN.
- completion run `benchmark_completion_3090_20260522` has been returned and verified locally.
- verified complete coverage after RamanNet-Lite extension: 13 models across raw/p1/p2/p3/p4/p5, six tasks, five random folds.
- each preprocessing variant has 390 fold-detail rows and 78 summary rows after RamanNet-Lite was appended.
- `benchmark_full_matrix.csv` contains 468 summary rows; `benchmark_best_by_task.csv` contains the six current best task rows.
- current best macro-F1 values: P1 0.996, P2 0.985, P3 0.994, G1 1.000, G2 0.966, G3 0.981.
- overall model mean ranking supports a classical-model narrative: ExtraTrees 0.971, XGBoost 0.969, RF 0.966 mean macro-F1 across the benchmark matrix; RamanNet-Lite is the strongest DL-family model at 0.877.

Evidence still needed:

- final figure/table generation from the returned benchmark matrix
- downstream ExtraTrees/p1 SHAP and soil-validation outputs

Allowed wording:

- conventional spectrum-level validation
- within-dataset screening benchmark

Forbidden wording:

- external validation
- leakage-free generalization
- validation on unseen sample preparations

## C002: Soil-Only CV Demonstrates Matrix-Screening Feasibility

Status: supported for same-matrix soil-screening feasibility; figure/table rendering still needed

Evidence needed:

- final figure/table rendering from the returned soil-validation files
- Optional: feature importance overlap between pure model and soil model

Evidence available:

- Formal script exists: `scripts/analysis/run_soil_only_cv.py`.
- Formal output path: `data/pure63_mainline/models/paper_main/soil_validation/`.
- Run id: `soil_only_cv_extratrees_p1_20260527_002017`.
- Scope: ExtraTrees/p1, full spectrum, 5-fold same-matrix CV, 79 design soil spectra plus 10 blank/control spectra included as negative samples.
- Primary metric is AUC; F1 and balanced accuracy use the fixed 0.5 threshold as auxiliary metrics.
- Soil-only CV summary:
  - P1 Thiram: AUC 0.996±0.008, F1 0.887±0.087, balanced accuracy 0.868±0.117.
  - P2 MG: AUC 0.956±0.052, F1 0.811±0.127, balanced accuracy 0.787±0.139.
  - P3 MBA: AUC 0.976±0.034, F1 0.919±0.085, balanced accuracy 0.892±0.109.
- OOF score label-permutation test with 10000 permutations:
  - P1 observed OOF_AUC 0.993, p=0.0001.
  - P2 observed OOF_AUC 0.955, p=0.0001.
  - P3 observed OOF_AUC 0.976, p=0.0001.
- Blank specificity after full soil+blank training:
  - P1: 10/10 blanks negative, max positive probability 0.065.
  - P2: 10/10 blanks negative, max positive probability 0.061.
  - P3: 10/10 blanks negative, max positive probability 0.085.

Historical context (pure→soil transfer, E019):

- Pure-trained ExtraTrees/p1 predicting soil: P1 F1=0.161, P2/P3 balanced acc=0.5
- This failure motivates the soil-only approach — domain shift is too large for direct transfer
- Retained as motivation in paper introduction/discussion, not as the soil result

Scope:

- Paper frames this as "soil-matrix screening feasibility" or "same-matrix CV"
- NOT "external validation" or "cross-matrix transfer success"
- Limitation: same-matrix CV, not cross-matrix generalization

## C003: Composition-Constrained Spectral Mixing Is a Chemistry-Grounded Augmentation

Status: supported as an implemented train-fold-only method with task-dependent DL benefit; do not overclaim universal improvement

Evidence needed:

- final figure/table rendering from the returned ablation files
- train-fold-only implementation evidence in Methods
- careful wording that the effect is task-dependent, not universal

Evidence available:

- implementation guard exists in `src/dataset.py`.
- DL wrappers pass train-fold composition keys from metadata.
- ablation script verified correct: `scripts/analysis/run_augmentation_ablation.py`.
- **Critical finding (2026-05-23)**: current benchmark DL results ALREADY include this augmentation (config.py defaults). The benchmark is NOT a no-aug baseline.
- Spectrum-KAN targeted audit returned locally under `paper_main/augmentation_ablation/` with run id `spectrum_kan_audit_3090_20260523_005715`.
- Targeted audit shows task-dependent benefit for Spectrum-KAN: on p1, G3 improves from 0.614 (no_aug) to 0.891 (composition_mixup), while G2 remains low at 0.467 -> 0.480. On raw, G2 improves from 0.514 to 0.588 but remains below classical ML.
- `composition_mixup/p1` exactly reproduces original benchmark Spectrum-KAN p1 results, confirming the audit is comparable to the benchmark configuration.
- Full 4-DL ablation returned and verified under `paper_main/augmentation_ablation/`, run id `full_dl_ablation_3090_20260523_171250`.
- RamanNet-Lite add-on ablation returned and verified under the same stage, run id `ramannet_lite_ablation_3090_20260524_182624`.
- Verified scope after the add-on: 5 DL models, p1/raw, no_aug/aug_no_mixup/composition_mixup, six tasks, five folds.
- Each mode/variant summary now has 30 rows with 6 RamanNet-Lite rows; each detail file has 150 rows with 30 RamanNet-Lite rows; each prediction file has 23,410 rows with 4,682 RamanNet-Lite rows.
- Strongest support is for G3 under p1:
  - 1D-CNN: 0.167 -> 0.745 from no_aug to composition_mixup.
  - 1D-ResNet: 0.223 -> 0.779.
  - KAN-CNN: 0.166 -> 0.710.
  - Spectrum-KAN: 0.614 -> 0.891.
- G2 support is modest and model-dependent:
  - 1D-CNN: 0.649 -> 0.710 from no_aug to composition_mixup.
  - KAN-CNN: 0.563 -> 0.678.
  - 1D-ResNet: 0.746 -> 0.751, effectively similar.
  - Spectrum-KAN: 0.467 -> 0.480, not rescued.
  - RamanNet-Lite: 0.580 -> 0.767 under p1; this gain is already achieved by ordinary augmentation, while composition-constrained mixup mainly reduces the fold-to-fold standard deviation (0.037 -> 0.021).
- RamanNet-Lite G3 under p1 improves from 0.871 (no_aug) to 0.888 (aug_no_mixup) and 0.910 (composition_mixup), supporting an additional composition-mixup benefit for MBA grade.
- RamanNet-Lite raw concentration grading remains weak: G2 0.518 -> 0.565 and G3 0.779 -> 0.739 from no_aug to composition_mixup.

Evidence still needed:

- final figure/table generation from the complete ablation matrix
- prose that explains why G2 and G3 respond differently

Scope:

- composition-constrained spectral mixing for ternary pesticide SERS.
- Do not claim first-ever linear spectral combination.
- Do not use the term "Beer-Lambert augmentation" in paper.
- Wording should emphasize "task-dependent gains, especially MBA grade classification, and RamanNet-Lite G2 stabilization under p1," not "augmentation universally improves DL."

## C006: RamanNet-Lite Is A Suitable DL-Family Representative

Status: supported for full benchmark inclusion and as a DL-family representative; formal optimization is skipped for the manuscript mainline

Evidence available:

- Candidate smoke run `candidate_smoke_after_full_3090_20260523_213809` returned and verified locally.
- Result path: `data/pure63_mainline/models/paper_main/dl_candidate_smoke/`.
- p1+composition_mixup macro-F1:
  - RamanNet-Lite: G2 0.767, G3 0.910.
  - PatchTransformer: G2 0.653, G3 0.830.
- RamanNet-Lite is more balanced than the existing DL baselines on the two concentration-grade tasks:
  - 1D-ResNet p1+composition_mixup: G2 0.751, G3 0.779.
  - Spectrum-KAN p1+composition_mixup: G2 0.480, G3 0.891.
- Full benchmark extension `ramannet_lite_benchmark_3090_20260524_141543` returned and verified locally under `data/pure63_mainline/models/paper_main/benchmark_random_cv/`.
- The benchmark matrix now has 468 rows = 13 models x 6 preprocessing variants x 6 tasks.
- RamanNet-Lite is the strongest DL-family model by cross-task/cross-preprocess mean macro-F1: 0.877.
- Full RamanNet-Lite G2/G3 macro-F1 by preprocessing:
  - raw: G2 0.565, G3 0.739.
  - p1: G2 0.767, G3 0.910.
  - p2: G2 0.766, G3 0.893.
  - p3: G2 0.561, G3 0.888.
  - p4: G2 0.709, G3 0.765.
  - p5: G2 0.759, G3 0.836.
- RamanNet-Lite p1/raw augmentation add-on `ramannet_lite_ablation_3090_20260524_182624` returned and is locally verified.
- Under p1, RamanNet-Lite G2/G3 macro-F1:
  - no_aug: G2 0.580, G3 0.871.
  - aug_no_mixup: G2 0.767, G3 0.888.
  - composition_mixup: G2 0.767, G3 0.910.
- Under raw, RamanNet-Lite G2/G3 macro-F1:
  - no_aug: G2 0.518, G3 0.779.
  - aug_no_mixup: G2 0.564, G3 0.756.
  - composition_mixup: G2 0.565, G3 0.739.

Evidence still needed:

- interpretability or peak-region analysis if it is discussed beyond performance.

Allowed wording:

- location-aware DL candidate tailored to fixed-wavenumber Raman/SERS spectra.
- strongest DL-family representative after smoke testing, full benchmark extension, and p1/raw augmentation add-on.
- representative DL-family comparator; optional sensitivity analysis only if explicitly needed for SI.

Forbidden wording:

- final best DL model before optimization.
- better than classical ML.
- first RamanNet-style model for SERS without a curated bibliography.
- proposed main model.
- star model or practical winner.

## C004: MG Grade Is the Hardest Semi-Quantitative Task

Status: partially supported

Evidence available:

- grouped G2 performance is weak.
- MG 1617 cm-1 c5/c6 separability is low.
- MBA suppresses MG peak intensity in current peak-statistics checks.
- formal RSD table exists in `paper_main/data_quality/`.

Evidence still needed:

- targeted MG peak separability table by concentration and MBA context
- optional high-vs-trace or within-one-level metric

## C005: SHAP/Interpretability Aligns With Chemistry at Peak-Cluster Level

Status: supported as partial-to-strong peak-cluster consistency depending on task; figure/table rendering still needed

Evidence needed:

- Final figure rendering from the returned SHAP tables
- Optional control windows or peak-ablation comparison if a stronger mechanism claim is desired

Evidence available:

- Final ExtraTrees/p1 TreeSHAP returned and verified under `paper_main/shap_peak_cluster/`.
- Run id: `paper_shap_extratrees_p1_20260525_010444`.
- Literature-peak match-only refresh run id: `paper_shap_match_only_p1_20260527_001443`.
- Outputs: top20 table 100 rows, cluster summary 24 rows, full mean|SHAP| 7005 rows, full SHAP npz present.
- Updated matched cluster counts: P1 4/7, P2 4/4, P3 2/2, G2 4/7, G3 3/4; total 17/24.
- P2 MG presence strongly aligns with MG 1172/1394/1616 windows.
- P3 MBA presence strongly aligns with MBA 1078/1590 windows.
- P1 Thiram: top cluster (918-926) matches Thiram_930 after update; 1510 also contributes; 768 cm⁻¹ remains unassigned.
- G2 MG grade: 1207-1216 cluster matches MG_1220 after update; 860 cm⁻¹ remains unassigned.
- G3 MBA grade has useful MBA peak support but also a mixed 1380/1394 region.

Risk:

- Current TreeSHAP supports chemistry-consistent attribution for several tasks, but does not justify strong MG-grade mechanism claims.
- Unassigned peaks (768, 860 cm⁻¹) should be framed as "candidate substrate-analyte interaction modes" in the paper.

## C007: SHAP-Guided Feature Selection Maintains Performance With Chemically Interpretable Regions Only

Status: supported for ExtraTrees/p1 SHAP-guided spectral masking; figure rendering still needed

Evidence needed:

- Overlay of retained features on reference SERS spectrum showing chemical correspondence

Evidence available:

- ExtraTrees/p1 TreeSHAP already computed: `shap_mean_abs_by_wavenumber.csv` (7005 rows = 5 tasks × 1401 wavenumbers)
- Script exists: `scripts/analysis/run_shap_feature_selection.py`.
- Output path: `data/pure63_mainline/models/paper_main/shap_feature_selection/`.
- Run id: `shap_feature_selection_extratrees_p1_20260527_001506`.
- Progressive elimination curve covers 5 tasks × 10 retention levels from 10% to 100%.
- At 30% retention, macro-F1 is P1 0.994, P2 0.977, P3 0.983, G2 0.961, G3 0.979.
- At 40% retention, macro-F1 is P1 0.996, P2 0.980, P3 0.988, G2 0.966, G3 0.979.
- At 100% retention, macro-F1 is P1 0.993, P2 0.978, P3 0.988, G2 0.966, G3 0.976.
- `shap_retention30_masks.npz` and `shap_retained_wavenumbers.csv` provide the retained-region artifacts for plotting.
- Peak-cluster matching shows high-SHAP regions correspond to known analyte peaks for P2/P3 and partially for P1/G2/G3.

Success criterion:

- Performance drop <5% when retaining only top 30-40% of features
- Retained features visually correspond to known Thiram/MG/MBA Raman peaks

Allowed wording:

- "SHAP-guided spectral region selection"
- "model performance maintained using only chemically interpretable spectral regions"
- "actionable interpretability" or "interpretability-driven feature engineering"

Forbidden wording:

- "novel feature selection algorithm" (we're applying existing SHAP-FS methodology)
- "optimal feature subset" (we're demonstrating chemical consistency, not optimizing)

Key citations:

- Marcilio & Wilson 2020 (IEEE): foundational SHAP-as-feature-selector
- shap-select (Kraev et al. 2024, arXiv): lightweight SHAP-based selection
- REFRESH (Sharma et al. 2023, AAAI/AIES): correlation-aware SHAP feature grouping
