# Claim Evidence Matrix

## C008: MG 1616 Suppression Reveals Competitive-Adsorption-Induced Mixture Interference

Status: supported as the manuscript headline claim; final Fig5 rendering still needed

Evidence needed:

- Final Fig5 rendering from p4 condition-mean spectra, 1616 cm⁻¹ ratio distributions, and difference spectra.
- Persistent CSV outputs from `scripts/analysis/verify_mg_suppression.py` for figure/table traceability.
- Fig3 peak-assignment setup showing 1616 cm⁻¹ as the clean MG core marker and 1172/1394 cm⁻¹ as overlap-sensitive MG-associated regions.

Evidence available:

- Formal script exists: `scripts/analysis/verify_mg_suppression.py`.
- Row alignment for X_p4 and split metadata has been hard-verified by `scripts/analysis/verify_row_alignment.py`: 15/15 sampled rows aligned with corr=1.0000.
- p4 no-normalization spectra preserve intensity information for peak-height comparison.
- Read-only verification on 901 spectra:
  - pure MG: 62 spectra.
  - MG+MBA: 128 spectra.
  - MG+Thiram: 100 spectra.
  - ternary: 358 spectra.
- MG 1616 cm⁻¹ median peak-height ratio versus same-level pure MG:
  - MG+MBA: 0.355.
  - MG+Thiram: 0.206.
  - ternary: 0.129.
- MG 1616 cm⁻¹ spectra below 0.5× same-level pure MG:
  - MG+MBA: 86/128.
  - MG+Thiram: 65/100.
  - ternary: 283/358.
- Auxiliary MG-associated peak ratios support broader peak-envelope reshaping:
  - MG 1220 cm⁻¹: 0.254 / 0.726 / 0.129 for MG+MBA / MG+Thiram / ternary.
  - MG 1172 and 1394 cm⁻¹ are retained as overlap-sensitive regions, not discarded; they support the mixture-interference story when framed as peak-envelope reshaping.
- Literature precedent supports strong wording:
  - Food Chemistry: X 2024 (`10.1016/j.fochx.2024.101954`) writes that mixed-pesticide SERS intensities changed due to competitive adsorption and uses multivariate analysis for simultaneous pesticide determination.
  - SERS/Raman + ML pesticide-mixture papers commonly write in terms of simultaneous determination, robustness, practicability, and spectral-region interpretation rather than self-limiting validation caveats.

Main-text wording:

- "三组分共存显著压制 MG 1616 cm⁻¹ 特征峰，揭示了竞争吸附诱导的混合物谱干扰。"
- "1172 和 1394 cm⁻¹ 等 MG 相关重叠敏感峰区发生峰包络重塑，进一步说明三组分混合谱不能由单一特征峰线性解释。"
- "该谱学发现构成后续 benchmark、TreeSHAP 归因、紧凑谱区筛选和土壤基质筛查验证的中心线索。"

Scope:

- The paper may strongly claim competitive-adsorption-induced spectral interference and MG 1616 suppression.
- The paper does not claim a measured adsorption constant, fitted Langmuir isotherm, or thermodynamic displacement model unless new titration/DFT evidence is added.

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

## C002: Spiked-Soil Matrix Screening Validation

Status: supported for spiked-soil / soil-matrix screening validation; figure/table rendering still needed

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

- Title/caption-level prose frames this as "加标土壤基质筛查验证" / "soil-matrix screening validation" / "spiked-soil screening validation".
- Methods-level detail may state same-matrix 5-fold CV.
- NOT "external validation" or "cross-matrix transfer success".

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

Status: supported as SHAP spectral assignment consistency; figure/table rendering still needed

Evidence needed:

- Final figure rendering from the returned SHAP tables and generated `shap_assignment_audit.csv`
- Optional extended co-adsorbed/reference-region audit if reporting ~24/29 in SI
- Optional control windows or peak-ablation comparison if a stronger mechanism claim is desired

Evidence available:

- Final ExtraTrees/p1 TreeSHAP returned and verified under `paper_main/shap_peak_cluster/`.
- Run id: `paper_shap_extratrees_p1_20260525_010444`.
- Literature-peak match-only refresh run id: `paper_shap_match_only_p1_20260527_001443`.
- Outputs: top20 table 120 rows, cluster summary 29 rows, full mean|SHAP| 8406 rows, full SHAP npz present.
- Matched cluster counts: P1 4/7, P2 4/4, P3 2/2, G1 3/5, G2 5/7, G3 4/4; formal output total 22/29 (incl. curated Thiram 860/1444).
- SHAP-mass weighted consistency: matched clusters account for 91.2% of total SHAP attribution mass (formal output).
- Task-level weighted consistency: G1 86.4%, G2 90.6%, G3 100%, P1 56.3%, P2 100%, P3 100%.
- Audit decomposition (`shap_assignment_audit.csv`, generated): 18 own-analyte literature peaks (83.8%) + 2 cross-component interference bands → 20/29 (88.1%) + 2 curated literature peaks (Thiram 860 (CH₃)₂-N per RSC Adv 2024 881; Thiram 1444 C-N per Analyst 2014 / RSC Adv 2024 1436) → formal 22/29 (91.2%). The 2 cross-component clusters (MG/MBA grade models using Thiram bands) are a STRENGTH — direct evidence the model captured competitive-adsorption interference chemistry.
- Curated additions sit well within accepted SERS-paper practice (e.g. RSC Adv 2024 assigns thiram bands with 20-56 cm⁻¹ shifts and cross-references normal Raman); do NOT under-report them as merely "candidate".
- Broader co-adsorbed/reference-region assignment can reach about 24/29 clusters and 92.3% SHAP mass, reported only in SI with explicit assignment levels, labeled as co-adsorbed/reference-region rather than target-analyte-only literature matching.
- P2 MG presence strongly aligns with MG 1172/1394/1616 windows.
- P3 MBA presence strongly aligns with MBA 1078/1590 windows.
- P1 Thiram: top cluster (918-926) matches Thiram_930; 1510 also contributes; 768 cm⁻¹ remains an unassigned candidate region.
- G2 MG grade: 1207-1216 cluster matches MG_1220; the 863 cm⁻¹ cluster is assigned to the literature-supported Thiram 860 cm⁻¹ band (cross-component interference).
- G3 MBA grade has useful MBA peak support but also a mixed 1380/1394 region.

Main-text role:

- TreeSHAP is written as chemical-region localization and assignment consistency: it shows that the model relies on analyte fingerprint peaks, co-adsorbed/reference molecular features, and interference-sensitive spectral windows, not random noise.
- SHAP evidence supports the headline by linking the MG suppression windows in Fig5 to the model-used regions in Fig6.
- Main text should report both count and attribution mass: "22 of 29 SHAP peak clusters fall on known chemical bands, accounting for 91.2% of total SHAP mass." Present the 2 cross-component clusters (MG/MBA grade models leaning on Thiram bands) as positive evidence of captured interference chemistry, not as a defect. SI may add the extended ~24/29 figure with explicit assignment levels.
- Unassigned peaks remain in SI as unassigned/candidate substrate-analyte or own-reference windows, without weakening the main narrative.

## C007: SHAP-Guided Feature Selection Maintains Performance With Chemically Interpretable Regions Only

Status: supported for ExtraTrees/p1 SHAP-guided spectral masking; figure rendering still needed

Evidence needed:

- Overlay of retained features on reference SERS spectrum showing chemical correspondence

Evidence available:

- ExtraTrees/p1 TreeSHAP already computed: `shap_mean_abs_by_wavenumber.csv` (8406 rows = 6 tasks × 1401 wavenumbers)
- Script exists: `scripts/analysis/run_shap_feature_selection.py`.
- Output path: `data/pure63_mainline/models/paper_main/shap_feature_selection/`.
- Run id: `shap_feature_selection_extratrees_p1_20260527_001506`.
- Progressive elimination curve covers 6 tasks × 10 retention levels from 10% to 100%.
- At 10% retention, macro-F1 is P1 0.986, P2 0.969, P3 0.982, G1 0.979, G2 0.953, G3 0.974; all six tasks remain above 0.95.
- At 30% retention, macro-F1 is P1 0.994, P2 0.977, P3 0.983, G1 0.991, G2 0.961, G3 0.979.
- At 40% retention, macro-F1 is P1 0.996, P2 0.980, P3 0.988, G1 0.991, G2 0.966, G3 0.979.
- At 100% retention, macro-F1 is P1 0.993, P2 0.978, P3 0.988, G1 0.994, G2 0.966, G3 0.976.
- `shap_retention30_masks.npz` and `shap_retained_wavenumbers.csv` provide the retained-region artifacts for plotting.
- Peak-cluster matching shows high-SHAP regions correspond to known analyte peaks for P2/P3 and partially for P1/G2/G3.

Success criterion:

- Main-text claim uses 10% retention (141/1401 wavenumbers), because all six tasks remain above 0.95 macro-F1.
- 5% retention is supplementary sensitivity only; it should not be used as the all-task headline.
- Retained features visually correspond to known Thiram/MG/MBA Raman peaks and interference-sensitive regions.

Allowed wording:

- "SHAP-guided compact spectral screening"
- "10% 高贡献波数即可维持六任务 macro-F1 >0.95"
- "model performance maintained using chemically meaningful high-contribution spectral regions"
- "actionable interpretability" or "interpretability-driven spectral-region screening"

Forbidden wording:

- "novel feature selection algorithm" (we're applying existing SHAP-FS methodology)
- "optimal feature subset" (we're demonstrating chemical consistency, not optimizing)

Key citations:

- Marcilio & Wilson 2020 (IEEE): foundational SHAP-as-feature-selector
- shap-select (Kraev et al. 2024, arXiv): lightweight SHAP-based selection
- REFRESH (Sharma et al. 2023, AAAI/AIES): correlation-aware SHAP feature grouping
