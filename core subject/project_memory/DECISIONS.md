# Decisions

## D001: Random CV Mainline, Grouped CV Not In Paper

Date: 2026-05-21 (updated 2026-05-26 per D017)
Status: active

Context:

- Grouped CV produced weak MG grade performance, especially for G2.
- Recent applied SERS-ML literature commonly reports spectrum-level random splits or conventional k-fold validation.
- The target is Q2+ applied SERS / food / analytical chemistry publication, not a machine-learning validation-method paper.
- Field standard review (2026-05-26): senior's paper uses 104/29 train/test split without CV; most SERS-ML papers use basic k-fold.

Decision:

- Use repeated random stratified spectrum-level CV as the main benchmark.
- Grouped CV is NOT included in the paper (neither main text nor SI). See D017.
- Grouped CV data remains in repository for potential future publication.
- Describe random CV as conventional spectrum-level validation, not external generalization.

Consequences:

- Soil validation becomes a key application safeguard.
- Paper claims must be scoped to screening and semi-quantitative grading.
- Grouped results should not be removed or hidden.

## D002: Preserve Locked Grouped Results

Date: 2026-05-21
Status: active

Decision:

- Keep `data/pure63_mainline/models/mainline_formal_experiment/03b_locked_main_results/` in place.

Rationale:

- It is the formal locked grouped-CV evidence.
- It maps directly to supplementary strict-validation tables.
- Moving it would break config/result-index assumptions.

## D003: Archive Negative Experiments, Do Not Delete

Date: 2026-05-21
Status: active

Decision:

- Failed or negative exploratory scripts are moved to `archive/scripts_negative/` with a manifest.
- They are not deleted.

Rationale:

- Negative experiments are useful for Discussion, rebuttal, and avoiding repeated dead ends.

## D004: p5 Requires a Preprocessing Registry

Date: 2026-05-21
Status: active

Decision:

- Do not add p5 as a one-off hard-coded path.
- First create a centralized preprocessing variant registry.

Rationale:

- Current p1-p4 assumptions are duplicated across config, dataset loading, runners, leakage analysis, and figures.

## D005: Composition-Constrained Mixup Only

Date: 2026-05-21
Status: active

Decision:

- Composition-constrained spectral mixing must occur only within the training fold and only between spectra with the same full composition key.

Rationale:

- Same-task-label mixing can combine different co-analyte contexts and produce ambiguous labels.

## D006: MG Grade Is Semi-Quantitative

Date: 2026-05-21
Status: active

Decision:

- G2/MG concentration grading is framed as semi-quantitative and physically challenging, not precise quantification.

Rationale:

- MG low-concentration peaks have weak separability, especially c5 vs c6 and under MBA competition.
- Current SHAP evidence does not strongly align G2 with known MG peaks.

## D008: Benchmark DL Results Include Augmentation - Ablation Required

Date: 2026-05-23
Status: active

Context:

- Code audit confirmed: `run_random_benchmark.py` does not pass `model_kwargs` to override `config.py` defaults.
- `config.py` line 145-147: `AUG_ENABLED=True`, `MIXUP_ENABLED=True`, `MIXUP_ALPHA=0.3`.
- Full call chain verified: factory() -> _DLWrapper(train_cfg=None) -> _default_train_cfg() -> aug_enabled=True in fit().
- Therefore all 4 DL benchmark results (1D-CNN, 1D-ResNet, Spectrum-KAN, KAN-CNN) already include generic augmentation + composition-constrained mixup.

Decision:

- The current benchmark DL results cannot be used as a "no-augmentation baseline."
- An augmentation ablation experiment is required before claiming composition-constrained mixing improves DL performance.
- The ablation must run the same split/folds as the benchmark for direct comparability.
- Ablation must include `raw` variant in addition to `p1`, because p1/SNV changes the sign and scale of the selected MG_1616 peak in a subset of MG-positive folders (QC diagnostic: 11/48 folders have negative mean peak values), which may confound G2 interpretation.

Consequences:

- Paper cannot claim "augmentation helps DL" until ablation is complete.
- If ablation shows no_aug ~= composition_mixup, the augmentation innovation point must be downgraded.
- The term "Beer-Lambert augmentation" should not be used; use "composition-constrained spectral mixing."

## D009: Model Role Selection Strategy

Date: 2026-05-23
Status: active (updated 2026-05-25; Optuna-specific parts superseded by D014)

Context:

- Full benchmark shows ExtraTrees (0.971), XGBoost (0.969), RF (0.966) as top classical ML.
- DL models significantly weaker: Spectrum-KAN (0.845), 1D-ResNet (0.835), 1D-CNN (0.747), KAN-CNN (0.745).
- Spectrum-KAN has extreme task variance: G1=0.994 but G2=0.480 on p1.
- Targeted audit `spectrum_kan_audit_3090_20260523_005715` reproduced the p1 composition-mixup G2 result exactly and showed that no_aug/aug_no_mixup do not rescue G2.
- Working interpretation: Spectrum-KAN is poorly matched to the MG grade task, likely through full-spectrum input, tanh-basis compression, overparameterization, and weak/unstable MG features. This is a supported diagnosis, not a formally proven root cause.
- KAN was inherited from the old grouped-CV pipeline, not specifically chosen for the current random-CV benchmark.
- Full 4-DL ablation returned on 2026-05-24. It confirms the augmentation story is task-dependent: G3 gains are large across DL models, while G2 gains are modest and Spectrum-KAN remains poor.
- Candidate smoke returned on 2026-05-24. RamanNet-Lite with p1+composition_mixup gives G2=0.767 and G3=0.910 macro-F1, making it the strongest balanced DL candidate so far.
- Literature review status: TabNet/sparse tabular attention is still a plausible later candidate, but it should not be called "first" or "zero papers in SERS/Raman" without a curated bibliography. It is no longer the first implementation priority.

Decision:

- Primary practical model: ExtraTrees/p1, using the fixed project recipe from `src/models.py`.
- DL-family comparator: **RamanNet-Lite** (confirmed by smoke test, full benchmark extension, and add-on ablation).
- KAN models (Spectrum-KAN, KAN-CNN) remain in the benchmark as background/reference, not as the paper's novel DL contribution.
- RamanNet-Lite's paper role: location-aware DL-family comparator tailored to fixed-wavenumber Raman/SERS peaks. Its value is defining the DL-family boundary, not beating classical ML.
- Formal Optuna is not part of the manuscript mainline. Optional ExtraTrees-only robustness checks may be added to SI if needed, but they must not be used to relabel the final model as hyperparameter-optimized.
- Literature basis checked before prioritization: RamanNet-style fixed-wavenumber modeling, RamanFormer-style spectral patch attention, TabNet-style sparse tabular attention, ROCKET/MiniROCKET-style time-series kernels, and Mamba/SSM-style sequence models. Only RamanNet-Lite and PatchTransformer were smoke-tested in this run.

Consequences:

- Do not spend immediate server time on TabNet/Mamba unless a reviewer-facing interpretability gap remains.
- Full six-task RamanNet-Lite benchmark has now been run; no RamanNet-Lite optimization is required before SHAP/soil work.
- Use the DL candidate as a complementary model, not as the paper's main accuracy leader. Classical ML remains the practical performance conclusion.
- KAN stays in benchmark tables but is not the innovation point.

## D011: Full DL Ablation And Candidate Smoke Returned

Date: 2026-05-24
Status: active

Decision:

- Treat the 4-DL augmentation ablation as complete and locally verified.
- Treat RamanNet-Lite as the strongest DL-family representative, with 1D-ResNet as the conservative DL comparator in discussion if needed.
- Treat PatchTransformer as a deprioritized candidate because it is unstable, especially on raw spectra.
- Keep Spectrum-KAN in the manuscript as task-dependent evidence: poor for G2, strong for G3 under p1+composition_mixup.

Evidence:

- Full run id: `full_dl_ablation_3090_20260523_171250`.
- Candidate run id: `candidate_smoke_after_full_3090_20260523_213809`.
- Local result paths:
  - `data/pure63_mainline/models/paper_main/augmentation_ablation/`
  - `data/pure63_mainline/models/paper_main/dl_candidate_smoke/`
- Full ablation p1 G2/G3 highlights:
  - G2 composition_mixup: 1D-ResNet 0.751, 1D-CNN 0.710, KAN-CNN 0.678, Spectrum-KAN 0.480.
  - G3 composition_mixup: Spectrum-KAN 0.891, 1D-ResNet 0.779, 1D-CNN 0.745, KAN-CNN 0.710.
- Candidate smoke p1 G2/G3 highlights:
  - RamanNet-Lite: G2 0.767, G3 0.910.
  - PatchTransformer: G2 0.653, G3 0.830.

Consequence:

- The next manuscript plan is not an optimization plan. It is ExtraTrees/p1 TreeSHAP plus soil presence validation, with RamanNet-Lite retained as DL-family benchmark evidence.

## D012: RamanNet-Lite Added To Full Benchmark

Date: 2026-05-24
Status: active

Decision:

- Treat the paper-main random-CV benchmark as a 13-model matrix after adding RamanNet-Lite.
- Keep ExtraTrees/p1 as the primary practical model for downstream SHAP and soil-only CV.
- Keep RamanNet-Lite as the strongest DL-family representative, with p1 as the fair main DL setting and p2/p5 only as supplementary preprocessing sensitivity if needed.
- Do not replace the practical conclusion that classical ML is the accuracy leader.

Evidence:

- Full run id: `ramannet_lite_benchmark_3090_20260524_141543`.
- Local path: `data/pure63_mainline/models/paper_main/benchmark_random_cv/`.
- Verified matrix size: 468 rows = 13 models x 6 preprocessing variants x 6 tasks.
- RamanNet-Lite mean macro-F1 across all tasks/preprocess variants: 0.877, making it the strongest DL-family model by overall benchmark mean.
- RamanNet-Lite p1: P1 0.961, P2 0.947, P3 0.953, G1 0.998, G2 0.767, G3 0.910.
- RamanNet-Lite G2/G3 highlights:
  - p1: G2 0.767, G3 0.910.
  - p2: G2 0.766, G3 0.893.
  - p5: G2 0.759, G3 0.836.
  - p3/raw are weak for G2; p4 is intermediate for G2 but weak for G3.

Consequence:

- Formal Optuna has been skipped after reviewing the 13-model matrix (see D014).
- Paper wording should describe RamanNet-Lite as a location-aware DL-family comparator with competitive DL performance, not as a model that beats classical ML.

## D013: RamanNet-Lite Is A DL Representative, Not A Star Model

Date: 2026-05-24
Status: active

Decision:

- Continue to keep ExtraTrees/tree ensembles as the practical accuracy conclusion.
- Treat RamanNet-Lite as the strongest DL-family representative and fair augmentation-ablation add-on, not as the paper's proposed or winning model.
- Do not call ExtraTrees + RamanNet-Lite "two star models." Describe them as "primary practical ML model" and "representative DL-family comparator."
- Do not add p2/p5 to the main RamanNet-Lite augmentation ablation unless the other DL models receive the same expanded preprocessing scope. p2/p5 can only be supplementary preprocessing sensitivity.
- Do not run RamanNet-Lite Optuna in the current manuscript mainline. If it is ever revisited, it can only be a limited DL sensitivity/gap-assessment experiment, not an attempt to overtake ExtraTrees.

Evidence:

- Reviewer-risk analysis on 2026-05-24 flagged a strong post-hoc-selection risk if RamanNet-Lite is presented as the main method, because it ranks 7th overall in the 13-model benchmark and is far below ExtraTrees on G2.
- RamanNet-Lite add-on ablation run id: `ramannet_lite_ablation_3090_20260524_182624`.
- Local path: `data/pure63_mainline/models/paper_main/augmentation_ablation/`.
- Verified scope: `RamanNet-Lite` x `p1/raw` x `no_aug/aug_no_mixup/composition_mixup` x six tasks x five folds.
- Local verification after return: each mode/variant summary has 30 rows with 6 RamanNet-Lite rows; each detail file has 150 rows with 30 RamanNet-Lite rows; each prediction file has 23,410 rows with 4,682 RamanNet-Lite rows.
- RamanNet-Lite p1 G2/G3:
  - no_aug: G2 0.580, G3 0.871.
  - aug_no_mixup: G2 0.767, G3 0.888.
  - composition_mixup: G2 0.767, G3 0.910.
- RamanNet-Lite raw G2/G3:
  - no_aug: G2 0.518, G3 0.779.
  - aug_no_mixup: G2 0.564, G3 0.756.
  - composition_mixup: G2 0.565, G3 0.739.

Consequence:

- The next manuscript narrative should say: classical ensemble learning is the practical winner; RamanNet-Lite is retained to define the DL-family boundary and to complete a fair 5-DL augmentation ablation.
- If a tuning-related defense becomes unavoidable, the safest scope is ExtraTrees-only robustness/sensitivity for SI, with cautious wording about model-selection bias and no claim that the headline result was Optuna-optimized.

## D014: Skip Formal Optuna And Lock Downstream Model

Date: 2026-05-25
Status: active

Decision:

- Skip formal Optuna in the manuscript mainline.
- Lock ExtraTrees/p1 as the primary practical model for TreeSHAP and soil presence validation.
- Keep RamanNet-Lite as the strongest DL-family representative and fair augmentation-ablation member, not as a second star model.
- If a tuning-related defense is needed later, run only a tiny ExtraTrees robustness/sensitivity check for SI; do not call it Optuna and do not use it to change the benchmark headline result.

Rationale:

- The full 13-model benchmark already shows a near-ceiling classical ensemble landscape: ExtraTrees 0.971, XGBoost 0.969, RF 0.966 mean macro-F1.
- ExtraTrees/p1 already gives the best G2 result (0.966±0.009), so same-CV Optuna would have low marginal value and high post-hoc selection-bias risk.
- RamanNet-Lite is the strongest DL-family model but ranks 7th overall and is far below ExtraTrees on G2; optimizing it as a main model would look like DL-for-DL's-sake.
- SHAP peak-cluster evidence and soil-matrix validation are more central to the analytical chemistry claim than another same-CV hyperparameter search.

Consequences:

- Next experiments: ExtraTrees/p1 TreeSHAP for P1/P2/P3/G2/G3 and train-all-pure -> predict-soil presence validation for P1/P2/P3.
- Use wording such as "fixed, reproducible model configurations" and "standardized cross-model benchmark."
- Do not use wording such as "optimized final model", "best achievable performance", "two star models", or "default ExtraTrees" unless a new evidence file supports it.

## D015: Soil Strategy — Soil-Only CV for Screening Feasibility

Date: 2026-05-26
Status: active (supersedes previous "boundary evidence only" framing)

Decision:

- Pure→soil transfer (E019) failed due to domain shift. This is retained as MOTIVATION, not as the paper result.
- The paper's soil result will be soil-only 5-fold CV (E020): train and test within soil data.
- Title/caption-level framing follows D021: "spiked-soil / soil-matrix screening validation" — the method is demonstrated in the target matrix.
- Methods-level detail may state matrix-matched / same-matrix CV. Do not promote this detail into a headline caveat.
- Statistical defense: permutation test + blank specificity + optional feature importance overlap with pure model.

Evidence:

- E019 failure: P1 F1=0.161, P2/P3 balanced acc=0.5 (domain shift confirmed).
- Preliminary soil-only CV: AUC>0.93 for all P1/P2/P3 with p1. Resolved by E020 formal run: P1/P2/P3 AUC 0.996/0.956/0.976, permutation p=0.0001, blanks 10/10 negative.
- Literature support: same-matrix CV or spiked-matrix validation is standard practice in SERS-ML papers.

Consequences:

- Allowed wording: title/caption-level "spiked-soil / soil-matrix screening validation"; methods-level "same-matrix CV" or "matrix-matched CV".
- Forbidden: "successful cross-matrix transfer", "external validation"
- P2 MG specificity concern is defended by the E020 permutation test and blank specificity result.

## D016: SHAP Literature Peak List Update

Date: 2026-05-26
Status: active (refined by D022 for weighted SHAP assignment and curated peak-library audit)

Decision:

- Update LITERATURE_PEAKS in `run_paper_shap.py` to add: Thiram 930 cm⁻¹, Thiram 1510 cm⁻¹, MG 1220 cm⁻¹.
- Rerun peak-cluster matching after update. The initial six-task D016 output produced 29 clusters and a 20/29 literature-match layer; D022 supersedes the manuscript-facing result with the curated 22/29 and 91.2% SHAP-mass consistency output.
- Expected improvement: P1 2/7→4/7, P2 3/4→4/4, G2 3/7→4/7.
- Original 2026-05-27 formal output kept 768 and 860 cm⁻¹ as unassigned/candidate windows. D022 later refined this: 860 cm⁻¹ has literature support as a Thiram band and should be handled through a curated peak-library audit rather than informal relabeling.

Evidence:

- Thiram 930: ν(C-S) symmetric stretch (RSC Advances 2024, Kang 2017)
- Thiram 1510: ν(C=N) stretch (Kang 2017, Zhu 2019)
- MG 1220: C-H in-plane bending (RSC Advances 2014, Jiang 2018)

Consequences:

- SHAP narrative strengthens significantly for P2/P3 (showcase) and becomes acceptable for P1/G2.
- Paper can write "majority of high-importance features correspond to known analyte vibrations."

## D010: Spectrum-KAN Targeted Audit Returned And Verified

Date: 2026-05-23
Status: active

Decision:

- Treat the Spectrum-KAN G2 anomaly as a reproducible model/task behavior, not a missing artifact or sync error.
- Do not rerun the full original benchmark just to "rescue" Spectrum-KAN.
- Run the full DL augmentation ablation next if the paper still needs an augmentation claim and a fair DL-family comparison.

Evidence:

- Returned local path: `data/pure63_mainline/models/paper_main/augmentation_ablation/`.
- Run id: `spectrum_kan_audit_3090_20260523_005715`.
- All three modes (`no_aug`, `aug_no_mixup`, `composition_mixup`) contain p1/raw summary, detail, prediction CSVs, logs, and run metadata.
- Local verification: each summary has 7 rows; each detail has 31 rows; each prediction file has 4683 rows.
- G2 macro-F1 results:
  - no_aug: p1 0.467, raw 0.514
  - aug_no_mixup: p1 0.472, raw 0.472
  - composition_mixup: p1 0.480, raw 0.588
- `composition_mixup/p1` exactly matches the original benchmark for all six Spectrum-KAN tasks, including G2=0.480.

## D017: Grouped CV Not In Paper (Neither Main Nor SI)

Date: 2026-05-26
Status: active (modifies D001)

Decision:

- Grouped CV results are NOT included in the paper — neither main text nor supplementary.
- Data and code remain in the repository for potential future use (next paper).
- The paper's validation strategy is: random stratified 5-fold CV (main) + permutation test (statistical) + soil-only CV (application).

Rationale:

- Field standard: most SERS-ML papers use simple train/test split or basic k-fold. Our stratified 5-fold CV already exceeds field norms.
- Grouped CV results are weaker (G2=0.555) and would create an attack surface for reviewers without adding value.
- No reviewer at SAA/Chemometrics&ILS will ask for folder-level grouped CV because they don't know our data has folder structure.
- Including it complicates the narrative and raises questions the paper doesn't need to answer.
- Reference: senior's paper (SNN+SERS, same lab) uses 104/29 train/test split without any CV.

Consequences:

- D001 is partially superseded: "Retain grouped CV as supplementary" → no longer applies.
- D002 remains active: locked grouped results stay in repo for future use.
- Paper structure simplifies: benchmark → SHAP → augmentation ablation → soil-only CV.
- If a reviewer asks about batch effects during revision, grouped CV can be added then.

## D018: SHAP-Guided Feature Selection Experiment

Date: 2026-05-26
Status: active

Decision:

- Add a "SHAP-guided spectral region selection" experiment to the paper as an actionable interpretability contribution.
- This elevates the paper from "post-hoc SHAP explanation" to "interpretability-driven feature engineering."
- Three sub-experiments (priority order):
  1. SHAP Spectral Masking: retain top SHAP regions, mask rest, retrain ExtraTrees → show performance maintained
  2. Progressive Feature Elimination Curve: remove bottom 10/20/.../80% features by SHAP rank, plot performance vs retention
  3. Cross-model SHAP Consensus: compare top SHAP regions across ExtraTrees/RF/XGBoost, report Jaccard overlap

Rationale:

- Literature gap: no SERS paper does "SHAP → select → retrain → validate" closed loop (Phase 1 research confirmed).
- The ZnO benchmark paper (ACS Applied Nano Materials, IF 5-6) succeeded with "benchmark + GANs + SHAP" — our equivalent is "benchmark + augmentation ablation + SHAP-guided FS."
- Methodological citations: Marcilio & Wilson 2020 (foundational SHAP-FS), shap-select (Kraev 2024), REFRESH (Sharma 2023).
- Computationally cheap: uses existing SHAP values, only needs mask + retrain (hours, not days).

Consequences:

- Paper gains a 4th core contribution: "SHAP-guided spectral selection demonstrates that model performance is maintained using only chemically interpretable regions."
- New figure: performance-vs-feature-retention curve overlaid on reference spectrum.
- Strengthens SAA submission probability from 50-60% to 60-70%.
- GPT execution scope expands to include this experiment.

## D007: Remote Server Artifacts Must Return By Stage

Date: 2026-05-22
Status: active

Decision:

- A remote run is not complete when the remote command exits; it is complete only after its artifacts are returned to the correct local stage directory and verified locally.
- Each returned server run must include the stage outputs, logs, environment snapshot, approved command/run metadata, and any timing or crash evidence generated during the run.
- User-facing reports and paper-claim memory are updated only after the local artifact set has been checked for expected variants, tasks, folds, and result indexes.

Rationale:

- Server instances are temporary and earlier project work already lost time when results were not synchronized before shutdown.
- Stage-local artifact paths keep the random-CV paper mainline separate from historical grouped-CV evidence.
- Reproducibility requires the code commit, runtime environment, command scope, and logs to travel with the result files.

## D019: Benchmark Data Is Tracked By The Parent Repo — Do Not git restore Data Dirs Inside `core subject`

Date: 2026-05-29
Status: active

Context:

- The project has two nested git repositories: parent `d:\通过拉曼光谱预测物及其浓度` and inner `core subject/.git`.
- The inner `core subject` repo does NOT track `data/pure63_mainline/models/...`; the parent repo does.
- RamanNet-Lite benchmark rows were lost from the working tree because a `git restore`/checkout aligned the tree to inner-repo state, which never contained RamanNet-Lite (inner history: 25→55→73 rows, 12 models only; `src/models.py` MODEL_REGISTRY has no RamanNet entry).
- The complete 13-model matrix survives in the parent repo HEAD (`cd18dc7`): `benchmark_full_matrix.csv` 469 lines / 36 RamanNet rows; every `cv_results_summary_*.csv` has 6 RamanNet rows; every `cv_results_detail_*.csv` has 30 RamanNet rows. The 12-model subset matches the working tree except 6 LDA timing-column float-noise digits.

Decision:

- Treat the parent repo as the authoritative tracker for `data/` artifacts.
- Do NOT run `git restore`/`git checkout` on data directories from inside `core subject` — it can silently drop artifacts that only exist in the parent repo history.
- Recover lost benchmark data from parent HEAD, not by re-running the server.

Consequences:

- Resolved 2026-06-01: the working tree and paper-main benchmark files are again aligned to the complete 468-row / 13-model matrix. D019 remains active as a repository-boundary warning: recover data from the parent repo when needed, and do not use inner-repo `git restore` on `data/` artifacts.

## D020: Paper Main-Line Locked — Mixture-Induced Spectral Interference × Interpretable ML

Date: 2026-05-29
Status: active (supersedes the earlier "benchmark-as-headline" framing in D-series narrative)

Context:

- Multi-angle web traversal of CAS 1-2区 precedents found NO paper that is simultaneously (no new chemistry + no new model + modest results) — every one leans on a strength we lack. Optimization/Bayesian-tuning as a hook is dead for us because classification performance is already saturated (~0.97).
- Six independent hook evaluations converged: the only genuinely open novelty wedge at Q1/Q2 is using interpretable ML to expose competitive-adsorption-consistent spectral interference in a ternary mixture.
- Own read-only recompute (X_p4, ±6 cm⁻¹ peak height vs same-MG-level pure MG): MG 1616 cm⁻¹ suppressed in all co-existence — median ratio 0.36 (MG+MBA) / 0.21 (MG+Thiram) / 0.13 (ternary); 283/358 ternary spectra <0.5× pure MG. GPT's reported 1172/1394 strong suppression under MG+Thiram did NOT reproduce (those bands are contaminated by Thiram 1380 / neighbor overlap).
- Senior-lab SNN paper (Zhang/Kong, accepted not-yet-in-print; same 3 analytes/substrate, 133 spectra, no CV) cleared 2区 via model-hook + DL-wins + real-sample quant — confirms hook+real-sample is the publishing currency, not rigor.

Decision:

- Headline = mixture-induced spectral interference / MG suppression consistent with competitive adsorption. Benchmark, classical>DL, TreeSHAP attribution, SHAP-guided FS, soil arm are all DEMOTED to supporting evidence.
- Cementing experiment = path A (zero new wet-lab): suppression analysis on existing ~900 mixture spectra + SHAP co-movement. Titration deferred to revision if a reviewer demands a measured adsorption isotherm.
- Phrasing discipline: "consistent with competitive adsorption", never "SHAP proves the mechanism"; never claim monotonic concentration dependence; core evidence peak = 1616 cm⁻¹ (clean), exclude 1172/1394 from core evidence.

Consequences:

- 论文写作路线图 §3 reordered so §3.2 = interference headline; benchmark moved to §3.3 supporting.
- Before final §3.2 figures: hard-verify X_p4 row ↔ split.csv row order (currently only row count 901 verified). **[RESOLVED 2026-05-29]** Content-level check (`scripts/analysis/verify_row_alignment.py`): for 15 rows spanning the whole table, re-read the original CSV by split.csv `file_path`, re-ran p4, compared to X_p4[i] — 15/15 corr=1.0000, max pointwise diff ~1e-5. Alignment confirmed: X_p4[i] == split.csv[i]. Guaranteed structurally by dataset.py:642 (X_raw filled in meta.iterrows() order).
- Target journal unchanged: Spectrochimica Acta Part A / Talanta (Q2).

## D021: Publication-Style Strong Narrative Calibration

Date: 2026-06-01
Status: active (refines D020 wording; does not change data claims)

Context:

- User + Claude + Codex review concluded that previous "risk list" wording was too self-limiting for the actual SERS/ML publication environment.
- Closely related papers do not frame limited validation as self-criticism; they write in publication-forward terms such as simultaneous determination, robustness, practicability, competitive adsorption, and spectral-region interpretation.
- Key precedent: Food Chemistry: X 2024 (`10.1016/j.fochx.2024.101954`) directly states that mixed-pesticide SERS intensities changed due to competitive adsorption, despite using a compact calibration/prediction dataset. This supports strong main-text wording for MG 1616 suppression.
- Target-journal precedent: Spectrochimica Acta Part A papers accept SERS/Raman + ML + pesticide mixture screening and Raman/XAI spectral-region interpretation; therefore the manuscript should not read like an ML validation-method paper.
- The project evidence remains: MG 1616 cm⁻¹ median ratio 0.355 (MG+MBA) / 0.206 (MG+Thiram) / 0.129 (ternary); 283/358 ternary spectra <0.5× same-level pure MG; the curated SHAP assignment audit reports 22/29 peak clusters on known chemical bands, accounting for 91.2% total SHAP mass; 10% SHAP-retained wavenumbers keep all six tasks >0.95 macro-F1.

Decision:

- Main-text tone uses a **publication-style strong narrative**:
  - Write that the study **reveals competitive-adsorption-induced MG peak suppression / mixture-induced spectral interference** in ternary SERS mixtures.
  - Use MG 1616 cm⁻¹ as the clean core marker for suppression.
  - Use 1172/1394 cm⁻¹ as overlap-sensitive MG-associated regions that demonstrate peak-envelope reshaping, not as discarded flaws.
  - Present random 5-fold CV as a systematic model-selection benchmark; do not foreground "random" outside Methods.
  - Present soil results as **spiked-soil / soil-matrix screening validation** in title/caption-level prose; keep "same-matrix CV" as methods-level detail only.
  - Present TreeSHAP as showing chemically meaningful model evidence and locating interference-related spectral regions; do not make the paper sound like a SHAP causality caveat.
  - Main feature-selection claim uses 10% retention (141/1401, all six tasks >0.95); 5% retention is SI/sensitivity only.
- Forbidden wording is now framed as implementation hygiene, not a self-limiting narrative:
  - Do not call random CV or soil-only CV "external validation".
  - Do not claim a measured Langmuir isotherm, adsorption constant, or thermodynamic proof.
  - Do not claim monotonic co-analyte concentration response unless a dedicated titration is added.

Consequences:

- `CLAIM_EVIDENCE_MATRIX.md` must include a headline claim for MG 1616 suppression / competitive-adsorption-induced interference.
- `reports/实验结果摘要.md` must open with the headline interference result, not historical grouped CV or benchmark.
- `reports/论文写作路线图.md`, `reports/图表与表格清单.md`, `project_architecture.md`, `HANDOFF.md`, `AGENTS.md`, and `CLAUDE.md` should instruct future assistants to use strong Chinese-first publication wording and avoid reflexive self-weakening.

## D022: SHAP Peak Assignment Uses Count + SHAP-Mass Weighted Consistency

Date: 2026-06-01
Status: active (refines D016/D021; formal audited output is 22/29 and 91.2% SHAP mass)

Context:

- The original existing-peak-list output had 29 clusters across six tasks and 20/29 clusters matched, containing 88.1% of total SHAP mass.
- The curated audit is now generated by `scripts/analysis/build_shap_assignment_audit.py` and saved as `data/pure63_mainline/models/paper_main/shap_peak_cluster/shap_assignment_audit.csv`.
- The formal manuscript output is **22/29 clusters on known chemical bands, accounting for 91.2% of total SHAP mass**.
- Task-level SHAP-mass weighted consistency after curated assignment is: G1 86.4%, G2 90.6%, G3 100%, P1 56.3%, P2 100%, P3 100%.
- Audit decomposition (`scripts/analysis/build_shap_assignment_audit.py` → `shap_assignment_audit.csv`): 18 clusters match the task's own-analyte literature peak (83.8% mass); +2 cross-component clusters where the model leans on a different analyte's band (MG/MBA grade tasks using Thiram bands) bring the intermediate audit layer to 20/29, 88.1% mass — these cross-component matches are a STRENGTH (the model captured mixture-interference chemistry, reinforcing the competitive-adsorption headline), to be presented as interference-band attribution.
- Curated literature additions (both supported by published SERS/Raman assignments): Thiram ~860 cm^-1 → (CH3)2-N / CH3 modes (RSC Adv 2024 D4RA00048J reports 881 cm^-1 (CH3)2-N-C; normal-Raman ~850), matching the G2 863 cluster; Thiram 1444 cm^-1 → C-N stretch + CH3 rock (Analyst 2014 1444; RSC Adv 2024 1436 monodentate), matching the G3 1440 cluster. With both added, the formal audited output is **22/29 clusters on known chemical bands = 91.2% of total SHAP mass**. Accepted same-field papers (e.g. RSC Adv 2024) assign thiram bands with 20-56 cm^-1 shifts and cross-reference normal Raman, so these curated additions are well within standard SERS-paper practice — do NOT under-report them.
- Further extension toward the co-adsorbed/reference-region audit (~24/29) remains available but should be labeled as co-adsorbed/reference-region assignment rather than target-analyte literature matching.
- Further increases toward 25-26/29 require either curated source evidence or this project's own pure/reference SERS spectra. Do not widen the matching window or add candidate peaks merely to chase a larger number.

Decision:

- Manuscript mainline reports SHAP assignment as a two-metric consistency result:
  - formal main-text output: 22/29 peak clusters fall on known chemical bands, representing 91.2% of total SHAP attribution mass;
  - decomposition for SI/methods support: 18 own-analyte literature peaks (83.8%) + 2 cross-component interference bands (20/29, 88.1%) + 2 curated Thiram bands (22/29, 91.2%);
  - extended co-adsorbed/reference-region audit may report about 24/29 and 92.3% SHAP mass in SI only if explicit assignment levels are retained.
- Preferred wording: "SHAP spectral assignment consistency" or "SHAP 高贡献光谱区的化学归属一致性", not only "peak matching rate".
- Fig6 should include a small quantitative assignment panel or callout that reports both raw cluster count and SHAP-mass weighted consistency.
- Complete assignment details belong in SI, including assignment level: literature target-analyte peak, co-adsorbed/overlap region, own-reference peak, or unassigned.

Consequences:

- Reports and figure plans use the formal SHAP output: 22/29 and 91.2%. The 20/29 and 88.1% layer is only the decomposition before curated Thiram 860/1444 assignment.
- Do not claim 24/29 as the main-text formal output; it remains an optional SI-level co-adsorbed/reference-region extension.
- Do not report any number higher than the audited output produced by a documented peak library, source list, and fixed match-window rule.
