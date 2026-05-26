# SERS Multi-Component Pesticide Screening

This repository contains the active SERS plus machine-learning pipeline for simultaneous screening of ternary pesticide mixtures:

- Thiram
- Malachite Green, abbreviated as MG
- Mercaptobenzoic acid, abbreviated as MBA

The current paper strategy is pragmatic and evidence-bounded:

- Main result: conventional spectrum-level random stratified 5-fold CV.
- Supplementary validation: grouped/folder-level transfer stress test.
- External matrix evidence: soil-only CV demonstrates screening feasibility in soil matrices.
- Interpretability: SHAP is interpreted at peak-window or cluster level unless a verified point-level result supports a stronger claim.
- Augmentation: composition-constrained spectral mixing is train-fold-only.
- MG concentration grading is treated as semi-quantitative and physically difficult.

## Repository Boundary

There are two Git boundaries in the larger workspace. Treat this folder, `core subject/`, as the active code repository.

Do not stage the parent directory blindly. The parent repository tracks historical overlapping files and reference assets.

## Current Data

| Item | Current value |
| --- | --- |
| Main data version | `pure63_mainline` |
| Pure-mixture spectra | 901 |
| Pure-mixture folders/compositions | 63 |
| Wavenumber range | 400-1800 cm^-1, 1401 points |
| Soil spectra detected by current parser | 79 design spectra plus 11 blank/background spectra across 13 folders |
| Grouped split | `data/pure63_mainline/splits/cv_split_pure63_main.csv` |
| Random split | `data/pure63_mainline/splits/cv_split_random_5fold.csv` |
| Preprocessed caches | `X_raw.npy`, `X_p1.npy`, `X_p2.npy`, `X_p3.npy`, `X_p4.npy`, `X_p5.npy` |

Note: the random split uses a six-label joint stratification key. Two exact-composition strata contain fewer than five spectra, so the split is best-effort joint stratified.

Note: lightweight RSD QC does not support calling p5 the most repeatable preprocessing. p5 should be treated as a second-derivative feature candidate, especially for interpretability/benchmark comparison.

## Quick Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

The random benchmark script skips XGBoost by default if it is unavailable. Use `--strict-models` only after installing requirements and verifying the server environment.

Run lightweight paper-main preparation tables:

```powershell
python scripts/analysis/run_data_quality.py
python scripts/analysis/run_shap_cluster.py
python scripts/analysis/run_soil_validation.py
```

Run a classical-model random-CV benchmark first:

```powershell
python scripts/analysis/run_random_benchmark.py --models RF ExtraTrees HistGradientBoosting SVM KNN PLS-DA LDA --variants raw p1 p2 p3 p4 p5
```

Run the full 12-model benchmark after dependencies and runtime are ready:

```powershell
python scripts/analysis/run_random_benchmark.py --strict-models
```

Run DL augmentation ablation:

```powershell
python scripts/analysis/run_augmentation_ablation.py --variants p5
```

## Active Project Layout

```text
core subject/
  project_memory/
    AGENTS.md
    CLAUDE.md
    project_architecture.md
    DECISIONS.md
    EXPERIMENT_TRACKER.md
    CLAIM_EVIDENCE_MATRIX.md
    FAILURE_LESSONS.md
    COLLABORATION_PROTOCOL.md
  reports/
    全局进度看板.md
    图表与表格清单.md
    实验结果摘要.md
    论文写作路线图.md
  src/
    config.py
    dataset.py
    models.py
    train_eval.py
    result_index.py
  scripts/analysis/
    run_random_benchmark.py
    run_augmentation_ablation.py
    run_data_quality.py
    run_shap_cluster.py
    run_soil_validation.py
    leakage_analysis.py
    benchmark_summary.py
  data/pure63_mainline/
    processed/
    splits/
    models/
      mainline_formal_experiment/
      paper_main/
  archive/
    MANIFEST.md
    scripts_negative/
    03_two_stage_context_aware/
```

## Historical Results Kept For Supplementary Evidence

Do not move these directories without a new approved migration plan:

- `data/pure63_mainline/models/mainline_formal_experiment/01_candidate_screening/`
- `data/pure63_mainline/models/mainline_formal_experiment/02_representative_model_optimization/`
- `data/pure63_mainline/models/mainline_formal_experiment/03b_locked_main_results/`
- `data/pure63_mainline/models/mainline_formal_experiment/04_leakage_analysis/`
- `data/pure63_mainline/models/mainline_formal_experiment/05_shap_explainability/`

The locked grouped-CV results in `03b_locked_main_results/` are the supplementary folder-level transfer evidence. They are not the paper-main headline after the strategy change, but they must remain intact.

## Paper Claim Guardrails

Allowed wording:

- conventional spectrum-level validation
- within-dataset screening benchmark
- folder-level transfer stress test
- composition-constrained spectral mixing
- semi-quantitative MG concentration grading

Forbidden or unsafe wording unless new evidence is generated:

- random CV as external validation
- leakage-free generalization
- first-ever Beer-Lambert augmentation
- SHAP alone proves competitive adsorption
- 17/20 SHAP peak matching
- precise MG quantification

## Persistent Memory

Before making strategic or code changes, read:

1. `project_memory/AGENTS.md`
2. `project_memory/DECISIONS.md`
3. `project_memory/EXPERIMENT_TRACKER.md`
4. `project_memory/CLAIM_EVIDENCE_MATRIX.md`
5. `project_memory/COLLABORATION_PROTOCOL.md`
6. `project_memory/project_architecture.md`

These files are the anti-forgetting layer for future Codex, Claude/Opus, and human sessions.

Stale editor tabs are not a source of truth. `project_memory/HANDOFF.md`, `project_memory/PROJECT_STATE.md`, and the retired long-form decision diary are not active repository memory files.

## User-Facing Reports

These documents are researcher-facing summaries. Agents still read `reports/全局进度看板.md` during startup:

- `reports/全局进度看板.md` — 项目总览入口
- `reports/实验结果摘要.md` — 实验结果数字
- `reports/图表与表格清单.md` — 论文图表状态
- `reports/论文写作路线图.md` — 论文结构与写作计划
