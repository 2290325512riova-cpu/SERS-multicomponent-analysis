# SERS Multi-Component Pesticide Screening

Systematic benchmarking of machine learning models for simultaneous detection of three pesticides (Thiram, Malachite Green, MBA) in ternary mixtures via surface-enhanced Raman spectroscopy on AgNPs.

## Highlights

- 13 models × 6 preprocessing pipelines benchmark under random stratified 5-fold CV
- Composition-constrained spectral mixing with task-dependent ablation
- SHAP-guided spectral region selection: 70% feature reduction with <2% performance loss
- Soil-matrix screening feasibility (AUC 0.956–0.996)

## Dataset

| Item | Value |
|------|-------|
| Pure-mixture spectra | 901 (63 compositions) |
| Soil spectra | 79 design + 11 blanks |
| Wavenumber range | 400–1800 cm⁻¹ (1401 points) |
| Tasks | 3 presence (P1–P3) + 3 concentration grade (G1–G3) |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full 13-model benchmark:

```bash
python scripts/analysis/run_random_benchmark.py --strict-models
```

Run augmentation ablation (DL models):

```bash
python scripts/analysis/run_augmentation_ablation.py --variants p1 raw
```

Run SHAP analysis and feature selection:

```bash
python scripts/analysis/run_paper_shap.py
python scripts/analysis/run_shap_feature_selection.py
```

Run soil-matrix validation:

```bash
python scripts/analysis/run_soil_only_cv.py
```

## Results

Best macro-F1 per task (ExtraTrees, random 5-fold CV):

| Task | F1 | Preprocessing |
|------|----|---------------|
| P1 Thiram presence | 0.996 | p1 |
| P2 MG presence | 0.985 | p2 |
| P3 MBA presence | 0.994 | p1 |
| G1 Thiram grade | 1.000 | p1 |
| G2 MG grade | 0.966 | p4 |
| G3 MBA grade | 0.981 | p1 |

SHAP-guided feature selection (30% spectral retention): all tasks maintain F1 > 0.96.

Soil-only CV: AUC 0.956–0.996 across P1–P3; permutation p < 0.001; 10/10 blanks correctly rejected.

## Project Structure

```
src/                  Core modules (config, dataset, models, training)
scripts/analysis/     Experiment scripts
data/pure63_mainline/ Preprocessed data, splits, and model outputs
reports/              Human-readable progress and results summaries
project_memory/       Project decisions and experiment tracking
archive/              Negative experiments with manifest
```

## Preprocessing Variants

| Tag | Pipeline |
|-----|----------|
| raw | Interpolated raw spectra |
| p1 | Cosmic removal → SG smooth → ALS baseline → SNV |
| p2 | SG smooth → ALS baseline → 1st derivative → SNV |
| p3 | ALS baseline → vector normalization |
| p4 | Cosmic removal → SG smooth → ALS baseline (no norm) |
| p5 | Cosmic removal → SG smooth → ALS baseline → 2nd derivative → SNV |

## Citation

Paper in preparation. Citation will be added upon publication.
