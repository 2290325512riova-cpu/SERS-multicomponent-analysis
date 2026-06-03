# SERS Multi-Component Pesticide Screening

Surface-enhanced Raman spectroscopy (SERS) plus interpretable machine learning for ternary pesticide-mixture screening on AgNPs. The current manuscript line is competitive-adsorption-induced MG 1616 cm⁻¹ peak suppression plus a compact, SHAP-guided screening workflow.

## Highlights

- MG 1616 cm⁻¹ suppression under co-existence: median peak-height ratio 0.355 (MG+MBA), 0.206 (MG+Thiram), and 0.129 (ternary) versus same-level pure MG
- 13 models × 6 preprocessing pipelines × 6 tasks benchmark under conventional spectrum-level 5-fold CV
- TreeSHAP peak-cluster attribution: 22/29 high-importance clusters map to known chemical bands, accounting for 91.2% total SHAP mass
- SHAP-guided compact screening: 10% wavenumber retention (141/1401) keeps all six tasks above 0.95 macro-F1
- Spiked-soil matrix screening validation: AUC 0.956–0.996 with 10/10 blanks rejected

## Dataset

| Item | Value |
|------|-------|
| Pure-mixture spectra | 901 (63 compositions) |
| Soil spectra | 79 design + 10 blanks |
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

Best macro-F1 per task in the complete 13-model benchmark:

| Task | Best model | F1 | Preprocessing |
|------|------------|----|---------------|
| P1 Thiram presence | KNN | 0.996 | p1 |
| P2 MG presence | RF | 0.985 | p1 |
| P3 MBA presence | ExtraTrees | 0.994 | p4 |
| G1 Thiram grade | ExtraTrees | 1.000 | p2 |
| G2 MG grade | ExtraTrees | 0.966 | p1 |
| G3 MBA grade | SVM | 0.981 | p1 |

Primary practical model for downstream interpretation and soil screening: ExtraTrees/p1.

SHAP-guided feature selection (10% spectral retention): all six tasks maintain macro-F1 > 0.95.

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
