# Archive Manifest

Date: 2026-05-21

This archive preserves negative or superseded experiments so they remain available for paper discussion, rebuttal planning, and future agent handoff. These files are not part of the new `paper_main` execution path.

## Archived Scripts

| Original path | Archived path | Reason | Paper role |
| --- | --- | --- | --- |
| `scripts/analysis/dl_rescue_experiment.py` | `archive/scripts_negative/dl_rescue_experiment.py` | DL rescue attempt was exploratory, slow, and did not become the main route. | Negative evidence: do not force DL to beat RF. |
| `scripts/analysis/dl_rescue_hybrid.py` | `archive/scripts_negative/dl_rescue_hybrid.py` | Self-supervised / hybrid rescue path did not provide a reliable improvement. | Negative evidence / internal methods record. |
| `scripts/analysis/dl_rescue_quick.py` | `archive/scripts_negative/dl_rescue_quick.py` | Quick DL comparison was superseded by the planned random-CV benchmark. | Historical exploration only. |
| `scripts/analysis/dl_rescue_round2.py` | `archive/scripts_negative/dl_rescue_round2.py` | Second rescue round was stopped or superseded. | Negative evidence / avoid repeating. |
| `scripts/analysis/dl_final_test.py` | `archive/scripts_negative/dl_final_test.py` | AttResNet final test is a useful historical result but not an active runner. | Architecture reference for future AttResNet extraction. |
| `scripts/analysis/compute_ordinal_metrics.py` | `archive/scripts_negative/compute_ordinal_metrics.py` | Ordinal framing did not solve the MG concentration limitation. | Optional supplementary metric reference only. |
| `scripts/analysis/run_two_stage_experiment.py` | `archive/scripts_negative/run_two_stage_experiment.py` | Two-stage/context-aware route was negative and is no longer active. | Negative result / possible rebuttal evidence. |

## Archived Result Directories

| Original path | Archived path | Reason | Paper role |
| --- | --- | --- | --- |
| `data/pure63_mainline/models/mainline_formal_experiment/03_two_stage_context_aware/` | `archive/03_two_stage_context_aware/` | Two-stage/context-aware results are negative and do not belong in the active grouped-CV or random-CV result tree. | Internal negative result; may support Discussion if needed. |

## Explicitly Not Archived

These directories remain in place because the new paper plan still uses them:

| Path | Role |
| --- | --- |
| `data/pure63_mainline/models/mainline_formal_experiment/01_candidate_screening/` | Supplementary grouped candidate screening baseline. |
| `data/pure63_mainline/models/mainline_formal_experiment/02_representative_model_optimization/` | Historical optimization and hyperparameter reference. |
| `data/pure63_mainline/models/mainline_formal_experiment/03b_locked_main_results/` | Locked grouped-CV evidence for supplementary strict validation. |
| `data/pure63_mainline/models/mainline_formal_experiment/04_leakage_analysis/` | Spectrum-level vs folder-level validation protocol evidence. |
| `data/pure63_mainline/models/mainline_formal_experiment/05_shap_explainability/` | Baseline for upgraded peak-cluster SHAP analysis. |
