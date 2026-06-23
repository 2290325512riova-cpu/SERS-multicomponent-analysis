# Project Agent Protocol

This project is a long-running SERS plus machine-learning research project. Do not rely on chat history as the source of truth. Before planning, editing, or running experiments, read the persistent memory files below.

## Required Startup Context

Read these files first:

1. `project_memory/DECISIONS.md`
2. `project_memory/EXPERIMENT_TRACKER.md`
3. `project_memory/CLAIM_EVIDENCE_MATRIX.md`
4. `project_memory/COLLABORATION_PROTOCOL.md`
5. `project_memory/project_architecture.md`
6. `reports/全局进度看板.md`

Then check:

```powershell
git status --short
```

If the requested action conflicts with an active decision, stop and ask for explicit user approval before changing direction.

## Frozen Project Decisions

- Random stratified spectral-level CV is the main paper benchmark.
- Grouped CV is NOT in the paper (neither main nor SI). Data preserved for future use.
- `03b_locked_main_results/` must stay in place; it is the locked grouped-CV evidence for future papers.
- Negative and failed experiments are archived with a manifest, not deleted.
- Soil screening uses soil-only CV (train and test within soil data) to support spiked-soil / soil-matrix screening validation.
- SHAP must be interpreted at peak-cluster/window level unless new point-level evidence proves otherwise.
- SHAP-guided spectral region selection (E021/E022) is a core contribution: mask non-SHAP regions, retrain, show performance maintained.
- Composition-constrained spectral mixing must be train-fold-only.
- MG concentration grading is semi-quantitative, not precise quantification.

## Decision Change Protocol

Do not overturn a frozen decision in chat only.

If a change is needed:

1. Add or update an ADR entry in `project_memory/DECISIONS.md`.
2. State the evidence that motivates the change.
3. List consequences for paper claims and experiment files.
4. Ask the user to approve before implementation.

## Execution Protocol

- Keep cleanup, archiving, feature development, and documentation in separate commits where possible.
- Before moving or deleting files, identify references with `rg`.
- Never move `01_candidate_screening/`, `02_representative_model_optimization/`, `03b_locked_main_results/`, `04_leakage_analysis/`, or `05_shap_explainability/` unless the user explicitly approves a new migration plan.
- Do not blindly `git add -A` in the parent repository; only stage in `core subject/`.
- Do not add preprocessing variants outside the registry in `src/config.py`.
- Do not describe p5 as the most stable preprocessing (its peak RSD is high due to second derivative).
- Before any remote server run, record the approved run scope, the local Git commit, the remote log path, and the local stage directory that will receive artifacts.
- After any remote server run, follow the server artifact return protocol in `project_memory/project_architecture.md`: return the full stage outputs, logs, environment snapshot, and run metadata before updating claims.
- Do not leave server sync archives or credentials in the repository root. Never write server passwords, API tokens, or SSH secrets into project files.
- Update `reports/全局进度看板.md` whenever the active phase or next actions change.
- If Codex and Claude/Opus disagree, preserve the disagreement in `project_memory/DECISIONS.md`; do not silently overwrite the project direction.

## Paper Claim Guardrails

- Follow D021/D022 publication-style narrative calibration: write Chinese-first, evidence-forward prose. Do not default to self-weakening caveat lists when the project data support a strong SERS/analytical-chemistry claim.
- Follow D022 for SHAP assignment wording: the formal audited output is 22/29 peak clusters on known chemical bands, accounting for 91.2% of total SHAP mass. The older 20/29 and 88.1% number is only the intermediate decomposition layer before curated Thiram 860/1444 assignment; do not present it as the final baseline.
- Calibrate wording against accepted SERS/ML pesticide-mixture papers, not against an abstract "perfect ML validation" ideal. The manuscript voice should be publication-forward: headline strong, methods precise, limitations placed where they belong.
- Main-text headline: the study reveals competitive-adsorption-induced MG 1616 cm⁻¹ peak suppression and mixture-induced spectral interference in ternary pesticide SERS mixtures.
- 1172/1394 cm⁻¹ are overlap-sensitive MG-associated regions demonstrating peak-envelope reshaping; do not frame them as discarded flaws in main-text planning.
- Present soil as spiked-soil / soil-matrix screening validation in title/caption-level prose; reserve "same-matrix CV" for methods-level detail.
- Do not call random split "external validation" or "leakage-free generalization".
- Use "conventional spectrum-level validation" or "within-dataset screening benchmark" for random CV.
- Grouped CV is NOT in the paper — do not reference it as supplementary evidence.
- SHAP peak assignment formal output is 22/29 clusters plus 91.2% SHAP-mass consistency. Optional ~24/29 extended co-adsorbed/reference-region assignment stays SI-level unless additional own-reference evidence makes it the better-supported main claim.
- Do not claim Beer-Lambert spectral mixing is first-ever; frame it as composition-constrained spectral mixing for ternary pesticide SERS.
- Soil result is title/caption-level "spiked-soil / soil-matrix screening validation"; method-level truth is matrix-matched/same-matrix CV. It is not "external validation" or "cross-matrix transfer".

## Manuscript Drafting Protocol

- During Chinese manuscript rewriting, work paragraph by paragraph or section by section.
- Before drafting a paragraph/section, find the closest published-paper precedent for that exact section role. Prefer `附/参考文献/正式写作参考背书/`, then `附/参考文献/新文献/`, then `附/参考文献/A组`-`H组`, then `附/师兄论文_full.txt`, then web/open literature. If no suitable source is available, ask the user to download a better paper instead of forcing a weak draft.
- Present the supporting paper and corresponding paragraph-level Chinese rendering before proposing the adapted manuscript text. Preserve source paragraph elements where present: citation markers, figure/table references, formulas, variables, and section placement.
- Do not use isolated sentence snippets from unrelated sections as "support" for a new paragraph. Prefer complete Methods-to-Methods, Results-to-Results, Introduction-to-Introduction, and Conclusion-to-Conclusion matches.
- Methods may define task labels, model candidates, validation strategy, metrics, peak-ratio calculations, and SHAP/feature-selection formulas. Results should contain model comparison, best-model selection, MG suppression numbers, SHAP assignment numbers, feature-retention performance, and soil validation outcomes.
