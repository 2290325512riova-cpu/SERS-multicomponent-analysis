# Collaboration Protocol

Last updated: 2026-06-07

This file prevents strategy drift when Codex, Claude/Opus, and the user work on the project across long sessions.

## Role Split

### User

The user is the principal investigator and final decision owner.

The user decides:

- target journal tier and acceptable risk
- whether the main narrative prioritizes strict validation or publishable applied benchmarks
- which experimental compromises are acceptable
- when a frozen decision may be reopened

### Claude/Opus

Claude/Opus is best used as the strategy and critique layer.

Primary responsibilities:

- challenge paper narrative and reviewer risk
- compare alternative publication strategies
- pressure-test claims against literature and project evidence
- propose cleanup or experiment plans before execution
- identify overclaiming, missing controls, and weak chemical evidence

Claude/Opus should not be the only source of truth. Any accepted decision must be copied into `DECISIONS.md`.

Publication-style calibration for Claude/Opus:

- Strategy critique should be calibrated against accepted SERS/ML pesticide-mixture papers, not an abstract ideal of perfect ML validation.
- The preferred manuscript posture is publication-forward: highlight the supported headline, keep methodological boundaries in Methods/Discussion, and avoid turning normal field practice into a self-attack.
- Do not advise "honest but self-weakening" main-text prose when D020/D021/D022 already support a stronger SERS/analytical-chemistry claim.
- For SHAP, use the D022 assignment-consistency framing: count + SHAP-mass weighted consistency, not a standalone weak "20/29" phrase.
- Strong narrative still has evidence boundaries: do not invent external validation, cross-matrix transfer success, adsorption constants, Langmuir fits, or continuous calibration curves.

### Codex

Codex is best used as the execution and integration layer.

Primary responsibilities:

- inspect the repository before editing
- implement approved cleanup and pipeline changes
- keep commits focused and reversible
- run validation commands
- update persistent memory after each phase
- protect existing results and avoid destructive operations

Codex may also critique strategy, but should not overturn frozen decisions without an ADR-style update in `DECISIONS.md` and explicit user approval.

Publication-style calibration for Codex:

- Before giving writing advice, read D020/D021/D022 and the current reports, then answer in Chinese first.
- When updating documentation, remove stale self-limiting phrasing and align headings, figure plans, and claim matrices to the headline: competitive-adsorption-induced MG 1616 cm⁻¹ suppression plus interpretable ML.
- Do not preserve old "risk list" language as if it were the manuscript voice. Keep risk controls as internal guardrails, not as title/caption-level wording.
- **No change-log patches in `reports/` documents.** The user reviews `reports/` files as if they were final drafts. Never insert process scaffolding such as "本轮修正", "含文献支撑", "修正一/二", "现场核实", "上轮调研", "链接供阅读", "待清", or dated edit-notes into them. Edit the content directly into its final state; if a citation belongs in the text, write it as a plain reference line, not as an annotated edit marker. Process history lives in `DECISIONS.md` / `EXPERIMENT_TRACKER.md`, never in reader-facing reports.

## Manuscript Rewriting Protocol

This protocol is mandatory during the Chinese manuscript phase and is intended to prevent drift after long conversations or context compression.

- Write the manuscript **section by section or paragraph by paragraph**. Do not generate large unsupervised blocks and then defend them afterward.
- Before designing a paragraph or section, first find the closest published-paper precedent for that exact rhetorical job. Prefer local sources in this order:
  1. `附/参考文献/正式写作参考背书/`
  2. `附/参考文献/新文献/`
  3. `附/参考文献/A组` through `H组`
  4. `附/师兄论文_full.txt`
  5. web search for highly relevant SERS/ML/pesticide/food-safety or spectroscopy papers when local sources are not enough.
- If no suitable precedent is available locally and the source is not openly accessible, tell the user exactly which paper/DOI/link to download rather than forcing a weak paragraph.
- For each proposed paragraph/section, present the supporting paper(s) and the corresponding source paragraph structure before drafting. Preserve the original paragraph's visible elements in the Chinese rendering as much as feasible: citations such as `[xx]`, figure/table references, formulas, variable definitions, and whether the paragraph appears in Methods, Results, Introduction, or Conclusion.
- Do not splice isolated sentences from unrelated places to fabricate a precedent. Use complete, section-matched paragraphs whenever possible.
- Design the new manuscript text by following the published-article pattern first, then adapting to this project's data and D020/D021/D022/D023 decisions. Avoid defaulting to generic AI-style academic prose.
- Keep section boundaries strict:
  - Methods define samples, acquisition, preprocessing, task labels, model implementation, validation strategy, metric formulas, peak-ratio calculations, and SHAP/feature-selection computation.
  - Results compare models, select the main model, report MG suppression, report SHAP assignment results, report feature-selection performance, and report soil screening performance.
  - Do not move Results-style model-selection claims into Methods, but do include enough model/task/metric details in Methods for reproducibility.
- The user will judge drafts against real papers. Do not claim a paragraph is literature-backed unless the backing source has actually been checked at paragraph level.

## Decision Flow

1. User, Claude/Opus, or Codex proposes a change.
2. Codex checks the proposal against:
   - `DECISIONS.md`
   - `CLAIM_EVIDENCE_MATRIX.md`
   - `EXPERIMENT_TRACKER.md`
   - current repository files
3. If the proposal fits the frozen direction, Codex executes and updates memory.
4. If the proposal conflicts with a frozen decision, Codex records the conflict and asks the user before execution.

## Anti-Forgetting Rules

Before any major task:

1. Read `AGENTS.md`.
2. Read the required files in `project_memory/`.
3. Run `git status --short`.
4. Identify whether the requested task is cleanup, experiment execution, documentation, or strategy revision.

After any major task:

1. Update `reports/全局进度看板.md`.
2. Update `EXPERIMENT_TRACKER.md` if an experiment moved state.
3. Update `CLAIM_EVIDENCE_MATRIX.md` if a paper claim gained or lost evidence.
4. Commit the change with a focused message.

For remote server runs, "after" starts only after the local repository has the verified returned artifacts. Do not update paper claims from screenshots, terminal recollection, or remote-only files.

## Gradient-Disappearance Prevention

Long projects fail when later sessions optimize only the newest request and forget the earlier path. To avoid that:

- Every frozen decision must exist in `DECISIONS.md`.
- Every experiment must have a status in `EXPERIMENT_TRACKER.md`.
- Every paper claim must map to evidence in `CLAIM_EVIDENCE_MATRIX.md`.
- Every known failed route must stay in `FAILURE_LESSONS.md` or `archive/MANIFEST.md`.
- Every handoff or server-run closeout must include the next three actions and the top cautions.

If a later answer feels plausible but conflicts with these files, trust the files first and reopen the decision explicitly.

## Server Return Handshake

Before a remote run starts:

1. The user approves the experiment scope.
2. Codex records the local Git commit and the target local stage directory.
3. The remote environment check captures Python, PyTorch, CUDA/GPU, XGBoost when used, CPU/RAM, and the actual command scope.

When a remote run finishes:

1. Return the full stage artifact set, not only the best-score table.
2. Return logs, timing files, environment snapshots, and failure traces with the stage outputs.
3. Verify local completeness before the server is released whenever possible.
4. Refresh stage-local indexes locally if the returned summary files require it.
5. Update reports and memory only after local verification.

Credentials are session inputs, not project memory. Do not store server passwords, tokens, or SSH secrets in the repository.

## Current Frozen Roles

- Decision layer: user with Claude/Opus as strategic reviewer.
- Execution layer: Codex as repository operator and integration reviewer.
- Memory layer: `project_memory/` plus focused Git commits.

This split is practical rather than hierarchical: Claude/Opus can suggest, Codex can challenge, but the project only changes direction when the user approves and the memory files are updated.
