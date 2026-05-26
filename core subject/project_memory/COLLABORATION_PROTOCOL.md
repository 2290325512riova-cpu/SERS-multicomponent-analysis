# Collaboration Protocol

Last updated: 2026-05-21

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
