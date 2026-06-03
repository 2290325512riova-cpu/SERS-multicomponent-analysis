# Project Memory

This folder is the persistent memory layer for the SERS pesticide-mixture project. It exists so Opus, Codex, and future sessions do not have to reconstruct decisions from chat history or stale editor tabs.

## Current Fact Sources

Read `../AGENTS.md` first. It points to `AGENTS.md`, which defines the required startup context and the frozen project guardrails.

Use these files as the active fact sources:

| Question | Active fact source |
| --- | --- |
| What decisions are currently approved? | `DECISIONS.md` |
| Which experiments exist, what is their status, and where are their artifacts? | `EXPERIMENT_TRACKER.md` |
| Which paper claims are supported or still risky? | `CLAIM_EVIDENCE_MATRIX.md` |
| How should Codex, Claude/Opus, and the user coordinate? | `COLLABORATION_PROTOCOL.md` |
| What is the active repository layout, execution order, and server artifact landing map? | `project_architecture.md` |
| What failed routes must not be rediscovered casually? | `FAILURE_LESSONS.md` and `../archive/MANIFEST.md` |
| What should the researcher read first? | `../reports/` user-facing reports |

`HANDOFF.md` is a compact onboarding summary, but the source of truth is still the decision/evidence stack above. `PROJECT_STATE.md` and the old long-form decision diary are not active fact sources. If an IDE has an old handoff tab open, reopen `HANDOFF.md` from disk before relying on it.

## Read Order

All agents should read `AGENTS.md` first, then:

1. `DECISIONS.md`
2. `EXPERIMENT_TRACKER.md`
3. `CLAIM_EVIDENCE_MATRIX.md`
4. `COLLABORATION_PROTOCOL.md`
5. `project_architecture.md`
6. `../reports/全局进度看板.md`
