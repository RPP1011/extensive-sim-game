# Pending Decisions

> Append-only log of human-gated decisions surfaced by the autonomous DAG run.
> Entries land here when the run encounters `plan-writer`, `spec-needed`, or
> `human-needed` work. The run continues with other eligible tasks (or
> terminates if none remain).
>
> User resolves each entry by editing the section to add an `**APPROVED:**`
> line (for plan-writer) or by starting an interactive brainstorm session
> (for spec-needed). Then re-run `dag-bootstrap.sh` to incorporate the
> resolution.

---

## 2026-04-25 — DAG hard-stop: stale-plan reconciliation needed

**Type:** Structural reconciliation (cannot proceed autonomously)

**Encountered iteration 2:** `2026-04-22-ir-interpreter.task-2` blocked on prerequisite `crates/dsl_ast/` not existing (P1a not landed).

**Root cause (broader):** Five plans in the DAG were **already executed earlier in this conversation** but their `- [ ] **Step N: ...**` checkboxes were never flipped to `[x]`. The bootstrap parses them as pending, so `dag-tick` keeps offering them up:

| Plan | Status in reality | Status in DAG |
|---|---|---|
| `2026-04-22-ir-interpreter` | Stalled — needs P1a | task-1 done; task-2 blocked |
| `2026-04-24-constitution-and-adr-impl` | **Executed** (constitution + ADR + AIS template + llms.txt + SessionStart hook) | 0/10 done |
| `2026-04-25-critic-skills-impl` | **Executed** (6 critic skills) | 0/9 done |
| `2026-04-25-dispatch-critics-hooks-impl` | **Executed** (dispatch-critics + Git pre-commit) | 0/10 done |
| `2026-04-25-engine-crate-restructure-impl` | **Executed** (engine_generated → engine_data rename) | 0/15 done |
| `2026-04-25-engine-crate-split-impl-v2` | **Executed** (Plan B1' — generic primitives, emit_step, engine_rules) | 0/21 done |
| `2026-04-25-legacy-src-sweep-impl` | **Executed** (legacy src/ sweep — Plan B3) | 0/5 done |

If `/loop /dag-tick` continues, it will dispatch sonnet subagents to "implement" each Task 1 of these plans. The subagents will find the work already in git history and either produce no diff or create confused duplicate commits. Either way: wasted budget, no progress.

**Options to resolve (pick one):**

1. **(A) Reconciliation script** — write a small helper that scans `git log` for each plan's expected commit signatures (commit message contains "spec-c-v2", "critic-skills", etc.) and auto-marks the associated state.json tasks as `done` with the matching commit SHAs. Re-bootstrap merges the reconciled state.

2. **(B) Manual mark-skipped** — call `dag-mark-done.sh <task-id> "executed-pre-dag" '{}'` for each task in each completed plan. Or write a one-shot jq update that marks all tasks in these plan IDs as `status: done` (effectively `skipped`).

3. **(C) Delete the plan files** — these plans are done; per the project conventions, "Items leave the doc entirely when **fully merged** — git history is the record." Move plan files to a `docs/superpowers/plans/archive/` dir or delete them; re-bootstrap; the DAG no longer offers them.

4. **(D) Defer the IR-interpreter plan** — pre-emptively mark its tasks `deferred` until P1a (dsl-ast-extraction) lands. Doesn't fix the broader reconciliation issue.

**Recommendation:** (C) is cleanest — these plans are done. Archive (or delete) them. If they need to be referenced later, git history has them.

**Status:** awaiting user
**To resolve:** pick an option, execute it, then re-run `/loop /dag-tick`. The DAG will then offer up the actually-pending work (Plan 4 debug & trace, Plan 5a ComputeBackend, ToM, tech-debt cleanup, dsl-ast-extraction, spec-c-v2 itself).

---

<!-- entries appended below by the agent -->

## 2026-04-26 — spec-needed: Collision detection (Tech-Debt T3)

**Roadmap source:** `docs/engine/status.md` open question #11

**Current state:** Agents can co-occupy a `Vec3`. Visualization works around via vertical voxel stacking (1 unit elevated per co-located agent).

**Decision required:**
- (a) Add collision detection as a real engine primitive (movement
  resolution rejects moves that would collide; emits `MoveBlocked` event).
  Cost: spatial-index queries on every movement; new event variant; non-trivial.
- (b) Keep co-occupancy semantics; document as intentional. Agents are
  point particles; collision is a rendering/visualization concern only.
- (c) Hybrid: collision detection only for specific kinds (e.g., NPCs
  collide with structures but not other NPCs).

**Status:** awaiting user
**To proceed:** add `**APPROVED:** [option]` to this section.
