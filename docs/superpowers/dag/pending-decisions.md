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

**Status:** **RESOLVED 2026-04-26 — option (B) hybrid**: sed-flipped 374 checkboxes across the 6 stale plan files; force-marked their state.json tasks as `done` with `completed_commit: executed-pre-dag`. ir-interpreter plan deferred until P1a lands (subsequently re-derived in Plan ToM, which used the existing hand-written SimState path instead of needing dsl_ast extraction). Loop resumed and drained the actual queue across iterations 4-40. Reference commit: `fab0c615`.

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

**Status:** **RESOLVED 2026-04-26 — option (b)** — co-occupancy intentional; viz handles overlap (`status.md` Q#11 closed). state.json T3 marked done in commit `6ef3a88e`.

## 2026-04-26 — spec-needed: Announce 3D vs planar distance (Tech-Debt T4)

**Roadmap source:** `docs/engine/status.md` open question #1

**Current state:** `Announce` event uses `spatial.within_radius()` which is 3D Euclidean. Spec §10 doesn't specify; this is an undocumented choice.

**Decision required:**
- (a) Confirm 3D — update spec to make it explicit; no code change.
- (b) Switch to planar (XZ-only) — cheaper computation, more intuitive for 2.5D scenes. Implementation: new `spatial.within_radius_xz()` primitive; update Announce dispatch.
- (c) Per-event-kind choice — `Announce` is planar (sound travels along the floor); `BroadcastSelf` is 3D (visual). Adds complexity.

**Status:** **RESOLVED 2026-04-26 — option (a)** — 3D Euclidean confirmed; spec/runtime.md §10 made explicit (`status.md` Q#1 closed). state.json T4 marked done in commit `6ef3a88e`.

## 2026-04-26 — human-needed: Passive triggers spec mismatch (Tech-Debt T7)

**Roadmap source:** `spec/ability.md` §6 / §23.1 markers say `runs-today`; no Trigger AST node or handler exists. Spec overclaims.

**Decision required:**
- (a) **Implement** — adds an `IrPassiveTrigger` IR node, parser/resolver extension, emit_physics generates per-trigger handlers, runtime fires them on emitted events. Substantial DSL grammar work; sub-plan needed. Estimated effort: 2-3 weeks (comparable to Theory of Mind Phase 1).
- (b) **Downgrade markers** — rewrite spec/ability.md §6 + §23.1 to mark `passive` block + listed triggers as `planned`, not `runs-today`. Removes the false claim. Future plan adds them.

**Status:** **RESOLVED 2026-04-26 — option (b)** — spec/ability.md §6 + §23.1 already mark passive block + triggers as `planned`; clarifying note added that the `ability` block "runs-today" status refers to parsing only. state.json T7 marked done in commit `6ef3a88e`.

---

## 2026-04-26 — TERMINAL: HUMAN_BLOCKED — autonomous queue drained

The agent loop terminated cleanly at iteration 40 after closing 37 implementer
tasks across 5 plans (Plan 4 debug & trace, Plan 5a ComputeBackend, Tech-Debt
cleanup, Spec C v2 bootstrap, Theory-of-Mind Phase 1). The 6 entries below
are the only remaining work — all `plan-writer` per (D) tiered autonomy.

User resolves each by adding `**APPROVED**` and (optionally) listing constraints
or scope changes; agent then drafts the plan via the `superpowers:writing-plans`
skill. Plans land at `docs/superpowers/plans/<date>-<slug>.md`. Re-bootstrap
populates the new tasks; loop continues.

---

## 2026-04-26 — plan-writer: Plan 5b–e — Remaining ComputeBackend phases

**Roadmap source:** `docs/spec/runtime.md` §8/§12/§14; `plans/2026-04-25-plan-5a-computebackend-mask-fill-impl.md` (Phase 1 of 5; Phases 5b–e deferred).

**Spec exists.** Plan 5a established the `ComputeBackend` trait + mask-fill threading. Subsequent phases:
- **5b** — cascade dispatch through backend (cascade currently goes through `CascadeRegistry::dispatch` directly; route through `backend.cascade_dispatch(...)`)
- **5c** — view fold through backend
- **5d** — real GPU kernel emit (Phase 1 used CPU pass-through stubs in `GpuBackend`)
- **5e** — full kernel-dispatch surface + cross-backend parity sweep

**Suggested scope:** one plan per phase (4 plans), each ~9 tasks like 5a. Or a single combined plan if phases are tightly interleaved. Direct continuation of 5a's pattern; mostly mechanical.

**Status:** awaiting user
**To proceed:** add `**APPROVED:** [scope choice — separate plans or combined]` and any constraints.

---

## 2026-04-26 — plan-writer: Plan 6 — `GpuBackend` foundation

**Roadmap source:** `docs/ROADMAP.md` "Engine plans not yet written" tier; depends on Plan 5 complete.

**Spec exists** at `docs/spec/gpu.md`. Goal: bridge the `ComputeBackend` trait's kernel-dispatch surface (after 5b–e) to the existing `engine_gpu` primitives (kernels under `crates/engine_gpu/src/`). Most of the underlying GPU kernels already exist (megakernel + cold-state replay landed); this is mostly plumbing — wiring `GpuBackend` impl methods to invoke the right kernels for each backend trait method.

**Suggested scope:** ~6-10 tasks. Depends on Plan 5b–e closing first.

**Status:** awaiting user
**To proceed:** add `**APPROVED:**` (typically deferred until 5b–e is in flight).

---

## 2026-04-26 — plan-writer: Plan 7+ — per-kernel GPU porting under the trait

**Roadmap source:** `docs/ROADMAP.md` "Engine plans not yet written" tier.

**Note:** This entry was previously framed as "many kernels still on CPU." Investigation 2026-04-26 (during this session) showed that nearly all kernels already exist in `engine_gpu/src/`. The genuinely-remaining items are:
- **GPU cold-state replay Phase 4** (memory + chronicle handlers per `spec/gpu.md` §4.5/§4.6) — small umbrella tail; could fold into Plan 6.
- **Subsystem 3 — Ability evaluation on GPU** (`pick_ability` kernel + `ability::tag(TAG)` scoring grammar + `per_ability` row type per `spec/gpu.md` §5) — this IS a real not-yet-ported kernel.

**Recommended action:** retire this catch-all plan-writer entry and replace with a focused plan for Subsystem 3 ability evaluation (the only genuinely-missing kernel). The "per-kernel porting" framing was overstated.

**Status:** awaiting user
**To proceed:** approve restructure (e.g., "draft a Subsystem 3 plan; drop the Plan 7+ umbrella").

---

## 2026-04-26 — plan-writer: Ability DSL implementation

**Roadmap source:** `docs/ROADMAP.md` "Drafted (spec exists, plan does not)" tier.

**Spec exists** at `docs/spec/ability.md` (~2000 LoC). Scope: voxel ops, control verbs (root/silence/fear/taunt), AI-state manipulation, structures, materials, passive triggers (note: Tech-Debt T7 resolved 3b — passive triggers are correctly marked `planned` in §23.1, so this plan covers their actual implementation).

**Suggested scope:** large — easily multi-week. Likely needs decomposition into multiple plans (one per verb category + structures/materials separately).

**Status:** awaiting user
**To proceed:** add `**APPROVED:** [decomposition strategy — one big plan or N sub-plans]`.

---

## 2026-04-26 — plan-writer: Economic depth implementation

**Roadmap source:** `docs/ROADMAP.md` "Drafted (spec exists, plan does not)" tier.

**Spec exists** at `docs/spec/economy.md` (~1400 LoC). Scope: recipes, contracts, labor, heterogeneity, information asymmetry, market structure, macro dynamics. Spec already designates 3 phases.

**Suggested scope:** 3 plans, one per phase. Each phase plan is ToM-scale (~14 tasks).

**Status:** awaiting user
**To proceed:** add `**APPROVED:**` with phase ordering preference.

---

## 2026-04-26 — plan-writer: GPU ability evaluation (Subsystem 3)

**Roadmap source:** `docs/ROADMAP.md` "Drafted (spec exists, plan does not)" tier; `docs/spec/gpu.md` §5.

**Spec exists.** Concrete deliverables: `pick_ability` kernel emitted from DSL → WGSL, new `ability::tag(TAG)` scoring grammar primitive, `per_ability` row type. This is the one genuinely-missing GPU kernel referenced in the Plan 7+ entry above.

**Suggested scope:** ~10 tasks (kernel emit + scoring grammar extension + parity test). Could be the natural successor to Plan 6 in the GPU stack.

**Status:** awaiting user
**To proceed:** add `**APPROVED:**` (often deferred until Plan 6 lands).

---
