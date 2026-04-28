# Pending Decisions

> Append-only log of human-gated decisions surfaced by the autonomous DAG run.
> Entries land here when the run encounters `plan-writer`, `spec-needed`, or
> `human-needed` work. The run continues with other eligible tasks (or
> terminates if none remain).
>
> User resolves each entry by adding an `**APPROVED:**` line, then the agent
> drafts the plan / executes the resolution. Once resolved, entries are
> **removed entirely** — git history retains the deliberation. Only genuinely-
> pending decisions live in this file.

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

## 2026-04-28 — plan-writer: engine_gpu --features gpu repair (surfaced by dispatch-emit T16)

**Background.** Dispatch-emit Plan COMPLETE (16/16, last commit `4474566c`). The plan deleted the 7 hand-written kernel modules from `crates/engine_gpu/src/`. After deletion, `cargo build -p engine_gpu --features gpu` has 100 errors — 49 pre-existing at HEAD, 51 newly exposed by the deletions.

**The pre-existing 49 errors** are NOT in the deleted hand-written kernels. They're in `step_batch`, `cascade.rs`, `cascade_resident.rs`, `backend/{resident,sync}_ctx.rs`, `event_ring.rs`, `view_storage_symmetric_pair.rs`. Symptoms:
- `state.views` field accesses (state-port migration removed `views` from `SimState`)
- `engine::generated` module missing
- `engine_rules::scoring` import unresolved
- `EventRing` / `CascadeRegistry` missing generic args
- `cascade.run_fixed_point_tel` arity mismatch (4 args expected, 3 supplied)

**The 51 newly-exposed errors** are dangling refs to the deleted modules in `lib.rs` (52 sites), `cascade.rs`, `cascade_resident.rs`, `backend/{resident,sync}_ctx.rs`. These references should be replaced with `engine_gpu_rules::*` types end-to-end.

**Suggested plan scope:** ~10 tasks. Roughly: (a) catalog every dangling reference; (b) port step_batch loop body to consume `engine_gpu_rules::*`; (c) rewire snapshot/sync helpers; (d) restore EventRing/CascadeRegistry generic args; (e) migrate or restore `state.views`; (f) trim cascade_resident.rs to skeleton; (g) re-enable parity tests as gates.

**Why this matters:** Plan invariant #5 (every parity test passes) cannot be verified until this lands. Default-features build is clean and unaffected; this only blocks `--features gpu`.

**Status:** awaiting user
**To proceed:** add `**APPROVED:**` to authorize plan-writing, optionally with sequencing preference (before vs. after the Ability DSL / Economic depth plans).

---
