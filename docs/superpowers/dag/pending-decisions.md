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

## 2026-04-26 — plan-writer: Plan 6 — `GpuBackend` foundation

**Roadmap source:** `docs/ROADMAP.md` "Engine plans not yet written" tier; depends on Plan 5 complete.

**Spec exists** at `docs/spec/engine.md` (GPU annexes §§9–12). Goal: bridge the `ComputeBackend` trait's kernel-dispatch surface (after 5b–e) to the existing `engine_gpu` primitives (kernels under `crates/engine_gpu/src/`). Most of the underlying GPU kernels already exist (megakernel + cold-state replay landed); this is mostly plumbing — wiring `GpuBackend` impl methods to invoke the right kernels for each backend trait method.

**Suggested scope:** ~6-10 tasks. Depends on Plan 5b–e closing first.

**Status:** awaiting user
**To proceed:** add `**APPROVED:**` (typically deferred until 5b–e is in flight).

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
