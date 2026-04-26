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

**Roadmap source:** `docs/spec/engine.md` §8/§4.2/§4; `plans/2026-04-25-plan-5a-computebackend-mask-fill-impl.md` (Phase 1 of 5; Phases 5b–e deferred).

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

**Spec exists** at `docs/spec/engine.md` (GPU annexes §§9–12). Goal: bridge the `ComputeBackend` trait's kernel-dispatch surface (after 5b–e) to the existing `engine_gpu` primitives (kernels under `crates/engine_gpu/src/`). Most of the underlying GPU kernels already exist (megakernel + cold-state replay landed); this is mostly plumbing — wiring `GpuBackend` impl methods to invoke the right kernels for each backend trait method.

**Suggested scope:** ~6-10 tasks. Depends on Plan 5b–e closing first.

**Status:** awaiting user
**To proceed:** add `**APPROVED:**` (typically deferred until 5b–e is in flight).

---

## 2026-04-26 — plan-writer: Plan 7+ — per-kernel GPU porting under the trait

**Roadmap source:** `docs/ROADMAP.md` "Engine plans not yet written" tier.

**Note:** This entry was previously framed as "many kernels still on CPU." Investigation 2026-04-26 (during this session) showed that nearly all kernels already exist in `engine_gpu/src/`. The genuinely-remaining items are:
- **GPU cold-state replay Phase 4** (memory + chronicle handlers per `spec/engine.md` §10) — small umbrella tail; could fold into Plan 6.
- **Subsystem 3 — Ability evaluation on GPU** (`pick_ability` kernel + `ability::tag(TAG)` scoring grammar + `per_ability` row type per `spec/engine.md` §11) — this IS a real not-yet-ported kernel.

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

**Roadmap source:** `docs/ROADMAP.md` "Drafted (spec exists, plan does not)" tier; `docs/spec/engine.md` §11.

**Spec exists.** Concrete deliverables: `pick_ability` kernel emitted from DSL → WGSL, new `ability::tag(TAG)` scoring grammar primitive, `per_ability` row type. This is the one genuinely-missing GPU kernel referenced in the Plan 7+ entry above.

**Suggested scope:** ~10 tasks (kernel emit + scoring grammar extension + parity test). Could be the natural successor to Plan 6 in the GPU stack.

**Status:** awaiting user
**To proceed:** add `**APPROVED:**` (often deferred until Plan 6 lands).

---
