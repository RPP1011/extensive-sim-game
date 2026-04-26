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

## 2026-04-26 — pivot-point: Subsystem 3 Group A → dispatch-emit brainstorm

**Trigger:** When Subsystem 3 Task A3 closes, the next dag-tick will claim Task B1 (`GpuPickAbilityEntry` upload types + `PickAbilityKernel` struct). **Do NOT dispatch B1.** Instead, pivot to a brainstorm session on dispatch-emit (kernel-wrapper automation).

**Rationale:** Group A (A1–A3) is pure compiler-emit work — emits Rust + WGSL for the `pick_ability` kernel from the DSL. Group B (B1–B5) is hand-written engine-core wiring: `PickAbilityKernel` Rust wrapper, buffer ownership in `ResidentPathContext`, dispatch sequencing in `step_batch`, binding-group construction. This is the same boilerplate every existing kernel has (`ScoringKernel`, `ApplyActionsKernel`, `MovementKernel`, `PhysicsKernel`, etc.) — currently hand-written by convention.

**Pipeline justifying the pivot:** ~15–25 more kernels coming up:
- Subsystem 3 `pick_ability` (in flight)
- 8 fold kernels (`cs_fold_<view>` per spec §12.3) — highly homogeneous; gold + standing landed, 6 still pending
- Cold-state replay Phases 2–4 (chronicle + remaining fold kernels)
- Ability DSL implementation (voxel ops, control verbs, structures, materials, passive triggers — each `EffectOp` GPU-side)
- NeuralBackend matmul kernels (currently `todo!()`)
- Economic depth Phase 1 (market matching, contract resolution)

The 8 fold kernels alone are a textbook first user — they all share binding-layout shape (events_in, view_storage_out, sim_cfg, agent_buf), workgroup pattern, and parity-test scaffolding. Hand-writing 8 nearly-identical wrappers is exactly the kind of waste an emitter eliminates.

**Pivot decision: Subsystem 3 Group B becomes the *first* user of dispatch-emit.** The 8 fold kernels become the *second* user. Plan 6 (`GpuBackend` foundation) likely subsumes into dispatch-emit rather than being a separate plan.

**Status:** awaiting brainstorm
**To proceed (after A3 lands):** spawn brainstorm session via `superpowers:brainstorming` skill targeting "kernel dispatch-emit abstraction"; deliverable is a design spec at `docs/superpowers/specs/<date>-kernel-dispatch-emit-design.md`. Subsystem 3 B1–D2 stay pending until the spec → plan cycle completes; they'll either be re-implemented under the new abstraction or left as the canonical hand-written reference for migration.

---

## 2026-04-26 — plan-writer: Plan 6 — `GpuBackend` foundation

**Roadmap source:** `docs/ROADMAP.md` "Engine plans not yet written" tier; depends on Plan 5 complete.

**Spec exists** at `docs/spec/engine.md` (GPU annexes §§9–12). Goal: bridge the `ComputeBackend` trait's kernel-dispatch surface (after 5b–e) to the existing `engine_gpu` primitives (kernels under `crates/engine_gpu/src/`). Most of the underlying GPU kernels already exist (megakernel + cold-state replay landed); this is mostly plumbing — wiring `GpuBackend` impl methods to invoke the right kernels for each backend trait method.

**Note 2026-04-26:** likely subsumed into dispatch-emit (see pivot-point entry above). Keep in pending until dispatch-emit brainstorm decides whether Plan 6 stays as a standalone plumbing plan or becomes redundant.

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
