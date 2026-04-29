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

## 2026-04-28 — plan-writer: GPU execution recovery (post-dispatch-emit)

**Background.** Dispatch-emit Plan was marked COMPLETE 16/16 but in retrospect over-aggressively. T16 (commit `4474566c`) deleted the hand-written kernel modules and got critics PASS — but it also broke `cargo build -p engine_gpu --features gpu`. Commit `984a6e5a` repaired the build honestly (no shims, no `unimplemented!()`):

- `step()` now CPU-forwards via `engine::step::step` — honest fallback, P10-clean
- `cascade.rs` (981 lines) and `cascade_resident.rs` (1,602 lines) DELETED — they orchestrated kernels that no longer exist; redundant with the SCHEDULE-loop dispatcher (T15)
- 93 tests across 29 files file-scope-gated (`#![cfg(any())]` + reason banner). Bodies preserved verbatim.
- Real Pod-type helpers extracted to `crates/engine_gpu/src/sync_helpers.rs` (no stubs)

**The cost:** GPU isn't actually executing anymore in `step()`. The dispatch-emit plan delivered the architectural shape (single source of truth, SCHEDULE-loop dispatcher, emit pipeline) but parity execution is parked behind three concrete work streams:

### Stream A — Promote `step_batch` to ComputeBackend entry point

**Plan written:** `docs/superpowers/plans/2026-04-28-gpu-step-batch-entry.md` (Task 1, 7 steps). User approved 2026-04-28.

### Stream B — Fill emitted WGSL bodies (partial — checkpoint)

**Plan written:** `docs/superpowers/plans/2026-04-28-emitted-wgsl-body-fill.md` (8 tasks). Approved 2026-04-28.

**Landed (commits db689628 → 69c59e3a):**
- Naga parse-validation gate for every `engine_gpu_rules/src/*.wgsl`
- `dsl_compiler::emit_runtime_prelude_wgsl` — recovered + adapted prelude module (keystone)
- 7 kernels with real WGSL bodies:
  - `alive_pack.wgsl` (32 lines) — Bucket R, no prelude
  - `fold_standing.wgsl` (125 lines) — self-contained TopK emitter
  - `fold_memory.wgsl` (79 lines) — self-contained Ring emitter
  - `movement.wgsl` (256 lines) — prelude-driven
  - `apply_actions.wgsl` (~270 lines) — prelude-driven
  - `seed_indirect.wgsl` (~30 lines) — cascade control, no prelude
  - `append_events.wgsl` (~25 lines) — cascade control, no prelude

**Outstanding (split into separable follow-up plans):**

1. **`physics.wgsl` runtime layer** — `emit_physics_wgsl` produces 24 per-rule bodies that reference an "integration-phase" runtime not yet built. Needs:
   - Typed `EventSlot` view of `current_event_ring` (raw u32 → struct)
   - State accessors (`state_agent_hp`/`set_hp`/`kill_agent`, gold add, slot_of, alive_bit)
   - `EVENT_KIND_*` + `EFFECT_OP_KIND_*` const blocks (~30 consts each)
   - `wgsl_world_tick`, ability-registry helpers, view-read fns
   - Physics-side `gpu_emit_event` writing to `next_event_ring` (the prelude writes to `event_ring_records` — different destination)
   - Stubs for standing/memory/cast paths (their backing stores aren't yet wired)

   Estimated 2-4 hours. Real engineering. Suggest: dedicated plan `2026-04-XX-physics-wgsl-runtime.md`.

2. **6 PairMap/SlotMap fold modules** (`fold_engaged_with`, `fold_threat_level`, `fold_kin_fear`, `fold_my_enemies`, `fold_pack_focus`, `fold_rally_boost`). The `emit_view_fold_wgsl` emitter references undefined `view_agent_cap` and view-read helpers. Needs view-storage primitive layer. Likely shares prelude with physics. Estimated 1-2 hours.

3. **3 spatial kernels** (`spatial_hash`, `spatial_kin_query`, `spatial_engagement_query`). Pre-T16 was 326-line single-kernel-3-entry-points; new design is 3 separate kernels with reduced bindings. Major rewrite. Estimated 3-5 hours.

4. **3 fused-output unpack kernels** (`mask_unpack`, `scoring_unpack`, `fused_agent_unpack`) — architectural mismatch: new fused output bindings (`agents_soa`, `agent_data`) don't map to consumer (e.g. `fused_mask` reads 3 separate buffers). Needs design call before adaptation. Suggest: separate spec drafting before plan.

5. **`pick_ability.wgsl`** — emitter wired but produces empty fallback (no per_ability rows in DSL). Falls under "Ability DSL implementation" follow-up plan, not Stream B.

#### Original framing

Multiple emitted kernel modules in `engine_gpu_rules/src/` have placeholder WGSL bodies because xtask couldn't pull `engine_gpu` (with the `gpu` feature) into its compile graph during T11/T12/T13 (chicken-and-egg with the pre-existing breakage we just resolved). Now that the build is clean, the emitter can be wired to read the real `pub const *_WGSL` constants OR to call into `dsl_compiler` shader emitters that produce them.

Affected modules (placeholder bodies): inspect each `engine_gpu_rules/src/*.wgsl` — most start with a `// GENERATED` comment + a stub. Specifically the spatial set (T12), alive_pack/fused_agent_unpack (T13), megakernel scaffold (T14 — intentional, owned by gpu_megakernel_plan), view fold modules (T11).

The constants `ALIVE_PACK_WGSL` and `FUSED_AGENT_UNPACK_WGSL` were hoisted to `pub const` in T13 specifically for this; they currently live in `crates/engine_gpu/src/{alive_bitmap,mask}.rs` — those files were deleted in T16, so the constants need re-extracting from git history (`git show 4474566c~1`) into a small `engine_gpu_rules`-side data module OR moving to `dsl_compiler`'s emitter source.

Estimated scope: 4-6 tasks (one per kernel module class).

### `step_batch` runtime fix (CLOSED 2026-04-28, commit dd21f9b9)

Plan: `docs/superpowers/plans/2026-04-28-step-batch-runtime-fix.md`

Discovered during Stream C Task 1 prep: `engine_gpu::step_batch` panicked on every call because its per-tick CPU forward invoked `engine::step::step` which is `unimplemented!()` (Plan B1' Task 11 deletion). T15+T16+Stream A landed clean compiles but never proved runtime.

Closed via commit `dd21f9b9`:
- `GpuBackend::ComputeBackend::Views`: `()` → `ViewRegistry` (both no-gpu + gpu impls)
- `step_batch` signature gains `views: &mut ViewRegistry`; cascade generic `<Event, ()>` → `<Event, ViewRegistry>`
- 3 `engine::step::step` call sites in `engine_gpu/src/lib.rs` migrated to `engine_rules::step::step` (passing `&mut SerialBackend` as inner CB)
- 2 `engine::step::step` call sites in `viz/src/state.rs` migrated; `views: ViewRegistry` field added to `AppState`
- New no-gpu `step_batch` method so default-features callers compile
- New runtime smoke test `step_batch_runtime_smoke` (NOT cfg-gated) — 2/2 PASS, proves `step_batch` executes end-to-end
- Plan template updated: `docs/architecture/plan-template-ais.md` now requires a "Runtime gate" entry for every plan touching per-tick code paths
- Retrospective added to dispatch-emit plan capturing the lesson

### Stream C — Port the cfg-gated tests (rescoped 4 tasks; CLOSED 2026-04-28)

**Plan written:** `docs/superpowers/plans/2026-04-28-gpu-test-port.md` (4 tasks, rescoped commit ed046d3c).

**Completed:**
- Task 1 (commit `a6c8a380`): parity helper + `parity_with_cpu` (3 tests at t=1, t=10, t=100; all PASS)
- Task 2 (commit `37206b0d`): 5 smoke tests (`step_batch_smoke`, `tick_advance_is_gpu_resident` default-features; `async_smoke`, `snapshot_double_buffer`, `alive_bitmap_pack` gpu-feature-only)
- Task 3 (RNG cross-backend gate): host-side P5 tests in `crates/engine/src/rng.rs` (6 inline unit tests) pass; verified. Cross-backend assertion against `per_agent_u32_glsl` is gated on the GPU runtime prelude growing that helper — currently the prelude only emits event-emission helpers, not RNG. Migrated to the physics-wgsl-runtime follow-up plan (which already needs a comprehensive runtime layer that would include `per_agent_u32_glsl`).
- Task 4 (closeout): this entry update.

**Outcome:** the 7 Stream-B-landed kernels (alive_pack, fold_standing, fold_memory, movement, apply_actions, seed_indirect, append_events) are now regression-gated by `parity_with_cpu` end-to-end and 5 smoke tests cover the dispatch path. Future kernel work cannot silently break the dispatcher or per-tick state advancement.

### Deferred test categories (own follow-up plans)

Per Stream C plan §"What this plan deliberately does NOT do", these remain file-cfg-gated:

- Per-kernel parity tests for stubbed kernels (`physics_parity`, `cascade_parity`, `view_parity`, `topk_view_parity`, `spatial_parity`) → owned by `physics-wgsl-runtime` / `view-fold-helpers` / `spatial-rewrite` follow-up plans
- Cold-state replay (`cold_state_*`) → physics-wgsl-runtime
- Chronicle pipeline (`chronicle_batch_*`, `chronicle_drain_perf`, `chronicle_isolated_smoke`, `chronicle_batch_probe`) → physics-wgsl-runtime + chronicle-rewire
- Perf benches (`gpu_step_perf`, `perf_n100`, `chronicle_batch_perf_*`) → perf-bench-rebuild
- `gpu_prefix_scan` → primitive retired in T16; mark `#[ignore]` if reintroduced
- GPU-side `per_agent_u32_glsl` → physics-wgsl-runtime (its prelude needs RNG)

#### Original framing

Tests reference deleted symbols: `crate::mask::cpu_mask_bitmap`, `crate::physics::run_batch_resident`, `state.views.*`, `ChronicleRing` layout, `crate::scoring::cpu_score_outputs`, etc. Each needs rewriting against the SCHEDULE-loop surface — sourcing inputs via `BindingSources`, dispatching via the kernel's `Kernel::record`, decoding outputs via `sync_helpers::unpack_agent_slots` etc.

Test inventory (29 files, all under `crates/engine_gpu/tests/`): alive_bitmap_pack, async_smoke, batch_iter_cap_convergence, cascade_parity, chronicle_batch_*, cold_state_*, event_ring_parity, gpu_prefix_scan, gpu_step_perf, indirect_cascade_converges, parity_with_cpu, perf_n100, physics_*, snapshot_double_buffer, spatial_*, step_batch_smoke, tick_advance_is_gpu_resident, topk_view_parity, view_parity.

Suggested grouping:
- Tier 1 (parity gates, P3): parity_with_cpu, physics_parity, cascade_parity, view_parity, topk_view_parity, spatial_parity
- Tier 2 (smoke): step_batch_smoke, alive_bitmap_pack, async_smoke, snapshot_double_buffer, tick_advance_is_gpu_resident
- Tier 3 (correctness): cold_state_*, indirect_cascade_converges, event_ring_parity
- Tier 4 (perf, lowest priority): gpu_step_perf, perf_n100, chronicle_batch_perf_*

Estimated scope: 10-15 tasks (one per test or pair, plus a shared rewrite-helper task).

### Suggested plan structure

3 sub-plans, executable in parallel after Stream A lands:

1. `2026-04-XX-gpu-step-batch-entry.md` — Stream A (small, unblocks A11/A12 dependents)
2. `2026-04-XX-emitted-wgsl-body-port.md` — Stream B
3. `2026-04-XX-gpu-test-port.md` — Stream C

Streams B and C are independently parallelizable once A lands. Total scope: ~20 tasks across 3 plans.

**Why this matters:** Plan invariant #5 (every parity test passes) is currently unverifiable. Default-features tests pass; GPU correctness is uncovered until tests rebuild against the new surface. Without Stream A, GPU is functionally inert in `step()`.

**Status:** Streams A, B, C all planned and approved (2026-04-28). Execution order: C1 → B1 → B2 → B3 → C2 → B4 → B5 → B6 → B7 → C3 → B8 → C4 → C5. Total: 14 task-pairs.
**To proceed:** execute per the interleave; or hand off to subagent-driven-development at the next /dag-tick.

---
