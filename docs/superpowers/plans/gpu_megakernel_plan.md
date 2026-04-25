# GPU Megakernel Plan

Umbrella for task #159. Port the DSL-emitted simulation layers (masks, scoring, physics, views) to GPU via `naga` → WGSL, with the CPU path preserved as reference and default.

## Reality check

A literal single-dispatch megakernel isn't achievable — WGSL has no device-wide barrier, so the cascade fixed-point can't run inside one kernel. The practical shape is **~3-5 sub-dispatches per tick**, each fusing all rules for one cascade iteration. Still a 40× reduction from the naive 200+ dispatches a 1:1 lift would produce.

## Rationale

CPU path today: 84K-787K ticks/sec at N=5-20 agents. The bottleneck is agent count, not tick rate. Target: break through the N-ceiling toward 1k-10k agents with useful throughput. 200k+ requires additional sparsification work (pair-map views become 640GB) — deferred.

## Architecture

Backend trait in `crates/engine/`:

```rust
pub trait SimBackend {
    fn step(&mut self, state: &mut SimState, scratch: &mut SimScratch,
            events: &mut EventRing, policy: &dyn PolicyBackend,
            cascade: &CascadeRegistry);
}
```

`CpuBackend` is today's `step::step` wrapped. `GpuBackend` (new `crates/engine_gpu/`) drives the sub-dispatch cascade.

GPU path:
1. CPU writes tick input (agent pool, event ring, view state) to GPU buffers (or keeps them resident)
2. For each cascade iteration until event ring stops growing:
   - One dispatched fused kernel covering: mask eval → scoring → argmax → physics rules → view folds
   - CPU reads event-ring atomic counter
3. CPU reads final state + event-ring contents for chronicle/tests

## Dependencies (feature-gated)

- `wgpu = "=26.x"` (pin minor, prevent silent re-resolves)
- `naga = "=26.x"`
- Isolated to a new `engine_gpu` crate. Enable with `cargo build --features gpu`.
- CPU-only builds stay dep-clean — no wgpu/naga leak into `engine` crate.

## Phased implementation

Each phase leaves the tree in a shippable state. CPU path default throughout.

### Phase 0 — Scaffolding + parity harness

- `trait SimBackend` in `engine`
- `CpuBackend` wrapping existing `step::step`
- `engine_gpu` crate with stub `GpuBackend` (identity, no actual GPU work yet)
- `--gpu` flag on `xtask chronicle` routes through `GpuBackend`
- Parity harness in tests: run N ticks via CPU backend, clone state, run same N ticks via GPU backend, assert byte-identical `SimState` + `EventRing`
- Passes trivially since GPU is CPU-forwarding; catches regressions once real GPU work lands

### Phase 1 — Attack mask E2E

- `emit_mask_wgsl.rs` mirrors `emit_mask.rs` — same AST walk, WGSL output
- wgpu device/queue setup, buffer pool
- Upload agent state, dispatch Attack mask kernel, read back action bitmap
- Parity harness passes with GpuBackend doing real work for Attack, CPU for everything else

### Phase 2 — Fuse all masks

- All 8 masks in one compute shader, branch by action kind
- Single dispatch per tick for mask phase

### Phase 3 — Scoring

- action × target × agent evaluation
- Argmax reduction on GPU (deterministic — sort-by-id before reduce, or segmented scan)

### Phase 4 — Views

- 9 view storage buffers (pair_map, slot_map variants)
- Fold events with deterministic accumulation
- Decay applied on read (`tick - last_update` in the get function) — no separate decay dispatch

### Phase 5 — Spatial hash on GPU

- Rebuild per tick from agent positions (fast at current scale)
- Support `query.nearest_hostile_to`, `query.nearby_kin` inside kernels

### Phase 6 — Physics + event ring

- Event ring as GPU buffer with atomic tail counter
- Physics rules fused, branch on event kind
- Cascade = CPU loops `dispatch → read tail → if grew, dispatch again`
- Eager view writes happen inside physics kernel for same-tick read-your-own-writes

### Phase 7 — Determinism

- GPU xorshift must match CPU byte-for-byte
- Event ordering deterministic: sort by (tick, event_kind, actor) within each cascade iteration
- All 373+ engine tests pass on both backends

### Phase 8 — Perf harness + handoff

- `xtask chronicle --bench --gpu` vs CPU
- Measure crossover agent count (where GPU wins)
- Document the number; default remains CPU

## Determinism policy

- Atomic adds in view folds break determinism on GPU. Use segmented reduction (sort by view key, then reduce) or accept that folds commutative-associative by construction (all current folds are `self += constant` — deterministic under any associative reduction)
- RNG: one seed per agent, deterministic xorshift, no shared mutable RNG
- Event ring ordering: sort by a stable key each cascade iteration before appending to the replay log
- Parity test: byte-exact `EventRing` between CPU and GPU at every tick, not just final state

## What ports easily (DSL emitter — ~30-40% of work)

- `emit_mask.rs` → `emit_mask_wgsl.rs` (same AST walk, different output)
- `emit_physics.rs` → `emit_physics_wgsl.rs`
- `emit_scoring.rs` → `emit_scoring_wgsl.rs`
- `emit_view.rs` → `emit_view_wgsl.rs`
- Task 158's GPU-emittability validator already guarantees the subset lifts cleanly

## What's hard (engine — 60-70% of work)

- GPU buffer layout for `SimState` (SoA on hot arrays is already close)
- `EventRing` with atomic tail
- Spatial hash GPU implementation
- Cascade sub-dispatch loop
- **Determinism engineering** — where weeks hide

## Build time mitigation

- `engine_gpu` as separate crate behind `--features gpu` — CPU-only dev loop unaffected
- Pin `wgpu` / `naga` minor versions
- Runtime WGSL parse, not build-time (saves ~100ms per build)
- `sccache` / `cargo-chef` in CI for dep caching

## Non-goals (explicit)

- **Sparse pair-map views** — needed at 10k+ agents, not this plan. Defer as task #182 or similar.
- **Spatial time-warp / optimistic PDES** — needed at 200k+ with large-map deployment. Defer.
- **Persistent kernel** — only if megakernel dispatch becomes the measured bottleneck. Defer.
- **Multi-GPU / distributed** — not in scope.

## Estimated scope

~10-14 tasks of the session-160-180 size. DSL-emitter half parallelizes well via subagents; engine half is sequential architectural work.

## Next action

Task 181: Phase 0 scaffolding + parity harness.
