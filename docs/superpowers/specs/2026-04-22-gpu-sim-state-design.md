# GPU Sim State (SimCfg + `@cpu_only` annotation)

**Status:** design approved, plan pending
**Date:** 2026-04-22
**Subsystem:** (1) of the GPU-everything follow-up
**Prerequisite for:** subsystems (2) and (3)

## Problem

After the GPU-resident cascade refactor, `step_batch(n)` runs N ticks GPU-resident with one submit per batch. But two pieces of simulation state still live on the CPU and are uploaded each tick via `queue.write_buffer`:

- `state.tick: u32` — advanced by `state.tick.wrapping_add(1)` in `step_batch`'s loop, then shipped to each kernel's cfg uniform.
- `state.rng_state: u64` — CPU-resident RNG state consumed by kernels that need per-agent random draws (movement jitter, event ordering).

Additionally, world-scalar fields (`engagement_range`, `attack_damage`, `attack_range`, `move_speed`, `move_speed_mult`, `kin_radius`, `cascade_max_iterations`) are duplicated across every per-kernel cfg uniform struct. A change to any of these requires touching ~5 kernels.

There is also no DSL-level escape hatch for rules that must stay CPU-resident (chronicle narrative text formatting, debug-only side effects). The GPU-emittable validator rejects them silently by type, which means the boundary is implicit rather than declared.

## Goal

A shared GPU-resident `SimCfg` storage buffer owning tick, world seed, and the cross-kernel world-scalar fields. Tick advance happens GPU-side via an atomic increment in the existing seed-indirect kernel. RNG follows because `per_agent_u32` is already a pure hash of `world_seed + tick + agent_id` — upload `world_seed` once at init, and the existing RNG function reads the current GPU tick.

A new DSL annotation `@cpu_only` marks rules that stay CPU-resident by design. The compiler emits only the CPU handler, skips WGSL emission, and omits the rule from the GPU event-kind dispatcher table.

`GpuBackend` is refactored into three sub-structs (`sync`, `resident`, `snapshot`) as part of this work — it has 24 fields today and subsystems (2)/(3) will add more; factoring now is cheaper than deferring.

## Non-goals

- Moving RNG state to a GPU atomic (the existing pure-hash approach makes this unnecessary — RNG is stateless given seed + tick).
- Eliminating `state.tick` from the CPU side entirely. CPU callers that read `state.tick` (sync `step()` path, test helpers) continue to see a valid value; it's just defined as "wall-clock as of last observer sync" during batch mode.
- Porting any game logic. This subsystem is pure infrastructure.
- Reducing the per-kernel cfg uniforms to zero. Kernel-local parameters (workgroup size, per-call slot indices, per-kernel thresholds) stay in their own small uniforms. SimCfg holds only sim-wide scalars.

## Architecture

### The `@cpu_only` annotation

Apply at the rule declaration site in DSL source files. Example:

```
@cpu_only physics chronicle_render_attack_narrative @phase(event) {
    on AgentAttacked { ... build String ... push chronicle text ... }
}
```

Compiler behaviour:
1. Parses the annotation, records it on the rule's IR node.
2. **Emits the CPU handler** as today (Rust function, registered in `CascadeRegistry::with_engine_builtins()`).
3. **Skips WGSL emission** for this rule. No entry in the GPU physics kernel's event dispatch table. No entry in the per-event-kind GPU dispatcher table that subsystem (2) introduces.
4. **Updates the GPU-emittable validator** to accept primitives it would otherwise reject (strings, unbounded allocation) inside `@cpu_only` rule bodies, since they'll never be lowered to WGSL.

Naming: `@cpu_only` wins over `@cpu`, `@async_cpu`, `@off_gpu` because it contrasts explicitly with the implicit default (`@gpu_emittable`). The implicit-`@gpu_emittable` default is unchanged — existing rules don't need annotations to keep working.

### `SimCfg` layout

One storage buffer, bound by every kernel that reads a sim-wide scalar. Initial shape (~64 bytes):

```
struct SimCfg {
    tick:                          u32,
    world_seed_lo:                 u32,
    world_seed_hi:                 u32,
    _pad0:                         u32,
    engagement_range:              f32,
    attack_damage:                 f32,
    attack_range:                  f32,
    move_speed:                    f32,
    move_speed_mult:               f32,
    kin_radius:                    f32,
    cascade_max_iterations:        u32,
    rules_registry_generation:     u32,
    abilities_registry_generation: u32,
    _reserved:                     [u32; 4],
}
```

`_reserved` gives 16 bytes of headroom — enough for ~4 additional sim-wide u32 or f32 fields before the next layout bump. Subsystems (2)/(3) are expected to consume some of this headroom (cold-state epoch, ability-registry cache key) but they're already called out as first-class fields above.

`rules_registry_generation` and `abilities_registry_generation` are cache-invalidation u32 counters. Incremented by the CPU when the `CascadeRegistry` or `PackedAbilityRegistry` changes shape. Used by kernel-side caches to skip redundant re-uploads. `abilities_registry_generation` has immediate utility — the perf gap analysis's suspect 4 (ability upload cache) uses it as the cache key.

Storage binding (not uniform) because the tick field is written by the GPU atomic. Read-mostly from other kernels; atomic writes happen only in the seed-indirect kernel.

### Tick advance

The existing seed-indirect kernel in `cascade_resident.rs` already runs once per tick to seed the cascade's first iteration. It grows one line:

```wgsl
atomicAdd(&sim_cfg.tick, 1u);
```

at end-of-tick. No new kernel needed. No new dispatch.

### RNG

`per_agent_u32(agent_id, purpose_tag)` is currently a pure hash of `(world_seed, tick, agent_id, purpose_tag)`. Migration is entirely mechanical:

- `world_seed` uploaded once at `ensure_resident_init` to `SimCfg.world_seed_{lo,hi}`.
- `tick` read from `SimCfg.tick` instead of the per-kernel cfg uniform field.
- No atomic state mutation. The function stays pure.

### CPU-side `state.tick` during batch

Stays stale across the whole batch by design (option (a) from the brainstorm). `step_batch` no longer does `state.tick.wrapping_add(1)` in its loop. At end of batch, `state.tick` is unchanged from start-of-batch. The next `snapshot()` call reads `SimCfg.tick` and reports it via `GpuSnapshot.tick`. Callers that want CPU `state.tick` to reflect reality call `snapshot()` first.

The sync `step()` path is unaffected — it continues to advance `state.tick` on the CPU side each call, since the sync path isn't using `SimCfg` yet. (Migrating sync to `SimCfg` is out of scope for this subsystem; acceptable because sync path already pays a per-tick CPU/GPU sync anyway.)

### `GpuBackend` sub-struct factoring

Rework `GpuBackend` from its current 24-field flat layout into:

```rust
pub struct GpuBackend {
    device: Arc<wgpu::Device>,
    queue:  Arc<wgpu::Queue>,
    backend_label: String,
    sync:     SyncPathContext,      // mask_kernel, scoring_kernel, view_storage, cascade_ctx, last_* diagnostics
    resident: ResidentPathContext,  // resident_agents_buf, resident_indirect_args, resident_cascade_ctx, sim_cfg_buf, mask_unpack, scoring_unpack
    snapshot: SnapshotContext,      // snapshot_front/back, watermarks, latest_recorded_tick
}
```

Each sub-struct owns its own lazy-init method. `GpuBackend` methods delegate. Existing callers of `step()`, `step_batch()`, `snapshot()` keep the same entry points — only internal organisation changes.

### Data flow — one tick in the batch path (post this subsystem)

```
(ensure_resident_init — once):
    SimCfg.world_seed_{lo,hi} ← state.world_seed
    SimCfg.engagement_range   ← state.config.combat.engagement_range
    ...all world-scalar fields uploaded once...
    SimCfg.tick               ← state.tick at batch entry

(per tick, on GPU, all within one encoder):
    mask → scoring → apply → movement → spatial → cascade(N iters)
    seed-indirect (end of tick) writes: atomicAdd(&sim_cfg.tick, 1u)
```

No CPU `queue.write_buffer` for tick. No per-tick CPU state mutation.

## Components

### New files

- `crates/engine_gpu/src/sim_cfg.rs` — `SimCfg` struct (bytemuck Pod), buffer allocation, initial upload helper.
- `crates/engine_gpu/src/backend/mod.rs`, `sync.rs`, `resident.rs`, `snapshot_ctx.rs` — sub-struct factoring. `GpuBackend` becomes a thin composite. Move existing fields into appropriate sub-structs.

### Modified files

- `crates/engine_gpu/src/cascade_resident.rs` — seed-indirect kernel gains the `atomicAdd` line. `SeedIndirectKernel` binds `SimCfg` as a storage buffer.
- `crates/engine_gpu/src/lib.rs` — `GpuBackend` struct replaced by the composite. `step_batch` no longer does `state.tick += 1`. `snapshot()` reads `SimCfg.tick` for `GpuSnapshot.tick`.
- All kernels (mask, scoring, apply, movement, physics, spatial): replace reads of duplicated per-kernel cfg fields (engagement_range, attack_damage, etc.) with reads from `SimCfg`. Per-kernel cfg uniforms shrink to kernel-local fields only.
- `crates/dsl_compiler/src/` — parse `@cpu_only`, thread it through IR, skip WGSL emit for annotated rules, update the GPU-emittable validator.

### Engine-core vs DSL-lowered split (for this subsystem)

- **Engine-core (hand-written)**: SimCfg struct, buffer allocation, seed-indirect's `atomicAdd` line, sub-struct factoring. Infrastructure.
- **DSL-lowered**: nothing new — this subsystem doesn't port any game rules. Just adds the `@cpu_only` annotation plumbing for future subsystems to use.

## Error handling

- **`SimCfg` field drift**: WGSL struct layout must match Rust struct layout (alignment, padding). Enforced by a compile-time assertion comparing field offsets to hand-written constants in WGSL. Drift panics at startup.
- **`rules_registry_generation` overflow**: u32 counter, 4B-tick lifetime at 1 Hz increment rate. Wraparound is fine for cache-invalidation purposes; consumers key caches on equality, not ordering.
- **Sub-struct factoring regression**: sync path's behaviour must be byte-identical post-factoring. Enforced by running `parity_with_cpu` + `physics_parity` + `cascade_parity` in CI.

## Testing

### New tests

- `sim_cfg_layout.rs` — unit test asserting `SimCfg`'s Rust-side struct layout (offset of each field) matches the WGSL struct layout. Regression fence.
- `tick_advance_is_gpu_resident.rs` — integration test: `step_batch(100)`, then `snapshot()`, assert `snap.tick == start_tick + 100`. Before this subsystem, `step_batch` advanced `state.tick` on CPU; after, `state.tick` stays at `start_tick` throughout, only `snap.tick` reflects the batch.
- `cpu_only_annotation_skips_wgsl_emit.rs` — DSL-compiler-level test: a rule annotated `@cpu_only` produces a CPU handler in `CascadeRegistry` and no entry in the GPU dispatcher table.

### Regression tests kept

Every existing test continues to pass. The sync `step()` path doesn't use `SimCfg` yet and behaves as before. `parity_with_cpu` must stay green post the sub-struct factoring.

### Non-goals for testing

- No parity test between CPU tick advance and GPU tick advance — they're known to diverge by one frame of lag during a batch (CPU stays at start_tick; GPU advances).
- No byte-exact replay test for batch-mode RNG. It's explicitly non-deterministic across GPUs (see GPU-resident cascade spec).

## Phase decomposition

One subsystem, two plannable phases:

**Phase 1 — `GpuBackend` sub-struct factoring.** Addresses the final code review's I4 item. No behaviour change. All regression tests stay green. Lands first because every subsequent change to `GpuBackend` is easier after factoring.

**Phase 2 — `SimCfg` + `@cpu_only` annotation + seed-indirect tick advance.** Adds the buffer, migrates all world-scalar field reads to `SimCfg`, adds the annotation plumbing, wires the atomic tick advance. `step_batch` stops advancing `state.tick` on CPU.

Both phases land in this subsystem's plan. Subsystem (2) starts after Phase 2 lands (it consumes `@cpu_only`).

## Open questions

- **Kernel-local cfg uniforms: how much shrinks?** After moving world-scalars to `SimCfg`, each kernel's local cfg uniform should be much smaller (only slot indices, workgroup sizes, per-call parameters). Worth measuring to make sure we're not over-engineering — if a kernel's local cfg becomes empty, we drop the binding.
- **Sync path migration to `SimCfg`**: out of scope for this subsystem. Defer until we want to retire the sync path's per-tick CPU increment too. Realistically only relevant if sync path becomes a performance bottleneck — currently it's the deterministic reference and doesn't need the optimization.
- **`@cpu_only` telemetry**: should the compiler emit a manifest of `@cpu_only` rules so runtime telemetry can count CPU-side executions? Probably yes as a small follow-up, not in this subsystem.
