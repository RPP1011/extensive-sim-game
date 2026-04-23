# GPU-resident cascade with snapshot-based observation

**Status:** design approved, plan pending
**Date:** 2026-04-22
**Scope:** `crates/engine_gpu` — additive, does not modify `SimBackend::step()` semantics.

## Problem

The current GPU backend (`engine_gpu::GpuBackend`) runs ~12 CPU/GPU round-trip fences per tick:

- mask + scoring: 2 readbacks
- spatial hash (×2 queries): ~4 readbacks (incl. CPU exclusive-scan between count and scatter)
- apply_actions + movement: 2 readbacks
- cascade physics iterations: ~1 readback per iter × ~3 iters/tick
- view fold final: 1 readback

At N=100k this dominates the steady-state: measured `gpu cascade: 4.5 s/tick` on a clean run, of which most is serialised readback latency, not kernel arithmetic. Each fence is a `device.poll(PollType::Wait)` that blocks the Rust driver thread until the GPU queue drains.

The cascade itself is also CPU-driven: `run_cascade` in `crates/engine_gpu/src/cascade.rs:187` is a Rust `for` loop that uploads `events_in`, dispatches physics, reads back `agent_slots_out` + `events_out`, then re-uploads for the next iteration.

## Goal

An additive batch API that runs the simulation GPU-resident for N ticks with zero per-tick CPU fences, and exposes state to the CPU via a cheap non-blocking snapshot. The existing `SimBackend::step()` sync path is preserved unchanged for tests, chronicle seed reproducibility, and the scenario/parity harnesses.

Success criterion: at N ≥ 512, batch mean µs/tick < 0.8× sync mean µs/tick. Non-goal: byte-exact parity between sync and batch paths.

## Non-goals

- Player actions / write-back of CPU-initiated state mutations. The batch API is read-only observation; mutation is a future spec.
- Replacing the sync `step()` path. It stays load-bearing for parity tests and deterministic chronicle output.
- Byte-exact GPU↔CPU parity. The batch path is explicitly non-deterministic in event fold order (same drift the sweep already logs).
- Cross-GPU reproducibility. Same hardware + same seed should reproduce; different GPUs may diverge due to non-commutative GPU folds.

## Architecture

**Shift in principle**: today every GPU kernel's outputs are copied to CPU, and the next kernel's inputs are re-uploaded from CPU. The batch path binds output buffers directly as the next kernel's inputs; CPU only observes when explicitly asked.

### New public surface

```rust
impl GpuBackend {
    /// Run N ticks GPU-resident. Records all N ticks (plus up to
    /// MAX_CASCADE_ITERATIONS indirect physics dispatches each) into
    /// one command buffer, submits once, polls once at end.
    /// Non-deterministic in event order. Does not populate the
    /// caller's EventRing — observe state via `snapshot()`.
    pub fn step_batch(&mut self, state: &mut SimState, n: u32,
                      cascade: &CascadeRegistry) -> Result<(), BatchError>;

    /// Cheap non-blocking snapshot via double-buffered staging.
    /// First call returns an empty snapshot. Subsequent calls return
    /// the state as-of the *previous* snapshot call (one frame lag).
    pub fn snapshot(&mut self) -> Result<GpuSnapshot, SnapshotError>;
}

pub struct GpuSnapshot {
    pub tick: u32,
    pub agents: Vec<GpuAgentSlot>,
    pub events_since_last: Vec<EventRecord>,
    pub chronicle_since_last: Vec<ChronicleEntry>,
}
```

### Data flow — one tick in the batch path

All GPU-resident. No CPU round-trips inside the tick:

```
agents_buf ──▶ [mask] ──▶ mask_bitmaps_buf ──▶ [scoring] ──▶ scoring_buf
                                                                 │
                                                                 ▼
agents_buf ◀── [movement] ◀── [apply_actions] ◀─────────────────┘
      │             │                │
      │             │                ▼
      │             │          apply_event_ring_buf
      ▼             │                │
[spatial: count → GPU-scan → scatter → sort → query]
      │                                           │
      ▼                                           │
kin_buf, nearest_buf ──▶ [cascade: N× indirect physics dispatch]
                                      │
                                      ▼
                            physics_event_ring_buf, updated agents_buf
                                      │
                                      ▼
                            [fold_iteration kernels ──▶ view_storage_buf]
                                      │
                                      ▼
                            events accumulate in main_event_ring_buf (GPU)
                            chronicle entries → chronicle_ring_buf (GPU)
```

### Indirect dispatch for cascade iterations

No per-iteration readback of a "converged?" flag. Instead:

- End-of-iter, the physics kernel writes indirect dispatch args `(workgroup_count, 1, 1)` to a small GPU buffer, where `workgroup_count = ceil(num_events_next_iter / PHYSICS_WORKGROUP_SIZE)` clamped to `ceil(agent_cap / PHYSICS_WORKGROUP_SIZE)`.
- When there are no follow-on events, the kernel writes `(0, 1, 1)`. Subsequent indirect dispatches are GPU no-ops (microseconds).
- `run_cascade_resident` pre-records MAX_CASCADE_ITERATIONS indirect dispatches into one encoder.
- `last_cascade_iterations()` becomes an inferred value from the args-buffer readback that rides along with `snapshot()`.

### `step_batch(n)` submit shape

One command encoder. Records N ticks: each tick = mask + scoring + spatial + apply + movement + cascade-indirect×MAX_ITERS + fold + a tiny tick-counter increment dispatch. One `queue.submit`. One `device.poll(Wait)` at end.

Per-tick state that must change (RNG seed, tick counter) lives in a GPU-side `PhysicsCfg` buffer updated by the tick-counter increment kernel, or recorded as `queue.write_buffer` calls inside the same encoder between kernel dispatches.

### `snapshot()` — double-buffered staging

Three staging-buffer pairs: `{agents, events, chronicle}` × `{front, back}`.

On call:
1. Encode `copy_buffer_to_buffer` for current `agents_buf` + `event_ring[last_read..tail]` + `chronicle_ring[last_read..tail]` into the **back** staging buffers. Update `last_read` watermarks.
2. `queue.submit`.
3. `map_async(Read)` on the **front** staging buffers (filled by the previous `snapshot()` call).
4. `device.poll(Wait)` — drives pending map callbacks from the previous frame and the copy just submitted.
5. Decode front staging → `GpuSnapshot`, unmap, swap front/back pointers.

First call returns `GpuSnapshot::empty()` because no previous frame exists to map.

The one-frame lag is acceptable because the rendering layer already interpolates via a delta value.

### Additive surface — existing callers untouched

- `SimBackend::step()` and its implementation on `GpuBackend` are not modified.
- Caller-provided `EventRing` is still populated by the sync path. The batch path does not touch it; batch events are observable only via `snapshot().events_since_last`.
- Existing `parity_with_cpu`, `perf_n100`, `chronicle_drain_perf`, all scenario tests: continue running against the sync path.

## Components

### Kernel modules (existing files, grow one method each)

Each kernel module keeps owning its WGSL, pipeline, and bind group layout. A new `*_resident` method is added alongside the existing `run_and_readback` / `run_batch`:

- `physics.rs` — `PhysicsKernel::run_batch_resident(...)`: indirect dispatch, no readback, outputs stay bound.
- `mask.rs` — `MaskKernel::run_resident(...)`.
- `scoring.rs` — `ScoringKernel::run_resident(...)`.
- `apply_actions.rs` — `ApplyActionsKernel::run_resident(...)`.
- `movement.rs` — `MovementKernel::run_resident(...)`.
- `spatial_gpu.rs` — `GpuSpatialHash::rebuild_and_query_resident(...)` plus a new GPU prefix-scan kernel that replaces the CPU exclusive-scan at `spatial_gpu.rs:917-926`. Also exposes `kin_buf` and `nearest_buf` as public device handles so downstream kernels can bind directly. Alive-filter is pushed into the query kernel, retiring `filter_dead_from_kin`.

### New driver modules

- `cascade_resident.rs` (~100–150 lines) — `run_cascade_resident`: bind inputs, encode MAX_CASCADE_ITERATIONS indirect dispatches, return. No Rust-side iteration loop.
- `snapshot.rs` — double-buffered staging, `GpuSnapshot`, `SnapshotError`. Uses `gpu_util::readback::readback_typed_async`.

### Shared utilities (new small files)

Factoring to avoid duplication between sync and resident drivers:

- `gpu_util/readback.rs` — `copy_buffer_to_buffer → map_async → poll → get_mapped_range → unmap` pattern, currently duplicated across ~8 sites. Factor into `readback_typed::<T>(device, queue, src_buf, byte_len) -> Vec<T>` + `readback_typed_async::<T>(...)` variant that returns a handle the caller polls later.
- `gpu_util/indirect.rs` — indirect dispatch args buffer layout, helper to write `(wg_x, 1, 1)` as a GPU store, `dispatch_indirect` call wrapper.
- `gpu_util/bind.rs` — deferred; add only if kernel modules grow too wide.

### `lib.rs` additions

`GpuBackend` grows:

- Persistent `agents_buf` handle (currently re-allocated per kernel call; batch path needs one instance bound across all kernels).
- Cumulative `event_ring_last_read: u64` + `chronicle_ring_last_read: u64` watermarks.
- Two `GpuStaging` instances (front/back) from `snapshot.rs`.

New methods: `step_batch(n)`, `snapshot()`. `SimBackend::step()` impl unchanged.

## Error handling

**Indirect args corruption.** If the physics kernel writes garbage workgroup counts, GPU could dispatch runaway workgroups. Mitigation: clamp workgroup count to `ceil(agent_cap / PHYSICS_WORKGROUP_SIZE)` inside the kernel before writing indirect args. Defence-in-depth: validate args on readback at snapshot time and log a warning.

**GPU ring overflow.** The main event ring fills up mid-batch (no per-tick drain). Mitigation: kernel writes an `overflowed` flag (same bit as today). `snapshot()` reads it and returns `Err(SnapshotError::RingOverflow { tick, events_dropped })`. Caller decides whether to drop events or shrink `step_batch(n)`.

**Cascade non-convergence.** Indirect loop runs MAX_CASCADE_ITERATIONS every tick (no early break on CPU). Correctness unchanged; perf cost is no-op dispatches (microseconds). A warning surfaces on snapshot if the final iter still had non-zero workgroup count.

**Staging map failure.** `map_async` callback returns `Err`. Surface as `snapshot() -> Result<GpuSnapshot, SnapshotError>`. First call (no previous staging) returns `Ok(GpuSnapshot::empty())`, explicitly documented.

**No CPU fallback on the batch path.** Unlike the sync path (which falls back to CPU cascade on GPU failure), the batch path returns an error. Caller re-issues via sync `step()` if graceful degradation is wanted.

**Telemetry.** `PhaseTimings` extended with `batch_submit_us` and `batch_poll_us` so `--perf-sweep --batch-ticks N` shows whether time is in submit-record or GPU execution.

## Testing

### Existing tests untouched

`parity_with_cpu`, `perf_n100`, `chronicle_drain_perf`, all scenario tests continue running against the sync path. Byte-exact regression coverage stays where it is.

### New tests

- **`async_smoke.rs`** — `step_batch(100)` at N=2048, one `snapshot()`, assert:
  - `snapshot.tick == 100`
  - `snapshot.agents.len() == N`
  - `alive_count` within ±25% of the sync path (same tolerance `perf_n100` uses).
  - `events_since_last.len() > 0`
  - `chronicle_since_last.len() > 0`
- **`snapshot_double_buffer.rs`** — three back-to-back `snapshot()` calls. First returns empty, second returns ticks `[0..k1)`, third returns `[k1..k2)`. No events dropped, no duplicates.
- **`indirect_cascade_converges.rs`** — unit test of `cascade_resident`: single tick with a fixture guaranteed to converge in 2 iterations. Assert `last_cascade_iterations() == 2` read via snapshot.
- **`gpu_prefix_scan.rs`** — unit test for the new scan kernel against randomised u32 arrays; verify exclusive-scan matches CPU reference.

### No parity test between sync and batch

Explicitly a non-goal. Same-seed state divergence is expected and tests must not assert otherwise.

### Perf regression gate

- `chronicle --perf-sweep --batch-ticks 100` added as a new optional flag. CI runs it at N=2048 (fast); local runs at N=100k.
- Target: batch mean µs/tick < 0.8× sync mean µs/tick at N ≥ 512. Looser at smaller N where per-tick overhead already dominates.
- Phase E does not merge if the gate fails — that is the whole point of the refactor.

## Phase decomposition

Each phase lands independently because it adds new code (new method or new module) rather than editing existing sync-path code:

- **Phase A — GPU spatial-hash scan + direct binding.**
  - New GPU prefix-scan kernel in `spatial_gpu.rs`.
  - New `GpuSpatialHash::rebuild_and_query_resident` method with direct `kin_buf` / `nearest_buf` exposure and alive-filter in the query kernel.
  - Existing `rebuild_and_query` + `filter_dead_from_kin` remain for the sync path.
  - Unit test: `gpu_prefix_scan.rs`.
- **Phase B — Resident entry points on mask / scoring / apply / movement / physics.**
  - `*_resident` method added to each kernel. Shared `gpu_util/readback.rs` + `gpu_util/indirect.rs` land here.
  - Existing sync entry points untouched.
  - No new end-to-end test at this phase (covered by Phase C/D integration tests).
- **Phase C — `cascade_resident.rs`.**
  - New driver consuming resident kernel entry points from A + B.
  - MAX_CASCADE_ITERATIONS indirect dispatches into one encoder.
  - Unit test: `indirect_cascade_converges.rs`.
- **Phase D — `step_batch(n)` + `snapshot.rs` + `GpuSnapshot`.**
  - `lib.rs` additions: `step_batch`, `snapshot`, persistent `agents_buf`, ring watermarks, two `GpuStaging` instances.
  - New module `snapshot.rs`.
  - Integration tests: `async_smoke.rs`, `snapshot_double_buffer.rs`.
  - Landable only after A + B + C.
- **Phase E — perf validation + docs.**
  - `--batch-ticks N` flag on `chronicle --perf-sweep`.
  - Confirm target (batch < 0.8× sync at N ≥ 512).
  - Update plan doc + technical overview.

## Open questions

- **RNG seed advancement across ticks** — currently the tick-counter increment is a tiny dispatch; the RNG state advance needs to also be GPU-side. Simplest path: treat `state.rng_state` as a field of the `PhysicsCfg` buffer and have the tick-increment kernel also update it via a deterministic hash step. To be resolved during Phase D implementation.
- **Chronicle ring capacity for N ticks between snapshots** — if callers snapshot every frame but the sim runs many ticks per frame, the chronicle ring must hold all chronicle entries across that window. Current default capacity is 1 M records (`event_ring.rs:99`); at 100k agents and ~20 chronicle/tick per agent that's tight. May need a config knob or a dedicated larger default for batch mode.
