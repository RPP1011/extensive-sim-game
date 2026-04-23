# GPU Sim State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `tick`, `world_seed`, and sim-wide world-scalar fields to a shared GPU-resident `SimCfg` storage buffer; refactor `GpuBackend` into `sync` / `resident` / `snapshot` sub-structs; add `@cpu_only` DSL annotation.

**Architecture:** Two phases — Phase 1 refactors `GpuBackend` (behaviour-neutral; parity tests stay green). Phase 2 introduces `SimCfg` as the source of truth for tick + world scalars, wires the atomic tick increment into the existing seed-indirect kernel, migrates kernel reads, removes the CPU `state.tick.wrapping_add(1)` from `step_batch`, and adds `@cpu_only` parser/IR/validator/emitter plumbing.

**Tech Stack:** Rust, wgpu 0.20+, WGSL, `bytemuck` for Pod structs, existing `engine_gpu` + `dsl_compiler` crates.

**Spec reference:** `docs/superpowers/specs/2026-04-22-gpu-sim-state-design.md`

---

## File structure

### Phase 1 — new files

- `crates/engine_gpu/src/backend/mod.rs` — `GpuBackend` composite type, delegating methods.
- `crates/engine_gpu/src/backend/sync_ctx.rs` — `SyncPathContext` sub-struct.
- `crates/engine_gpu/src/backend/resident_ctx.rs` — `ResidentPathContext` sub-struct.
- `crates/engine_gpu/src/backend/snapshot_ctx.rs` — `SnapshotContext` sub-struct.

### Phase 2 — new files

- `crates/engine_gpu/src/sim_cfg.rs` — `SimCfg` Pod struct + buffer allocation + initial upload.
- `crates/engine_gpu/tests/sim_cfg_layout.rs` — struct-layout regression test.
- `crates/engine_gpu/tests/tick_advance_is_gpu_resident.rs` — integration test.
- `crates/engine_gpu/tests/cpu_only_annotation_skips_wgsl_emit.rs` — DSL-level test.

### Existing files modified (Phase 1)

- `crates/engine_gpu/src/lib.rs` — `GpuBackend` struct replaced by the `backend` module's composite; public API unchanged.

### Existing files modified (Phase 2)

- `crates/engine_gpu/src/cascade_resident.rs` — `SeedIndirectKernel` binds `SimCfg`; adds one WGSL line for `atomicAdd(&sim_cfg.tick, 1u)`.
- `crates/engine_gpu/src/lib.rs` — `step_batch` stops advancing `state.tick` on CPU; `snapshot()` reads `SimCfg.tick` for `GpuSnapshot.tick`.
- `crates/engine_gpu/src/mask.rs`, `scoring.rs`, `apply_actions.rs`, `movement.rs`, `physics.rs`, `spatial_gpu.rs` — each kernel's per-call cfg uniform shrinks; reads of world-scalars (engagement_range, attack_damage, attack_range, move_speed, move_speed_mult, kin_radius, cascade_max_iterations) migrate to `SimCfg`.
- `crates/dsl_compiler/src/parser.rs` — parse `@cpu_only` annotation.
- `crates/dsl_compiler/src/ir.rs` — carry `@cpu_only` flag on rule IR nodes.
- `crates/dsl_compiler/src/emit_physics_wgsl.rs` — skip WGSL emit for `@cpu_only` rules; skip dispatch-table entry.
- `crates/dsl_compiler/src/emit_physics.rs` — still emit CPU handler for `@cpu_only` rules.
- `crates/dsl_compiler/src/` — update GPU-emittable validator to accept primitives (strings, unbounded allocs) inside `@cpu_only` rule bodies.

**Untouched (intentional):** all sync-path tests, `SimBackend::step()` public API, the sync cascade driver in `crates/engine/src/cascade.rs`.

---

# Phase 1 — GpuBackend sub-struct factoring

**Goal**: behaviour-neutral refactor. Existing parity tests must stay green. This phase unblocks Phase 2 (which grows the struct further) by organising the 24 flat fields into three purposeful sub-structs.

## Task 1.1: Create backend module skeleton

**Files:**
- Create: `crates/engine_gpu/src/backend/mod.rs`
- Create: `crates/engine_gpu/src/backend/sync_ctx.rs`
- Create: `crates/engine_gpu/src/backend/resident_ctx.rs`
- Create: `crates/engine_gpu/src/backend/snapshot_ctx.rs`
- Modify: `crates/engine_gpu/src/lib.rs` (add `mod backend;`)

- [ ] **Step 1: Create `backend/sync_ctx.rs` with empty skeleton**

```rust
//! Sync-path state on the GPU backend — kernels and buffers used
//! exclusively by `SimBackend::step()`.
//!
//! Fields move here from the flat `GpuBackend` struct in commit
//! that follows this module's creation. This file is intentionally
//! empty at skeleton time so the `mod backend;` wiring lands first.

#![cfg(feature = "gpu")]

pub struct SyncPathContext {
    // Fields land in Task 1.2.
}

impl SyncPathContext {
    pub fn new() -> Self {
        Self {}
    }
}
```

- [ ] **Step 2: Create `backend/resident_ctx.rs` with empty skeleton**

```rust
//! Resident-path (batch) state — buffers and kernels used by
//! `step_batch()` for GPU-resident execution.

#![cfg(feature = "gpu")]

pub struct ResidentPathContext {
    // Fields land in Task 1.3.
}

impl ResidentPathContext {
    pub fn new() -> Self {
        Self {}
    }
}
```

- [ ] **Step 3: Create `backend/snapshot_ctx.rs` with empty skeleton**

```rust
//! Snapshot staging — double-buffered read-back state for `snapshot()`.

#![cfg(feature = "gpu")]

pub struct SnapshotContext {
    // Fields land in Task 1.4.
}

impl SnapshotContext {
    pub fn new() -> Self {
        Self {}
    }
}
```

- [ ] **Step 4: Create `backend/mod.rs`**

```rust
//! GpuBackend composite structure. The public `GpuBackend` type
//! delegates to `sync`, `resident`, and `snapshot` sub-contexts —
//! see `docs/superpowers/specs/2026-04-22-gpu-sim-state-design.md`
//! for rationale.

#![cfg(feature = "gpu")]

pub mod resident_ctx;
pub mod snapshot_ctx;
pub mod sync_ctx;

pub use resident_ctx::ResidentPathContext;
pub use snapshot_ctx::SnapshotContext;
pub use sync_ctx::SyncPathContext;
```

- [ ] **Step 5: Wire the module into `lib.rs`**

In `crates/engine_gpu/src/lib.rs`, near the other `mod` declarations at top of file:

```rust
#[cfg(feature = "gpu")]
pub mod backend;
```

- [ ] **Step 6: Verify the crate builds**

Run: `cargo build --features gpu -p engine_gpu`
Expected: clean build, no warnings (unused structs are fine for now because they're `pub`).

- [ ] **Step 7: Commit**

```bash
git add crates/engine_gpu/src/backend/ crates/engine_gpu/src/lib.rs
git commit -m "refactor(engine_gpu): scaffold backend/ sub-struct module"
```

---

## Task 1.2: Move sync-path fields into `SyncPathContext`

**Files:**
- Modify: `crates/engine_gpu/src/backend/sync_ctx.rs`
- Modify: `crates/engine_gpu/src/lib.rs`

**Context**: the current flat `GpuBackend` struct at `lib.rs:151` has fields related to sync-path kernels. Read the existing struct definition to identify which fields belong to the sync path:
- `mask_kernel`
- `scoring_kernel`
- `view_storage`
- `cascade_ctx` (also used by resident — leave in sync since sync owns it)
- `last_mask_bitmaps`, `last_scoring_outputs`, `last_cascade_iterations`, `last_cascade_error`, `last_phase_us`
- `skip_scoring_sidecar`
- `backend_label`

Move these into `SyncPathContext`. Update accessor methods (e.g. `last_mask_bitmaps()`) to delegate.

- [ ] **Step 1: Read current `GpuBackend` struct definition**

Run: `grep -n "pub struct GpuBackend" crates/engine_gpu/src/lib.rs` — find the `#[cfg(feature = "gpu")]` version (around line 151). Read 40 lines to see all fields.

- [ ] **Step 2: Define `SyncPathContext` fields**

Replace the empty body in `backend/sync_ctx.rs`:

```rust
use crate::cascade::CascadeCtx;
use crate::mask::MaskKernel;
use crate::scoring::{ScoringKernel, ScoreOutput};
use crate::view_storage::ViewStorage;
use crate::PhaseTimings;

pub struct SyncPathContext {
    pub mask_kernel:                MaskKernel,
    pub scoring_kernel:             ScoringKernel,
    pub view_storage:               ViewStorage,
    pub cascade_ctx:                Option<CascadeCtx>,
    pub last_mask_bitmaps:          Vec<Vec<u32>>,
    pub last_scoring_outputs:       Vec<ScoreOutput>,
    pub last_cascade_iterations:    Option<u32>,
    pub last_cascade_error:         Option<String>,
    pub last_phase_us:              PhaseTimings,
    pub skip_scoring_sidecar:       bool,
    pub backend_label:              String,
}

impl SyncPathContext {
    pub fn new(
        mask_kernel: MaskKernel,
        scoring_kernel: ScoringKernel,
        view_storage: ViewStorage,
        backend_label: String,
    ) -> Self {
        Self {
            mask_kernel,
            scoring_kernel,
            view_storage,
            cascade_ctx: None,
            last_mask_bitmaps: Vec::new(),
            last_scoring_outputs: Vec::new(),
            last_cascade_iterations: None,
            last_cascade_error: None,
            last_phase_us: PhaseTimings::default(),
            skip_scoring_sidecar: true,
            backend_label,
        }
    }
}
```

- [ ] **Step 3: Remove those fields from `GpuBackend`**

In `crates/engine_gpu/src/lib.rs`, the `#[cfg(feature = "gpu")]` `pub struct GpuBackend` gets the removed fields replaced with:

```rust
pub sync: crate::backend::SyncPathContext,
```

Leave the other fields (`device`, `queue`, resident-path fields, snapshot fields) in place for now — Tasks 1.3 and 1.4 move them.

- [ ] **Step 4: Update `GpuBackend::new` to construct `SyncPathContext`**

Find the existing `GpuBackend::new` / `new_async` at `lib.rs:488` and replace the field-by-field construction of sync fields with:

```rust
let sync = crate::backend::SyncPathContext::new(
    mask_kernel,
    scoring_kernel,
    view_storage,
    backend_label,
);
```

Then use `sync` in the `Ok(Self { ... })` block.

- [ ] **Step 5: Update callers**

Every method on `impl GpuBackend` that previously read e.g. `self.mask_kernel` now reads `self.sync.mask_kernel`. Likely ~20-30 callsites. Grep for each moved field name and rewrite. Keep the public method signatures unchanged — delegation is internal.

Example:
```rust
// Before:
pub fn last_mask_bitmaps(&self) -> &[Vec<u32>] { &self.last_mask_bitmaps }
// After:
pub fn last_mask_bitmaps(&self) -> &[Vec<u32>] { &self.sync.last_mask_bitmaps }
```

- [ ] **Step 6: Build + run the full engine_gpu test suite**

Run: `cargo test --features gpu -p engine_gpu`
Expected: all tests pass (this is a behaviour-neutral refactor).

If a test fails, the refactor broke something. Revert the failed callsite and retry more carefully.

- [ ] **Step 7: Commit**

```bash
git add crates/engine_gpu/src/backend/sync_ctx.rs crates/engine_gpu/src/lib.rs
git commit -m "refactor(engine_gpu): move sync-path fields into SyncPathContext"
```

---

## Task 1.3: Move resident-path fields into `ResidentPathContext`

**Files:**
- Modify: `crates/engine_gpu/src/backend/resident_ctx.rs`
- Modify: `crates/engine_gpu/src/lib.rs`

Resident-path fields on current `GpuBackend`:
- `resident_agents_buf`, `resident_agents_cap`
- `resident_indirect_args`
- `resident_cascade_ctx`
- `mask_unpack_kernel` (if present after the 9b730988 fix)
- `scoring_unpack_kernel` (if present)
- Any fields added by the later C1+C2 fix agent's work (`batch_events_ring` container, append kernel, etc.)

Grep the `#[cfg(feature = "gpu")]` `pub struct GpuBackend` block in `lib.rs` to enumerate all resident-prefixed and batch-related fields.

- [ ] **Step 1: Enumerate resident fields**

Run:
```
grep -nE "resident_|mask_unpack|scoring_unpack|append_events|batch_events" crates/engine_gpu/src/lib.rs | head -20
```

- [ ] **Step 2: Add fields to `ResidentPathContext`**

```rust
use crate::cascade_resident::CascadeResidentCtx;
use crate::gpu_util::indirect::IndirectArgsBuffer;
use crate::mask::MaskUnpackKernel;
use crate::scoring::ScoringUnpackKernel;

pub struct ResidentPathContext {
    pub resident_agents_buf:        Option<wgpu::Buffer>,
    pub resident_agents_cap:        u32,
    pub resident_indirect_args:     Option<IndirectArgsBuffer>,
    pub resident_cascade_ctx:       Option<CascadeResidentCtx>,
    pub mask_unpack_kernel:         Option<MaskUnpackKernel>,
    pub scoring_unpack_kernel:      Option<ScoringUnpackKernel>,
    // (add any additional resident fields discovered in Step 1)
}

impl ResidentPathContext {
    pub fn new() -> Self {
        Self {
            resident_agents_buf:     None,
            resident_agents_cap:     0,
            resident_indirect_args:  None,
            resident_cascade_ctx:    None,
            mask_unpack_kernel:      None,
            scoring_unpack_kernel:   None,
        }
    }
}

impl Default for ResidentPathContext {
    fn default() -> Self { Self::new() }
}
```

- [ ] **Step 3: Remove those fields from `GpuBackend`**

Replace them with:

```rust
pub resident: crate::backend::ResidentPathContext,
```

- [ ] **Step 4: Update `GpuBackend::new`**

Add `resident: ResidentPathContext::new(),` to the `Ok(Self { ... })` block.

- [ ] **Step 5: Update callers**

Rewrite all `self.resident_*` and `self.mask_unpack_kernel` reads through `self.resident.*`. Grep for each field name; expect ~15-30 callsites mainly in `step_batch`, `snapshot`, `ensure_resident_init`.

- [ ] **Step 6: Build + run test suite**

Run: `cargo test --features gpu -p engine_gpu`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/engine_gpu/src/backend/resident_ctx.rs crates/engine_gpu/src/lib.rs
git commit -m "refactor(engine_gpu): move resident-path fields into ResidentPathContext"
```

---

## Task 1.4: Move snapshot fields into `SnapshotContext`

**Files:**
- Modify: `crates/engine_gpu/src/backend/snapshot_ctx.rs`
- Modify: `crates/engine_gpu/src/lib.rs`

Snapshot fields on current `GpuBackend`:
- `snapshot_front: Option<GpuStaging>`
- `snapshot_back: Option<GpuStaging>`
- `snapshot_event_ring_read: u64`
- `snapshot_chronicle_ring_read: u64`
- `latest_recorded_tick: u32`

- [ ] **Step 1: Add fields to `SnapshotContext`**

```rust
use crate::snapshot::GpuStaging;

pub struct SnapshotContext {
    pub snapshot_front:                Option<GpuStaging>,
    pub snapshot_back:                 Option<GpuStaging>,
    pub snapshot_event_ring_read:      u64,
    pub snapshot_chronicle_ring_read:  u64,
    pub latest_recorded_tick:          u32,
}

impl SnapshotContext {
    pub fn new() -> Self {
        Self {
            snapshot_front:               None,
            snapshot_back:                None,
            snapshot_event_ring_read:     0,
            snapshot_chronicle_ring_read: 0,
            latest_recorded_tick:         0,
        }
    }
}

impl Default for SnapshotContext {
    fn default() -> Self { Self::new() }
}
```

- [ ] **Step 2: Remove fields from `GpuBackend`, add `pub snapshot: SnapshotContext`**

Same pattern as Tasks 1.2 / 1.3.

- [ ] **Step 3: Update `GpuBackend::new`**

```rust
snapshot: SnapshotContext::new(),
```

- [ ] **Step 4: Update callers**

Callsites mainly in `GpuBackend::snapshot()` and `step_batch`. Rewrite to `self.snapshot.snapshot_front`, etc.

- [ ] **Step 5: Build + run test suite**

Run: `cargo test --features gpu -p engine_gpu`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/engine_gpu/src/backend/snapshot_ctx.rs crates/engine_gpu/src/lib.rs
git commit -m "refactor(engine_gpu): move snapshot fields into SnapshotContext"
```

---

## Task 1.5: Phase 1 final regression sweep

- [ ] **Step 1: Run the full test suite in release mode**

Run: `cargo test --release --features gpu -p engine_gpu`
Expected: all tests pass. Pre-existing failures `scoring::tests::scoring_binding_count_matches_emitter` and `view_parity::pair_map_decay_kin_fear_parity` were resolved earlier in this project's history; they should stay passing.

- [ ] **Step 2: Run the perf sweep, confirm no regression**

Run: `cargo run --release --features gpu --bin xtask -- chronicle --perf-sweep --use-batch --batch-ticks 100 2>&1 | tail -12`
Expected: N=2048 batch µs/tick within ±5% of the baseline established by commits `ce72f9c0` / `69fdbfbd` (~5600-5900 µs/tick on llvmpipe). If the refactor regressed perf, there's an unintended `Clone` or bounce somewhere.

- [ ] **Step 3: Verify no new `unsafe` blocks**

Run: `git diff HEAD~4 -- crates/engine_gpu/src | grep -n unsafe`
Expected: no new `unsafe` appearances; the refactor is a pure structural change.

- [ ] **Step 4: Commit a tag or annotation if everything green**

No commit needed — Phase 1 ends at the head of Task 1.4.

---

# Phase 2 — SimCfg + `@cpu_only` + atomic tick advance

**Goal**: GPU-resident tick + world seed + world-scalar fields; compiler-level `@cpu_only` escape hatch; existing seed-indirect kernel gains atomic increment. `step_batch` stops advancing `state.tick` on CPU.

## Task 2.1: Create the `SimCfg` module

**Files:**
- Create: `crates/engine_gpu/src/sim_cfg.rs`
- Modify: `crates/engine_gpu/src/lib.rs` (add `pub mod sim_cfg;`)
- Test: `crates/engine_gpu/tests/sim_cfg_layout.rs`

- [ ] **Step 1: Write the failing test**

```rust
//! Regression fence on SimCfg's struct layout. Any drift between the
//! Rust Pod layout and the WGSL struct layout silently corrupts
//! GPU-side reads; this test asserts field offsets match.

#![cfg(feature = "gpu")]

use engine_gpu::sim_cfg::SimCfg;
use bytemuck::offset_of;

#[test]
fn sim_cfg_field_offsets_are_stable() {
    assert_eq!(std::mem::size_of::<SimCfg>(), 64);
    assert_eq!(offset_of!(zeroed SimCfg, tick), 0);
    assert_eq!(offset_of!(zeroed SimCfg, world_seed_lo), 4);
    assert_eq!(offset_of!(zeroed SimCfg, world_seed_hi), 8);
    assert_eq!(offset_of!(zeroed SimCfg, _pad0), 12);
    assert_eq!(offset_of!(zeroed SimCfg, engagement_range), 16);
    assert_eq!(offset_of!(zeroed SimCfg, attack_damage), 20);
    assert_eq!(offset_of!(zeroed SimCfg, attack_range), 24);
    assert_eq!(offset_of!(zeroed SimCfg, move_speed), 28);
    assert_eq!(offset_of!(zeroed SimCfg, move_speed_mult), 32);
    assert_eq!(offset_of!(zeroed SimCfg, kin_radius), 36);
    assert_eq!(offset_of!(zeroed SimCfg, cascade_max_iterations), 40);
    assert_eq!(offset_of!(zeroed SimCfg, rules_registry_generation), 44);
    assert_eq!(offset_of!(zeroed SimCfg, abilities_registry_generation), 48);
    // _reserved[0..4] at 52, 56, 60 — size_of ends at 64.
}

#[test]
fn sim_cfg_is_pod_zeroable() {
    let _: SimCfg = bytemuck::Zeroable::zeroed();
}
```

Note: if `bytemuck::offset_of!` isn't available in the version in use, fall back to `memoffset::offset_of!` (add `memoffset` to dev-dependencies) or hand-compute with `unsafe { (&(*p).field as *const _ as usize) - (p as usize) }`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --features gpu -p engine_gpu --test sim_cfg_layout`
Expected: FAIL — `sim_cfg` module doesn't exist.

- [ ] **Step 3: Create `crates/engine_gpu/src/sim_cfg.rs`**

```rust
//! Shared GPU-resident sim state — tick, world seed, world-scalar
//! fields, and cache-invalidation generation counters. Bound as a
//! `storage` buffer (not uniform) because the tick field is atomically
//! incremented by the seed-indirect kernel.
//!
//! WGSL-side struct layout must match this Rust definition byte-for-
//! byte. The `sim_cfg_layout` regression test fences field offsets.

#![cfg(feature = "gpu")]

use bytemuck::{Pod, Zeroable};
use engine::state::SimState;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct SimCfg {
    pub tick:                          u32,
    pub world_seed_lo:                 u32,
    pub world_seed_hi:                 u32,
    pub _pad0:                         u32,
    pub engagement_range:              f32,
    pub attack_damage:                 f32,
    pub attack_range:                  f32,
    pub move_speed:                    f32,
    pub move_speed_mult:               f32,
    pub kin_radius:                    f32,
    pub cascade_max_iterations:        u32,
    pub rules_registry_generation:     u32,
    pub abilities_registry_generation: u32,
    pub _reserved:                     [u32; 3],
}

impl SimCfg {
    /// Populate from a `SimState` at batch-entry time. Reads world
    /// seed + config scalars + current tick.
    pub fn from_state(state: &SimState) -> Self {
        Self {
            tick:                          state.tick,
            world_seed_lo:                 (state.rng_state & 0xFFFF_FFFF) as u32,
            world_seed_hi:                 (state.rng_state >> 32) as u32,
            _pad0:                         0,
            engagement_range:              state.config.combat.engagement_range,
            attack_damage:                 state.config.combat.attack_damage,
            attack_range:                  state.config.combat.attack_range,
            move_speed:                    state.config.movement.move_speed,
            move_speed_mult:               state.config.movement.move_speed_mult,
            kin_radius:                    state.config.combat.kin_radius,
            cascade_max_iterations:        state.config.cascade.max_iterations,
            rules_registry_generation:     0, // bumped by future subsystems
            abilities_registry_generation: 0,
            _reserved:                     [0; 3],
        }
    }
}

/// Allocate the GPU-side SimCfg buffer, sized for one instance. Usage
/// flags include STORAGE (kernel reads/atomic writes), COPY_SRC
/// (snapshot readback), and COPY_DST (host upload of initial values).
pub fn create_sim_cfg_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("engine_gpu::sim_cfg"),
        size:               std::mem::size_of::<SimCfg>() as u64,
        usage:              wgpu::BufferUsages::STORAGE
                         | wgpu::BufferUsages::COPY_SRC
                         | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Upload a fresh `SimCfg` snapshot to the device buffer.
pub fn upload_sim_cfg(queue: &wgpu::Queue, buf: &wgpu::Buffer, cfg: &SimCfg) {
    queue.write_buffer(buf, 0, bytemuck::bytes_of(cfg));
}
```

Note: field sizes sum to 60 bytes; `_reserved: [u32; 3]` adds 12 bytes — total 64. Adjust `_reserved` length if the offset math works out differently; the test locks it.

Update test field offset expectations to match the final layout.

- [ ] **Step 4: Wire into lib.rs**

```rust
#[cfg(feature = "gpu")]
pub mod sim_cfg;
```

- [ ] **Step 5: Run the test; confirm it passes**

Run: `cargo test --features gpu -p engine_gpu --test sim_cfg_layout`
Expected: 2/2 pass.

- [ ] **Step 6: Commit**

```bash
git add crates/engine_gpu/src/sim_cfg.rs crates/engine_gpu/src/lib.rs crates/engine_gpu/tests/sim_cfg_layout.rs
git commit -m "feat(engine_gpu): SimCfg Pod struct + regression-fenced layout"
```

---

## Task 2.2: Add `sim_cfg_buf` to `ResidentPathContext`; upload initial values

**Files:**
- Modify: `crates/engine_gpu/src/backend/resident_ctx.rs`
- Modify: `crates/engine_gpu/src/lib.rs` (inside `ensure_resident_init`)

- [ ] **Step 1: Add field to `ResidentPathContext`**

Add after `resident_cascade_ctx`:

```rust
pub sim_cfg_buf: Option<wgpu::Buffer>,
```

Update `new()` to initialise `sim_cfg_buf: None`.

- [ ] **Step 2: In `ensure_resident_init`, allocate and populate `sim_cfg_buf`**

Find `ensure_resident_init` (added by the D2/D4 work, now inside `ResidentPathContext` or `GpuBackend`). Before `resident_cascade_ctx` init, add:

```rust
if self.resident.sim_cfg_buf.is_none() {
    let buf = crate::sim_cfg::create_sim_cfg_buffer(&self.device);
    let cfg = crate::sim_cfg::SimCfg::from_state(state);
    crate::sim_cfg::upload_sim_cfg(&self.queue, &buf, &cfg);
    self.resident.sim_cfg_buf = Some(buf);
}
```

- [ ] **Step 3: Build + run tests**

Run: `cargo test --features gpu -p engine_gpu`
Expected: all tests pass. The new buffer is allocated but not yet read by any kernel — behaviour unchanged.

- [ ] **Step 4: Commit**

```bash
git add crates/engine_gpu/src/backend/resident_ctx.rs crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): allocate + upload SimCfg in ensure_resident_init"
```

---

## Task 2.3: Bind `SimCfg` into the seed-indirect kernel; add atomic tick increment

**Files:**
- Modify: `crates/engine_gpu/src/cascade_resident.rs`

**Context**: the seed-indirect kernel runs once per tick to seed iter 0 of the cascade. Currently it reads `apply_event_ring.tail` and writes `indirect_args[0]`. We extend it to also atomically increment `sim_cfg.tick`.

- [ ] **Step 1: Extend the WGSL source**

Find the WGSL source for `SeedIndirectKernel` in `cascade_resident.rs`. Add a new binding:

```wgsl
@group(0) @binding(N) var<storage, read_write> sim_cfg: SimCfg;

struct SimCfg {
    tick:                          atomic<u32>,
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
    _reserved0:                    u32,
    _reserved1:                    u32,
    _reserved2:                    u32,
}
```

N is the next free binding index in the seed-indirect BGL.

At the end of the `@compute` entry point, add:

```wgsl
atomicAdd(&sim_cfg.tick, 1u);
```

- [ ] **Step 2: Extend the BGL and bind group**

In the Rust builder for `SeedIndirectKernel`, add a `BindGroupLayoutEntry` for `sim_cfg` with `ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, ... }`.

Add a `BindGroupEntry` referencing `self.resident.sim_cfg_buf.as_ref().unwrap().as_entire_binding()`.

- [ ] **Step 3: Update the seed-indirect dispatch callsite**

In `run_cascade_resident` (or wherever seed-indirect is dispatched), pass the `sim_cfg_buf` reference when building the bind group.

- [ ] **Step 4: Build + run async_smoke + parity tests**

Run: `cargo test --features gpu -p engine_gpu --test async_smoke --test parity_with_cpu`
Expected: pass. The atomic increment doesn't affect any current observer (no code reads `sim_cfg.tick` yet).

- [ ] **Step 5: Commit**

```bash
git add crates/engine_gpu/src/cascade_resident.rs
git commit -m "feat(engine_gpu): seed-indirect kernel binds SimCfg + atomic tick++"
```

---

## Task 2.4–2.9: Migrate each kernel to read world-scalars from `SimCfg`

Six kernels to migrate, one task per: mask (2.4), scoring (2.5), apply_actions (2.6), movement (2.7), physics (2.8), spatial (2.9). Each follows the same pattern:

1. Identify which world-scalar fields the kernel's current cfg uniform carries (engagement_range, attack_damage, attack_range, move_speed, move_speed_mult, kin_radius, cascade_max_iterations).
2. Add a `sim_cfg` binding to the kernel's BGL.
3. Extend the kernel's WGSL to declare `SimCfg` struct + bind the buffer.
4. Replace kernel-internal reads of the migrated fields with `sim_cfg.<field>` reads.
5. Remove the migrated fields from the kernel's per-call cfg uniform struct (on both Rust and WGSL sides).
6. Update the kernel's `run_resident` method to not populate those fields in its cfg writes.
7. Run the full test suite.
8. Commit.

Because each task follows the same mechanical pattern, the example below is the full Task 2.4 template; Tasks 2.5–2.9 repeat substituting the kernel name.

### Task 2.4: Migrate mask kernel to `SimCfg`

**Files:**
- Modify: `crates/engine_gpu/src/mask.rs`

- [ ] **Step 1: Identify migrated fields**

Open `mask.rs`. Find the cfg struct used by `MaskKernel::run_resident` — likely `MaskCfg` or equivalent. Enumerate its fields. Of those, which are world-scalar (engagement_range, attack_damage, attack_range, move_speed, move_speed_mult, kin_radius)? Mask likely reads `engagement_range` and `attack_range`.

- [ ] **Step 2: Extend BGL**

Find the BGL for `MaskKernel`. Add a read-only storage buffer entry for `sim_cfg` at the next free binding.

- [ ] **Step 3: Extend WGSL**

At the top of the mask WGSL source (in `build_masks_wgsl_fused` or equivalent emitter), add the `SimCfg` struct declaration + binding:

```wgsl
@group(0) @binding(N) var<storage, read> sim_cfg: SimCfg;

struct SimCfg {
    tick:                          u32,  // atomic on writer side; read is fine
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
    _reserved0:                    u32,
    _reserved1:                    u32,
    _reserved2:                    u32,
}
```

Then in the kernel body, replace reads of the per-call cfg's migrated fields with `sim_cfg.engagement_range`, `sim_cfg.attack_range`, etc.

- [ ] **Step 4: Update Rust cfg struct**

Remove the migrated fields from `MaskCfg` (or equivalent). Re-emit. The struct shrinks to kernel-local fields only.

- [ ] **Step 5: Update `run_resident` bind group**

Pass `sim_cfg_buf` from `ResidentPathContext` into `MaskKernel::run_resident`:

```rust
pub fn run_resident(
    &mut self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    agents_buf: &wgpu::Buffer,
    sim_cfg_buf: &wgpu::Buffer,  // new
    agent_cap: u32,
) -> Result<(), KernelError>
```

Thread through the BGL bind group.

- [ ] **Step 6: Update callsite in `step_batch`**

In `step_batch`'s per-tick loop, pass `self.resident.sim_cfg_buf.as_ref().unwrap()` to the mask resident call.

- [ ] **Step 7: Run tests**

Run: `cargo test --release --features gpu -p engine_gpu --test async_smoke --test parity_with_cpu --test spatial_parity`
Expected: all pass. Mask reads the same values as before; sync path untouched.

- [ ] **Step 8: Commit**

```bash
git add crates/engine_gpu/src/mask.rs crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): MaskKernel reads world-scalars from SimCfg"
```

### Tasks 2.5 through 2.9

Apply the Task 2.4 template to each of:

- **Task 2.5**: `crates/engine_gpu/src/scoring.rs` — scoring kernel. Commit: `ScoringKernel reads world-scalars from SimCfg`.
- **Task 2.6**: `crates/engine_gpu/src/apply_actions.rs` — apply_actions kernel. Commit: `ApplyActionsKernel reads world-scalars from SimCfg`.
- **Task 2.7**: `crates/engine_gpu/src/movement.rs` — movement kernel. Commit: `MovementKernel reads world-scalars from SimCfg`.
- **Task 2.8**: `crates/engine_gpu/src/physics.rs` — physics kernel (big one; carefully thread `sim_cfg_buf` through the resident path). Commit: `PhysicsKernel reads world-scalars from SimCfg`.
- **Task 2.9**: `crates/engine_gpu/src/spatial_gpu.rs` — spatial hash kernels. Commit: `spatial_gpu reads world-scalars from SimCfg`.

Each task: adds the binding, updates the WGSL struct + reads, removes fields from the kernel's per-call cfg struct, runs the test suite, commits.

After Task 2.9, the per-kernel cfg structs carry only kernel-local fields (slot indices, workgroup sizes). If any kernel ends up with an empty cfg uniform, drop its binding entirely in a follow-on cleanup.

---

## Task 2.10: Remove CPU `state.tick.wrapping_add(1)` from `step_batch`

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/engine_gpu/tests/tick_advance_is_gpu_resident.rs`:

```rust
//! step_batch must not advance state.tick on CPU. GPU SimCfg.tick
//! advances each tick; snapshot() reads it back to populate
//! GpuSnapshot.tick.

#![cfg(feature = "gpu")]

use engine::backend::SimBackend;
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::GpuBackend;
use glam::Vec3;

#[test]
fn step_batch_does_not_advance_cpu_tick() {
    let mut gpu = GpuBackend::new().expect("gpu init");
    let mut state = SimState::new(16, 0xDEAD);
    for i in 0..4 {
        state.spawn_agent(AgentSpawn {
            creature_type: if i % 2 == 0 {
                CreatureType::Human
            } else {
                CreatureType::Wolf
            },
            pos: Vec3::new(i as f32 * 2.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        }).unwrap();
    }
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = CascadeRegistry::with_engine_builtins();

    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    let tick_after_sync_warmup = state.tick;

    gpu.step_batch(
        &mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 10,
    );

    // CPU state.tick must NOT have advanced during step_batch.
    assert_eq!(
        state.tick, tick_after_sync_warmup,
        "state.tick should stay stale during batch; moved from {} to {}",
        tick_after_sync_warmup, state.tick,
    );

    // Snapshot must report the advanced tick.
    let _empty = gpu.snapshot().expect("first snapshot is empty");
    let _kick = gpu.snapshot().expect("kick snapshot");
    let snap = gpu.snapshot().expect("third snapshot");
    assert_eq!(
        snap.tick,
        tick_after_sync_warmup + 10,
        "snapshot.tick should reflect 10 ticks of GPU-side advance",
    );
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --release --features gpu -p engine_gpu --test tick_advance_is_gpu_resident`
Expected: FAIL — `state.tick` did advance (current `step_batch` behaviour).

- [ ] **Step 3: Remove the CPU tick increment**

In `step_batch`'s per-tick loop in `crates/engine_gpu/src/lib.rs`, find:

```rust
state.tick = state.tick.wrapping_add(1);
```

Delete it. Also remove:

```rust
self.snapshot.latest_recorded_tick = state.tick;
```

(its role is taken over by `SimCfg.tick` reads in `snapshot()`).

- [ ] **Step 4: Run the test to verify it still fails for the snapshot assertion**

Run: `cargo test --release --features gpu -p engine_gpu --test tick_advance_is_gpu_resident`
Expected: FAIL — `state.tick` no longer advances (correct), but `snap.tick` still reads from `latest_recorded_tick` which is now 0. Task 2.11 fixes that.

- [ ] **Step 5: Commit (partial)**

```bash
git add crates/engine_gpu/src/lib.rs crates/engine_gpu/tests/tick_advance_is_gpu_resident.rs
git commit -m "feat(engine_gpu): step_batch no longer advances state.tick on CPU"
```

The test stays failing on the `snap.tick` assertion until Task 2.11.

---

## Task 2.11: `snapshot()` reads `SimCfg.tick` for `GpuSnapshot.tick`

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs`
- Modify: `crates/engine_gpu/src/snapshot.rs` (if needed)

- [ ] **Step 1: Add a `SimCfg` tick readback to `snapshot()`**

In `GpuBackend::snapshot()` in `lib.rs`, before computing the snapshot struct, read the current GPU tick:

```rust
let gpu_tick = {
    let buf = self.resident.sim_cfg_buf.as_ref().ok_or_else(|| {
        crate::snapshot::SnapshotError::Ring(
            "sim_cfg_buf not initialised; call step_batch first".into(),
        )
    })?;
    let bytes = crate::gpu_util::readback::readback_typed::<u32>(
        &self.device, &self.queue, buf, 4,
    ).map_err(|e| crate::snapshot::SnapshotError::Ring(e))?;
    bytes[0]
};
```

- [ ] **Step 2: Pass `gpu_tick` into the staging's `kick_copy`**

`GpuStaging::kick_copy` already takes a `tick: u32` parameter (per D1). Replace the existing `self.snapshot.latest_recorded_tick` argument with `gpu_tick`. The returned `GpuSnapshot.tick` flows through naturally.

- [ ] **Step 3: Run the tick-advance test**

Run: `cargo test --release --features gpu -p engine_gpu --test tick_advance_is_gpu_resident`
Expected: PASS — `state.tick` stays stale; `snap.tick == tick_after_sync_warmup + 10`.

- [ ] **Step 4: Run the async_smoke + full suite**

Run: `cargo test --release --features gpu -p engine_gpu`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): snapshot() reads SimCfg.tick for GpuSnapshot.tick"
```

---

## Task 2.12: DSL parser — parse `@cpu_only` annotation

**Files:**
- Modify: `crates/dsl_compiler/src/parser.rs`
- Modify: `crates/dsl_compiler/src/ast.rs`

- [ ] **Step 1: Write the failing test**

In `crates/dsl_compiler/src/parser.rs`, append to the `#[cfg(test)] mod tests`:

```rust
#[test]
fn cpu_only_annotation_parses_on_physics_rule() {
    let src = r#"
@cpu_only physics some_cpu_only_rule @phase(event) {
    on AgentDied { }
}
"#;
    let program = parse_program(src).expect("parse OK");
    let rule = program.physics_rules.iter().find(|r| r.name == "some_cpu_only_rule").expect("rule present");
    assert!(rule.cpu_only, "cpu_only flag should be set");
}
```

(Adjust `parse_program` / struct names to match actual parser API.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p dsl_compiler cpu_only_annotation_parses_on_physics_rule`
Expected: FAIL — `cpu_only` field doesn't exist on the rule AST.

- [ ] **Step 3: Add `cpu_only` flag to the physics-rule AST node**

In `crates/dsl_compiler/src/ast.rs`, find the physics rule struct (likely `PhysicsRuleAst` or `PhysicsDecl`). Add:

```rust
pub cpu_only: bool,
```

Default to `false` in every constructor.

- [ ] **Step 4: Extend the parser**

In `parser.rs`, find the annotation-prefix parser (the bit that handles `@phase`, `@replayable`, etc.). Add recognition of `@cpu_only`:

```rust
// Existing annotation-loop continues; add:
if peek_keyword(input, "@cpu_only") {
    consume(input, "@cpu_only");
    cpu_only = true;
    continue;
}
```

Thread `cpu_only` into the rule AST construction.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test -p dsl_compiler cpu_only_annotation_parses_on_physics_rule`
Expected: PASS.

- [ ] **Step 6: Verify the full test suite**

Run: `cargo test -p dsl_compiler`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add crates/dsl_compiler/src/ast.rs crates/dsl_compiler/src/parser.rs
git commit -m "feat(dsl_compiler): parse @cpu_only annotation on physics rules"
```

---

## Task 2.13: DSL IR — carry `cpu_only` through lowering

**Files:**
- Modify: `crates/dsl_compiler/src/ir.rs`
- Modify: `crates/dsl_compiler/src/lower.rs` (or equivalent)

- [ ] **Step 1: Add `cpu_only: bool` to the IR physics rule node**

In `ir.rs`, find the IR physics rule struct (`IrPhysicsRule` or similar). Add:

```rust
pub cpu_only: bool,
```

- [ ] **Step 2: Propagate from AST to IR**

In the lowering code (likely `lower.rs`), set `cpu_only: ast.cpu_only` when constructing the IR node.

- [ ] **Step 3: Verify lowering tests still pass**

Run: `cargo test -p dsl_compiler`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add crates/dsl_compiler/src/ir.rs crates/dsl_compiler/src/lower.rs
git commit -m "feat(dsl_compiler): thread @cpu_only through IR lowering"
```

---

## Task 2.14: GPU-emittable validator — accept non-GPU primitives inside `@cpu_only` bodies

**Files:**
- Modify: `crates/dsl_compiler/src/` (validator location; grep for "gpu_emittable" or "GpuEmittableValidator")

- [ ] **Step 1: Locate the validator**

Run: `grep -rn "gpu_emittable\|GpuEmittableValidator\|emit_validation" crates/dsl_compiler/src/ | head -5`

- [ ] **Step 2: Short-circuit validation for `@cpu_only` rules**

At the validator's entry for each physics rule body:

```rust
if rule.cpu_only {
    // Rule stays CPU-only — no WGSL emission, no validator constraints.
    return Ok(());
}
// Existing validation continues.
```

- [ ] **Step 3: Add a test fixture with a `@cpu_only` rule using a non-GPU primitive**

Create a minimal test inline in the validator's `tests` module that exercises a `@cpu_only` rule calling e.g. `format!(...)` or an unbounded `Vec::push` — primitives that would normally fail validation.

```rust
#[test]
fn cpu_only_rules_bypass_gpu_emittable_validation() {
    let src = r#"
@cpu_only physics narrative_formatter @phase(event) {
    on AgentDied {
        // imagine this contains String-building logic
    }
}
"#;
    let program = parse_program(src).unwrap();
    let result = validate_gpu_emittable(&program);
    assert!(result.is_ok(), "cpu_only rule should not trigger GPU validator");
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p dsl_compiler cpu_only_rules_bypass_gpu_emittable_validation`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src/
git commit -m "feat(dsl_compiler): @cpu_only rules bypass GPU-emittable validator"
```

---

## Task 2.15: DSL WGSL emitter — skip `@cpu_only` rules

**Files:**
- Modify: `crates/dsl_compiler/src/emit_physics_wgsl.rs`

- [ ] **Step 1: Find the per-rule emit loop**

Grep: `grep -n "for rule in" crates/dsl_compiler/src/emit_physics_wgsl.rs | head`

- [ ] **Step 2: Skip `@cpu_only` rules**

In the emit loop:

```rust
for rule in &program.physics_rules {
    if rule.cpu_only {
        continue; // CPU-only rules have no WGSL body
    }
    // Existing emit logic
    ...
}
```

Also: skip adding the rule to the event-kind dispatcher table that the physics kernel consumes.

- [ ] **Step 3: Write the end-to-end test**

Create `crates/engine_gpu/tests/cpu_only_annotation_skips_wgsl_emit.rs`:

```rust
//! End-to-end: an @cpu_only physics rule produces a CPU handler
//! but no entry in the GPU dispatch manifest.

use dsl_compiler::{compile_from_str, CompileOutput};

#[test]
fn cpu_only_rule_has_no_wgsl_emit() {
    let src = r#"
@cpu_only physics test_cpu_rule @phase(event) {
    on AgentDied { }
}

physics test_gpu_rule @phase(event) {
    on AgentDied { }
}
"#;
    let out: CompileOutput = compile_from_str(src).expect("compile OK");

    // GPU dispatcher manifest must contain test_gpu_rule but not test_cpu_rule.
    assert!(!out.gpu_dispatch_manifest.iter().any(|r| r.name == "test_cpu_rule"));
    assert!(out.gpu_dispatch_manifest.iter().any(|r| r.name == "test_gpu_rule"));

    // CPU handler registry must contain both.
    assert!(out.cpu_handlers.iter().any(|r| r.name == "test_cpu_rule"));
    assert!(out.cpu_handlers.iter().any(|r| r.name == "test_gpu_rule"));
}
```

(Adjust `compile_from_str` / struct names to actual API.)

- [ ] **Step 4: Run the test; confirm it passes**

Run: `cargo test -p dsl_compiler --test cpu_only_annotation_skips_wgsl_emit`
Expected: PASS.

- [ ] **Step 5: Run the full compiler test suite**

Run: `cargo test -p dsl_compiler`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add crates/dsl_compiler/src/emit_physics_wgsl.rs crates/engine_gpu/tests/cpu_only_annotation_skips_wgsl_emit.rs
git commit -m "feat(dsl_compiler): skip WGSL emit for @cpu_only rules"
```

---

## Task 2.16: CPU emitter — still emit CPU handler for `@cpu_only` rules

**Files:**
- Modify: `crates/dsl_compiler/src/emit_physics.rs`

- [ ] **Step 1: Verify the CPU emitter doesn't skip `@cpu_only`**

Open `emit_physics.rs`. Confirm no `if rule.cpu_only { continue; }` was added when threading the flag through IR. If the CPU emitter was inadvertently gated, remove the gate.

This is a no-op if nothing was accidentally gated — Task 2.13 set the field without touching emit paths.

- [ ] **Step 2: Run the tests**

Run: `cargo test -p dsl_compiler`
Expected: the `cpu_only_rule_has_no_wgsl_emit` test from Task 2.15 asserts CPU handlers DO exist for `@cpu_only` rules. If it passes, Task 2.16 is a verification no-op; commit is skipped.

- [ ] **Step 3: (conditional) Commit if a gate was removed**

```bash
git commit --allow-empty -m "test(dsl_compiler): confirm CPU emitter handles @cpu_only rules"
```

---

## Task 2.17: Full regression sweep + perf sweep

- [ ] **Step 1: Full test suite in release**

Run: `cargo test --release --features gpu -p engine_gpu && cargo test --release -p dsl_compiler`
Expected: all pass.

- [ ] **Step 2: Perf sweep to confirm no regression**

Run: `cargo run --release --features gpu --bin xtask -- chronicle --perf-sweep --use-batch --batch-ticks 100 2>&1 | tail -12`
Expected: N=2048 batch µs/tick within ±10% of baseline. The new SimCfg binding + atomic tick add a few microseconds per tick — acceptable.

- [ ] **Step 3: Schema-hash baseline bump (if necessary)**

The per-kernel cfg uniform schema changes when fields migrate out. If the `crates/engine/.schema_hash` baseline test fails, update the baseline:

Run: `cargo test -p engine schema_hash 2>&1 | tail -5`
If FAIL with a new hash, echo the new hash into `crates/engine/.schema_hash`, commit:
```bash
git add crates/engine/.schema_hash
git commit -m "chore(engine): bump schema_hash for SimCfg migration"
```

- [ ] **Step 4: Final commit or tag**

No commit needed — Phase 2 ends. Subsystems (2) / (3) can begin.

---

## Notes for the implementing engineer

- **Migration order matters**: SimCfg module (2.1) must land before any kernel migration (2.4+). The buffer (2.2) before the seed-indirect kernel binds it (2.3). Tick advance removal (2.10) before snapshot reads (2.11) — between those two commits, one test fails; that's OK as a two-commit transaction.
- **WGSL struct layout**: alignment matters. `vec3<u32>` in WGSL pads to 16 bytes in storage buffers; we intentionally avoid vec types in SimCfg for this reason. Each field is a scalar u32/f32.
- **Sync path isn't migrated**: `SimBackend::step()` continues to use per-kernel cfg uniforms with the old fields. That's fine — we're not consolidating the sync path; its per-tick CPU/GPU sync overhead dominates anyway.
- **Don't commit hand-generated files**: the DSL compiler auto-regenerates files under `crates/engine/src/generated/` and `crates/engine_generated/src/generated/`. After any DSL source or compiler change, run `cargo run --bin xtask -- compile-dsl` to regenerate; commit the regenerated files alongside the generator change.
- **Parallel agent activity**: during this implementation, other agents may be working on subsystems (2)/(3). They'll likely touch different files, but watch for merge conflicts in `lib.rs` (shared by many tasks).

## Open questions surfaced during planning

- **`MaskCfg` may have kernel-local fields beyond world scalars** (workgroup size, bitmap stride, etc.). After migration, if the local cfg uniform is ≤ 8 bytes, consider dropping it entirely — the kernel can read those constants inline via WGSL `const`. Defer until after the migration lands; measure, then decide.
- **Sync path migration**: out of scope for this plan. If sync's per-tick CPU `queue.write_buffer` ever becomes a bottleneck, open a follow-up subsystem to migrate it to SimCfg too.
