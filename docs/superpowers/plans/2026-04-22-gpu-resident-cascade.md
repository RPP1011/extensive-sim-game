# GPU-Resident Cascade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an additive batch API (`step_batch(n)` + `snapshot()`) to `engine_gpu::GpuBackend` that runs N ticks GPU-resident with zero per-tick CPU fences and exposes state via a double-buffered non-blocking snapshot. The existing `SimBackend::step()` sync path is preserved unchanged.

**Architecture:** Each kernel module grows a `*_resident` entry point alongside its existing sync path. A new `cascade_resident.rs` module encodes MAX_CASCADE_ITERATIONS indirect dispatches into one command buffer per tick. A new `snapshot.rs` module provides double-buffered staging with `map_async` on the front buffer and `copy_buffer_to_buffer` into the back buffer on each call.

**Tech Stack:** Rust, wgpu 0.20+, WGSL compute shaders, `bytemuck` for buffer layouts, existing engine_gpu crate.

**Spec reference:** `docs/superpowers/specs/2026-04-22-gpu-resident-cascade-design.md`

---

## File structure

**New files:**
- `crates/engine_gpu/src/gpu_util/mod.rs` — umbrella module for shared GPU helpers
- `crates/engine_gpu/src/gpu_util/readback.rs` — `readback_typed::<T>()` and `readback_typed_async::<T>()`
- `crates/engine_gpu/src/gpu_util/indirect.rs` — indirect dispatch args buffer + writer helper
- `crates/engine_gpu/src/cascade_resident.rs` — `run_cascade_resident(...)`
- `crates/engine_gpu/src/snapshot.rs` — `GpuStaging`, `GpuSnapshot`, `SnapshotError`
- `crates/engine_gpu/tests/gpu_prefix_scan.rs` — unit test for the scan kernel
- `crates/engine_gpu/tests/indirect_cascade_converges.rs` — unit test for resident cascade
- `crates/engine_gpu/tests/async_smoke.rs` — integration test for step_batch + snapshot
- `crates/engine_gpu/tests/snapshot_double_buffer.rs` — back-to-back snapshot test

**Existing files modified:**
- `crates/engine_gpu/src/lib.rs` — add `snapshot()`, rewrite `step_batch()`, add persistent `agents_buf` + staging fields
- `crates/engine_gpu/src/physics.rs` — add `run_batch_resident` method on `PhysicsKernel`
- `crates/engine_gpu/src/mask.rs` — add `run_resident` method on `MaskKernel`
- `crates/engine_gpu/src/scoring.rs` — add `run_resident` method on `ScoringKernel`
- `crates/engine_gpu/src/apply_actions.rs` — add `run_resident` method
- `crates/engine_gpu/src/movement.rs` — add `run_resident` method
- `crates/engine_gpu/src/spatial_gpu.rs` — add GPU prefix-scan kernel + `rebuild_and_query_resident`
- `src/bin/xtask/chronicle_cmd.rs` — add `--batch-ticks N` flag and batch-path timing

**Untouched (intentional):** all existing `*.rs` test files, `SimBackend::step()` implementation, `scenarios/`, `docs/plans/gpu_megakernel_plan.md`.

---

# Phase A — GPU spatial-hash scan + direct binding

**Goal**: Add a GPU prefix-scan kernel to `spatial_gpu.rs`, add a `rebuild_and_query_resident` method that exposes `kin_buf` + `nearest_buf` as bindable device handles, and move the alive-filter into the query kernel.

## Task A1: GPU prefix-scan kernel

**Files:**
- Modify: `crates/engine_gpu/src/spatial_gpu.rs` (add scan pipeline + WGSL + `run_scan` method on `GpuSpatialHash`)
- Test: `crates/engine_gpu/tests/gpu_prefix_scan.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/engine_gpu/tests/gpu_prefix_scan.rs`:

```rust
//! Unit test for the GPU exclusive prefix-scan kernel in spatial_gpu.
//! Replaces the CPU exclusive-scan at spatial_gpu.rs:917-926.

#![cfg(feature = "gpu")]

use engine_gpu::spatial_gpu::{GpuSpatialHash, GRID_CELLS};

fn cpu_exclusive_scan(input: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; input.len()];
    let mut running: u32 = 0;
    for i in 0..input.len() {
        out[i] = running;
        running = running.saturating_add(input[i]);
    }
    out
}

#[test]
fn prefix_scan_matches_cpu_reference() {
    let mut hash = GpuSpatialHash::new_for_test().expect("spatial init");

    // Deterministic pseudo-random counts.
    let mut counts = vec![0u32; GRID_CELLS as usize];
    let mut s: u32 = 0x1234_5678;
    for c in counts.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *c = s % 64;
    }

    let expected = cpu_exclusive_scan(&counts);
    let actual = hash.run_scan_for_test(&counts).expect("scan dispatch");

    assert_eq!(actual, expected, "GPU scan must equal CPU exclusive-scan");
}

#[test]
fn prefix_scan_all_zeros() {
    let mut hash = GpuSpatialHash::new_for_test().expect("spatial init");
    let counts = vec![0u32; GRID_CELLS as usize];
    let actual = hash.run_scan_for_test(&counts).expect("scan dispatch");
    assert!(actual.iter().all(|&x| x == 0));
}

#[test]
fn prefix_scan_saturates() {
    // If total > u32::MAX, cpu_exclusive_scan saturates; GPU must too.
    let mut hash = GpuSpatialHash::new_for_test().expect("spatial init");
    let mut counts = vec![0u32; GRID_CELLS as usize];
    counts[0] = u32::MAX;
    counts[1] = 10;
    let expected = cpu_exclusive_scan(&counts);
    let actual = hash.run_scan_for_test(&counts).expect("scan dispatch");
    assert_eq!(actual, expected);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --features gpu -p engine_gpu --test gpu_prefix_scan`
Expected: FAIL — `new_for_test` / `run_scan_for_test` do not exist.

- [ ] **Step 3: Add WGSL prefix-scan kernel source**

Modify `crates/engine_gpu/src/spatial_gpu.rs`. Find the existing pipeline-build block (search for `count_pipeline`). Add a new `scan_pipeline` field on `GpuSpatialHash` and its creation. The WGSL source (add as a `const` near the top of the file):

```rust
/// Single-workgroup exclusive prefix-scan over `GRID_CELLS` u32s using
/// Hillis-Steele double-buffering in shared memory. Correct because
/// `GRID_CELLS` fits in one workgroup (32×32×4 = 4096 cells; workgroup
/// size 1024 with 4 serial rounds). If `GRID_CELLS` grows past 4096,
/// switch to a multi-pass Blelloch scan.
const PREFIX_SCAN_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> cell_counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> cell_offsets: array<u32>;

const CELLS: u32 = {GRID_CELLS}u;
const WG: u32 = 1024u;
const CHUNKS: u32 = (CELLS + WG - 1u) / WG;

var<workgroup> scratch_a: array<u32, WG>;
var<workgroup> scratch_b: array<u32, WG>;

@compute @workgroup_size(1024, 1, 1)
fn scan_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    // Running total carried across chunks so we produce a single
    // exclusive scan over CELLS items using only one workgroup.
    var carry: u32 = 0u;

    for (var chunk: u32 = 0u; chunk < CHUNKS; chunk = chunk + 1u) {
        let global_idx = chunk * WG + tid;
        let val = select(0u, cell_counts[global_idx], global_idx < CELLS);
        scratch_a[tid] = val;
        workgroupBarrier();

        // Hillis-Steele inclusive scan over the chunk.
        var offset: u32 = 1u;
        var src_is_a = true;
        while (offset < WG) {
            let prev = select(
                scratch_b[tid - offset],
                scratch_a[tid - offset],
                src_is_a,
            );
            let cur = select(scratch_b[tid], scratch_a[tid], src_is_a);
            let sum = select(cur, saturate_add(cur, prev), tid >= offset);
            if (src_is_a) { scratch_b[tid] = sum; } else { scratch_a[tid] = sum; }
            src_is_a = !src_is_a;
            workgroupBarrier();
            offset = offset << 1u;
        }

        let inclusive = select(scratch_b[tid], scratch_a[tid], src_is_a);
        let exclusive = saturate_sub(inclusive, val);

        if (global_idx < CELLS) {
            cell_offsets[global_idx] = saturate_add(exclusive, carry);
        }
        // Update carry to inclusive-sum of the chunk (value at tid=WG-1).
        workgroupBarrier();
        let chunk_total = select(scratch_b[WG - 1u], scratch_a[WG - 1u], src_is_a);
        carry = saturate_add(carry, chunk_total);
        workgroupBarrier();
    }
}

fn saturate_add(a: u32, b: u32) -> u32 {
    let sum = a + b;
    return select(sum, 0xFFFFFFFFu, sum < a);
}

fn saturate_sub(a: u32, b: u32) -> u32 {
    return select(a - b, 0u, a < b);
}
"#;
```

The `{GRID_CELLS}` placeholder is substituted at runtime — see next step.

- [ ] **Step 4: Compile and bind the scan pipeline**

In `spatial_gpu.rs`, find `impl GpuSpatialHash { pub fn new(...) ... }` (around line 582). Add a `scan_pipeline: wgpu::ComputePipeline` field to the struct (near `count_pipeline`). Build it after the existing pipelines using the same `device.create_shader_module` + `create_compute_pipeline` pattern, substituting `{GRID_CELLS}` via `format!` or `.replace`:

```rust
// In new(), after count_pipeline is built:
let scan_wgsl = PREFIX_SCAN_WGSL.replace("{GRID_CELLS}", &GRID_CELLS.to_string());
let scan_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("engine_gpu::spatial::scan_shader"),
    source: wgpu::ShaderSource::Wgsl(scan_wgsl.into()),
});
let scan_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    label: Some("engine_gpu::spatial::scan_layout"),
    bind_group_layouts: &[&scan_bg_layout], // new BGL, 2 storage buffers
    push_constant_ranges: &[],
});
let scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: Some("engine_gpu::spatial::scan_pipeline"),
    layout: Some(&scan_layout),
    module: &scan_module,
    entry_point: "scan_main",
    compilation_options: Default::default(),
    cache: None,
});
```

Also create `scan_bg_layout` earlier in `new()` with two read_write storage buffers at bindings 0 and 1.

- [ ] **Step 5: Add `new_for_test` + `run_scan_for_test` helpers**

At the bottom of the `impl GpuSpatialHash` block in `spatial_gpu.rs`, add:

```rust
/// Test-only: build a hash without the full rebuild state.
#[doc(hidden)]
pub fn new_for_test() -> Result<Self, SpatialError> {
    let (device, queue) = crate::test_device()?;
    Self::new(&device)
}

/// Test-only: run just the prefix-scan kernel on a user-supplied count
/// vector. Uploads `counts` to cell_counts_buf, dispatches scan, reads
/// back cell_offsets_buf. Used by gpu_prefix_scan.rs only.
#[doc(hidden)]
pub fn run_scan_for_test(
    &mut self,
    counts: &[u32],
) -> Result<Vec<u32>, SpatialError> {
    assert_eq!(counts.len(), GRID_CELLS as usize);
    let device = self.device();
    let queue = self.queue();

    // Acquire or build a pool (pools are per-agent_cap; any agent_cap works).
    let pool = self.ensure_pool_for_test(32)?;

    queue.write_buffer(&pool.cell_counts_buf, 0, bytemuck::cast_slice(counts));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("engine_gpu::spatial::scan_test_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine_gpu::spatial::scan_test_cpass"),
            timestamp_writes: None,
        });
        cpass.set_bind_group(0, &pool.scan_bind_group, &[]);
        cpass.set_pipeline(&self.scan_pipeline);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &pool.cell_offsets_buf,
        0,
        &pool.cell_offsets_readback,
        0,
        (GRID_CELLS as u64) * 4,
    );
    queue.submit(Some(encoder.finish()));

    map_read_u32(&pool.cell_offsets_readback, device, GRID_CELLS as usize)
}
```

`ensure_pool_for_test` + `cell_offsets_readback` may require minor additions to `PoolEntry` — add a `cell_offsets_readback` buffer alongside `cell_counts_readback` and a `scan_bind_group` built during pool construction.

`crate::test_device()` helper — add to `lib.rs` next to `GpuBackend::new()`:

```rust
#[cfg(feature = "gpu")]
#[doc(hidden)]
pub fn test_device() -> Result<(wgpu::Device, wgpu::Queue), GpuInitError> {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or(GpuInitError::NoAdapter)?;
        adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|e| GpuInitError::RequestDevice(e.to_string()))
    })
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test --features gpu -p engine_gpu --test gpu_prefix_scan -- --nocapture`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add crates/engine_gpu/src/spatial_gpu.rs crates/engine_gpu/src/lib.rs crates/engine_gpu/tests/gpu_prefix_scan.rs
git commit -m "feat(engine_gpu): GPU exclusive prefix-scan kernel in spatial hash"
```

---

## Task A2: `rebuild_and_query_resident` method with alive-filter in query kernel

**Files:**
- Modify: `crates/engine_gpu/src/spatial_gpu.rs`

- [ ] **Step 1: Write the failing test**

Append to `crates/engine_gpu/tests/gpu_prefix_scan.rs` (or create `crates/engine_gpu/tests/spatial_resident.rs`):

```rust
#[test]
fn rebuild_and_query_resident_matches_sync_path() {
    // 32 agents in a cluster. Sync path result must equal resident path
    // result for nearby_kin + nearest_hostile at kin_radius=12.
    use engine::creature::CreatureType;
    use engine::state::{AgentSpawn, SimState};
    use engine_gpu::spatial_gpu::GpuSpatialHash;
    use glam::Vec3;

    let mut state = SimState::new(64, 0xCAFE);
    for i in 0..32 {
        let angle = (i as f32) * 0.2;
        let r = 8.0 + (i as f32 % 4.0);
        state
            .spawn_agent(AgentSpawn {
                creature_type: if i % 2 == 0 {
                    CreatureType::Human
                } else {
                    CreatureType::Wolf
                },
                pos: Vec3::new(r * angle.cos(), r * angle.sin(), 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .unwrap();
    }

    let mut hash = GpuSpatialHash::new_for_test().expect("spatial init");
    let sync = hash.rebuild_and_query(hash.device(), hash.queue(), &state, 12.0)
        .expect("sync query");

    // Resident path exposes the same data via GPU buffer handles AND
    // optionally reads it back for inspection.
    let resident = hash.rebuild_and_query_resident_for_test(&state, 12.0)
        .expect("resident query");

    assert_eq!(resident.nearest_hostile, sync.nearest_hostile);
    for i in 0..32 {
        assert_eq!(resident.nearby_kin[i].count, sync.nearby_kin[i].count);
        let sa = resident.nearby_kin[i].as_slice();
        let sb = sync.nearby_kin[i].as_slice();
        assert_eq!(sa, sb, "kin list diverged at slot {i}");
    }
}

#[test]
fn alive_filter_in_query_kernel() {
    // Spawn 4 agents, kill one, assert its id does not appear in any
    // nearby_kin list. Exercises the alive-filter that used to live in
    // filter_dead_from_kin on CPU.
    use engine::creature::CreatureType;
    use engine::state::{AgentSpawn, SimState};
    use engine_gpu::spatial_gpu::GpuSpatialHash;
    use glam::Vec3;

    let mut state = SimState::new(8, 0xFEED);
    for i in 0..4 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32 * 2.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        }).unwrap();
    }
    // Kill agent 2 (id=3 under 1-based indexing).
    state.kill_agent_by_id(3);

    let mut hash = GpuSpatialHash::new_for_test().expect("spatial init");
    let res = hash.rebuild_and_query_resident_for_test(&state, 12.0)
        .expect("resident query");

    for (slot, q) in res.nearby_kin.iter().enumerate() {
        for i in 0..(q.count as usize).min(q.ids.len()) {
            assert_ne!(q.ids[i], 3, "dead id 3 appeared in kin list at slot {slot}");
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --features gpu -p engine_gpu --test gpu_prefix_scan rebuild_and_query_resident -- --nocapture`
Expected: FAIL — `rebuild_and_query_resident_for_test` not defined.

- [ ] **Step 3: Modify the query WGSL to filter dead ids**

Find the existing query kernel WGSL in `spatial_gpu.rs` (search for `nearby_kin — same species`). In the inner loop where kin candidates are pushed, gate on `agents[candidate_slot].alive != 0u`:

```wgsl
// Existing:
//   if (agents[cand_slot].species == self_species && cand_id != self_id) {
//       kin.ids[kin.count] = cand_id;
//       kin.count = kin.count + 1u;
//   }
// Replace with:
if (agents[cand_slot].alive != 0u
    && agents[cand_slot].species == self_species
    && cand_id != self_id) {
    kin.ids[kin.count] = cand_id;
    kin.count = kin.count + 1u;
}
```

Also gate `nearest_hostile` candidate selection on `alive != 0u` — find the `nearest_hostile_to` branch in the same shader.

- [ ] **Step 4: Add `rebuild_and_query_resident`**

Add alongside `rebuild_and_query` in `spatial_gpu.rs`. Signature:

```rust
/// Resident-path sibling to `rebuild_and_query`. Runs the same
/// clear/count/scatter/sort/query pipeline but (a) uses the GPU
/// prefix-scan kernel instead of CPU exclusive-scan, (b) does NOT copy
/// the outputs back to CPU — the caller binds `kin_buf()` /
/// `nearest_buf()` directly into downstream kernels.
///
/// Returns the pool id so the caller can fetch buffer handles. The
/// encoder argument lets the caller record this as part of a larger
/// command buffer (e.g. a whole tick).
pub fn rebuild_and_query_resident(
    &mut self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    state: &SimState,
    radius: f32,
) -> Result<PoolHandle, SpatialError> {
    // 1. Ensure pool exists for this agent_cap (identical to sync path).
    // 2. Upload agents + cfg (queue.write_buffer, not copy).
    // 3. Encode clear + count + scan + scatter + sort + query into the
    //    caller's encoder. No queue.submit, no map_async.
    // 4. Return a handle the caller can use to get buffer references.
    ...
}

pub fn kin_buf(&self, handle: PoolHandle) -> &wgpu::Buffer { ... }
pub fn nearest_buf(&self, handle: PoolHandle) -> &wgpu::Buffer { ... }
```

`PoolHandle` is a tiny wrapper type (a u32 pool index) returned by `ensure_pool`. The buffer accessors return existing fields.

Test-only helper `rebuild_and_query_resident_for_test` builds its own encoder, submits, and reads back for parity assertion:

```rust
#[doc(hidden)]
pub fn rebuild_and_query_resident_for_test(
    &mut self,
    state: &SimState,
    radius: f32,
) -> Result<SpatialQueryResults, SpatialError> {
    let device = self.device();
    let queue = self.queue();
    let mut encoder = device.create_command_encoder(&Default::default());
    let handle = self.rebuild_and_query_resident(device, queue, &mut encoder, state, radius)?;
    // Encode readback.
    let pool = self.pool_ref(handle);
    let agent_cap = state.agent_cap() as u64;
    let kin_bytes = agent_cap * std::mem::size_of::<GpuQueryResult>() as u64;
    let nearest_bytes = agent_cap * 4;
    encoder.copy_buffer_to_buffer(&pool.kin_buf, 0, &pool.kin_readback, 0, kin_bytes);
    encoder.copy_buffer_to_buffer(&pool.nearest_buf, 0, &pool.nearest_readback, 0, nearest_bytes);
    queue.submit(Some(encoder.finish()));
    let kin = map_read_query_results(&pool.kin_readback, device, state.agent_cap() as usize)?;
    let nearest = map_read_u32(&pool.nearest_readback, device, state.agent_cap() as usize)?;
    Ok(SpatialQueryResults { within_radius: vec![], nearby_kin: kin, nearest_hostile: nearest })
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --features gpu -p engine_gpu --test gpu_prefix_scan -- --nocapture`
Expected: 5 passed (3 scan + 2 resident).

Also re-run existing spatial tests to confirm no regression:

Run: `cargo test --features gpu -p engine_gpu --test spatial_parity`
Expected: all pass (sync path unchanged).

- [ ] **Step 6: Commit**

```bash
git add crates/engine_gpu/src/spatial_gpu.rs crates/engine_gpu/tests/gpu_prefix_scan.rs
git commit -m "feat(engine_gpu): rebuild_and_query_resident + alive-filter in query kernel"
```

---

# Phase B — Shared utilities + resident kernel entry points

**Goal**: Add `gpu_util::readback` + `gpu_util::indirect` shared utilities, then add `run_resident` entry points to each kernel module.

## Task B1: `gpu_util::readback` helper

**Files:**
- Create: `crates/engine_gpu/src/gpu_util/mod.rs`
- Create: `crates/engine_gpu/src/gpu_util/readback.rs`
- Modify: `crates/engine_gpu/src/lib.rs` (add `mod gpu_util;`)

- [ ] **Step 1: Write the failing test**

Append to `gpu_prefix_scan.rs` (or create `crates/engine_gpu/tests/readback_helper.rs`):

```rust
#[test]
fn readback_typed_roundtrips_u32_slice() {
    use engine_gpu::gpu_util::readback::readback_typed;

    let (device, queue) = engine_gpu::test_device().expect("test device");
    let src: Vec<u32> = (0..256u32).collect();
    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test::src"),
        contents: bytemuck::cast_slice(&src),
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
    });
    let out: Vec<u32> = readback_typed(&device, &queue, &buf, src.len() * 4).expect("readback");
    assert_eq!(out, src);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --features gpu -p engine_gpu readback_typed_roundtrips -- --nocapture`
Expected: FAIL — `gpu_util` module not defined.

- [ ] **Step 3: Create `gpu_util/mod.rs`**

```rust
//! Shared GPU helpers used by both sync and resident kernel drivers.
//!
//! Factored out of duplicated patterns across physics.rs, mask.rs,
//! spatial_gpu.rs, etc. Each helper is a thin wrapper around a wgpu
//! idiom — kept here so the batch-path drivers in `cascade_resident`
//! and `snapshot` don't reintroduce the duplication.

pub mod indirect;
pub mod readback;
```

- [ ] **Step 4: Create `gpu_util/readback.rs`**

```rust
//! Typed readback helpers. Collapses the `copy_buffer_to_buffer →
//! map_async → device.poll(Wait) → get_mapped_range → cast_slice →
//! unmap` pattern repeated across ~8 sites into one function.

use bytemuck::Pod;

/// Blocking readback. Allocates a throwaway staging buffer, copies
/// `byte_len` bytes from `src`, polls until the callback fires, casts
/// the mapped range to `Vec<T>`.
///
/// Not for the hot path — every call allocates a staging buffer. Use
/// the kernel's pooled staging if the call is per-tick. Fine for
/// tests, init, and the snapshot path (which owns its staging).
pub fn readback_typed<T: Pod + Copy>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    byte_len: usize,
) -> Result<Vec<T>, String> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_util::readback::staging"),
        size: byte_len as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu_util::readback::enc"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, byte_len as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);

    rx.recv()
        .map_err(|e| format!("readback channel closed: {e}"))?
        .map_err(|e| format!("readback map_async: {e:?}"))?;

    let data = slice.get_mapped_range();
    let out: Vec<T> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(out)
}

/// Non-blocking readback handle. Encodes the copy into the caller's
/// encoder (does NOT submit), returns a handle the caller polls later.
/// Used by `snapshot.rs` for the double-buffered staging path.
pub struct PendingReadback<T: Pod + Copy> {
    staging: wgpu::Buffer,
    byte_len: u64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Pod + Copy> PendingReadback<T> {
    pub fn new(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        byte_len: u64,
    ) -> Self {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_util::readback::pending_staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, byte_len);
        Self {
            staging,
            byte_len,
            _marker: std::marker::PhantomData,
        }
    }

    /// Issue `map_async` on the staging buffer. Does NOT poll —
    /// caller polls once after all pending readbacks are kicked off.
    pub fn map_read(&self) -> std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        rx
    }

    /// After `map_read` is awaited, decode mapped range into Vec<T>.
    pub fn take(self) -> Vec<T> {
        let slice = self.staging.slice(..);
        let data = slice.get_mapped_range();
        let out: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging.unmap();
        out
    }
}
```

- [ ] **Step 5: Wire `gpu_util` into `lib.rs`**

Edit `crates/engine_gpu/src/lib.rs`. Near the other `mod` declarations at top of file:

```rust
#[cfg(feature = "gpu")]
pub mod gpu_util;
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test --features gpu -p engine_gpu readback_typed_roundtrips -- --nocapture`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/engine_gpu/src/gpu_util crates/engine_gpu/src/lib.rs crates/engine_gpu/tests/gpu_prefix_scan.rs
git commit -m "feat(engine_gpu): gpu_util::readback helpers (typed + pending)"
```

---

## Task B2: `gpu_util::indirect` helper

**Files:**
- Create: `crates/engine_gpu/src/gpu_util/indirect.rs`

- [ ] **Step 1: Write the failing test**

Append to test file:

```rust
#[test]
fn indirect_args_buffer_layout() {
    use engine_gpu::gpu_util::indirect::{IndirectArgs, IndirectArgsBuffer};

    let (device, queue) = engine_gpu::test_device().expect("test device");
    let ia = IndirectArgsBuffer::new(&device, 4); // 4 slots
    // Initial state: all zeros → no-op dispatch.
    let vals = ia.read(&device, &queue);
    for v in &vals {
        assert_eq!(v.x, 0);
        assert_eq!(v.y, 1);
        assert_eq!(v.z, 1);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `indirect` module not defined.

- [ ] **Step 3: Create `gpu_util/indirect.rs`**

```rust
//! Indirect dispatch args buffer layout + helper.
//!
//! An indirect dispatch reads `(workgroup_x, workgroup_y, workgroup_z)`
//! as three consecutive u32s from a device buffer at
//! `dispatch_workgroups_indirect(buf, offset)` time. A buffer of N
//! slots × 12 B can stage N consecutive indirect dispatches whose
//! workgroup counts are computed by preceding kernels.

use bytemuck::{Pod, Zeroable};

/// One (x, y, z) tuple. Writable by WGSL via
/// `array<vec3<u32>>` binding.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct IndirectArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

pub struct IndirectArgsBuffer {
    buf: wgpu::Buffer,
    slots: u32,
}

impl IndirectArgsBuffer {
    /// `slots` = how many consecutive indirect dispatches this buffer
    /// can drive. Initialised to `(0, 1, 1)` everywhere so dispatches
    /// that read from uninitialised slots no-op.
    pub fn new(device: &wgpu::Device, slots: u32) -> Self {
        let bytes = (slots as u64) * 12;
        let initial: Vec<IndirectArgs> = (0..slots)
            .map(|_| IndirectArgs { x: 0, y: 1, z: 1 })
            .collect();
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_util::indirect::args"),
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let _ = bytes;
        Self { buf, slots }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }

    pub fn slot_offset(&self, slot: u32) -> u64 {
        assert!(slot < self.slots);
        (slot as u64) * 12
    }

    /// Test-only blocking readback.
    #[doc(hidden)]
    pub fn read(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<IndirectArgs> {
        crate::gpu_util::readback::readback_typed(
            device,
            queue,
            &self.buf,
            (self.slots as usize) * 12,
        )
        .expect("indirect args readback")
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --features gpu -p engine_gpu indirect_args_buffer_layout -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/engine_gpu/src/gpu_util/indirect.rs crates/engine_gpu/tests/gpu_prefix_scan.rs
git commit -m "feat(engine_gpu): gpu_util::indirect args buffer helper"
```

---

## Task B3: `MaskKernel::run_resident`

**Files:**
- Modify: `crates/engine_gpu/src/mask.rs`

- [ ] **Step 1: Add `run_resident` method**

Find `impl MaskKernel` in `mask.rs`. The existing `run_and_readback` (or whatever the sync entry point is called — grep for `pub fn run` in the file) follows this shape:

1. Upload agents
2. Encode mask dispatch
3. Copy output bitmaps to readback staging
4. Submit
5. `map_async` + poll + decode → `Vec<Vec<u32>>`

`run_resident` keeps only steps 1 + 2, binds the output buffer directly, and takes the caller's encoder as an argument. Signature:

```rust
/// Resident-path sibling to `run_and_readback`. Records the mask
/// dispatch into `encoder` and does NOT copy the output bitmaps to
/// CPU. The caller binds `mask_bitmaps_buf()` directly into the next
/// kernel (typically scoring).
///
/// Inputs: agent SoA buffer handle. Must be the same agents_buf that
/// downstream kernels bind as input, so the caller owns the layout.
pub fn run_resident(
    &mut self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    agents_buf: &wgpu::Buffer,
    agent_cap: u32,
) -> Result<(), KernelError> {
    self.ensure_pool_cap(device, agent_cap)?;
    let pool = self.pool_ref(); // existing pooled bindings, sized to agent_cap
    queue.write_buffer(&pool.cfg_buf, 0, bytemuck::bytes_of(&self.cfg(agent_cap)));
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine_gpu::mask::cpass_resident"),
            timestamp_writes: None,
        });
        cpass.set_bind_group(0, &pool.bind_group, &[]);
        cpass.set_pipeline(&self.pipeline);
        cpass.dispatch_workgroups(agent_cap.div_ceil(MASK_WORKGROUP_SIZE).max(1), 1, 1);
    }
    Ok(())
}

/// Buffer handle for the mask bitmap output. Stable across ticks; the
/// kernel writes into this same buffer each call.
pub fn mask_bitmaps_buf(&self) -> &wgpu::Buffer {
    &self.pool_ref().bitmap_buf
}
```

Note: the existing `run_and_readback` currently allocates its own `agents_buf` inside the pool. For the resident path, add a **second** pool entry (`resident_pool: Option<PoolEntry>`) whose bind group binds an externally-provided `agents_buf` handle instead of the pool-owned one. The sync path keeps its existing pool untouched. This duplicates ~30 lines of pool-construction code; acceptable to keep the sync path risk-free. Track the handle via raw pointer identity so the resident pool is rebuilt only when a new `agents_buf` is supplied.

- [ ] **Step 2: Smoke-test the new path via an integration check**

No new standalone test file for this task — coverage comes from Phase C (`indirect_cascade_converges`) and Phase D (`async_smoke`). But verify the existing sync path still passes:

Run: `cargo test --features gpu -p engine_gpu --test parity_with_cpu`
Expected: all pass.

Also verify the crate builds:

Run: `cargo build --features gpu -p engine_gpu`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add crates/engine_gpu/src/mask.rs
git commit -m "feat(engine_gpu): MaskKernel::run_resident (resident entry point)"
```

---

## Task B4: `ScoringKernel::run_resident`

**Files:**
- Modify: `crates/engine_gpu/src/scoring.rs`

Same pattern as Task B3, applied to `ScoringKernel`.

- [ ] **Step 1: Add `run_resident` method**

Signature:

```rust
pub fn run_resident(
    &mut self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    agents_buf: &wgpu::Buffer,
    mask_bitmaps_buf: &wgpu::Buffer,
    agent_cap: u32,
) -> Result<(), ScoringError> {
    // Identical shape to mask::run_resident but binds mask_bitmaps_buf
    // as input and writes to self.pool_ref().scoring_buf as output.
    ...
}

pub fn scoring_buf(&self) -> &wgpu::Buffer {
    &self.pool_ref().scoring_buf
}
```

- [ ] **Step 2: Verify the sync path still builds and tests pass**

Run: `cargo test --features gpu -p engine_gpu --test parity_with_cpu`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add crates/engine_gpu/src/scoring.rs
git commit -m "feat(engine_gpu): ScoringKernel::run_resident (resident entry point)"
```

---

## Task B5: `ApplyActionsKernel::run_resident`

**Files:**
- Modify: `crates/engine_gpu/src/apply_actions.rs`

Same pattern. Signature:

```rust
pub fn run_resident(
    &mut self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    agents_buf: &wgpu::Buffer,
    scoring_buf: &wgpu::Buffer,
    event_ring: &GpuEventRing,
    agent_cap: u32,
) -> Result<(), ApplyActionsError> {
    // Encode the apply_actions dispatch; writes into agents_buf
    // (mutating) and appends events to event_ring via atomic add on
    // the tail counter.
    ...
}
```

- [ ] **Step 1: Add method**
- [ ] **Step 2: Verify existing tests pass** — `cargo test --features gpu -p engine_gpu --test parity_with_cpu`
- [ ] **Step 3: Commit** — `git commit -m "feat(engine_gpu): ApplyActionsKernel::run_resident"`

---

## Task B6: `MovementKernel::run_resident`

**Files:**
- Modify: `crates/engine_gpu/src/movement.rs`

- [ ] **Step 1: Add `run_resident` method**

Signature:

```rust
pub fn run_resident(
    &mut self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    agents_buf: &wgpu::Buffer,
    scoring_buf: &wgpu::Buffer,
    event_ring: &GpuEventRing,
    agent_cap: u32,
) -> Result<(), MovementError> {
    self.ensure_pool_cap(device, agent_cap)?;
    let pool = self.pool_ref();
    queue.write_buffer(&pool.cfg_buf, 0, bytemuck::bytes_of(&self.cfg(agent_cap)));
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine_gpu::movement::cpass_resident"),
            timestamp_writes: None,
        });
        cpass.set_bind_group(0, &pool.resident_bind_group, &[]);
        cpass.set_pipeline(&self.pipeline);
        cpass.dispatch_workgroups(agent_cap.div_ceil(MOVEMENT_WORKGROUP_SIZE).max(1), 1, 1);
    }
    Ok(())
}
```

The `resident_bind_group` is a new per-pool field that binds the externally-provided `agents_buf`, `scoring_buf`, and `event_ring` buffers (rather than the pooled agents/scoring copies the sync path uses). Build it during `ensure_pool_cap` when those handles are first seen. Store them as `Option<(u64, u64, u64)>` resource IDs so the bind group is only rebuilt when handles change.

- [ ] **Step 2: Verify existing tests pass**

Run: `cargo test --features gpu -p engine_gpu --test parity_with_cpu`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add crates/engine_gpu/src/movement.rs
git commit -m "feat(engine_gpu): MovementKernel::run_resident (resident entry point)"
```

---

## Task B7: `PhysicsKernel::run_batch_resident` with indirect dispatch

**Files:**
- Modify: `crates/engine_gpu/src/physics.rs`

This is the biggest B task. The existing `PhysicsKernel::run_batch` at `physics.rs:1272` does the full dispatch + readback. `run_batch_resident` differs in three ways:

1. Takes the caller's encoder (does not submit).
2. Uses `dispatch_workgroups_indirect` reading its workgroup count from an `IndirectArgsBuffer` slot.
3. End of kernel: writes `(wg, 1, 1)` to the NEXT slot of the indirect args buffer, where `wg = ceil(out_event_count / PHYSICS_WORKGROUP_SIZE)` clamped to `ceil(agent_cap / PHYSICS_WORKGROUP_SIZE)`. If no events emitted, writes `(0, 1, 1)`.

- [ ] **Step 1: Extend the physics WGSL to write indirect args for the next iter**

Find the WGSL emitter in `physics.rs` (`build_physics_shader`, line 379, or similar). After the main atomic-event-ring-append logic, add at end-of-kernel (single-thread):

```wgsl
// Bind group addition: indirect args buffer + slot_index uniform.
// @group(G) @binding(B) var<storage, read_write> indirect_args: array<vec3<u32>>;
// @group(G) @binding(B+1) var<uniform> next_slot: u32;
// @group(G) @binding(B+2) var<uniform> agent_cap_val: u32;

if (global_id.x == 0u && local_id.x == 0u) {
    // Event ring tail at end of this iter = u32 read from
    // event_ring_tail[0]. Clamp to agent_cap-worth of workgroups.
    let emitted = event_ring_tail;
    let cap_wg = (agent_cap_val + PHYSICS_WORKGROUP_SIZE - 1u) / PHYSICS_WORKGROUP_SIZE;
    let requested = (emitted + PHYSICS_WORKGROUP_SIZE - 1u) / PHYSICS_WORKGROUP_SIZE;
    let wg = min(requested, cap_wg);
    indirect_args[next_slot] = vec3<u32>(wg, 1u, 1u);
}
```

Exact placement depends on the existing emitter's structure. The condition `global_id.x == 0u && local_id.x == 0u` means only one invocation writes, avoiding the race.

- [ ] **Step 2: Add `run_batch_resident` method**

```rust
/// Resident-path sibling to `run_batch`. Records ONE physics
/// iteration dispatch into `encoder` using `dispatch_workgroups_
/// indirect` reading from `indirect_args_buf[read_slot]`. At end of
/// dispatch, the kernel writes the workgroup count for iteration
/// `write_slot` into `indirect_args_buf[write_slot]`.
///
/// The caller records MAX_CASCADE_ITERATIONS calls with chained
/// (read_slot, write_slot) pairs: iter 0 reads slot 0 (pre-seeded by
/// the caller), writes slot 1; iter 1 reads slot 1, writes slot 2;
/// etc. The first slot must be seeded by the caller with the initial
/// event count (typically via copy_buffer_to_buffer from the
/// apply_event_ring tail).
pub fn run_batch_resident(
    &mut self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    agents_buf: &wgpu::Buffer,
    abilities_buf: &wgpu::Buffer,
    kin_buf: &wgpu::Buffer,
    nearest_hostile_buf: &wgpu::Buffer,
    event_ring: &GpuEventRing,
    chronicle_ring: &GpuChronicleRing,
    indirect_args: &IndirectArgsBuffer,
    read_slot: u32,
    write_slot: u32,
    cfg: PhysicsCfg,
) -> Result<(), PhysicsError> {
    self.ensure_pool_cap(device, cfg.agent_cap)?;
    let pool = self.pool_ref();
    queue.write_buffer(&pool.cfg_buf, 0, bytemuck::bytes_of(&cfg));
    queue.write_buffer(&pool.next_slot_buf, 0, bytemuck::bytes_of(&write_slot));

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine_gpu::physics::cpass_resident"),
            timestamp_writes: None,
        });
        cpass.set_bind_group(0, &pool.resident_bind_group, &[]);
        cpass.set_pipeline(&self.pipeline);
        cpass.dispatch_workgroups_indirect(
            indirect_args.buffer(),
            indirect_args.slot_offset(read_slot),
        );
    }
    Ok(())
}
```

`resident_bind_group` is a new per-pool field built during `ensure_pool_cap` that binds `agents_buf`, `kin_buf`, `nearest_hostile_buf`, `event_ring`, `chronicle_ring`, `indirect_args.buffer()`, and `next_slot_buf` as passed in (or held internally).

Note: bind group layouts are baked into the pipeline at compile time, so the *layout* must match what the WGSL declares. Follow the existing pattern at `physics.rs:1150-1240` where bindings are established.

- [ ] **Step 3: Verify the sync path still builds and tests pass**

Run: `cargo test --features gpu -p engine_gpu --test parity_with_cpu --test physics_parity`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add crates/engine_gpu/src/physics.rs
git commit -m "feat(engine_gpu): PhysicsKernel::run_batch_resident (indirect dispatch)"
```

---

# Phase C — `cascade_resident.rs`

**Goal**: New driver module that encodes MAX_CASCADE_ITERATIONS indirect physics dispatches + fold kernels into a single encoder, with no Rust-side iteration loop.

## Task C1: `cascade_resident.rs` module

**Files:**
- Create: `crates/engine_gpu/src/cascade_resident.rs`
- Modify: `crates/engine_gpu/src/lib.rs` (add `mod cascade_resident;`)

- [ ] **Step 1: Create the module skeleton**

```rust
//! GPU-resident cascade driver. Records MAX_CASCADE_ITERATIONS
//! physics iterations + per-iter fold kernels into a single
//! command encoder. No Rust-side convergence check, no per-iter
//! readback — convergence is encoded as indirect dispatch args
//! written by each iteration's physics kernel.
//!
//! Paired with sync cascade in `cascade.rs` — do NOT use this path
//! when determinism or per-tick CPU observability is required.

#![cfg(feature = "gpu")]

use crate::cascade::{MAX_CASCADE_ITERATIONS, CascadeCtx};
use crate::event_ring::GpuEventRing;
use crate::gpu_util::indirect::IndirectArgsBuffer;
use crate::physics::{PhysicsCfg, PhysicsKernel};
use crate::spatial_gpu::GpuSpatialHash;
use crate::view_storage::ViewStorage;
use engine::state::SimState;

/// One-tick resident cascade. Assumes:
/// - `agents_buf`, `apply_event_ring`, `chronicle_ring` are already
///   populated by upstream apply+movement kernels.
/// - `indirect_args` is a buffer of at least `MAX_CASCADE_ITERATIONS+1`
///   slots. Slot 0 holds the initial workgroup count derived from
///   `apply_event_ring.tail`; slots 1..=MAX_ITERS are filled by the
///   physics kernel.
///
/// Encodes N indirect physics dispatches + N fold dispatches into
/// `encoder`. Does NOT submit.
pub fn run_cascade_resident(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    state: &SimState,
    ctx: &mut CascadeCtx,
    view_storage: &mut ViewStorage,
    agents_buf: &wgpu::Buffer,
    apply_event_ring: &GpuEventRing,
    indirect_args: &IndirectArgsBuffer,
    cfg_template: PhysicsCfg,
) -> Result<(), CascadeResidentError> {
    // 1. Spatial precompute (resident path). Two calls, kin_radius +
    //    engagement_range, into the same encoder.
    let kin_radius = state.config.combat.kin_radius;
    let engagement_range = state.config.combat.engagement_range;
    let kin_handle = ctx.spatial.rebuild_and_query_resident(
        device, queue, encoder, state, kin_radius,
    )?;
    let hostile_handle = ctx.spatial.rebuild_and_query_resident(
        device, queue, encoder, state, engagement_range,
    )?;

    // 2. Seed iter 0 indirect args from apply_event_ring.tail.
    //    Kernel `seed_indirect_slot0` reads tail and writes
    //    (ceil(tail / PHYSICS_WORKGROUP_SIZE), 1, 1) into slot 0.
    //    Recorded as a tiny 1-workgroup dispatch.
    ctx.seed_indirect_kernel.record(
        encoder,
        apply_event_ring.tail_buffer(),
        indirect_args,
        /*target_slot*/ 0,
    );

    // 3. MAX_CASCADE_ITERATIONS physics iterations.
    for iter in 0..MAX_CASCADE_ITERATIONS {
        ctx.physics.run_batch_resident(
            device, queue, encoder,
            agents_buf,
            ctx.abilities.buffer(),
            ctx.spatial.kin_buf(kin_handle),
            ctx.spatial.nearest_buf(hostile_handle),
            &ctx.physics_event_ring,
            &ctx.chronicle_ring,
            indirect_args,
            /*read_slot*/ iter,
            /*write_slot*/ iter + 1,
            cfg_template,
        )?;

        // Fold iter events into view_storage (GPU-resident).
        view_storage.fold_iteration_events_resident(
            device, queue, encoder,
            &ctx.physics_event_ring,
            state.tick,
        )?;
    }

    Ok(())
}

#[derive(Debug)]
pub enum CascadeResidentError {
    Physics(crate::physics::PhysicsError),
    Spatial(crate::spatial_gpu::SpatialError),
    ViewStorage(crate::view_storage::ViewStorageError),
}

// ... From impls elided; match cascade.rs conventions ...
```

The `ctx.seed_indirect_kernel` + `view_storage.fold_iteration_events_resident` are minor additions — a 1-line kernel that writes slot 0 from the apply ring tail, and a resident version of the existing fold pipeline that doesn't submit.

- [ ] **Step 2: Wire into `lib.rs`**

```rust
#[cfg(feature = "gpu")]
pub mod cascade_resident;
```

- [ ] **Step 3: Build check**

Run: `cargo build --features gpu -p engine_gpu`
Expected: no errors. (Integration test comes in the next task.)

- [ ] **Step 4: Commit**

```bash
git add crates/engine_gpu/src/cascade_resident.rs crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): cascade_resident driver module (no runtime test yet)"
```

---

## Task C2: `indirect_cascade_converges` unit test

**Files:**
- Create: `crates/engine_gpu/tests/indirect_cascade_converges.rs`

- [ ] **Step 1: Write the test**

```rust
//! Single-tick resident cascade on a fixture that converges in 2
//! iterations. Asserts the indirect args buffer reflects that.

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

const SEED: u64 = 0xC0DE_FACE_CAFE_0001;

#[test]
fn cascade_resident_converges_in_two_iters() {
    // 1 human + 1 wolf 1 m apart — guarantees a single attack event
    // that kills neither in one hit, so cascade needs 2 iters (attack
    // + follow-on fear from kin) then converges.
    let mut gpu = GpuBackend::new().expect("gpu init");
    let mut state = SimState::new(8, SEED);
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(0.0, 0.0, 0.0),
        hp: 100.0,
        ..Default::default()
    }).unwrap();
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(1.0, 0.0, 0.0),
        hp: 80.0,
        ..Default::default()
    }).unwrap();

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warmup.
    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

    // Take a snapshot to read indirect args via the debug accessor.
    // step_batch records the resident cascade; one tick is enough.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 1);
    let snap = gpu.snapshot().expect("snapshot");

    // last_cascade_iterations() reports the number of slots whose
    // workgroup count was > 0 at the end of the batch. For a
    // 2-iteration convergence we expect exactly 2.
    let iters = gpu.last_cascade_iterations().expect("iter count");
    assert!(
        iters >= 2 && iters <= 4,
        "expected 2-4 cascade iterations, got {iters}"
    );
    let _ = snap; // snapshot used to trigger the args readback
}
```

- [ ] **Step 2: Run the test**

Cannot run until Phase D lands (`step_batch` + `snapshot` rewrite). Leave the test in place; it's the integration proof-point for Phase D.

- [ ] **Step 3: Commit**

```bash
git add crates/engine_gpu/tests/indirect_cascade_converges.rs
git commit -m "test(engine_gpu): indirect_cascade_converges (deferred, runs after phase D)"
```

---

# Phase D — `step_batch(n)` + `snapshot()`

**Goal**: Rewrite `GpuBackend::step_batch` to record N ticks' worth of resident dispatches into one command buffer with one submit + one poll. Add `GpuBackend::snapshot()` with double-buffered staging and non-blocking map_async.

## Task D1: `snapshot.rs` module with `GpuStaging` + `GpuSnapshot`

**Files:**
- Create: `crates/engine_gpu/src/snapshot.rs`
- Modify: `crates/engine_gpu/src/lib.rs` (add `mod snapshot;`)

- [ ] **Step 1: Create the module**

```rust
//! Double-buffered snapshot staging. Feeds `GpuBackend::snapshot()`.
//!
//! Three staging pairs: {agents, events, chronicle} × {front, back}.
//! On each snapshot call:
//!   - map_async on front (filled by the previous snapshot call)
//!   - copy_buffer_to_buffer from live GPU buffers into back
//!   - poll → decode front → GpuSnapshot
//!   - swap front / back
//! First call returns GpuSnapshot::empty().

#![cfg(feature = "gpu")]

use crate::event_ring::{EventRecord, GpuEventRing, RECORD_BYTES};
use crate::physics::GpuAgentSlot;
use bytemuck::Pod;

#[derive(Debug, Clone)]
pub struct GpuSnapshot {
    pub tick: u32,
    pub agents: Vec<GpuAgentSlot>,
    pub events_since_last: Vec<EventRecord>,
    // pub chronicle_since_last: Vec<ChronicleEntry>,  // once ChronicleEntry is pub
}

impl GpuSnapshot {
    pub fn empty() -> Self {
        Self {
            tick: 0,
            agents: Vec::new(),
            events_since_last: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub enum SnapshotError {
    Ring(String),
    Map(String),
}

/// One side of the double buffer. Holds staging buffers for agents,
/// events, chronicle, and a watermark tuple (event_ring_read,
/// chronicle_ring_read, tick).
pub struct GpuStaging {
    agents_staging: wgpu::Buffer,
    events_staging: wgpu::Buffer,
    // chronicle_staging: wgpu::Buffer,
    events_len_bytes: u64,
    tick: u32,
    /// Set to true between `kick_copy` and `take_snapshot`; prevents
    /// double-take.
    filled: bool,
}

impl GpuStaging {
    pub fn new(device: &wgpu::Device, agent_cap: u32, event_ring_cap: u32) -> Self {
        let agent_bytes = (agent_cap as u64) * (std::mem::size_of::<GpuAgentSlot>() as u64);
        let event_bytes = (event_ring_cap as u64) * RECORD_BYTES;
        Self {
            agents_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("snapshot::agents_staging"),
                size: agent_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            events_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("snapshot::events_staging"),
                size: event_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            events_len_bytes: 0,
            tick: 0,
            filled: false,
        }
    }

    /// Encode copy_buffer_to_buffer for agents + event slice into
    /// this staging. Does not submit.
    pub fn kick_copy(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        agents_buf: &wgpu::Buffer,
        agent_bytes: u64,
        event_ring: &GpuEventRing,
        event_ring_start: u64,
        event_ring_end: u64,
        tick: u32,
    ) {
        encoder.copy_buffer_to_buffer(agents_buf, 0, &self.agents_staging, 0, agent_bytes);
        let slice_len = event_ring_end.saturating_sub(event_ring_start);
        if slice_len > 0 {
            encoder.copy_buffer_to_buffer(
                event_ring.records_buffer(),
                event_ring_start * RECORD_BYTES,
                &self.events_staging,
                0,
                slice_len * RECORD_BYTES,
            );
        }
        self.events_len_bytes = slice_len * RECORD_BYTES;
        self.tick = tick;
        self.filled = true;
    }

    /// Map the staging for read, poll, decode into a snapshot,
    /// unmap. Assumes `kick_copy` was called on a previous frame
    /// AND that `queue.submit` has been called since.
    pub fn take_snapshot(
        &mut self,
        device: &wgpu::Device,
        agent_count: usize,
    ) -> Result<GpuSnapshot, SnapshotError> {
        if !self.filled {
            return Ok(GpuSnapshot::empty());
        }
        let agent_bytes = (agent_count * std::mem::size_of::<GpuAgentSlot>()) as u64;

        let agents_slice = self.agents_staging.slice(..agent_bytes);
        let events_slice = self.events_staging.slice(..self.events_len_bytes);

        let (atx, arx) = std::sync::mpsc::channel();
        agents_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = atx.send(r); });
        let (etx, erx) = if self.events_len_bytes > 0 {
            let (etx, erx) = std::sync::mpsc::channel();
            events_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = etx.send(r); });
            (Some(etx), Some(erx))
        } else {
            (None, None)
        };
        let _ = device.poll(wgpu::PollType::Wait);

        arx.recv().map_err(|e| SnapshotError::Map(format!("agents rx closed: {e}")))?
            .map_err(|e| SnapshotError::Map(format!("agents map: {e:?}")))?;
        let agents: Vec<GpuAgentSlot> =
            bytemuck::cast_slice(&agents_slice.get_mapped_range()).to_vec();
        self.agents_staging.unmap();

        let events = if let Some(erx) = erx {
            erx.recv().map_err(|e| SnapshotError::Map(format!("events rx closed: {e}")))?
                .map_err(|e| SnapshotError::Map(format!("events map: {e:?}")))?;
            let data: Vec<EventRecord> =
                bytemuck::cast_slice(&events_slice.get_mapped_range()).to_vec();
            self.events_staging.unmap();
            data
        } else {
            Vec::new()
        };

        self.filled = false;
        Ok(GpuSnapshot {
            tick: self.tick,
            agents,
            events_since_last: events,
        })
    }
}
```

- [ ] **Step 2: Wire into `lib.rs`**

```rust
#[cfg(feature = "gpu")]
pub mod snapshot;
```

- [ ] **Step 3: Build check**

Run: `cargo build --features gpu -p engine_gpu`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add crates/engine_gpu/src/snapshot.rs crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): snapshot module — GpuStaging + GpuSnapshot"
```

---

## Task D2: `GpuBackend` persistent agent buffer + staging fields

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs`

- [ ] **Step 1: Add fields to `GpuBackend` struct**

Find `pub struct GpuBackend { ... }` at `lib.rs:133` (the `#[cfg(feature = "gpu")]` version). Add:

```rust
/// Persistent agent SoA buffer for the resident path. Allocated on
/// first step_batch call, reused across ticks. Sync path still uses
/// per-kernel pooled buffers.
resident_agents_buf: Option<wgpu::Buffer>,
resident_agents_cap: u32,

/// Indirect dispatch args for the resident cascade. MAX_ITERS+1 slots.
resident_indirect_args: Option<crate::gpu_util::indirect::IndirectArgsBuffer>,

/// Double-buffered snapshot staging. `front` is the one that will be
/// read next; `back` is the one currently filling.
snapshot_front: Option<crate::snapshot::GpuStaging>,
snapshot_back: Option<crate::snapshot::GpuStaging>,

/// Watermarks for event / chronicle ring: what has been snapshotted.
snapshot_event_ring_read: u64,
snapshot_chronicle_ring_read: u64,
```

Update `GpuBackend::new` to initialise these as `None` / `0`.

- [ ] **Step 2: Build check**

Run: `cargo build --features gpu -p engine_gpu`
Expected: no errors (unused fields are OK for now).

- [ ] **Step 3: Commit**

```bash
git add crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): persistent agent buffer + snapshot staging fields on GpuBackend"
```

---

## Task D3: `GpuBackend::snapshot()` method

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs`

- [ ] **Step 1: Implement `snapshot()`**

Add to `impl GpuBackend` block (above or below `step_batch`):

```rust
/// Cheap non-blocking snapshot via double-buffered staging. First
/// call returns `GpuSnapshot::empty()` (no previous frame to map).
/// Subsequent calls return state as-of the *previous* snapshot call.
///
/// One `device.poll(Wait)` per call.
pub fn snapshot(&mut self) -> Result<snapshot::GpuSnapshot, snapshot::SnapshotError> {
    let device = &self.device;
    let queue = &self.queue;

    // Lazy-init both staging instances.
    if self.snapshot_front.is_none() {
        self.snapshot_front = Some(snapshot::GpuStaging::new(
            device,
            self.resident_agents_cap.max(32),
            event_ring::DEFAULT_CAPACITY,
        ));
        self.snapshot_back = Some(snapshot::GpuStaging::new(
            device,
            self.resident_agents_cap.max(32),
            event_ring::DEFAULT_CAPACITY,
        ));
    }

    // Take snapshot of the front (filled by previous call).
    let front = self.snapshot_front.as_mut().unwrap();
    let snap = front.take_snapshot(device, self.resident_agents_cap as usize)?;

    // Kick copy into the back (filling for next call).
    let back = self.snapshot_back.as_mut().unwrap();
    let agents_buf = self.resident_agents_buf
        .as_ref()
        .ok_or_else(|| snapshot::SnapshotError::Ring(
            "no resident agents buffer; call step_batch first".into()
        ))?;
    let agent_bytes = (self.resident_agents_cap as u64)
        * (std::mem::size_of::<physics::GpuAgentSlot>() as u64);
    let event_ring = /* reference to the main event ring on self */;
    let event_ring_end = /* read event_ring.tail GPU-side or track a CPU-side counter */;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("engine_gpu::snapshot::kick_copy"),
    });
    back.kick_copy(
        &mut encoder,
        agents_buf,
        agent_bytes,
        event_ring,
        self.snapshot_event_ring_read,
        event_ring_end,
        /*tick*/ self.latest_recorded_tick,
    );
    queue.submit(Some(encoder.finish()));
    self.snapshot_event_ring_read = event_ring_end;

    // Swap front / back.
    std::mem::swap(&mut self.snapshot_front, &mut self.snapshot_back);

    Ok(snap)
}
```

**Implementation note**: `event_ring_end` is the tricky part. The main event ring's tail is a GPU-side atomic u32. Options:
- (a) Read tail via a blocking readback just before kick_copy (one small 4-byte readback; acceptable cost per snapshot).
- (b) Track tail on the CPU by having each kernel write its emitted-count into a secondary CPU-tracked buffer.

Go with (a) for simplicity — it's one 4-byte readback per snapshot, not per tick.

- [ ] **Step 2: Build check**

Run: `cargo build --features gpu -p engine_gpu`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): GpuBackend::snapshot() double-buffered observer API"
```

---

## Task D4: Rewrite `GpuBackend::step_batch`

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs`

- [ ] **Step 1: Replace the body of `step_batch`**

Find `pub fn step_batch` at `lib.rs:713`. Replace its body:

```rust
pub fn step_batch<B: PolicyBackend>(
    &mut self,
    state: &mut SimState,
    scratch: &mut SimScratch,
    events: &mut EventRing,
    policy: &B,
    cascade: &CascadeRegistry,
    n_ticks: u32,
) {
    // Batch path: record N ticks into one command buffer, submit
    // once, poll once. Does NOT populate `events` — caller observes
    // via snapshot().
    //
    // Fall back to the sync path (one step() per tick) if the
    // resident cascade init fails, which matches the existing
    // `last_cascade_error` / `CPU fallback` pattern.
    let _ = (scratch, events, policy); // unused on batch path

    if let Err(e) = self.ensure_resident_init(state) {
        eprintln!("engine_gpu::step_batch: resident init failed ({e}), falling back to sync loop");
        for _ in 0..n_ticks {
            <Self as SimBackend>::step(self, state, scratch, events, policy, cascade);
        }
        return;
    }

    let device = &self.device;
    let queue = &self.queue;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("engine_gpu::step_batch::enc"),
    });

    for _ in 0..n_ticks {
        // 1. Mask resident.
        self.mask.run_resident(
            device, queue, &mut encoder,
            self.resident_agents_buf.as_ref().unwrap(),
            state.agent_cap(),
        ).expect("mask resident");
        // 2. Scoring resident.
        self.scoring.run_resident(
            device, queue, &mut encoder,
            self.resident_agents_buf.as_ref().unwrap(),
            self.mask.mask_bitmaps_buf(),
            state.agent_cap(),
        ).expect("scoring resident");
        // 3. Apply + movement resident.
        let ctx = self.ensure_cascade_initialized().expect("cascade init");
        ctx.apply_actions.run_resident(
            device, queue, &mut encoder,
            self.resident_agents_buf.as_ref().unwrap(),
            self.scoring.scoring_buf(),
            &ctx.apply_event_ring,
            state.agent_cap(),
        ).expect("apply resident");
        ctx.movement.run_resident(
            device, queue, &mut encoder,
            self.resident_agents_buf.as_ref().unwrap(),
            self.scoring.scoring_buf(),
            &ctx.apply_event_ring,
            state.agent_cap(),
        ).expect("movement resident");
        // 4. Cascade resident (N iters).
        crate::cascade_resident::run_cascade_resident(
            device, queue, &mut encoder,
            state, ctx,
            self.view_storage.as_mut().unwrap(),
            self.resident_agents_buf.as_ref().unwrap(),
            &ctx.apply_event_ring,
            self.resident_indirect_args.as_ref().unwrap(),
            self.build_physics_cfg(state),
        ).expect("cascade resident");

        // 5. Tick increment (tiny 1-workgroup kernel that updates
        //    state.tick + rng_state on GPU).
        self.tick_advance.record(&mut encoder, &self.cfg_buf);
        state.tick = state.tick.wrapping_add(1);
        self.latest_recorded_tick = state.tick;
    }

    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);
}

fn ensure_resident_init(&mut self, state: &SimState) -> Result<(), String> {
    let cap = state.agent_cap();
    if self.resident_agents_buf.is_none() || self.resident_agents_cap < cap {
        let bytes = (cap as u64) * (std::mem::size_of::<physics::GpuAgentSlot>() as u64);
        self.resident_agents_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::resident_agents"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.resident_agents_cap = cap;
        // Upload initial state from SimState.
        let slots = physics::pack_agent_slots(state);
        self.queue.write_buffer(
            self.resident_agents_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&slots),
        );
    }
    if self.resident_indirect_args.is_none() {
        self.resident_indirect_args = Some(
            crate::gpu_util::indirect::IndirectArgsBuffer::new(
                &self.device,
                (cascade::MAX_CASCADE_ITERATIONS + 1) as u32,
            )
        );
    }
    Ok(())
}
```

`tick_advance` is a new tiny kernel on `GpuBackend` — a 1-workgroup compute shader that increments `state.tick` and advances `rng_state` in the GPU-side `cfg_buf`. Its WGSL is ~10 lines. Add alongside the existing pipeline fields.

- [ ] **Step 2: Build check**

Run: `cargo build --features gpu -p engine_gpu`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): step_batch rewrite — GPU-resident N ticks, one submit"
```

---

## Task D5: `async_smoke` integration test

**Files:**
- Create: `crates/engine_gpu/tests/async_smoke.rs`

- [ ] **Step 1: Write the test**

```rust
//! End-to-end smoke: step_batch(100) at N=2048, one snapshot, assert
//! structural invariants. Does NOT compare to sync path — non-
//! deterministic by design.

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

const SEED: u64 = 0xAAAA_BBBB_CCCC_DDDD;
const N_AGENTS: u32 = 2048;
const BATCH_TICKS: u32 = 100;

fn spawn_fixture() -> SimState {
    let mut state = SimState::new(N_AGENTS + 16, SEED);
    let area = (N_AGENTS as f32 * 10.0).sqrt().ceil();
    let mut s: u64 = SEED;
    for i in 0..N_AGENTS {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let x = (s as f32 / u64::MAX as f32) * area - area * 0.5;
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let y = (s as f32 / u64::MAX as f32) * area - area * 0.5;
        let (ct, hp) = match i % 5 {
            0 | 1 => (CreatureType::Human, 100.0),
            2 | 3 => (CreatureType::Wolf, 80.0),
            _ => (CreatureType::Deer, 60.0),
        };
        state.spawn_agent(AgentSpawn {
            creature_type: ct,
            pos: Vec3::new(x, y, 0.0),
            hp,
            ..Default::default()
        }).unwrap();
    }
    state
}

#[test]
fn step_batch_then_snapshot() {
    let mut gpu = GpuBackend::new().expect("gpu init");
    let mut state = spawn_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warmup: one sync step to pay shader-compile cost.
    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    let tick_after_warmup = state.tick;

    // First snapshot call — returns empty.
    let empty = gpu.snapshot().expect("first snapshot");
    assert!(empty.agents.is_empty(), "first snapshot should be empty");

    // Run batch.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, BATCH_TICKS);
    assert_eq!(state.tick, tick_after_warmup + BATCH_TICKS, "tick must advance by BATCH_TICKS");

    // Second snapshot — should contain state as of previous snapshot
    // (empty staging at that point) → still empty agent slice. But
    // the event slice may still be empty since no kick_copy-and-take
    // happened during the batch.
    let _ = gpu.snapshot();

    // Third snapshot — now returns the batch state.
    let snap = gpu.snapshot().expect("third snapshot");
    assert_eq!(snap.agents.len(), (N_AGENTS + 16) as usize);
    assert!(snap.events_since_last.len() > 0, "events must accumulate during batch");

    // Agent count alive: use the sim's own tracker for cross-check.
    let alive_in_state = state.agents_alive().count();
    let alive_in_snap = snap.agents.iter().filter(|a| a.alive != 0).count();
    let lo = (alive_in_state as f64 * 0.75) as usize;
    let hi = (alive_in_state as f64 * 1.25) as usize;
    assert!(
        alive_in_snap >= lo && alive_in_snap <= hi,
        "snapshot alive count {alive_in_snap} outside ±25% of state {alive_in_state}"
    );
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test --release --features gpu -p engine_gpu --test async_smoke -- --nocapture`
Expected: PASS. If FAIL, the issue is in Phase C/D integration — debug step_batch or snapshot.

- [ ] **Step 3: Commit**

```bash
git add crates/engine_gpu/tests/async_smoke.rs
git commit -m "test(engine_gpu): async_smoke — step_batch + snapshot end-to-end"
```

---

## Task D6: `snapshot_double_buffer` test

**Files:**
- Create: `crates/engine_gpu/tests/snapshot_double_buffer.rs`

- [ ] **Step 1: Write the test**

```rust
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
fn snapshots_do_not_drop_or_duplicate_events() {
    let mut gpu = GpuBackend::new().expect("gpu init");
    let mut state = SimState::new(64, 0xDEAD_BEEF);
    for i in 0..8 {
        state.spawn_agent(AgentSpawn {
            creature_type: if i % 2 == 0 { CreatureType::Human } else { CreatureType::Wolf },
            pos: Vec3::new((i as f32) * 2.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        }).unwrap();
    }
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::with_engine_builtins();

    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

    // Empty snapshot, then two batches, then two snapshots.
    let _empty = gpu.snapshot().unwrap();
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 10);
    let _swap = gpu.snapshot().unwrap();
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 10);
    let snap_a = gpu.snapshot().unwrap();
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 10);
    let snap_b = gpu.snapshot().unwrap();

    // Each snapshot after a batch should contain events strictly
    // from that batch — no overlap with the previous.
    for ea in snap_a.events_since_last.iter() {
        for eb in snap_b.events_since_last.iter() {
            assert_ne!(
                (ea.tick, ea.kind, ea.payload),
                (eb.tick, eb.kind, eb.payload),
                "event appears in two consecutive snapshots"
            );
        }
    }
    // Ticks must be monotonic.
    assert!(snap_b.tick > snap_a.tick);
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test --release --features gpu -p engine_gpu --test snapshot_double_buffer -- --nocapture`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/engine_gpu/tests/snapshot_double_buffer.rs
git commit -m "test(engine_gpu): snapshot_double_buffer — no drops, no duplicates"
```

---

## Task D7: Run the deferred Phase C test

- [ ] **Step 1: Run the deferred `indirect_cascade_converges` test**

Run: `cargo test --release --features gpu -p engine_gpu --test indirect_cascade_converges -- --nocapture`
Expected: PASS.

If FAIL: the physics kernel WGSL from Task B7 is not writing indirect args correctly. Debug by reading back the `IndirectArgsBuffer` contents after one tick and comparing against expected workgroup counts.

- [ ] **Step 2: Run the full test suite to verify no regression**

Run: `cargo test --release --features gpu -p engine_gpu`
Expected: all existing tests + new tests pass.

- [ ] **Step 3: Commit (only if any fixups were needed)**

If Step 1/2 needed code fixes, commit them as a small fixup commit.

---

# Phase E — Perf validation + docs

## Task E1: `--batch-ticks N` flag on `chronicle --perf-sweep`

**Files:**
- Modify: `src/bin/xtask/chronicle_cmd.rs`

- [ ] **Step 1: Add the flag to `ChronicleArgs`**

Find `struct ChronicleArgs { ... }` in `chronicle_cmd.rs`. Add:

```rust
/// Run the perf sweep via the batch API (step_batch + snapshot)
/// instead of tick-by-tick step(). Mutually exclusive with --bench /
/// --sweep. `--batch-ticks` sets how many ticks each step_batch call
/// runs (default 100). Ignored outside --perf-sweep.
#[arg(long, default_value_t = 100)]
batch_ticks: u32,

/// Enable the batch path in --perf-sweep. Without this, --perf-sweep
/// uses the tick-by-tick sync path (today's behaviour). With it,
/// --perf-sweep runs `ticks / batch_ticks` step_batch calls and
/// reports per-batch wall clock and per-tick amortised.
#[arg(long)]
use_batch: bool,
```

- [ ] **Step 2: Add a batch timer alongside `time_gpu_sweep`**

In `chronicle_cmd.rs`, add `time_gpu_batch_sweep` next to `time_gpu_sweep`:

```rust
#[cfg(feature = "gpu")]
fn time_gpu_batch_sweep(
    n: u32,
    ticks: u32,
    batch_ticks: u32,
    seed: u64,
) -> Result<GpuSweepResult, String> {
    use engine::backend::SimBackend as _;

    let mut backend = engine_gpu::GpuBackend::new().map_err(|e| format!("init: {e}"))?;
    backend.rebuild_view_storage(n).map_err(|e| format!("view_storage resize: {e}"))?;

    let (mut state, _) = spawn_perf_fixture(n, seed);
    let mut scratch = engine::step::SimScratch::new(state.agent_cap() as usize);
    let mut events = engine::event::EventRing::with_cap(EVENT_RING_SOFT_CAP);
    let cascade = engine::cascade::CascadeRegistry::with_engine_builtins();

    // Warmup.
    for _ in 0..PERF_WARMUP_TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }

    let total_batches = (ticks + batch_ticks - 1) / batch_ticks;
    let mut per_batch_ns = Vec::with_capacity(total_batches as usize);
    let events_before = events.total_pushed();
    for _ in 0..total_batches {
        let t0 = std::time::Instant::now();
        backend.step_batch(
            &mut state, &mut scratch, &mut events,
            &UtilityBackend, &cascade, batch_ticks,
        );
        per_batch_ns.push(t0.elapsed().as_nanos());
    }
    let events_after = events.total_pushed();
    let total_events = (events_after - events_before) as u64;

    // Per-tick ns = batch ns / batch_ticks.
    let per_tick_ns: Vec<u128> = per_batch_ns.iter()
        .map(|&n| n / batch_ticks as u128)
        .collect();

    Ok(GpuSweepResult {
        per_tick_ns,
        total_events,
        cascade_iters_sum: 0,
        cascade_iters_samples: 0,
    })
}
```

- [ ] **Step 3: Gate the sweep caller on `args.use_batch`**

In `run_perf_sweep`, where `time_gpu_sweep(n, ticks, PERF_SWEEP_SEED)` is called, dispatch on `use_batch`:

```rust
let gpu = if use_batch {
    time_gpu_batch_sweep(n, ticks, batch_ticks, PERF_SWEEP_SEED)
} else {
    time_gpu_sweep(n, ticks, PERF_SWEEP_SEED)
};
```

Propagate `use_batch` + `batch_ticks` from `ChronicleArgs` through `run_perf_sweep`'s signature.

- [ ] **Step 4: Run the sweep on both paths**

Run: `cargo run --release --features gpu --bin xtask -- chronicle --perf-sweep --use-batch --batch-ticks 100 2>&1 | tail -30`
Expected: clean sweep output with the new per-phase table still rendering.

Compare:

Run: `cargo run --release --features gpu --bin xtask -- chronicle --perf-sweep 2>&1 | tail -30`
Expected: sync sweep output.

At N ≥ 512, batch mean µs/tick should be measurably lower than sync mean µs/tick.

- [ ] **Step 5: Commit**

```bash
git add src/bin/xtask/chronicle_cmd.rs
git commit -m "feat(xtask): --use-batch + --batch-ticks flags on chronicle --perf-sweep"
```

---

## Task E2: Perf regression gate check + doc update

- [ ] **Step 1: Run batch sweep at N=2048 and record baseline**

Run: `cargo run --release --features gpu --bin xtask -- chronicle --perf-sweep --use-batch --batch-ticks 100 2>&1 | tail -20`
Expected: table row for N=2048 shows GPU µs/tick clearly less than sync GPU µs/tick (from the earlier sweep, that was ~6,250 µs/tick).

Record the number. If batch µs/tick at N=2048 is not <0.8× sync (5,000 µs/tick), the refactor has not achieved its goal — debug before closing out the plan. Likely culprits:
- Staging buffer allocations inside the hot path (they should be lazy-initialised, not per-call).
- `ensure_resident_init` reallocating each call (check `resident_agents_cap` logic).
- Indirect args buffer not being reused across batches.

- [ ] **Step 2: Update `docs/overview.md` (or equivalent) with the new batch API**

Add a short section under the GPU backend description explaining that `step_batch` + `snapshot` is the recommended path for headless / rendering use, and `step` remains for tests / determinism.

- [ ] **Step 3: Commit**

```bash
git add docs/
git commit -m "docs: batch GPU path in engine_gpu backend overview"
```

---

## Final verification

- [ ] **Run the full test suite in release**

Run: `cargo test --release --features gpu -p engine_gpu`
Expected: all pass, including new `gpu_prefix_scan`, `async_smoke`, `snapshot_double_buffer`, `indirect_cascade_converges`.

- [ ] **Run the perf sweep in both modes**

Run: `cargo run --release --features gpu --bin xtask -- chronicle --perf-sweep`
Run: `cargo run --release --features gpu --bin xtask -- chronicle --perf-sweep --use-batch --batch-ticks 100`

Confirm: N ≥ 512, batch < 0.8× sync µs/tick.

- [ ] **Confirm sync path is untouched**

Run: `cargo test --release --features gpu -p engine_gpu --test parity_with_cpu --test physics_parity --test cascade_parity --test spatial_parity`
Expected: all pass, proving existing callers still work byte-exactly.

---

## Notes for the implementing engineer

- **WGSL gotchas**: `dispatch_workgroups_indirect` requires the args buffer to have `BufferUsages::INDIRECT`. Also `STORAGE` for kernels that write to it. Both are set in `IndirectArgsBuffer::new`.
- **Bind group layout** is baked into pipeline at compile time. When adding a new binding (e.g. the indirect args slot), both the WGSL `@binding` decl AND the Rust `BindGroupLayoutEntry` must match, or pipeline creation panics at runtime with a confusing diagnostic.
- **`queue.write_buffer` inside an encoder**: `queue.write_buffer` is implicitly ordered before any subsequent `queue.submit`, so calling it between kernel dispatches is fine, but it does NOT interleave with the encoder's compute passes — it happens before the first submit.
- **Determinism**: do NOT add a parity assertion between sync and batch paths. Event fold order differs and the test will flake. This is spec'd as a non-goal.
- **First-call empty snapshot**: by design. The `async_smoke` test expects `gpu.snapshot()` before any `step_batch` to return empty.
- **Fallback to sync loop on init failure**: preserves existing behaviour for callers that hit a GPU issue. The current sync path's CPU fallback remains unchanged.

---

## Open questions during implementation

1. **RNG advancement**: `state.rng_state` is CPU-side today. The batch path either (a) leaves it unchanged and accepts non-deterministic RNG advance, (b) adds a GPU-side mirror updated by the tick_advance kernel. Pick (a) for Phase D and defer (b) to a follow-up if needed — the batch path is already non-deterministic.

2. **Chronicle slice in snapshot**: deferred from the spec's snapshot contents because `ChronicleEntry` is not yet public in `engine_gpu`. Add chronicle_since_last to `GpuSnapshot` once the chronicle ring is exposed publicly; for Phase D keep the snapshot to agents + events only. Spec marks this as an open question; acceptable deferral.

3. **View storage fold in resident path**: the existing `fold_iteration_events` at `cascade.rs:364` uses dispatch+submit per view. A resident variant needs to accept an encoder argument. If the fold kernels are fire-and-forget (no readback), the resident variant is just "take an encoder" — minor.
