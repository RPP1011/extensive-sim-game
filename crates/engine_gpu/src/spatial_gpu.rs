//! Phase 5 — GPU spatial hash + query primitives.
//!
//! Port of the CPU `SpatialHash` in `crates/engine/src/spatial.rs` to a
//! GPU-resident grid of cells. Supports the three query primitives the
//! DSL physics rules rely on:
//!
//!   * `within_radius(pos, radius)` — alive agents whose position is
//!     within Euclidean radius of `pos`.
//!   * `nearest_hostile_to(center, radius)` — closest alive hostile,
//!     tied on distance broken by lowest raw `AgentId`.
//!   * `nearby_kin(center, radius)` — alive same-species neighbours of
//!     `center`, excluding `center` itself.
//!
//! ## Rebuild scheme (per tick)
//!
//! The hash rebuilds from scratch each tick — agent motion in a single
//! tick is a small fraction of the cell size for typical speeds, so
//! incremental maintenance would amortise to the same work. Four passes:
//!
//!   1. **Clear + count.** Zero the per-cell counter, then per alive
//!      agent `atomicAdd(&counts[cell_id], 1)`.
//!   2. **Prefix sum.** On CPU — the grid is 64×64=4096 cells, which
//!      trivially fits in an exclusive scan on the host. Result is the
//!      per-cell write offset into the flat cell_data buffer.
//!   3. **Scatter.** Per alive agent, `atomicAdd(&fill[cell_id], 1)`
//!      yields a local index; `cell_data[offsets[cell_id] + local_idx] =
//!      raw_agent_id`.
//!   4. **Sort.** Per cell with more than one occupant, sort the bucket
//!      by raw `AgentId` with a bounded insertion sort (K ≤ 32 →
//!      ~496 compares worst-case). Matches the CPU's
//!      "`within_radius` sorts result by raw id before returning"
//!      discipline, so queries see the same ordering.
//!
//! ## Determinism contract
//!
//! - Atomic increments during scatter give non-deterministic write
//!   ordering *within* a cell, but the post-sort pass restores raw-id
//!   ordering before any query reads it. This matches the CPU's
//!   `within_radius` (`hits.sort_unstable_by_key(|id| id.raw())`).
//! - Query results go through a second insertion sort before readback
//!   since a multi-cell walk aggregates across buckets.
//!
//! ## Fixed-cap K
//!
//! Query results are written into fixed-capacity `array<u32, K>` per
//! agent slot. K = 32 handles dense fixtures up to ~2000 agents in a
//! 20×20 world (expected ~25 neighbours in a radius-5 query — see the
//! parity test's `dense` variant). If a query oversets, the write path
//! clamps to K and sets a `truncated` flag — the parity test asserts
//! the flag is never set for fixtures inside the documented density
//! envelope.
//!
//! ## Not yet: kernel-side exposure
//!
//! Phase 5's deliverable is the spatial hash + a batched query kernel
//! the parity test drives. Inlining the query helpers into
//! scoring/physics kernels (so those can call `within_radius` without a
//! host round-trip) is a follow-on integration — the WGSL helpers in
//! this module are structured to be `#include`-able once the fused
//! scoring / physics kernels land.

use std::fmt;

use bytemuck::{Pod, Zeroable};
use engine::ids::AgentId;
use engine::state::SimState;
use wgpu::util::DeviceExt;

/// GPU cell size in world units. Matches the CPU's `spatial::CELL_SIZE`
/// for apples-to-apples neighbour counts. A 16 m cell gives ~3x3 cell
/// scans for typical 5 m queries; larger radii grow the scan linearly
/// but the constant factor stays small.
pub const CELL_SIZE: f32 = 16.0;

/// Grid extent per axis — 64×64 = 4096 cells, covering a 1024 m × 1024 m
/// playable area centred on `world_origin`. Agents whose mapped cell
/// falls outside this range are clamped to the boundary cell; the
/// parity test uses fixtures well inside this envelope so no clamping
/// fires.
pub const GRID_DIM: u32 = 64;

/// Number of cells in the flat grid buffer. `GRID_DIM * GRID_DIM`.
pub const GRID_CELLS: u32 = GRID_DIM * GRID_DIM;

/// Fixed-cap K per cell — max occupants a single cell can hold without
/// truncation, and also the fixed-cap on `within_radius` result lists.
/// K=32 is documented in the module header; sized for 2000-agent
/// fixtures in ~20×20 worlds (expected ~25 neighbours on a radius-5
/// query with default CELL_SIZE). If a fixture exceeds this density,
/// the truncation flag fires and the parity test fails loudly.
pub const K: u32 = 32;

/// Workgroup size for per-agent kernels. Matches `emit_mask_wgsl::WORKGROUP_SIZE`
/// so the backend can share a layout if it chooses.
pub const WORKGROUP_SIZE: u32 = 64;

/// Workgroup size for per-cell sort kernel. 64 threads each handle one
/// cell's bucket (K=32 items max) via an in-register insertion sort.
pub const SORT_WORKGROUP_SIZE: u32 = 64;

// ---------------------------------------------------------------------------
// Error + GPU-POD wire types
// ---------------------------------------------------------------------------

/// Error surface for spatial-hash init + dispatch. Narrow mirror of
/// `mask::KernelError`.
#[derive(Debug)]
pub enum SpatialError {
    ShaderCompile(String),
    Dispatch(String),
}

impl fmt::Display for SpatialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpatialError::ShaderCompile(s) => write!(f, "shader compile: {s}"),
            SpatialError::Dispatch(s) => write!(f, "dispatch: {s}"),
        }
    }
}

impl std::error::Error for SpatialError {}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct GpuPos {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}

/// Per-spatial-hash config uniform. Packs world origin, cell size, agent
/// cap, etc. Aligns to 16 bytes — wgpu enforces that on uniform blocks.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct SpatialCfg {
    world_origin_x: f32,
    world_origin_y: f32,
    cell_size: f32,
    grid_dim: u32,
    agent_cap: u32,
    k_cap: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct QueryCfg {
    /// Radius for within_radius queries, in world units.
    radius: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// ---------------------------------------------------------------------------
// CPU-side parity + query results layout
// ---------------------------------------------------------------------------

/// Fixed-cap query result. `count` is the number of valid AgentIds in
/// `ids[..count]`. `truncated != 0` iff the kernel tried to push more
/// than K hits and had to clamp. All slots beyond `count` are
/// `u32::MAX` (sentinel).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct GpuQueryResult {
    pub count: u32,
    pub truncated: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub ids: [u32; K as usize],
}

impl Default for GpuQueryResult {
    fn default() -> Self {
        Self {
            count: 0,
            truncated: 0,
            _pad0: 0,
            _pad1: 0,
            ids: [u32::MAX; K as usize],
        }
    }
}

impl GpuQueryResult {
    /// Valid AgentId prefix of `ids`, in raw-id ascending order (the
    /// query kernel insertion-sorts before readback).
    pub fn as_slice(&self) -> &[u32] {
        &self.ids[..self.count as usize]
    }
}

// ---------------------------------------------------------------------------
// WGSL source
// ---------------------------------------------------------------------------

/// WGSL module covering the spatial-hash rebuild passes + the three
/// query primitives. Emitted as one fused source so the host only has
/// to manage one shader module; individual entry points drive each
/// pass.
pub const SPATIAL_WGSL: &str = r#"
// === Bindings ===
//
//  @binding(0)  agent_pos: array<Pos>
//  @binding(1)  agent_alive: array<u32>           // 1 = alive
//  @binding(2)  agent_creature_type: array<u32>   // u8 CreatureType, u32::MAX for unspawned
//  @binding(3)  cell_counts: array<atomic<u32>>   // GRID_CELLS words
//  @binding(4)  cell_offsets: array<u32>          // GRID_CELLS words (CPU-written after prefix sum)
//  @binding(5)  cell_fills: array<atomic<u32>>    // GRID_CELLS words (reset by clear pass)
//  @binding(6)  cell_data: array<u32>             // agent_cap words, raw ids in cell order
//  @binding(7)  within_results: array<QueryResult>
//  @binding(8)  kin_results: array<QueryResult>
//  @binding(9)  nearest_results: array<u32>       // AgentId.raw() or NO_HOSTILE
//  @binding(10) cfg: SpatialCfg
//  @binding(11) qcfg: QueryCfg

struct Pos { x: f32, y: f32, z: f32, _pad: f32 };

struct SpatialCfg {
    world_origin_x: f32,
    world_origin_y: f32,
    cell_size: f32,
    grid_dim: u32,
    agent_cap: u32,
    k_cap: u32,
    _pad0: u32,
    _pad1: u32,
};

struct QueryCfg {
    radius: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct QueryResult {
    count: u32,
    truncated: u32,
    _pad0: u32,
    _pad1: u32,
    ids: array<u32, K_CAP_REPLACE>,
};

@group(0) @binding(0) var<storage, read> agent_pos: array<Pos>;
@group(0) @binding(1) var<storage, read> agent_alive: array<u32>;
@group(0) @binding(2) var<storage, read> agent_creature_type: array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> cell_fills: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> cell_data: array<u32>;
@group(0) @binding(7) var<storage, read_write> within_results: array<QueryResult>;
@group(0) @binding(8) var<storage, read_write> kin_results: array<QueryResult>;
@group(0) @binding(9) var<storage, read_write> nearest_results: array<u32>;
@group(0) @binding(10) var<uniform> cfg: SpatialCfg;
@group(0) @binding(11) var<uniform> qcfg: QueryCfg;

// === Hostility table ===
//
// Mirrors the symmetric closure in `engine_rules::entities::CreatureType::is_hostile_to`.
// Enum encoding: 0=Human, 1=Wolf, 2=Deer, 3=Dragon. Humans↔Wolves, Wolves↔Deer,
// Dragons↔everyone are mutually hostile. Unspawned slots carry
// `u32::MAX` as their creature_type, which fails every clause here
// (no hostility with the void).
fn is_hostile(a: u32, b: u32) -> bool {
    if (a == 0u && b == 1u) { return true; } // Human vs Wolf
    if (a == 0u && b == 3u) { return true; } // Human vs Dragon
    if (a == 1u && b == 0u) { return true; } // Wolf vs Human
    if (a == 1u && b == 2u) { return true; } // Wolf vs Deer
    if (a == 1u && b == 3u) { return true; } // Wolf vs Dragon
    if (a == 2u && b == 1u) { return true; } // Deer vs Wolf
    if (a == 2u && b == 3u) { return true; } // Deer vs Dragon
    if (a == 3u && b == 0u) { return true; } // Dragon vs Human
    if (a == 3u && b == 1u) { return true; } // Dragon vs Wolf
    if (a == 3u && b == 2u) { return true; } // Dragon vs Deer
    return false;
}

// === Cell hash ===
//
// Map world-space `(x, y)` to a flat cell index in
// `[0, grid_dim * grid_dim)`. Out-of-range agents clamp to the grid
// edge; callers guarantee fixtures sit inside the envelope.
fn pos_to_cell(px: f32, py: f32) -> u32 {
    let dx = px - cfg.world_origin_x;
    let dy = py - cfg.world_origin_y;
    var cx = i32(floor(dx / cfg.cell_size));
    var cy = i32(floor(dy / cfg.cell_size));
    let max_idx = i32(cfg.grid_dim) - 1;
    if (cx < 0) { cx = 0; }
    if (cy < 0) { cy = 0; }
    if (cx > max_idx) { cx = max_idx; }
    if (cy > max_idx) { cy = max_idx; }
    return u32(cy) * cfg.grid_dim + u32(cx);
}

// Returns the number of cell steps in each axis a query must scan to
// guarantee no in-range agent is missed. Matches
// `cell_reach_for_radius` on the CPU side.
fn cell_reach_for_radius(radius: f32) -> i32 {
    if (radius <= 0.0) { return 0; }
    let cells = i32(ceil(radius / cfg.cell_size));
    if (cells < 1) { return 1; }
    if (cells > 256) { return 256; }
    return cells;
}

// === Pass 1a: clear counters ===
@compute @workgroup_size(64)
fn cs_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cfg.grid_dim * cfg.grid_dim) { return; }
    atomicStore(&cell_counts[idx], 0u);
    atomicStore(&cell_fills[idx], 0u);
}

// === Pass 1b: count ===
@compute @workgroup_size(64)
fn cs_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= cfg.agent_cap) { return; }
    if (agent_alive[slot] == 0u) { return; }
    let p = agent_pos[slot];
    let cell = pos_to_cell(p.x, p.y);
    atomicAdd(&cell_counts[cell], 1u);
}

// === Pass 1c: scatter ===
@compute @workgroup_size(64)
fn cs_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= cfg.agent_cap) { return; }
    if (agent_alive[slot] == 0u) { return; }
    let p = agent_pos[slot];
    let cell = pos_to_cell(p.x, p.y);
    let local = atomicAdd(&cell_fills[cell], 1u);
    let offset = cell_offsets[cell];
    cell_data[offset + local] = slot + 1u;
}

// === Pass 1d: sort each cell's bucket by raw AgentId ascending ===
//
// With the query kernel doing top-K-by-lowest-id selection, the cell
// bucket itself doesn't need to be pre-sorted for correctness — but
// sorting it lets the query cut off scanning once it's seen an id
// greater than its current K-th smallest, and keeps rebuild behaviour
// deterministic for diagnostics. One thread per cell, insertion-sort
// the bucket. At the densest Phase-5 fixture (~500 agents/cell in the
// dense parity test) this is ~125K compares per cell on a handful of
// cells — still cheap next to the global query scan.
@compute @workgroup_size(64)
fn cs_sort(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= cfg.grid_dim * cfg.grid_dim) { return; }
    let offset = cell_offsets[cell];
    let n = atomicLoad(&cell_counts[cell]);
    if (n <= 1u) { return; }
    for (var i: u32 = 1u; i < n; i = i + 1u) {
        let x = cell_data[offset + i];
        var j: i32 = i32(i) - 1;
        loop {
            if (j < 0) { break; }
            let yv = cell_data[offset + u32(j)];
            if (yv <= x) { break; }
            cell_data[offset + u32(j + 1)] = yv;
            j = j - 1;
        }
        cell_data[offset + u32(j + 1)] = x;
    }
}

// === Sorted-insert helper ===
//
// Insert `new_id` into the top-K-by-lowest-id buffer `ids` (length
// `count`, cap `k_cap`). If the buffer isn't full, insertion-sort as
// usual. If it is full, drop the new id unless it's smaller than the
// current max — in that case shift the tail and replace. Sets the
// truncated flag whenever a push was refused. Callers maintain the
// buffer in ascending-order throughout.
fn sorted_insert_topk(count_ptr: ptr<function, u32>,
                      trunc_ptr: ptr<function, u32>,
                      ids: ptr<function, array<u32, K_CAP_REPLACE>>,
                      new_id: u32) {
    let c = *count_ptr;
    if (c < cfg.k_cap) {
        var j: i32 = i32(c) - 1;
        loop {
            if (j < 0) { break; }
            let yv = (*ids)[u32(j)];
            if (yv <= new_id) { break; }
            (*ids)[u32(j + 1)] = yv;
            j = j - 1;
        }
        (*ids)[u32(j + 1)] = new_id;
        *count_ptr = c + 1u;
    } else {
        let max_idx_k = cfg.k_cap - 1u;
        if (new_id < (*ids)[max_idx_k]) {
            var j: i32 = i32(max_idx_k) - 1;
            loop {
                if (j < 0) { break; }
                let yv = (*ids)[u32(j)];
                if (yv <= new_id) { break; }
                (*ids)[u32(j + 1)] = yv;
                j = j - 1;
            }
            (*ids)[u32(j + 1)] = new_id;
        }
        *trunc_ptr = 1u;
    }
}

// === Pass 2: batched query ===
//
// One thread per agent slot. Alive agents compute all three query
// primitives in a single cell walk:
//   * within_radius (3-D) → within_results[slot]
//   * nearby_kin          → kin_results[slot] (same species, excludes self)
//   * nearest_hostile_to  → nearest_results[slot] (raw id or NO_HOSTILE)
// Dead / unspawned slots write empty results and NO_HOSTILE.
@compute @workgroup_size(64)
fn cs_query(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= cfg.agent_cap) { return; }

    // Default-init outputs. Slots past `count` stay at u32::MAX
    // sentinel.
    var w_count: u32 = 0u;
    var w_trunc: u32 = 0u;
    var w_ids: array<u32, K_CAP_REPLACE>;
    for (var i: u32 = 0u; i < cfg.k_cap; i = i + 1u) { w_ids[i] = 0xFFFFFFFFu; }

    var k_count: u32 = 0u;
    var k_trunc: u32 = 0u;
    var k_ids: array<u32, K_CAP_REPLACE>;
    for (var i: u32 = 0u; i < cfg.k_cap; i = i + 1u) { k_ids[i] = 0xFFFFFFFFu; }

    var nearest: u32 = 0xFFFFFFFFu;

    if (agent_alive[slot] == 0u) {
        within_results[slot].count = 0u;
        within_results[slot].truncated = 0u;
        kin_results[slot].count = 0u;
        kin_results[slot].truncated = 0u;
        for (var i: u32 = 0u; i < cfg.k_cap; i = i + 1u) {
            within_results[slot].ids[i] = 0xFFFFFFFFu;
            kin_results[slot].ids[i] = 0xFFFFFFFFu;
        }
        nearest_results[slot] = nearest;
        return;
    }

    let my_id = slot + 1u;
    let my_pos = agent_pos[slot];
    let my_ct = agent_creature_type[slot];
    let radius = qcfg.radius;
    let r2 = radius * radius;
    let cx = i32(floor((my_pos.x - cfg.world_origin_x) / cfg.cell_size));
    let cy = i32(floor((my_pos.y - cfg.world_origin_y) / cfg.cell_size));
    let reach = cell_reach_for_radius(radius);
    let max_idx = i32(cfg.grid_dim) - 1;

    var best_id: u32 = 0xFFFFFFFFu;
    var best_d2: f32 = 3.4e38;

    for (var dy: i32 = -reach; dy <= reach; dy = dy + 1) {
        for (var dx: i32 = -reach; dx <= reach; dx = dx + 1) {
            let tx = cx + dx;
            let ty = cy + dy;
            if (tx < 0) { continue; }
            if (ty < 0) { continue; }
            if (tx > max_idx) { continue; }
            if (ty > max_idx) { continue; }
            let cell = u32(ty) * cfg.grid_dim + u32(tx);
            let offset = cell_offsets[cell];
            let raw_count = atomicLoad(&cell_counts[cell]);
            for (var i: u32 = 0u; i < raw_count; i = i + 1u) {
                let other_id = cell_data[offset + i];
                if (other_id == 0u) { continue; }
                let other_slot = other_id - 1u;
                if (agent_alive[other_slot] == 0u) { continue; }
                let op = agent_pos[other_slot];
                let ex = op.x - my_pos.x;
                let ey = op.y - my_pos.y;
                let ez = op.z - my_pos.z;
                let d2 = ex * ex + ey * ey + ez * ez;
                if (d2 > r2) { continue; }

                // within_radius includes self when alive + in range —
                // matches `SpatialHash::within_radius`. Callers who need
                // self-exclusion strip it themselves.
                sorted_insert_topk(&w_count, &w_trunc, &w_ids, other_id);

                if (other_id == my_id) { continue; }

                let other_ct = agent_creature_type[other_slot];

                // nearby_kin — same species, excludes self.
                if (other_ct == my_ct) {
                    sorted_insert_topk(&k_count, &k_trunc, &k_ids, other_id);
                }

                // nearest_hostile_to — closest alive hostile, ties
                // broken on ascending raw id. Using squared distance
                // matches the CPU helper's ordering up to f32 precision.
                if (is_hostile(my_ct, other_ct)) {
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_id = other_id;
                    } else if (d2 == best_d2 && other_id < best_id) {
                        best_id = other_id;
                    }
                }
            }
        }
    }

    within_results[slot].count = w_count;
    within_results[slot].truncated = w_trunc;
    for (var i: u32 = 0u; i < cfg.k_cap; i = i + 1u) { within_results[slot].ids[i] = w_ids[i]; }
    kin_results[slot].count = k_count;
    kin_results[slot].truncated = k_trunc;
    for (var i: u32 = 0u; i < cfg.k_cap; i = i + 1u) { kin_results[slot].ids[i] = k_ids[i]; }
    nearest_results[slot] = best_id;
}
"#;

// ---------------------------------------------------------------------------
// Host-side pipeline + buffer pool
// ---------------------------------------------------------------------------

/// Compiled spatial-hash pipelines + per-tick buffer pool. One instance
/// per `GpuBackend`; `rebuild_and_query` owns the full dispatch sequence.
pub struct GpuSpatialHash {
    clear_pipeline: wgpu::ComputePipeline,
    count_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    sort_pipeline: wgpu::ComputePipeline,
    query_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pool: Option<BufferPool>,
}

#[allow(dead_code)] // Several buffers only ever referenced through the bind group.
struct BufferPool {
    agent_cap: u32,
    pos_buf: wgpu::Buffer,
    alive_buf: wgpu::Buffer,
    creature_type_buf: wgpu::Buffer,
    cell_counts_buf: wgpu::Buffer,
    cell_counts_readback: wgpu::Buffer,
    cell_offsets_buf: wgpu::Buffer,
    cell_fills_buf: wgpu::Buffer,
    cell_data_buf: wgpu::Buffer,
    within_buf: wgpu::Buffer,
    within_readback: wgpu::Buffer,
    kin_buf: wgpu::Buffer,
    kin_readback: wgpu::Buffer,
    nearest_buf: wgpu::Buffer,
    nearest_readback: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    qcfg_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Aggregated per-agent query results, one entry per slot in spawn
/// order. Empty slots (dead / unspawned) report `count=0` for the
/// lists and `NO_HOSTILE` for `nearest_hostile`.
pub struct SpatialQueryResults {
    pub within_radius: Vec<GpuQueryResult>,
    pub nearby_kin: Vec<GpuQueryResult>,
    pub nearest_hostile: Vec<u32>,
}

impl GpuSpatialHash {
    /// Build the pipelines on `device`. Parses the shared WGSL source
    /// once via wgpu's `naga` frontend; five compute pipelines share
    /// the one shader module + one bind-group layout.
    pub fn new(device: &wgpu::Device) -> Result<Self, SpatialError> {
        let src = SPATIAL_WGSL.replace("K_CAP_REPLACE", &K.to_string());

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine_gpu::spatial::wgsl"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(SpatialError::ShaderCompile(format!("{err}")));
        }

        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let entries = [
            storage_entry(0, true),  // agent_pos
            storage_entry(1, true),  // agent_alive
            storage_entry(2, true),  // agent_creature_type
            storage_entry(3, false), // cell_counts (atomic)
            storage_entry(4, true),  // cell_offsets (CPU-written)
            storage_entry(5, false), // cell_fills (atomic)
            storage_entry(6, false), // cell_data
            storage_entry(7, false), // within_results
            storage_entry(8, false), // kin_results
            storage_entry(9, false), // nearest_results
            uniform_entry(10),       // cfg
            uniform_entry(11),       // qcfg
        ];
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine_gpu::spatial::bgl"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine_gpu::spatial::pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let make_pipe = |entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("engine_gpu::spatial::cp_{entry}")),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let clear_pipeline = make_pipe("cs_clear");
        let count_pipeline = make_pipe("cs_count");
        let scatter_pipeline = make_pipe("cs_scatter");
        let sort_pipeline = make_pipe("cs_sort");
        let query_pipeline = make_pipe("cs_query");

        Ok(Self {
            clear_pipeline,
            count_pipeline,
            scatter_pipeline,
            sort_pipeline,
            query_pipeline,
            bind_group_layout,
            pool: None,
        })
    }

    fn ensure_pool(&mut self, device: &wgpu::Device, agent_cap: u32) {
        if let Some(p) = &self.pool {
            if p.agent_cap == agent_cap {
                return;
            }
        }
        let grid_cells = GRID_CELLS as usize;
        let agent_cap_u = agent_cap as usize;

        let pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::spatial::agent_pos"),
            size: (agent_cap_u * std::mem::size_of::<GpuPos>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let alive_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::spatial::agent_alive"),
            size: (agent_cap_u * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let creature_type_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::spatial::agent_creature_type"),
            size: (agent_cap_u * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::spatial::cell_counts"),
            size: (grid_cells * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cell_counts_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::spatial::cell_counts_readback"),
            size: (grid_cells * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::spatial::cell_offsets"),
            size: (grid_cells * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_fills_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::spatial::cell_fills"),
            size: (grid_cells * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_data_bytes = (agent_cap_u.max(1) * 4) as u64;
        let cell_data_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::spatial::cell_data"),
            size: cell_data_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let qr_bytes = (agent_cap_u * std::mem::size_of::<GpuQueryResult>()) as u64;
        let within_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::spatial::within_results"),
            size: qr_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let within_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::spatial::within_results_readback"),
            size: qr_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let kin_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::spatial::kin_results"),
            size: qr_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let kin_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::spatial::kin_results_readback"),
            size: qr_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let nearest_bytes = (agent_cap_u * 4) as u64;
        let nearest_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::spatial::nearest_results"),
            size: nearest_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let nearest_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::spatial::nearest_results_readback"),
            size: nearest_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cfg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine_gpu::spatial::cfg"),
            contents: bytemuck::cast_slice(&[SpatialCfg {
                world_origin_x: 0.0,
                world_origin_y: 0.0,
                cell_size: CELL_SIZE,
                grid_dim: GRID_DIM,
                agent_cap,
                k_cap: K,
                _pad0: 0,
                _pad1: 0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let qcfg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine_gpu::spatial::qcfg"),
            contents: bytemuck::cast_slice(&[QueryCfg {
                radius: 0.0,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine_gpu::spatial::bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: alive_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: creature_type_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cell_counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cell_offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cell_fills_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: cell_data_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: within_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: kin_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: nearest_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: cfg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: qcfg_buf.as_entire_binding() },
            ],
        });

        self.pool = Some(BufferPool {
            agent_cap,
            pos_buf,
            alive_buf,
            creature_type_buf,
            cell_counts_buf,
            cell_counts_readback,
            cell_offsets_buf,
            cell_fills_buf,
            cell_data_buf,
            within_buf,
            within_readback,
            kin_buf,
            kin_readback,
            nearest_buf,
            nearest_readback,
            cfg_buf,
            qcfg_buf,
            bind_group,
        });
    }

    /// Rebuild the spatial hash from `state`, then run the batched
    /// `within_radius` kernel with the given radius. Per-slot results
    /// are indexed by `(AgentId.raw() - 1)`.
    pub fn rebuild_and_query(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &SimState,
        radius: f32,
    ) -> Result<SpatialQueryResults, SpatialError> {
        let agent_cap = state.agent_cap();
        self.ensure_pool(device, agent_cap);

        let pos_src: Vec<GpuPos> = state
            .hot_pos()
            .iter()
            .map(|v| GpuPos { x: v.x, y: v.y, z: v.z, _pad: 0.0 })
            .collect();
        let alive_src: Vec<u32> = state
            .hot_alive()
            .iter()
            .map(|&b| if b { 1u32 } else { 0u32 })
            .collect();
        let ct_src: Vec<u32> = (0..agent_cap)
            .map(|slot| {
                let id = AgentId::new(slot + 1).unwrap();
                match state.agent_creature_type(id) {
                    Some(ct) => ct as u8 as u32,
                    None => u32::MAX,
                }
            })
            .collect();
        let (world_origin_x, world_origin_y) = compute_world_origin(state);

        let pool = self.pool.as_ref().expect("pool ensured");
        queue.write_buffer(&pool.pos_buf, 0, bytemuck::cast_slice(&pos_src));
        queue.write_buffer(&pool.alive_buf, 0, bytemuck::cast_slice(&alive_src));
        queue.write_buffer(&pool.creature_type_buf, 0, bytemuck::cast_slice(&ct_src));
        queue.write_buffer(
            &pool.cfg_buf,
            0,
            bytemuck::cast_slice(&[SpatialCfg {
                world_origin_x,
                world_origin_y,
                cell_size: CELL_SIZE,
                grid_dim: GRID_DIM,
                agent_cap,
                k_cap: K,
                _pad0: 0,
                _pad1: 0,
            }]),
        );
        queue.write_buffer(
            &pool.qcfg_buf,
            0,
            bytemuck::cast_slice(&[QueryCfg {
                radius,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            }]),
        );

        // --- Phase A: clear + count ---
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine_gpu::spatial::enc_count"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::spatial::cpass_clear_count"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &pool.bind_group, &[]);
            cpass.set_pipeline(&self.clear_pipeline);
            cpass.dispatch_workgroups(GRID_CELLS.div_ceil(WORKGROUP_SIZE).max(1), 1, 1);
            cpass.set_pipeline(&self.count_pipeline);
            cpass.dispatch_workgroups(agent_cap.div_ceil(WORKGROUP_SIZE).max(1), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &pool.cell_counts_buf,
            0,
            &pool.cell_counts_readback,
            0,
            (GRID_CELLS as u64) * 4,
        );
        queue.submit(Some(encoder.finish()));

        // Readback counts, exclusive-scan on CPU, upload offsets.
        let counts_vec = map_read_u32(&pool.cell_counts_readback, device, GRID_CELLS as usize)?;
        let mut offsets = vec![0u32; GRID_CELLS as usize];
        let mut running: u32 = 0;
        for i in 0..(GRID_CELLS as usize) {
            offsets[i] = running;
            let c = counts_vec[i];
            running = running.saturating_add(c);
        }
        queue.write_buffer(&pool.cell_offsets_buf, 0, bytemuck::cast_slice(&offsets));

        // --- Phase B: scatter + sort + query ---
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine_gpu::spatial::enc_scatter_query"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine_gpu::spatial::cpass_scatter_query"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &pool.bind_group, &[]);
            cpass.set_pipeline(&self.scatter_pipeline);
            cpass.dispatch_workgroups(agent_cap.div_ceil(WORKGROUP_SIZE).max(1), 1, 1);
            cpass.set_pipeline(&self.sort_pipeline);
            cpass.dispatch_workgroups(GRID_CELLS.div_ceil(SORT_WORKGROUP_SIZE).max(1), 1, 1);
            cpass.set_pipeline(&self.query_pipeline);
            cpass.dispatch_workgroups(agent_cap.div_ceil(WORKGROUP_SIZE).max(1), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &pool.within_buf,
            0,
            &pool.within_readback,
            0,
            (agent_cap as u64) * std::mem::size_of::<GpuQueryResult>() as u64,
        );
        encoder.copy_buffer_to_buffer(
            &pool.kin_buf,
            0,
            &pool.kin_readback,
            0,
            (agent_cap as u64) * std::mem::size_of::<GpuQueryResult>() as u64,
        );
        encoder.copy_buffer_to_buffer(
            &pool.nearest_buf,
            0,
            &pool.nearest_readback,
            0,
            (agent_cap as u64) * 4,
        );
        queue.submit(Some(encoder.finish()));

        let within = map_read_query_results(&pool.within_readback, device, agent_cap as usize)?;
        let kin = map_read_query_results(&pool.kin_readback, device, agent_cap as usize)?;
        let nearest = map_read_u32(&pool.nearest_readback, device, agent_cap as usize)?;
        Ok(SpatialQueryResults {
            within_radius: within,
            nearby_kin: kin,
            nearest_hostile: nearest,
        })
    }
}

/// Compute the AABB of alive agent positions (XY) and pick a world
/// origin that fits the grid over them. Falls back to `(0, 0)` if no
/// agent is alive.
fn compute_world_origin(state: &SimState) -> (f32, f32) {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let positions = state.hot_pos();
    let alive = state.hot_alive();
    for (i, p) in positions.iter().enumerate() {
        if i < alive.len() && alive[i] {
            if p.x < min_x { min_x = p.x; }
            if p.y < min_y { min_y = p.y; }
            if p.x > max_x { max_x = p.x; }
            if p.y > max_y { max_y = p.y; }
        }
    }
    if !min_x.is_finite() {
        return (0.0, 0.0);
    }
    let span = GRID_DIM as f32 * CELL_SIZE;
    let extent_x = max_x - min_x;
    let extent_y = max_y - min_y;
    let origin_x = if extent_x >= span { min_x } else { min_x - (span - extent_x) * 0.5 };
    let origin_y = if extent_y >= span { min_y } else { min_y - (span - extent_y) * 0.5 };
    (origin_x, origin_y)
}

fn map_read_u32(
    buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    n: usize,
) -> Result<Vec<u32>, SpatialError> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    let map_result = rx
        .recv()
        .map_err(|e| SpatialError::Dispatch(format!("map_async channel closed: {e}")))?;
    map_result.map_err(|e| SpatialError::Dispatch(format!("map_async: {e:?}")))?;
    let data = slice.get_mapped_range();
    let mut out = vec![0u32; n];
    out.copy_from_slice(&bytemuck::cast_slice(&data)[..n]);
    drop(data);
    buffer.unmap();
    Ok(out)
}

// ---------------------------------------------------------------------------
// CPU reference — portable scan via the engine's own `SpatialHash`. Used
// by the parity test. Each entry holds the raw `AgentId::raw()` values
// in ascending order — matches the GPU `GpuQueryResult::as_slice`
// semantics so the parity test can compare with `assert_eq!` on the
// underlying `&[u32]`.
// ---------------------------------------------------------------------------

/// `within_radius` CPU reference for the whole agent_cap. Kept for
/// callers from the Phase 5 commit 1 that only wanted the within_radius
/// arm — the kin / nearest arms live in `cpu_reference`.
pub struct CpuWithinRadiusReference {
    pub within_radius: Vec<Vec<u32>>,
}

/// Full CPU reference — mirrors `SpatialQueryResults` layout. Per-slot
/// entries are in raw-id ascending order (within / kin), or carry
/// `NO_HOSTILE` when no hostile is in range (nearest_hostile).
pub struct CpuSpatialReference {
    pub within_radius: Vec<Vec<u32>>,
    pub nearby_kin: Vec<Vec<u32>>,
    pub nearest_hostile: Vec<u32>,
}

/// Sentinel returned by `nearest_hostile_to` when no hostile is in
/// range. Matches the CPU's `None`.
pub const NO_HOSTILE: u32 = u32::MAX;

/// Compute the CPU-side `within_radius` reference for a given radius.
/// Per-slot result is the engine's `SpatialHash::within_radius`
/// filtered to alive agents and mapped to `raw()` ids — matches the
/// GPU kernel's "include-self-if-alive-and-in-range" semantics.
pub fn cpu_reference_within(state: &SimState, radius: f32) -> CpuWithinRadiusReference {
    let agent_cap = state.agent_cap() as usize;
    let mut within: Vec<Vec<u32>> = vec![Vec::new(); agent_cap];

    let spatial = state.spatial();
    for slot in 0..(agent_cap as u32) {
        let id = match AgentId::new(slot + 1) {
            Some(id) => id,
            None => continue,
        };
        if !state.agent_alive(id) {
            continue;
        }
        let pos = match state.agent_pos(id) {
            Some(p) => p,
            None => continue,
        };
        let mut hits: Vec<u32> = spatial
            .within_radius(state, pos, radius)
            .into_iter()
            .filter(|other| state.agent_alive(*other))
            .map(|a| a.raw())
            .collect();
        hits.sort_unstable();
        within[slot as usize] = hits;
    }
    CpuWithinRadiusReference { within_radius: within }
}

/// Full CPU reference for all three query primitives.
///
/// `within_radius` goes through `SpatialHash::within_radius`; `kin` and
/// `nearest_hostile` use the dedicated `engine::spatial::*` helpers so
/// parity is checked against the exact code paths physics rules call
/// (`engagement_on_move`, `fear_spread_on_death`).
pub fn cpu_reference(state: &SimState, radius: f32) -> CpuSpatialReference {
    let agent_cap = state.agent_cap() as usize;
    let mut within: Vec<Vec<u32>> = vec![Vec::new(); agent_cap];
    let mut kin: Vec<Vec<u32>> = vec![Vec::new(); agent_cap];
    let mut nearest: Vec<u32> = vec![NO_HOSTILE; agent_cap];

    let spatial = state.spatial();
    for slot in 0..(agent_cap as u32) {
        let id = match AgentId::new(slot + 1) {
            Some(id) => id,
            None => continue,
        };
        if !state.agent_alive(id) {
            continue;
        }
        let pos = match state.agent_pos(id) {
            Some(p) => p,
            None => continue,
        };
        let mut hits: Vec<u32> = spatial
            .within_radius(state, pos, radius)
            .into_iter()
            .filter(|other| state.agent_alive(*other))
            .map(|a| a.raw())
            .collect();
        hits.sort_unstable();
        within[slot as usize] = hits;

        kin[slot as usize] = engine::spatial::nearby_kin(state, id, radius)
            .into_iter()
            .map(|a| a.raw())
            .collect();

        nearest[slot as usize] = engine::spatial::nearest_hostile_to(state, id, radius)
            .map(|a| a.raw())
            .unwrap_or(NO_HOSTILE);
    }
    CpuSpatialReference {
        within_radius: within,
        nearby_kin: kin,
        nearest_hostile: nearest,
    }
}

fn map_read_query_results(
    buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    n: usize,
) -> Result<Vec<GpuQueryResult>, SpatialError> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    let map_result = rx
        .recv()
        .map_err(|e| SpatialError::Dispatch(format!("map_async channel closed: {e}")))?;
    map_result.map_err(|e| SpatialError::Dispatch(format!("map_async: {e:?}")))?;
    let data = slice.get_mapped_range();
    let casted: &[GpuQueryResult] = bytemuck::cast_slice(&data);
    let out = casted[..n].to_vec();
    drop(data);
    buffer.unmap();
    Ok(out)
}
