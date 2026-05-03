//! Compile-time spatial-grid configuration.
//!
//! Single source of truth for the per-fixture uniform-grid hash — the
//! constants here are emitted into every kernel that touches the
//! `spatial_grid_*` bindings AND consulted by per-fixture runtime
//! crates when allocating the matching wgpu buffers. Mismatches
//! between the two are silent corruption (writes to the wrong cell,
//! out-of-bounds slot reads), so both surfaces route through the
//! same constants.
//!
//! ## Sizing knobs
//!
//! - [`CELL_SIZE`] — edge length of one grid cell in world units. Set
//!   to the largest fold radius any fixture queries (today: boids'
//!   `perception_radius = 6.0`). Larger cells mean fewer cells to
//!   walk per query (always a 3³=27 neighborhood for a single-cell-
//!   radius query) at the cost of more candidates per cell.
//! - [`WORLD_HALF_EXTENT`] — half-edge of the cubic world along each
//!   axis. Coordinates outside `[-extent, +extent]` clamp to the
//!   edge cell. Set to 64.0 for boids — comfortably exceeds the ±8
//!   initial spread so the flock has room to drift.
//! - [`MAX_PER_CELL`] — cap on how many agents any single cell can
//!   hold. Bounded-list counting sort: each agent does
//!   `slot = atomicAdd(&offsets[cell], 1)`; if `slot >= MAX_PER_CELL`
//!   the agent is silently dropped from queries that tick. 32 is
//!   conservative for boids at thousand-scale; cluster-heavy
//!   fixtures or higher densities may need more.
//!
//! ## Derived
//!
//! [`grid_dim`] is `(2 * WORLD_HALF_EXTENT / CELL_SIZE).ceil() as u32`
//! per axis. [`num_cells`] is `grid_dim ^ 3`.

/// Cell edge length, world units.
pub const CELL_SIZE: f32 = 6.0;

/// Half-extent of the cubic world along each axis, world units.
pub const WORLD_HALF_EXTENT: f32 = 64.0;

/// Max agents per cell. Overflows are silently dropped from spatial
/// queries on the same tick.
///
/// Workgroup-shared memory budget at MAX_PER_CELL=32 (the tiled
/// MoveBoid kernel's tile arrays):
///   - tile_pos: 27 × 32 × 16 B = 14 KB
///   - tile_vel: 27 × 32 × 16 B = 14 KB
///   - tile_count: 108 B
///   = ~28 KB total, comfortably under the 32 KB lower bound on
///   workgroup-memory across adapters.
///
/// **Density caveat**: MAX_PER_CELL is also the workgroup size for
/// PerCell-shaped kernels (see `PER_CELL_WORKGROUP_X`). Bumping it
/// to absorb higher density (e.g. 64 to fit 1M agents in 22³ cells
/// without truncation) doubles the workgroup size — most lanes go
/// idle at typical density and the cooperative-load preamble pays
/// for them. Decoupling these two knobs is a future refactor; for
/// now MAX_PER_CELL=32 is the right balance for the boids fixture
/// at ≤100k agents (~10/cell average — no overflow). Higher counts
/// need either smaller cells or per-lane multi-iteration.
pub const MAX_PER_CELL: u32 = 32;

/// Cells per axis: `ceil(2 * WORLD_HALF_EXTENT / CELL_SIZE)`. For the
/// defaults above (128 / 6) ≈ 22, so we use 22.
pub const fn grid_dim() -> u32 {
    let extent = 2.0 * WORLD_HALF_EXTENT;
    let raw = extent / CELL_SIZE;
    // const-fn `ceil` workaround: integer division + 1 if there's a
    // remainder. The cast loses precision but the boids defaults pin
    // a clean integer answer anyway.
    let truncated = raw as u32;
    if (truncated as f32) * CELL_SIZE < extent {
        truncated + 1
    } else {
        truncated
    }
}

/// Total cell count: `grid_dim ^ 3`.
pub const fn num_cells() -> u32 {
    let d = grid_dim();
    d * d * d
}

/// Byte size of the `spatial_grid_offsets` buffer
/// (`num_cells * 4` for `array<atomic<u32>>`).
pub const fn offsets_bytes() -> u64 {
    (num_cells() as u64) * 4
}

/// Byte size of the `spatial_grid_cells` buffer
/// (`num_cells * MAX_PER_CELL * 4` for `array<u32>`).
pub const fn cells_bytes() -> u64 {
    (num_cells() as u64) * (MAX_PER_CELL as u64) * 4
}

/// Render the WGSL constants + helper functions every kernel with
/// spatial bindings needs. Returns an empty string when `body` does
/// not reference any spatial binding (the substring scan keys on
/// `spatial_grid_cells` / `spatial_grid_offsets`, the only two
/// identifiers that appear when a spatial binding is in use).
pub fn compose_spatial_prelude(body: &str) -> String {
    let touches_spatial =
        body.contains("spatial_grid_cells") || body.contains("spatial_grid_offsets");
    if !touches_spatial {
        return String::new();
    }
    let dim = grid_dim();
    let mut out = String::new();
    out.push_str(&format!(
        "// --- Spatial grid configuration (must match boids_runtime allocation) ---\n\
         const SPATIAL_CELL_SIZE: f32 = {cell_size:?};\n\
         const SPATIAL_WORLD_HALF_EXTENT: f32 = {extent:?};\n\
         const SPATIAL_GRID_DIM: u32 = {dim}u;\n\
         const SPATIAL_MAX_PER_CELL: u32 = {max_per_cell}u;\n\
         \n\
         /// Map a world-space position to a flat cell index in [0, GRID_DIM^3).\n\
         /// Coordinates outside the world extent clamp to the boundary cell\n\
         /// (silent clamp; no overflow path).\n\
         fn pos_to_cell(p: vec3<f32>) -> u32 {{\n\
         \x20   let shifted = p + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT);\n\
         \x20   let cell_xyz_f = shifted / SPATIAL_CELL_SIZE;\n\
         \x20   let max_idx = SPATIAL_GRID_DIM - 1u;\n\
         \x20   let cx = clamp(u32(max(cell_xyz_f.x, 0.0)), 0u, max_idx);\n\
         \x20   let cy = clamp(u32(max(cell_xyz_f.y, 0.0)), 0u, max_idx);\n\
         \x20   let cz = clamp(u32(max(cell_xyz_f.z, 0.0)), 0u, max_idx);\n\
         \x20   return (cz * SPATIAL_GRID_DIM + cy) * SPATIAL_GRID_DIM + cx;\n\
         }}\n\
         \n\
         /// Recombine a (cx, cy, cz) triple into a flat cell index. Out-of-\n\
         /// range components wrap via clamp; callers using neighbor offsets\n\
         /// pass values in [-1, GRID_DIM] and rely on the clamp at the world\n\
         /// boundary.\n\
         fn cell_index(cx: i32, cy: i32, cz: i32) -> u32 {{\n\
         \x20   let max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20   let xx = u32(clamp(cx, 0, max_idx));\n\
         \x20   let yy = u32(clamp(cy, 0, max_idx));\n\
         \x20   let zz = u32(clamp(cz, 0, max_idx));\n\
         \x20   return (zz * SPATIAL_GRID_DIM + yy) * SPATIAL_GRID_DIM + xx;\n\
         }}\n\n",
        cell_size = CELL_SIZE,
        extent = WORLD_HALF_EXTENT,
        dim = dim,
        max_per_cell = MAX_PER_CELL,
    ));
    out
}
