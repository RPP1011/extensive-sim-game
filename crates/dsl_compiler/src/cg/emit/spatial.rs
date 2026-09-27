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
//! ## Grid layout
//!
//! The grid is rebuilt every tick by a three-phase counting sort
//! (`BuildHashCount` → `BuildHashScan*` → `BuildHashScatter`, see
//! `cg/emit/kernel.rs`). Cell `c` owns the slice
//! `spatial_grid_cells[spatial_grid_starts[c] .. spatial_grid_starts[c+1]]`,
//! with agent ids in ascending order inside the slice. The layout is
//! **uncapped**: a cell holds every agent that maps to it, however
//! many. `spatial_grid_offsets` is scratch for the build only (per-cell
//! count, then write cursor) and is left all-zero when the build ends,
//! so the next tick's count starts from zero without any host clear.
//!
//! ## Walk semantics (the contract every `spatial.<query>(self)` walk follows)
//!
//! - **Declared radius.** A `spatial_query` / `query` carrying
//!   `@spatial(radius = R)` walks every cell overlapping the axis-aligned
//!   box `origin ± R` (computed per thread, so `R` may exceed
//!   [`CELL_SIZE`] and may be a `@runtime` config value) and keeps
//!   exactly the candidates with `|candidate - origin|² <= R²`. An
//!   un-annotated query keeps the historical one-ring (3³ = 27 cell)
//!   walk with its loose cell-diagonal gate.
//! - **Each cell at most once.** Out-of-grid neighbour coordinates are
//!   skipped (`cell_in_grid`), never clamped onto the edge cell — a
//!   clamp would make a walk at the edge visit the same edge cell (and
//!   every agent in it) two, four or eight times.
//! - **Outside the world extent.** Positions beyond
//!   ±[`WORLD_HALF_EXTENT`] still clamp into the edge cells (the grid
//!   has no overflow bucket). Walks stay correct — the radius-derived
//!   box is clamped the same way, so the edge cell holding them is
//!   visited once and the per-candidate distance gate rejects the far
//!   ones — but every out-of-world agent in the same edge column/row
//!   is a candidate of every walk that touches that edge cell, so worlds
//!   much larger than the extent pay for it in walk length.
//!
//! ## Sizing knobs
//!
//! - [`CELL_SIZE`] — edge length of one grid cell in world units.
//!   Matches boids' `perception_radius = 6.0` (the un-annotated
//!   one-ring walk covers exactly one cell of reach). Radius-annotated
//!   queries walk `~(2R / CELL_SIZE + 1)` cells per axis instead.
//! - [`WORLD_HALF_EXTENT`] — half-edge of the cubic world along each
//!   axis. See "Outside the world extent" above.
//! - [`MAX_PER_CELL`] — the tiled PerCell kernel's per-neighbour-cell
//!   workgroup tile budget (and its workgroup size). It does not cap
//!   what a query sees: cells denser than this are read past the tile
//!   from global memory.
//!
//! ## Derived
//!
//! [`grid_dim`] is `(2 * WORLD_HALF_EXTENT / CELL_SIZE).ceil() as u32`
//! per axis. [`num_cells`] is `grid_dim ^ 3`.

/// Cell edge length, world units.
pub const CELL_SIZE: f32 = 6.0;

/// Half-extent of the cubic world along each axis, world units.
pub const WORLD_HALF_EXTENT: f32 = 64.0;

/// Per-neighbour-cell tile budget of the tiled PerCell kernel (and its
/// workgroup size, `PER_CELL_WORKGROUP_X`).
///
/// The counting-sort grid itself is uncapped (see the module docs); this
/// constant only bounds how many agents per neighbour cell the tiled
/// kernel caches in workgroup memory. Candidates past the tile are read
/// from global memory, so no agent is dropped from a query.
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
/// PerCell-shaped kernels (see `PER_CELL_WORKGROUP_X`); the home-cell
/// loop strides by it, so dense home cells are processed in several
/// passes and dense neighbour cells spill to global reads — correct at
/// any density, fastest at <= MAX_PER_CELL per cell.
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
         // Block scratch for the single-workgroup stable scatter (unused by\n\
         // the other spatial kernels).\n\
         var<workgroup> _scatter_blk_cell: array<u32, 256>;\n\
         const SPATIAL_CELL_SIZE: f32 = {cell_size:?};\n\
         const SPATIAL_WORLD_HALF_EXTENT: f32 = {extent:?};\n\
         const SPATIAL_GRID_DIM: u32 = {dim}u;\n\
         const SPATIAL_MAX_PER_CELL: u32 = {max_per_cell}u;\n\
         \n\
         /// Map a world-space position to a flat cell index in [0, GRID_DIM^3).\n\
         /// Coordinates outside the world extent clamp to the boundary cell\n\
         /// (there is no overflow bucket; walks re-check distance).\n\
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
         /// Per-axis cell coordinates of a world position, clamped exactly\n\
         /// like `pos_to_cell` (so an out-of-world position maps to the edge\n\
         /// cell its agent was binned into). Radius walks bound their cell\n\
         /// box with `pos_to_cell_xyz(origin -/+ R)`.\n\
         fn pos_to_cell_xyz(p: vec3<f32>) -> vec3<i32> {{\n\
         \x20   let f = (p + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20   let m = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20   return vec3<i32>(\n\
         \x20       clamp(i32(max(f.x, 0.0)), 0, m),\n\
         \x20       clamp(i32(max(f.y, 0.0)), 0, m),\n\
         \x20       clamp(i32(max(f.z, 0.0)), 0, m),\n\
         \x20   );\n\
         }}\n\
         \n\
         /// True iff (cx, cy, cz) is a real grid cell. Neighbour walks skip\n\
         /// out-of-grid offsets with this instead of clamping them: a clamp\n\
         /// would revisit the edge cell and count its agents repeatedly.\n\
         fn cell_in_grid(cx: i32, cy: i32, cz: i32) -> bool {{\n\
         \x20   let d = i32(SPATIAL_GRID_DIM);\n\
         \x20   return cx >= 0 && cx < d && cy >= 0 && cy < d && cz >= 0 && cz < d;\n\
         }}\n\
         \n\
         /// Recombine a (cx, cy, cz) triple into a flat cell index. Callers\n\
         /// pass in-grid coordinates (neighbour walks gate on `cell_in_grid`\n\
         /// first); the clamp only keeps a stray call in bounds.\n\
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
