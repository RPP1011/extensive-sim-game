//! Voxel-engine adapter — bridges `~/Projects/voxel_engine` to the sim
//! engine via the `TerrainQuery` seam. **Phase A** of the 5-phase voxel
//! integration plan (`docs/superpowers/plans/2026-05-09-voxel-engine-integration.md`).
//!
//! # What this crate does today
//!
//! Wraps `voxel_engine::voxel::grid::VoxelGrid` — a dense, flat `Vec<u8>`
//! backed 3D array — and implements the engine's `TerrainQuery` trait.
//! Cells are 1 world-unit on a side; the cell at integer coordinate
//! `(x, y, z)` covers the AABB `[x, x+1) × [y, y+1) × [z, z+1)`. World
//! origin sits at grid origin; negative coordinates fall outside the
//! grid and behave as empty (return defaults: height 0, walkable true,
//! LOS clear — same as the engine's `FlatPlane` default impl).
//!
//! Voxel value `0` = air (empty); any non-zero value = solid. The
//! per-kind material distinction (palisade vs. quarried-stone vs.
//! placed-block) lives in the upper bits but is not interpreted here —
//! Phase B fills in the chronicle consumer that maps `place_voxel
//! <kind_hash>` to a non-zero material code.
//!
//! # What this crate does NOT do
//!
//! - **No chronicle event consumer wiring.** `apply_voxel_chronicle_record`
//!   is a `None`-returning stub. Phase B fills it in.
//! - **No GPU mirror.** Phase C ships a `wgpu::Buffer` mirror of the
//!   grid that kernels can read directly.
//! - **No production runtime opt-in.** Existing runtimes continue to use
//!   the `FlatPlane` default; Phase E migrates wave_defense to opt in.
//!
//! # Determinism audit (load-bearing per P5)
//!
//! `voxel_engine` uses `std::collections::HashMap`/`HashSet` in three
//! files — `voxel/svdag.rs`, `voxel/articulation.rs`,
//! `voxel/connectivity.rs` — but **none** of them are reachable from
//! the modules we use (`voxel::grid` + `voxel::raycast`). The
//! `VoxelGrid` struct is internally a `Vec<u8>` indexed by
//! `z*H*W + y*W + x`, so iteration order is deterministic by
//! construction. No HashMap-derived nondeterminism enters the adapter
//! boundary.
//!
//! Verified by `grep -rn HashMap /home/ricky/Projects/voxel_engine/src/world/`
//! (zero hits) and `grep -n` on `voxel/grid.rs` + `voxel/raycast.rs`
//! (zero hits). Re-audit when bumping the `voxel_engine` path-dep.
//!
//! # Vulkan deps cost
//!
//! `voxel_engine`'s `Cargo.toml` lists `ash`, `gpu-allocator`,
//! `ash-window`, `raw-window-handle`, `dot_vox`, plus a `shaderc`
//! build-dep — all unconditional. The `default = []` feature set keeps
//! `winit`/`egui`/`tracing` out of the dep tree, but Vulkan still
//! compiles. Sim engine crates that don't import `engine_voxel` aren't
//! affected.

use engine::terrain::TerrainQuery;
use glam::Vec3;
use voxel_engine::voxel::grid::VoxelGrid;
use voxel_engine::voxel::raycast::ray_cast_grid;

pub use engine::state::agent::MovementMode;

/// Default grid extent in voxel cells. 256³ cells (= 256 world-unit
/// cube) is enough for the Phase B `voxel_probe` fixture and the
/// Phase E `wave_defense` opt-in. Larger worlds will need a chunked
/// representation; that's a Phase C concern.
pub const DEFAULT_EXTENT: u32 = 256;

/// CPU-side voxel terrain backend. Implements the engine's
/// `TerrainQuery` trait by consulting a wrapped
/// `voxel_engine::voxel::grid::VoxelGrid`.
///
/// Cells are 1 world-unit on a side; the cell at integer `(x, y, z)`
/// covers `[x, x+1) × [y, y+1) × [z, z+1)`. Negative coordinates and
/// coordinates beyond the grid extent fall outside the populated region
/// and behave as empty (height 0, walkable true, LOS clear).
///
/// `apply_voxel_chronicle_record` takes `&mut self`; consumers call it
/// from a chronicle drainer that owns the terrain or holds a unique
/// reference. Phase B will decide whether to add interior mutability
/// (e.g. an `RwLock`) once the actual drain shape is wired up.
pub struct VoxelTerrain {
    grid: VoxelGrid,
    extent: u32,
}

impl VoxelTerrain {
    /// Construct an empty voxel world at the default extent.
    ///
    /// All cells start as air (value 0). Subsequent chronicle drains
    /// (Phase B) populate the grid as agents fire `place_voxel` and
    /// `harvest` abilities.
    pub fn new() -> Self {
        Self::with_extent(DEFAULT_EXTENT)
    }

    /// Construct an empty voxel world at a caller-chosen cubic extent.
    pub fn with_extent(extent: u32) -> Self {
        Self {
            grid: VoxelGrid::new(extent, extent, extent),
            extent,
        }
    }

    /// Cubic extent (in cells) of the underlying grid. Cells outside
    /// `[0, extent)` on any axis are treated as empty by the trait
    /// methods.
    pub fn extent(&self) -> u32 {
        self.extent
    }

    /// Read a single cell's material id. Returns `0` for out-of-bounds
    /// coordinates (same as for genuinely empty cells). Phase A only
    /// uses this in tests; Phase B will use it inside the chronicle
    /// consumer.
    pub fn cell_at(&self, x: i32, y: i32, z: i32) -> u8 {
        if x < 0 || y < 0 || z < 0 {
            return 0;
        }
        self.grid.get(x as u32, y as u32, z as u32).unwrap_or(0)
    }

    /// Internal helper for tests / Phase B chronicle consumer:
    /// directly write a cell's material id. Out-of-bounds writes are
    /// silently dropped (matches `VoxelGrid::set` semantics).
    #[doc(hidden)]
    pub fn set_cell(&mut self, x: u32, y: u32, z: u32, value: u8) {
        self.grid.set(x, y, z, value);
    }

    /// Drain a single chronicle record into the voxel world.
    ///
    /// **Phase A stub** — always returns `None`. Phase B implements
    /// the actual decode for `kind = EffectPlaceVoxelApplied (60)` and
    /// `kind = EffectHarvestApplied (59)`.
    ///
    /// Returns:
    /// - `Some(())` once Phase B lands and the record was successfully
    ///   applied.
    /// - `None` if the record was not a voxel-mutating event (Phase B+),
    ///   or — today — for every record (Phase A stub).
    pub fn apply_voxel_chronicle_record(&mut self, _rec: &[u32]) -> Option<()> {
        None
    }
}

impl Default for VoxelTerrain {
    fn default() -> Self {
        Self::new()
    }
}

impl TerrainQuery for VoxelTerrain {
    /// World-space ground height at horizontal `(x, y)`. Walks the
    /// column at integer `(floor(x), floor(y))` from top down and
    /// returns the highest occupied cell's `z + 1` (top face) as f32.
    /// Empty column or out-of-bounds returns `0.0`.
    fn height_at(&self, x: f32, y: f32) -> f32 {
        let cx = x.floor() as i32;
        let cy = y.floor() as i32;
        if cx < 0 || cy < 0 || (cx as u32) >= self.extent || (cy as u32) >= self.extent {
            return 0.0;
        }
        // Walk top-down so we return the highest occupied cell first.
        for cz in (0..self.extent as i32).rev() {
            if self.grid.get(cx as u32, cy as u32, cz as u32).unwrap_or(0) != 0 {
                return (cz + 1) as f32;
            }
        }
        0.0
    }

    /// `Walk` / `Climb` / `Swim` modes can occupy a cell only if it's
    /// air. `Fly` and `Fall` always pass (matches the constitutional
    /// stance that flying entities are not bound by ground terrain).
    /// Out-of-bounds cells default to walkable (matches FlatPlane).
    fn walkable(&self, pos: Vec3, mode: MovementMode) -> bool {
        if matches!(mode, MovementMode::Fly | MovementMode::Fall) {
            return true;
        }
        let cx = pos.x.floor() as i32;
        let cy = pos.y.floor() as i32;
        let cz = pos.z.floor() as i32;
        if cx < 0
            || cy < 0
            || cz < 0
            || (cx as u32) >= self.extent
            || (cy as u32) >= self.extent
            || (cz as u32) >= self.extent
        {
            return true;
        }
        self.grid.get(cx as u32, cy as u32, cz as u32).unwrap_or(0) == 0
    }

    /// Whether the straight-line segment from `from` to `to` is
    /// unobstructed. Uses voxel_engine's Amanatides-Woo DDA raymarcher
    /// (`voxel::raycast::ray_cast_grid`). Returns `true` (clear) if
    /// the ray exits the grid AABB without hitting a solid cell, or if
    /// the endpoints coincide.
    fn line_of_sight(&self, from: Vec3, to: Vec3) -> bool {
        let delta = to - from;
        let length = delta.length();
        if length < 1e-6 {
            return true;
        }
        let dir = delta / length;
        let hit = ray_cast_grid(
            &self.grid,
            [from.x, from.y, from.z],
            [dir.x, dir.y, dir.z],
        );
        match hit {
            // Ray exits without hitting anything → unobstructed.
            None => true,
            // Hit found, but only counts as a blocker if it's between
            // `from` and `to`. `t` is in voxel-grid units, which are
            // identical to world units here (1 cell = 1 unit), so
            // compare directly to `length`.
            Some(h) => h.t > length,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_world_height_is_zero() {
        let t = VoxelTerrain::new();
        assert_eq!(t.height_at(0.0, 0.0), 0.0);
        assert_eq!(t.height_at(50.0, 50.0), 0.0);
        assert_eq!(t.height_at(-10.0, -10.0), 0.0);
    }

    #[test]
    fn empty_world_is_walkable_everywhere() {
        let t = VoxelTerrain::new();
        assert!(t.walkable(Vec3::new(1.0, 2.0, 3.0), MovementMode::Walk));
        assert!(t.walkable(Vec3::new(-99.0, 0.0, 0.0), MovementMode::Walk));
        assert!(t.walkable(Vec3::new(50.0, 50.0, 50.0), MovementMode::Fly));
        assert!(t.walkable(Vec3::new(0.0, 0.0, -10.0), MovementMode::Swim));
    }

    #[test]
    fn empty_world_has_clear_line_of_sight() {
        let t = VoxelTerrain::new();
        assert!(t.line_of_sight(Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)));
        assert!(t.line_of_sight(
            Vec3::new(-5.0, -5.0, -5.0),
            Vec3::new(5.0, 5.0, 5.0)
        ));
    }

    #[test]
    fn coincident_endpoints_have_clear_line_of_sight() {
        let t = VoxelTerrain::new();
        let p = Vec3::new(7.0, 7.0, 7.0);
        assert!(t.line_of_sight(p, p));
    }

    #[test]
    fn out_of_bounds_is_walkable() {
        let t = VoxelTerrain::with_extent(16);
        // Far outside the 16³ grid.
        assert!(t.walkable(Vec3::new(1000.0, 1000.0, 1000.0), MovementMode::Walk));
        assert!(t.walkable(Vec3::new(-50.0, -50.0, -50.0), MovementMode::Walk));
    }

    #[test]
    fn fly_ignores_solid_cells() {
        let mut t = VoxelTerrain::with_extent(16);
        t.set_cell(5, 5, 5, 1);
        // Walk into solid → blocked.
        assert!(!t.walkable(Vec3::new(5.5, 5.5, 5.5), MovementMode::Walk));
        // Fly through solid → still passes (per the doc-comment contract).
        assert!(t.walkable(Vec3::new(5.5, 5.5, 5.5), MovementMode::Fly));
    }

    #[test]
    fn solid_cell_increases_height_at() {
        let mut t = VoxelTerrain::with_extent(16);
        t.set_cell(5, 5, 5, 1);
        // Top face of cell (5, 5, 5) sits at z = 6.
        assert_eq!(t.height_at(5.5, 5.5), 6.0);
        // Adjacent column still empty.
        assert_eq!(t.height_at(6.5, 5.5), 0.0);
    }

    #[test]
    fn solid_cell_blocks_line_of_sight() {
        let mut t = VoxelTerrain::with_extent(16);
        t.set_cell(5, 5, 5, 1);
        // Ray from (0, 5.5, 5.5) → (10, 5.5, 5.5) crosses cell (5, 5, 5).
        assert!(!t.line_of_sight(
            Vec3::new(0.0, 5.5, 5.5),
            Vec3::new(10.0, 5.5, 5.5)
        ));
        // Parallel ray two cells away is clear.
        assert!(t.line_of_sight(
            Vec3::new(0.0, 7.5, 5.5),
            Vec3::new(10.0, 7.5, 5.5)
        ));
    }

    #[test]
    fn ray_stopping_short_of_obstacle_is_clear() {
        let mut t = VoxelTerrain::with_extent(16);
        t.set_cell(5, 5, 5, 1);
        // Segment from (0, 5.5, 5.5) → (3, 5.5, 5.5) ends before the
        // obstacle at x∈[5,6); should be unobstructed.
        assert!(t.line_of_sight(
            Vec3::new(0.0, 5.5, 5.5),
            Vec3::new(3.0, 5.5, 5.5)
        ));
    }

    #[test]
    fn cell_at_returns_zero_for_negative_coords() {
        let t = VoxelTerrain::new();
        assert_eq!(t.cell_at(-1, 0, 0), 0);
        assert_eq!(t.cell_at(0, -1, 0), 0);
        assert_eq!(t.cell_at(0, 0, -1), 0);
    }

    #[test]
    fn apply_voxel_chronicle_record_is_phase_a_stub() {
        let mut t = VoxelTerrain::new();
        // Phase A stub: returns None for every input. Phase B fills in
        // the actual decode + mutate.
        let dummy_rec = [0u32; 8];
        assert_eq!(t.apply_voxel_chronicle_record(&dummy_rec), None);
    }

    #[test]
    fn implements_terrain_query_via_arc_dyn() {
        // Confirm the adapter is usable through the same Arc<dyn> seam
        // SimState already exposes — i.e. callers can drop in
        // VoxelTerrain without further plumbing.
        use std::sync::Arc;
        let t: Arc<dyn TerrainQuery + Send + Sync> = Arc::new(VoxelTerrain::new());
        assert_eq!(t.height_at(0.0, 0.0), 0.0);
        assert!(t.walkable(Vec3::ZERO, MovementMode::Walk));
        assert!(t.line_of_sight(Vec3::ZERO, Vec3::ONE));
    }
}
