//! Voxel-engine adapter — bridges `~/Projects/voxel_engine` to the sim
//! engine via the `TerrainQuery` seam. **Phases A + B** of the 5-phase
//! voxel integration plan
//! (`docs/superpowers/plans/2026-05-09-voxel-engine-integration.md`).
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
//! the Phase B consumer stores a deterministic non-zero material code
//! derived from `kind_hash` (low 8 bits or'd with 1 to avoid colliding
//! with the air sentinel), and the per-kind registry mapping is a
//! Phase D concern.
//!
//! # Phase B chronicle consumer (this slice)
//!
//! `apply_voxel_chronicle_record` decodes engine chronicle records
//! produced by the GPU `apply_ability` dispatcher when a verb's program
//! includes `EffectOp::PlaceVoxel` (kind=26 → engine kind=60) or
//! `EffectOp::Harvest` (kind=25 → engine kind=59). The runtime drives
//! the call once per drained record:
//!
//! - **`EffectPlaceVoxelApplied` (kind=60)** — `slot[2]=caster_slot`,
//!   `slot[3]=kind_hash`. The consumer writes a non-zero voxel value
//!   (`((kind_hash & 0xFF) as u8) | 1`) at the cell directly above the
//!   caster's position so the place lands on the column floor (matches
//!   `floor()` cell semantics — the caster stands on `z=0`, the placed
//!   block goes at `z=0` and lifts `height_at` to `1.0`).
//! - **`EffectHarvestApplied` (kind=59)** — `slot[2]=caster_slot`,
//!   `slot[3]=kind_hash`, `slot[4]=amount`. The consumer scans cells
//!   in a small neighborhood around the caster and clears up to
//!   `amount` non-zero cells (deterministic top-down x/y/z iteration).
//!
//! ## Caller signature: option (A) — pass caster_pos in
//!
//! The plan considered two consumer signatures: (A) caller resolves
//! `caster_slot → pos` and passes a `Vec3`, (B) caller passes the full
//! `SimState`. We picked (A) — `engine_voxel` stays free of any
//! `engine::SimState` dep and the per-call coupling is minimal (one
//! `Vec3` per drained record). The runtime owns the agent SoA; it
//! looks up `caster_slot`'s position before calling.
//!
//! # What this crate does NOT do
//!
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
    /// `rec` is one 10-word slice off the engine event ring (see
    /// `engine::gpu::EVENT_STRIDE_U32`). `caster_pos` is the position
    /// of `slot[2]`'s agent at consume tick — the runtime resolves
    /// `caster_slot → pos` against its agent SoA and passes the
    /// world-space `Vec3` in. See option (A) discussion in the crate
    /// doc-comment.
    ///
    /// Recognised event kinds (header word at `rec[0]`):
    ///
    /// - **60 = `EffectPlaceVoxelApplied`** — writes a non-zero voxel
    ///   at `(floor(caster_pos.x), floor(caster_pos.y),
    ///   floor(caster_pos.z))`. The material code is
    ///   `((kind_hash & 0xFF) as u8) | 1` so distinct kinds map to
    ///   distinct (deterministic) byte values; the `| 1` ensures we
    ///   never collide with the `0 = air` sentinel.
    /// - **59 = `EffectHarvestApplied`** — clears up to `amount`
    ///   non-zero voxels of the matching kind around the caster. The
    ///   scan walks a 3-cell-radius cube (small enough to be cheap,
    ///   large enough to catch the place + harvest pin's adjacent
    ///   cell) in deterministic (z, y, x) ascending order so the
    ///   removal sequence is byte-stable for the same input.
    ///
    /// Returns:
    /// - `Some(())` if the record was a voxel-mutating event and was
    ///   applied (or attempted — out-of-bounds writes drop silently
    ///   per `VoxelGrid::set` semantics).
    /// - `None` if the record was not one of the recognised kinds, or
    ///   if `rec` is shorter than the expected 10 words.
    pub fn apply_voxel_chronicle_record(
        &mut self,
        rec: &[u32],
        caster_pos: Vec3,
    ) -> Option<()> {
        // The engine ring stride is 10 u32; defend against a caller
        // that passes a clipped slice.
        if rec.len() < 5 {
            return None;
        }
        let kind_tag = rec[0];
        match kind_tag {
            60 => {
                // EffectPlaceVoxelApplied: rec[2]=caster_slot,
                // rec[3]=kind_hash. caster_slot is informational here —
                // the runtime already used it to resolve `caster_pos`.
                let kind_hash = rec[3];
                // `| 1` so we never accidentally write the air
                // sentinel for kinds whose low byte hashes to 0.
                let value: u8 = ((kind_hash & 0xFF) as u8) | 1;
                let cx = caster_pos.x.floor() as i32;
                let cy = caster_pos.y.floor() as i32;
                let cz = caster_pos.z.floor() as i32;
                if cx >= 0
                    && cy >= 0
                    && cz >= 0
                    && (cx as u32) < self.extent
                    && (cy as u32) < self.extent
                    && (cz as u32) < self.extent
                {
                    self.grid.set(cx as u32, cy as u32, cz as u32, value);
                }
                Some(())
            }
            59 => {
                // EffectHarvestApplied: rec[2]=caster_slot,
                // rec[3]=kind_hash, rec[4]=amount.
                let kind_hash = rec[3];
                let target_value: u8 = ((kind_hash & 0xFF) as u8) | 1;
                let mut amount_remaining = rec[4];
                if amount_remaining == 0 {
                    // Treat zero amount as a no-op (matches
                    // `apply_summon_event_to_state`'s zero-count
                    // tolerance).
                    return Some(());
                }
                // Bounding cube around caster: 3-cell radius. The
                // Phase B fixture places at the caster's own cell and
                // harvests from there too, so a tight neighborhood is
                // enough; a larger world-sim opt-in (Phase E) can
                // pass an explicit radius via the EffectOp::Harvest
                // payload once that's wired.
                const HARVEST_RADIUS: i32 = 3;
                let cx = caster_pos.x.floor() as i32;
                let cy = caster_pos.y.floor() as i32;
                let cz = caster_pos.z.floor() as i32;
                // Deterministic ascending z/y/x walk so the removal
                // order is byte-stable per (caster_pos, amount) input.
                'outer: for dz in -HARVEST_RADIUS..=HARVEST_RADIUS {
                    let z = cz + dz;
                    if z < 0 || (z as u32) >= self.extent {
                        continue;
                    }
                    for dy in -HARVEST_RADIUS..=HARVEST_RADIUS {
                        let y = cy + dy;
                        if y < 0 || (y as u32) >= self.extent {
                            continue;
                        }
                        for dx in -HARVEST_RADIUS..=HARVEST_RADIUS {
                            let x = cx + dx;
                            if x < 0 || (x as u32) >= self.extent {
                                continue;
                            }
                            let cur = self.grid.get(x as u32, y as u32, z as u32).unwrap_or(0);
                            if cur != 0 && cur == target_value {
                                self.grid.set(x as u32, y as u32, z as u32, 0);
                                amount_remaining -= 1;
                                if amount_remaining == 0 {
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
                Some(())
            }
            _ => None,
        }
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
    fn apply_voxel_chronicle_record_unknown_kind_returns_none() {
        let mut t = VoxelTerrain::new();
        // Header kind=42 (some non-voxel event) → no-op, returns None.
        let mut rec = [0u32; 10];
        rec[0] = 42;
        rec[2] = 0;
        rec[3] = 0xCAFEBABE;
        assert_eq!(t.apply_voxel_chronicle_record(&rec, Vec3::ZERO), None);
    }

    #[test]
    fn apply_voxel_chronicle_record_short_slice_returns_none() {
        let mut t = VoxelTerrain::new();
        let rec = [60u32, 0, 0]; // shorter than 5 words
        assert_eq!(t.apply_voxel_chronicle_record(&rec, Vec3::ZERO), None);
    }

    #[test]
    fn place_voxel_record_writes_cell_at_caster_pos() {
        let mut t = VoxelTerrain::with_extent(16);
        let mut rec = [0u32; 10];
        rec[0] = 60; // EffectPlaceVoxelApplied
        rec[2] = 0; // caster_slot
        rec[3] = 0x12345678; // kind_hash
        let pos = Vec3::new(5.5, 7.25, 0.0);
        assert_eq!(t.apply_voxel_chronicle_record(&rec, pos), Some(()));
        // Floor → cell (5, 7, 0). Material code = (0x78 | 1) = 0x79.
        assert_eq!(t.cell_at(5, 7, 0), 0x79);
        // Adjacent cells stay empty.
        assert_eq!(t.cell_at(6, 7, 0), 0);
        assert_eq!(t.cell_at(5, 8, 0), 0);
    }

    #[test]
    fn place_voxel_lifts_height_at() {
        let mut t = VoxelTerrain::with_extent(16);
        let mut rec = [0u32; 10];
        rec[0] = 60;
        rec[3] = 1;
        // Drive the consumer at world origin (caster on z=0 cell).
        assert_eq!(t.apply_voxel_chronicle_record(&rec, Vec3::ZERO), Some(()));
        // Top face of cell (0,0,0) = z+1 = 1.0.
        assert!((t.height_at(0.5, 0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn harvest_record_clears_matching_voxel() {
        let mut t = VoxelTerrain::with_extent(16);
        // Place via the consumer so the material code byte matches
        // what the harvest path looks for.
        let mut place = [0u32; 10];
        place[0] = 60;
        place[3] = 7; // kind_hash → value 0x07 | 1 = 7
        let pos = Vec3::new(2.5, 2.5, 0.0);
        assert_eq!(t.apply_voxel_chronicle_record(&place, pos), Some(()));
        assert_eq!(t.cell_at(2, 2, 0), 7);

        // Now harvest matching kind, amount=1.
        let mut harv = [0u32; 10];
        harv[0] = 59;
        harv[3] = 7;
        harv[4] = 1;
        assert_eq!(t.apply_voxel_chronicle_record(&harv, pos), Some(()));
        assert_eq!(t.cell_at(2, 2, 0), 0);
    }

    #[test]
    fn harvest_record_does_not_clear_mismatched_kind() {
        let mut t = VoxelTerrain::with_extent(16);
        // Place kind_hash=7 (material 7).
        let mut place = [0u32; 10];
        place[0] = 60;
        place[3] = 7;
        let pos = Vec3::new(2.5, 2.5, 0.0);
        t.apply_voxel_chronicle_record(&place, pos);
        // Harvest a DIFFERENT kind (low-byte 9 → material 9). Same cell
        // is not material 9, so it stays.
        let mut harv = [0u32; 10];
        harv[0] = 59;
        harv[3] = 9;
        harv[4] = 5;
        assert_eq!(t.apply_voxel_chronicle_record(&harv, pos), Some(()));
        assert_eq!(t.cell_at(2, 2, 0), 7);
    }

    #[test]
    fn harvest_record_amount_bounds_removals() {
        let mut t = VoxelTerrain::with_extent(16);
        // Manually set 5 cells of the same material in the harvest
        // radius (use kind_hash=3 → material 3).
        let target_value: u8 = (3u32 & 0xFF) as u8 | 1;
        for i in 0..5 {
            t.set_cell(2, 2 + i, 0, target_value);
        }
        // Harvest with amount=2 → only 2 cleared.
        let mut harv = [0u32; 10];
        harv[0] = 59;
        harv[3] = 3;
        harv[4] = 2;
        let pos = Vec3::new(2.5, 2.5, 0.0);
        t.apply_voxel_chronicle_record(&harv, pos);
        let cleared: u32 = (0..5)
            .map(|i| if t.cell_at(2, 2 + i, 0) == 0 { 1 } else { 0 })
            .sum();
        assert_eq!(cleared, 2, "exactly `amount` cells should be cleared");
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
