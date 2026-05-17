//! Navgrid index per spec §7.2 (built-in index kind v1).
//!
//! A `NavgridIndex` is a per-region 2D walkability + cell-height
//! table covering the (x, z) extent of a [`VoxelRegion`]'s AABB.
//! Each cell stores a u32 packed as:
//!
//!   bits  0..7  — walkable flag (0 = blocked, 1 = walkable)
//!   bits  8..23 — top-of-column height (u16, voxel y of the top
//!                  solid voxel; 0 means "no solid" / "air column")
//!   bits 24..31 — reserved for future adjacency cost / region tags
//!
//! ## Phase 4a scope
//!
//! Engine-side build + attachment + tests. No DSL surface — Phase
//! 4b adds `navgrid.walkable(region, cell)` for physics rules.
//!
//! ## Design decisions
//!
//! - **Simplified spec build.** The spec example chains
//!   `column_reduce_xz → per_cell_classify → connect_neighbors`.
//!   This implementation collapses the three steps into one
//!   `build_navgrid` pass — the per-column scan finds top-of-column,
//!   classifies walkable based on "top + 1 is air" (single-voxel
//!   step), and writes the packed u32 in one go. Splitting back
//!   into the spec's three-stage form is a Phase 4b/4c refactor
//!   when the WGSL build kernel lands (the spec stages map naturally
//!   to GPU passes).
//!
//! - **No adjacency cost yet.** Bits 24..31 reserved. The
//!   assassination fixture (Phase 5) uses gradient-steering on
//!   believed-distance, not real path costs — adjacency lights up
//!   when a fixture needs A* over the navgrid directly.
//!
//! - **One navgrid per region**. The spec attaches indices to
//!   regions via `region.indices: BTreeMap<IndexKind, IndexHandle>`.
//!   This module ships the `NavgridIndex` value; Phase 4b wires
//!   the attachment + per-runtime allocator. For now, callers
//!   build a `NavgridIndex` directly via `build_navgrid`.

use crate::region::{Aabb, VoxelRegion, VoxelRegionBounds};

/// Per spec §7.2 storage: `per_cell_2d(max_cells = 16384,
/// bytes_per_cell = 4)`. Caps a navgrid at 16384 cells (e.g.
/// 128×128 in (x, z)) per region.
pub const NAVGRID_MAX_CELLS: u32 = 16384;
pub const NAVGRID_BYTES_PER_CELL: u32 = 4;

/// Default agent step height (per spec §7.2's `AGENT_STEP_HEIGHT`
/// constant referenced in the build body). 1 voxel for now.
pub const AGENT_STEP_HEIGHT: u16 = 1;

/// Packed u32 navgrid cell. Read via [`NavgridCell::from_u32`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NavgridCell(pub u32);

impl NavgridCell {
    pub fn walkable(self) -> bool {
        (self.0 & 0xFF) != 0
    }
    pub fn height(self) -> u16 {
        ((self.0 >> 8) & 0xFFFF) as u16
    }
    pub fn from_parts(walkable: bool, height: u16) -> Self {
        let w = if walkable { 1u32 } else { 0u32 };
        let h = (height as u32) << 8;
        Self(w | h)
    }
}

#[derive(Debug, Clone)]
pub struct NavgridIndex {
    /// (x, z) extent in voxel cells. `cells.len() == size_x * size_z`.
    pub size_x: u32,
    pub size_z: u32,
    /// World-space origin (the AABB's min.x / min.z, truncated to
    /// voxel-grid integers). Used by callers to map world pos →
    /// (cx, cz) cell coords.
    pub origin_x: i32,
    pub origin_z: i32,
    /// Y of the cell's top-of-column scan range. Caller controls
    /// the y range; for fixtures with flat-ish heightmaps,
    /// `(min_y, max_y) = (region.bounds.min.y, region.bounds.max.y)`.
    pub scan_min_y: i32,
    pub scan_max_y: i32,
    /// Per-cell packed data, row-major by (z, x): index =
    /// `cz * size_x + cx`.
    pub cells: Vec<NavgridCell>,
}

impl NavgridIndex {
    pub fn cell_at(&self, cx: u32, cz: u32) -> Option<NavgridCell> {
        if cx >= self.size_x || cz >= self.size_z {
            return None;
        }
        Some(self.cells[(cz * self.size_x + cx) as usize])
    }
    pub fn world_to_cell(&self, world_x: f32, world_z: f32) -> Option<(u32, u32)> {
        let cx_i = world_x.floor() as i32 - self.origin_x;
        let cz_i = world_z.floor() as i32 - self.origin_z;
        if cx_i < 0 || cz_i < 0 {
            return None;
        }
        let cx = cx_i as u32;
        let cz = cz_i as u32;
        if cx >= self.size_x || cz >= self.size_z {
            return None;
        }
        Some((cx, cz))
    }
}

/// Build a [`NavgridIndex`] for `region` against the voxel grid
/// in `solid_at` (a closure: `(x, y, z) -> bool` returning true
/// for solid). Per spec, region bounds drive the (x, z) extent;
/// y is scanned over the region's AABB.
///
/// Returns `Err` if the region's bounds aren't AABB (chunk-set
/// bounds need the chunk-coord transform that Phase 4b ships) or
/// if the resulting cell count exceeds `NAVGRID_MAX_CELLS`.
pub fn build_navgrid<F>(region: &VoxelRegion, solid_at: F) -> Result<NavgridIndex, NavgridBuildError>
where
    F: Fn(i32, i32, i32) -> bool,
{
    let aabb = match &region.bounds {
        VoxelRegionBounds::Aabb(a) => *a,
        VoxelRegionBounds::ChunkSet(_) => {
            return Err(NavgridBuildError::ChunkSetBoundsNotSupported);
        }
    };

    let (origin_x, origin_z, size_x, size_z, scan_min_y, scan_max_y) = aabb_to_cell_extent(aabb);
    let cell_count = size_x * size_z;
    if cell_count > NAVGRID_MAX_CELLS {
        return Err(NavgridBuildError::TooManyCells {
            requested: cell_count,
            max: NAVGRID_MAX_CELLS,
        });
    }

    let mut cells: Vec<NavgridCell> = Vec::with_capacity(cell_count as usize);

    for cz in 0..size_z {
        for cx in 0..size_x {
            let wx = origin_x + cx as i32;
            let wz = origin_z + cz as i32;
            // Column scan: find top of column (highest y with
            // solid_at(wx, y, wz) true). Iterate top-down so the
            // first hit IS the top.
            let mut top_y: Option<i32> = None;
            for wy in (scan_min_y..scan_max_y).rev() {
                if solid_at(wx, wy, wz) {
                    top_y = Some(wy);
                    break;
                }
            }
            let cell = match top_y {
                Some(y) => {
                    // Walkable if the voxel ABOVE the top is air
                    // (i.e. an agent can stand on top_y). For
                    // top_y == scan_max_y - 1 the column reaches
                    // the top of the scan range; we conservatively
                    // call it non-walkable (the air-above can't
                    // be sampled).
                    let walkable = (y + 1) < scan_max_y && !solid_at(wx, y + 1, wz);
                    NavgridCell::from_parts(walkable, (y - scan_min_y).clamp(0, u16::MAX as i32) as u16)
                }
                None => {
                    // No solid in the column — treat as non-walkable
                    // (no ground to stand on). A future "fly" mode
                    // would flip this.
                    NavgridCell::from_parts(false, 0)
                }
            };
            cells.push(cell);
        }
    }

    Ok(NavgridIndex {
        size_x,
        size_z,
        origin_x,
        origin_z,
        scan_min_y,
        scan_max_y,
        cells,
    })
}

fn aabb_to_cell_extent(aabb: Aabb) -> (i32, i32, u32, u32, i32, i32) {
    let origin_x = aabb.min[0].floor() as i32;
    let origin_z = aabb.min[2].floor() as i32;
    let size_x = ((aabb.max[0] - aabb.min[0]).ceil() as i32).max(0) as u32;
    let size_z = ((aabb.max[2] - aabb.min[2]).ceil() as i32).max(0) as u32;
    let scan_min_y = aabb.min[1].floor() as i32;
    let scan_max_y = aabb.max[1].ceil() as i32;
    (origin_x, origin_z, size_x, size_z, scan_min_y, scan_max_y)
}

#[derive(Debug, Clone, PartialEq)]
pub enum NavgridBuildError {
    /// `ChunkSet` bounds aren't supported until the chunk-coord
    /// transform lands. Convert to AABB at the caller.
    ChunkSetBoundsNotSupported,
    /// Cell count exceeds the spec's per-region budget.
    TooManyCells { requested: u32, max: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::{Aabb, VoxelRegion, VoxelRegionBounds, VoxelRegionId, VoxelRegionKind};

    fn region_aabb(min: [f32; 3], max: [f32; 3]) -> VoxelRegion {
        VoxelRegion {
            id: VoxelRegionId::from_raw_for_test(),
            bounds: VoxelRegionBounds::Aabb(Aabb { min, max }),
            kind: VoxelRegionKind(0),
            created_at_tick: 0,
        }
    }

    #[test]
    fn flat_ground_classifies_walkable() {
        // 4x4 cells, ground at y=0 across the entire region.
        let region = region_aabb([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);
        let navgrid = build_navgrid(&region, |_, y, _| y == 0).expect("build");
        assert_eq!(navgrid.size_x, 4);
        assert_eq!(navgrid.size_z, 4);
        for cz in 0..navgrid.size_z {
            for cx in 0..navgrid.size_x {
                let cell = navgrid.cell_at(cx, cz).unwrap();
                assert!(
                    cell.walkable(),
                    "cell ({cx}, {cz}) should be walkable (flat ground at y=0)"
                );
                assert_eq!(cell.height(), 0); // height = top_y - scan_min_y = 0
            }
        }
    }

    #[test]
    fn wall_columns_not_walkable() {
        // 4x4 cells, ground at y=0, plus a wall at x=2 spanning
        // y=0..4 (column full of solid → no air above the top).
        let region = region_aabb([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);
        let navgrid = build_navgrid(&region, |x, y, _| {
            // Ground everywhere at y=0, plus the wall at x=2.
            y == 0 || (x == 2 && (0..4).contains(&y))
        })
        .expect("build");
        for cz in 0..navgrid.size_z {
            // Wall column (x=2 in world = cx=2): top_y=3 → above
            // is scan_max_y (out of scan range) → non-walkable.
            let wall = navgrid.cell_at(2, cz).unwrap();
            assert!(
                !wall.walkable(),
                "wall column at cx=2 cz={cz} should be non-walkable; got {wall:?}"
            );
            // Non-wall columns: top_y=0 → above (y=1) is air →
            // walkable.
            for cx in [0, 1, 3] {
                let floor = navgrid.cell_at(cx, cz).unwrap();
                assert!(
                    floor.walkable(),
                    "floor column at cx={cx} cz={cz} should be walkable; got {floor:?}"
                );
            }
        }
    }

    #[test]
    fn air_column_not_walkable() {
        // 2x2 region, NO solids at all — purely air. No ground to
        // stand on → every cell non-walkable.
        let region = region_aabb([0.0, 0.0, 0.0], [2.0, 4.0, 2.0]);
        let navgrid = build_navgrid(&region, |_, _, _| false).expect("build");
        for cell in &navgrid.cells {
            assert!(!cell.walkable(), "air-only cell should be non-walkable");
        }
    }

    #[test]
    fn world_to_cell_handles_origin_offset() {
        // Region at origin (10, _, 20) → world (15, _, 25) maps to (5, 5).
        let region = region_aabb([10.0, 0.0, 20.0], [20.0, 4.0, 30.0]);
        let navgrid = build_navgrid(&region, |_, y, _| y == 0).expect("build");
        assert_eq!(navgrid.world_to_cell(15.0, 25.0), Some((5, 5)));
        assert_eq!(navgrid.world_to_cell(10.0, 20.0), Some((0, 0)));
        assert_eq!(navgrid.world_to_cell(5.0, 25.0), None); // x < origin
    }

    #[test]
    fn chunk_set_bounds_rejected() {
        let region = VoxelRegion {
            id: VoxelRegionId::from_raw_for_test(),
            bounds: VoxelRegionBounds::ChunkSet(vec![0, 1, 2]),
            kind: VoxelRegionKind(0),
            created_at_tick: 0,
        };
        let err = build_navgrid(&region, |_, _, _| false).unwrap_err();
        assert_eq!(err, NavgridBuildError::ChunkSetBoundsNotSupported);
    }

    #[test]
    fn too_many_cells_rejected() {
        // 200×200 = 40000 cells > NAVGRID_MAX_CELLS (16384).
        let region = region_aabb([0.0, 0.0, 0.0], [200.0, 4.0, 200.0]);
        let err = build_navgrid(&region, |_, _, _| false).unwrap_err();
        assert!(matches!(err, NavgridBuildError::TooManyCells { .. }));
    }
}
