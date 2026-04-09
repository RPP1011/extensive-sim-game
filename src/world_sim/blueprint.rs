//! Blueprint data structure for NPC-driven building construction.
//!
//! A blueprint describes which voxels a building is made of and where they go,
//! relative to a blueprint origin. NPCs construct buildings by placing blueprint
//! voxels one at a time, guided by `sort_by_support_order()` which ensures
//! structural validity (ground first, roof last).

use crate::world_sim::state::BuildingType;
use crate::world_sim::systems::buildings::{
    building_floor_material, building_wall_material, wall_height,
};
use crate::world_sim::voxel::VoxelMaterial;

/// A single voxel placement within a blueprint.
#[derive(Clone, Debug)]
pub struct BlueprintVoxel {
    /// Position relative to blueprint origin (x, y, z).
    pub offset: (i8, i8, i8),
    /// Material to place.
    pub material: VoxelMaterial,
    /// Whether this voxel has been placed in the world.
    pub placed: bool,
}

/// A building blueprint: an ordered list of voxel placements.
#[derive(Clone, Debug)]
pub struct Blueprint {
    pub voxels: Vec<BlueprintVoxel>,
    /// World-space voxel origin (bottom-left corner). Set when attached to a building.
    pub origin: (i32, i32, i32),
}

impl Blueprint {
    /// Stable-sort placements by Z ascending (ground first, roof last).
    /// Within a Z layer, perimeter voxels sort before interior voxels.
    pub fn sort_by_support_order(&mut self) {
        self.voxels.sort_by(|a, b| {
            let az = a.offset.2;
            let bz = b.offset.2;
            az.cmp(&bz).then_with(|| {
                // Perimeter voxels (walls) before interior — we use a heuristic:
                // a voxel is "perimeter" if it's a wall material, interior otherwise.
                let a_perim = is_perimeter_material(a.material);
                let b_perim = is_perimeter_material(b.material);
                // false < true, so we reverse: perimeter (true) should come first
                b_perim.cmp(&a_perim)
            })
        });
    }

    /// Count of unplaced voxels.
    pub fn remaining(&self) -> usize {
        self.voxels.iter().filter(|v| !v.placed).count()
    }

    /// First unplaced voxel (index and reference).
    pub fn next_unplaced(&self) -> Option<(usize, &BlueprintVoxel)> {
        self.voxels
            .iter()
            .enumerate()
            .find(|(_, v)| !v.placed)
    }

    /// Generate a blueprint for a building: floor + perimeter walls.
    ///
    /// `fp_w` and `fp_h` are the footprint dimensions (in voxels).
    /// `_tier` is reserved for future use (material upgrades, etc.).
    pub fn generate(building_type: BuildingType, fp_w: usize, fp_h: usize, _tier: u32) -> Self {
        let floor_mat = building_floor_material(building_type);
        let wall_mat = building_wall_material(building_type);
        let wh = wall_height(building_type);

        let mut voxels = Vec::new();

        // Floor at z=0
        for dy in 0..fp_h as i8 {
            for dx in 0..fp_w as i8 {
                voxels.push(BlueprintVoxel {
                    offset: (dx, dy, 0),
                    material: floor_mat,
                    placed: false,
                });
            }
        }

        // Perimeter walls at z=1..=wh
        for story in 0..wh {
            let z = 1 + story as i8;
            for dy in 0..fp_h as i8 {
                for dx in 0..fp_w as i8 {
                    let on_edge = dx == 0
                        || dy == 0
                        || dx == (fp_w as i8 - 1)
                        || dy == (fp_h as i8 - 1);
                    if on_edge {
                        voxels.push(BlueprintVoxel {
                            offset: (dx, dy, z),
                            material: wall_mat,
                            placed: false,
                        });
                    }
                }
            }
        }

        let mut bp = Blueprint { voxels, origin: (0, 0, 0) };
        bp.sort_by_support_order();
        bp
    }
}

/// Heuristic: wall/structural materials are "perimeter".
fn is_perimeter_material(mat: VoxelMaterial) -> bool {
    matches!(
        mat,
        VoxelMaterial::WoodPlanks
            | VoxelMaterial::StoneBlock
            | VoxelMaterial::StoneBrick
            | VoxelMaterial::Thatch
            | VoxelMaterial::Iron
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_by_support_order_puts_ground_first() {
        let mut bp = Blueprint {
            origin: (0, 0, 0),
            voxels: vec![
                BlueprintVoxel {
                    offset: (0, 0, 2),
                    material: VoxelMaterial::WoodPlanks,
                    placed: false,
                },
                BlueprintVoxel {
                    offset: (0, 0, 0),
                    material: VoxelMaterial::WoodPlanks,
                    placed: false,
                },
                BlueprintVoxel {
                    offset: (0, 0, 1),
                    material: VoxelMaterial::WoodPlanks,
                    placed: false,
                },
            ],
        };
        bp.sort_by_support_order();
        let zs: Vec<i8> = bp.voxels.iter().map(|v| v.offset.2).collect();
        assert_eq!(zs, vec![0, 1, 2]);
    }

    #[test]
    fn generate_house_blueprint_has_floor_and_walls() {
        let bp = Blueprint::generate(BuildingType::House, 3, 3, 1);

        // Floor: 3x3 = 9 voxels at z=0
        let floor_voxels: Vec<_> = bp.voxels.iter().filter(|v| v.offset.2 == 0).collect();
        assert_eq!(floor_voxels.len(), 9, "expected 9 floor voxels at z=0");

        // Wall height for House is 2, so walls at z=1 and z=2
        let wall_voxels: Vec<_> = bp.voxels.iter().filter(|v| v.offset.2 > 0).collect();
        assert!(!wall_voxels.is_empty(), "expected wall voxels at z>0");

        // For a 3x3 footprint, all edge cells = 8 perimeter cells per layer, 2 layers = 16
        let expected_wall_count = 8 * 2;
        assert_eq!(
            wall_voxels.len(),
            expected_wall_count,
            "expected {} wall voxels for 3x3 with wall_height=2",
            expected_wall_count
        );

        // Verify support order: all z=0 before z=1 before z=2
        let zs: Vec<i8> = bp.voxels.iter().map(|v| v.offset.2).collect();
        let mut sorted_zs = zs.clone();
        sorted_zs.sort();
        assert_eq!(zs, sorted_zs, "voxels should be sorted by z ascending");

        // All voxels start unplaced
        assert!(bp.voxels.iter().all(|v| !v.placed));
        assert_eq!(bp.remaining(), bp.voxels.len());
    }
}
