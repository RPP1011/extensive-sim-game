//! Blueprint-based voxel construction system.
//!
//! Instead of stamping all building voxels at once, NPCs place them one at a
//! time from blueprints. `advance_blueprint_construction` places one voxel per
//! call, consuming the required commodity from the worker's inventory.

use crate::world_sim::blueprint::Blueprint;
use crate::world_sim::commodity;
use crate::world_sim::interior_gen::footprint_size;
use crate::world_sim::state::WorldState;
use crate::world_sim::voxel::{world_to_voxel, Voxel, VoxelMaterial};

/// Map a blueprint voxel material to the commodity index required to place it.
fn material_commodity(mat: VoxelMaterial) -> usize {
    match mat {
        VoxelMaterial::WoodLog | VoxelMaterial::WoodPlanks | VoxelMaterial::Thatch => {
            commodity::WOOD
        }
        VoxelMaterial::StoneBlock
        | VoxelMaterial::StoneBrick
        | VoxelMaterial::CutStone
        | VoxelMaterial::Sandstone
        | VoxelMaterial::Marble
        | VoxelMaterial::Basalt
        | VoxelMaterial::Concrete
        | VoxelMaterial::Brick
        | VoxelMaterial::Obsidian => commodity::CRYSTAL,
        VoxelMaterial::Iron | VoxelMaterial::Steel | VoxelMaterial::Bronze => commodity::IRON,
        VoxelMaterial::Glass | VoxelMaterial::Ceramic => commodity::CRYSTAL,
        VoxelMaterial::Bone => commodity::HIDE,
        // Farmland, Crop, and other materials default to WOOD as a catch-all.
        _ => commodity::WOOD,
    }
}

/// Advance blueprint construction for a building.
///
/// Places one voxel per call if the worker has the required material.
/// Returns `true` if a voxel was placed.
pub fn advance_blueprint_construction(
    state: &mut WorldState,
    building_entity_idx: usize,
    worker_entity_idx: usize,
) -> bool {
    // --- Validate indices ---
    if building_entity_idx >= state.entities.len() || worker_entity_idx >= state.entities.len() {
        return false;
    }

    // --- Find next unplaced voxel and its requirements ---
    let (voxel_idx, material, offset) = {
        let building = match state.entities[building_entity_idx].building.as_ref() {
            Some(b) => b,
            None => return false,
        };
        let bp = match building.blueprint.as_ref() {
            Some(bp) => bp,
            None => return false,
        };
        let (idx, bv) = match bp.next_unplaced() {
            Some(pair) => pair,
            None => return false,
        };
        (idx, bv.material, bv.offset)
    };

    let required_commodity = material_commodity(material);

    // --- Check worker inventory ---
    {
        let inv = match state.entities[worker_entity_idx].inventory.as_ref() {
            Some(inv) => inv,
            None => return false,
        };
        if inv.commodities[required_commodity] < 1.0 {
            return false;
        }
    }

    // --- Deduct commodity from worker ---
    state.entities[worker_entity_idx]
        .inventory
        .as_mut()
        .unwrap()
        .commodities[required_commodity] -= 1.0;

    // --- Compute world position and place voxel ---
    let origin = state.entities[building_entity_idx]
        .building
        .as_ref()
        .unwrap()
        .blueprint
        .as_ref()
        .unwrap()
        .origin;

    let wx = origin.0 + offset.0 as i32;
    let wy = origin.1 + offset.1 as i32;
    let wz = origin.2 + offset.2 as i32;

    state.voxel_world.set_voxel(wx, wy, wz, Voxel::new(material));

    // --- Mark placement as placed ---
    state.entities[building_entity_idx]
        .building
        .as_mut()
        .unwrap()
        .blueprint
        .as_mut()
        .unwrap()
        .voxels[voxel_idx]
        .placed = true;

    true
}

/// Generate and attach a blueprint to a building entity.
pub fn attach_blueprint(state: &mut WorldState, building_entity_idx: usize) {
    if building_entity_idx >= state.entities.len() {
        return;
    }

    let entity = &state.entities[building_entity_idx];
    let building = match entity.building.as_ref() {
        Some(b) => b,
        None => return,
    };

    let building_type = building.building_type;
    let tier = building.tier;
    let (fp_w, fp_h) = footprint_size(building_type, tier);

    let mut bp = Blueprint::generate(building_type, fp_w, fp_h, tier as u32);

    // Set origin from entity world position converted to voxel coords.
    let pos = entity.pos;
    let (vx, vy) = {
        let (x, y, _) = world_to_voxel(pos.0, pos.1, 0.0);
        (x, y)
    };
    let vz = state.voxel_world.surface_height(vx, vy);
    bp.origin = (vx, vy, vz);

    state.entities[building_entity_idx]
        .building
        .as_mut()
        .unwrap()
        .blueprint = Some(bp);
}

/// Find non-terrain voxels in the blueprint footprint that need clearing.
///
/// Returns positions of solid voxels (trees, boulders, etc.) that are not
/// natural terrain and would interfere with construction.
pub fn site_clearing_targets(
    state: &WorldState,
    blueprint: &Blueprint,
) -> Vec<(i32, i32, i32)> {
    let origin = blueprint.origin;

    // Determine XY extent from blueprint voxels.
    let (max_x, max_y, max_z) = blueprint.voxels.iter().fold((0i8, 0i8, 0i8), |acc, v| {
        (acc.0.max(v.offset.0), acc.1.max(v.offset.1), acc.2.max(v.offset.2))
    });

    let mut targets = Vec::new();

    for dy in 0..=max_y as i32 {
        for dx in 0..=max_x as i32 {
            let wx = origin.0 + dx;
            let wy = origin.1 + dy;
            // Scan upward from origin Z through the blueprint height.
            for dz in 0..=max_z as i32 {
                let wz = origin.2 + dz;
                let voxel = state.voxel_world.get_voxel(wx, wy, wz);
                let mat = voxel.material;
                if !mat.is_solid() {
                    continue;
                }
                // Skip natural terrain materials.
                if is_terrain_material(mat) {
                    continue;
                }
                targets.push((wx, wy, wz));
            }
        }
    }

    targets
}

/// Materials considered natural terrain (not cleared during site preparation).
fn is_terrain_material(mat: VoxelMaterial) -> bool {
    matches!(
        mat,
        VoxelMaterial::Dirt
            | VoxelMaterial::Stone
            | VoxelMaterial::Granite
            | VoxelMaterial::Sand
            | VoxelMaterial::Clay
            | VoxelMaterial::Gravel
            | VoxelMaterial::Grass
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::blueprint::BlueprintVoxel;
    use crate::world_sim::state::{BuildingData, BuildingType, Entity, Inventory};

    /// Helper: create a minimal WorldState with voxel world.
    fn test_state() -> WorldState {
        WorldState::new(42)
    }

    /// Helper: create a building entity with a small blueprint attached.
    fn make_building_entity(id: u32) -> Entity {
        let bp = Blueprint {
            origin: (10, 10, 5),
            voxels: vec![
                BlueprintVoxel {
                    offset: (0, 0, 0),
                    material: VoxelMaterial::WoodPlanks,
                    placed: false,
                },
                BlueprintVoxel {
                    offset: (1, 0, 0),
                    material: VoxelMaterial::StoneBlock,
                    placed: false,
                },
            ],
        };
        let mut building = BuildingData::default();
        building.building_type = BuildingType::House;
        building.blueprint = Some(bp);

        let mut e = Entity::new_npc(id, (0.0, 0.0));
        e.building = Some(building);
        e
    }

    /// Helper: create a worker entity with specified commodities.
    fn make_worker_entity(id: u32, wood: f32, crystal: f32) -> Entity {
        let mut inv = Inventory::default();
        inv.commodities[commodity::WOOD] = wood;
        inv.commodities[commodity::CRYSTAL] = crystal;
        let mut e = Entity::new_npc(id, (0.0, 0.0));
        e.inventory = Some(inv);
        e
    }

    #[test]
    fn advance_construction_places_voxel() {
        let mut state = test_state();
        state.entities.push(make_building_entity(0));
        state.entities.push(make_worker_entity(1, 5.0, 5.0));

        let placed = advance_blueprint_construction(&mut state, 0, 1);
        assert!(placed, "should place a voxel when worker has materials");

        // First voxel (WoodPlanks) should be marked placed.
        let bp = state.entities[0]
            .building
            .as_ref()
            .unwrap()
            .blueprint
            .as_ref()
            .unwrap();
        assert!(bp.voxels[0].placed);
        assert!(!bp.voxels[1].placed);

        // Wood should have been deducted.
        let inv = state.entities[1].inventory.as_ref().unwrap();
        assert!((inv.commodities[commodity::WOOD] - 4.0).abs() < f32::EPSILON);

        // Voxel should exist in the world at origin + offset.
        let v = state.voxel_world.get_voxel(10, 10, 5);
        assert_eq!(v.material, VoxelMaterial::WoodPlanks);
    }

    #[test]
    fn construction_requires_material() {
        let mut state = test_state();
        state.entities.push(make_building_entity(0));
        // Worker with zero inventory.
        state.entities.push(make_worker_entity(1, 0.0, 0.0));

        let placed = advance_blueprint_construction(&mut state, 0, 1);
        assert!(!placed, "should not place voxel without materials");

        // Blueprint should remain unchanged.
        let bp = state.entities[0]
            .building
            .as_ref()
            .unwrap()
            .blueprint
            .as_ref()
            .unwrap();
        assert!(!bp.voxels[0].placed);
        assert!(!bp.voxels[1].placed);
    }

    #[test]
    fn site_clearing_finds_obstructions() {
        let mut state = test_state();

        let bp = Blueprint {
            origin: (5, 5, 10),
            voxels: vec![
                BlueprintVoxel {
                    offset: (0, 0, 0),
                    material: VoxelMaterial::WoodPlanks,
                    placed: false,
                },
                BlueprintVoxel {
                    offset: (1, 0, 0),
                    material: VoxelMaterial::WoodPlanks,
                    placed: false,
                },
            ],
        };

        // Place a tree log (obstruction) and a dirt block (terrain, should be skipped).
        state
            .voxel_world
            .set_voxel(5, 5, 10, Voxel::new(VoxelMaterial::WoodLog));
        state
            .voxel_world
            .set_voxel(6, 5, 10, Voxel::new(VoxelMaterial::Dirt));

        let targets = site_clearing_targets(&state, &bp);
        assert_eq!(targets.len(), 1, "only the WoodLog should need clearing");
        assert_eq!(targets[0], (5, 5, 10));
    }
}
