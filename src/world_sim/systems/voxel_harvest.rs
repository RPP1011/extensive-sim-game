//! Voxel harvest system — NPCs mine targeted voxels each work tick.
//!
//! When an NPC has an active `harvest_target`, `harvest_tick()` applies damage
//! to the target voxel. When the voxel breaks, yield is added to NPC inventory
//! and the next adjacent voxel of the same material is auto-selected.

use crate::world_sim::state::{BuildingType, WorldState};
use crate::world_sim::voxel::{world_to_voxel, VoxelMaterial};

/// Damage applied per harvest tick.
const HARVEST_DAMAGE_PER_TICK: u8 = 1;

/// Process one harvest tick for an NPC with an active harvest_target.
///
/// Returns `Some((commodity_index, amount))` if the target voxel broke and
/// yielded a commodity, `None` otherwise (still mining, target gone, or no target).
pub fn harvest_tick(state: &mut WorldState, entity_idx: usize) -> Option<(usize, f32)> {
    // Get harvest target from NPC data.
    let (vx, vy, vz) = {
        let npc = state.entities[entity_idx].npc.as_ref()?;
        npc.harvest_target?
    };

    // Check target voxel is still solid.
    let target_voxel = state.voxel_world.get_voxel(vx, vy, vz);
    if !target_voxel.material.is_solid() {
        // Voxel is gone (mined by someone else or collapsed).
        if let Some(npc) = state.entities[entity_idx].npc.as_mut() {
            npc.harvest_target = None;
        }
        return None;
    }

    let target_material = target_voxel.material;

    // Apply mining damage.
    let result = state.voxel_world.mine_voxel(vx, vy, vz, HARVEST_DAMAGE_PER_TICK);

    match result {
        Some((_broken_material, yield_info)) => {
            // Voxel broke. Add yield to NPC inventory if applicable.
            if let Some((commodity_idx, amount)) = yield_info {
                if let Some(inv) = state.entities[entity_idx].inventory.as_mut() {
                    inv.commodities[commodity_idx] += amount;
                }
            }

            // Find next adjacent voxel of same material.
            let next_target = state.voxel_world.find_nearest_harvestable(vx, vy, target_material, 4);
            if let Some(npc) = state.entities[entity_idx].npc.as_mut() {
                npc.harvest_target = next_target;
            }

            yield_info
        }
        None => {
            // Still mining, voxel not broken yet.
            None
        }
    }
}

/// Find a harvest target based on NPC knowledge and desired material.
pub fn select_harvest_target(
    state: &WorldState,
    _entity_idx: usize,
    desired_material: VoxelMaterial,
    search_center: (f32, f32),
    search_radius: i32,
) -> Option<(i32, i32, i32)> {
    let (cvx, cvy, _cvz) = world_to_voxel(search_center.0, search_center.1, 0.0);
    state.voxel_world.find_nearest_harvestable(cvx, cvy, desired_material, search_radius)
}

/// Map a building type to the voxel material it harvests.
pub fn required_harvest_material(building_type: BuildingType) -> Option<VoxelMaterial> {
    match building_type {
        BuildingType::Sawmill => Some(VoxelMaterial::WoodLog),
        BuildingType::Mine => Some(VoxelMaterial::IronOre),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::{Entity, EntityKind, Inventory, NpcData, WorldTeam};
    use crate::world_sim::voxel::{Voxel, VoxelMaterial};

    /// Helper: create a minimal NPC entity with inventory and harvest_target.
    fn make_npc_entity(id: u32, harvest_target: Option<(i32, i32, i32)>) -> Entity {
        let mut npc = NpcData::default();
        npc.harvest_target = harvest_target;
        Entity {
            id,
            kind: EntityKind::Npc,
            team: WorldTeam::Neutral,
            pos: (0.0, 0.0),
            grid_id: None,
            local_pos: None,
            alive: true,
            hp: 100.0,
            max_hp: 100.0,
            shield_hp: 0.0,
            armor: 0.0,
            magic_resist: 0.0,
            attack_damage: 10.0,
            attack_range: 1.0,
            move_speed: 1.0,
            level: 1,
            status_effects: Vec::new(),
            npc: Some(npc),
            building: None,
            item: None,
            resource: None,
            inventory: Some(Inventory::default()),
            move_target: None,
            move_speed_mult: 1.0,
            enemy_capabilities: None,
        }
    }

    #[test]
    fn harvest_tick_damages_and_breaks_voxel() {
        let mut state = WorldState::new(42);
        // Place a Dirt voxel (hardness = 5) at (2, 2, 2).
        state.voxel_world.set_voxel(2, 2, 2, Voxel::new(VoxelMaterial::Dirt));
        // Create an NPC entity targeting that voxel.
        state.entities.push(make_npc_entity(1, Some((2, 2, 2))));

        // First 4 ticks: voxel not broken yet.
        for _ in 0..4 {
            let result = harvest_tick(&mut state, 0);
            assert!(result.is_none(), "voxel should not break before hardness reached");
            // Voxel should still be Dirt.
            assert_eq!(state.voxel_world.get_voxel(2, 2, 2).material, VoxelMaterial::Dirt);
        }

        // 5th tick: voxel breaks, should yield something.
        let result = harvest_tick(&mut state, 0);
        // Dirt yields (FOOD, 0.1).
        assert!(result.is_some(), "voxel should break on 5th tick");
        let (commodity_idx, amount) = result.unwrap();
        assert_eq!(commodity_idx, crate::world_sim::commodity::FOOD);
        assert!((amount - 0.1).abs() < 1e-6);
        // Voxel should now be Air.
        assert_eq!(state.voxel_world.get_voxel(2, 2, 2).material, VoxelMaterial::Air);
        // Inventory should have the yield.
        let inv = state.entities[0].inventory.as_ref().unwrap();
        assert!((inv.commodities[commodity_idx] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn harvest_tick_clears_target_when_voxel_gone() {
        let mut state = WorldState::new(42);
        // Set harvest_target to an Air position (no voxel there).
        // We don't place any voxel at (5, 5, 5) — it defaults to Air.
        state.entities.push(make_npc_entity(1, Some((5, 5, 5))));

        let result = harvest_tick(&mut state, 0);
        assert!(result.is_none());
        // harvest_target should be cleared.
        let npc = state.entities[0].npc.as_ref().unwrap();
        assert!(npc.harvest_target.is_none(), "harvest_target should be None when voxel is Air");
    }

    #[test]
    fn harvest_tick_returns_none_without_target() {
        let mut state = WorldState::new(42);
        // NPC with no harvest target.
        state.entities.push(make_npc_entity(1, None));

        let result = harvest_tick(&mut state, 0);
        assert!(result.is_none());
    }

    #[test]
    fn harvest_tick_auto_selects_next_adjacent() {
        let mut state = WorldState::new(42);
        // Place two adjacent Dirt voxels.
        state.voxel_world.set_voxel(3, 3, 3, Voxel::new(VoxelMaterial::Dirt));
        state.voxel_world.set_voxel(4, 3, 3, Voxel::new(VoxelMaterial::Dirt));
        state.entities.push(make_npc_entity(1, Some((3, 3, 3))));

        // Mine through the first voxel (hardness 5).
        for _ in 0..5 {
            harvest_tick(&mut state, 0);
        }
        // First voxel should be gone.
        assert_eq!(state.voxel_world.get_voxel(3, 3, 3).material, VoxelMaterial::Air);
        // harvest_target should now point to the adjacent Dirt voxel.
        let npc = state.entities[0].npc.as_ref().unwrap();
        assert_eq!(npc.harvest_target, Some((4, 3, 3)));
    }

    #[test]
    fn required_harvest_material_mapping() {
        assert_eq!(required_harvest_material(BuildingType::Sawmill), Some(VoxelMaterial::WoodLog));
        assert_eq!(required_harvest_material(BuildingType::Mine), Some(VoxelMaterial::IronOre));
        assert_eq!(required_harvest_material(BuildingType::Farm), None);
        assert_eq!(required_harvest_material(BuildingType::Inn), None);
    }
}
