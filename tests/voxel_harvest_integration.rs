//! Integration tests for the voxel harvest and structural collapse loop.
//!
//! These verify end-to-end behavior: NPCs harvesting trees yield wood commodities,
//! and removing trunk support causes canopy collapse via structural_tick.

use game::world_sim::state::{Entity, EntityKind, Inventory, NpcData, WorldState, WorldTeam};
use game::world_sim::systems::structural_tick::structural_tick;
use game::world_sim::systems::voxel_harvest::harvest_tick;
use game::world_sim::voxel::{ChunkPos, Voxel, VoxelMaterial};

/// Helper: create a minimal NPC entity with inventory and harvest_target.
fn make_harvest_npc(id: u32, pos: (f32, f32), harvest_target: Option<(i32, i32, i32)>) -> Entity {
    let mut npc = NpcData::default();
    npc.harvest_target = harvest_target;
    Entity {
        id,
        kind: EntityKind::Npc,
        team: WorldTeam::Neutral,
        pos,
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
fn npc_harvests_tree_and_collects_wood() {
    let mut state = WorldState::new(42);

    // Generate terrain chunks around origin so surface_height works.
    for cx in -1..=1 {
        for cy in -1..=1 {
            for cz in 0..=3 {
                state
                    .voxel_world
                    .generate_chunk(ChunkPos::new(cx, cy, cz), 42);
            }
        }
    }

    // Find surface height at (8, 8) and place a 6-voxel WoodLog trunk.
    let vx = 8;
    let vy = 8;
    let surface = state.voxel_world.surface_height(vx, vy);
    for dz in 0..6 {
        state
            .voxel_world
            .set_voxel(vx, vy, surface + dz, Voxel::new(VoxelMaterial::WoodLog));
    }

    // Create NPC targeting the lowest trunk voxel.
    let target = (vx, vy, surface);
    state
        .entities
        .push(make_harvest_npc(0, (vx as f32, vy as f32), Some(target)));

    // Count initial WoodLog voxels.
    let initial_wood_count = (0..6)
        .filter(|dz| {
            state.voxel_world.get_voxel(vx, vy, surface + dz).material == VoxelMaterial::WoodLog
        })
        .count();
    assert_eq!(initial_wood_count, 6, "should start with 6 WoodLog voxels");

    // Call harvest_tick repeatedly — WoodLog hardness is 15, so up to 100 ticks
    // should break at least one (and auto-advance to the next).
    for _ in 0..100 {
        harvest_tick(&mut state, 0);
    }

    // Count remaining WoodLog voxels.
    let remaining_wood_count = (0..6)
        .filter(|dz| {
            state.voxel_world.get_voxel(vx, vy, surface + dz).material == VoxelMaterial::WoodLog
        })
        .count();

    assert!(
        remaining_wood_count < initial_wood_count,
        "at least one WoodLog should have been mined: {} remaining out of {}",
        remaining_wood_count,
        initial_wood_count
    );

    // NPC inventory should have WOOD commodity > 0.
    let inv = state.entities[0].inventory.as_ref().unwrap();
    let wood_amount = inv.commodities[game::world_sim::commodity::WOOD];
    assert!(
        wood_amount > 0.0,
        "NPC should have collected WOOD, got {}",
        wood_amount
    );

    // harvest_target should have auto-advanced (or be None if all wood gone).
    let npc = state.entities[0].npc.as_ref().unwrap();
    if remaining_wood_count == 0 {
        // All wood gone, target should be None (no more WoodLog nearby).
        assert!(
            npc.harvest_target.is_none(),
            "harvest_target should be None when all wood is gone"
        );
    } else {
        // Still some wood left; target should differ from the original lowest voxel
        // (which was mined away).
        assert_ne!(
            npc.harvest_target,
            Some(target),
            "harvest_target should have advanced past the original lowest voxel"
        );
    }

    println!(
        "Harvested {} WoodLog voxels, collected {} WOOD, target now: {:?}",
        initial_wood_count - remaining_wood_count,
        wood_amount,
        npc.harvest_target
    );
}

#[test]
fn structural_collapse_on_tree_chop() {
    let mut state = WorldState::new(42);

    // Don't generate terrain — build a free-standing tree from z=0 so the only
    // ground anchor is the trunk base itself. This ensures the canopy has no
    // accidental support from terrain in adjacent columns.

    let vx = 8;
    let vy = 8;
    let base_z = 0;

    // Place trunk: 4 WoodLog voxels vertically starting at z=0.
    for dz in 0..4 {
        state
            .voxel_world
            .set_voxel(vx, vy, base_z + dz, Voxel::new(VoxelMaterial::WoodLog));
    }

    // Place canopy: single-layer 3x3 slab of Grass voxels above trunk.
    // A single layer ensures no canopy voxel has another solid voxel directly
    // below it — the only vertical support came from the trunk at (vx, vy).
    let canopy_base = base_z + 4;
    for dx in -1..=1 {
        for dy in -1..=1 {
            state.voxel_world.set_voxel(
                vx + dx,
                vy + dy,
                canopy_base,
                Voxel::new(VoxelMaterial::Grass),
            );
        }
    }

    // Count initial canopy voxels.
    let initial_canopy = count_canopy(&state, vx, vy, canopy_base);
    assert_eq!(initial_canopy, 9, "should start with 3x3 = 9 canopy voxels");

    // Chop: remove all trunk voxels (set to Air).
    for dz in 0..4 {
        state
            .voxel_world
            .set_voxel(vx, vy, base_z + dz, Voxel::default());
    }

    // Clear any pre-existing structural events.
    state.structural_events.clear();

    // Run structural_tick (may need multiple passes since MAX_CHUNKS_PER_TICK limits work).
    for _ in 0..10 {
        structural_tick(&mut state);
    }

    // Count remaining canopy voxels.
    let remaining_canopy = count_canopy(&state, vx, vy, canopy_base);

    println!(
        "Canopy: {} of {} voxels collapsed, {} structural events",
        initial_canopy - remaining_canopy,
        initial_canopy,
        state.structural_events.len()
    );

    assert!(
        remaining_canopy < initial_canopy,
        "canopy voxels should mostly collapse: {} remaining out of {}",
        remaining_canopy,
        initial_canopy
    );

    assert!(
        !state.structural_events.is_empty(),
        "structural_events should contain FragmentCollapse events"
    );
}

/// Count solid canopy voxels in a 3x3 single-layer region.
fn count_canopy(state: &WorldState, vx: i32, vy: i32, canopy_base: i32) -> usize {
    let mut count = 0;
    for dx in -1..=1 {
        for dy in -1..=1 {
            if state
                .voxel_world
                .get_voxel(vx + dx, vy + dy, canopy_base)
                .material
                .is_solid()
            {
                count += 1;
            }
        }
    }
    count
}
