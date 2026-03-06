use bevy::prelude::*;

use crate::ai::core::{SimVec2, Team};
use crate::mission::{
    enemy_templates::{default_enemy_wave, generate_boss, is_climax_room, BOSS_UNIT_ID},
    room_gen::{generate_room, spawn_room, RoomFloor, RoomObstacle, RoomWall},
    sim_bridge::{EnemyAiState, MissionOutcome, MissionSimState, PlayerUnitMarker},
    unit_vis::{spawn_unit_visual, UnitVisual},
};

use super::types::{MissionRoomSequence, RoomDoor, spawn_boss_visual};

// ---------------------------------------------------------------------------
// System: spawn_room_door_system
// ---------------------------------------------------------------------------

/// Spawns a gold door mesh at the far end of the current room when all enemies
/// are dead and the mission is not yet on the last room.
pub fn spawn_room_door_system(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sim_state: Option<Res<MissionSimState>>,
    room_seq: Option<Res<MissionRoomSequence>>,
    door_query: Query<Entity, With<RoomDoor>>,
) {
    let (Some(sim), Some(seq)) = (sim_state, room_seq) else {
        return;
    };

    if sim.outcome.is_some() {
        return;
    }

    if seq.is_last_room() {
        return;
    }

    if door_query.iter().next().is_some() {
        return;
    }

    let all_enemies_dead = sim
        .sim
        .units
        .iter()
        .filter(|u| u.team == Team::Enemy)
        .all(|u| u.hp <= 0);

    if !all_enemies_dead {
        return;
    }

    let Some(room_type) = seq.current_room_type() else {
        return;
    };
    let layout = generate_room(seq.seed + seq.current_index as u64, *room_type);

    let local_x = layout.width / 2.0;
    let local_y = 1.5_f32;
    let local_z = layout.depth - 1.0;

    let world_pos = seq.current_room_origin + Vec3::new(local_x, local_y, local_z);

    let door_mat = materials.add(StandardMaterial {
        base_color: Color::rgb(1.0, 0.84, 0.0),
        emissive: Color::rgb(1.0, 0.7, 0.0),
        perceptual_roughness: 0.3,
        metallic: 0.8,
        ..default()
    });
    let door_mesh = meshes.add(Cuboid::new(1.0, 3.0, 0.3));

    commands.spawn((
        PbrBundle {
            mesh: door_mesh,
            material: door_mat,
            transform: Transform::from_translation(world_pos),
            ..default()
        },
        RoomDoor,
        Name::new("Advance to next room"),
    ));
}

// ---------------------------------------------------------------------------
// System: advance_room_system
// ---------------------------------------------------------------------------

/// Detects when any hero unit is within 3 world units of the `RoomDoor` entity
/// and triggers a room transition.
pub fn advance_room_system(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut sim_state: Option<ResMut<MissionSimState>>,
    mut room_seq: Option<ResMut<MissionRoomSequence>>,
    door_query: Query<(Entity, &Transform), With<RoomDoor>>,
    floor_query: Query<Entity, With<RoomFloor>>,
    wall_query: Query<Entity, With<RoomWall>>,
    obstacle_query: Query<Entity, With<RoomObstacle>>,
    unit_vis_query: Query<(Entity, &UnitVisual)>,
    _player_marker_query: Query<Entity, With<PlayerUnitMarker>>,
) {
    let (Some(ref mut sim), Some(ref mut seq)) = (sim_state.as_mut(), room_seq.as_mut()) else {
        return;
    };

    // Last-room victory check.
    if seq.is_last_room() && sim.outcome.is_none() {
        let all_enemies_dead = sim
            .sim
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy)
            .all(|u| u.hp <= 0);

        if all_enemies_dead {
            sim.outcome = Some(MissionOutcome::Victory);
        }
        return;
    }

    let Ok((door_entity, door_transform)) = door_query.get_single() else {
        return;
    };

    let door_pos = door_transform.translation;

    let hero_near = sim.sim.units.iter().any(|u| {
        if u.team != Team::Hero || u.hp <= 0 {
            return false;
        }
        let dx = u.position.x - door_pos.x;
        let dz = u.position.y - door_pos.z;
        (dx * dx + dz * dz).sqrt() < 3.0
    });

    if !hero_near {
        return;
    }

    // 1. Despawn current room geometry.
    for entity in floor_query.iter() {
        commands.entity(entity).despawn_recursive();
    }
    for entity in wall_query.iter() {
        commands.entity(entity).despawn_recursive();
    }
    for entity in obstacle_query.iter() {
        commands.entity(entity).despawn_recursive();
    }
    commands.entity(door_entity).despawn_recursive();

    // 2. Despawn dead enemy unit visuals.
    let dead_enemy_ids: std::collections::HashSet<u32> = sim
        .sim
        .units
        .iter()
        .filter(|u| u.team == Team::Enemy && u.hp <= 0)
        .map(|u| u.id)
        .collect();

    for (entity, vis) in unit_vis_query.iter() {
        if dead_enemy_ids.contains(&vis.sim_unit_id) {
            commands.entity(entity).despawn_recursive();
        }
    }

    sim.sim.units.retain(|u| {
        u.team == Team::Hero || (u.team == Team::Enemy && u.hp > 0)
    });

    // 3. Advance room index.
    seq.current_index += 1;

    // 4. Generate new room.
    let new_room_type = match seq.current_room_type() {
        Some(&rt) => rt,
        None => return,
    };
    let new_seed = seq.seed + seq.current_index as u64;
    let new_layout = generate_room(new_seed, new_room_type);

    seq.current_room_origin.z -= new_layout.depth + 5.0;
    let origin = seq.current_room_origin;

    spawn_room(&new_layout, &mut commands, &mut meshes, &mut materials);

    // 5. Build fresh enemy wave.
    let spawn_positions: Vec<SimVec2> = new_layout
        .enemy_spawn
        .positions
        .iter()
        .map(|&p| SimVec2 {
            x: p.x + origin.x,
            y: p.y + origin.z,
        })
        .collect();

    let boss_spawn_pos = if !spawn_positions.is_empty() {
        let cx = spawn_positions.iter().map(|p| p.x).sum::<f32>() / spawn_positions.len() as f32;
        let cy = spawn_positions.iter().map(|p| p.y).sum::<f32>() / spawn_positions.len() as f32;
        SimVec2 { x: cx, y: cy }
    } else {
        SimVec2 { x: origin.x + new_layout.width / 2.0, y: origin.z + new_layout.depth / 2.0 }
    };

    let new_enemies: Vec<crate::ai::core::UnitState> = if is_climax_room(&new_room_type) {
        let flashpoint_chain_id: u64 = 42;
        let difficulty = seq.current_index as u32;
        let mut boss = generate_boss(flashpoint_chain_id, difficulty);
        boss.position = boss_spawn_pos;
        vec![boss]
    } else {
        let id_offset: u32 = 2000 + seq.current_index as u32 * 100;
        let enemy_count = 4_usize;
        let mut wave = default_enemy_wave(enemy_count, new_seed, &spawn_positions);
        for (i, unit) in wave.iter_mut().enumerate() {
            unit.id = id_offset + i as u32;
        }
        wave
    };

    // 6. Spawn visuals for new enemies.
    for unit in &new_enemies {
        if unit.id == BOSS_UNIT_ID {
            spawn_boss_visual(unit.id, unit.position, &mut commands, &mut meshes, &mut materials);
        } else {
            spawn_unit_visual(unit.id, unit.team, unit.position, &mut commands, &mut meshes, &mut materials);
        }
    }

    // 7. Add new enemies to the sim.
    sim.sim.units.extend(new_enemies);

    // 8. Refresh AI state.
    sim.enemy_ai = EnemyAiState::new(&sim.sim);

    // 9. Reset outcome.
    sim.outcome = None;
}
