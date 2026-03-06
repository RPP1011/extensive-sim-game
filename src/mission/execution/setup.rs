use bevy::prelude::*;

use crate::ai::core::Team;
use crate::game_core::{HubScreen, HubUiState, RoomType};
use crate::mission::{
    enemy_templates::{generate_boss, is_climax_room, BOSS_UNIT_ID},
    room_gen::{generate_room, spawn_room, RoomFloor, RoomObstacle, RoomWall},
    room_sequence::MissionRoomSequence,
    sim_bridge::{
        build_default_sim, EnemyAiState, MissionSimState, PlayerOrderState, PlayerUnitMarker,
    },
    unit_vis::{spawn_unit_visual, UnitHealthData, UnitPositionData, UnitSelection, UnitVisual},
};

// ---------------------------------------------------------------------------
// Context resource
// ---------------------------------------------------------------------------

/// Marker resource set when entering the mission scene, cleared on exit.
#[derive(Resource)]
pub struct ActiveMissionContext {
    pub room_type: RoomType,
    pub player_unit_count: usize,
    pub enemy_unit_count: usize,
    pub seed: u64,
    /// Difficulty level, used to determine the number of rooms in the sequence.
    pub difficulty: u32,
    /// The global campaign turn at the time this mission was started.
    pub global_turn: u32,
}

impl Default for ActiveMissionContext {
    fn default() -> Self {
        Self {
            room_type: RoomType::Entry,
            player_unit_count: 4,
            enemy_unit_count: 4,
            seed: 42,
            difficulty: 2,
            global_turn: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Boss visual spawning
// ---------------------------------------------------------------------------

/// Spawns a gold/yellow 1.5x-scale visual for the climax-room boss.
fn spawn_boss_visual_execution(
    sim_unit_id: u32,
    position: crate::ai::core::SimVec2,
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) -> Entity {
    use crate::ai::core::Team;
    use crate::mission::unit_vis::{HpBarBg, HpBarFg, UnitVisual};
    use bevy::prelude::Capsule3d;

    let world_pos = Vec3::new(position.x, 0.0, position.y);

    let body_material = materials.add(StandardMaterial {
        base_color: Color::rgb(1.0, 0.84, 0.0),
        emissive: Color::rgb(0.6, 0.4, 0.0),
        metallic: 0.3,
        perceptual_roughness: 0.5,
        ..default()
    });
    let body_mesh = meshes.add(Capsule3d {
        radius: 0.3,
        half_length: 0.5,
    });

    let bar_bg_mesh = meshes.add(Cuboid::new(0.8, 0.08, 0.08));
    let bar_bg_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.2, 0.2, 0.2),
        ..default()
    });

    let bar_fg_mesh = meshes.add(Cuboid::new(0.8, 0.08, 0.08));
    let bar_fg_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.1, 0.9, 0.1),
        ..default()
    });

    commands
        .spawn((
            SpatialBundle {
                transform: Transform {
                    translation: world_pos,
                    scale: Vec3::splat(1.5),
                    ..default()
                },
                ..default()
            },
            UnitVisual { sim_unit_id, team: Team::Enemy },
            Name::new("Boss"),
        ))
        .with_children(|parent| {
            parent.spawn(PbrBundle {
                mesh: body_mesh,
                material: body_material,
                transform: Transform::from_xyz(0.0, 0.8, 0.0),
                ..default()
            });
            parent.spawn((
                PbrBundle {
                    mesh: bar_bg_mesh,
                    material: bar_bg_material,
                    transform: Transform::from_xyz(0.0, 2.0, 0.0),
                    ..default()
                },
                HpBarBg,
            ));
            parent.spawn((
                PbrBundle {
                    mesh: bar_fg_mesh,
                    material: bar_fg_material,
                    transform: Transform::from_xyz(0.0, 2.0, 0.01),
                    ..default()
                },
                HpBarFg,
            ));
        })
        .id()
}

// ---------------------------------------------------------------------------
// Setup / teardown
// ---------------------------------------------------------------------------

pub(crate) fn mission_enter(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    ctx_opt: Option<&ActiveMissionContext>,
    cameras: &mut Query<(&mut crate::camera::OrbitCameraController, &mut Transform)>,
) {
    let (room_type, player_count, enemy_count, seed, difficulty) = match ctx_opt {
        Some(ctx) => (ctx.room_type, ctx.player_unit_count, ctx.enemy_unit_count, ctx.seed, ctx.difficulty),
        None => (RoomType::Entry, 4, 4, 42u64, 2u32),
    };

    let layout = generate_room(seed, room_type);
    spawn_room(&layout, commands, meshes, materials);

    let mut sim = if is_climax_room(&room_type) {
        use crate::mission::sim_bridge::build_sim_with_templates;
        let mut boss = generate_boss(42u64, difficulty);
        let boss_pos = if !layout.enemy_spawn.positions.is_empty() {
            let cx = layout.enemy_spawn.positions.iter().map(|p| p.x).sum::<f32>()
                / layout.enemy_spawn.positions.len() as f32;
            let cy = layout.enemy_spawn.positions.iter().map(|p| p.y).sum::<f32>()
                / layout.enemy_spawn.positions.len() as f32;
            crate::ai::core::SimVec2 { x: cx, y: cy }
        } else {
            crate::ai::core::SimVec2 { x: layout.width / 2.0, y: layout.depth / 2.0 }
        };
        boss.position = boss_pos;
        build_sim_with_templates(player_count, vec![boss], seed)
    } else {
        build_default_sim(player_count, enemy_count, seed)
    };
    let _ = &mut sim;
    let enemy_ai = EnemyAiState::new(&sim);

    for unit in &sim.units {
        let entity = if unit.id == BOSS_UNIT_ID {
            spawn_boss_visual_execution(unit.id, unit.position, commands, meshes, materials)
        } else {
            spawn_unit_visual(unit.id, unit.team, unit.position, commands, meshes, materials)
        };
        if unit.team == Team::Hero {
            commands.entity(entity).insert(PlayerUnitMarker { sim_unit_id: unit.id });
        }
    }

    let grid_nav = layout.nav.to_gridnav();
    commands.insert_resource(MissionSimState {
        sim,
        tick_remainder_ms: 0,
        outcome: None,
        enemy_ai,
        hero_intents: Vec::new(),
        grid_nav: Some(grid_nav),
    });
    commands.insert_resource(PlayerOrderState::default());
    commands.insert_resource(UnitSelection::default());
    commands.insert_resource(UnitHealthData::default());
    commands.insert_resource(UnitPositionData::default());
    commands.insert_resource(MissionRoomSequence::new(difficulty, seed));

    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 0.4,
    });
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::WHITE,
            illuminance: 18_000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_4,
            0.0,
        )),
        ..default()
    });

    let cx = layout.width * 0.5;
    let cz = layout.depth * 0.5;
    for (mut controller, mut transform) in cameras.iter_mut() {
        controller.focus = Vec3::new(cx, 0.0, cz);
        controller.radius = 28.0;
        controller.yaw = 0.0;
        controller.pitch = 1.0;
        let horizontal = controller.radius * controller.pitch.cos();
        let offset = Vec3::new(
            horizontal * controller.yaw.sin(),
            controller.radius * controller.pitch.sin(),
            horizontal * controller.yaw.cos(),
        );
        transform.translation = controller.focus + offset;
        transform.look_at(controller.focus, Vec3::Y);
    }
}

pub(crate) fn mission_exit(
    commands: &mut Commands,
    floor_query: &Query<Entity, With<RoomFloor>>,
    wall_query: &Query<Entity, With<RoomWall>>,
    obstacle_query: &Query<Entity, With<RoomObstacle>>,
    unit_query: &Query<Entity, With<UnitVisual>>,
) {
    for entity in floor_query.iter() {
        commands.entity(entity).despawn_recursive();
    }
    for entity in wall_query.iter() {
        commands.entity(entity).despawn_recursive();
    }
    for entity in obstacle_query.iter() {
        commands.entity(entity).despawn_recursive();
    }
    for entity in unit_query.iter() {
        commands.entity(entity).despawn_recursive();
    }

    commands.remove_resource::<MissionSimState>();
    commands.remove_resource::<PlayerOrderState>();
    commands.remove_resource::<UnitSelection>();
    commands.remove_resource::<UnitHealthData>();
    commands.remove_resource::<UnitPositionData>();
    commands.remove_resource::<MissionRoomSequence>();
}

// ---------------------------------------------------------------------------
// Transition watcher
// ---------------------------------------------------------------------------

/// Watches `hub_ui.screen` for transitions into and out of `MissionExecution`,
/// then calls mission_enter / mission_exit accordingly.
pub fn mission_scene_transition_system(
    hub_ui: Res<HubUiState>,
    mut last_screen: Local<Option<HubScreen>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    ctx_opt: Option<Res<ActiveMissionContext>>,
    mut cameras: Query<(&mut crate::camera::OrbitCameraController, &mut Transform)>,
    floor_query: Query<Entity, With<RoomFloor>>,
    wall_query: Query<Entity, With<RoomWall>>,
    obstacle_query: Query<Entity, With<RoomObstacle>>,
    unit_query: Query<Entity, With<UnitVisual>>,
) {
    let current = hub_ui.screen;
    let previous = *last_screen;

    if previous == Some(current) {
        return;
    }

    let entered_mission = current == HubScreen::MissionExecution;
    let exited_mission = previous == Some(HubScreen::MissionExecution);

    if exited_mission {
        mission_exit(&mut commands, &floor_query, &wall_query, &obstacle_query, &unit_query);
    }

    if entered_mission {
        mission_enter(&mut commands, &mut meshes, &mut materials, ctx_opt.as_deref(), &mut cameras);
    }

    *last_screen = Some(current);
}

/// Bridges sim state into visual data resources each frame.
pub fn sync_sim_to_visuals_system(
    sim_state: Option<Res<MissionSimState>>,
    pos_data: Option<ResMut<UnitPositionData>>,
    hp_data: Option<ResMut<UnitHealthData>>,
) {
    let (Some(sim), Some(mut pos_data), Some(mut hp_data)) = (sim_state, pos_data, hp_data) else {
        return;
    };

    for unit in &sim.sim.units {
        pos_data.positions.insert(unit.id, (unit.position.x, unit.position.y));
        hp_data.hp.insert(unit.id, (unit.hp, unit.max_hp));
    }
}
