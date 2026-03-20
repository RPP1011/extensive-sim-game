use bevy::prelude::*;

use crate::game_core::{HubScreen, HubUiState, RoomType};
use crate::mission::{
    room_sequence::MissionRoomSequence,
    sim_bridge::{
        EnemyAiState, LastMissionReplay, MissionCombatRecording, MissionSimState,
        PlayerOrderState,
    },
    unit_vis::{UnitHealthData, UnitPositionData, UnitSelection},
};
use crate::scenario::{ScenarioCfg, build_combat, scenario_from_campaign};

// ---------------------------------------------------------------------------
// Context resource
// ---------------------------------------------------------------------------

/// Marker resource set when entering the mission scene, cleared on exit.
#[derive(Resource, Clone)]
pub struct ActiveMissionContext {
    pub room_type: RoomType,
    pub player_unit_count: usize,
    pub enemy_unit_count: usize,
    pub seed: u64,
    /// Difficulty level, used to determine the number of rooms in the sequence.
    pub difficulty: u32,
    /// The global campaign turn at the time this mission was started.
    pub global_turn: u32,
    /// Hero template names from the campaign roster (e.g. "knight", "mage").
    /// When non-empty, these are used instead of default hero units.
    pub hero_templates: Vec<String>,
    /// Pre-built scenario config. When `Some`, used directly instead of
    /// generating one from the other fields (e.g. flashpoints with custom composition).
    pub scenario_cfg: Option<ScenarioCfg>,
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
            hero_templates: Vec::new(),
            scenario_cfg: None,
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
    ctx_opt: Option<&ActiveMissionContext>,
) {
    let ctx = ctx_opt.cloned().unwrap_or_default();

    // Build or reuse a ScenarioCfg, then run it through build_combat()
    let cfg = if let Some(pre_built) = ctx.scenario_cfg.clone() {
        pre_built
    } else if !ctx.hero_templates.is_empty() {
        scenario_from_campaign(
            &ctx.hero_templates,
            ctx.difficulty,
            ctx.global_turn,
            ctx.room_type,
            ctx.seed,
            None,
        )
    } else {
        // Fallback: build a minimal ScenarioCfg from legacy fields
        ScenarioCfg {
            name: "mission".into(),
            seed: ctx.seed,
            hero_count: ctx.player_unit_count,
            enemy_count: ctx.enemy_unit_count,
            difficulty: ctx.difficulty,
            room_type: match ctx.room_type {
                RoomType::Entry => "Entry",
                RoomType::Pressure => "Pressure",
                RoomType::Pivot => "Pivot",
                RoomType::Setpiece => "Setpiece",
                RoomType::Recovery => "Recovery",
                RoomType::Climax => "Climax",
                RoomType::Open => "Open",
            }
            .to_string(),
            ..ScenarioCfg::default()
        }
    };

    let setup = build_combat(&cfg);

    let enemy_ai = EnemyAiState::new(&setup.sim);

    // Start recording combat frames for post-mission replay (clone initial frame before move)
    let initial_frame = setup.sim.clone();

    commands.insert_resource(MissionSimState {
        sim: setup.sim,
        tick_remainder_ms: 0,
        outcome: None,
        enemy_ai,
        hero_intents: Vec::new(),
        grid_nav: Some(setup.grid_nav),
    });
    commands.insert_resource(PlayerOrderState::default());
    commands.insert_resource(UnitSelection::default());
    commands.insert_resource(UnitHealthData::default());
    commands.insert_resource(UnitPositionData::default());
    commands.insert_resource(MissionRoomSequence::new(ctx.difficulty, ctx.seed));

    // Start recording combat frames for post-mission replay
    commands.insert_resource(MissionCombatRecording {
        frames: vec![initial_frame],
        events_per_frame: vec![Vec::new()],
        active: true,
    });
}

pub(crate) fn mission_exit(
    commands: &mut Commands,
    sim_state: Option<&MissionSimState>,
    recording: Option<&MissionCombatRecording>,
) {
    // Convert recording to replay for hub UI viewing
    if let Some(rec) = recording {
        if !rec.frames.is_empty() {
            commands.insert_resource(LastMissionReplay {
                name: "Last Mission".to_string(),
                frames: rec.frames.clone(),
                events_per_frame: rec.events_per_frame.clone(),
                grid_nav: sim_state.and_then(|s| s.grid_nav.clone()),
            });
        }
    }

    commands.remove_resource::<MissionSimState>();
    commands.remove_resource::<PlayerOrderState>();
    commands.remove_resource::<UnitSelection>();
    commands.remove_resource::<UnitHealthData>();
    commands.remove_resource::<UnitPositionData>();
    commands.remove_resource::<MissionRoomSequence>();
    commands.remove_resource::<MissionCombatRecording>();
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
    ctx_opt: Option<Res<ActiveMissionContext>>,
    sim_state: Option<Res<MissionSimState>>,
    recording: Option<Res<MissionCombatRecording>>,
) {
    let current = hub_ui.screen;
    let previous = *last_screen;

    if previous == Some(current) {
        return;
    }

    let entered_mission = current == HubScreen::MissionExecution;
    let exited_mission = previous == Some(HubScreen::MissionExecution);

    if exited_mission {
        mission_exit(
            &mut commands,
            sim_state.as_deref(),
            recording.as_deref(),
        );
    }

    if entered_mission {
        mission_enter(&mut commands, ctx_opt.as_deref());
    }

    *last_screen = Some(current);
}

// ---------------------------------------------------------------------------
// Replay viewer
// ---------------------------------------------------------------------------

/// Tracks replay viewer playback state.
#[derive(Resource)]
pub struct ReplayViewerState {
    pub frames: Vec<crate::ai::core::SimState>,
    pub events_per_frame: Vec<Vec<crate::ai::core::SimEvent>>,
    pub frame_index: usize,
    pub tick_seconds: f32,
    pub tick_accumulator: f32,
    pub paused: bool,
    pub previous_screen: HubScreen,
}

/// Watches for transitions into/out of `ReplayViewer` and spawns/despawns the
/// replay scene accordingly.
pub fn replay_viewer_transition_system(
    hub_ui: Res<HubUiState>,
    mut last_screen: Local<Option<HubScreen>>,
    mut commands: Commands,
    last_replay: Option<Res<LastMissionReplay>>,
) {
    let current = hub_ui.screen;
    let previous = *last_screen;

    if previous == Some(current) {
        return;
    }

    let entered_replay = current == HubScreen::ReplayViewer;
    let exited_replay = previous == Some(HubScreen::ReplayViewer);

    if exited_replay {
        commands.remove_resource::<ReplayViewerState>();
        commands.remove_resource::<UnitPositionData>();
        commands.remove_resource::<UnitHealthData>();
    }

    if entered_replay {
        if let Some(replay) = last_replay.as_ref() {
            if !replay.frames.is_empty() {
                let prev = previous.unwrap_or(HubScreen::GuildManagement);
                commands.insert_resource(ReplayViewerState {
                    frames: replay.frames.clone(),
                    events_per_frame: replay.events_per_frame.clone(),
                    frame_index: 0,
                    tick_seconds: 0.1,
                    tick_accumulator: 0.0,
                    paused: false,
                    previous_screen: prev,
                });
                commands.insert_resource(UnitPositionData::default());
                commands.insert_resource(UnitHealthData::default());
            }
        }
    }

    *last_screen = Some(current);
}

/// Advances the replay viewer frame-by-frame.
pub fn advance_replay_viewer_system(
    time: Res<Time>,
    viewer: Option<ResMut<ReplayViewerState>>,
) {
    let Some(mut viewer) = viewer else {
        return;
    };
    if viewer.paused {
        return;
    }
    if viewer.frame_index + 1 >= viewer.frames.len() {
        return;
    }

    viewer.tick_accumulator += time.delta_seconds();
    while viewer.tick_accumulator >= viewer.tick_seconds
        && viewer.frame_index + 1 < viewer.frames.len()
    {
        viewer.tick_accumulator -= viewer.tick_seconds;
        viewer.frame_index += 1;
    }
}

/// Keyboard controls for the replay viewer.
pub fn replay_viewer_keyboard_system(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mut hub_ui: ResMut<HubUiState>,
    mut viewer: Option<ResMut<ReplayViewerState>>,
) {
    let Some(keyboard) = keyboard else { return };
    let Some(ref mut viewer) = viewer else { return };

    if keyboard.just_pressed(KeyCode::Escape) {
        hub_ui.screen = viewer.previous_screen;
        return;
    }
    if keyboard.just_pressed(KeyCode::Space) {
        viewer.paused = !viewer.paused;
    }
    if keyboard.just_pressed(KeyCode::ArrowLeft) {
        viewer.frame_index = viewer.frame_index.saturating_sub(1);
        viewer.tick_accumulator = 0.0;
        viewer.paused = true;
    }
    if keyboard.just_pressed(KeyCode::ArrowRight) {
        viewer.frame_index = (viewer.frame_index + 1).min(viewer.frames.len().saturating_sub(1));
        viewer.tick_accumulator = 0.0;
        viewer.paused = true;
    }
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
