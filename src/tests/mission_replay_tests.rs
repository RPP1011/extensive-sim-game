use super::*;
use std::panic::{self, AssertUnwindSafe};

use bevy::prelude::*;
use bevy::time::Time;

use crate::ai::core::{self, SimState, Team, FIXED_TICK_MS};
use crate::game_core::{HubScreen, HubUiState, RoomType};
use crate::mission::execution::ActiveMissionContext;
use crate::mission::sim_bridge::{
    advance_sim_system, EnemyAiState, LastMissionReplay, MissionCombatRecording,
    MissionSimState, PlayerOrderState, SimEventBuffer,
};
use crate::mission::unit_vis::{UnitHealthData, UnitPositionData};
use crate::mission::execution::{
    replay_viewer_transition_system, advance_replay_viewer_system,
    replay_viewer_keyboard_system,
};
use crate::mission::execution::setup::ReplayViewerState;
use crate::scenario::{build_combat, scenario_from_campaign, ScenarioCfg};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal SimState for testing (no file I/O, no templates).
fn test_sim(seed: u64) -> SimState {
    let cfg = ScenarioCfg {
        name: "test".into(),
        seed,
        hero_count: 2,
        enemy_count: 2,
        ..ScenarioCfg::default()
    };
    build_combat(&cfg).sim
}

/// Create a MissionSimState resource with a test sim.
fn test_mission_sim(seed: u64) -> MissionSimState {
    let sim = test_sim(seed);
    let enemy_ai = EnemyAiState::new(&sim);
    MissionSimState {
        sim,
        tick_remainder_ms: 0,
        outcome: None,
        enemy_ai,
        hero_intents: Vec::new(),
        grid_nav: None,
    }
}

/// Build a LastMissionReplay from a short headless combat run.
fn build_test_replay(ticks: usize) -> LastMissionReplay {
    let cfg = ScenarioCfg {
        name: "replay_test".into(),
        seed: 42,
        hero_count: 2,
        enemy_count: 2,
        ..ScenarioCfg::default()
    };
    let setup = build_combat(&cfg);
    let mut sim = setup.sim;
    let mut squad_ai = setup.squad_ai;

    let mut frames = vec![sim.clone()];
    let mut events_per_frame = vec![Vec::new()];

    for _ in 0..ticks {
        let intents = crate::ai::squad::generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
        let (next, events) = core::step(sim, &intents, FIXED_TICK_MS);
        sim = next;
        frames.push(sim.clone());
        events_per_frame.push(events);
    }

    LastMissionReplay {
        name: "Test Replay".to_string(),
        frames,
        events_per_frame,
        grid_nav: None,
    }
}

// ===========================================================================
// Test: advance_sim_system records frames into MissionCombatRecording
// ===========================================================================

#[test]
fn advance_sim_system_records_frames() {
    let mut app = App::new();
    app.insert_resource(Time::<()>::default());

    let sim_state = test_mission_sim(42);
    let initial_units = sim_state.sim.units.len();
    app.insert_resource(sim_state);
    app.insert_resource(UnitPositionData::default());
    app.insert_resource(UnitHealthData::default());
    app.insert_resource(SimEventBuffer::default());
    app.insert_resource(MissionCombatRecording {
        frames: vec![],
        events_per_frame: vec![],
        active: true,
    });

    app.add_systems(Update, advance_sim_system);

    // Manually advance time so the system processes at least one fixed tick
    {
        let mut time = app.world.resource_mut::<Time<()>>();
        // Advance by 200ms — should trigger 2 fixed ticks (100ms each)
        time.advance_by(std::time::Duration::from_millis(200));
    }
    app.update();

    let recording = app.world.resource::<MissionCombatRecording>();
    assert!(
        recording.frames.len() >= 2,
        "should have recorded at least 2 frames, got {}",
        recording.frames.len(),
    );
    assert_eq!(
        recording.frames.len(),
        recording.events_per_frame.len(),
        "frames and events_per_frame should have same length",
    );

    // Verify recorded frames have the right unit count
    for frame in &recording.frames {
        assert_eq!(frame.units.len(), initial_units);
    }
}

#[test]
fn advance_sim_skips_recording_when_inactive() {
    let mut app = App::new();
    app.insert_resource(Time::<()>::default());
    app.insert_resource(test_mission_sim(42));
    app.insert_resource(UnitPositionData::default());
    app.insert_resource(UnitHealthData::default());
    app.insert_resource(SimEventBuffer::default());
    app.insert_resource(MissionCombatRecording {
        frames: vec![],
        events_per_frame: vec![],
        active: false, // recording disabled
    });

    app.add_systems(Update, advance_sim_system);

    {
        let mut time = app.world.resource_mut::<Time<()>>();
        time.advance_by(std::time::Duration::from_millis(200));
    }
    app.update();

    let recording = app.world.resource::<MissionCombatRecording>();
    assert_eq!(
        recording.frames.len(),
        0,
        "should not record when active=false",
    );
}

#[test]
fn advance_sim_works_without_recording_resource() {
    // Ensure the system doesn't panic when no MissionCombatRecording is present
    let mut app = App::new();
    app.insert_resource(Time::<()>::default());
    app.insert_resource(test_mission_sim(42));
    app.insert_resource(UnitPositionData::default());
    app.insert_resource(UnitHealthData::default());
    app.insert_resource(SimEventBuffer::default());
    // No MissionCombatRecording inserted

    app.add_systems(Update, advance_sim_system);

    {
        let mut time = app.world.resource_mut::<Time<()>>();
        time.advance_by(std::time::Duration::from_millis(200));
    }

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        app.update();
    }));
    assert!(result.is_ok(), "advance_sim_system panicked without recording resource");
}

// ===========================================================================
// Test: MissionCombatRecording → LastMissionReplay conversion
// ===========================================================================

#[test]
fn recording_converts_to_last_replay_on_mission_exit() {
    // We can't easily call mission_exit as a Bevy system (needs queries),
    // so test the conversion logic directly: simulate what mission_exit does.
    let mut app = App::new();

    // Run a few ticks of sim to accumulate frames
    app.insert_resource(Time::<()>::default());
    let sim_state = test_mission_sim(42);
    app.insert_resource(sim_state);
    app.insert_resource(UnitPositionData::default());
    app.insert_resource(UnitHealthData::default());
    app.insert_resource(SimEventBuffer::default());
    app.insert_resource(MissionCombatRecording {
        frames: vec![],
        events_per_frame: vec![],
        active: true,
    });
    app.add_systems(Update, advance_sim_system);

    // Run several frames
    for _ in 0..5 {
        let mut time = app.world.resource_mut::<Time<()>>();
        time.advance_by(std::time::Duration::from_millis(100));
        drop(time);
        app.update();
    }

    let recording = app.world.resource::<MissionCombatRecording>();
    let frame_count = recording.frames.len();
    assert!(frame_count >= 5, "should have at least 5 recorded frames");

    // Simulate what mission_exit does: convert recording to LastMissionReplay
    let replay = LastMissionReplay {
        name: "Last Mission".to_string(),
        frames: recording.frames.clone(),
        events_per_frame: recording.events_per_frame.clone(),
        grid_nav: None,
    };
    app.insert_resource(replay);

    let replay = app.world.resource::<LastMissionReplay>();
    assert_eq!(replay.frames.len(), frame_count);
    assert_eq!(replay.events_per_frame.len(), frame_count);
    assert_eq!(replay.name, "Last Mission");
}

// ===========================================================================
// Test: Replay viewer transition system lifecycle
// ===========================================================================

#[test]
fn replay_viewer_creates_state_on_enter() {
    let mut app = App::new();

    // Insert LastMissionReplay
    let replay = build_test_replay(20);
    let frame_count = replay.frames.len();
    app.insert_resource(replay);

    // Start at GuildManagement, then transition to ReplayViewer
    app.insert_resource(HubUiState {
        screen: HubScreen::GuildManagement,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    });
    app.init_resource::<Assets<Mesh>>();
    app.init_resource::<Assets<StandardMaterial>>();
    app.add_systems(Update, replay_viewer_transition_system);

    // First update: establishes GuildManagement as last_screen
    app.update();
    assert!(
        app.world.get_resource::<ReplayViewerState>().is_none(),
        "should not have ReplayViewerState yet",
    );

    // Transition to ReplayViewer
    app.world.resource_mut::<HubUiState>().screen = HubScreen::ReplayViewer;
    app.update();

    let viewer = app
        .world
        .get_resource::<ReplayViewerState>()
        .expect("ReplayViewerState should exist after entering ReplayViewer");
    assert_eq!(viewer.frame_index, 0);
    assert_eq!(viewer.frames.len(), frame_count);
    assert!(!viewer.paused);
    assert_eq!(viewer.previous_screen, HubScreen::GuildManagement);

    // Should have position/health data resources
    assert!(app.world.get_resource::<UnitPositionData>().is_some());
    assert!(app.world.get_resource::<UnitHealthData>().is_some());
}

#[test]
fn replay_viewer_cleans_up_on_exit() {
    let mut app = App::new();

    let replay = build_test_replay(10);
    app.insert_resource(replay);
    app.insert_resource(HubUiState {
        screen: HubScreen::GuildManagement,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    });
    app.init_resource::<Assets<Mesh>>();
    app.init_resource::<Assets<StandardMaterial>>();
    app.add_systems(Update, replay_viewer_transition_system);

    // Establish initial screen
    app.update();

    // Enter replay viewer
    app.world.resource_mut::<HubUiState>().screen = HubScreen::ReplayViewer;
    app.update();
    assert!(app.world.get_resource::<ReplayViewerState>().is_some());

    // Exit replay viewer
    app.world.resource_mut::<HubUiState>().screen = HubScreen::GuildManagement;
    app.update();

    assert!(
        app.world.get_resource::<ReplayViewerState>().is_none(),
        "ReplayViewerState should be removed on exit",
    );
}

#[test]
fn replay_viewer_no_op_without_replay_data() {
    let mut app = App::new();

    // No LastMissionReplay inserted
    app.insert_resource(HubUiState {
        screen: HubScreen::GuildManagement,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    });
    app.init_resource::<Assets<Mesh>>();
    app.init_resource::<Assets<StandardMaterial>>();
    app.add_systems(Update, replay_viewer_transition_system);

    app.update();
    app.world.resource_mut::<HubUiState>().screen = HubScreen::ReplayViewer;
    app.update();

    // Should not panic, and should not create ReplayViewerState
    assert!(
        app.world.get_resource::<ReplayViewerState>().is_none(),
        "should not create ReplayViewerState without replay data",
    );
}

// ===========================================================================
// Test: Replay viewer advance system
// ===========================================================================

#[test]
fn replay_viewer_advances_frames() {
    let mut app = App::new();
    app.insert_resource(Time::<()>::default());

    let replay = build_test_replay(50);
    let total_frames = replay.frames.len();

    app.insert_resource(ReplayViewerState {
        frames: replay.frames,
        events_per_frame: replay.events_per_frame,
        frame_index: 0,
        tick_seconds: 0.1,
        tick_accumulator: 0.0,
        paused: false,
        previous_screen: HubScreen::GuildManagement,
    });

    app.add_systems(Update, advance_replay_viewer_system);

    // Advance time by 500ms = 5 frames at 100ms tick_seconds
    {
        let mut time = app.world.resource_mut::<Time<()>>();
        time.advance_by(std::time::Duration::from_millis(500));
    }
    app.update();

    let viewer = app.world.resource::<ReplayViewerState>();
    assert!(
        viewer.frame_index >= 4,
        "should have advanced at least 4 frames, got {}",
        viewer.frame_index,
    );
    assert!(viewer.frame_index < total_frames);
}

#[test]
fn replay_viewer_paused_does_not_advance() {
    let mut app = App::new();
    app.insert_resource(Time::<()>::default());

    let replay = build_test_replay(20);

    app.insert_resource(ReplayViewerState {
        frames: replay.frames,
        events_per_frame: replay.events_per_frame,
        frame_index: 5,
        tick_seconds: 0.1,
        tick_accumulator: 0.0,
        paused: true, // paused
        previous_screen: HubScreen::GuildManagement,
    });
    app.insert_resource(UnitPositionData::default());
    app.insert_resource(UnitHealthData::default());
    app.init_resource::<Assets<Mesh>>();
    app.init_resource::<Assets<StandardMaterial>>();

    app.add_systems(Update, advance_replay_viewer_system);

    {
        let mut time = app.world.resource_mut::<Time<()>>();
        time.advance_by(std::time::Duration::from_millis(500));
    }
    app.update();

    let viewer = app.world.resource::<ReplayViewerState>();
    assert_eq!(viewer.frame_index, 5, "paused viewer should not advance");
}

#[test]
fn replay_viewer_stops_at_last_frame() {
    let mut app = App::new();
    app.insert_resource(Time::<()>::default());

    let replay = build_test_replay(10);
    let last_frame = replay.frames.len() - 1;

    app.insert_resource(ReplayViewerState {
        frames: replay.frames,
        events_per_frame: replay.events_per_frame,
        frame_index: last_frame, // already at end
        tick_seconds: 0.1,
        tick_accumulator: 0.0,
        paused: false,
        previous_screen: HubScreen::GuildManagement,
    });
    app.insert_resource(UnitPositionData::default());
    app.insert_resource(UnitHealthData::default());
    app.init_resource::<Assets<Mesh>>();
    app.init_resource::<Assets<StandardMaterial>>();

    app.add_systems(Update, advance_replay_viewer_system);

    {
        let mut time = app.world.resource_mut::<Time<()>>();
        time.advance_by(std::time::Duration::from_millis(500));
    }
    app.update();

    let viewer = app.world.resource::<ReplayViewerState>();
    assert_eq!(viewer.frame_index, last_frame, "should stay at last frame");
}

// ===========================================================================
// Test: Replay viewer keyboard controls
// ===========================================================================

#[test]
fn replay_keyboard_space_toggles_pause() {
    let mut app = App::new();

    let replay = build_test_replay(20);
    app.insert_resource(ReplayViewerState {
        frames: replay.frames,
        events_per_frame: replay.events_per_frame,
        frame_index: 5,
        tick_seconds: 0.1,
        tick_accumulator: 0.0,
        paused: false,
        previous_screen: HubScreen::GuildManagement,
    });
    app.insert_resource(HubUiState {
        screen: HubScreen::ReplayViewer,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    });

    let mut keyboard = ButtonInput::<KeyCode>::default();
    keyboard.press(KeyCode::Space);
    app.insert_resource(keyboard);

    app.add_systems(Update, replay_viewer_keyboard_system);
    app.update();

    let viewer = app.world.resource::<ReplayViewerState>();
    assert!(viewer.paused, "space should pause");
}

#[test]
fn replay_keyboard_arrows_step_frames() {
    let mut app = App::new();

    let replay = build_test_replay(20);
    app.insert_resource(ReplayViewerState {
        frames: replay.frames,
        events_per_frame: replay.events_per_frame,
        frame_index: 10,
        tick_seconds: 0.1,
        tick_accumulator: 0.0,
        paused: false,
        previous_screen: HubScreen::GuildManagement,
    });
    app.insert_resource(HubUiState {
        screen: HubScreen::ReplayViewer,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    });

    // Press left arrow
    let mut keyboard = ButtonInput::<KeyCode>::default();
    keyboard.press(KeyCode::ArrowLeft);
    app.insert_resource(keyboard);

    app.add_systems(Update, replay_viewer_keyboard_system);
    app.update();

    let viewer = app.world.resource::<ReplayViewerState>();
    assert_eq!(viewer.frame_index, 9, "left arrow should step back");
    assert!(viewer.paused, "stepping should pause");
}

#[test]
fn replay_keyboard_escape_returns_to_previous_screen() {
    let mut app = App::new();

    let replay = build_test_replay(10);
    app.insert_resource(ReplayViewerState {
        frames: replay.frames,
        events_per_frame: replay.events_per_frame,
        frame_index: 5,
        tick_seconds: 0.1,
        tick_accumulator: 0.0,
        paused: false,
        previous_screen: HubScreen::OverworldMap,
    });
    app.insert_resource(HubUiState {
        screen: HubScreen::ReplayViewer,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    });

    let mut keyboard = ButtonInput::<KeyCode>::default();
    keyboard.press(KeyCode::Escape);
    app.insert_resource(keyboard);

    app.add_systems(Update, replay_viewer_keyboard_system);
    app.update();

    let hub_ui = app.world.resource::<HubUiState>();
    assert_eq!(
        hub_ui.screen,
        HubScreen::OverworldMap,
        "Escape should return to previous screen",
    );
}

// ===========================================================================
// Test: Mission scene transition with ActiveMissionContext
// ===========================================================================

#[test]
fn mission_transition_creates_sim_with_hero_templates() {
    let mut app = App::new();
    app.insert_resource(Time::<()>::default());

    // Insert context with hero templates
    app.insert_resource(ActiveMissionContext {
        hero_templates: vec!["knight".into(), "mage".into()],
        difficulty: 2,
        global_turn: 5,
        seed: 42,
        ..Default::default()
    });

    app.insert_resource(HubUiState {
        screen: HubScreen::GuildManagement,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    });

    app.init_resource::<Assets<Mesh>>();
    app.init_resource::<Assets<StandardMaterial>>();

    app.add_systems(Update, crate::mission::execution::mission_scene_transition_system);

    // Establish initial screen
    app.update();

    // Transition to mission
    app.world.resource_mut::<HubUiState>().screen = HubScreen::MissionExecution;
    app.update();

    // Verify resources were created
    let sim_state = app
        .world
        .get_resource::<MissionSimState>()
        .expect("MissionSimState should exist");

    let heroes: Vec<_> = sim_state
        .sim
        .units
        .iter()
        .filter(|u| u.team == Team::Hero)
        .collect();
    assert_eq!(heroes.len(), 2, "should have 2 heroes from templates");

    let enemies: Vec<_> = sim_state
        .sim
        .units
        .iter()
        .filter(|u| u.team == Team::Enemy)
        .collect();
    assert!(enemies.len() >= 2, "should have enemies from campaign scaling");

    // Verify recording was started
    let recording = app
        .world
        .get_resource::<MissionCombatRecording>()
        .expect("recording should exist");
    assert!(recording.active);
    assert_eq!(recording.frames.len(), 1, "should have initial frame");
}

#[test]
fn mission_transition_with_prebuilt_scenario() {
    let mut app = App::new();
    app.insert_resource(Time::<()>::default());

    let cfg = ScenarioCfg {
        name: "custom_flashpoint".into(),
        seed: 99,
        hero_templates: vec!["paladin".into()],
        enemy_hero_templates: vec!["assassin".into(), "berserker".into()],
        ..ScenarioCfg::default()
    };

    app.insert_resource(ActiveMissionContext {
        scenario_cfg: Some(cfg),
        ..Default::default()
    });

    app.insert_resource(HubUiState {
        screen: HubScreen::GuildManagement,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    });

    app.init_resource::<Assets<Mesh>>();
    app.init_resource::<Assets<StandardMaterial>>();

    app.add_systems(Update, crate::mission::execution::mission_scene_transition_system);

    app.update();
    app.world.resource_mut::<HubUiState>().screen = HubScreen::MissionExecution;
    app.update();

    let sim_state = app.world.resource::<MissionSimState>();
    let heroes = sim_state
        .sim
        .units
        .iter()
        .filter(|u| u.team == Team::Hero)
        .count();
    let enemies = sim_state
        .sim
        .units
        .iter()
        .filter(|u| u.team == Team::Enemy)
        .count();
    assert_eq!(heroes, 1, "should have 1 hero (paladin)");
    assert_eq!(enemies, 2, "should have 2 enemies (assassin + berserker)");
}

#[test]
fn mission_transition_default_context_creates_valid_sim() {
    // Test the fallback path: no hero_templates, no scenario_cfg
    let mut app = App::new();
    app.insert_resource(Time::<()>::default());

    // No ActiveMissionContext — will use defaults
    app.insert_resource(HubUiState {
        screen: HubScreen::GuildManagement,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    });

    app.init_resource::<Assets<Mesh>>();
    app.init_resource::<Assets<StandardMaterial>>();

    app.add_systems(Update, crate::mission::execution::mission_scene_transition_system);

    app.update();
    app.world.resource_mut::<HubUiState>().screen = HubScreen::MissionExecution;
    app.update();

    let sim_state = app
        .world
        .get_resource::<MissionSimState>()
        .expect("should create sim state with defaults");
    assert!(
        sim_state.sim.units.len() >= 4,
        "default context should create at least 4 units",
    );
}

// ===========================================================================
// Test: Full mission → recording → replay lifecycle
// ===========================================================================

#[test]
fn full_mission_to_replay_lifecycle() {
    let mut app = App::new();
    app.insert_resource(Time::<()>::default());

    // --- Phase 1: Enter mission ---
    app.insert_resource(ActiveMissionContext {
        hero_templates: vec!["knight".into()],
        difficulty: 1,
        seed: 42,
        ..Default::default()
    });

    app.insert_resource(HubUiState {
        screen: HubScreen::GuildManagement,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    });

    app.init_resource::<Assets<Mesh>>();
    app.init_resource::<Assets<StandardMaterial>>();
    app.insert_resource(SimEventBuffer::default());

    // Transition systems must run before sim/replay systems so resources
    // inserted via commands are available on the NEXT frame.
    app.add_systems(Update, (
        crate::mission::execution::mission_scene_transition_system,
        replay_viewer_transition_system,
    ));
    app.add_systems(
        Update,
        advance_sim_system
            .after(crate::mission::execution::mission_scene_transition_system)
            .run_if(|sim: Option<Res<MissionSimState>>| sim.is_some()),
    );
    app.add_systems(
        Update,
        advance_replay_viewer_system
            .after(replay_viewer_transition_system)
            .run_if(|v: Option<Res<ReplayViewerState>>| v.is_some()),
    );

    // Establish initial screen
    app.update();

    // Enter mission
    app.world.resource_mut::<HubUiState>().screen = HubScreen::MissionExecution;
    app.update();

    assert!(app.world.get_resource::<MissionSimState>().is_some());
    assert!(app.world.get_resource::<MissionCombatRecording>().is_some());

    // --- Phase 2: Run a few sim ticks ---
    for _ in 0..10 {
        let mut time = app.world.resource_mut::<Time<()>>();
        time.advance_by(std::time::Duration::from_millis(100));
        drop(time);
        app.update();
    }

    let recording = app.world.resource::<MissionCombatRecording>();
    let recorded_frames = recording.frames.len();
    assert!(
        recorded_frames >= 10,
        "should have recorded at least 10 frames, got {}",
        recorded_frames,
    );

    // --- Phase 3: Exit mission → creates LastMissionReplay ---
    app.world.resource_mut::<HubUiState>().screen = HubScreen::GuildManagement;
    app.update();

    assert!(
        app.world.get_resource::<MissionSimState>().is_none(),
        "MissionSimState should be removed after exit",
    );
    assert!(
        app.world.get_resource::<MissionCombatRecording>().is_none(),
        "MissionCombatRecording should be removed after exit",
    );

    let last_replay = app
        .world
        .get_resource::<LastMissionReplay>()
        .expect("LastMissionReplay should be created on exit");
    assert_eq!(last_replay.frames.len(), recorded_frames);

    // --- Phase 4: Enter replay viewer ---
    app.world.resource_mut::<HubUiState>().screen = HubScreen::ReplayViewer;
    app.update();

    let viewer = app
        .world
        .get_resource::<ReplayViewerState>()
        .expect("ReplayViewerState should exist");
    assert_eq!(viewer.frames.len(), recorded_frames);
    // frame_index may have advanced if time accumulated from sim phase
    assert!(viewer.frame_index < recorded_frames);
    assert_eq!(viewer.previous_screen, HubScreen::GuildManagement);

    // --- Phase 5: Advance replay ---
    {
        let mut time = app.world.resource_mut::<Time<()>>();
        time.advance_by(std::time::Duration::from_millis(500));
    }
    app.update();

    let viewer = app.world.resource::<ReplayViewerState>();
    assert!(
        viewer.frame_index > 0,
        "replay should have advanced",
    );

    // --- Phase 6: Exit replay viewer ---
    app.world.resource_mut::<HubUiState>().screen = HubScreen::GuildManagement;
    app.update();

    assert!(
        app.world.get_resource::<ReplayViewerState>().is_none(),
        "ReplayViewerState should be cleaned up",
    );

    // LastMissionReplay should persist (can be viewed again)
    assert!(
        app.world.get_resource::<LastMissionReplay>().is_some(),
        "LastMissionReplay should persist after replay viewer exit",
    );
}

// ===========================================================================
// Test: scenario_from_campaign interface (what the UI calls)
// ===========================================================================

#[test]
fn scenario_from_campaign_interface_contract() {
    // Test the interface that the campaign UI calls to generate encounter configs

    // Case 1: Normal campaign mission
    let cfg = scenario_from_campaign(
        &["knight".into(), "mage".into(), "ranger".into()],
        3, 10, RoomType::Pressure, 42, None,
    );
    assert_eq!(cfg.hero_templates.len(), 3);
    assert!(!cfg.enemy_hero_templates.is_empty());
    assert!(cfg.enemy_count >= 2);
    assert!(cfg.hp_multiplier >= 1.0);
    assert_eq!(cfg.room_type, "Pressure");

    // Verify the produced config actually builds successfully
    let setup = build_combat(&cfg);
    let hero_count = setup.sim.units.iter().filter(|u| u.team == Team::Hero).count();
    assert_eq!(hero_count, 3);
    assert!(setup.sim.grid_nav.is_some());
    assert!(!setup.grid_nav.blocked.is_empty());

    // Case 2: Early game (low difficulty, turn 0)
    let cfg_early = scenario_from_campaign(
        &["knight".into()], 1, 0, RoomType::Entry, 1, None,
    );
    assert_eq!(cfg_early.hp_multiplier, 1.0, "turn 0 should have no HP scaling");
    assert!(cfg_early.enemy_count <= 4, "low difficulty should not overwhelm");

    // Case 3: Late game (high difficulty, high turn)
    let cfg_late = scenario_from_campaign(
        &["knight".into()], 5, 30, RoomType::Climax, 1, None,
    );
    assert!(cfg_late.hp_multiplier > cfg_early.hp_multiplier);
    assert!(cfg_late.enemy_count > cfg_early.enemy_count);
}

#[test]
fn active_mission_context_paths_all_produce_valid_combat() {
    // Test every path through mission_enter's config construction

    // Path 1: hero_templates set → scenario_from_campaign
    let ctx1 = ActiveMissionContext {
        hero_templates: vec!["knight".into(), "mage".into()],
        difficulty: 3,
        global_turn: 10,
        seed: 42,
        ..Default::default()
    };
    let cfg1 = scenario_from_campaign(
        &ctx1.hero_templates, ctx1.difficulty, ctx1.global_turn,
        ctx1.room_type, ctx1.seed, None,
    );
    let setup1 = build_combat(&cfg1);
    assert!(setup1.sim.units.len() >= 3);

    // Path 2: scenario_cfg set → use directly
    let ctx2 = ActiveMissionContext {
        scenario_cfg: Some(ScenarioCfg {
            name: "flashpoint".into(),
            seed: 99,
            hero_templates: vec!["paladin".into()],
            enemy_hero_templates: vec!["assassin".into()],
            ..ScenarioCfg::default()
        }),
        ..Default::default()
    };
    let setup2 = build_combat(ctx2.scenario_cfg.as_ref().unwrap());
    let heroes = setup2.sim.units.iter().filter(|u| u.team == Team::Hero).count();
    let enemies = setup2.sim.units.iter().filter(|u| u.team == Team::Enemy).count();
    assert_eq!(heroes, 1);
    assert_eq!(enemies, 1);

    // Path 3: empty hero_templates, no scenario_cfg → fallback defaults
    let ctx3 = ActiveMissionContext::default();
    let cfg3 = ScenarioCfg {
        name: "mission".into(),
        seed: ctx3.seed,
        hero_count: ctx3.player_unit_count,
        enemy_count: ctx3.enemy_unit_count,
        difficulty: ctx3.difficulty,
        room_type: "Entry".to_string(),
        ..ScenarioCfg::default()
    };
    let setup3 = build_combat(&cfg3);
    assert_eq!(
        setup3.sim.units.iter().filter(|u| u.team == Team::Hero).count(),
        4,
    );
    assert_eq!(
        setup3.sim.units.iter().filter(|u| u.team == Team::Enemy).count(),
        4,
    );
}

// ===========================================================================
// Test: Full user journey hotpath
//
// Simulates the exact sequence a player follows through the hub UI:
//   OverworldMap → RegionView → LocalEagleEyeIntro (auto-advances)
//   → MissionExecution (combat plays, sim ticks, recording)
//   → "Return to Overworld" (mission_exit, LastMissionReplay created)
//   → "Review Last Mission" button (enters ReplayViewer)
//   → Replay controls (pause, step, Esc to exit)
// ===========================================================================

#[test]
fn user_journey_overworld_to_mission_to_replay() {
    // -----------------------------------------------------------------------
    // Setup: build a minimal App with the systems the hotpath touches
    // -----------------------------------------------------------------------
    let mut app = App::new();
    app.insert_resource(Time::<()>::default());
    app.init_resource::<Assets<Mesh>>();
    app.init_resource::<Assets<StandardMaterial>>();
    app.insert_resource(SimEventBuffer::default());

    // Campaign state: roster with active heroes
    let mut roster = game_core::CampaignRoster {
        heroes: Vec::new(),
        recruit_pool: Vec::new(),
        player_hero_id: None,
        next_id: 1,
        generation_counter: 0,
    };
    // Add 3 heroes with different archetypes (what the roster would have)
    roster.heroes.push(game_core::HeroCompanion {
        id: 1,
        name: "Aldric".into(),
        origin_faction_id: 0,
        origin_region_id: 0,
        backstory: "Test hero".into(),
        archetype: game_core::PersonalityArchetype::Vanguard,
        loyalty: 70.0,
        stress: 10.0,
        fatigue: 0.0,
        injury: 0.0,
        resolve: 80.0,
        active: true,
        deserter: false,
        xp: 0,
        level: 1,
        equipment: Default::default(),
        traits: Vec::new(),
    });
    roster.heroes.push(game_core::HeroCompanion {
        id: 2,
        name: "Brynn".into(),
        origin_faction_id: 0,
        origin_region_id: 0,
        backstory: "Test hero".into(),
        archetype: game_core::PersonalityArchetype::Guardian,
        loyalty: 60.0,
        stress: 15.0,
        fatigue: 0.0,
        injury: 0.0,
        resolve: 75.0,
        active: true,
        deserter: false,
        xp: 0,
        level: 1,
        equipment: Default::default(),
        traits: Vec::new(),
    });
    roster.heroes.push(game_core::HeroCompanion {
        id: 3,
        name: "Cael".into(),
        origin_faction_id: 0,
        origin_region_id: 0,
        backstory: "Test hero".into(),
        archetype: game_core::PersonalityArchetype::Tactician,
        loyalty: 55.0,
        stress: 20.0,
        fatigue: 0.0,
        injury: 0.0,
        resolve: 65.0,
        active: true,
        deserter: false,
        xp: 0,
        level: 1,
        equipment: Default::default(),
        traits: Vec::new(),
    });
    roster.next_id = 4;

    let run_state = RunState { global_turn: 8 };
    app.insert_resource(roster);
    app.insert_resource(run_state);

    // Start at OverworldMap (where the player picks a region)
    app.insert_resource(HubUiState {
        screen: HubScreen::OverworldMap,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    });

    // Register the systems involved in the hotpath
    app.add_systems(Update, (
        crate::mission::execution::mission_scene_transition_system,
        replay_viewer_transition_system,
    ));
    app.add_systems(
        Update,
        advance_sim_system
            .after(crate::mission::execution::mission_scene_transition_system)
            .run_if(|sim: Option<Res<MissionSimState>>| sim.is_some()),
    );
    app.add_systems(
        Update,
        (
            advance_replay_viewer_system,
            replay_viewer_keyboard_system,
        )
            .after(replay_viewer_transition_system)
            .run_if(|v: Option<Res<ReplayViewerState>>| v.is_some()),
    );

    // Initial frame
    app.update();
    assert_eq!(app.world.resource::<HubUiState>().screen, HubScreen::OverworldMap);

    // -----------------------------------------------------------------------
    // Step 1: Player clicks "Enter Region" → RegionView
    // (In the real UI this goes through request_enter_selected_region + region
    //  transition system. We simulate the result: hub_ui.screen = RegionView)
    // -----------------------------------------------------------------------
    app.world.resource_mut::<HubUiState>().screen = HubScreen::RegionView;
    app.update();
    assert_eq!(app.world.resource::<HubUiState>().screen, HubScreen::RegionView);

    // -----------------------------------------------------------------------
    // Step 2: Player clicks "Enter Local Eagle-Eye Intro"
    // (bootstrap_local_eagle_eye_intro sets phase → HiddenInside)
    // We skip the intro animation by directly simulating what
    // local_intro_sequence_system does: set input_handoff_ready = true
    // and populate ActiveMissionContext from roster.
    // -----------------------------------------------------------------------

    // Simulate what local_intro_sequence_system does when intro completes:
    // Map roster archetypes → hero template names (same logic as hub_outcome.rs)
    let hero_templates: Vec<String> = app
        .world
        .resource::<game_core::CampaignRoster>()
        .heroes
        .iter()
        .filter(|h| h.active && !h.deserter)
        .map(|h| match h.archetype {
            game_core::PersonalityArchetype::Vanguard => "knight".to_string(),
            game_core::PersonalityArchetype::Guardian => "paladin".to_string(),
            game_core::PersonalityArchetype::Tactician => "ranger".to_string(),
        })
        .collect();
    assert_eq!(hero_templates, vec!["knight", "paladin", "ranger"]);

    let global_turn = app.world.resource::<RunState>().global_turn;
    assert_eq!(global_turn, 8);

    // Insert ActiveMissionContext (what local_intro_sequence_system does)
    app.insert_resource(ActiveMissionContext {
        hero_templates: hero_templates.clone(),
        global_turn,
        seed: 42,
        difficulty: 2,
        ..Default::default()
    });

    // Transition to MissionExecution (what local_intro_sequence_system does)
    app.world.resource_mut::<HubUiState>().screen = HubScreen::MissionExecution;
    app.update();

    // -----------------------------------------------------------------------
    // Step 3: Verify mission started with roster heroes
    // -----------------------------------------------------------------------
    {
        let sim_state = app
            .world
            .get_resource::<MissionSimState>()
            .expect("MissionSimState should exist after entering mission");

        let heroes: Vec<_> = sim_state
            .sim
            .units
            .iter()
            .filter(|u| u.team == Team::Hero)
            .collect();
        assert_eq!(
            heroes.len(), 3,
            "should have 3 heroes from roster (knight, paladin, ranger)"
        );

        let enemies: Vec<_> = sim_state
            .sim
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy)
            .collect();
        assert!(
            enemies.len() >= 2,
            "should have enemies from campaign scaling (difficulty=2)"
        );

        // Recording should have started
        let recording = app.world.resource::<MissionCombatRecording>();
        assert!(recording.active);
        assert_eq!(recording.frames.len(), 1, "initial frame recorded");
    }

    // -----------------------------------------------------------------------
    // Step 4: Combat plays out — run sim ticks until outcome
    // -----------------------------------------------------------------------
    let mut outcome_found = false;
    for frame in 0..500 {
        {
            let mut time = app.world.resource_mut::<Time<()>>();
            time.advance_by(std::time::Duration::from_millis(100));
        }
        app.update();

        if let Some(sim) = app.world.get_resource::<MissionSimState>() {
            if sim.outcome.is_some() {
                outcome_found = true;
                break;
            }
        }

        // If the sim was removed (e.g. early exit), stop
        if app.world.get_resource::<MissionSimState>().is_none() {
            break;
        }
    }

    // Verify combat completed
    let sim_state = app
        .world
        .get_resource::<MissionSimState>()
        .expect("sim should still exist");
    assert!(
        sim_state.outcome.is_some(),
        "combat should have resolved (Victory or Defeat)"
    );
    let outcome = sim_state.outcome.unwrap();

    // Recording should have many frames
    let recording = app.world.resource::<MissionCombatRecording>();
    let combat_frame_count = recording.frames.len();
    assert!(
        combat_frame_count >= 10,
        "should have recorded multiple combat frames, got {}",
        combat_frame_count,
    );

    // -----------------------------------------------------------------------
    // Step 5: Player clicks "Return to Overworld"
    // (mission_outcome_ui_system sets hub_ui.screen = OverworldMap)
    // -----------------------------------------------------------------------
    app.world.resource_mut::<HubUiState>().screen = HubScreen::OverworldMap;
    app.update();

    // Mission resources cleaned up
    assert!(
        app.world.get_resource::<MissionSimState>().is_none(),
        "MissionSimState should be removed"
    );
    assert!(
        app.world.get_resource::<MissionCombatRecording>().is_none(),
        "MissionCombatRecording should be removed"
    );

    // LastMissionReplay should exist (this is what powers "Review Last Mission")
    let last_replay = app
        .world
        .get_resource::<LastMissionReplay>()
        .expect("LastMissionReplay should be created on mission exit");
    assert_eq!(last_replay.frames.len(), combat_frame_count);
    assert_eq!(last_replay.events_per_frame.len(), combat_frame_count);

    // -----------------------------------------------------------------------
    // Step 6: Player sees "Review Last Mission" button and clicks it
    // (hub_ui_draw checks last_replay.is_some() and shows button;
    //  clicking sets hub_ui.screen = ReplayViewer)
    // -----------------------------------------------------------------------
    assert!(
        app.world.get_resource::<LastMissionReplay>().is_some(),
        "button should be visible because LastMissionReplay exists"
    );
    app.world.resource_mut::<HubUiState>().screen = HubScreen::ReplayViewer;
    app.update();

    let viewer = app
        .world
        .get_resource::<ReplayViewerState>()
        .expect("ReplayViewerState should be created");
    assert_eq!(viewer.frames.len(), combat_frame_count);
    assert_eq!(viewer.previous_screen, HubScreen::OverworldMap);

    // -----------------------------------------------------------------------
    // Step 7: Player uses replay controls
    // -----------------------------------------------------------------------

    // 7a. Advance a few frames
    {
        let mut time = app.world.resource_mut::<Time<()>>();
        time.advance_by(std::time::Duration::from_millis(300));
    }
    app.update();

    let frame_after_advance = app.world.resource::<ReplayViewerState>().frame_index;
    assert!(frame_after_advance > 0, "replay should have advanced");

    // 7b. Press Space to pause
    {
        let mut keyboard = ButtonInput::<KeyCode>::default();
        keyboard.press(KeyCode::Space);
        app.insert_resource(keyboard);
    }
    app.update();
    assert!(
        app.world.resource::<ReplayViewerState>().paused,
        "Space should pause"
    );

    // 7c. Press ArrowLeft to step back
    let frame_before_step = app.world.resource::<ReplayViewerState>().frame_index;
    {
        let mut keyboard = ButtonInput::<KeyCode>::default();
        keyboard.press(KeyCode::ArrowLeft);
        app.insert_resource(keyboard);
    }
    app.update();
    assert_eq!(
        app.world.resource::<ReplayViewerState>().frame_index,
        frame_before_step.saturating_sub(1),
        "ArrowLeft should step back one frame"
    );

    // 7d. Press ArrowRight to step forward
    let frame_before_fwd = app.world.resource::<ReplayViewerState>().frame_index;
    {
        let mut keyboard = ButtonInput::<KeyCode>::default();
        keyboard.press(KeyCode::ArrowRight);
        app.insert_resource(keyboard);
    }
    app.update();
    assert_eq!(
        app.world.resource::<ReplayViewerState>().frame_index,
        frame_before_fwd + 1,
        "ArrowRight should step forward one frame"
    );

    // -----------------------------------------------------------------------
    // Step 8: Player presses Escape to exit replay → returns to OverworldMap
    // -----------------------------------------------------------------------
    {
        let mut keyboard = ButtonInput::<KeyCode>::default();
        keyboard.press(KeyCode::Escape);
        app.insert_resource(keyboard);
    }
    app.update();

    assert_eq!(
        app.world.resource::<HubUiState>().screen,
        HubScreen::OverworldMap,
        "Escape should return to OverworldMap (the previous screen)"
    );

    // Cleanup: replay viewer state removed, visuals despawned
    // (transition system runs on next frame after screen change)
    app.update();

    assert!(
        app.world.get_resource::<ReplayViewerState>().is_none(),
        "ReplayViewerState should be cleaned up after exit"
    );

    // LastMissionReplay persists — player can watch again
    assert!(
        app.world.get_resource::<LastMissionReplay>().is_some(),
        "LastMissionReplay should persist for re-watching"
    );
}
