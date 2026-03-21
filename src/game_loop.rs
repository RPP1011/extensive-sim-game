use bevy::app::AppExit;
use bevy::prelude::*;

use crate::game_core::{HubScreen, HubUiState, RunState};
use crate::ui::settings::SettingsMenuState;

#[derive(Resource)]
pub struct SimulationSteps(pub Option<u32>);

#[derive(Resource, Clone)]
pub struct TurnPacingState {
    pub paused: bool,
    pub seconds_per_turn: f32,
    pub accumulator_seconds: f32,
    pub pending_steps: u32,
}

impl Default for TurnPacingState {
    fn default() -> Self {
        Self {
            paused: false,
            seconds_per_turn: 0.75,
            accumulator_seconds: 0.0,
            pending_steps: 0,
        }
    }
}

#[derive(Resource, Clone, Copy)]
pub struct StartSceneState {
    pub active: bool,
}

#[derive(Resource, Clone, Copy)]
pub struct RuntimeModeState {
    pub hub_mode: bool,
    pub dev_mode: bool,
}

#[derive(Component)]
pub struct MissionHudText;

pub fn start_scene_input_system(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mut start_scene: ResMut<StartSceneState>,
    mut turn_pacing: ResMut<TurnPacingState>,
    simulation_steps: Res<SimulationSteps>,
    settings_menu: Res<SettingsMenuState>,
    scenario_replay: Option<Res<super::scenario_3d::ScenarioReplay>>,
) {
    if simulation_steps.0.is_some()
        || settings_menu.is_open
        || scenario_replay.is_some()
        || !start_scene.active
    {
        return;
    }
    let Some(keyboard) = keyboard else {
        return;
    };
    if keyboard.just_pressed(KeyCode::Enter) {
        start_scene.active = false;
        turn_pacing.paused = false;
        turn_pacing.accumulator_seconds = 0.0;
        turn_pacing.pending_steps = 0;
    }
}

pub fn turn_pacing_input_system(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mut pacing: ResMut<TurnPacingState>,
    simulation_steps: Res<SimulationSteps>,
    settings_menu: Res<SettingsMenuState>,
    scenario_replay: Option<Res<super::scenario_3d::ScenarioReplay>>,
    start_scene: Option<Res<StartSceneState>>,
) {
    if simulation_steps.0.is_some()
        || settings_menu.is_open
        || scenario_replay.is_some()
        || start_scene.as_ref().is_some_and(|s| s.active)
    {
        return;
    }
    let Some(keyboard) = keyboard else {
        return;
    };
    if keyboard.just_pressed(KeyCode::Space) {
        pacing.paused = !pacing.paused;
    }
    if pacing.paused && keyboard.just_pressed(KeyCode::Enter) {
        pacing.pending_steps = pacing.pending_steps.saturating_add(1);
    }
}

pub fn increment_global_turn(
    mut run_state: ResMut<RunState>,
    simulation_steps: Res<SimulationSteps>,
    time: Option<Res<Time>>,
    pacing: Option<ResMut<TurnPacingState>>,
    scenario_replay: Option<Res<super::scenario_3d::ScenarioReplay>>,
    start_scene: Option<Res<StartSceneState>>,
) {
    if let Some(steps) = simulation_steps.0 {
        if steps > 0 {
            run_state.global_turn += 1;
        }
        return;
    }
    if scenario_replay.is_some() {
        run_state.global_turn += 1;
        return;
    }
    if start_scene.as_ref().is_some_and(|s| s.active) {
        return;
    }

    let Some(mut pacing) = pacing else {
        run_state.global_turn += 1;
        return;
    };

    if pacing.paused {
        if pacing.pending_steps > 0 {
            pacing.pending_steps -= 1;
            run_state.global_turn += 1;
        }
        return;
    }

    let Some(time) = time else {
        run_state.global_turn += 1;
        return;
    };
    pacing.accumulator_seconds += time.delta_seconds();
    let seconds_per_turn = pacing.seconds_per_turn.max(0.05);
    while pacing.accumulator_seconds >= seconds_per_turn {
        pacing.accumulator_seconds -= seconds_per_turn;
        run_state.global_turn += 1;
    }
}

pub fn exit_after_steps(
    mut app_exit_events: EventWriter<AppExit>,
    mut simulation_steps: ResMut<SimulationSteps>,
) {
    if let Some(steps) = &mut simulation_steps.0 {
        if *steps > 0 {
            *steps -= 1;
        } else {
            println!("Simulation complete after specified steps.");
            app_exit_events.send(AppExit);
        }
    }
}

pub fn run_if_simulation_steps_exist(simulation_steps: Option<Res<SimulationSteps>>) -> bool {
    simulation_steps.map_or(false, |steps| steps.0.is_some())
}

pub fn run_if_gameplay_active(start_scene: Option<Res<StartSceneState>>) -> bool {
    !start_scene.map_or(false, |state| state.active)
}

pub fn hub_runtime_input_enabled(
    hub_ui: &HubUiState,
    transition_state: Option<&super::region_nav::RegionLayerTransitionState>,
    local_intro_state: Option<&super::local_intro::LocalEagleEyeIntroState>,
) -> bool {
    if transition_state.is_some_and(|state| state.interaction_locked) {
        return false;
    }
    match hub_ui.screen {
        HubScreen::GuildManagement | HubScreen::Overworld | HubScreen::OverworldMap => true,
        HubScreen::LocalEagleEyeIntro => {
            local_intro_state.is_some_and(|state| {
                state.phase == super::local_intro::LocalIntroPhase::GameplayControl
            })
        }
        _ => false,
    }
}

pub fn run_if_hub_runtime_active(
    hub_ui: Option<Res<HubUiState>>,
    transition_state: Option<Res<super::region_nav::RegionLayerTransitionState>>,
    local_intro_state: Option<Res<super::local_intro::LocalEagleEyeIntroState>>,
) -> bool {
    match hub_ui.as_ref() {
        Some(ui) => hub_runtime_input_enabled(
            ui,
            transition_state.as_deref(),
            local_intro_state.as_deref(),
        ),
        None => false,
    }
}

pub fn run_if_mission_execution_active(hub_ui: Option<Res<HubUiState>>) -> bool {
    hub_ui.is_some_and(|ui| ui.screen == HubScreen::MissionExecution)
}

pub fn run_if_replay_viewer_active(hub_ui: Option<Res<HubUiState>>) -> bool {
    hub_ui.is_some_and(|ui| ui.screen == HubScreen::ReplayViewer)
}
