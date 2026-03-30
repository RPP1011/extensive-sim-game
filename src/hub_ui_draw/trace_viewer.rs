//! Campaign trace viewer state and Bevy systems.
//!
//! Resource-based playback with transition/advance/keyboard systems.
//! Originally in `src/headless_campaign/trace_viewer.rs`, moved here
//! when headless_campaign was split into its own crate (which can't
//! depend on Bevy).

use bevy::prelude::*;

use headless_campaign::trace::{CampaignTrace, TraceEvent};
use headless_campaign::state::CampaignState;
use crate::game_core::HubScreen;
use crate::game_core::HubUiState;

// ---------------------------------------------------------------------------
// Bevy resource
// ---------------------------------------------------------------------------

/// Tracks campaign trace playback state.
#[derive(Resource)]
pub struct CampaignTraceViewerState {
    pub trace: CampaignTrace,
    pub current_tick: u64,
    pub current_state: CampaignState,
    pub tick_speed: f32,
    pub tick_accumulator: f32,
    pub paused: bool,
    pub previous_screen: HubScreen,
    pub visible_events: Vec<TraceEvent>,
    pub event_window: u64,
}

impl CampaignTraceViewerState {
    pub fn new(trace: CampaignTrace, previous_screen: HubScreen) -> Self {
        let initial_state = trace
            .state_at_tick(0)
            .unwrap_or_else(|| CampaignState::default_test_campaign(trace.seed));
        let visible = trace.events_in_range(0, 500).into_iter().cloned().collect();
        Self {
            trace,
            current_tick: 0,
            current_state: initial_state,
            tick_speed: 100.0,
            tick_accumulator: 0.0,
            paused: true,
            previous_screen,
            visible_events: visible,
            event_window: 500,
        }
    }

    pub fn seek_to(&mut self, tick: u64) {
        let tick = tick.min(self.trace.total_ticks);
        self.current_tick = tick;
        if let Some(state) = self.trace.state_at_tick(tick) {
            self.current_state = state;
        }
        let start = tick.saturating_sub(self.event_window / 2);
        let end = tick + self.event_window / 2;
        self.visible_events = self.trace.events_in_range(start, end).into_iter().cloned().collect();
    }
}

/// Holds a loaded trace file. Persists across screen transitions.
#[derive(Resource)]
pub struct LoadedCampaignTrace {
    pub trace: CampaignTrace,
    pub file_path: Option<String>,
}

// ---------------------------------------------------------------------------
// Bevy systems
// ---------------------------------------------------------------------------

pub fn campaign_trace_viewer_transition_system(
    hub_ui: Res<HubUiState>,
    mut last_screen: Local<Option<HubScreen>>,
    mut commands: Commands,
    loaded_trace: Option<Res<LoadedCampaignTrace>>,
) {
    let current = hub_ui.screen;
    let previous = *last_screen;
    if previous == Some(current) { return; }

    if previous == Some(HubScreen::CampaignTraceViewer) {
        commands.remove_resource::<CampaignTraceViewerState>();
    }
    if current == HubScreen::CampaignTraceViewer {
        if let Some(ref loaded) = loaded_trace {
            let prev = previous.unwrap_or(HubScreen::StartMenu);
            commands.insert_resource(CampaignTraceViewerState::new(loaded.trace.clone(), prev));
        }
    }
    *last_screen = Some(current);
}

pub fn advance_campaign_trace_viewer_system(
    time: Res<Time>,
    viewer: Option<ResMut<CampaignTraceViewerState>>,
) {
    let Some(mut viewer) = viewer else { return };
    if viewer.paused || viewer.current_tick >= viewer.trace.total_ticks { return; }

    viewer.tick_accumulator += time.delta_seconds() * viewer.tick_speed;
    let ticks_to_advance = viewer.tick_accumulator as u64;
    if ticks_to_advance > 0 {
        viewer.tick_accumulator -= ticks_to_advance as f32;
        let new_tick = (viewer.current_tick + ticks_to_advance).min(viewer.trace.total_ticks);
        viewer.seek_to(new_tick);
    }
}

pub fn campaign_trace_viewer_keyboard_system(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mut hub_ui: ResMut<HubUiState>,
    mut viewer: Option<ResMut<CampaignTraceViewerState>>,
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

    let shift = keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight);
    let step = if shift { 1000 } else { 100 };

    if keyboard.just_pressed(KeyCode::ArrowLeft) {
        let new_tick = viewer.current_tick.saturating_sub(step);
        viewer.seek_to(new_tick);
        viewer.paused = true;
    }
    if keyboard.just_pressed(KeyCode::ArrowRight) {
        let new_tick = (viewer.current_tick + step).min(viewer.trace.total_ticks);
        viewer.seek_to(new_tick);
        viewer.paused = true;
    }

    if keyboard.just_pressed(KeyCode::Equal) || keyboard.just_pressed(KeyCode::NumpadAdd) {
        viewer.tick_speed = (viewer.tick_speed * 2.0).min(10000.0);
    }
    if keyboard.just_pressed(KeyCode::Minus) || keyboard.just_pressed(KeyCode::NumpadSubtract) {
        viewer.tick_speed = (viewer.tick_speed / 2.0).max(10.0);
    }

    if keyboard.just_pressed(KeyCode::KeyF) {
        let save_path = "generated/forked_campaign.json";
        if let Ok(json) = serde_json::to_string_pretty(&viewer.current_state) {
            if std::fs::write(save_path, &json).is_ok() {
                eprintln!("Forked campaign at tick {} → {}", viewer.current_tick, save_path);
            }
        }
    }
}
