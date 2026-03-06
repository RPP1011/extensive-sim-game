use bevy::prelude::*;

use crate::game_core::{HubScreen, HubUiState};

pub const LOCAL_INTRO_HIDDEN_FRAMES: u16 = 20;
pub const LOCAL_INTRO_EXIT_FRAMES: u16 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalIntroPhase {
    Idle,
    HiddenInside,
    ExitingBuilding,
    GameplayControl,
}

#[derive(Debug, Clone)]
pub struct LocalIntroAnchor {
    pub prefab_id: &'static str,
    pub building_anchor_world: Vec3,
    pub player_spawn_world: Vec3,
    pub player_exit_world: Vec3,
}

#[derive(Resource, Debug, Clone)]
pub struct LocalEagleEyeIntroState {
    pub source_region_id: Option<usize>,
    pub anchor: Option<LocalIntroAnchor>,
    pub phase: LocalIntroPhase,
    pub phase_frames: u16,
    pub intro_completed: bool,
    pub input_handoff_ready: bool,
    pub status: String,
}

impl Default for LocalEagleEyeIntroState {
    fn default() -> Self {
        Self {
            source_region_id: None,
            anchor: None,
            phase: LocalIntroPhase::Idle,
            phase_frames: 0,
            intro_completed: false,
            input_handoff_ready: false,
            status: "Local intro idle.".to_string(),
        }
    }
}

pub fn local_intro_anchor_for_region(region_id: usize) -> Option<LocalIntroAnchor> {
    let prefab_id = match region_id {
        0 => "prefabs/local_intro/dilapidated_hamlet_a",
        1 => "prefabs/local_intro/dilapidated_hamlet_b",
        2 => "prefabs/local_intro/dilapidated_hamlet_c",
        3 => "prefabs/local_intro/dilapidated_hamlet_d",
        _ => return None,
    };
    Some(LocalIntroAnchor {
        prefab_id,
        building_anchor_world: Vec3::new(0.0, 0.0, 0.0),
        player_spawn_world: Vec3::new(-1.1, 0.0, -0.8),
        player_exit_world: Vec3::new(1.8, 0.0, 1.2),
    })
}

pub fn bootstrap_local_eagle_eye_intro(
    hub_ui: &mut HubUiState,
    local_intro: &mut LocalEagleEyeIntroState,
    region_transition: &super::region_nav::RegionLayerTransitionState,
    overworld: &crate::game_core::OverworldMap,
) -> String {
    let Some(payload) = region_transition.active_payload.as_ref() else {
        local_intro.source_region_id = None;
        local_intro.anchor = None;
        local_intro.phase = LocalIntroPhase::Idle;
        local_intro.phase_frames = 0;
        local_intro.intro_completed = false;
        local_intro.input_handoff_ready = false;
        local_intro.status =
            "Local intro bootstrap aborted: no active region context is available.".to_string();
        hub_ui.screen = HubScreen::RegionView;
        return local_intro.status.clone();
    };
    let Some(anchor) = local_intro_anchor_for_region(payload.region_id) else {
        local_intro.source_region_id = Some(payload.region_id);
        local_intro.anchor = None;
        local_intro.phase = LocalIntroPhase::Idle;
        local_intro.phase_frames = 0;
        local_intro.intro_completed = false;
        local_intro.input_handoff_ready = false;
        local_intro.status = format!(
            "Local intro bootstrap aborted: anchor prefab/geometry unavailable for region {} (id {}).",
            overworld
                .regions
                .get(payload.region_id)
                .map(|region| region.name.as_str())
                .unwrap_or("Unknown"),
            payload.region_id
        );
        hub_ui.screen = HubScreen::RegionView;
        return local_intro.status.clone();
    };

    local_intro.source_region_id = Some(payload.region_id);
    local_intro.anchor = Some(anchor);
    local_intro.phase = LocalIntroPhase::HiddenInside;
    local_intro.phase_frames = 0;
    local_intro.intro_completed = false;
    local_intro.input_handoff_ready = false;
    local_intro.status = format!(
        "Local intro bootstrapped for region {} (id {}). Building anchor placed; player hidden inside.",
        overworld
            .regions
            .get(payload.region_id)
            .map(|region| region.name.as_str())
            .unwrap_or("Unknown"),
        payload.region_id
    );
    hub_ui.screen = HubScreen::LocalEagleEyeIntro;
    local_intro.status.clone()
}

pub fn advance_local_eagle_eye_intro(
    local_intro: &mut LocalEagleEyeIntroState,
) -> Option<String> {
    match local_intro.phase {
        LocalIntroPhase::Idle | LocalIntroPhase::GameplayControl => None,
        LocalIntroPhase::HiddenInside => {
            local_intro.phase_frames = local_intro.phase_frames.saturating_add(1);
            if local_intro.phase_frames >= LOCAL_INTRO_HIDDEN_FRAMES {
                local_intro.phase = LocalIntroPhase::ExitingBuilding;
                local_intro.phase_frames = 0;
                local_intro.status =
                    "Local intro: character exits the dilapidated building.".to_string();
                return Some(local_intro.status.clone());
            }
            None
        }
        LocalIntroPhase::ExitingBuilding => {
            local_intro.phase_frames = local_intro.phase_frames.saturating_add(1);
            if local_intro.phase_frames >= LOCAL_INTRO_EXIT_FRAMES {
                local_intro.phase = LocalIntroPhase::GameplayControl;
                local_intro.phase_frames = 0;
                local_intro.intro_completed = true;
                local_intro.input_handoff_ready = true;
                local_intro.status =
                    "Local intro complete. Gameplay input handed off to player control."
                        .to_string();
                return Some(local_intro.status.clone());
            }
            None
        }
    }
}
