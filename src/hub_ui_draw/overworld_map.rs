//! Overworld strategic map — right-panel crises and left-panel map rendering.

use bevy::prelude::*;
use bevy_egui::egui;
use crate::camera::{CameraFocusTransitionState, OrbitCameraController, SceneViewBounds};
use crate::campaign_ops::{enter_start_menu, truncate_for_hud};
use crate::game_core::{self, CharacterCreationState, HubScreen, HubUiState};
use crate::hub_types::{HubMenuState, StartMenuState};
use crate::game_loop::RuntimeModeState;
use crate::region_nav::{
    RegionLayerTransitionState, RegionTargetPickerState,
    request_enter_selected_region,
};
use super::overworld_map_parties;
use super::overworld_map_strategic;

/// Draw the left-panel overworld map content.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_overworld_map_panel(
    ui: &mut egui::Ui,
    hub_ui: &mut HubUiState,
    hub_menu: &mut HubMenuState,
    start_menu: &mut StartMenuState,
    overworld: &mut game_core::OverworldMap,
    parties: &mut game_core::CampaignParties,
    roster: &mut game_core::CampaignRoster,
    target_picker: &mut RegionTargetPickerState,
    camera_focus_transition: &mut CameraFocusTransitionState,
    region_transition: &mut RegionLayerTransitionState,
    character_creation: &CharacterCreationState,
    camera_query: &Query<&OrbitCameraController>,
    bounds: &SceneViewBounds,
    party_snapshots: &[game_core::CampaignParty],
    _runtime_mode: &RuntimeModeState,
    transition_locked: bool,
) {
    // Navigation buttons
    ui.horizontal(|ui| {
        if ui
            .add_enabled(!transition_locked, egui::Button::new("Back To Start Menu"))
            .clicked()
        {
            enter_start_menu(hub_ui, start_menu);
        }
        if ui
            .add_enabled(!transition_locked, egui::Button::new("Switch To Guild"))
            .clicked()
        {
            hub_ui.screen = HubScreen::GuildManagement;
        }
        if ui
            .add_enabled(!transition_locked, egui::Button::new("Overworld Details"))
            .clicked()
        {
            hub_ui.screen = HubScreen::Overworld;
        }
    });
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.label(format!("Travel CD: {}", overworld.travel_cooldown_turns));
        ui.label(format!("Seed: {}", overworld.map_seed));
        ui.label("Map View: campaign terrain");
    });
    let selected_region_name = overworld
        .regions
        .get(overworld.selected_region)
        .map(|r| r.name.as_str())
        .unwrap_or("Unknown");
    ui.horizontal_wrapped(|ui| {
        if ui
            .add_enabled(
                !transition_locked,
                egui::Button::new(format!("Enter {}", selected_region_name)),
            )
            .clicked()
        {
            let notice = request_enter_selected_region(
                hub_ui,
                target_picker,
                camera_focus_transition,
                region_transition,
                overworld,
                character_creation,
            );
            parties.notice = notice.clone();
            hub_menu.notice = notice;
        }
        ui.small(truncate_for_hud(&region_transition.status, 120));
    });
    if transition_locked {
        ui.colored_label(
            egui::Color32::from_rgb(112, 207, 242),
            "Region transition in progress. Overworld inputs are temporarily locked.",
        );
    }

    ui.separator();
    overworld_map_strategic::draw_strategic_map(ui, overworld, target_picker, parties, party_snapshots, transition_locked);

    ui.separator();
    overworld_map_parties::draw_faction_control(ui, overworld);

    ui.separator();
    overworld_map_parties::draw_player_parties(
        ui,
        overworld,
        parties,
        roster,
        target_picker,
        camera_focus_transition,
        camera_query,
        bounds,
    );
}


