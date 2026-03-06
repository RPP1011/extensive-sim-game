//! Region view and local eagle-eye intro rendering.

use bevy_egui::egui;

use crate::campaign_ops::truncate_for_hud;
use crate::game_core::{self, HubScreen, HubUiState};
use crate::hub_types::HubMenuState;
use crate::local_intro::{
    bootstrap_local_eagle_eye_intro, LocalEagleEyeIntroState, LocalIntroPhase,
};
use crate::region_nav::RegionLayerTransitionState;
use crate::runtime_assets::RegionArtState;

/// Draw the RegionView side-panel content.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_region_view(
    ui: &mut egui::Ui,
    hub_ui: &mut HubUiState,
    hub_menu: &mut HubMenuState,
    local_intro: &mut LocalEagleEyeIntroState,
    parties: &mut game_core::CampaignParties,
    region_transition: &RegionLayerTransitionState,
    overworld: &game_core::OverworldMap,
    region_art: &RegionArtState,
    region_art_texture_id: Option<egui::TextureId>,
) {
    ui.horizontal(|ui| {
        if ui.button("Return To Overworld Map").clicked() {
            hub_ui.screen = HubScreen::OverworldMap;
        }
    });
    ui.separator();
    ui.label(egui::RichText::new("Region Layer").strong());
    if let Some(payload) = region_transition.active_payload.as_ref() {
        let region_name = overworld
            .regions
            .get(payload.region_id)
            .map(|region| region.name.as_str())
            .unwrap_or("Unknown");
        let faction_name = overworld
            .factions
            .get(payload.faction_index)
            .map(|faction| faction.name.as_str())
            .unwrap_or("Unknown");
        ui.label(format!(
            "Loaded region {} (id {}).",
            region_name, payload.region_id
        ));
        ui.small(format!(
            "Faction context: {} ({})",
            payload.faction_id, faction_name
        ));
        ui.small(format!(
            "Deterministic seeds: campaign={} region={}",
            payload.campaign_seed, payload.region_seed
        ));
        // Display pre-generated environment art for this region.
        if let Some(texture_id) = region_art_texture_id {
            let available = ui.available_width().min(480.0);
            let art_size = egui::vec2(available, available * 0.5625);
            ui.image(egui::load::SizedTexture::new(texture_id, art_size));
        } else if region_art.loaded_region_id == Some(payload.region_id) {
            ui.colored_label(
                egui::Color32::from_rgb(140, 150, 160),
                "No environment art generated for this region yet.",
            );
            if !region_art.status.is_empty() {
                ui.small(truncate_for_hud(&region_art.status, 120));
            }
        }
        if ui.button("Enter Local Eagle-Eye Intro").clicked() {
            let status = bootstrap_local_eagle_eye_intro(
                hub_ui,
                local_intro,
                region_transition,
                overworld,
            );
            parties.notice = status.clone();
            hub_menu.notice = status;
        }
    } else {
        ui.colored_label(
            egui::Color32::from_rgb(235, 95, 95),
            "Region payload missing. Returning to overworld is recommended.",
        );
    }
    ui.small(truncate_for_hud(&region_transition.status, 120));
    if !local_intro.status.is_empty() {
        ui.small(truncate_for_hud(&local_intro.status, 120));
    }
}

/// Draw the LocalEagleEyeIntro side-panel content.
pub(crate) fn draw_local_eagle_eye_intro(
    ui: &mut egui::Ui,
    hub_ui: &mut HubUiState,
    local_intro: &LocalEagleEyeIntroState,
    overworld: &game_core::OverworldMap,
) {
    ui.horizontal(|ui| {
        if ui.button("Return To Region Layer").clicked() {
            hub_ui.screen = HubScreen::RegionView;
        }
    });
    ui.separator();
    ui.label(egui::RichText::new("Local Eagle-Eye Intro").strong());
    if let Some(region_id) = local_intro.source_region_id {
        let region_name = overworld
            .regions
            .get(region_id)
            .map(|region| region.name.as_str())
            .unwrap_or("Unknown");
        ui.label(format!("Region context: {} (id {}).", region_name, region_id));
    } else {
        ui.label("Region context: unavailable");
    }
    if let Some(anchor) = local_intro.anchor.as_ref() {
        ui.small(format!("Building anchor prefab: {}", anchor.prefab_id));
        ui.small(format!(
            "Anchor world position: ({:.1}, {:.1}, {:.1})",
            anchor.building_anchor_world.x,
            anchor.building_anchor_world.y,
            anchor.building_anchor_world.z
        ));
        ui.small(format!(
            "Player path: inside ({:.1}, {:.1}, {:.1}) -> exit ({:.1}, {:.1}, {:.1})",
            anchor.player_spawn_world.x,
            anchor.player_spawn_world.y,
            anchor.player_spawn_world.z,
            anchor.player_exit_world.x,
            anchor.player_exit_world.y,
            anchor.player_exit_world.z
        ));
    } else {
        ui.colored_label(
            egui::Color32::from_rgb(235, 95, 95),
            "Building anchor unavailable. Intro can be retried from Region View.",
        );
    }
    let player_state = match local_intro.phase {
        LocalIntroPhase::Idle => "idle",
        LocalIntroPhase::HiddenInside => "hidden inside building",
        LocalIntroPhase::ExitingBuilding => "exiting building",
        LocalIntroPhase::GameplayControl => "outside and controllable",
    };
    ui.small(format!("Player state: {}", player_state));
    ui.small(format!(
        "Intro completed: {} | Gameplay input handoff: {}",
        if local_intro.intro_completed {
            "yes"
        } else {
            "no"
        },
        if local_intro.input_handoff_ready {
            "ready"
        } else {
            "pending"
        }
    ));
    ui.small(truncate_for_hud(&local_intro.status, 120));
}
