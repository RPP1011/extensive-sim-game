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
        if super::common::ascii_button(ui, "Return To Overworld Map") {
            hub_ui.screen = HubScreen::OverworldMap;
        }
    });
    super::common::ascii_separator(ui);
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
        ui.label(format!("Region: {}", region_name));
        ui.small(format!("Controlled by: {}", faction_name));
        // Display pre-generated environment art for this region.
        if let Some(texture_id) = region_art_texture_id {
            let available = ui.available_width();
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
        if super::common::ascii_button(ui, "Scout Region") {
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
        if super::common::ascii_button(ui, "Return To Region Layer") {
            hub_ui.screen = HubScreen::RegionView;
        }
    });
    super::common::ascii_separator(ui);
    ui.label(egui::RichText::new("Eagle Eye Scouting").strong());
    ui.add_space(4.0);

    let region_name = local_intro
        .source_region_id
        .and_then(|id| overworld.regions.get(id))
        .map(|r| r.name.as_str())
        .unwrap_or("Unknown");

    // --- Narrative lines that reveal progressively by phase ---
    let muted = egui::Color32::from_rgb(160, 170, 180);
    let dim = egui::Color32::from_rgb(120, 130, 140);

    // Phase 0 (Idle / Preparing) — first batch of lines
    ui.colored_label(muted, format!("Target: {}", region_name));
    ui.colored_label(dim, "Your scout approaches the outskirts...");

    // Phase 1+ (HiddenInside) — reveal more detail
    if matches!(
        local_intro.phase,
        LocalIntroPhase::HiddenInside
            | LocalIntroPhase::ExitingBuilding
            | LocalIntroPhase::GameplayControl
    ) {
        ui.add_space(2.0);
        ui.colored_label(dim, "Slipping past the outer patrols...");
        ui.colored_label(dim, "The garrison looks thin today...");
        ui.colored_label(muted, "Counting heads from a rooftop vantage.");
    }

    // Phase 2+ (ExitingBuilding) — enemy assessment
    if matches!(
        local_intro.phase,
        LocalIntroPhase::ExitingBuilding | LocalIntroPhase::GameplayControl
    ) {
        ui.add_space(2.0);
        ui.colored_label(dim, "Assessing enemy strength...");
        ui.colored_label(dim, "Marking supply routes and escape paths.");
        ui.colored_label(muted, "Your scout signals: almost done.");
    }

    // Phase 3 (GameplayControl) — final readiness
    if local_intro.phase == LocalIntroPhase::GameplayControl {
        ui.add_space(2.0);
        ui.colored_label(
            egui::Color32::from_rgb(180, 210, 180),
            "Scouting complete. The area is mapped.",
        );
    }

    // --- Progress bar ---
    ui.add_space(8.0);
    let (progress, status_label) = match local_intro.phase {
        LocalIntroPhase::Idle => (0.2, "Preparing..."),
        LocalIntroPhase::HiddenInside => (0.5, "Infiltrating..."),
        LocalIntroPhase::ExitingBuilding => (0.8, "Emerging..."),
        LocalIntroPhase::GameplayControl => (1.0, "Ready to explore"),
    };
    ui.small(
        egui::RichText::new(status_label).color(egui::Color32::from_rgb(190, 195, 200)),
    );
    super::common::ascii_progress_bar(ui, progress, 15);
}
