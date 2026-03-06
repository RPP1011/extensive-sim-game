//! Overworld details side-panel (not the map view).

use bevy_egui::egui;

use crate::campaign_ops::{enter_start_menu, truncate_for_hud};
use crate::game_core::{self, HubScreen, HubUiState};
use crate::hub_types::StartMenuState;

/// Draw the Overworld details side-panel (region list, diplomacy, offers).
pub(crate) fn draw_overworld_details(
    ui: &mut egui::Ui,
    hub_ui: &mut HubUiState,
    start_menu: &mut StartMenuState,
    overworld: &mut game_core::OverworldMap,
    diplomacy: &game_core::DiplomacyState,
    interactions: &game_core::InteractionBoard,
) {
    ui.horizontal(|ui| {
        if ui.button("Back To Start Menu").clicked() {
            enter_start_menu(hub_ui, start_menu);
        }
        if ui.button("Switch To Guild").clicked() {
            hub_ui.screen = HubScreen::GuildManagement;
        }
        if ui.button("Strategic Map").clicked() {
            hub_ui.screen = HubScreen::OverworldMap;
        }
    });
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.label(format!("Travel CD: {}", overworld.travel_cooldown_turns));
        ui.label(format!("Seed: {}", overworld.map_seed));
        ui.label(format!(
            "Current: {}",
            overworld
                .regions
                .get(overworld.current_region)
                .map(|r| r.name.as_str())
                .unwrap_or("Unknown")
        ));
        ui.label(format!(
            "Selected: {}",
            overworld
                .regions
                .get(overworld.selected_region)
                .map(|r| r.name.as_str())
                .unwrap_or("Unknown")
        ));
    });
    ui.small("Controls: J/L select region | T travel | 1/2/3 flashpoint intent");

    ui.separator();
    ui.label(egui::RichText::new("Regions").strong());
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(13, 16, 22))
        .inner_margin(egui::Margin::same(8.0))
        .show(ui, |ui| {
            for r in overworld.regions.iter().take(10) {
                let marker = if r.id == overworld.current_region {
                    "*"
                } else if r.id == overworld.selected_region {
                    ">"
                } else {
                    " "
                };
                let telemetry = if r.intel_level >= 65.0 {
                    format!("unrest={:.0} control={:.0}", r.unrest, r.control)
                } else if r.intel_level >= 35.0 {
                    format!(
                        "unrest~{:.0} control~{:.0}",
                        (r.unrest / 10.0).round() * 10.0,
                        (r.control / 10.0).round() * 10.0
                    )
                } else {
                    "unrest=? control=?".to_string()
                };
                ui.label(format!(
                    "{marker} {} [F{}] intel={:.0} {}",
                    r.name, r.owner_faction_id, r.intel_level, telemetry
                ));
            }
        });

    ui.separator();
    ui.label(egui::RichText::new("Diplomacy & Offers").strong());
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(13, 16, 22))
        .inner_margin(egui::Margin::same(8.0))
        .show(ui, |ui| {
            for (idx, f) in overworld
                .factions
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != diplomacy.player_faction_id)
                .take(5)
            {
                let rel = diplomacy.relations[diplomacy.player_faction_id][idx];
                ui.label(format!("{} relation={}", f.name, rel));
            }
            ui.separator();
            if interactions.offers.is_empty() {
                ui.label("offers: none");
            } else {
                for (idx, o) in interactions.offers.iter().enumerate().take(5) {
                    let marker = if idx == interactions.selected { ">" } else { " " };
                    let region = overworld
                        .regions
                        .iter()
                        .find(|r| r.id == o.region_id)
                        .map(|r| r.name.as_str())
                        .unwrap_or("Unknown Region");
                    ui.label(format!(
                        "{marker} #{} {:?} [{}] {}",
                        o.id,
                        o.kind,
                        region,
                        truncate_for_hud(&o.summary, 72)
                    ));
                }
            }
        });
}
