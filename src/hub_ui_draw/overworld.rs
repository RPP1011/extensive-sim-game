//! Overworld details side-panel (not the map view).

use bevy_egui::egui;

use crate::campaign_ops::{enter_start_menu, truncate_for_hud};
use crate::game_core::{self, HubScreen, HubUiState};
use crate::hub_types::StartMenuState;

fn format_bar(value: f32, max: f32) -> String {
    let filled = ((value / max).clamp(0.0, 1.0) * 10.0).round() as usize;
    let empty = 10 - filled;
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

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
        if super::common::ascii_button(ui, "Back To Start Menu") {
            enter_start_menu(hub_ui, start_menu);
        }
        if super::common::ascii_button(ui, "Switch To Guild") {
            hub_ui.screen = HubScreen::GuildManagement;
        }
        if super::common::ascii_button(ui, "Strategic Map") {
            hub_ui.screen = HubScreen::OverworldMap;
        }
    });
    super::common::ascii_separator(ui);
    ui.horizontal_wrapped(|ui| {
        ui.label(format!("Travel CD: {}", overworld.travel_cooldown_turns));
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

    super::common::ascii_separator(ui);
    ui.label(egui::RichText::new("Regions").strong());
    egui::Frame::none()
        .fill(egui::Color32::TRANSPARENT)
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
                let faction_name = overworld.factions.get(r.owner_faction_id)
                    .map(|f| f.name.as_str()).unwrap_or("Unknown");
                ui.label(format!("{marker} {} — {}", r.name, faction_name));
                if r.intel_level >= 65.0 {
                    ui.small(format!("  Intel {} Unrest {} Control {}",
                        format_bar(r.intel_level, 100.0), format_bar(r.unrest, 100.0), format_bar(r.control, 100.0)));
                } else if r.intel_level >= 35.0 {
                    ui.small(format!("  Intel {} (partial visibility)", format_bar(r.intel_level, 100.0)));
                } else {
                    ui.small("  Intel [░░░░░░░░░░] (unknown)");
                }
            }
        });

    super::common::ascii_separator(ui);
    ui.label(egui::RichText::new("Diplomacy & Offers").strong());
    egui::Frame::none()
        .fill(egui::Color32::TRANSPARENT)
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
                let tier = if rel >= 75 { "Allied" } else if rel >= 25 { "Neutral" } else { "Hostile" };
                let tier_color = match tier {
                    "Allied" => egui::Color32::from_rgb(126, 200, 122),
                    "Hostile" => egui::Color32::from_rgb(220, 100, 100),
                    _ => egui::Color32::from_rgb(180, 180, 160),
                };
                ui.horizontal(|ui| {
                    ui.label(&f.name);
                    ui.colored_label(tier_color, tier);
                });
            }
            super::common::ascii_separator(ui);
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
                    ui.label(format!("{marker} {} — {}", region, truncate_for_hud(&o.summary, 72)));
                }
            }
        });
}
