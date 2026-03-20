//! Character creation center panels — faction and backstory selection (full-screen variants).

use bevy_egui::egui;

use crate::campaign_ops::{enter_start_menu, truncate_for_hud};
use crate::character_select::{
    build_backstory_selection_choices, build_faction_selection_choices, confirm_backstory_selection,
    confirm_faction_selection,
};
use crate::game_core::{
    self, CharacterCreationState, HubScreen, HubUiState,
};
use crate::hub_types::{CharacterCreationUiState, StartMenuState};
use crate::ui_helpers::split_faction_impact_sections;
use super::faction_color;

#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_faction_center(
    ctx: &egui::Context,
    ui: &mut egui::Ui,
    hub_ui: &mut HubUiState,
    start_menu: &mut StartMenuState,
    character_creation: &mut CharacterCreationState,
    creation_ui: &mut CharacterCreationUiState,
    diplomacy: &mut game_core::DiplomacyState,
    overworld: &game_core::OverworldMap,
) {
    let choices = build_faction_selection_choices(overworld);
    ui.heading("Step 1 of 2 \u{00b7} Faction");

    // Flavor text frame
    egui::Frame::none()
        .fill(egui::Color32::TRANSPARENT)
        .inner_margin(egui::Margin::same(10.0))
        .show(ui, |ui| {
            ui.colored_label(
                egui::Color32::from_rgb(160, 175, 195),
                "Choose the faction that shapes your campaign. Each offers different doctrine and recruits.",
            );
        });

    if !choices.is_empty() {
        let mut selected_pos = character_creation
            .selected_faction_index
            .and_then(|idx| choices.iter().position(|choice| choice.index == idx))
            .unwrap_or(0);
        let move_down = ctx.input(|input| input.key_pressed(egui::Key::ArrowDown));
        let move_up = ctx.input(|input| input.key_pressed(egui::Key::ArrowUp));
        if move_down {
            selected_pos = (selected_pos + 1).min(choices.len() - 1);
        }
        if move_up {
            selected_pos = selected_pos.saturating_sub(1);
        }
        if move_down || move_up {
            if let Some(choice) = choices.get(selected_pos) {
                character_creation.selected_faction_index = Some(choice.index);
                character_creation.selected_faction_id = Some(choice.id.clone());
                creation_ui.status = format!("Selected '{}'. Continue when ready.", choice.name);
            }
        }
        if ctx.input(|input| input.key_pressed(egui::Key::Enter)) {
            let _ = confirm_faction_selection(
                hub_ui,
                character_creation,
                creation_ui,
                diplomacy,
                overworld,
            );
        }
    }

    ui.add_space(6.0);

    if choices.is_empty() {
        ui.colored_label(
            egui::Color32::from_rgb(235, 95, 95),
            "No factions available. Return to Start Menu.",
        );
    } else {
        egui::ScrollArea::vertical()
            .max_height(430.0)
            .show(ui, |ui| {
                for choice in &choices {
                    let is_selected =
                        character_creation.selected_faction_index == Some(choice.index);
                    let accent = faction_color(choice.index);
                    let marker = if is_selected { "(\u{25cf})" } else { "( )" };
                    let bg = if is_selected {
                        egui::Color32::from_rgba_premultiplied(
                            accent.r(),
                            accent.g(),
                            accent.b(),
                            40,
                        )
                    } else {
                        egui::Color32::from_rgb(16, 22, 32)
                    };

                    let resp = egui::Frame::none()
                        .fill(bg)
                        .stroke(egui::Stroke::new(
                            if is_selected { 2.0 } else { 1.0 },
                            if is_selected {
                                accent
                            } else {
                                egui::Color32::from_rgb(40, 50, 65)
                            },
                        ))
                        .rounding(egui::Rounding::same(4.0))
                        .inner_margin(egui::Margin::same(8.0))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(marker).monospace().color(accent),
                                );
                                ui.label(
                                    egui::RichText::new(&choice.name).strong().color(accent),
                                );
                            });
                            let (doctrine, profile, recruit) =
                                split_faction_impact_sections(&choice.impact);
                            ui.small(doctrine);
                            ui.small(profile);
                            ui.small(recruit);
                        });

                    if resp.response.clicked() {
                        character_creation.selected_faction_index = Some(choice.index);
                        character_creation.selected_faction_id = Some(choice.id.clone());
                        creation_ui.status =
                            format!("Selected '{}'. Continue when ready.", choice.name);
                    }

                    ui.add_space(4.0);
                }
            });
    }

    // Story-so-far bar
    super::common::ascii_separator(ui);
    ui.horizontal(|ui| {
        ui.colored_label(egui::Color32::from_rgb(100, 115, 135), "So far:");
        if let Some(idx) = character_creation.selected_faction_index {
            if let Some(choice) = choices.iter().find(|c| c.index == idx) {
                ui.colored_label(faction_color(idx), format!("[Faction: {}]", choice.name));
            }
        }
    });

    super::common::ascii_separator(ui);
    ui.horizontal_wrapped(|ui| {
        if super::common::ascii_button(ui, "Back") {
            enter_start_menu(hub_ui, start_menu);
        }
        let has_selection = character_creation.selected_faction_index.is_some();
        if super::common::ascii_button_enabled(ui, "Continue To Backstory", has_selection)
        {
            let _ = confirm_faction_selection(
                hub_ui,
                character_creation,
                creation_ui,
                diplomacy,
                overworld,
            );
        }
        ui.small("Keyboard: Up/Down selects, Enter continues.");
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_backstory_center(
    ctx: &egui::Context,
    ui: &mut egui::Ui,
    hub_ui: &mut HubUiState,
    character_creation: &mut CharacterCreationState,
    creation_ui: &mut CharacterCreationUiState,
    roster: &mut game_core::CampaignRoster,
    parties: &mut game_core::CampaignParties,
    overworld: &game_core::OverworldMap,
) {
    let backstory_choices = build_backstory_selection_choices();
    let faction_choices = build_faction_selection_choices(overworld);
    ui.heading("Step 2 of 2 \u{00b7} Backstory");

    // Flavor text frame
    egui::Frame::none()
        .fill(egui::Color32::TRANSPARENT)
        .inner_margin(egui::Margin::same(10.0))
        .show(ui, |ui| {
            ui.colored_label(
                egui::Color32::from_rgb(160, 175, 195),
                "Define your archetype and opening campaign pressure.",
            );
        });

    if !backstory_choices.is_empty() {
        let mut selected_pos = character_creation
            .selected_backstory_id
            .as_deref()
            .and_then(|id| backstory_choices.iter().position(|choice| choice.id == id))
            .unwrap_or(0);
        let move_down = ctx.input(|input| input.key_pressed(egui::Key::ArrowDown));
        let move_up = ctx.input(|input| input.key_pressed(egui::Key::ArrowUp));
        if move_down {
            selected_pos = (selected_pos + 1).min(backstory_choices.len() - 1);
        }
        if move_up {
            selected_pos = selected_pos.saturating_sub(1);
        }
        if move_down || move_up {
            if let Some(choice) = backstory_choices.get(selected_pos) {
                character_creation.selected_backstory_id = Some(choice.id.to_string());
                creation_ui.status =
                    format!("Selected '{}'. Confirm to proceed.", choice.name);
            }
        }
        if ctx.input(|input| input.key_pressed(egui::Key::Enter)) {
            let _ = confirm_backstory_selection(
                hub_ui,
                character_creation,
                creation_ui,
                roster,
                parties,
                overworld,
            );
        }
    }

    ui.add_space(6.0);

    let backstory_accent = egui::Color32::from_rgb(154, 182, 240);
    egui::ScrollArea::vertical()
        .max_height(430.0)
        .show(ui, |ui| {
            for choice in &backstory_choices {
                let is_selected = character_creation
                    .selected_backstory_id
                    .as_deref()
                    == Some(choice.id);
                let marker = if is_selected { "(\u{25cf})" } else { "( )" };
                let accent = if is_selected {
                    backstory_accent
                } else {
                    egui::Color32::from_rgb(130, 155, 190)
                };
                let bg = if is_selected {
                    egui::Color32::from_rgba_premultiplied(
                        backstory_accent.r(),
                        backstory_accent.g(),
                        backstory_accent.b(),
                        40,
                    )
                } else {
                    egui::Color32::from_rgb(16, 22, 32)
                };

                let resp = egui::Frame::none()
                    .fill(bg)
                    .stroke(egui::Stroke::new(
                        if is_selected { 2.0 } else { 1.0 },
                        if is_selected {
                            backstory_accent
                        } else {
                            egui::Color32::from_rgb(40, 50, 65)
                        },
                    ))
                    .rounding(egui::Rounding::same(4.0))
                    .inner_margin(egui::Margin::same(8.0))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(marker).monospace().color(accent),
                            );
                            ui.label(
                                egui::RichText::new(choice.name).strong().color(accent),
                            );
                        });
                        ui.small(truncate_for_hud(choice.summary, 130));
                        ui.small(format!(
                            "Stats: {}",
                            choice.stat_modifiers.join(", ")
                        ));
                    });

                if resp.response.clicked() {
                    character_creation.selected_backstory_id =
                        Some(choice.id.to_string());
                    creation_ui.status = format!(
                        "Selected '{}'. Confirm to proceed.",
                        choice.name
                    );
                }

                ui.add_space(4.0);
            }
        });

    // Story-so-far bar
    super::common::ascii_separator(ui);
    ui.horizontal(|ui| {
        ui.colored_label(egui::Color32::from_rgb(100, 115, 135), "So far:");
        if let Some(idx) = character_creation.selected_faction_index {
            if let Some(fc) = faction_choices.iter().find(|c| c.index == idx) {
                ui.colored_label(faction_color(idx), format!("[Faction: {}]", fc.name));
            }
        }
        if let Some(ref bs_id) = character_creation.selected_backstory_id {
            if let Some(bc) = backstory_choices.iter().find(|c| c.id == bs_id.as_str()) {
                ui.colored_label(backstory_accent, format!("[Backstory: {}]", bc.name));
            }
        }
    });

    super::common::ascii_separator(ui);
    ui.horizontal_wrapped(|ui| {
        if super::common::ascii_button(ui, "Back To Faction Step") {
            hub_ui.screen = HubScreen::CharacterCreationFaction;
        }
        let has_selection = character_creation.selected_backstory_id.is_some();
        if super::common::ascii_button_enabled(ui, "Confirm And Enter Overworld", has_selection)
        {
            let _ = confirm_backstory_selection(
                hub_ui,
                character_creation,
                creation_ui,
                roster,
                parties,
                overworld,
            );
        }
        ui.small("Keyboard: Up/Down selects, Enter confirms.");
    });
}
