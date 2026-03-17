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
use crate::ui_helpers::{illustration_placeholder_tile, split_faction_impact_sections};
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
    ui.label("Choose your faction identity and campaign doctrine.");

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

    ui.separator();
    ui.columns(2, |columns| {
        columns[0].vertical(|ui| {
            if let Some(selected) = choices
                .iter()
                .find(|choice| character_creation.selected_faction_index == Some(choice.index))
            {
                let accent = faction_color(selected.index);
                let (doctrine, profile, recruit) =
                    split_faction_impact_sections(&selected.impact);
                illustration_placeholder_tile(
                    ui,
                    &format!("{} Key Art", selected.name),
                    &format!(
                        "Prompt: {} faction banner scene with terrain motifs.",
                        selected.name
                    ),
                    accent,
                );
                egui::Frame::none()
                    .fill(egui::Color32::from_rgba_premultiplied(9, 13, 18, 208))
                    .stroke(egui::Stroke::new(1.0, accent))
                    .rounding(egui::Rounding::same(8.0))
                    .inner_margin(egui::Margin::same(12.0))
                    .show(ui, |ui| {
                        ui.colored_label(accent, format!("Selected: {}", selected.name));
                        ui.small(doctrine);
                        ui.small(profile);
                        ui.small(recruit);
                    });
            } else {
                ui.small("Select a faction to preview narrative tone and doctrine.");
            }
        });
        columns[1].vertical(|ui| {
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
                            let (doctrine, profile, recruit) =
                                split_faction_impact_sections(&choice.impact);
                            let bg = if is_selected {
                                egui::Color32::from_rgba_premultiplied(
                                    accent.r(),
                                    accent.g(),
                                    accent.b(),
                                    44,
                                )
                            } else {
                                egui::Color32::from_rgba_premultiplied(12, 18, 26, 195)
                            };
                            egui::Frame::none()
                                .fill(bg)
                                .stroke(egui::Stroke::new(
                                    if is_selected { 2.0 } else { 1.0 },
                                    if is_selected {
                                        accent
                                    } else {
                                        egui::Color32::from_rgba_premultiplied(130, 162, 186, 82)
                                    },
                                ))
                                .rounding(egui::Rounding::same(8.0))
                                .inner_margin(egui::Margin::same(10.0))
                                .show(ui, |ui| {
                                    ui.horizontal_wrapped(|ui| {
                                        if ui
                                            .add(
                                                egui::Button::new(&choice.name)
                                                    .min_size(egui::vec2(180.0, 30.0)),
                                            )
                                            .clicked()
                                        {
                                            character_creation.selected_faction_index =
                                                Some(choice.index);
                                            character_creation.selected_faction_id =
                                                Some(choice.id.clone());
                                            creation_ui.status = format!(
                                                "Selected '{}'. Continue when ready.",
                                                choice.name
                                            );
                                        }
                                        if is_selected {
                                            ui.colored_label(accent, "Selected");
                                        }
                                    });
                                    ui.small(doctrine);
                                    ui.small(profile);
                                    ui.small(recruit);
                                });
                            ui.add_space(8.0);
                        }
                    });
            }
        });
    });
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        if ui.button("Back").clicked() {
            enter_start_menu(hub_ui, start_menu);
        }
        let has_selection = character_creation.selected_faction_index.is_some();
        if ui
            .add_enabled(has_selection, egui::Button::new("Continue To Backstory"))
            .clicked()
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
    ui.small(format!(
        "Status: {}",
        truncate_for_hud(&creation_ui.status, 120)
    ));
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
    ui.heading("Step 2 of 2 \u{00b7} Backstory");
    ui.label("Define your archetype and opening campaign pressure.");
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
    ui.separator();
    ui.columns(2, |columns| {
        columns[0].vertical(|ui| {
            let selected = backstory_choices
                .iter()
                .find(|choice| character_creation.selected_backstory_id.as_deref() == Some(choice.id));
            if let Some(choice) = selected {
                illustration_placeholder_tile(
                    ui,
                    &format!("{} Portrait", choice.name),
                    &format!(
                        "Prompt: {} hero line-art portrait for narrative cinematic.",
                        choice.name
                    ),
                    egui::Color32::from_rgb(224, 187, 120),
                );
                illustration_placeholder_tile(
                    ui,
                    "Backstory Scene Kit",
                    &format!(
                        "Prompt: cinematic line-art scene reflecting '{}'.",
                        choice.summary
                    ),
                    egui::Color32::from_rgb(148, 180, 240),
                );
            } else {
                ui.small("Select a backstory to preview cinematic narrative prompts.");
            }
        });
        columns[1].vertical(|ui| {
            egui::ScrollArea::vertical()
                .max_height(430.0)
                .show(ui, |ui| {
                    for choice in &backstory_choices {
                        let is_selected = character_creation
                            .selected_backstory_id
                            .as_deref()
                            == Some(choice.id);
                        egui::Frame::none()
                            .fill(if is_selected {
                                egui::Color32::from_rgb(37, 49, 74)
                            } else {
                                egui::Color32::from_rgba_premultiplied(12, 18, 26, 195)
                            })
                            .stroke(egui::Stroke::new(
                                if is_selected { 2.0 } else { 1.0 },
                                if is_selected {
                                    egui::Color32::from_rgb(154, 182, 240)
                                } else {
                                    egui::Color32::from_rgba_premultiplied(130, 162, 186, 82)
                                },
                            ))
                            .rounding(egui::Rounding::same(8.0))
                            .inner_margin(egui::Margin::same(10.0))
                            .show(ui, |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    if ui
                                        .add(
                                            egui::Button::new(choice.name)
                                                .min_size(egui::vec2(180.0, 30.0)),
                                        )
                                        .clicked()
                                    {
                                        character_creation.selected_backstory_id =
                                            Some(choice.id.to_string());
                                        creation_ui.status = format!(
                                            "Selected '{}'. Confirm to proceed.",
                                            choice.name
                                        );
                                    }
                                    if is_selected {
                                        ui.colored_label(
                                            egui::Color32::from_rgb(154, 182, 240),
                                            "Selected",
                                        );
                                    }
                                });
                                ui.small(truncate_for_hud(choice.summary, 130));
                                ui.small(format!(
                                    "Stat modifiers: {}",
                                    choice.stat_modifiers.join(" | ")
                                ));
                            });
                        ui.add_space(8.0);
                    }
                });
        });
    });
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        if ui.button("Back To Faction Step").clicked() {
            hub_ui.screen = HubScreen::CharacterCreationFaction;
        }
        let has_selection = character_creation.selected_backstory_id.is_some();
        if ui
            .add_enabled(
                has_selection,
                egui::Button::new("Confirm And Enter Overworld"),
            )
            .clicked()
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
    ui.small(format!(
        "Status: {}",
        truncate_for_hud(&creation_ui.status, 120)
    ));
}
