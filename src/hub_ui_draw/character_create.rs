//! Character creation side-panel screens (faction + backstory).

use bevy_egui::egui;

use crate::campaign_ops::{enter_start_menu, truncate_for_hud};
use crate::character_select::{
    build_backstory_selection_choices, build_faction_selection_choices, confirm_backstory_selection,
    confirm_faction_selection,
};
use crate::game_core::{self, CharacterCreationState, HubScreen, HubUiState};
use crate::game_loop::RuntimeModeState;
use crate::hub_types::{CharacterCreationUiState, StartMenuState};

/// Side-panel faction selection screen.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_faction_side_panel(
    ui: &mut egui::Ui,
    hub_ui: &mut HubUiState,
    start_menu: &mut StartMenuState,
    character_creation: &mut CharacterCreationState,
    creation_ui: &mut CharacterCreationUiState,
    diplomacy: &mut game_core::DiplomacyState,
    overworld: &game_core::OverworldMap,
    _runtime_mode: &RuntimeModeState,
) {
    let choices = build_faction_selection_choices(overworld);
    ui.horizontal(|ui| {
        if ui.button("Back To Start Menu").clicked() {
            enter_start_menu(hub_ui, start_menu);
        }
    });
    ui.separator();
    ui.label(egui::RichText::new("Character Creation - Faction").strong());
    ui.small("Choose your faction before continuing to backstory.");
    if choices.is_empty() {
        ui.colored_label(
            egui::Color32::from_rgb(235, 95, 95),
            "No factions available. Return to Start Menu and create a new campaign.",
        );
    } else {
        for choice in &choices {
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(13, 16, 22))
                .inner_margin(egui::Margin::same(8.0))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        let is_selected =
                            character_creation.selected_faction_index == Some(choice.index);
                        if ui
                            .add(egui::Button::new(format!("Select {}", choice.name)))
                            .clicked()
                        {
                            character_creation.selected_faction_index = Some(choice.index);
                            character_creation.selected_faction_id = Some(choice.id.clone());
                            creation_ui.status = format!(
                                "Selected '{}'. Review impact and continue.",
                                choice.name
                            );
                        }
                        if is_selected {
                            ui.colored_label(
                                egui::Color32::from_rgb(138, 206, 125),
                                "Selected",
                            );
                        }
                    });
                    ui.small(format!("ID: {}", choice.id));
                    ui.label(truncate_for_hud(&choice.impact, 180));
                });
        }
        ui.separator();
        if ui.button("Continue to Backstory").clicked() {
            let _ = confirm_faction_selection(
                hub_ui,
                character_creation,
                creation_ui,
                diplomacy,
                overworld,
            );
        }
    }
    ui.small(format!(
        "Status: {}",
        truncate_for_hud(&creation_ui.status, 120)
    ));
}

/// Side-panel backstory selection screen.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_backstory_side_panel(
    ui: &mut egui::Ui,
    hub_ui: &mut HubUiState,
    character_creation: &mut CharacterCreationState,
    creation_ui: &mut CharacterCreationUiState,
    roster: &mut game_core::CampaignRoster,
    parties: &mut game_core::CampaignParties,
    overworld: &game_core::OverworldMap,
    runtime_mode: &RuntimeModeState,
) {
    let backstory_choices = build_backstory_selection_choices();
    ui.horizontal(|ui| {
        if ui.button("Back To Faction Step").clicked() {
            hub_ui.screen = HubScreen::CharacterCreationFaction;
        }
        if runtime_mode.dev_mode && ui.button("Overworld Map (Dev)").clicked() {
            hub_ui.screen = HubScreen::OverworldMap;
        }
    });
    ui.separator();
    ui.label(egui::RichText::new("Character Creation - Backstory").strong());
    let faction_text = character_creation
        .selected_faction_id
        .as_deref()
        .unwrap_or("none");
    ui.small(format!("Confirmed faction id: {}", faction_text));
    if backstory_choices.is_empty() {
        ui.colored_label(
            egui::Color32::from_rgb(235, 95, 95),
            "No backstory archetypes configured. Return to faction step.",
        );
    } else {
        ui.small(
            "Choose an archetype. Stat modifiers and recruit-generation bias apply immediately on confirmation.",
        );
        for choice in &backstory_choices {
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(13, 16, 22))
                .inner_margin(egui::Margin::same(8.0))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        let is_selected = character_creation
                            .selected_backstory_id
                            .as_deref()
                            == Some(choice.id);
                        if ui
                            .add(egui::Button::new(format!("Select {}", choice.name)))
                            .clicked()
                        {
                            character_creation.selected_backstory_id =
                                Some(choice.id.to_string());
                            creation_ui.status = format!(
                                "Selected '{}' archetype. Review effects and confirm.",
                                choice.name
                            );
                        }
                        if is_selected {
                            ui.colored_label(
                                egui::Color32::from_rgb(138, 206, 125),
                                "Selected",
                            );
                        }
                    });
                    ui.small(format!("ID: {}", choice.id));
                    ui.label(truncate_for_hud(choice.summary, 180));
                    ui.small(format!(
                        "Stat modifiers: {}",
                        choice.stat_modifiers.join(" | ")
                    ));
                    ui.small(format!(
                        "Recruit bias: {}",
                        choice.recruit_bias_modifiers.join(" | ")
                    ));
                });
        }
        ui.separator();
        if ui.button("Confirm Backstory and Enter Overworld").clicked() {
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
    ui.small(format!(
        "Status: {}",
        truncate_for_hud(&creation_ui.status, 120)
    ));
}
