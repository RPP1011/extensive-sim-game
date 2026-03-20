//! Character creation side-panel screens (faction + backstory).

use bevy_egui::egui;

use super::faction_color;
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
        if super::common::ascii_button(ui, "Back To Start Menu") {
            enter_start_menu(hub_ui, start_menu);
        }
    });
    super::common::ascii_separator(ui);
    ui.label(egui::RichText::new("Character Creation - Faction").strong());
    ui.small("Choose your faction before continuing to backstory.");
    if choices.is_empty() {
        ui.colored_label(
            egui::Color32::from_rgb(235, 95, 95),
            "No factions available. Return to Start Menu and create a new campaign.",
        );
    } else {
        for choice in &choices {
            let is_selected =
                character_creation.selected_faction_index == Some(choice.index);
            let accent = faction_color(choice.index);
            egui::Frame::none()
                .fill(egui::Color32::TRANSPARENT)
                .inner_margin(egui::Margin::same(8.0))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        if super::common::ascii_button(ui, &format!("Select {}", choice.name))
                        {
                            character_creation.selected_faction_index = Some(choice.index);
                            character_creation.selected_faction_id = Some(choice.id.clone());
                            creation_ui.status = format!(
                                "Selected '{}'. Review impact and continue.",
                                choice.name
                            );
                        }
                        if is_selected {
                            ui.colored_label(accent, "Selected");
                        }
                    });
                    ui.label(truncate_for_hud(&choice.impact, 180));
                });
        }
        super::common::ascii_separator(ui);
        if super::common::ascii_button(ui, "Continue to Backstory") {
            let _ = confirm_faction_selection(
                hub_ui,
                character_creation,
                creation_ui,
                diplomacy,
                overworld,
            );
        }
    }
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
        if super::common::ascii_button(ui, "Back To Faction Step") {
            hub_ui.screen = HubScreen::CharacterCreationFaction;
        }
        if runtime_mode.dev_mode && super::common::ascii_button(ui, "Overworld Map (Dev)") {
            hub_ui.screen = HubScreen::OverworldMap;
        }
    });
    super::common::ascii_separator(ui);
    ui.label(egui::RichText::new("Character Creation - Backstory").strong());
    let faction_name = character_creation
        .selected_faction_index
        .and_then(|idx| overworld.factions.get(idx))
        .map(|f| f.name.as_str())
        .unwrap_or("None");
    ui.small(format!("Faction: {}", faction_name));
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
            let is_selected = character_creation
                .selected_backstory_id
                .as_deref()
                == Some(choice.id);
            egui::Frame::none()
                .fill(egui::Color32::TRANSPARENT)
                .inner_margin(egui::Margin::same(8.0))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        if super::common::ascii_button(ui, &format!("Select {}", choice.name))
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
                                egui::Color32::from_rgb(154, 182, 240),
                                "Selected",
                            );
                        }
                    });
                    ui.label(truncate_for_hud(choice.summary, 180));
                    ui.small(format!(
                        "Stats: {}",
                        choice.stat_modifiers.join(", ")
                    ));
                    ui.small(format!(
                        "Recruit focus: {}",
                        choice.recruit_bias_modifiers.join(", ")
                    ));
                });
        }
        super::common::ascii_separator(ui);
        if super::common::ascii_button(ui, "Confirm Backstory and Enter Overworld") {
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
}
