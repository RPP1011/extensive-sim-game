//! Start menu rendering — GUI-only full-screen and side-panel variants.

use bevy_egui::egui;

use crate::campaign_ops::{format_slot_badge, truncate_for_hud};
use crate::game_core::{
    self, CharacterCreationState, HubScreen, HubUiState,
};
use crate::game_loop::RuntimeModeState;
use crate::hub_types::{CharacterCreationUiState, StartMenuState};
use crate::ui::save_browser::{CampaignSaveIndexState, CampaignSavePanelState};
use crate::ui::settings::SettingsMenuState;
use super::character_creation_center;

/// Full-screen GUI-only overlay for StartMenu, CharacterCreationFaction, and
/// CharacterCreationBackstory screens.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_gui_only_screens(
    ctx: &egui::Context,
    hub_ui: &mut HubUiState,
    start_menu: &mut StartMenuState,
    settings_menu: &mut SettingsMenuState,
    character_creation: &mut CharacterCreationState,
    creation_ui: &mut CharacterCreationUiState,
    diplomacy: &mut game_core::DiplomacyState,
    roster: &mut game_core::CampaignRoster,
    parties: &mut game_core::CampaignParties,
    overworld: &game_core::OverworldMap,
    can_continue: bool,
    first_launch_lock: bool,
    runtime_mode: &RuntimeModeState,
    slot1: &str,
    slot2: &str,
    slot3: &str,
    autosave: &str,
) {
    let show_start_menu_sidebar = hub_ui.screen == HubScreen::StartMenu;
    if show_start_menu_sidebar {
        draw_hamburger_nav(
            ctx,
            hub_ui,
            start_menu,
            settings_menu,
            runtime_mode,
            can_continue,
            first_launch_lock,
        );
    }

    egui::Area::new(egui::Id::new("hub_gui_only_content"))
        .anchor(
            egui::Align2::CENTER_CENTER,
            egui::vec2(if show_start_menu_sidebar { 90.0 } else { 0.0 }, 0.0),
        )
        .show(ctx, |ui| {
            let width = if show_start_menu_sidebar {
                (ctx.screen_rect().width() * 0.62).clamp(420.0, 930.0)
            } else {
                (ctx.screen_rect().width() * 0.86).clamp(560.0, 1180.0)
            };
            egui::Frame::none()
                .fill(egui::Color32::from_rgba_premultiplied(7, 12, 18, 224))
                .stroke(egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgba_premultiplied(120, 160, 184, 98),
                ))
                .inner_margin(egui::Margin::same(
                    if show_start_menu_sidebar { 18.0 } else { 24.0 },
                ))
                .show(ui, |ui| {
                    ui.set_width(width);
                    match hub_ui.screen {
                        HubScreen::StartMenu => {
                            draw_start_menu_center(
                                ui,
                                hub_ui,
                                start_menu,
                                first_launch_lock,
                                slot1,
                                slot2,
                                slot3,
                                autosave,
                            );
                        }
                        HubScreen::CharacterCreationFaction => {
                            character_creation_center::draw_faction_center(
                                ctx,
                                ui,
                                hub_ui,
                                start_menu,
                                character_creation,
                                creation_ui,
                                diplomacy,
                                overworld,
                            );
                        }
                        HubScreen::CharacterCreationBackstory => {
                            character_creation_center::draw_backstory_center(
                                ctx,
                                ui,
                                hub_ui,
                                character_creation,
                                creation_ui,
                                roster,
                                parties,
                                overworld,
                            );
                        }
                        _ => {}
                    }
                });
        });
}

fn draw_hamburger_nav(
    ctx: &egui::Context,
    hub_ui: &mut HubUiState,
    start_menu: &mut StartMenuState,
    settings_menu: &mut SettingsMenuState,
    runtime_mode: &RuntimeModeState,
    can_continue: bool,
    first_launch_lock: bool,
) {
    let nav_width = if start_menu.hamburger_expanded { 340.0 } else { 72.0 };
    egui::SidePanel::left("hub_hamburger_nav")
        .resizable(false)
        .exact_width(nav_width)
        .frame(
            egui::Frame::none()
                .fill(egui::Color32::from_rgba_premultiplied(8, 12, 18, 236))
                .inner_margin(egui::Margin::same(12.0)),
        )
        .show(ctx, |ui| {
            if ui
                .add(egui::Button::new("|||").min_size(egui::vec2(44.0, 34.0)))
                .clicked()
            {
                start_menu.hamburger_expanded = !start_menu.hamburger_expanded;
            }
            if !start_menu.hamburger_expanded {
                return;
            }
            ui.separator();
            ui.heading("Guild Menu");
            ui.small(truncate_for_hud(&start_menu.subtitle, 110));
            ui.separator();
            if ui
                .add(egui::Button::new("New Campaign").min_size(egui::vec2(260.0, 46.0)))
                .clicked()
            {
                hub_ui.request_new_campaign = true;
            }
            if ui
                .add_enabled(
                    can_continue && !first_launch_lock,
                    egui::Button::new("Continue Campaign").min_size(egui::vec2(260.0, 42.0)),
                )
                .clicked()
            {
                hub_ui.request_continue_campaign = true;
            }
            if runtime_mode.dev_mode
                && ui
                    .add(egui::Button::new("Overworld Map (Dev)").min_size(egui::vec2(260.0, 36.0)))
                    .clicked()
            {
                hub_ui.screen = HubScreen::OverworldMap;
            }
            if ui
                .add(egui::Button::new("Settings").min_size(egui::vec2(260.0, 36.0)))
                .clicked()
            {
                settings_menu.is_open = true;
            }
            if ui
                .add(egui::Button::new("Credits").min_size(egui::vec2(260.0, 36.0)))
                .clicked()
            {
                hub_ui.show_credits = true;
            }
            if ui
                .add(egui::Button::new("Quit").min_size(egui::vec2(260.0, 36.0)))
                .clicked()
            {
                hub_ui.request_quit = true;
            }
        });
}

fn draw_start_menu_center(
    ui: &mut egui::Ui,
    _hub_ui: &mut HubUiState,
    start_menu: &StartMenuState,
    first_launch_lock: bool,
    slot1: &str,
    slot2: &str,
    slot3: &str,
    autosave: &str,
) {
    ui.heading("Adventurer's Guild");
    ui.label("A GUI-only command deck with a landscape backsplash.");
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.small(format!("Slot1: {}", slot1));
        ui.small(format!("Slot2: {}", slot2));
        ui.small(format!("Slot3: {}", slot3));
        ui.small(format!("Autosave: {}", autosave));
    });
    if first_launch_lock {
        ui.colored_label(
            egui::Color32::from_rgb(240, 185, 100),
            "First launch detected: start with New Campaign.",
        );
    }
    ui.small(format!(
        "Status: {}",
        truncate_for_hud(&start_menu.status, 120)
    ));
}

/// Side-panel variant of the start menu (used when the left SidePanel is active).
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_start_menu_side_panel(
    ui: &mut egui::Ui,
    hub_ui: &mut HubUiState,
    start_menu: &mut StartMenuState,
    settings_menu: &mut SettingsMenuState,
    runtime_mode: &RuntimeModeState,
    can_continue: bool,
    first_launch_lock: bool,
    slot1: &str,
    slot2: &str,
    slot3: &str,
    autosave: &str,
    save_index: &CampaignSaveIndexState,
    save_panel: &CampaignSavePanelState,
) {
    ui.label(egui::RichText::new("Campaign Start").strong());
    if ui
        .add(egui::Button::new("New Campaign").min_size(egui::vec2(300.0, 48.0)))
        .clicked()
    {
        hub_ui.request_new_campaign = true;
    }
    ui.horizontal_wrapped(|ui| {
        if ui
            .add_enabled(
                can_continue && !first_launch_lock,
                egui::Button::new("Continue Campaign").min_size(egui::vec2(220.0, 40.0)),
            )
            .clicked()
        {
            hub_ui.request_continue_campaign = true;
        }
        if runtime_mode.dev_mode
            && ui
                .add(egui::Button::new("Overworld Map (Dev)").min_size(egui::vec2(180.0, 40.0)))
                .clicked()
        {
            hub_ui.screen = HubScreen::OverworldMap;
        }
        if ui
            .add(egui::Button::new("Settings").min_size(egui::vec2(120.0, 40.0)))
            .clicked()
        {
            settings_menu.is_open = true;
        }
        if ui
            .add(egui::Button::new("Credits").min_size(egui::vec2(110.0, 40.0)))
            .clicked()
        {
            hub_ui.show_credits = true;
        }
        if ui
            .add(egui::Button::new("Quit").min_size(egui::vec2(90.0, 40.0)))
            .clicked()
        {
            hub_ui.request_quit = true;
        }
    });
    if first_launch_lock {
        ui.small("First launch detected: start with New Campaign. Use --dev to bypass.");
    } else if !can_continue {
        ui.small("No compatible saves found. Start a new campaign.");
    }

    ui.separator();
    egui::CollapsingHeader::new("Saves")
        .default_open(false)
        .show(ui, |ui| {
            ui.small("Load shortcuts: F9 slot1 | Shift+F9 slot2 | Ctrl+F9 slot3");
            ui.small("Save panel: F6");
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(13, 16, 22))
                .inner_margin(egui::Margin::same(8.0))
                .show(ui, |ui| {
                    let slot1_badge =
                        format_slot_badge(save_index.index.slots.iter().find(|m| m.slot == "slot1"));
                    let slot2_badge =
                        format_slot_badge(save_index.index.slots.iter().find(|m| m.slot == "slot2"));
                    let slot3_badge =
                        format_slot_badge(save_index.index.slots.iter().find(|m| m.slot == "slot3"));
                    ui.label(format!("{slot1_badge} {slot1}"));
                    ui.label(format!("{slot2_badge} {slot2}"));
                    ui.label(format!("{slot3_badge} {slot3}"));
                    ui.label(format!(
                        "{} {}",
                        format_slot_badge(save_index.index.autosave.as_ref()),
                        autosave
                    ));
                    ui.label(format!(
                        "Panel: {}",
                        if save_panel.open {
                            truncate_for_hud(&save_panel.preview, 96)
                        } else {
                            "closed".to_string()
                        }
                    ));
                });
        });
    ui.small(format!(
        "Status: {}",
        truncate_for_hud(&start_menu.status, 96)
    ));
}
