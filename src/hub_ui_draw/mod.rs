//! Hub UI draw system — extracted from main.rs.
//!
//! The Bevy system `draw_hub_egui_system` lives here. It delegates to
//! sub-module helpers for each logical screen section.

mod start_menu;
mod character_creation_center;
mod character_create;
mod guild;
mod overworld;
mod overworld_map;
mod overworld_map_parties;
mod overworld_map_strategic;
mod region;
mod common;

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::camera::{
    CameraFocusTransitionState,
    OrbitCameraController, SceneViewBounds,
};
use crate::game_core::{self, HubScreen, HubUiState, MissionData, MissionProgress, MissionResult};
use crate::game_core::CharacterCreationState;
use crate::hub_types::{
    CharacterCreationUiState, HubActionQueue, HubMenuState, HeroDetailUiState,
    StartMenuState,
};
use crate::region_nav::{
    RegionLayerTransitionState, RegionTargetPickerState,
};
use crate::local_intro::LocalEagleEyeIntroState;
use crate::runtime_assets::RegionArtState;
use crate::ui::settings::SettingsMenuState;
use crate::game_loop::RuntimeModeState;
use crate::campaign_ops::{
    format_slot_meta, truncate_for_hud,
};
use crate::ui::save_browser::{
    continue_campaign_candidates, CampaignSaveIndexState, CampaignSavePanelState,
};

/// Colour palette for factions. Shared across sub-modules.
pub(crate) const FACTION_PALETTE: [egui::Color32; 6] = [
    egui::Color32::from_rgb(88, 160, 255),
    egui::Color32::from_rgb(226, 130, 70),
    egui::Color32::from_rgb(126, 200, 122),
    egui::Color32::from_rgb(190, 150, 240),
    egui::Color32::from_rgb(230, 205, 90),
    egui::Color32::from_rgb(120, 212, 210),
];

pub(crate) fn faction_color(idx: usize) -> egui::Color32 {
    FACTION_PALETTE[idx % FACTION_PALETTE.len()]
}

/// Pre-computed board rows from the mission query.
pub(crate) type BoardRow = (String, MissionResult, u32, f32, f32, f32);

pub(crate) fn draw_hub_egui_system(
    mut contexts: EguiContexts,
    mut hub_ui: ResMut<HubUiState>,
    mut hub_menu: ResMut<HubMenuState>,
    mut start_menu: ResMut<StartMenuState>,
    runtime_context: (Res<RuntimeModeState>, Res<game_core::AttentionState>),
    mission_query: Query<(&MissionData, &MissionProgress)>,
    camera_context: (Res<SceneViewBounds>, Query<&OrbitCameraController>),
    mut overworld: ResMut<game_core::OverworldMap>,
    mut diplomacy: ResMut<game_core::DiplomacyState>,
    interactions: Res<game_core::InteractionBoard>,
    mut action_queue: ResMut<HubActionQueue>,
    mut character_creation: ResMut<CharacterCreationState>,
    mut creation_ui: ResMut<CharacterCreationUiState>,
    campaign_state: (
        ResMut<game_core::CampaignRoster>,
        ResMut<game_core::CampaignParties>,
        Res<game_core::CampaignLedger>,
        Res<game_core::CampaignEventLog>,
        ResMut<RegionTargetPickerState>,
        ResMut<CameraFocusTransitionState>,
        ResMut<RegionLayerTransitionState>,
        ResMut<LocalEagleEyeIntroState>,
        Res<game_core::FlashpointState>,
        ResMut<HeroDetailUiState>,
    ),
    mut settings_menu: ResMut<SettingsMenuState>,
    save_state: (Res<CampaignSaveIndexState>, Res<CampaignSavePanelState>, Res<RegionArtState>),
) {
    let (runtime_mode, attention) = runtime_context;
    let (bounds, camera_query) = camera_context;
    let (
        mut roster,
        mut parties,
        ledger,
        event_log,
        mut target_picker,
        mut camera_focus_transition,
        mut region_transition,
        mut local_intro,
        flashpoint_state,
        mut hero_detail,
    ) = campaign_state;
    let (save_index, save_panel, region_art) = save_state;

    // Pre-register the region art texture with egui before ctx_mut() borrows contexts.
    let region_art_texture_id: Option<egui::TextureId> = {
        let active_region_id = region_transition.active_payload.as_ref().map(|p| p.region_id);
        region_art
            .texture_handle
            .as_ref()
            .filter(|_| region_art.loaded_region_id == active_region_id && active_region_id.is_some())
            .map(|handle| contexts.add_image(handle.clone()))
    };
    let ctx = contexts.ctx_mut();
    common::apply_hub_style(ctx);

    if hub_ui.screen == HubScreen::BackstoryCinematic {
        return;
    }

    // Pre-compute shared values
    let slot1 = format_slot_meta(save_index.index.slots.iter().find(|m| m.slot == "slot1"));
    let slot2 = format_slot_meta(save_index.index.slots.iter().find(|m| m.slot == "slot2"));
    let slot3 = format_slot_meta(save_index.index.slots.iter().find(|m| m.slot == "slot3"));
    let autosave = format_slot_meta(save_index.index.autosave.as_ref());
    let continue_candidates = continue_campaign_candidates(&save_index.index);
    let can_continue = !continue_candidates.is_empty();
    let party_snapshots = parties.parties.clone();
    let first_launch_lock = !runtime_mode.dev_mode && !can_continue;
    let transition_locked = region_transition.interaction_locked;
    let panel_width = (ctx.screen_rect().width() * 0.42).clamp(420.0, 680.0);
    let panel_min_width = if ctx.screen_rect().width() < 1100.0 { 320.0 } else { 420.0 };

    let mut board_rows: Vec<BoardRow> = mission_query
        .iter()
        .map(|(data, progress)| {
            (
                data.mission_name.clone(),
                progress.result,
                progress.turns_remaining,
                progress.sabotage_progress,
                progress.alert_level,
                progress.reactor_integrity,
            )
        })
        .collect();
    board_rows.sort_by(|a, b| b.4.total_cmp(&a.4));

    // -----------------------------------------------------------------------
    // GUI-only screens (StartMenu, CharCreation) — full-screen overlay
    // -----------------------------------------------------------------------
    let gui_only_screen = matches!(
        hub_ui.screen,
        HubScreen::StartMenu
            | HubScreen::CharacterCreationFaction
            | HubScreen::CharacterCreationBackstory
    );
    if gui_only_screen {
        start_menu::draw_gui_only_screens(
            ctx,
            &mut hub_ui,
            &mut start_menu,
            &mut settings_menu,
            &mut character_creation,
            &mut creation_ui,
            &mut diplomacy,
            &mut roster,
            &mut parties,
            &overworld,
            can_continue,
            first_launch_lock,
            &runtime_mode,
            &slot1,
            &slot2,
            &slot3,
            &autosave,
        );
        if hub_ui.show_credits {
            common::draw_credits_window(ctx, &mut hub_ui);
        }
        return;
    }

    // -----------------------------------------------------------------------
    // Active Crises + Faction Strength right panel — OverworldMap only
    // -----------------------------------------------------------------------
    if hub_ui.screen == HubScreen::OverworldMap {
        overworld_map_parties::draw_crises_right_panel(ctx, &overworld, &flashpoint_state);
    }

    // -----------------------------------------------------------------------
    // Main left side-panel
    // -----------------------------------------------------------------------
    egui::SidePanel::left("hub_panel")
        .frame(
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(10, 12, 16))
                .inner_margin(egui::Margin::same(12.0)),
        )
        .resizable(false)
        .default_width(panel_width)
        .min_width(panel_min_width)
        .show(ctx, |ui| {
            ui.heading("Adventurer's Guild Hub");
            let subtitle = if hub_ui.screen == HubScreen::StartMenu {
                &start_menu.subtitle
            } else {
                &hub_menu.notice
            };
            ui.label(egui::RichText::new(truncate_for_hud(subtitle, 110)).italics());
            ui.separator();

            match hub_ui.screen {
                HubScreen::StartMenu => {
                    start_menu::draw_start_menu_side_panel(
                        ui,
                        &mut hub_ui,
                        &mut start_menu,
                        &mut settings_menu,
                        &runtime_mode,
                        can_continue,
                        first_launch_lock,
                        &slot1,
                        &slot2,
                        &slot3,
                        &autosave,
                        &save_index,
                        &save_panel,
                    );
                }
                HubScreen::CharacterCreationFaction => {
                    character_create::draw_faction_side_panel(
                        ui,
                        &mut hub_ui,
                        &mut start_menu,
                        &mut character_creation,
                        &mut creation_ui,
                        &mut diplomacy,
                        &overworld,
                        &runtime_mode,
                    );
                }
                HubScreen::CharacterCreationBackstory => {
                    character_create::draw_backstory_side_panel(
                        ui,
                        &mut hub_ui,
                        &mut character_creation,
                        &mut creation_ui,
                        &mut roster,
                        &mut parties,
                        &overworld,
                        &runtime_mode,
                    );
                }
                HubScreen::GuildManagement => {
                    guild::draw_guild_management(
                        ui,
                        &mut hub_ui,
                        &mut hub_menu,
                        &mut start_menu,
                        &mut action_queue,
                        &mut roster,
                        &mut hero_detail,
                        &attention,
                        &ledger,
                        &board_rows,
                    );
                }
                HubScreen::Overworld => {
                    overworld::draw_overworld_details(
                        ui,
                        &mut hub_ui,
                        &mut start_menu,
                        &mut overworld,
                        &diplomacy,
                        &interactions,
                    );
                }
                HubScreen::OverworldMap => {
                    overworld_map::draw_overworld_map_panel(
                        ui,
                        &mut hub_ui,
                        &mut hub_menu,
                        &mut start_menu,
                        &mut overworld,
                        &mut parties,
                        &mut roster,
                        &mut target_picker,
                        &mut camera_focus_transition,
                        &mut region_transition,
                        &character_creation,
                        &camera_query,
                        &bounds,
                        &party_snapshots,
                        &runtime_mode,
                        transition_locked,
                    );
                }
                HubScreen::RegionView => {
                    region::draw_region_view(
                        ui,
                        &mut hub_ui,
                        &mut hub_menu,
                        &mut local_intro,
                        &mut parties,
                        &region_transition,
                        &overworld,
                        &region_art,
                        region_art_texture_id,
                    );
                }
                HubScreen::LocalEagleEyeIntro => {
                    region::draw_local_eagle_eye_intro(
                        ui,
                        &mut hub_ui,
                        &local_intro,
                        &overworld,
                    );
                }
                HubScreen::BackstoryCinematic => {
                    ui.label(
                        "Backstory cinematic is rendering in the full-screen cinematic layer.",
                    );
                }
                HubScreen::MissionExecution => {
                    ui.label("Mission in progress.");
                }
            }

            ui.separator();
            ui.small("Global: F6 save panel | F5/F9 slot1 | Shift+F5/F9 slot2 | Ctrl+F5/F9 slot3 | Esc settings");
            if !event_log.entries.is_empty() {
                ui.small(format!(
                    "Recent events tracked: {}",
                    event_log.entries.len()
                ));
            }
        });

    if hub_ui.show_credits {
        common::draw_credits_window(ctx, &mut hub_ui);
    }
}
