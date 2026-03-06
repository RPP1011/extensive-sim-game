use bevy::app::AppExit;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::game_core::{self, HubScreen, HubUiState};
use crate::hub_types::{CampaignOutcomeState, HubMenuState, StartMenuState};
use crate::local_intro::{self, LocalEagleEyeIntroState};
use crate::region_nav::{RegionLayerTransitionState, advance_region_layer_transition};
use crate::runtime_assets::RuntimeAssetGenState;
use crate::campaign_ops::truncate_for_hud;

pub fn campaign_outcome_check_system(
    flashpoint: Res<game_core::FlashpointState>,
    roster: Res<game_core::CampaignRoster>,
    mut outcome_state: ResMut<CampaignOutcomeState>,
) {
    if outcome_state.shown {
        return;
    }
    if let Some(outcome) = game_core::check_campaign_outcome(&flashpoint, &roster) {
        outcome_state.outcome = Some(outcome);
        outcome_state.shown = true;
    }
}

/// Renders a full-screen egui overlay when a campaign outcome (victory or
/// defeat) has been detected.
pub fn campaign_outcome_ui_system(
    mut contexts: EguiContexts,
    mut outcome_state: ResMut<CampaignOutcomeState>,
    mut hub_ui: ResMut<HubUiState>,
    mut start_menu: ResMut<StartMenuState>,
    run_state: Res<game_core::RunState>,
    roster: Res<game_core::CampaignRoster>,
    flashpoint: Res<game_core::FlashpointState>,
    ledger: Res<game_core::CampaignLedger>,
) {
    let outcome = match outcome_state.outcome {
        Some(o) => o,
        None => return,
    };

    let ctx = contexts.ctx_mut();
    let screen = ctx.screen_rect();
    let window_style = (*ctx.style()).clone();

    let title = match outcome {
        game_core::CampaignOutcome::Victory => "CAMPAIGN COMPLETE",
        game_core::CampaignOutcome::Defeat => "YOUR COMPANY FALLS",
    };

    let mut new_campaign_clicked = false;

    egui::Window::new(title)
    .collapsible(false)
    .resizable(false)
    .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
    .fixed_size(egui::vec2(screen.width().min(640.0), screen.height().min(520.0)))
    .frame(
        egui::Frame::window(&window_style)
            .fill(egui::Color32::from_rgba_premultiplied(8, 10, 18, 248)),
    )
    .show(ctx, |ui| {
        ui.set_min_size(ui.available_size());
        ui.vertical_centered(|ui| {
            match outcome {
                game_core::CampaignOutcome::Victory => {
                    ui.colored_label(
                        egui::Color32::from_rgb(230, 210, 100),
                        egui::RichText::new("CAMPAIGN COMPLETE")
                            .heading()
                            .strong(),
                    );
                    ui.separator();
                    ui.label("The realm is secured. Your legend is written.");
                    ui.separator();
                    let heroes_alive = game_core::active_hero_count(&roster);
                    let missions_completed = ledger
                        .records
                        .iter()
                        .filter(|r| r.result == game_core::MissionResult::Victory)
                        .count();
                    let chains_completed = flashpoint.chains.iter().filter(|c| c.completed).count();
                    ui.label(format!("Turns taken: {}", run_state.global_turn));
                    ui.label(format!("Heroes still active: {}", heroes_alive));
                    ui.label(format!("Missions completed: {}", missions_completed));
                    ui.label(format!("Flashpoint chains resolved: {}", chains_completed));
                }
                game_core::CampaignOutcome::Defeat => {
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 80, 80),
                        egui::RichText::new("YOUR COMPANY FALLS")
                            .heading()
                            .strong(),
                    );
                    ui.separator();
                    ui.label("All heroes are lost. The realm descends into chaos.");
                    ui.separator();
                    let turns_lasted = run_state.global_turn;
                    let farthest = flashpoint
                        .chains
                        .iter()
                        .map(|c| c.stage)
                        .max()
                        .unwrap_or(0);
                    ui.label(format!("Turns lasted: {}", turns_lasted));
                    ui.label(format!("Farthest flashpoint stage reached: {}", farthest));
                }
            }
            ui.separator();
            if ui
                .add(egui::Button::new("New Campaign").min_size(egui::vec2(200.0, 44.0)))
                .clicked()
            {
                new_campaign_clicked = true;
            }
        });
    });

    if new_campaign_clicked {
        outcome_state.outcome = None;
        outcome_state.shown = false;
        hub_ui.request_new_campaign = true;
        crate::campaign_ops::enter_start_menu(&mut hub_ui, &mut start_menu);
    }
}

pub fn draw_runtime_asset_gen_egui_system(
    mut contexts: EguiContexts,
    hub_ui: Res<HubUiState>,
    mut runtime_asset_gen: ResMut<RuntimeAssetGenState>,
    runtime_asset_preview: Res<crate::runtime_assets::RuntimeAssetPreviewState>,
) {
    if matches!(
        hub_ui.screen,
        HubScreen::BackstoryCinematic
            | HubScreen::CharacterCreationFaction
            | HubScreen::CharacterCreationBackstory
    ) {
        return;
    }
    let preview_texture_id = runtime_asset_preview
        .texture_handle
        .as_ref()
        .map(|handle| contexts.add_image(handle.clone()));

    let is_menu_layer = matches!(
        hub_ui.screen,
        HubScreen::StartMenu | HubScreen::CharacterCreationFaction | HubScreen::CharacterCreationBackstory
    );
    let window_anchor = if is_menu_layer {
        egui::Align2::RIGHT_BOTTOM
    } else {
        egui::Align2::RIGHT_TOP
    };
    egui::Window::new("Runtime Asset Gen")
        .default_width(420.0)
        .resizable(false)
        .collapsible(true)
        .anchor(window_anchor, egui::vec2(-16.0, if is_menu_layer { -16.0 } else { 16.0 }))
        .show(contexts.ctx_mut(), |ui| {
            ui.small(truncate_for_hud(&runtime_asset_gen.status, 140));
            ui.separator();
            ui.small(format!(
                "Provider: Gemini (swappable) | Model: {}",
                runtime_asset_gen.model
            ));
            ui.small(format!(
                "Pending: {} | In-flight: {}/{}",
                runtime_asset_gen.pending.len(),
                runtime_asset_gen.in_flight_jobs,
                runtime_asset_gen.max_parallel_jobs
            ));
            if ui.button("Queue 2 More Environment Concepts").clicked() {
                match crate::runtime_assets::queue_runtime_environment_jobs(&mut runtime_asset_gen, 2) {
                    Ok(count) => {
                        runtime_asset_gen.status =
                            format!("Queued {} additional runtime environment jobs.", count);
                    }
                    Err(err) => {
                        runtime_asset_gen.status = err;
                    }
                }
            }
            ui.separator();
            for recent in runtime_asset_gen.recent.iter().take(4) {
                let marker = if recent.success { "OK" } else { "ERR" };
                ui.small(format!(
                    "[{}] #{} {} ({})",
                    marker, recent.job_id, recent.source_title, recent.source_id
                ));
                ui.small(format!("prompt: {}", recent.prompt_file.display()));
                if let Some(path) = &recent.image_file {
                    ui.small(format!("image: {}", path.display()));
                }
            }
            if let Some(texture_id) = preview_texture_id {
                ui.separator();
                ui.label(egui::RichText::new("Latest Generated Preview").strong());
                let max_w = ui.available_width().max(120.0);
                let size = egui::vec2(max_w, max_w * 0.56);
                ui.image((texture_id, size));
            } else if let Some(err) = &runtime_asset_preview.last_error {
                ui.separator();
                ui.colored_label(egui::Color32::from_rgb(235, 95, 95), err);
            }
        });
}

pub fn region_layer_transition_system(
    mut hub_ui: ResMut<HubUiState>,
    mut region_transition: ResMut<RegionLayerTransitionState>,
    overworld: Res<game_core::OverworldMap>,
    mut hub_menu: ResMut<HubMenuState>,
) {
    if let Some(status) =
        advance_region_layer_transition(&mut hub_ui, &mut region_transition, &overworld)
    {
        hub_menu.notice = status;
    }
}

pub fn local_intro_sequence_system(
    mut hub_ui: ResMut<HubUiState>,
    mut local_intro: ResMut<LocalEagleEyeIntroState>,
    mut hub_menu: ResMut<HubMenuState>,
) {
    if hub_ui.screen != HubScreen::LocalEagleEyeIntro {
        return;
    }
    if let Some(status) = local_intro::advance_local_eagle_eye_intro(&mut local_intro) {
        hub_menu.notice = status;
    }
    // When the intro completes and input is handed off, enter the mission.
    if local_intro.input_handoff_ready {
        hub_ui.screen = HubScreen::MissionExecution;
    }
}

pub fn hub_quit_requested_system(
    mut hub_ui: ResMut<HubUiState>,
    mut app_exit_events: EventWriter<AppExit>,
) {
    if hub_ui.request_quit {
        hub_ui.request_quit = false;
        app_exit_events.send(AppExit);
    }
}
