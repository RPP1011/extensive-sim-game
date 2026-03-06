//! Overworld map — player party panel, crises panel, delegated party controls,
//! faction control, and focus helpers.

use bevy::prelude::*;
use bevy_egui::egui;

use crate::camera::{
    CameraFocusTransitionQueueResult, CameraFocusTransitionState, CameraFocusTrigger,
    OrbitCameraController, SceneViewBounds,
};
use crate::campaign_ops::truncate_for_hud;
use crate::game_core;
use crate::region_nav::{
    RegionTargetPickerState,
    begin_region_target_picker,
    confirm_region_target_picker, cancel_region_target_picker,
    party_target_region_label, party_panel_label, default_camera_focus,
    queue_party_camera_focus_transition, transfer_direct_command_to_selected,
};
use super::faction_color;

/// Draw the right-side crises + faction strength panel (OverworldMap only).
pub(crate) fn draw_crises_right_panel(
    ctx: &egui::Context,
    overworld: &game_core::OverworldMap,
    flashpoint_state: &game_core::FlashpointState,
) {
    egui::SidePanel::right("hub_crises_panel")
        .frame(
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(10, 12, 16))
                .inner_margin(egui::Margin::same(10.0)),
        )
        .resizable(false)
        .default_width(240.0)
        .min_width(180.0)
        .show(ctx, |ui| {
            ui.label(egui::RichText::new("Active Crises").strong());
            ui.separator();
            if flashpoint_state.chains.is_empty() {
                ui.small("No active flashpoint chains.");
            } else {
                for chain in &flashpoint_state.chains {
                    let status_label = if chain.completed { "Completed" } else { "Active" };
                    let intent_label = match chain.intent {
                        game_core::FlashpointIntent::StealthPush => "Stealth Push",
                        game_core::FlashpointIntent::DirectAssault => "Direct Assault",
                        game_core::FlashpointIntent::CivilianFirst => "Civilian First",
                    };
                    egui::Frame::none()
                        .fill(egui::Color32::from_rgba_premultiplied(20, 24, 32, 180))
                        .inner_margin(egui::Margin::same(4.0))
                        .show(ui, |ui| {
                            let title = if chain.objective.is_empty() {
                                format!("Chain #{}", chain.id)
                            } else {
                                truncate_for_hud(&chain.objective, 36).to_string()
                            };
                            ui.small(egui::RichText::new(title).strong());
                            ui.small(format!("Stage {} / 3", chain.stage));
                            ui.small(format!("Status: {}", status_label));
                            ui.small(format!("Intent: {}", intent_label));
                            ui.small(format!("Region: {}", chain.region_id));
                        });
                    ui.add_space(2.0);
                }
            }
            ui.separator();
            ui.label(egui::RichText::new("Faction Strength").strong());
            ui.separator();
            for faction in &overworld.factions {
                let color = faction_color(faction.id);
                ui.colored_label(color, egui::RichText::new(&faction.name).small().strong());
                let strength_norm = (faction.strength / 100.0).clamp(0.0, 1.0);
                let bar_total = ui.available_width().max(60.0);
                let bar_filled = bar_total * strength_norm;
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(bar_total, 8.0), egui::Sense::hover());
                ui.painter().rect_filled(
                    rect,
                    2.0,
                    egui::Color32::from_rgba_premultiplied(40, 44, 52, 200),
                );
                if bar_filled > 0.0 {
                    let filled_rect = egui::Rect::from_min_size(
                        rect.min,
                        egui::vec2(bar_filled, rect.height()),
                    );
                    ui.painter().rect_filled(filled_rect, 2.0, color);
                }
                ui.small(format!("{:.0}%", faction.strength.clamp(0.0, 100.0)));
                ui.add_space(4.0);
            }
        });
}

pub(crate) fn draw_faction_control(ui: &mut egui::Ui, overworld: &game_core::OverworldMap) {
    ui.label(egui::RichText::new("Faction Control").strong());
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(13, 16, 22))
        .inner_margin(egui::Margin::same(8.0))
        .show(ui, |ui| {
            let mut territory = vec![0usize; overworld.factions.len()];
            for region in &overworld.regions {
                if region.owner_faction_id < territory.len() {
                    territory[region.owner_faction_id] += 1;
                }
            }
            for faction in &overworld.factions {
                let color = faction_color(faction.id);
                let owned = territory.get(faction.id).copied().unwrap_or(0);
                ui.colored_label(
                    color,
                    format!(
                        "{}: {} regions | strength {:.0} | cohesion {:.0}",
                        faction.name, owned, faction.strength, faction.cohesion
                    ),
                );
            }
        });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_player_parties(
    ui: &mut egui::Ui,
    overworld: &mut game_core::OverworldMap,
    parties: &mut game_core::CampaignParties,
    roster: &mut game_core::CampaignRoster,
    target_picker: &mut RegionTargetPickerState,
    camera_focus_transition: &mut CameraFocusTransitionState,
    camera_query: &Query<&OrbitCameraController>,
    bounds: &SceneViewBounds,
) {
    ui.label(egui::RichText::new("Player Parties").strong());
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(13, 16, 22))
        .inner_margin(egui::Margin::same(8.0))
        .show(ui, |ui| {
            if let Some(active_transition) = camera_focus_transition.active.as_ref() {
                let party_name = parties
                    .parties
                    .iter()
                    .find(|party| party.id == active_transition.target_party_id)
                    .map(|party| party.name.as_str())
                    .unwrap_or("selected party");
                ui.colored_label(
                    egui::Color32::from_rgb(112, 207, 242),
                    format!(
                        "Camera transition: {} -> {} ({}%, region {}). Pan/orbit/zoom locked until complete.",
                        active_transition.trigger.label(),
                        party_name,
                        (active_transition.progress() * 100.0).round() as u32,
                        active_transition.target_region_id + 1
                    ),
                );
                ui.small("New focus actions safely retarget the active transition.");
            }
            let mut set_selected = parties.selected_party_id;
            for party in &parties.parties {
                let leader = roster
                    .heroes
                    .iter()
                    .find(|h| h.id == party.leader_hero_id)
                    .map(|h| h.name.as_str())
                    .unwrap_or("Unknown");
                let region = overworld
                    .regions
                    .get(party.region_id)
                    .map(|r| r.name.as_str())
                    .unwrap_or("Unknown");
                let target_region = party_target_region_label(party, overworld);
                let label = party_panel_label(
                    party,
                    parties.selected_party_id == Some(party.id),
                    leader,
                    region,
                    &target_region,
                );
                if ui
                    .selectable_label(parties.selected_party_id == Some(party.id), label)
                    .clicked()
                {
                    set_selected = Some(party.id);
                }
            }
            parties.selected_party_id = set_selected;
            if let Some(active_party_id) = target_picker.active_party_id() {
                if parties.selected_party_id != Some(active_party_id) {
                    target_picker.clear();
                }
            }
            if let Some(selected) = parties.selected_party_id {
                if let Some(party_idx) =
                    parties.parties.iter().position(|p| p.id == selected)
                {
                    if !parties.parties[party_idx].is_player_controlled {
                        draw_delegated_party_controls(
                            ui,
                            overworld,
                            parties,
                            roster,
                            target_picker,
                            camera_focus_transition,
                            camera_query,
                            bounds,
                            party_idx,
                        );
                    } else {
                        if target_picker.is_active_for_party(parties.parties[party_idx].id) {
                            target_picker.clear();
                        }
                        ui.small(
                            "Player party uses direct control; select a delegated party for orders.",
                        );
                        if ui.button("Focus Selected Party").clicked() {
                            focus_player_party(
                                ui,
                                overworld,
                                parties,
                                camera_focus_transition,
                                camera_query,
                                bounds,
                                party_idx,
                            );
                        }
                    }
                }
            }
            ui.small(format!(
                "Party Notice: {}",
                truncate_for_hud(&parties.notice, 96)
            ));
        });
}

#[allow(clippy::too_many_arguments)]
fn draw_delegated_party_controls(
    ui: &mut egui::Ui,
    overworld: &mut game_core::OverworldMap,
    parties: &mut game_core::CampaignParties,
    roster: &mut game_core::CampaignRoster,
    target_picker: &mut RegionTargetPickerState,
    camera_focus_transition: &mut CameraFocusTransitionState,
    camera_query: &Query<&OrbitCameraController>,
    bounds: &SceneViewBounds,
    party_idx: usize,
) {
    let party_id = parties.parties[party_idx].id;
    let party_name = parties.parties[party_idx].name.clone();
    let active_target = parties.parties[party_idx]
        .order_target_region_id
        .and_then(|id| overworld.regions.get(id).map(|r| r.name.as_str()))
        .unwrap_or("none")
        .to_string();
    let picker_pending_region = target_picker
        .selected_region_id()
        .and_then(|id| overworld.regions.get(id))
        .map(|r| r.name.clone())
        .unwrap_or_else(|| "none".to_string());
    ui.horizontal_wrapped(|ui| {
        ui.small("Target Region:");
        ui.small(active_target);
    });
    if target_picker.is_active_for_party(party_id) {
        ui.small("Picker active: click a map region, then confirm or cancel.");
        ui.horizontal_wrapped(|ui| {
            ui.small(format!("Selected in picker: {}", picker_pending_region));
            if ui.button("Confirm Target").clicked() {
                match confirm_region_target_picker(target_picker, parties, party_id, overworld) {
                    Ok(notice) => parties.notice = notice,
                    Err(reason) => parties.notice = reason,
                }
            }
            if ui.button("Cancel Picker").clicked() {
                match cancel_region_target_picker(target_picker, party_id) {
                    Ok(notice) => parties.notice = notice,
                    Err(reason) => parties.notice = reason,
                }
            }
        });
    } else if ui.button("Pick Region Target").clicked() {
        let party_ref = &parties.parties[party_idx];
        parties.notice = begin_region_target_picker(target_picker, party_ref);
    }

    let mut order_update: Option<(game_core::PartyOrderKind, Option<usize>, String)> = None;
    let fallback_target_region = overworld
        .selected_region
        .min(overworld.regions.len().saturating_sub(1));
    let assigned_target_region = parties.parties[party_idx].order_target_region_id;
    ui.separator();
    ui.small("Delegated Orders:");
    ui.horizontal_wrapped(|ui| {
        if ui.button("Hold").clicked() {
            order_update = Some((
                game_core::PartyOrderKind::HoldPosition,
                None,
                format!("{} ordered to hold position.", party_name),
            ));
        }
        if ui.button("Patrol").clicked() {
            let target = assigned_target_region;
            let target_suffix = target
                .and_then(|id| overworld.regions.get(id))
                .map(|r| format!(" toward {}.", r.name))
                .unwrap_or_else(|| ".".to_string());
            order_update = Some((
                game_core::PartyOrderKind::PatrolNearby,
                target,
                format!("{} ordered to patrol nearby{}", party_name, target_suffix),
            ));
        }
        if ui.button("Reinforce Selected").clicked() {
            let target_region = assigned_target_region.unwrap_or(fallback_target_region);
            order_update = Some((
                game_core::PartyOrderKind::ReinforceFront,
                Some(target_region),
                format!(
                    "{} ordered to reinforce {}.",
                    party_name,
                    overworld
                        .regions
                        .get(target_region)
                        .map(|r| r.name.as_str())
                        .unwrap_or("selected region")
                ),
            ));
        }
        if ui.button("Recruit/Train").clicked() {
            let target_region = assigned_target_region.unwrap_or(fallback_target_region);
            order_update = Some((
                game_core::PartyOrderKind::RecruitAndTrain,
                Some(target_region),
                format!("{} ordered to recruit and train.", party_name),
            ));
        }
        if ui.button("Take Command").clicked() {
            match transfer_direct_command_to_selected(parties) {
                Ok(handoff) => {
                    target_picker.clear();
                    overworld.current_region = handoff.new_region_id;
                    overworld.selected_region = handoff.new_region_id;
                    roster.player_hero_id = Some(handoff.new_leader_hero_id);
                    let camera_focus_start = camera_query
                        .iter()
                        .next()
                        .map(|controller| controller.focus)
                        .unwrap_or_else(|| default_camera_focus(bounds));
                    let transition_notice = match queue_party_camera_focus_transition(
                        camera_focus_transition,
                        camera_focus_start,
                        overworld,
                        bounds,
                        handoff.new_party_id,
                        handoff.new_region_id,
                        CameraFocusTrigger::TakeCommand,
                    ) {
                        Ok(CameraFocusTransitionQueueResult::Started) => {
                            "Camera refocus started."
                        }
                        Ok(CameraFocusTransitionQueueResult::Retargeted) => {
                            "Camera refocus retargeted."
                        }
                        Err(reason) => {
                            parties.notice = reason;
                            ""
                        }
                    };
                    if !transition_notice.is_empty() {
                        parties.notice = format!(
                            "Command handed off: {} now directly controlled; {} delegated. {}",
                            handoff.new_party_name,
                            handoff.previous_party_name,
                            transition_notice
                        );
                    }
                }
                Err(reason) => {
                    parties.notice = reason;
                }
            }
        }
    });
    if let Some((order, target, notice)) = order_update {
        let party = &mut parties.parties[party_idx];
        party.order = order;
        party.order_target_region_id = target;
        parties.notice = notice;
    }
}

fn focus_player_party(
    _ui: &mut egui::Ui,
    overworld: &mut game_core::OverworldMap,
    parties: &mut game_core::CampaignParties,
    camera_focus_transition: &mut CameraFocusTransitionState,
    camera_query: &Query<&OrbitCameraController>,
    bounds: &SceneViewBounds,
    party_idx: usize,
) {
    let party_id = parties.parties[party_idx].id;
    let region_id = parties.parties[party_idx].region_id;
    overworld.current_region = region_id;
    overworld.selected_region = region_id;
    let camera_focus_start = camera_query
        .iter()
        .next()
        .map(|controller| controller.focus)
        .unwrap_or_else(|| default_camera_focus(bounds));
    let transition_notice = match queue_party_camera_focus_transition(
        camera_focus_transition,
        camera_focus_start,
        overworld,
        bounds,
        party_id,
        region_id,
        CameraFocusTrigger::FocusSelectedParty,
    ) {
        Ok(CameraFocusTransitionQueueResult::Started) => "Camera refocus started.",
        Ok(CameraFocusTransitionQueueResult::Retargeted) => "Camera refocus retargeted.",
        Err(reason) => {
            parties.notice = reason;
            ""
        }
    };
    if !transition_notice.is_empty() {
        parties.notice = format!(
            "Focused on {}. {}",
            parties.parties[party_idx].name, transition_notice
        );
    }
}
