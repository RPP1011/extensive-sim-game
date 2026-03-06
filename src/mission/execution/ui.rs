use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::ai::core::{IntentAction, Team, UnitIntent};
use crate::game_core::{HubScreen, HubUiState, RunState};
use crate::mission::{
    room_sequence::MissionRoomSequence,
    sim_bridge::{threat_level, threat_level_roman, MissionOutcome, MissionSimState},
    unit_vis::UnitSelection,
};

// ---------------------------------------------------------------------------
// UI systems
// ---------------------------------------------------------------------------

/// Draws the in-mission egui HUD, outcome panel, and retreat button.
pub fn mission_outcome_ui_system(
    mut contexts: EguiContexts,
    sim_state: Option<Res<MissionSimState>>,
    room_seq: Option<Res<MissionRoomSequence>>,
    run_state: Option<Res<RunState>>,
    mut hub_ui: ResMut<HubUiState>,
) {
    let ctx = contexts.ctx_mut();

    let Some(sim) = sim_state else {
        return;
    };

    let global_turn = run_state.as_ref().map(|r| r.global_turn).unwrap_or(0);
    let tl = threat_level(global_turn);
    let tl_roman = threat_level_roman(tl);

    let rooms_cleared = room_seq
        .as_ref()
        .map(|s| s.current_index + 1)
        .unwrap_or(1);

    match sim.outcome {
        Some(MissionOutcome::Victory) => {
            let heroes_survived = sim
                .sim
                .units
                .iter()
                .filter(|u| u.team == Team::Hero && u.hp > 0)
                .count();
            let enemies_defeated = sim
                .sim
                .units
                .iter()
                .filter(|u| u.team == Team::Enemy && u.hp <= 0)
                .count();

            let mut return_clicked = false;
            egui::Window::new("Mission Result##victory")
                .collapsible(false)
                .resizable(false)
                .fixed_size(egui::vec2(400.0, 300.0))
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.heading("VICTORY");
                    ui.separator();
                    ui.label(format!("Threat Level: {}", tl_roman));
                    ui.label(format!("Heroes survived: {}", heroes_survived));
                    ui.label(format!("Enemies defeated: {}", enemies_defeated));
                    ui.label(format!("Rooms cleared: {}", rooms_cleared));
                    ui.separator();
                    ui.label("Loot: None");
                    ui.separator();
                    if ui.button("Return to Overworld").clicked() {
                        return_clicked = true;
                    }
                });
            if return_clicked {
                hub_ui.screen = HubScreen::OverworldMap;
            }
        }
        Some(MissionOutcome::Defeat) => {
            let enemies_remaining = sim
                .sim
                .units
                .iter()
                .filter(|u| u.team == Team::Enemy && u.hp > 0)
                .count();

            let mut return_clicked = false;
            egui::Window::new("Mission Result##defeat")
                .collapsible(false)
                .resizable(false)
                .fixed_size(egui::vec2(400.0, 300.0))
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.heading("DEFEAT");
                    ui.separator();
                    ui.label("All heroes have fallen.");
                    ui.separator();
                    ui.label(format!("Enemies remaining: {}", enemies_remaining));
                    ui.label(format!("Rooms reached: {}", rooms_cleared));
                    ui.separator();
                    if ui.button("Return to Overworld").clicked() {
                        return_clicked = true;
                    }
                });
            if return_clicked {
                hub_ui.screen = HubScreen::OverworldMap;
            }
        }
        None => {
            let heroes_alive = sim
                .sim
                .units
                .iter()
                .filter(|u| u.team == Team::Hero && u.hp > 0)
                .count();
            let enemies_alive = sim
                .sim
                .units
                .iter()
                .filter(|u| u.team == Team::Enemy && u.hp > 0)
                .count();

            let mut retreat_clicked = false;
            egui::Window::new("Mission HUD")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::LEFT_TOP, egui::vec2(8.0, 8.0))
                .show(ctx, |ui| {
                    ui.label(format!("Threat Level: {}", tl_roman));
                    ui.label(format!("Units: {} alive", heroes_alive));
                    ui.label(format!("Enemies: {} alive", enemies_alive));
                    ui.separator();
                    if ui.button("Retreat").clicked() {
                        retreat_clicked = true;
                    }
                });

            if retreat_clicked {
                hub_ui.screen = HubScreen::OverworldMap;
            }
        }
    }
}

/// Draws the ability HUD panel at the bottom of the screen for the selected unit.
pub fn ability_hud_system(
    mut contexts: EguiContexts,
    selection: Option<Res<UnitSelection>>,
    mut sim_state: Option<ResMut<MissionSimState>>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    let (Some(sel), Some(ref mut sim)) = (selection, sim_state.as_mut()) else {
        return;
    };

    let selected_id = match sel.selected_ids.first().copied() {
        Some(id) => id,
        None => return,
    };

    let unit = match sim.sim.units.iter().find(|u| u.id == selected_id && u.hp > 0) {
        Some(u) => u.clone(),
        None => return,
    };

    let nearest_enemy_id: Option<u32> = sim
        .sim
        .units
        .iter()
        .filter(|u| u.team == Team::Enemy && u.hp > 0)
        .map(|u| {
            let dx = u.position.x - unit.position.x;
            let dy = u.position.y - unit.position.y;
            let dist_sq = dx * dx + dy * dy;
            (u.id, dist_sq)
        })
        .min_by(|(_, da), (_, db)| da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, _)| id);

    let ctx = contexts.ctx_mut();

    egui::Window::new("Ability HUD")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -8.0))
        .show(ctx, |ui| {
            let label = if unit.team == Team::Hero {
                format!("Selected: Hero #{}", unit.id)
            } else {
                format!("Selected: Unit #{}", unit.id)
            };
            ui.label(label);

            ui.label(format!("HP: {} / {}", unit.hp, unit.max_hp));

            if let Some(cast) = &unit.casting {
                ui.label(format!("Casting... ({} ms remaining)", cast.remaining_ms));
            }

            ui.separator();

            ui.horizontal(|ui| {
                // [Q] Attack
                let atk_cd = unit.cooldown_remaining_ms;
                let atk_label = if atk_cd > 0 {
                    format!("[Q] Attack ({}ms)", atk_cd)
                } else {
                    "[Q] Attack".to_owned()
                };
                let q_pressed = ui
                    .add_enabled(atk_cd == 0, egui::Button::new(atk_label))
                    .clicked()
                    || (keyboard.just_pressed(KeyCode::KeyQ) && atk_cd == 0);

                // [W] CastAbility
                let ab_cd = unit.ability_cooldown_remaining_ms;
                let ab_label = if ab_cd > 0 {
                    format!("[W] Ability ({}ms)", ab_cd)
                } else {
                    "[W] Ability".to_owned()
                };
                let w_pressed = ui
                    .add_enabled(ab_cd == 0, egui::Button::new(ab_label))
                    .clicked()
                    || (keyboard.just_pressed(KeyCode::KeyW) && ab_cd == 0);

                // [E] CastHeal (hero) or CastControl
                let e_unit_id = unit.id;
                let e_is_hero = unit.team == Team::Hero;
                let e_cd = if e_is_hero {
                    unit.heal_cooldown_remaining_ms
                } else {
                    unit.control_cooldown_remaining_ms
                };
                let e_label_base = if e_is_hero { "Heal" } else { "Control" };
                let e_label = if e_cd > 0 {
                    format!("[E] {} ({}ms)", e_label_base, e_cd)
                } else {
                    format!("[E] {}", e_label_base)
                };
                let e_pressed = ui
                    .add_enabled(e_cd == 0, egui::Button::new(e_label))
                    .clicked()
                    || (keyboard.just_pressed(KeyCode::KeyE) && e_cd == 0);

                // Dispatch intents
                if q_pressed {
                    if let Some(target_id) = nearest_enemy_id {
                        sim.hero_intents.retain(|i| i.unit_id != selected_id);
                        sim.hero_intents.push(UnitIntent {
                            unit_id: selected_id,
                            action: IntentAction::Attack { target_id },
                        });
                    }
                }
                if w_pressed {
                    if let Some(target_id) = nearest_enemy_id {
                        sim.hero_intents.retain(|i| i.unit_id != selected_id);
                        sim.hero_intents.push(UnitIntent {
                            unit_id: selected_id,
                            action: IntentAction::CastAbility { target_id },
                        });
                    }
                }
                if e_pressed {
                    let action = if e_is_hero {
                        IntentAction::CastHeal { target_id: e_unit_id }
                    } else if let Some(target_id) = nearest_enemy_id {
                        IntentAction::CastControl { target_id }
                    } else {
                        IntentAction::Hold
                    };
                    sim.hero_intents.retain(|i| i.unit_id != selected_id);
                    sim.hero_intents.push(UnitIntent {
                        unit_id: selected_id,
                        action,
                    });
                }
            });
        });
}
