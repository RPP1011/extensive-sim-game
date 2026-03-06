use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::game_core::{CampaignRoster, CompanionQuestStatus, CompanionStoryState};

// ---------------------------------------------------------------------------
// Resource
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
pub struct QuestLogState {
    pub open: bool,
    pub selected_hero_id: Option<u32>,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

pub fn quest_log_toggle_system(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mut quest_log_state: ResMut<QuestLogState>,
) {
    let Some(keyboard) = keyboard else {
        return;
    };
    if keyboard.just_pressed(KeyCode::KeyJ) {
        quest_log_state.open = !quest_log_state.open;
    }
}

pub fn draw_quest_log_system(
    mut contexts: EguiContexts,
    mut quest_log_state: ResMut<QuestLogState>,
    companion_story_state: Res<CompanionStoryState>,
    campaign_roster: Res<CampaignRoster>,
) {
    if !quest_log_state.open {
        return;
    }

    let ctx = contexts.ctx_mut();

    egui::Window::new("Quest Journal")
        .default_width(620.0)
        .resizable(true)
        .collapsible(false)
        .show(ctx, |ui| {
            // Auto-select the first active hero if nothing is selected yet.
            if quest_log_state.selected_hero_id.is_none() {
                if let Some(hero) = campaign_roster
                    .heroes
                    .iter()
                    .find(|h| h.active && !h.deserter)
                {
                    quest_log_state.selected_hero_id = Some(hero.id);
                }
            }

            let active_heroes: Vec<_> = campaign_roster
                .heroes
                .iter()
                .filter(|h| h.active && !h.deserter)
                .collect();

            ui.horizontal_top(|ui| {
                // -------------------------------------------------------
                // Left panel: hero list (~180 px)
                // -------------------------------------------------------
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(14, 16, 22))
                    .inner_margin(egui::Margin::same(6.0))
                    .show(ui, |ui| {
                        ui.set_width(180.0);
                        ui.label(
                            egui::RichText::new("Heroes")
                                .strong()
                                .color(egui::Color32::from_rgb(180, 180, 200)),
                        );
                        ui.separator();

                        for hero in &active_heroes {
                            let active_quest_count = companion_story_state
                                .quests
                                .iter()
                                .filter(|q| {
                                    q.hero_id == hero.id
                                        && q.status == CompanionQuestStatus::Active
                                })
                                .count();

                            let label_text = format!(
                                "{} [{} active]",
                                hero.name, active_quest_count
                            );

                            let is_selected =
                                quest_log_state.selected_hero_id == Some(hero.id);

                            let text = if is_selected {
                                egui::RichText::new(label_text)
                                    .strong()
                                    .color(egui::Color32::from_rgb(100, 200, 255))
                            } else {
                                egui::RichText::new(label_text)
                                    .color(egui::Color32::from_rgb(200, 200, 210))
                            };

                            if ui.selectable_label(is_selected, text).clicked() {
                                quest_log_state.selected_hero_id = Some(hero.id);
                            }
                        }

                        if active_heroes.is_empty() {
                            ui.label(
                                egui::RichText::new("No active heroes.")
                                    .color(egui::Color32::from_rgb(140, 140, 150)),
                            );
                        }
                    });

                ui.separator();

                // -------------------------------------------------------
                // Right panel: quest details
                // -------------------------------------------------------
                ui.vertical(|ui| {
                    let selected_id = quest_log_state.selected_hero_id;

                    let Some(hero_id) = selected_id else {
                        ui.label(
                            egui::RichText::new("Select a hero to view quests.")
                                .color(egui::Color32::from_rgb(140, 140, 150)),
                        );
                        return;
                    };

                    let hero_opt = campaign_roster.heroes.iter().find(|h| h.id == hero_id);
                    let hero_name = hero_opt.map(|h| h.name.as_str()).unwrap_or("Unknown");

                    ui.label(
                        egui::RichText::new(hero_name)
                            .strong()
                            .heading()
                            .color(egui::Color32::from_rgb(220, 220, 240)),
                    );
                    ui.separator();

                    let hero_quests: Vec<_> = companion_story_state
                        .quests
                        .iter()
                        .filter(|q| q.hero_id == hero_id)
                        .collect();

                    if hero_quests.is_empty() {
                        ui.label(
                            egui::RichText::new("No quests recorded for this hero.")
                                .color(egui::Color32::from_rgb(140, 140, 150)),
                        );
                    } else {
                        // Active quests
                        let active: Vec<_> = hero_quests
                            .iter()
                            .filter(|q| q.status == CompanionQuestStatus::Active)
                            .collect();

                        if !active.is_empty() {
                            ui.label(
                                egui::RichText::new("Active Quests")
                                    .strong()
                                    .color(egui::Color32::from_rgb(180, 200, 255)),
                            );
                            for quest in active {
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgb(18, 22, 32))
                                    .inner_margin(egui::Margin::same(6.0))
                                    .show(ui, |ui| {
                                        // Title in white
                                        ui.label(
                                            egui::RichText::new(&quest.title)
                                                .strong()
                                                .color(egui::Color32::WHITE),
                                        );
                                        // Objective in grey
                                        ui.label(
                                            egui::RichText::new(&quest.objective)
                                                .color(egui::Color32::from_rgb(160, 160, 170)),
                                        );

                                        // Progress bar
                                        let progress_frac = if quest.target > 0 {
                                            (quest.progress as f32 / quest.target as f32)
                                                .clamp(0.0, 1.0)
                                        } else {
                                            0.0
                                        };
                                        let bar_width =
                                            ui.available_width().min(340.0).max(60.0);
                                        let (bar_rect, _) = ui.allocate_exact_size(
                                            egui::vec2(bar_width, 10.0),
                                            egui::Sense::hover(),
                                        );
                                        let painter = ui.painter();
                                        painter.rect_filled(
                                            bar_rect,
                                            2.0,
                                            egui::Color32::from_rgb(30, 40, 30),
                                        );
                                        let fill_rect = egui::Rect::from_min_size(
                                            bar_rect.min,
                                            egui::vec2(
                                                bar_rect.width() * progress_frac,
                                                bar_rect.height(),
                                            ),
                                        );
                                        painter.rect_filled(
                                            fill_rect,
                                            2.0,
                                            egui::Color32::from_rgb(60, 180, 80),
                                        );

                                        ui.label(
                                            egui::RichText::new(format!(
                                                "{}/{}",
                                                quest.progress, quest.target
                                            ))
                                            .small()
                                            .color(egui::Color32::from_rgb(
                                                140, 180, 140,
                                            )),
                                        );

                                        // Reward line
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "Reward: +{:.0} loyalty, +{:.0} resolve",
                                                quest.reward_loyalty, quest.reward_resolve
                                            ))
                                            .small()
                                            .color(egui::Color32::from_rgb(
                                                160, 160, 100,
                                            )),
                                        );
                                    });
                                ui.add_space(4.0);
                            }
                        }

                        // Completed quests
                        let completed: Vec<_> = hero_quests
                            .iter()
                            .filter(|q| q.status == CompanionQuestStatus::Completed)
                            .collect();

                        if !completed.is_empty() {
                            ui.add_space(6.0);
                            ui.label(
                                egui::RichText::new("Completed")
                                    .strong()
                                    .color(egui::Color32::from_rgb(130, 150, 130)),
                            );
                            for quest in completed {
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgb(14, 18, 14))
                                    .inner_margin(egui::Margin::same(4.0))
                                    .show(ui, |ui| {
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "✓ {}",
                                                quest.title
                                            ))
                                            .color(egui::Color32::from_rgb(
                                                110, 130, 110,
                                            )),
                                        );
                                        ui.label(
                                            egui::RichText::new("Completed")
                                                .small()
                                                .color(egui::Color32::from_rgb(
                                                    90, 120, 90,
                                                )),
                                        );
                                    });
                                ui.add_space(2.0);
                            }
                        }

                        // Failed quests
                        let failed: Vec<_> = hero_quests
                            .iter()
                            .filter(|q| q.status == CompanionQuestStatus::Failed)
                            .collect();

                        if !failed.is_empty() {
                            ui.add_space(6.0);
                            ui.label(
                                egui::RichText::new("Failed")
                                    .strong()
                                    .color(egui::Color32::from_rgb(180, 80, 80)),
                            );
                            for quest in failed {
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgb(22, 12, 12))
                                    .inner_margin(egui::Margin::same(4.0))
                                    .show(ui, |ui| {
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "✗ {}",
                                                quest.title
                                            ))
                                            .color(egui::Color32::from_rgb(
                                                200, 80, 80,
                                            )),
                                        );
                                        ui.label(
                                            egui::RichText::new("Failed")
                                                .small()
                                                .color(egui::Color32::from_rgb(
                                                    160, 70, 70,
                                                )),
                                        );
                                    });
                                ui.add_space(2.0);
                            }
                        }
                    }

                    // ---------------------------------------------------
                    // Stress / injury strip at bottom
                    // ---------------------------------------------------
                    if let Some(hero) = hero_opt {
                        ui.add_space(10.0);
                        ui.separator();
                        ui.label(
                            egui::RichText::new("Hero Status")
                                .strong()
                                .color(egui::Color32::from_rgb(180, 180, 200)),
                        );

                        let bars: &[(&str, f32, egui::Color32)] = &[
                            (
                                "Loyalty",
                                hero.loyalty,
                                egui::Color32::from_rgb(60, 120, 220),
                            ),
                            (
                                "Stress",
                                hero.stress,
                                egui::Color32::from_rgb(220, 130, 40),
                            ),
                            (
                                "Injury",
                                hero.injury,
                                egui::Color32::from_rgb(200, 60, 60),
                            ),
                            (
                                "Fatigue",
                                hero.fatigue,
                                egui::Color32::from_rgb(200, 190, 50),
                            ),
                        ];

                        for (label, value, color) in bars {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!("{:<8}", label))
                                        .small()
                                        .color(egui::Color32::from_rgb(160, 160, 170)),
                                );
                                let bar_width = 160.0_f32;
                                let (bar_rect, _) = ui.allocate_exact_size(
                                    egui::vec2(bar_width, 10.0),
                                    egui::Sense::hover(),
                                );
                                let painter = ui.painter();
                                painter.rect_filled(
                                    bar_rect,
                                    2.0,
                                    egui::Color32::from_rgb(30, 32, 40),
                                );
                                let frac = (value / 100.0).clamp(0.0, 1.0);
                                let fill_rect = egui::Rect::from_min_size(
                                    bar_rect.min,
                                    egui::vec2(bar_rect.width() * frac, bar_rect.height()),
                                );
                                painter.rect_filled(fill_rect, 2.0, *color);
                                ui.label(
                                    egui::RichText::new(format!("{:.0}", value))
                                        .small()
                                        .color(egui::Color32::from_rgb(160, 160, 170)),
                                );
                            });
                        }
                    }
                });
            });

            ui.separator();
            if ui.button("Close").clicked() {
                quest_log_state.open = false;
            }
        });
}
