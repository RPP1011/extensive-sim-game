//! Guild management screen rendering.

use bevy_egui::egui;

use crate::campaign_ops::{enter_start_menu, truncate_for_hud};
use crate::game_core::{self, HubScreen, HubUiState, MissionResult};
use crate::hub_types::{
    HubAction, HubActionQueue, HubMenuState, HeroDetailUiState, StartMenuState,
};
use super::BoardRow;

/// Draw the guild management side-panel content.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_guild_management(
    ui: &mut egui::Ui,
    hub_ui: &mut HubUiState,
    hub_menu: &mut HubMenuState,
    start_menu: &mut StartMenuState,
    action_queue: &mut HubActionQueue,
    roster: &mut game_core::CampaignRoster,
    hero_detail: &mut HeroDetailUiState,
    attention: &game_core::AttentionState,
    ledger: &game_core::CampaignLedger,
    board_rows: &[BoardRow],
) {
    let options = [
        "Assemble Expedition",
        "Review Recruits",
        "Intel Sweep",
        "Dispatch Relief",
        "Leave Guild",
    ];

    ui.horizontal(|ui| {
        if ui.button("Back To Start Menu").clicked() {
            enter_start_menu(hub_ui, start_menu);
        }
        if ui.button("Switch To Overworld").clicked() {
            hub_ui.screen = HubScreen::OverworldMap;
        }
    });
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.label(format!(
            "Attention: {:.0}/{:.0}",
            attention.global_energy, attention.max_energy
        ));
        ui.label(format!("Actions: {}", action_queue.actions_taken));
    });
    ui.label(format!(
        "Last Consequence: {}",
        truncate_for_hud(
            &ledger
                .records
                .last()
                .map(|r| format!("t{} {} {}", r.turn, r.mission_name, r.summary))
                .unwrap_or_else(|| "none".to_string()),
            96
        )
    ));
    ui.separator();
    ui.label(egui::RichText::new("Guild Actions").strong());
    egui::Grid::new("action_grid")
        .num_columns(2)
        .spacing(egui::vec2(8.0, 8.0))
        .show(ui, |ui| {
            for (idx, label) in options.iter().enumerate() {
                let is_selected = idx == hub_menu.selected;
                let mut button =
                    egui::Button::new(*label).min_size(egui::vec2(240.0, 34.0));
                if is_selected {
                    button = button.fill(egui::Color32::from_rgb(52, 66, 88));
                }
                if ui.add(button).clicked() {
                    hub_menu.selected = idx;
                    let action = HubAction::from_selected(idx);
                    if action == HubAction::LeaveGuild {
                        enter_start_menu(hub_ui, start_menu);
                        hub_menu.notice = "Returned to start menu.".to_string();
                    } else if action_queue.pending.is_none() {
                        action_queue.pending = Some(action);
                        hub_menu.notice = format!("Executing '{}'...", action.label());
                    }
                }
                if idx % 2 == 1 {
                    ui.end_row();
                }
            }
        });

    ui.separator();
    draw_mission_board(ui, board_rows);

    ui.separator();
    draw_roster(ui, hub_menu, roster);

    // Hero Detail Panel
    if !roster.heroes.is_empty() {
        ui.separator();
        draw_hero_detail(ui, roster, hero_detail);
    }
}

fn draw_mission_board(ui: &mut egui::Ui, board_rows: &[BoardRow]) {
    ui.label(egui::RichText::new("Mission Board").strong());
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(13, 16, 22))
        .inner_margin(egui::Margin::same(8.0))
        .show(ui, |ui| {
            egui::Grid::new("mission_grid")
                .striped(true)
                .num_columns(6)
                .show(ui, |ui| {
                    ui.strong("Mission");
                    ui.strong("Status");
                    ui.strong("T");
                    ui.strong("Prog");
                    ui.strong("Alert");
                    ui.strong("Integ");
                    ui.end_row();
                    for (name, result, turns, prog, alert, integ) in board_rows.iter().take(6) {
                        ui.label(truncate_for_hud(name, 20));
                        ui.label(match result {
                            MissionResult::InProgress => "In Progress",
                            MissionResult::Victory => "Victory",
                            MissionResult::Defeat => "Defeat",
                        });
                        ui.label(turns.to_string());
                        ui.label(format!("{:.0}%", prog));
                        let alert_color = if *alert >= 70.0 {
                            egui::Color32::from_rgb(235, 95, 95)
                        } else if *alert >= 35.0 {
                            egui::Color32::from_rgb(220, 165, 80)
                        } else {
                            egui::Color32::from_rgb(138, 206, 125)
                        };
                        ui.colored_label(alert_color, format!("{:.0}", alert));
                        let integ_color = if *integ <= 35.0 {
                            egui::Color32::from_rgb(235, 95, 95)
                        } else if *integ <= 70.0 {
                            egui::Color32::from_rgb(220, 165, 80)
                        } else {
                            egui::Color32::from_rgb(138, 206, 125)
                        };
                        ui.colored_label(integ_color, format!("{:.0}", integ));
                        ui.end_row();
                    }
                });
        });
}

fn draw_roster(
    ui: &mut egui::Ui,
    hub_menu: &mut HubMenuState,
    roster: &mut game_core::CampaignRoster,
) {
    ui.label(egui::RichText::new("Roster").strong());
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(13, 16, 22))
        .inner_margin(egui::Margin::same(8.0))
        .show(ui, |ui| {
            let mut selected_player = None;
            for hero in roster.heroes.iter().take(6) {
                ui.horizontal(|ui| {
                    let is_player = roster.player_hero_id == Some(hero.id);
                    ui.label(format!(
                        "{}{} (L{:.0} S{:.0} F{:.0})",
                        if is_player { "[PLAYER] " } else { "" },
                        hero.name,
                        hero.loyalty,
                        hero.stress,
                        hero.fatigue
                    ));
                    if ui
                        .add_enabled(
                            !is_player,
                            egui::Button::new(format!("Set Player##{}", hero.id)),
                        )
                        .clicked()
                    {
                        selected_player = Some(hero.id);
                    }
                });
            }
            if let Some(id) = selected_player {
                roster.player_hero_id = Some(id);
                if let Some(hero) = roster.heroes.iter().find(|h| h.id == id) {
                    hub_menu.notice = format!("Player character set to {}.", hero.name);
                }
            } else if roster.heroes.is_empty() {
                ui.label("heroes: none");
            }
        });
}

fn draw_hero_detail(
    ui: &mut egui::Ui,
    roster: &mut game_core::CampaignRoster,
    hero_detail: &mut HeroDetailUiState,
) {
    ui.label(egui::RichText::new("Hero Detail").strong());

    // Hero selector tabs
    ui.horizontal(|ui| {
        for hero in roster.heroes.iter().take(6) {
            let selected =
                hero_detail.selected_hero_id.unwrap_or(roster.heroes[0].id) == hero.id;
            let mut btn = egui::Button::new(truncate_for_hud(&hero.name, 12))
                .min_size(egui::vec2(80.0, 28.0));
            if selected {
                btn = btn.fill(egui::Color32::from_rgb(52, 80, 120));
            }
            if ui.add(btn).clicked() {
                hero_detail.selected_hero_id = Some(hero.id);
                hero_detail.pending_loot = None;
            }
        }
    });

    let selected_id = hero_detail
        .selected_hero_id
        .unwrap_or_else(|| roster.heroes[0].id);

    // Collect mutations to apply after the immutable borrow.
    let mut equip_weapon_item: Option<game_core::EquipmentItem> = None;
    let mut gen_loot_seed: Option<u64> = None;

    if let Some(hero) = roster.heroes.iter().find(|h| h.id == selected_id) {
        egui::Frame::none()
            .fill(egui::Color32::from_rgb(18, 22, 32))
            .inner_margin(egui::Margin::same(8.0))
            .show(ui, |ui| {
                draw_hero_identity(ui, hero, roster.player_hero_id);
                ui.separator();
                draw_hero_stats(ui, hero);
                ui.separator();
                draw_hero_equipment(ui, hero);
                ui.separator();
                draw_hero_loot(ui, hero, hero_detail, &mut equip_weapon_item, &mut gen_loot_seed);
            });
    }

    // Apply deferred mutations now that immutable borrow is released.
    if let Some(seed) = gen_loot_seed {
        hero_detail.loot_seed_counter = hero_detail.loot_seed_counter.wrapping_add(1);
        hero_detail.pending_loot = game_core::generate_loot_drop(seed, 2);
    }
    if let Some(item) = equip_weapon_item {
        if let Some(hero) = roster.heroes.iter_mut().find(|h| h.id == selected_id) {
            hero.equipment.weapon = Some(item);
            game_core::check_level_up(hero);
        }
        hero_detail.pending_loot = None;
    }
}

fn draw_hero_identity(
    ui: &mut egui::Ui,
    hero: &game_core::HeroCompanion,
    player_hero_id: Option<u32>,
) {
    let is_player = player_hero_id == Some(hero.id);
    let player_tag = if is_player { " [PLAYER]" } else { "" };
    ui.label(
        egui::RichText::new(format!("{}{} -- Level {}", hero.name, player_tag, hero.level))
            .strong(),
    );
    // XP progress bar
    let xp_threshold = hero.level * hero.level * 50;
    let xp_frac = if xp_threshold > 0 {
        (hero.xp as f32 / xp_threshold as f32).clamp(0.0, 1.0)
    } else {
        1.0
    };
    ui.horizontal(|ui| {
        ui.label("XP:");
        let xp_bar = egui::ProgressBar::new(xp_frac)
            .desired_width(160.0)
            .text(format!("{}/{}", hero.xp, xp_threshold));
        ui.add(xp_bar);
    });
    // Status
    let status_text = if hero.active {
        egui::RichText::new("Active").color(egui::Color32::from_rgb(100, 220, 100))
    } else {
        egui::RichText::new("Recovering").color(egui::Color32::from_rgb(220, 160, 60))
    };
    ui.horizontal(|ui| {
        ui.label("Status:");
        ui.label(status_text);
    });
}

fn draw_hero_stats(ui: &mut egui::Ui, hero: &game_core::HeroCompanion) {
    let stat_rows = [
        ("Loyalty", hero.loyalty, egui::Color32::from_rgb(80, 180, 255)),
        ("Stress", hero.stress, egui::Color32::from_rgb(235, 95, 95)),
        ("Fatigue", hero.fatigue, egui::Color32::from_rgb(220, 165, 80)),
    ];
    egui::Grid::new(format!("hero_stats_{}", hero.id))
        .num_columns(2)
        .spacing(egui::vec2(6.0, 4.0))
        .show(ui, |ui| {
            for (label, value, color) in stat_rows {
                ui.label(label);
                let bar = egui::ProgressBar::new((value / 100.0).clamp(0.0, 1.0))
                    .desired_width(140.0)
                    .fill(color)
                    .text(format!("{:.0}", value));
                ui.add(bar);
                ui.end_row();
            }
        });
}

fn draw_hero_equipment(ui: &mut egui::Ui, hero: &game_core::HeroCompanion) {
    ui.label(egui::RichText::new("Equipment").strong());
    let slots: [(&str, &Option<game_core::EquipmentItem>); 5] = [
        ("Weapon", &hero.equipment.weapon),
        ("Offhand", &hero.equipment.offhand),
        ("Chest", &hero.equipment.chest),
        ("Boots", &hero.equipment.boots),
        ("Accessory", &hero.equipment.accessory),
    ];
    egui::Grid::new(format!("hero_equip_{}", hero.id))
        .num_columns(2)
        .spacing(egui::vec2(6.0, 4.0))
        .show(ui, |ui| {
            for (slot_name, slot_item) in slots.iter() {
                ui.label(*slot_name);
                match slot_item {
                    None => {
                        ui.colored_label(
                            egui::Color32::from_rgb(100, 100, 100),
                            "[ empty ]",
                        );
                    }
                    Some(item) => {
                        let rarity_label = match item.rarity {
                            game_core::ItemRarity::Rare => "Rare",
                            game_core::ItemRarity::Standard => "Standard",
                        };
                        let rarity_color = match item.rarity {
                            game_core::ItemRarity::Rare => {
                                egui::Color32::from_rgb(180, 130, 255)
                            }
                            game_core::ItemRarity::Standard => {
                                egui::Color32::from_rgb(200, 200, 200)
                            }
                        };
                        let summary = format!(
                            "{} ({}) +{}atk +{}hp",
                            item.name, rarity_label, item.attack_bonus, item.hp_bonus
                        );
                        let tooltip = format!(
                            "{}\nRarity: {}\nAttack: +{}\nHP: +{}\nSpeed: +{:.2}\nCooldown mult: {:.2}",
                            item.name,
                            rarity_label,
                            item.attack_bonus,
                            item.hp_bonus,
                            item.speed_bonus,
                            item.cooldown_mult
                        );
                        let response = ui.colored_label(rarity_color, summary);
                        response.on_hover_text(tooltip);
                    }
                }
                ui.end_row();
            }
        });
}

fn draw_hero_loot(
    ui: &mut egui::Ui,
    hero: &game_core::HeroCompanion,
    hero_detail: &HeroDetailUiState,
    equip_weapon_item: &mut Option<game_core::EquipmentItem>,
    gen_loot_seed: &mut Option<u64>,
) {
    ui.label(egui::RichText::new("Test Loot").strong());
    if let Some(ref item) = hero_detail.pending_loot.clone() {
        let rarity_label = match item.rarity {
            game_core::ItemRarity::Rare => "Rare",
            game_core::ItemRarity::Standard => "Standard",
        };
        ui.label(format!(
            "Pending: {} ({}) +{}atk +{}hp",
            item.name, rarity_label, item.attack_bonus, item.hp_bonus
        ));
        if ui.button("Equip as Weapon").clicked() {
            *equip_weapon_item = Some(item.clone());
        }
    } else {
        ui.colored_label(
            egui::Color32::from_rgb(100, 100, 100),
            "No pending item.",
        );
    }
    if ui.button("Generate Test Item").clicked() {
        *gen_loot_seed = Some(hero.id as u64 + hero_detail.loot_seed_counter);
    }
}
