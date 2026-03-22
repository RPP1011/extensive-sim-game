//! Colored ASCII combat viewport and overworld CentralPanel.
//!
//! Replaces all 3D rendering with egui-only text viewports:
//! - CentralPanel: overworld hex map (always visible on map screens)
//! - Floating Window: mission combat ASCII grid (draggable, closeable)

use std::collections::VecDeque;

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::ai::core::{SimEvent, SimState, Team};
use crate::ai::pathing::GridNav;
use crate::game_core::{self, HubScreen, HubUiState};
use crate::hub_ui_draw::faction_color;
use crate::mission::sim_bridge::{MissionSimState, LastMissionReplay, SimEventBuffer};
use crate::mission::execution::ReplayViewerState;
use crate::region_nav::RegionTargetPickerState;

pub mod glyph_atlas;
pub mod batched_renderer;

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// State for the floating mission combat pane.
#[derive(Resource)]
pub struct MissionPaneState {
    pub open: bool,
    pub event_feed: VecDeque<(String, egui::Color32)>,
    pub max_feed: usize,
}

impl Default for MissionPaneState {
    fn default() -> Self {
        Self {
            open: false,
            event_feed: VecDeque::new(),
            max_feed: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// Color palette
// ---------------------------------------------------------------------------

const COLOR_WALL: egui::Color32 = egui::Color32::from_rgb(60, 55, 50);
const COLOR_FLOOR: egui::Color32 = egui::Color32::from_rgb(38, 40, 35);
const COLOR_ELEVATED: egui::Color32 = egui::Color32::from_rgb(80, 75, 55);
const COLOR_HALF_COVER: egui::Color32 = egui::Color32::from_rgb(90, 80, 60);
const COLOR_HERO: egui::Color32 = egui::Color32::from_rgb(80, 200, 120); // #50C878
const COLOR_ALLY: egui::Color32 = egui::Color32::from_rgb(80, 160, 255); // #50A0FF
const COLOR_ENEMY: egui::Color32 = egui::Color32::from_rgb(255, 90, 80); // #FF5A50
const COLOR_DEAD: egui::Color32 = egui::Color32::from_rgb(80, 80, 80);
const COLOR_CC: egui::Color32 = egui::Color32::from_rgb(230, 200, 80);
const COLOR_ZONE: egui::Color32 = egui::Color32::from_rgb(160, 120, 220);

const COLOR_DMG: egui::Color32 = egui::Color32::from_rgb(255, 180, 60);
const COLOR_HEAL: egui::Color32 = egui::Color32::from_rgb(80, 220, 80);
const COLOR_DEATH: egui::Color32 = egui::Color32::from_rgb(255, 60, 60);
const COLOR_CC_EVENT: egui::Color32 = egui::Color32::from_rgb(230, 200, 80);

// HP bar colors
const COLOR_HP_HIGH: egui::Color32 = egui::Color32::from_rgb(80, 200, 80);
const COLOR_HP_MID: egui::Color32 = egui::Color32::from_rgb(220, 200, 50);
const COLOR_HP_LOW: egui::Color32 = egui::Color32::from_rgb(220, 60, 40);
const COLOR_HP_BG: egui::Color32 = egui::Color32::from_rgb(50, 50, 50);

// UI chrome colors
const COLOR_HEADER: egui::Color32 = egui::Color32::from_rgb(160, 170, 185);
const COLOR_DIM: egui::Color32 = egui::Color32::from_rgb(55, 60, 68);
const COLOR_SECTION: egui::Color32 = egui::Color32::from_rgb(100, 110, 125);

// ---------------------------------------------------------------------------
// Grid rendering
// ---------------------------------------------------------------------------

/// Renders the combat grid as colored ASCII using LayoutJob for per-character coloring.
/// Each cell is 2 chars wide: `. ` floor, `░░` half cover, `██` wall, `▲▲` elevation.
/// Units: `@H` player hero (green), `a1`..`aN` allies (blue), `e1`..`eN` enemies (red).
/// Dead units render as `xx` in gray. Row numbers on the left, box-drawing border.
fn draw_combat_grid(ui: &mut egui::Ui, sim: &SimState, grid_nav: Option<&GridNav>, time_secs: f64) {
    let Some(nav) = grid_nav else {
        ui.colored_label(COLOR_DIM, "No grid data available.");
        return;
    };

    let cols = ((nav.max_x - nav.min_x) / nav.cell_size).ceil() as i32;
    let rows = ((nav.max_y - nav.min_y) / nav.cell_size).ceil() as i32;

    // Build unit position lookup: cell -> (label, color)
    let mut unit_cells: std::collections::HashMap<(i32, i32), (String, egui::Color32)> =
        std::collections::HashMap::new();

    let mut hero_idx = 0u32;
    let mut enemy_idx = 0u32;
    for (slot, unit) in sim.units.iter().enumerate() {
        let cx = ((unit.position.x - nav.min_x) / nav.cell_size).floor() as i32;
        let cy = ((unit.position.y - nav.min_y) / nav.cell_size).floor() as i32;

        let is_dead = unit.hp <= 0;
        let is_casting = unit.casting.is_some();
        let is_cc = unit.control_remaining_ms > 0;

        let (label, base_color) = match unit.team {
            Team::Hero => {
                let s = hero_idx;
                hero_idx += 1;
                if slot == 0 {
                    ("@H".to_string(), COLOR_HERO)
                } else {
                    (format!("a{}", s), COLOR_ALLY)
                }
            }
            Team::Enemy => {
                enemy_idx += 1;
                (format!("e{}", enemy_idx), COLOR_ENEMY)
            }
        };

        if is_dead {
            unit_cells.insert((cx, cy), ("xx".to_string(), COLOR_DEAD));
            continue;
        }

        let color = if is_cc {
            COLOR_CC
        } else if is_casting {
            let flash = (time_secs * 4.0).sin() > 0.0;
            if flash {
                match unit.team {
                    Team::Hero => egui::Color32::from_rgb(180, 255, 200),
                    Team::Enemy => egui::Color32::from_rgb(255, 160, 40),
                }
            } else {
                base_color
            }
        } else {
            base_color
        };

        unit_cells.insert((cx, cy), (label, color));
    }

    let grid_font = egui::FontId::monospace(15.0);
    let row_num_font = egui::FontId::monospace(12.0);

    // Grid border top: "   ┌──────...──┐"
    {
        let mut border_job = egui::text::LayoutJob::default();
        let top_str = format!("   \u{250c}{}\u{2510}", "\u{2500}".repeat(cols as usize * 2));
        border_job.append(
            &top_str,
            0.0,
            egui::TextFormat { font_id: grid_font.clone(), color: COLOR_DIM, ..Default::default() },
        );
        ui.label(border_job);
    }

    for row in 0..rows {
        let mut job = egui::text::LayoutJob::default();

        // Row number (right-aligned, 2 chars + space)
        let row_label = format!("{:>2} ", row);
        job.append(
            &row_label,
            0.0,
            egui::TextFormat { font_id: row_num_font.clone(), color: COLOR_SECTION, ..Default::default() },
        );
        // Left border
        job.append(
            "\u{2502}",
            0.0,
            egui::TextFormat { font_id: grid_font.clone(), color: COLOR_DIM, ..Default::default() },
        );

        for col in 0..cols {
            let cell_key = (col, row);
            let (text, color) = if let Some((label, color)) = unit_cells.get(&cell_key) {
                (format!("{:<2}", label), *color)
            } else if nav.blocked.contains(&cell_key) {
                ("\u{2588}\u{2588}".to_string(), COLOR_WALL)
            } else if nav.elevation_by_cell.get(&cell_key).copied().unwrap_or(0.0) > 1.0 {
                ("\u{25b2}\u{25b2}".to_string(), COLOR_ELEVATED)
            } else if nav.elevation_by_cell.get(&cell_key).copied().unwrap_or(0.0) > 0.3 {
                ("\u{2591}\u{2591}".to_string(), COLOR_HALF_COVER)
            } else {
                (". ".to_string(), COLOR_FLOOR)
            };

            job.append(
                &text,
                0.0,
                egui::TextFormat { font_id: grid_font.clone(), color, ..Default::default() },
            );
        }

        // Right border
        job.append(
            "\u{2502}",
            0.0,
            egui::TextFormat { font_id: grid_font.clone(), color: COLOR_DIM, ..Default::default() },
        );

        ui.label(job);
    }

    // Grid border bottom: "   └──────...──┘"
    {
        let mut border_job = egui::text::LayoutJob::default();
        let bot_str = format!("   \u{2514}{}\u{2518}", "\u{2500}".repeat(cols as usize * 2));
        border_job.append(
            &bot_str,
            0.0,
            egui::TextFormat { font_id: grid_font.clone(), color: COLOR_DIM, ..Default::default() },
        );
        ui.label(border_job);
    }
}

// ---------------------------------------------------------------------------
// Unit roster
// ---------------------------------------------------------------------------

fn hp_color(ratio: f32) -> egui::Color32 {
    if ratio > 0.6 {
        COLOR_HP_HIGH
    } else if ratio > 0.3 {
        COLOR_HP_MID
    } else {
        COLOR_HP_LOW
    }
}

/// Renders two-column unit roster: allies on left, enemies on right.
/// Format per unit: `label  HP_BAR hp/max [status]`
/// HP bar is 8 chars: `█` filled, `░` empty. Green >60%, yellow >30%, red <=30%.
fn draw_unit_roster(ui: &mut egui::Ui, sim: &SimState) {
    let font = egui::FontId::monospace(13.0);
    let header_font = egui::FontId::monospace(12.0);

    // Collect units by team
    let heroes: Vec<_> = sim.units.iter().enumerate()
        .filter(|(_, u)| u.team == Team::Hero)
        .collect();
    let enemies: Vec<_> = sim.units.iter().filter(|u| u.team == Team::Enemy).collect();
    let max_rows = heroes.len().max(enemies.len());

    // Section headers side by side using ╔═ format
    {
        let mut header_job = egui::text::LayoutJob::default();
        header_job.append(
            "\u{2554}\u{2550} ALLIES ",
            0.0,
            egui::TextFormat { font_id: header_font.clone(), color: COLOR_ALLY, ..Default::default() },
        );
        header_job.append(
            &"\u{2550}".repeat(24),
            0.0,
            egui::TextFormat { font_id: header_font.clone(), color: COLOR_DIM, ..Default::default() },
        );
        header_job.append(
            "  ",
            0.0,
            egui::TextFormat { font_id: header_font.clone(), color: COLOR_DIM, ..Default::default() },
        );
        header_job.append(
            "\u{2554}\u{2550} ENEMIES ",
            0.0,
            egui::TextFormat { font_id: header_font.clone(), color: COLOR_ENEMY, ..Default::default() },
        );
        header_job.append(
            &"\u{2550}".repeat(22),
            0.0,
            egui::TextFormat { font_id: header_font.clone(), color: COLOR_DIM, ..Default::default() },
        );
        ui.label(header_job);
    }

    let bar_len = 8usize;

    for row in 0..max_rows {
        let mut job = egui::text::LayoutJob::default();

        // Hero/ally column
        if let Some((slot, unit)) = heroes.get(row) {
            let label = if *slot == 0 {
                "@H".to_string()
            } else {
                format!("a{}", slot)
            };
            let label_color = if *slot == 0 { COLOR_HERO } else { COLOR_ALLY };

            // Label
            job.append(
                &format!("{:<3}", label),
                0.0,
                egui::TextFormat { font_id: font.clone(), color: label_color, ..Default::default() },
            );

            if unit.hp <= 0 {
                // Dead unit
                job.append(
                    "DEAD                           ",
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: COLOR_DEAD, ..Default::default() },
                );
            } else {
                let ratio = unit.hp as f32 / unit.max_hp.max(1) as f32;
                let filled = ((ratio * bar_len as f32).round() as usize).min(bar_len);
                let bar_filled: String = "\u{2588}".repeat(filled);
                let bar_empty: String = "\u{2591}".repeat(bar_len - filled);

                job.append(
                    &bar_filled,
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: hp_color(ratio), ..Default::default() },
                );
                job.append(
                    &bar_empty,
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: COLOR_HP_BG, ..Default::default() },
                );

                let hp_text = format!(" {:>4}/{:<4}", unit.hp, unit.max_hp);
                job.append(
                    &hp_text,
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: egui::Color32::LIGHT_GRAY, ..Default::default() },
                );

                // Status indicators
                if unit.control_remaining_ms > 0 {
                    job.append(
                        " CC",
                        0.0,
                        egui::TextFormat { font_id: font.clone(), color: COLOR_CC, ..Default::default() },
                    );
                } else if unit.casting.is_some() {
                    job.append(
                        " ...",
                        0.0,
                        egui::TextFormat { font_id: font.clone(), color: COLOR_HEADER, ..Default::default() },
                    );
                } else {
                    job.append(
                        "    ",
                        0.0,
                        egui::TextFormat { font_id: font.clone(), color: COLOR_DIM, ..Default::default() },
                    );
                }
            }
        } else {
            // Empty padding for alignment
            job.append(
                &" ".repeat(31),
                0.0,
                egui::TextFormat { font_id: font.clone(), color: COLOR_DIM, ..Default::default() },
            );
        }

        // Separator between columns
        job.append(
            "  ",
            0.0,
            egui::TextFormat { font_id: font.clone(), color: COLOR_DIM, ..Default::default() },
        );

        // Enemy column
        if let Some(unit) = enemies.get(row) {
            let label = format!("e{}", row + 1);
            job.append(
                &format!("{:<3}", label),
                0.0,
                egui::TextFormat { font_id: font.clone(), color: COLOR_ENEMY, ..Default::default() },
            );

            if unit.hp <= 0 {
                job.append(
                    "DEAD",
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: COLOR_DEAD, ..Default::default() },
                );
            } else {
                let ratio = unit.hp as f32 / unit.max_hp.max(1) as f32;
                let filled = ((ratio * bar_len as f32).round() as usize).min(bar_len);
                let bar_filled: String = "\u{2588}".repeat(filled);
                let bar_empty: String = "\u{2591}".repeat(bar_len - filled);

                job.append(
                    &bar_filled,
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: hp_color(ratio), ..Default::default() },
                );
                job.append(
                    &bar_empty,
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: COLOR_HP_BG, ..Default::default() },
                );

                let hp_text = format!(" {:>4}/{:<4}", unit.hp, unit.max_hp);
                job.append(
                    &hp_text,
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: egui::Color32::LIGHT_GRAY, ..Default::default() },
                );

                if unit.control_remaining_ms > 0 {
                    job.append(
                        " CC",
                        0.0,
                        egui::TextFormat { font_id: font.clone(), color: COLOR_CC, ..Default::default() },
                    );
                } else if unit.casting.is_some() {
                    job.append(
                        " ...",
                        0.0,
                        egui::TextFormat { font_id: font.clone(), color: COLOR_HEADER, ..Default::default() },
                    );
                }
            }
        }

        ui.label(job);
    }
}

// ---------------------------------------------------------------------------
// Event feed
// ---------------------------------------------------------------------------

/// Processes new SimEvents into the event feed.
fn ingest_events(feed: &mut VecDeque<(String, egui::Color32)>, events: &[SimEvent], max: usize) {
    for event in events {
        let (msg, color) = match event {
            SimEvent::DamageApplied { source_id, target_id, amount, target_hp_after, .. } => {
                (format!("#{} \u{2192} #{}: {} dmg (\u{2192}{}hp)", source_id, target_id, amount, target_hp_after), COLOR_DMG)
            }
            SimEvent::HealApplied { source_id, target_id, amount, target_hp_after, .. } => {
                (format!("#{} heals #{} +{} (\u{2192}{}hp)", source_id, target_id, amount, target_hp_after), COLOR_HEAL)
            }
            SimEvent::UnitDied { unit_id, .. } => {
                (format!("#{} has fallen!", unit_id), COLOR_DEATH)
            }
            SimEvent::ControlApplied { source_id, target_id, duration_ms, .. } => {
                (format!("#{} CC \u{2192} #{} ({}ms)", source_id, target_id, duration_ms), COLOR_CC_EVENT)
            }
            _ => continue,
        };
        feed.push_back((msg, color));
        while feed.len() > max {
            feed.pop_front();
        }
    }
}

/// Draws the combat event log. Each line prefixed with `▸ ` in dim color.
/// Damage in orange, heals in green, deaths in red, CC in yellow.
/// Shows "Awaiting events..." when empty.
fn draw_event_feed(ui: &mut egui::Ui, feed: &VecDeque<(String, egui::Color32)>) {
    let font = egui::FontId::monospace(13.0);
    if feed.is_empty() {
        ui.colored_label(COLOR_DIM, "  Awaiting events...");
        return;
    }
    for (msg, color) in feed.iter() {
        let mut job = egui::text::LayoutJob::default();
        job.append(
            "\u{25b8} ",
            0.0,
            egui::TextFormat { font_id: font.clone(), color: COLOR_DIM, ..Default::default() },
        );
        job.append(
            msg,
            0.0,
            egui::TextFormat { font_id: font.clone(), color: *color, ..Default::default() },
        );
        ui.label(job);
    }
}

// ---------------------------------------------------------------------------
// Section header helper
// ---------------------------------------------------------------------------

/// Renders `╔═ SECTION_NAME ═══════...` format section header at 12px.
fn section_header(ui: &mut egui::Ui, title: &str) {
    let font = egui::FontId::monospace(12.0);
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &format!("\u{2554}\u{2550} {} ", title),
        0.0,
        egui::TextFormat { font_id: font.clone(), color: COLOR_HEADER, ..Default::default() },
    );
    let fill_count = (60usize).saturating_sub(title.len() + 4);
    job.append(
        &"\u{2550}".repeat(fill_count),
        0.0,
        egui::TextFormat { font_id: font.clone(), color: COLOR_DIM, ..Default::default() },
    );
    ui.label(job);
}

// ---------------------------------------------------------------------------
// Horizontal separator helper
// ---------------------------------------------------------------------------

fn draw_separator(ui: &mut egui::Ui) {
    let font = egui::FontId::monospace(12.0);
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &"\u{2500}".repeat(66),
        0.0,
        egui::TextFormat { font_id: font.clone(), color: COLOR_DIM, ..Default::default() },
    );
    ui.label(job);
}

// ---------------------------------------------------------------------------
// Hero command panel
// ---------------------------------------------------------------------------

// Colors for hero command panel
const COLOR_READY: egui::Color32 = egui::Color32::from_rgb(80, 220, 80);
const COLOR_COOLDOWN: egui::Color32 = egui::Color32::from_rgb(120, 120, 130);
const COLOR_KEY: egui::Color32 = egui::Color32::from_rgb(200, 180, 100);
const COLOR_LABEL: egui::Color32 = egui::Color32::from_rgb(180, 190, 200);
const COLOR_VALUE: egui::Color32 = egui::Color32::from_rgb(150, 160, 175);
const COLOR_MOVEMENT: egui::Color32 = egui::Color32::from_rgb(140, 150, 165);

/// Draws the hero command panel showing abilities for the first living hero unit.
fn draw_hero_commands(ui: &mut egui::Ui, sim: &SimState) {
    let font = egui::FontId::monospace(13.0);
    let small_font = egui::FontId::monospace(12.0);

    // Find first living hero
    let hero = sim.units.iter().find(|u| u.team == Team::Hero && u.hp > 0);
    let Some(hero) = hero else {
        return;
    };

    section_header(ui, "HERO COMMANDS");

    // --- [Q] Attack ---
    {
        let mut job = egui::text::LayoutJob::default();
        let ready = hero.cooldown_remaining_ms == 0;

        job.append("[Q] ", 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_KEY, ..Default::default() });
        job.append(&format!("{:<10}", "Attack"), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_LABEL, ..Default::default() });

        if ready {
            job.append("\u{25cf}ready   ", 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_READY, ..Default::default() });
        } else {
            let secs = hero.cooldown_remaining_ms as f32 / 1000.0;
            job.append(&format!("\u{25cb} {:<5.1}s ", secs), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_COOLDOWN, ..Default::default() });
        }

        if hero.attack_range > 0.0 {
            job.append(&format!("Range: {:<4.1} ", hero.attack_range), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_VALUE, ..Default::default() });
        }
        if hero.attack_damage > 0 {
            job.append(&format!("Dmg: {}", hero.attack_damage), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_DMG, ..Default::default() });
        }
        ui.label(job);
    }

    // --- [W] Ability (only if ability_damage > 0 or ability_range > 0) ---
    if hero.ability_damage > 0 || hero.ability_range > 0.0 {
        let mut job = egui::text::LayoutJob::default();
        let ready = hero.ability_cooldown_remaining_ms == 0;

        job.append("[W] ", 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_KEY, ..Default::default() });
        job.append(&format!("{:<10}", "Ability"), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_LABEL, ..Default::default() });

        if ready {
            job.append("\u{25cf}ready   ", 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_READY, ..Default::default() });
        } else {
            let secs = hero.ability_cooldown_remaining_ms as f32 / 1000.0;
            job.append(&format!("\u{25cb} {:<5.1}s ", secs), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_COOLDOWN, ..Default::default() });
        }

        if hero.ability_range > 0.0 {
            job.append(&format!("Range: {:<4.1} ", hero.ability_range), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_VALUE, ..Default::default() });
        }
        if hero.ability_damage > 0 {
            job.append(&format!("Dmg: {}", hero.ability_damage), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_DMG, ..Default::default() });
        }
        ui.label(job);
    }

    // --- [E] Heal (only if heal_amount > 0) ---
    if hero.heal_amount > 0 {
        let mut job = egui::text::LayoutJob::default();
        let ready = hero.heal_cooldown_remaining_ms == 0;

        job.append("[E] ", 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_KEY, ..Default::default() });
        job.append(&format!("{:<10}", "Heal"), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_LABEL, ..Default::default() });

        if ready {
            job.append("\u{25cf}ready   ", 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_READY, ..Default::default() });
        } else {
            let secs = hero.heal_cooldown_remaining_ms as f32 / 1000.0;
            job.append(&format!("\u{25cb} {:<5.1}s ", secs), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_COOLDOWN, ..Default::default() });
        }

        if hero.heal_range > 0.0 {
            job.append(&format!("Range: {:<4.1} ", hero.heal_range), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_VALUE, ..Default::default() });
        }
        job.append(&format!("+{} HP", hero.heal_amount), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_HEAL, ..Default::default() });
        ui.label(job);
    }

    // --- [R] Control (only if control_duration_ms > 0) ---
    if hero.control_duration_ms > 0 {
        let mut job = egui::text::LayoutJob::default();
        let ready = hero.control_cooldown_remaining_ms == 0;

        job.append("[R] ", 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_KEY, ..Default::default() });
        job.append(&format!("{:<10}", "Control"), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_LABEL, ..Default::default() });

        if ready {
            job.append("\u{25cf}ready   ", 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_READY, ..Default::default() });
        } else {
            let secs = hero.control_cooldown_remaining_ms as f32 / 1000.0;
            job.append(&format!("\u{25cb} {:<5.1}s ", secs), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_COOLDOWN, ..Default::default() });
        }

        if hero.control_range > 0.0 {
            job.append(&format!("Range: {:<4.1} ", hero.control_range), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_VALUE, ..Default::default() });
        }
        let cc_secs = hero.control_duration_ms as f32 / 1000.0;
        job.append(&format!("CC {:.1}s", cc_secs), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_CC, ..Default::default() });
        ui.label(job);
    }

    // --- Hero engine abilities (from abilities vec) ---
    if !hero.abilities.is_empty() {
        let mut sep_job = egui::text::LayoutJob::default();
        sep_job.append(
            "\u{2500}\u{2500} Abilities \u{2500}\u{2500}",
            0.0,
            egui::TextFormat { font_id: small_font.clone(), color: COLOR_DIM, ..Default::default() },
        );
        ui.label(sep_job);

        for (i, slot) in hero.abilities.iter().enumerate() {
            let mut job = egui::text::LayoutJob::default();
            let ready = slot.cooldown_remaining_ms == 0;

            job.append(&format!("[{}] ", i + 1), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_KEY, ..Default::default() });

            let name: String = slot.def.name.chars().take(14).collect();
            job.append(&format!("{:<15}", name), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_LABEL, ..Default::default() });

            if ready {
                job.append("\u{25cf}ready", 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_READY, ..Default::default() });
            } else {
                let secs = slot.cooldown_remaining_ms as f32 / 1000.0;
                job.append(&format!("\u{25cb} {:.1}s cd", secs), 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_COOLDOWN, ..Default::default() });
            }

            ui.label(job);
        }
    }

    // --- Movement commands ---
    {
        let mut sep_job = egui::text::LayoutJob::default();
        sep_job.append(
            "\u{2500}\u{2500} Movement \u{2500}\u{2500}",
            0.0,
            egui::TextFormat { font_id: small_font.clone(), color: COLOR_DIM, ..Default::default() },
        );
        ui.label(sep_job);

        let mut job = egui::text::LayoutJob::default();
        job.append("[Click] ", 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_KEY, ..Default::default() });
        job.append("Move   ", 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_MOVEMENT, ..Default::default() });
        job.append("[H] ", 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_KEY, ..Default::default() });
        job.append("Hold   ", 0.0, egui::TextFormat { font_id: font.clone(), color: COLOR_MOVEMENT, ..Default::default() });
        job.append("[Retreat]", 0.0, egui::TextFormat { font_id: font.clone(), color: egui::Color32::from_rgb(255, 130, 80), ..Default::default() });
        ui.label(job);
    }
}

// ---------------------------------------------------------------------------
// Combat pane (floating window)
// ---------------------------------------------------------------------------

/// Returns true if the "Retreat" button was clicked.
fn draw_combat_pane(
    ctx: &egui::Context,
    pane: &mut MissionPaneState,
    sim: &SimState,
    grid_nav: Option<&GridNav>,
    title: &str,
    replay_info: Option<(usize, usize)>,
    show_retreat: bool,
) -> bool {
    let mut open = pane.open;
    let mut retreat_clicked = false;

    let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
    let heroes_total = sim.units.iter().filter(|u| u.team == Team::Hero).count();
    let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
    let enemies_total = sim.units.iter().filter(|u| u.team == Team::Enemy).count();

    egui::Window::new(format!("\u{2694} {}", title))
        .default_size([560.0, 600.0])
        .resizable(true)
        .collapsible(true)
        .open(&mut open)
        .frame(
            egui::Frame::window(&ctx.style())
                .fill(egui::Color32::from_rgb(8, 10, 14))
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 70, 85))),
        )
        .show(ctx, |ui| {
            // ── Header bar ──────────────────────────────────────────────────
            let tick = sim.tick;
            let header_font = egui::FontId::monospace(12.0);
            {
                let mut header_job = egui::text::LayoutJob::default();
                header_job.append(
                    &format!("Tk {:>5}  \u{25b6}  ", tick),
                    0.0,
                    egui::TextFormat { font_id: header_font.clone(), color: COLOR_SECTION, ..Default::default() },
                );
                header_job.append(
                    &format!("Allies {}/{}  ", heroes_alive, heroes_total),
                    0.0,
                    egui::TextFormat { font_id: header_font.clone(), color: COLOR_ALLY, ..Default::default() },
                );
                header_job.append(
                    "vs  ",
                    0.0,
                    egui::TextFormat { font_id: header_font.clone(), color: COLOR_SECTION, ..Default::default() },
                );
                header_job.append(
                    &format!("Enemies {}/{}  ", enemies_alive, enemies_total),
                    0.0,
                    egui::TextFormat { font_id: header_font.clone(), color: COLOR_ENEMY, ..Default::default() },
                );
                ui.label(header_job);
            }

            // Objective / replay info / retreat button line
            if let Some((frame, total)) = replay_info {
                let mut replay_job = egui::text::LayoutJob::default();
                replay_job.append(
                    &format!("Frame {:>4}/{:>4}  ", frame + 1, total),
                    0.0,
                    egui::TextFormat { font_id: header_font.clone(), color: COLOR_HEADER, ..Default::default() },
                );
                replay_job.append(
                    "Space: play/pause  \u{2190}/\u{2192}: step  Esc: exit",
                    0.0,
                    egui::TextFormat { font_id: header_font.clone(), color: COLOR_SECTION, ..Default::default() },
                );
                ui.label(replay_job);
            } else if show_retreat {
                ui.horizontal(|ui| {
                    let mut obj_job = egui::text::LayoutJob::default();
                    obj_job.append(
                        "Obj: ",
                        0.0,
                        egui::TextFormat { font_id: header_font.clone(), color: COLOR_DIM, ..Default::default() },
                    );
                    obj_job.append(
                        "Eliminate",
                        0.0,
                        egui::TextFormat { font_id: header_font.clone(), color: COLOR_SECTION, ..Default::default() },
                    );
                    ui.label(obj_job);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("Retreat").color(egui::Color32::from_rgb(255, 130, 80)),
                                )
                                .min_size(egui::vec2(80.0, 26.0)),
                            )
                            .clicked()
                        {
                            retreat_clicked = true;
                        }
                    });
                });
            }

            draw_separator(ui);

            // ── Combat grid ─────────────────────────────────────────────────
            egui::ScrollArea::both().max_height(340.0).show(ui, |ui| {
                draw_combat_grid(ui, sim, grid_nav, ctx.input(|i| i.time));
            });

            draw_separator(ui);

            // ── Roster ──────────────────────────────────────────────────────
            draw_unit_roster(ui, sim);

            draw_separator(ui);

            // ── Event feed ──────────────────────────────────────────────────
            section_header(ui, "COMBAT LOG");
            egui::ScrollArea::vertical()
                .max_height(130.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    draw_event_feed(ui, &pane.event_feed);
                });

            // ── Hero commands (live combat only) ────────────────────────────
            if show_retreat {
                draw_separator(ui);
                draw_hero_commands(ui, sim);
            }
        });

    pane.open = open;
    retreat_clicked
}

// ---------------------------------------------------------------------------
// CentralPanel dispatch
// ---------------------------------------------------------------------------

/// Renders the overworld terrain map in the CentralPanel.
fn draw_overworld_central_panel(
    ctx: &egui::Context,
    overworld: &mut game_core::OverworldMap,
    target_picker: &mut RegionTargetPickerState,
    parties: &mut game_core::CampaignParties,
    party_snapshots: &[game_core::CampaignParty],
    transition_locked: bool,
) {
    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(6, 9, 13))
                .inner_margin(egui::Margin::same(8.0)),
        )
        .show(ctx, |ui| {
            crate::hub_ui_draw::overworld_map_strategic::draw_strategic_map(
                ui, overworld, target_picker, parties, party_snapshots, transition_locked,
            );
        });
}

/// Renders 19-hex overworld as ASCII art with faction coloring.
fn draw_ascii_overworld(ui: &mut egui::Ui, overworld: &mut game_core::OverworldMap) {
    let font = egui::FontId::monospace(15.0);
    let small_font = egui::FontId::monospace(11.0);
    let dim = egui::Color32::from_rgb(45, 50, 58);
    let border_color = egui::Color32::from_rgb(55, 65, 78);

    // Map title
    {
        let mut title_job = egui::text::LayoutJob::default();
        let top_border = format!("\u{2554}{}\u{2557}", "\u{2550}".repeat(62));
        title_job.append(
            &top_border,
            0.0,
            egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() },
        );
        ui.label(title_job);

        let mut title2_job = egui::text::LayoutJob::default();
        title2_job.append(
            "\u{2551}  ",
            0.0,
            egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() },
        );
        title2_job.append(
            "OVERWORLD MAP",
            0.0,
            egui::TextFormat { font_id: font.clone(), color: egui::Color32::from_rgb(200, 215, 235), ..Default::default() },
        );
        title2_job.append(
            "                                             \u{2551}",
            0.0,
            egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() },
        );
        ui.label(title2_job);

        let mut title3_job = egui::text::LayoutJob::default();
        let bot_border = format!("\u{255a}{}\u{255d}", "\u{2550}".repeat(62));
        title3_job.append(
            &bot_border,
            0.0,
            egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() },
        );
        ui.label(title3_job);
    }

    ui.add_space(6.0);

    // Build axial hex coords (same as generation.rs)
    let mut hex_coords: Vec<(i32, i32)> = Vec::new();
    for q in -2..=2i32 {
        let r_min = (-2).max(-q - 2);
        let r_max = 2.min(-q + 2);
        for r in r_min..=r_max {
            hex_coords.push((q, r));
        }
    }
    hex_coords.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

    // Map from axial coord to region index
    let coord_to_idx: std::collections::HashMap<(i32, i32), usize> = hex_coords
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    // Group hexes by r-coordinate (row)
    let mut rows: std::collections::BTreeMap<i32, Vec<i32>> = std::collections::BTreeMap::new();
    for &(q, r) in &hex_coords {
        rows.entry(r).or_default().push(q);
    }
    for qs in rows.values_mut() {
        qs.sort();
    }

    let cell_w = 16usize; // chars per hex cell

    for (r_idx, (r, qs)) in rows.iter().enumerate() {
        // For hex stagger: rows with fewer hexes get extra indent
        let indent_chars = (5 - qs.len()) * (cell_w / 2);

        // Top border row for this hex row
        {
            let mut top_job = egui::text::LayoutJob::default();
            if indent_chars > 0 {
                top_job.append(
                    &" ".repeat(indent_chars),
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: dim, ..Default::default() },
                );
            }
            for (qi, _) in qs.iter().enumerate() {
                if qi == 0 {
                    top_job.append("\u{250c}", 0.0, egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() });
                } else {
                    top_job.append("\u{252c}", 0.0, egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() });
                }
                top_job.append(
                    &"\u{2500}".repeat(cell_w - 1),
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() },
                );
            }
            top_job.append("\u{2510}", 0.0, egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() });
            ui.label(top_job);
        }

        // Content row: markers line
        {
            let mut marker_job = egui::text::LayoutJob::default();
            if indent_chars > 0 {
                marker_job.append(
                    &" ".repeat(indent_chars),
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: dim, ..Default::default() },
                );
            }
            for &q in qs.iter() {
                marker_job.append("\u{2502}", 0.0, egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() });

                let region_idx = coord_to_idx.get(&(q, *r)).copied().unwrap_or(usize::MAX);
                if region_idx < overworld.regions.len() {
                    let region = &overworld.regions[region_idx];
                    let faction_col = faction_color(region.owner_faction_id);
                    let is_current = region_idx == overworld.current_region;
                    let is_selected = region_idx == overworld.selected_region;
                    let has_mission = region.mission_slot.is_some();

                    let marker = if is_current && is_selected {
                        "@ >"
                    } else if is_current {
                        "@  "
                    } else if is_selected {
                        ">  "
                    } else if has_mission {
                        "!  "
                    } else {
                        "   "
                    };
                    let marker_color = if is_current {
                        egui::Color32::from_rgb(255, 255, 100)
                    } else if is_selected {
                        egui::Color32::from_rgb(180, 220, 255)
                    } else if has_mission {
                        egui::Color32::from_rgb(255, 200, 80)
                    } else {
                        faction_col
                    };

                    marker_job.append(
                        &format!(" {:<width$}", marker, width = cell_w - 2),
                        0.0,
                        egui::TextFormat { font_id: font.clone(), color: marker_color, ..Default::default() },
                    );
                } else {
                    marker_job.append(
                        &" ".repeat(cell_w - 1),
                        0.0,
                        egui::TextFormat { font_id: font.clone(), color: dim, ..Default::default() },
                    );
                }
            }
            marker_job.append("\u{2502}", 0.0, egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() });
            ui.label(marker_job);
        }

        // Content row: region name line
        {
            let mut name_job = egui::text::LayoutJob::default();
            if indent_chars > 0 {
                name_job.append(
                    &" ".repeat(indent_chars),
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: dim, ..Default::default() },
                );
            }
            for &q in qs.iter() {
                name_job.append("\u{2502}", 0.0, egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() });

                let region_idx = coord_to_idx.get(&(q, *r)).copied().unwrap_or(usize::MAX);
                if region_idx < overworld.regions.len() {
                    let region = &overworld.regions[region_idx];
                    let faction_col = faction_color(region.owner_faction_id);
                    let is_current = region_idx == overworld.current_region;
                    let is_selected = region_idx == overworld.selected_region;

                    // Truncate name to fit cell
                    let max_name = cell_w - 2;
                    let name: String = region.name.chars().take(max_name).collect();

                    let name_color = if is_current {
                        egui::Color32::WHITE
                    } else if is_selected {
                        egui::Color32::from_rgb(
                            faction_col.r().saturating_add(70),
                            faction_col.g().saturating_add(70),
                            faction_col.b().saturating_add(70),
                        )
                    } else {
                        faction_col
                    };

                    name_job.append(
                        &format!(" {:<width$}", name, width = max_name),
                        0.0,
                        egui::TextFormat { font_id: font.clone(), color: name_color, ..Default::default() },
                    );
                } else {
                    name_job.append(
                        &" ".repeat(cell_w - 1),
                        0.0,
                        egui::TextFormat { font_id: font.clone(), color: dim, ..Default::default() },
                    );
                }
            }
            name_job.append("\u{2502}", 0.0, egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() });
            ui.label(name_job);
        }

        // Content row: faction owner line
        {
            let mut faction_job = egui::text::LayoutJob::default();
            if indent_chars > 0 {
                faction_job.append(
                    &" ".repeat(indent_chars),
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: dim, ..Default::default() },
                );
            }
            for &q in qs.iter() {
                faction_job.append("\u{2502}", 0.0, egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() });

                let region_idx = coord_to_idx.get(&(q, *r)).copied().unwrap_or(usize::MAX);
                if region_idx < overworld.regions.len() {
                    let region = &overworld.regions[region_idx];
                    let faction_col = faction_color(region.owner_faction_id);
                    let faction_name = overworld.factions.get(region.owner_faction_id)
                        .map(|f| f.name.as_str())
                        .unwrap_or("???");
                    let max_f = cell_w - 3;
                    let fname: String = faction_name.chars().take(max_f).collect();

                    faction_job.append(
                        " \u{25a0} ",
                        0.0,
                        egui::TextFormat { font_id: small_font.clone(), color: faction_col, ..Default::default() },
                    );
                    faction_job.append(
                        &format!("{:<width$}", fname, width = max_f),
                        0.0,
                        egui::TextFormat { font_id: small_font.clone(), color: egui::Color32::from_rgb(150, 160, 175), ..Default::default() },
                    );
                } else {
                    faction_job.append(
                        &" ".repeat(cell_w - 1),
                        0.0,
                        egui::TextFormat { font_id: font.clone(), color: dim, ..Default::default() },
                    );
                }
            }
            faction_job.append("\u{2502}", 0.0, egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() });
            ui.label(faction_job);
        }

        // Bottom border (only for last row)
        let is_last_row = r_idx + 1 == rows.len();
        if is_last_row {
            let mut bot_job = egui::text::LayoutJob::default();
            if indent_chars > 0 {
                bot_job.append(
                    &" ".repeat(indent_chars),
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: dim, ..Default::default() },
                );
            }
            for (qi, _) in qs.iter().enumerate() {
                if qi == 0 {
                    bot_job.append("\u{2514}", 0.0, egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() });
                } else {
                    bot_job.append("\u{2534}", 0.0, egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() });
                }
                bot_job.append(
                    &"\u{2500}".repeat(cell_w - 1),
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() },
                );
            }
            bot_job.append("\u{2518}", 0.0, egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() });
            ui.label(bot_job);
        }
    }

    // Legend
    ui.add_space(10.0);
    {
        let legend_font = egui::FontId::monospace(12.0);
        let mut legend = egui::text::LayoutJob::default();
        legend.append(
            "@ current  ",
            0.0,
            egui::TextFormat { font_id: legend_font.clone(), color: egui::Color32::from_rgb(255, 255, 100), ..Default::default() },
        );
        legend.append(
            "> selected  ",
            0.0,
            egui::TextFormat { font_id: legend_font.clone(), color: egui::Color32::from_rgb(180, 220, 255), ..Default::default() },
        );
        legend.append(
            "! mission  ",
            0.0,
            egui::TextFormat { font_id: legend_font.clone(), color: egui::Color32::from_rgb(255, 200, 80), ..Default::default() },
        );
        legend.append(
            "\u{2502}  Factions: ",
            0.0,
            egui::TextFormat { font_id: legend_font.clone(), color: COLOR_SECTION, ..Default::default() },
        );
        for (i, faction) in overworld.factions.iter().enumerate() {
            let fc = faction_color(i);
            legend.append(
                &format!("\u{25a0} {}  ", faction.name),
                0.0,
                egui::TextFormat { font_id: legend_font.clone(), color: fc, ..Default::default() },
            );
        }
        ui.label(legend);
    }
}

fn draw_start_menu_central_panel(ctx: &egui::Context) {
    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(8, 12, 18))
                .inner_margin(egui::Margin::same(0.0)),
        )
        .show(ctx, |ui| {
            // Center the title art vertically and horizontally
            let available = ui.available_size();
            ui.allocate_ui_with_layout(
                available,
                egui::Layout::top_down(egui::Align::Center),
                |ui| {
                    ui.add_space((available.y * 0.22).max(40.0));
                    draw_ascii_title_art(ui);
                },
            );
        });
}

fn draw_ascii_title_art(ui: &mut egui::Ui) {
    let font = egui::FontId::monospace(18.0);
    let border_color = egui::Color32::from_rgb(100, 130, 170);
    let title_color = egui::Color32::from_rgb(240, 245, 255);
    let subtitle_color = egui::Color32::from_rgb(160, 180, 210);
    let dim_color = egui::Color32::from_rgb(60, 75, 95);

    let bar = "\u{2550}".repeat(43);
    let lines: &[(&str, egui::Color32)] = &[
        ("\u{2551}                                           \u{2551}", dim_color),
        ("\u{2551}        ADVENTURER'S  GUILD                \u{2551}", title_color),
        ("\u{2551}                                           \u{2551}", dim_color),
        ("\u{2551}      Command your company.                \u{2551}", subtitle_color),
        ("\u{2551}      Shape the realm.                     \u{2551}", subtitle_color),
        ("\u{2551}                                           \u{2551}", dim_color),
        ("\u{2551}                                           \u{2551}", dim_color),
        ("\u{2551}   A tactical RPG of strategy & sacrifice  \u{2551}", egui::Color32::from_rgb(100, 120, 150)),
        ("\u{2551}                                           \u{2551}", dim_color),
    ];

    // Top border
    {
        let mut job = egui::text::LayoutJob::default();
        job.append(
            &format!("\u{2554}{}\u{2557}", bar),
            0.0,
            egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() },
        );
        ui.label(job);
    }

    for (text, color) in lines {
        let mut job = egui::text::LayoutJob::default();
        job.append(
            text,
            0.0,
            egui::TextFormat { font_id: font.clone(), color: *color, ..Default::default() },
        );
        ui.label(job);
    }

    // Mid separator
    {
        let mut job = egui::text::LayoutJob::default();
        job.append(
            &format!("\u{2560}{}\u{2563}", bar),
            0.0,
            egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() },
        );
        ui.label(job);
    }

    // Bottom border
    {
        let mut job = egui::text::LayoutJob::default();
        job.append(
            &format!("\u{255a}{}\u{255d}", bar),
            0.0,
            egui::TextFormat { font_id: font.clone(), color: border_color, ..Default::default() },
        );
        ui.label(job);
    }

    // Decorative mountain landscape below the title box
    ui.add_space(8.0);
    let mountain_font = egui::FontId::monospace(14.0);
    let mountain_color = egui::Color32::from_rgb(40, 55, 75);
    let star_color = egui::Color32::from_rgb(60, 80, 110);

    let mountain_lines: &[&[(& str, egui::Color32)]] = &[
        &[("        /\\       ", mountain_color), (".  *  .", star_color), ("       /\\", mountain_color)],
        &[("       /  \\  ", mountain_color), (".", star_color), ("       ", mountain_color), ("*", star_color), ("       /  \\   ", mountain_color), (".", star_color)],
        &[("      /    \\     ", mountain_color), (".", star_color), ("       ", mountain_color), (".", star_color), ("  /    \\", mountain_color)],
        &[("     /______\\  ", mountain_color), ("*", star_color), ("    ", mountain_color), (".", star_color), ("      /______\\    ", mountain_color), ("*", star_color)],
        &[("        ||          ", mountain_color), (".", star_color), ("         ||", mountain_color)],
    ];

    for segments in mountain_lines {
        let mut job = egui::text::LayoutJob::default();
        for (text, color) in *segments {
            job.append(
                text,
                0.0,
                egui::TextFormat { font_id: mountain_font.clone(), color: *color, ..Default::default() },
            );
        }
        ui.label(job);
    }
}

// ---------------------------------------------------------------------------
// Region local map
// ---------------------------------------------------------------------------

/// A simple seeded PRNG for deterministic town layout.
fn region_rng(seed: u64, step: u64) -> u64 {
    let mut s = seed.wrapping_add(step).wrapping_mul(6364136223846793005);
    s ^= s >> 33;
    s = s.wrapping_mul(0xff51afd7ed558ccd);
    s ^= s >> 33;
    s
}

/// Renders a procedural ASCII town/settlement map in the CentralPanel for the RegionView screen.
fn draw_region_local_map(
    ctx: &egui::Context,
    overworld: &game_core::OverworldMap,
    region_transition: &crate::region_nav::RegionLayerTransitionState,
) {
    const W: usize = 50;
    const H: usize = 25;

    // Colors
    const COL_GROUND: egui::Color32 = egui::Color32::from_rgb(35, 40, 35);
    const COL_WALL: egui::Color32 = egui::Color32::from_rgb(80, 75, 65);
    const COL_DOOR: egui::Color32 = egui::Color32::from_rgb(140, 120, 80);
    const COL_ROAD: egui::Color32 = egui::Color32::from_rgb(110, 110, 100);
    const COL_PATH: egui::Color32 = egui::Color32::from_rgb(90, 90, 80);
    const COL_PLAYER: egui::Color32 = egui::Color32::from_rgb(80, 200, 120);
    const COL_LABEL: egui::Color32 = egui::Color32::from_rgb(180, 175, 150);


    // Determine seed from region payload or fallback
    let (seed, faction_idx) = if let Some(payload) = region_transition.active_payload.as_ref() {
        (payload.region_seed, payload.faction_index)
    } else {
        (overworld.selected_region as u64, 0)
    };
    let entrance_color = faction_color(faction_idx);

    // --- Build the grid ---
    // Grid cells: (char, color)
    let mut grid = vec![('.', COL_GROUND); W * H];

    // Helper to set a cell
    let set = |grid: &mut Vec<(char, egui::Color32)>, x: usize, y: usize, ch: char, col: egui::Color32| {
        if x < W && y < H {
            grid[y * W + x] = (ch, col);
        }
    };

    // 1. Draw 2 horizontal roads
    let road_y1 = 8;
    let road_y2 = 16;
    for x in 2..W - 2 {
        set(&mut grid, x, road_y1, '\u{2550}', COL_ROAD); // ═
        set(&mut grid, x, road_y2, '\u{2550}', COL_ROAD);
    }

    // 2. Draw 2-3 vertical paths connecting roads
    let num_paths = 2 + (region_rng(seed, 0) % 2) as usize; // 2 or 3
    let mut path_xs: Vec<usize> = Vec::new();
    for i in 0..num_paths {
        let px = 10 + (region_rng(seed, 10 + i as u64) % 12) as usize * 3;
        let px = px.min(W - 4);
        path_xs.push(px);
        for y in (road_y1 + 1)..road_y2 {
            set(&mut grid, px, y, '|', COL_PATH);
        }
        // crossing marks
        set(&mut grid, px, road_y1, '\u{256a}', COL_ROAD); // ╪
        set(&mut grid, px, road_y2, '\u{256a}', COL_ROAD);
    }

    // 3. Place buildings
    let building_names = ["Tavern", "Market", "Smithy", "Garrison", "Temple", "Inn", "Stables", "Healer"];
    let num_buildings = 4 + (region_rng(seed, 1) % 3) as usize; // 4-6
    let num_buildings = num_buildings.min(building_names.len());

    // Shuffle building names deterministically
    let mut name_indices: Vec<usize> = (0..building_names.len()).collect();
    for i in (1..name_indices.len()).rev() {
        let j = (region_rng(seed, 100 + i as u64) % (i as u64 + 1)) as usize;
        name_indices.swap(i, j);
    }

    struct Building {
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        name: &'static str,
        door_x: usize,
        door_y: usize,
        entrance_x: usize,
        entrance_y: usize,
    }

    let mut buildings: Vec<Building> = Vec::new();
    let mut player_placed = false;

    for i in 0..num_buildings {
        let name = building_names[name_indices[i]];
        let bw = 8 + (region_rng(seed, 20 + i as u64) % 7) as usize; // 8-14
        let bh = 3 + (region_rng(seed, 30 + i as u64) % 3) as usize; // 3-5

        // Place buildings: first half above road_y1, second half below road_y2
        let (bx, by, door_side_y) = if i < (num_buildings + 1) / 2 {
            // Above road_y1
            let bx = 3 + (region_rng(seed, 40 + i as u64) % ((W - bw - 6) as u64).max(1)) as usize;
            let by = road_y1.saturating_sub(bh + 1);
            (bx, by, by + bh - 1) // door on bottom (road-facing)
        } else {
            // Below road_y2
            let bx = 3 + (region_rng(seed, 40 + i as u64) % ((W - bw - 6) as u64).max(1)) as usize;
            let by = road_y2 + 2;
            (bx, by, by) // door on top (road-facing)
        };

        // Check overlap with existing buildings (simple bounding box)
        let overlaps = buildings.iter().any(|b| {
            bx < b.x + b.w + 1 && bx + bw + 1 > b.x && by < b.y + b.h + 1 && by + bh + 1 > b.y
        });
        if overlaps || bx + bw >= W - 1 || by + bh >= H - 1 {
            continue;
        }

        // Draw walls
        for wx in bx..bx + bw {
            for wy in by..by + bh {
                set(&mut grid, wx, wy, '#', COL_WALL);
            }
        }

        // Door position (on road-facing side, near center)
        let door_x = bx + bw / 2;
        set(&mut grid, door_x, door_side_y, '+', COL_DOOR);

        // Entrance marker inside building
        let ent_x = bx + bw / 2;
        let ent_y = if door_side_y == by { by + 1 } else { by + bh - 2 };
        let ent_y = ent_y.min(by + bh - 1).max(by);
        set(&mut grid, ent_x, ent_y, '\u{2302}', entrance_color); // ⌂

        // Place player at first building entrance
        let (px, py) = if !player_placed {
            player_placed = true;
            let px = ent_x + 1;
            let py = ent_y;
            if px < bx + bw - 1 {
                set(&mut grid, px, py, '@', COL_PLAYER);
            }
            (px, py)
        } else {
            (ent_x, ent_y)
        };
        let _ = (px, py);

        buildings.push(Building {
            x: bx,
            y: by,
            w: bw,
            h: bh,
            name,
            door_x,
            door_y: door_side_y,
            entrance_x: ent_x,
            entrance_y: ent_y,
        });
    }

    // --- Render ---
    let font = egui::FontId::monospace(14.0);
    let label_font = egui::FontId::monospace(11.0);
    let legend_font = egui::FontId::monospace(12.0);

    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(8, 10, 14))
                .inner_margin(egui::Margin::same(14.0)),
        )
        .show(ctx, |ui| {
            // Region title
            if let Some(payload) = region_transition.active_payload.as_ref() {
                let region_name = overworld
                    .regions
                    .get(payload.region_id)
                    .map(|r| r.name.as_str())
                    .unwrap_or("Unknown");
                ui.label(
                    egui::RichText::new(format!("Local Map — {}", region_name))
                        .font(egui::FontId::monospace(15.0))
                        .color(egui::Color32::from_rgb(190, 200, 215)),
                );
                ui.add_space(4.0);
            }

            // Building labels (rendered above each building)
            // We collect label positions first so we can render them as part of the grid output
            // Actually, we render each row of the grid as a LayoutJob. Labels appear on the row
            // just above each building if there's space.

            // Build a label overlay: row -> Vec<(start_x, text)>
            let mut label_overlay: std::collections::HashMap<usize, Vec<(usize, &str)>> =
                std::collections::HashMap::new();
            for b in &buildings {
                let label_y = if b.y > 0 { b.y - 1 } else { 0 };
                // Center the name above the building
                let label_x = b.x + (b.w.saturating_sub(b.name.len())) / 2;
                label_overlay.entry(label_y).or_default().push((label_x, b.name));
            }

            // Render grid rows
            for y in 0..H {
                let mut job = egui::text::LayoutJob::default();

                // Check if this row has any labels
                let labels = label_overlay.get(&y);

                // We need to merge labels into the grid row
                let mut x = 0;
                while x < W {
                    // Check if a label starts at this x position
                    let label = labels.and_then(|ls| ls.iter().find(|(lx, _)| *lx == x));
                    if let Some((_, text)) = label {
                        job.append(
                            text,
                            0.0,
                            egui::TextFormat {
                                font_id: label_font.clone(),
                                color: COL_LABEL,
                                ..Default::default()
                            },
                        );
                        x += text.len();
                        continue;
                    }

                    // Check if we're inside a label span (skip those chars)
                    let inside_label = labels.map_or(false, |ls| {
                        ls.iter().any(|(lx, t)| x > *lx && x < *lx + t.len())
                    });
                    if inside_label {
                        x += 1;
                        continue;
                    }

                    // Normal grid cell
                    let (ch, col) = grid[y * W + x];
                    let s: String = ch.to_string();
                    job.append(
                        &s,
                        0.0,
                        egui::TextFormat {
                            font_id: font.clone(),
                            color: col,
                            ..Default::default()
                        },
                    );
                    x += 1;
                }

                ui.label(job);
            }

            // Legend
            ui.add_space(6.0);
            let faction_col = overworld.regions.get(overworld.selected_region).map(|r| crate::hub_ui_draw::faction_color(r.owner_faction_id)).unwrap_or(egui::Color32::WHITE);
            let mut legend = egui::text::LayoutJob::default();
            let lc = egui::Color32::from_rgb(100, 110, 125);
            legend.append("@ ", 0.0, egui::TextFormat { font_id: legend_font.clone(), color: COL_PLAYER, ..Default::default() });
            legend.append("You   ", 0.0, egui::TextFormat { font_id: legend_font.clone(), color: lc, ..Default::default() });
            legend.append("⌂ ", 0.0, egui::TextFormat { font_id: legend_font.clone(), color: faction_col, ..Default::default() });
            legend.append("Entrance   ", 0.0, egui::TextFormat { font_id: legend_font.clone(), color: lc, ..Default::default() });
            legend.append("+ ", 0.0, egui::TextFormat { font_id: legend_font.clone(), color: COL_DOOR, ..Default::default() });
            legend.append("Door   ", 0.0, egui::TextFormat { font_id: legend_font.clone(), color: lc, ..Default::default() });
            legend.append("# ", 0.0, egui::TextFormat { font_id: legend_font.clone(), color: COL_WALL, ..Default::default() });
            legend.append("Wall   ", 0.0, egui::TextFormat { font_id: legend_font.clone(), color: lc, ..Default::default() });
            legend.append("═ ", 0.0, egui::TextFormat { font_id: legend_font.clone(), color: COL_ROAD, ..Default::default() });
            legend.append("Road", 0.0, egui::TextFormat { font_id: legend_font.clone(), color: lc, ..Default::default() });
            ui.label(legend);
        });
}

fn draw_empty_central_panel(ctx: &egui::Context) {
    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(6, 9, 13))
                .inner_margin(egui::Margin::same(12.0)),
        )
        .show(ctx, |_ui| {});
}

// ---------------------------------------------------------------------------
// Main system
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn draw_ascii_viewport_system(
    mut contexts: EguiContexts,
    mut hub_ui: ResMut<HubUiState>,
    sim_state: Option<Res<MissionSimState>>,
    mut pane_state: ResMut<MissionPaneState>,
    event_buf: Option<Res<SimEventBuffer>>,
    replay_viewer: Option<Res<ReplayViewerState>>,
    last_replay: Option<Res<LastMissionReplay>>,
    mut overworld: ResMut<game_core::OverworldMap>,
    mut parties: ResMut<game_core::CampaignParties>,
    mut target_picker: ResMut<RegionTargetPickerState>,
    region_transition: Res<crate::region_nav::RegionLayerTransitionState>,
) {
    let ctx = contexts.ctx_mut();

    // For character creation and cinematic screens, skip all rendering
    if matches!(
        hub_ui.screen,
        HubScreen::CharacterCreationFaction
            | HubScreen::CharacterCreationBackstory
            | HubScreen::BackstoryCinematic
    ) {
        return;
    }

    // For the start menu, render the ASCII title art as the CentralPanel background
    if hub_ui.screen == HubScreen::StartMenu {
        draw_start_menu_central_panel(ctx);
        return;
    }

    let transition_locked = region_transition.interaction_locked;
    let party_snapshots = parties.parties.clone();

    // Auto-open mission pane when entering MissionExecution
    if hub_ui.screen == HubScreen::MissionExecution && sim_state.is_some() {
        pane_state.open = true;
    }

    // Ingest new events from the sim event buffer
    if let Some(ref buf) = event_buf {
        if !buf.events.is_empty() {
            let max = pane_state.max_feed;
            ingest_events(&mut pane_state.event_feed, &buf.events, max);
        }
    }

    // CentralPanel: overworld map on map-related screens, empty otherwise
    match hub_ui.screen {
        HubScreen::OverworldMap | HubScreen::MissionExecution | HubScreen::ReplayViewer => {
            let snaps = parties.parties.clone();
            draw_overworld_central_panel(
                ctx, &mut overworld, &mut target_picker, &mut parties,
                &snaps, transition_locked,
            );
        }
        HubScreen::RegionView => {
            draw_region_local_map(ctx, &overworld, &region_transition);
        }
        _ => {
            draw_empty_central_panel(ctx);
        }
    }

    // Floating combat pane: live mission
    if hub_ui.screen == HubScreen::MissionExecution {
        if let Some(ref sim) = sim_state {
            let retreat = draw_combat_pane(
                ctx,
                &mut pane_state,
                &sim.sim,
                sim.grid_nav.as_ref(),
                "Mission",
                None,
                true, // show retreat button
            );
            if retreat {
                hub_ui.screen = HubScreen::OverworldMap;
            }
        }
    }

    // Floating combat pane: replay viewer
    if hub_ui.screen == HubScreen::ReplayViewer {
        if let Some(ref viewer) = replay_viewer {
            if let Some(frame) = viewer.frames.get(viewer.frame_index) {
                let grid_nav = last_replay.as_ref().and_then(|r| r.grid_nav.as_ref());
                pane_state.open = true;
                draw_combat_pane(
                    ctx,
                    &mut pane_state,
                    frame,
                    grid_nav,
                    "Replay",
                    Some((viewer.frame_index, viewer.frames.len())),
                    false, // no retreat in replay
                );
            }
        }
    }

    // "View Battle" reopener — handled in side panel via MissionPaneState.open
}
