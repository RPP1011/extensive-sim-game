//! ASCII combat grid renderer.
//!
//! Renders the tactical grid as colored ASCII text using LayoutJob.
//! Each cell is 2 chars wide. Units are shown as colored labels,
//! terrain as block/box-drawing characters.

use bevy_egui::egui;
use crate::ai::core::{SimState, Team};
use crate::ai::pathing::GridNav;

const COLOR_WALL: egui::Color32 = egui::Color32::from_rgb(60, 55, 50);
const COLOR_FLOOR: egui::Color32 = egui::Color32::from_rgb(38, 40, 35);
const COLOR_ELEVATED: egui::Color32 = egui::Color32::from_rgb(80, 75, 55);
const COLOR_HALF_COVER: egui::Color32 = egui::Color32::from_rgb(90, 80, 60);
const COLOR_HERO: egui::Color32 = egui::Color32::from_rgb(80, 200, 120);
const COLOR_ALLY: egui::Color32 = egui::Color32::from_rgb(80, 160, 255);
const COLOR_ENEMY: egui::Color32 = egui::Color32::from_rgb(255, 90, 80);
const COLOR_DEAD: egui::Color32 = egui::Color32::from_rgb(80, 80, 80);
const COLOR_CC: egui::Color32 = egui::Color32::from_rgb(230, 200, 80);

use super::COLOR_DIM;
use super::COLOR_SECTION;

/// Renders the combat grid as colored ASCII using LayoutJob for per-character coloring.
pub fn draw_combat_grid(ui: &mut egui::Ui, sim: &SimState, grid_nav: Option<&GridNav>, time_secs: f64) {
    let Some(nav) = grid_nav else {
        ui.colored_label(COLOR_DIM, "No grid data available.");
        return;
    };

    let cols = ((nav.max_x - nav.min_x) / nav.cell_size).ceil() as i32;
    let rows = ((nav.max_y - nav.min_y) / nav.cell_size).ceil() as i32;

    // Build unit position lookup
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

    // Top border
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

        let row_label = format!("{:>2} ", row);
        job.append(
            &row_label,
            0.0,
            egui::TextFormat { font_id: row_num_font.clone(), color: COLOR_SECTION, ..Default::default() },
        );
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

        job.append(
            "\u{2502}",
            0.0,
            egui::TextFormat { font_id: grid_font.clone(), color: COLOR_DIM, ..Default::default() },
        );
        ui.label(job);
    }

    // Bottom border
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
