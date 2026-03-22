//! Unit roster panel — allies and enemies side by side with HP bars.

use bevy_egui::egui;
use crate::ai::core::{SimState, Team};

use super::{COLOR_ALLY, COLOR_ENEMY, COLOR_DIM};

const COLOR_HERO: egui::Color32 = egui::Color32::from_rgb(80, 200, 120);
const COLOR_DEAD: egui::Color32 = egui::Color32::from_rgb(80, 80, 80);
const COLOR_CC: egui::Color32 = egui::Color32::from_rgb(230, 200, 80);
const COLOR_HP_HIGH: egui::Color32 = egui::Color32::from_rgb(80, 200, 80);
const COLOR_HP_MID: egui::Color32 = egui::Color32::from_rgb(220, 200, 50);
const COLOR_HP_LOW: egui::Color32 = egui::Color32::from_rgb(220, 60, 40);
const COLOR_HP_BG: egui::Color32 = egui::Color32::from_rgb(50, 50, 50);

fn hp_color(ratio: f32) -> egui::Color32 {
    if ratio > 0.6 {
        COLOR_HP_HIGH
    } else if ratio > 0.3 {
        COLOR_HP_MID
    } else {
        COLOR_HP_LOW
    }
}

/// Renders a two-column unit roster: allies on left, enemies on right.
pub fn draw_unit_roster(ui: &mut egui::Ui, sim: &SimState) {
    let font = egui::FontId::monospace(13.0);
    let header_font = egui::FontId::monospace(12.0);

    let heroes: Vec<_> = sim.units.iter().enumerate()
        .filter(|(_, u)| u.team == Team::Hero)
        .collect();
    let enemies: Vec<_> = sim.units.iter().filter(|u| u.team == Team::Enemy).collect();
    let max_rows = heroes.len().max(enemies.len());

    // Section headers
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
            let label = if *slot == 0 { "@H".to_string() } else { format!("a{}", slot) };
            let label_color = if *slot == 0 { COLOR_HERO } else { COLOR_ALLY };

            job.append(
                &format!("{:<3}", label),
                0.0,
                egui::TextFormat { font_id: font.clone(), color: label_color, ..Default::default() },
            );

            if unit.hp <= 0 {
                job.append(
                    "DEAD                           ",
                    0.0,
                    egui::TextFormat { font_id: font.clone(), color: COLOR_DEAD, ..Default::default() },
                );
            } else {
                render_hp_bar(&mut job, &font, unit.hp, unit.max_hp, bar_len);
                render_status(&mut job, &font, unit.control_remaining_ms, unit.casting.is_some());
            }
        } else {
            job.append(
                &" ".repeat(31),
                0.0,
                egui::TextFormat { font_id: font.clone(), color: COLOR_DIM, ..Default::default() },
            );
        }

        // Separator
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
                render_hp_bar(&mut job, &font, unit.hp, unit.max_hp, bar_len);
                render_status(&mut job, &font, unit.control_remaining_ms, unit.casting.is_some());
            }
        }

        ui.label(job);
    }
}

fn render_hp_bar(
    job: &mut egui::text::LayoutJob,
    font: &egui::FontId,
    hp: i32,
    max_hp: i32,
    bar_len: usize,
) {
    let ratio = hp as f32 / max_hp.max(1) as f32;
    let filled = ((ratio * bar_len as f32).round() as usize).min(bar_len);

    job.append(
        &"\u{2588}".repeat(filled),
        0.0,
        egui::TextFormat { font_id: font.clone(), color: hp_color(ratio), ..Default::default() },
    );
    job.append(
        &"\u{2591}".repeat(bar_len - filled),
        0.0,
        egui::TextFormat { font_id: font.clone(), color: COLOR_HP_BG, ..Default::default() },
    );

    let hp_text = format!(" {:>4}/{:<4}", hp, max_hp);
    job.append(
        &hp_text,
        0.0,
        egui::TextFormat { font_id: font.clone(), color: egui::Color32::LIGHT_GRAY, ..Default::default() },
    );
}

fn render_status(
    job: &mut egui::text::LayoutJob,
    font: &egui::FontId,
    control_remaining_ms: u32,
    is_casting: bool,
) {
    if control_remaining_ms > 0 {
        job.append(
            " CC",
            0.0,
            egui::TextFormat { font_id: font.clone(), color: COLOR_CC, ..Default::default() },
        );
    } else if is_casting {
        job.append(
            " ...",
            0.0,
            egui::TextFormat { font_id: font.clone(), color: super::COLOR_HEADER, ..Default::default() },
        );
    } else {
        job.append(
            "    ",
            0.0,
            egui::TextFormat { font_id: font.clone(), color: COLOR_DIM, ..Default::default() },
        );
    }
}
