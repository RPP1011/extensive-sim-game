//! Hero command panel — shows abilities, cooldowns, and movement commands.

use bevy_egui::egui;
use crate::ai::core::{SimState, Team};
use super::{section_header, COLOR_DIM};

const COLOR_READY: egui::Color32 = egui::Color32::from_rgb(80, 220, 80);
const COLOR_COOLDOWN: egui::Color32 = egui::Color32::from_rgb(120, 120, 130);
const COLOR_KEY: egui::Color32 = egui::Color32::from_rgb(200, 180, 100);
const COLOR_LABEL: egui::Color32 = egui::Color32::from_rgb(180, 190, 200);
const COLOR_VALUE: egui::Color32 = egui::Color32::from_rgb(150, 160, 175);
const COLOR_MOVEMENT: egui::Color32 = egui::Color32::from_rgb(140, 150, 165);
const COLOR_DMG: egui::Color32 = egui::Color32::from_rgb(255, 180, 60);
const COLOR_HEAL: egui::Color32 = egui::Color32::from_rgb(80, 220, 80);
const COLOR_CC: egui::Color32 = egui::Color32::from_rgb(230, 200, 80);

/// Draws the hero command panel for the first living hero unit.
pub fn draw_hero_commands(ui: &mut egui::Ui, sim: &SimState) {
    let font = egui::FontId::monospace(13.0);
    let small_font = egui::FontId::monospace(12.0);

    let hero = sim.units.iter().find(|u| u.team == Team::Hero && u.hp > 0);
    let Some(hero) = hero else {
        return;
    };

    section_header(ui, "HERO COMMANDS");

    // [Q] Attack
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

    // [W] Ability
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

    // [E] Heal
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

    // [R] Control
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

    // Engine abilities
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

    // Movement commands
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
