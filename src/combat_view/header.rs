//! Mission header bar — tick count, team counts, objective, replay info.

use bevy_egui::egui;
use crate::ai::core::Team;
use super::data_source::CombatDataSource;
use super::{COLOR_ALLY, COLOR_ENEMY, COLOR_SECTION, COLOR_HEADER};

/// Draw the combat header showing tick, team counts, and objective/replay info.
pub fn draw_header(ui: &mut egui::Ui, source: &dyn CombatDataSource) {
    let sim = source.sim_state();
    let header_font = egui::FontId::monospace(12.0);

    let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
    let heroes_total = sim.units.iter().filter(|u| u.team == Team::Hero).count();
    let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
    let enemies_total = sim.units.iter().filter(|u| u.team == Team::Enemy).count();

    // Main header line
    {
        let mut job = egui::text::LayoutJob::default();
        job.append(
            &format!("Tk {:>5}  \u{25b6}  ", source.tick()),
            0.0,
            egui::TextFormat { font_id: header_font.clone(), color: COLOR_SECTION, ..Default::default() },
        );
        job.append(
            &format!("Allies {}/{}  ", heroes_alive, heroes_total),
            0.0,
            egui::TextFormat { font_id: header_font.clone(), color: COLOR_ALLY, ..Default::default() },
        );
        job.append(
            "vs  ",
            0.0,
            egui::TextFormat { font_id: header_font.clone(), color: COLOR_SECTION, ..Default::default() },
        );
        job.append(
            &format!("Enemies {}/{}  ", enemies_alive, enemies_total),
            0.0,
            egui::TextFormat { font_id: header_font.clone(), color: COLOR_ENEMY, ..Default::default() },
        );
        ui.label(job);
    }

    // Replay info or objective
    if let Some((frame, total)) = source.replay_info() {
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
    } else {
        let obj = source.objective_text();
        if !obj.is_empty() {
            let mut obj_job = egui::text::LayoutJob::default();
            obj_job.append(
                "Obj: ",
                0.0,
                egui::TextFormat { font_id: header_font.clone(), color: super::COLOR_DIM, ..Default::default() },
            );
            obj_job.append(
                obj,
                0.0,
                egui::TextFormat { font_id: header_font.clone(), color: COLOR_SECTION, ..Default::default() },
            );
            ui.label(obj_job);
        }
    }
}
