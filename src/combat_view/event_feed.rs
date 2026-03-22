//! Combat event feed — ingests SimEvents and renders colored log lines.

use std::collections::VecDeque;
use bevy_egui::egui;
use crate::ai::core::SimEvent;

use super::COLOR_DIM;

const COLOR_DMG: egui::Color32 = egui::Color32::from_rgb(255, 180, 60);
const COLOR_HEAL: egui::Color32 = egui::Color32::from_rgb(80, 220, 80);
const COLOR_DEATH: egui::Color32 = egui::Color32::from_rgb(255, 60, 60);
const COLOR_CC_EVENT: egui::Color32 = egui::Color32::from_rgb(230, 200, 80);

/// Process new SimEvents into the event feed.
pub fn ingest_events(
    feed: &mut VecDeque<(String, egui::Color32)>,
    events: &[SimEvent],
    max: usize,
) {
    for event in events {
        let (msg, color) = match event {
            SimEvent::DamageApplied { source_id, target_id, amount, target_hp_after, .. } => {
                (
                    format!("#{} \u{2192} #{}: {} dmg (\u{2192}{}hp)", source_id, target_id, amount, target_hp_after),
                    COLOR_DMG,
                )
            }
            SimEvent::HealApplied { source_id, target_id, amount, target_hp_after, .. } => {
                (
                    format!("#{} heals #{} +{} (\u{2192}{}hp)", source_id, target_id, amount, target_hp_after),
                    COLOR_HEAL,
                )
            }
            SimEvent::UnitDied { unit_id, .. } => {
                (format!("#{} has fallen!", unit_id), COLOR_DEATH)
            }
            SimEvent::ControlApplied { source_id, target_id, duration_ms, .. } => {
                (
                    format!("#{} CC \u{2192} #{} ({}ms)", source_id, target_id, duration_ms),
                    COLOR_CC_EVENT,
                )
            }
            _ => continue,
        };
        feed.push_back((msg, color));
        while feed.len() > max {
            feed.pop_front();
        }
    }
}

/// Draw the event feed as colored log lines.
pub fn draw_event_feed(ui: &mut egui::Ui, feed: &VecDeque<(String, egui::Color32)>) {
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
