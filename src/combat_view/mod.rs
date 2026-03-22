//! Reusable combat window component.
//!
//! Provides a `CombatView` widget that composes grid rendering, unit roster,
//! event feed, hero command panel, and mission header into a single reusable
//! egui window. Works in both live combat and replay modes via the
//! `CombatDataSource` trait abstraction.

mod data_source;
mod grid;
mod roster;
mod event_feed;
mod command_panel;
mod header;

pub use data_source::CombatDataSource;

use std::collections::VecDeque;
use bevy_egui::egui;

use crate::ai::core::SimEvent;

// Re-export colors used across submodules.
pub(crate) const COLOR_DIM: egui::Color32 = egui::Color32::from_rgb(55, 60, 68);
pub(crate) const COLOR_HEADER: egui::Color32 = egui::Color32::from_rgb(160, 170, 185);
pub(crate) const COLOR_SECTION: egui::Color32 = egui::Color32::from_rgb(100, 110, 125);
pub(crate) const COLOR_ALLY: egui::Color32 = egui::Color32::from_rgb(80, 160, 255);
pub(crate) const COLOR_ENEMY: egui::Color32 = egui::Color32::from_rgb(255, 90, 80);

/// Reusable combat view widget.
///
/// Holds persistent UI state (event feed, scroll, selection) and renders
/// a full combat window using any `CombatDataSource` implementation.
pub struct CombatView {
    /// Rolling event feed with colored messages.
    pub event_feed: VecDeque<(String, egui::Color32)>,
    /// Maximum number of feed entries to retain.
    pub max_feed: usize,
    /// Currently selected unit (for command panel).
    pub selected_unit: Option<u32>,
}

impl Default for CombatView {
    fn default() -> Self {
        Self {
            event_feed: VecDeque::new(),
            max_feed: 20,
            selected_unit: None,
        }
    }
}

impl CombatView {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest new simulation events into the event feed.
    pub fn ingest_events(&mut self, events: &[SimEvent]) {
        event_feed::ingest_events(&mut self.event_feed, events, self.max_feed);
    }

    /// Draw the full combat view inside the provided egui::Ui.
    ///
    /// The `source` provides all simulation data. `show_commands` controls
    /// whether the hero command panel is displayed (true for live combat,
    /// false for replay).
    pub fn draw(
        &mut self,
        ui: &mut egui::Ui,
        source: &dyn CombatDataSource,
        show_commands: bool,
    ) {
        let sim = source.sim_state();

        // ── Header ──
        header::draw_header(ui, source);

        draw_separator(ui);

        // ── Grid ──
        egui::ScrollArea::both().max_height(340.0).show(ui, |ui| {
            grid::draw_combat_grid(ui, sim, source.grid_nav(), ui.ctx().input(|i| i.time));
        });

        draw_separator(ui);

        // ── Roster ──
        roster::draw_unit_roster(ui, sim);

        draw_separator(ui);

        // ── Event feed ──
        section_header(ui, "COMBAT LOG");
        egui::ScrollArea::vertical()
            .max_height(130.0)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                event_feed::draw_event_feed(ui, &self.event_feed);
            });

        // ── Hero commands (live only) ──
        if show_commands {
            draw_separator(ui);
            command_panel::draw_hero_commands(ui, sim);
        }
    }
}

/// Section header in `╔═ TITLE ═══...` format.
pub(crate) fn section_header(ui: &mut egui::Ui, title: &str) {
    let font = egui::FontId::monospace(12.0);
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &format!("\u{2554}\u{2550} {} ", title),
        0.0,
        egui::TextFormat { font_id: font.clone(), color: COLOR_HEADER, ..Default::default() },
    );
    let fill_count = 60usize.saturating_sub(title.len() + 4);
    job.append(
        &"\u{2550}".repeat(fill_count),
        0.0,
        egui::TextFormat { font_id: font.clone(), color: COLOR_DIM, ..Default::default() },
    );
    ui.label(job);
}

/// Horizontal separator line.
pub(crate) fn draw_separator(ui: &mut egui::Ui) {
    let font = egui::FontId::monospace(12.0);
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &"\u{2500}".repeat(66),
        0.0,
        egui::TextFormat { font_id: font.clone(), color: COLOR_DIM, ..Default::default() },
    );
    ui.label(job);
}
