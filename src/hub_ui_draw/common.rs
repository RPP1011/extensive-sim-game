//! Shared drawing helpers used across hub UI sub-modules.

use bevy_egui::egui;
use crate::game_core::HubUiState;

/// Apply the standard hub UI style to the egui context.
pub(crate) fn apply_hub_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(10.0, 10.0);
    style.spacing.button_padding = egui::vec2(10.0, 8.0);
    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::new(28.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(18.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::new(18.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Small,
        egui::FontId::new(14.0, egui::FontFamily::Proportional),
    );
    ctx.set_style(style);
}

/// ASCII separator line replacing `ui.separator()`.
pub(crate) fn ascii_separator(ui: &mut egui::Ui) {
    let font = egui::FontId::monospace(12.0);
    let width = (ui.available_width() / 7.8) as usize; // approximate char count
    let line: String = "─".repeat(width.max(20).min(80));
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &line,
        0.0,
        egui::TextFormat {
            font_id: font,
            color: egui::Color32::from_rgb(45, 52, 62),
            ..Default::default()
        },
    );
    ui.label(job);
}

/// ASCII button rendered as clickable `[ label ]` monospace text.
/// Returns true if clicked. Replaces native `ui.button()`.
pub(crate) fn ascii_button(ui: &mut egui::Ui, label: &str) -> bool {
    ascii_button_colored(ui, label, egui::Color32::from_rgb(170, 180, 195))
}

/// ASCII button with custom color.
pub(crate) fn ascii_button_colored(ui: &mut egui::Ui, label: &str, color: egui::Color32) -> bool {
    let font = egui::FontId::monospace(14.0);
    let text = format!("[ {} ]", label);
    let response = ui.add(
        egui::Label::new(
            egui::RichText::new(&text).font(font).color(color)
        ).sense(egui::Sense::click())
    );
    response.clicked()
}

/// ASCII button that's conditionally enabled.
pub(crate) fn ascii_button_enabled(ui: &mut egui::Ui, label: &str, enabled: bool) -> bool {
    if !enabled {
        let font = egui::FontId::monospace(14.0);
        let text = format!("[ {} ]", label);
        ui.label(egui::RichText::new(&text).font(font).color(egui::Color32::from_rgb(60, 65, 72)));
        return false;
    }
    ascii_button(ui, label)
}

/// ASCII progress bar rendered as `████░░░░ XX%` via LayoutJob.
pub(crate) fn ascii_progress_bar(ui: &mut egui::Ui, progress: f32, width: usize) {
    let font = egui::FontId::monospace(13.0);
    let clamped = progress.clamp(0.0, 1.0);
    let filled = (clamped * width as f32).round() as usize;
    let empty = width - filled;
    let bar_filled: String = "█".repeat(filled);
    let bar_empty: String = "░".repeat(empty);
    let pct = format!(" {:>3.0}%", clamped * 100.0);

    let bar_color = if clamped > 0.6 {
        egui::Color32::from_rgb(80, 200, 80)
    } else if clamped > 0.3 {
        egui::Color32::from_rgb(200, 180, 50)
    } else {
        egui::Color32::from_rgb(200, 60, 40)
    };

    let mut job = egui::text::LayoutJob::default();
    job.append(&bar_filled, 0.0, egui::TextFormat { font_id: font.clone(), color: bar_color, ..Default::default() });
    job.append(&bar_empty, 0.0, egui::TextFormat { font_id: font.clone(), color: egui::Color32::from_rgb(50, 50, 50), ..Default::default() });
    job.append(&pct, 0.0, egui::TextFormat { font_id: font, color: egui::Color32::from_rgb(140, 150, 165), ..Default::default() });
    ui.label(job);
}

/// Draw the credits overlay window.
pub(crate) fn draw_credits_window(ctx: &egui::Context, hub_ui: &mut HubUiState) {
    egui::Window::new("Credits")
        .collapsible(false)
        .resizable(false)
        .show(ctx, |ui| {
            ui.heading("Adventurer's Guild Prototype");
            ui.label("Built with Rust + Bevy.");
            ui.label("UI: bevy_egui immediate mode interface.");
            ui.label("Design focus: deterministic tactical orchestration.");
            ascii_separator(ui);
            if ascii_button(ui, "Close") {
                hub_ui.show_credits = false;
            }
        });
}
