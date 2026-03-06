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
            ui.separator();
            if ui.button("Close").clicked() {
                hub_ui.show_credits = false;
            }
        });
}
