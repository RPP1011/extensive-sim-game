use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FadeDirection {
    FadeIn,
    FadeOut,
    None,
}

impl Default for FadeDirection {
    fn default() -> Self {
        FadeDirection::None
    }
}

#[derive(Resource)]
pub struct FadeState {
    pub alpha: f32,
    pub direction: FadeDirection,
    pub duration: f32,
}

impl Default for FadeState {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            direction: FadeDirection::None,
            duration: 0.5,
        }
    }
}

pub fn draw_fade_system(mut contexts: EguiContexts, fade_state: Res<FadeState>) {
    if fade_state.alpha <= 0.0 {
        return;
    }
    let ctx = contexts.ctx_mut();
    let screen_rect = ctx.screen_rect();
    let alpha_byte = (fade_state.alpha.clamp(0.0, 1.0) * 255.0) as u8;
    let color = egui::Color32::from_black_alpha(alpha_byte);

    egui::Area::new("fade_overlay".into())
        .fixed_pos(egui::pos2(0.0, 0.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.painter().rect_filled(screen_rect, 0.0, color);
        });
}

pub fn update_fade_system(mut fade_state: ResMut<FadeState>, time: Res<Time>) {
    match fade_state.direction {
        FadeDirection::FadeOut => {
            let delta = time.delta_seconds() / fade_state.duration;
            fade_state.alpha = (fade_state.alpha + delta).clamp(0.0, 1.0);
            if fade_state.alpha >= 1.0 {
                fade_state.direction = FadeDirection::None;
            }
        }
        FadeDirection::FadeIn => {
            let delta = time.delta_seconds() / fade_state.duration;
            fade_state.alpha = (fade_state.alpha - delta).clamp(0.0, 1.0);
            if fade_state.alpha <= 0.0 {
                fade_state.direction = FadeDirection::None;
            }
        }
        FadeDirection::None => {}
    }
}
