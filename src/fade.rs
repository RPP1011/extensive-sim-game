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

use crate::game_core::{HubScreen, HubUiState};
use crate::hub_types::ScreenHistory;

/// Drives screen transitions with fade-out → switch → fade-in.
#[derive(Resource, Default)]
pub struct ScreenTransition {
    /// Target screen to navigate to after fade-out completes.
    pub target: Option<HubScreen>,
    /// Whether we're waiting for fade-out to complete before switching.
    pub pending: bool,
}

impl ScreenTransition {
    pub fn request(&mut self, target: HubScreen) {
        self.target = Some(target);
        self.pending = true;
    }
}

/// System that orchestrates fade-based screen transitions.
/// When a transition is pending, starts fade-out. When fade-out completes,
/// switches screen and starts fade-in.
pub fn screen_transition_system(
    mut transition: ResMut<ScreenTransition>,
    mut fade: ResMut<FadeState>,
    mut hub_ui: ResMut<HubUiState>,
    mut history: ResMut<ScreenHistory>,
) {
    if !transition.pending {
        return;
    }

    let Some(target) = transition.target else {
        transition.pending = false;
        return;
    };

    match fade.direction {
        FadeDirection::None if fade.alpha <= 0.0 => {
            // Start fade-out
            fade.direction = FadeDirection::FadeOut;
            fade.duration = 0.25;
        }
        FadeDirection::None if fade.alpha >= 1.0 => {
            // Fade-out complete — switch screen, push history, start fade-in
            let old_screen = hub_ui.screen;
            hub_ui.screen = target;
            history.push(old_screen);
            fade.direction = FadeDirection::FadeIn;
            fade.duration = 0.25;
            transition.target = None;
            transition.pending = false;
        }
        _ => {
            // Fade is in progress — wait
        }
    }
}

/// Helper to navigate back to the previous screen (with transition).
pub fn navigate_back(
    transition: &mut ScreenTransition,
    history: &mut ScreenHistory,
) -> bool {
    if let Some(prev) = history.pop() {
        transition.request(prev);
        true
    } else {
        false
    }
}
