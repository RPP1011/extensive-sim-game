use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::HubScreen;
use crate::HubUiState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TIPS: [&str; 5] = [
    "Welcome! Click units to select them. Click ground to move selected units.",
    "Press Q/W/E to trigger unit abilities. Cooldowns shown on the ability bar.",
    "Defeat all enemies to advance to the next room. Reach the gold door when it appears.",
    "Press J to open the Quest Journal. Track companion stories and faction relations.",
    "Press Escape to retreat from a mission. Heroes can recover from injury between missions.",
];

// ---------------------------------------------------------------------------
// Resource
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
pub struct TutorialState {
    /// Persisted via serde; if true, never show again.
    pub completed: bool,
    pub current_tip: usize,
    pub dismissed: bool,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// On first frame after game starts, if not completed, ensure the tutorial is
/// visible by clearing `dismissed`.
pub fn tutorial_toggle_system(
    mut tutorial_state: ResMut<TutorialState>,
    mut first_frame_done: Local<bool>,
) {
    if !*first_frame_done {
        *first_frame_done = true;
        if !tutorial_state.completed {
            tutorial_state.dismissed = false;
        }
    }
}

/// Renders the tutorial tooltip window at bottom-center of the screen.
/// Only shows when the tutorial is not completed and the player is in an
/// active gameplay scene (MissionExecution or any Overworld screen).
pub fn draw_tutorial_system(
    mut contexts: EguiContexts,
    mut tutorial_state: ResMut<TutorialState>,
    hub_ui: Res<HubUiState>,
) {
    if tutorial_state.completed || tutorial_state.dismissed {
        return;
    }

    let in_active_scene = matches!(
        hub_ui.screen,
        HubScreen::MissionExecution
            | HubScreen::Overworld
            | HubScreen::OverworldMap
            | HubScreen::RegionView
    );
    if !in_active_scene {
        return;
    }

    let ctx = contexts.ctx_mut();
    let screen_rect = ctx.screen_rect();
    let window_width = 480.0_f32;
    let window_x = screen_rect.center().x - window_width / 2.0;
    let window_y = screen_rect.max.y - 120.0;

    egui::Window::new("Tutorial")
        .fixed_pos(egui::pos2(window_x, window_y))
        .fixed_size(egui::vec2(window_width, 90.0))
        .collapsible(false)
        .resizable(false)
        .title_bar(true)
        .show(ctx, |ui| {
            let tip_index = tutorial_state.current_tip.min(TIPS.len() - 1);
            ui.label(TIPS[tip_index]);

            ui.add_space(8.0);

            ui.horizontal(|ui| {
                let is_last_tip = tip_index == TIPS.len() - 1;
                if is_last_tip {
                    if ui.button("Got it!").clicked() {
                        tutorial_state.completed = true;
                    }
                } else if ui.button("Next").clicked() {
                    tutorial_state.current_tip += 1;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(egui::Button::new(
                            egui::RichText::new("Skip").weak(),
                        ))
                        .clicked()
                    {
                        tutorial_state.completed = true;
                    }
                });
            });
        });
}
