use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use bevy_game::mapgen_gemini;
use std::env;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use crate::campaign_ops::truncate_for_hud;
use crate::game_core::{self, CharacterCreationState, HubScreen, HubUiState};
use crate::runtime_assets::{
    RuntimeAssetGenState, detect_image_extension_from_bytes, decode_preview_image,
};

use super::builders::*;
use super::types::*;

pub fn backstory_cinematic_state_reset_system(
    hub_ui: Res<HubUiState>,
    mut cinematic: ResMut<BackstoryCinematicState>,
    mut narrative_gen: ResMut<BackstoryNarrativeGenState>,
) {
    if hub_ui.screen != HubScreen::BackstoryCinematic && cinematic.phase != BackstoryCinematicPhase::Idle
    {
        *cinematic = BackstoryCinematicState::default();
        narrative_gen.in_flight = false;
        narrative_gen.requested_seed = None;
        if let Ok(mut slot) = narrative_gen.shared_result.lock() {
            *slot = None;
        }
    }
}

pub fn backstory_cinematic_bootstrap_system(
    hub_ui: Res<HubUiState>,
    character_creation: Res<CharacterCreationState>,
    overworld: Res<game_core::OverworldMap>,
    mut cinematic: ResMut<BackstoryCinematicState>,
    mut asset_gen: ResMut<RuntimeAssetGenState>,
) {
    if hub_ui.screen != HubScreen::BackstoryCinematic {
        return;
    }
    if cinematic.phase != BackstoryCinematicPhase::Idle || !cinematic.beats.is_empty() {
        return;
    }

    let beats = build_backstory_cinematic_beats(&character_creation, &overworld);
    if beats.is_empty() {
        cinematic.status = "Cinematic bootstrap failed: no beat prompts.".to_string();
        return;
    }
    asset_gen
        .pending
        .retain(|job| job.scene_tag.as_deref() != Some("backstory_cinematic"));
    let portrait_prompt = build_character_portrait_prompt(&character_creation, &overworld);
    enqueue_backstory_portrait_job(&mut asset_gen, portrait_prompt);
    cinematic.initialized_for_campaign_seed = Some(overworld.map_seed);
    cinematic.phase = BackstoryCinematicPhase::Loading;
    cinematic.narrative_summary =
        build_backstory_narrative_summary(&character_creation, &overworld);
    cinematic.current_beat = 0;
    cinematic.beat_elapsed_seconds = 0.0;
    cinematic.beats_enqueued = false;
    cinematic.portrait_image_file = None;
    cinematic.status =
        "Forging your portrait reference first, then story scenes.".to_string();
    cinematic.beats = beats;
}

pub fn backstory_narrative_gen_dispatch_system(
    hub_ui: Res<HubUiState>,
    character_creation: Res<CharacterCreationState>,
    overworld: Res<game_core::OverworldMap>,
    cinematic: Res<BackstoryCinematicState>,
    mut narrative_gen: ResMut<BackstoryNarrativeGenState>,
) {
    if hub_ui.screen != HubScreen::BackstoryCinematic {
        return;
    }
    if cinematic.beats.is_empty() || cinematic.initialized_for_campaign_seed.is_none() {
        return;
    }
    let campaign_seed = cinematic.initialized_for_campaign_seed.unwrap_or(overworld.map_seed);
    if narrative_gen.in_flight || narrative_gen.requested_seed == Some(campaign_seed) {
        return;
    }

    narrative_gen.in_flight = true;
    narrative_gen.requested_seed = Some(campaign_seed);
    let model = narrative_gen.model.clone();
    let prompt = build_backstory_narrative_generation_prompt(
        &character_creation,
        &overworld,
        &cinematic.beats,
    );
    let beat_count = cinematic.beats.len();
    let shared_result = Arc::clone(&narrative_gen.shared_result);
    std::thread::spawn(move || {
        let mut success = false;
        let mut status = String::new();
        let mut summary = String::new();
        let mut beat_subtitles = Vec::new();
        let _ = mapgen_gemini::load_dotenv_if_present(Path::new(".env"));
        let api_key = match env::var("GEMINI_API_KEY") {
            Ok(v) if !v.trim().is_empty() => v,
            _ => {
                status = "Narrative text fallback active: GEMINI_API_KEY is missing.".to_string();
                String::new()
            }
        };
        if !api_key.is_empty() {
            match mapgen_gemini::call_gemini_text(&model, &prompt, &api_key) {
                Ok(response) => match mapgen_gemini::extract_parts(&response) {
                    Ok(outputs) => {
                        let Some(text) = outputs.text.as_deref() else {
                            status = "Narrative text fallback active: provider returned no text."
                                .to_string();
                            if let Ok(mut slot) = shared_result.lock() {
                                *slot = Some(BackstoryNarrativeResult {
                                    campaign_seed,
                                    summary,
                                    beat_subtitles,
                                    status,
                                    success,
                                });
                            }
                            return;
                        };
                        match parse_backstory_narrative_payload(text, beat_count) {
                            Ok((generated_summary, generated_subtitles)) => {
                                summary = generated_summary;
                                beat_subtitles = generated_subtitles;
                                success = true;
                                status = "Narrative text generated from Gemini.".to_string();
                            }
                            Err(err) => {
                                status =
                                    format!("Narrative text fallback active: provider output invalid ({err}).");
                            }
                        }
                    }
                    Err(err) => {
                        status = format!(
                            "Narrative text fallback active: provider parse failed ({err})."
                        );
                    }
                },
                Err(err) => {
                    status = format!("Narrative text fallback active: request failed ({err}).");
                }
            }
        }
        if let Ok(mut slot) = shared_result.lock() {
            *slot = Some(BackstoryNarrativeResult {
                campaign_seed,
                summary,
                beat_subtitles,
                status,
                success,
            });
        }
    });
}

pub fn backstory_narrative_gen_collect_system(
    hub_ui: Res<HubUiState>,
    mut cinematic: ResMut<BackstoryCinematicState>,
    mut narrative_gen: ResMut<BackstoryNarrativeGenState>,
) {
    if hub_ui.screen != HubScreen::BackstoryCinematic {
        return;
    }
    let maybe_result = if let Ok(mut slot) = narrative_gen.shared_result.lock() {
        slot.take()
    } else {
        None
    };
    let Some(result) = maybe_result else {
        return;
    };
    narrative_gen.in_flight = false;
    if cinematic.initialized_for_campaign_seed != Some(result.campaign_seed) {
        return;
    }
    if result.success {
        cinematic.narrative_summary = result.summary;
        for (idx, subtitle) in result.beat_subtitles.iter().enumerate() {
            if let Some(beat) = cinematic.beats.get_mut(idx) {
                beat.subtitle = subtitle.clone();
            }
        }
        cinematic.status = "Narrative text ready.".to_string();
    } else if !result.status.is_empty() {
        cinematic.status = result.status;
    }
}

pub fn backstory_cinematic_collect_system(
    hub_ui: Res<HubUiState>,
    mut runtime_asset_gen: ResMut<RuntimeAssetGenState>,
    mut cinematic: ResMut<BackstoryCinematicState>,
) {
    if hub_ui.screen != HubScreen::BackstoryCinematic {
        return;
    }
    for result in runtime_asset_gen.recent.iter() {
        if result.scene_tag.as_deref() != Some("backstory_cinematic") {
            continue;
        }
        if !cinematic.seen_job_ids.insert(result.job_id) {
            continue;
        }
        if result.source_id == "backstory-character-portrait" {
            if result.success {
                cinematic.portrait_image_file = result.image_file.clone();
                cinematic.status = "Portrait reference ready. Generating story beats in parallel."
                    .to_string();
            } else {
                cinematic.status = result.status.clone();
            }
            continue;
        }
        let Some(beat_idx) = result.sequence_index else {
            continue;
        };
        let Some(beat) = cinematic.beats.get_mut(beat_idx) else {
            continue;
        };
        if result.success {
            beat.image_file = result.image_file.clone();
            cinematic.status = format!("Beat '{}' ready.", beat.title);
        } else {
            cinematic.status = result.status.clone();
        }
    }
    if cinematic.portrait_image_file.is_some() && !cinematic.beats_enqueued {
        enqueue_backstory_cinematic_jobs(
            &mut runtime_asset_gen,
            &cinematic.beats,
            cinematic.portrait_image_file.clone(),
        );
        cinematic.beats_enqueued = true;
    }
    if cinematic.phase == BackstoryCinematicPhase::Loading
        && cinematic
            .beats
            .first()
            .is_some_and(|beat| beat.image_file.is_some())
    {
        cinematic.phase = BackstoryCinematicPhase::Playing;
        cinematic.status = "Backstory cinematic started.".to_string();
    }
}

pub fn backstory_cinematic_texture_load_system(
    hub_ui: Res<HubUiState>,
    mut cinematic: ResMut<BackstoryCinematicState>,
    mut images: ResMut<Assets<Image>>,
) {
    if hub_ui.screen != HubScreen::BackstoryCinematic {
        return;
    }
    for beat in cinematic.beats.iter_mut() {
        if beat.texture_handle.is_some() {
            continue;
        }
        let Some(path) = beat.image_file.as_ref() else {
            continue;
        };
        let Ok(bytes) = fs::read(path) else {
            continue;
        };
        let extension = detect_image_extension_from_bytes(&bytes);
        if let Ok(image) = decode_preview_image(&bytes, extension) {
            beat.texture_handle = Some(images.add(image));
        }
    }
}

pub fn backstory_cinematic_playback_system(
    time: Res<Time>,
    mut hub_ui: ResMut<HubUiState>,
    mut cinematic: ResMut<BackstoryCinematicState>,
) {
    if hub_ui.screen != HubScreen::BackstoryCinematic {
        return;
    }
    if cinematic.phase != BackstoryCinematicPhase::Playing || cinematic.beats.is_empty() {
        return;
    }
    cinematic.beat_elapsed_seconds += time.delta_seconds();
    if cinematic.beat_elapsed_seconds < cinematic.beat_duration_seconds {
        return;
    }

    let next = cinematic.current_beat + 1;
    if next >= cinematic.beats.len() {
        cinematic.status = "Backstory cinematic complete. Entering overworld.".to_string();
        hub_ui.screen = HubScreen::OverworldMap;
        return;
    }
    if cinematic.beats[next].texture_handle.is_some() || cinematic.beats[next].image_file.is_some() {
        cinematic.current_beat = next;
        cinematic.beat_elapsed_seconds = 0.0;
    } else {
        cinematic.status = format!(
            "Waiting for beat '{}' render to finish...",
            cinematic.beats[next].title
        );
        cinematic.beat_elapsed_seconds = cinematic.beat_duration_seconds * 0.8;
    }
}

pub fn draw_backstory_cinematic_egui_system(
    mut contexts: EguiContexts,
    hub_ui: Res<HubUiState>,
    cinematic: Res<BackstoryCinematicState>,
) {
    if hub_ui.screen != HubScreen::BackstoryCinematic {
        return;
    }
    let current = cinematic.beats.get(cinematic.current_beat);
    let texture_id = current
        .and_then(|beat| beat.texture_handle.as_ref().cloned())
        .map(|handle| contexts.add_image(handle));
    egui::CentralPanel::default()
        .frame(egui::Frame::none().fill(egui::Color32::from_rgb(6, 8, 12)))
        .show(contexts.ctx_mut(), |ui| {
            let rect = ui.max_rect();
            let painter = ui.painter_at(rect);
            if let Some(texture) = texture_id {
                let t = (cinematic.beat_elapsed_seconds / cinematic.beat_duration_seconds)
                    .clamp(0.0, 1.0);
                let zoom = 1.02 + 0.08 * t;
                let uv_w = 1.0 / zoom;
                let uv_h = 1.0 / zoom;
                let dir = if cinematic.current_beat % 2 == 0 { 1.0 } else { -1.0 };
                let pan_x = (0.5 - uv_w * 0.5) + dir * 0.1 * t * (1.0 - uv_w);
                let pan_y = (0.5 - uv_h * 0.5) + 0.06 * (1.0 - t) * (1.0 - uv_h);
                let uv = egui::Rect::from_min_max(
                    egui::pos2(pan_x.clamp(0.0, 1.0 - uv_w), pan_y.clamp(0.0, 1.0 - uv_h)),
                    egui::pos2(
                        (pan_x + uv_w).clamp(uv_w, 1.0),
                        (pan_y + uv_h).clamp(uv_h, 1.0),
                    ),
                );
                painter.image(texture, rect, uv, egui::Color32::WHITE);
                painter.rect_filled(
                    rect,
                    0.0,
                    egui::Color32::from_rgba_premultiplied(5, 7, 10, 45),
                );
            } else {
                painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(6, 8, 12));
                let ready = cinematic
                    .beats
                    .iter()
                    .filter(|beat| beat.image_file.is_some())
                    .count();
                let total = cinematic.beats.len();
                painter.text(
                    rect.center_top() + egui::vec2(0.0, 120.0),
                    egui::Align2::CENTER_TOP,
                    format!("Forging Backstory Scene... ({}/{})", ready, total),
                    egui::FontId::proportional(28.0),
                    egui::Color32::from_rgb(205, 220, 230),
                );
            }

            let title = current
                .map(|beat| beat.title.as_str())
                .unwrap_or("Loading Beat");
            let subtitle = current
                .map(|beat| beat.subtitle.as_str())
                .unwrap_or("Forging your chapter...");
            let summary = if cinematic.narrative_summary.is_empty() {
                "Unwritten origins."
            } else {
                cinematic.narrative_summary.as_str()
            };

            painter.text(
                egui::pos2(rect.left() + 32.0, rect.top() + 28.0),
                egui::Align2::LEFT_TOP,
                truncate_for_hud(summary, 180),
                egui::FontId::proportional(18.0),
                egui::Color32::from_rgb(204, 214, 225),
            );
            painter.text(
                egui::pos2(rect.left() + 32.0, rect.bottom() - 82.0),
                egui::Align2::LEFT_BOTTOM,
                title,
                egui::FontId::proportional(34.0),
                egui::Color32::from_rgb(232, 236, 240),
            );
            painter.text(
                egui::pos2(rect.left() + 32.0, rect.bottom() - 56.0),
                egui::Align2::LEFT_BOTTOM,
                truncate_for_hud(subtitle, 130),
                egui::FontId::proportional(22.0),
                egui::Color32::from_rgb(216, 225, 235),
            );
            painter.text(
                egui::pos2(rect.left() + 32.0, rect.bottom() - 28.0),
                egui::Align2::LEFT_BOTTOM,
                truncate_for_hud(&cinematic.status, 130),
                egui::FontId::proportional(16.0),
                egui::Color32::from_rgb(186, 198, 210),
            );
        });
}
