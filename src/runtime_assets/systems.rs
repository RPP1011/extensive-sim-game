use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::texture::{CompressedImageFormats, ImageSampler, ImageType};
use bevy_egui::{egui, EguiContexts};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use crate::game_core::HubScreen;
use crate::game_core::HubUiState;
use crate::region_nav::RegionLayerTransitionState;
use crate::ui_helpers::paint_landscape_backsplash;

use super::types::*;

pub fn load_runtime_env_prompt_corpus(path: &Path) -> Result<Vec<RuntimeEnvPromptRow>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("Failed to read prompt corpus {}: {err}", path.display()))?;
    serde_json::from_str::<Vec<RuntimeEnvPromptRow>>(&raw)
        .map_err(|err| format!("Failed to parse prompt corpus {}: {err}", path.display()))
}

pub fn compose_runtime_environment_prompt(base_prompt: &str, style: RuntimeAssetStyle) -> String {
    format!(
        "{base_prompt}\nStyle direction: {}.\nHard constraints: environment only, no people, no humanoids, no foreground creatures, no text, no watermark, no UI.",
        style.as_prompt_suffix()
    )
}

pub fn compose_runtime_backstory_scene_prompt(base_prompt: &str, style: RuntimeAssetStyle) -> String {
    format!(
        "{base_prompt}\nStyle direction: {}.\nHard constraints: include exactly one human protagonist, keep face/features/wardrobe consistent with the provided reference portrait, no extra characters, no text, no watermark, no UI.",
        style.as_prompt_suffix()
    )
}

pub fn detect_image_extension_from_bytes(bytes: &[u8]) -> &'static str {
    if bytes.len() >= 8 && bytes[..8] == [137, 80, 78, 71, 13, 10, 26, 10] {
        return "png";
    }
    if bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
        return "jpg";
    }
    if bytes.len() >= 12 && bytes[..4] == *b"RIFF" && bytes[8..12] == *b"WEBP" {
        return "webp";
    }
    if bytes.len() >= 6 && bytes[..6] == *b"GIF89a" {
        return "gif";
    }
    if bytes.len() >= 6 && bytes[..6] == *b"GIF87a" {
        return "gif";
    }
    "png"
}

pub fn detect_mime_type_from_bytes(bytes: &[u8]) -> &'static str {
    match detect_image_extension_from_bytes(bytes) {
        "jpg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        _ => "image/png",
    }
}

pub fn decode_preview_image(bytes: &[u8], extension: &str) -> Result<Image, String> {
    if let Ok(image) = Image::from_buffer(
        bytes,
        ImageType::Extension(extension),
        CompressedImageFormats::NONE,
        true,
        ImageSampler::Default,
        RenderAssetUsages::RENDER_WORLD,
    ) {
        return Ok(image);
    }
    let dyn_img =
        image::load_from_memory(bytes).map_err(|err| format!("fallback decode failed: {}", err))?;
    let rgba = dyn_img.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba.into_raw(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    ))
}

pub fn queue_runtime_environment_jobs(
    asset_gen: &mut RuntimeAssetGenState,
    count: usize,
) -> Result<usize, String> {
    let corpus = load_runtime_env_prompt_corpus(&asset_gen.prompt_corpus_path)?;
    if corpus.is_empty() {
        return Ok(0);
    }
    let base_index = asset_gen.next_job_id as usize;
    let mut queued = 0usize;
    for offset in 0..count {
        let row = &corpus[(base_index + offset) % corpus.len()];
        let style = RuntimeAssetStyle::Concept;
        let job = RuntimeAssetJob {
            id: asset_gen.next_job_id,
            source_id: row.id.clone(),
            source_title: row.title.clone(),
            prompt: compose_runtime_environment_prompt(&row.prompt, style),
            style,
            kind: RuntimeAssetJobKind::EnvironmentScene,
            reference_image_path: None,
            scene_tag: None,
            sequence_index: None,
        };
        asset_gen.next_job_id += 1;
        asset_gen.pending.push_back(job);
        queued += 1;
    }
    Ok(queued)
}

pub fn runtime_asset_gen_bootstrap_system(
    hub_ui: Res<HubUiState>,
    mut asset_gen: ResMut<RuntimeAssetGenState>,
) {
    if asset_gen.auto_seeded {
        return;
    }
    if !matches!(
        hub_ui.screen,
        HubScreen::OverworldMap | HubScreen::GuildManagement | HubScreen::Overworld
    ) {
        return;
    }

    let seed_count = match queue_runtime_environment_jobs(&mut asset_gen, 4) {
        Ok(count) => count,
        Err(err) => {
            asset_gen.status = err;
            return;
        }
    };
    if seed_count == 0 {
        asset_gen.status = "Asset gen skipped: prompt corpus is empty.".to_string();
        return;
    }
    asset_gen.auto_seeded = true;
    asset_gen.status = format!(
        "Runtime asset generation queued {} environment jobs.",
        seed_count
    );
}

pub fn runtime_asset_gen_collect_system(mut asset_gen: ResMut<RuntimeAssetGenState>) {
    let mut drained: Vec<RuntimeAssetResult> = Vec::new();
    let shared_results = Arc::clone(&asset_gen.shared_results);
    if let Ok(mut shared) = shared_results.lock() {
        drained.extend(shared.drain(..));
    } else {
        asset_gen.status = "Asset gen internal error: result queue lock poisoned.".to_string();
        return;
    }
    if drained.is_empty() {
        return;
    }

    asset_gen.in_flight_jobs = asset_gen.in_flight_jobs.saturating_sub(drained.len());
    for result in drained {
        asset_gen.status = result.status.clone();
        asset_gen.recent.push_front(result);
        while asset_gen.recent.len() > 8 {
            asset_gen.recent.pop_back();
        }
    }
}

pub fn runtime_asset_gen_dispatch_system(mut asset_gen: ResMut<RuntimeAssetGenState>) {
    while asset_gen.in_flight_jobs < asset_gen.max_parallel_jobs {
        let Some(job) = asset_gen.pending.pop_front() else {
            break;
        };
        asset_gen.in_flight_jobs += 1;
        let output_dir = asset_gen.output_dir.clone();
        let shared_results = Arc::clone(&asset_gen.shared_results);
        std::thread::spawn(move || {
            let image_stem = format!("runtime_{:03}_{}_{}", job.id, job.source_id, job.style.as_slug());
            let prompt_file = output_dir.join(format!("{image_stem}.prompt.txt"));
            let _ = fs::create_dir_all(&output_dir);
            let _ = fs::write(&prompt_file, format!("{}\n", job.prompt));

            let payload = RuntimeAssetResult {
                job_id: job.id,
                source_id: job.source_id,
                source_title: job.source_title,
                prompt_file,
                image_file: None,
                success: false,
                status: "No image generation provider configured.".to_string(),
                scene_tag: job.scene_tag,
                sequence_index: job.sequence_index,
            };
            if let Ok(mut queue) = shared_results.lock() {
                queue.push(payload);
            }
        });
    }
}

pub fn runtime_asset_preview_update_system(
    runtime_asset_gen: Res<RuntimeAssetGenState>,
    mut runtime_asset_preview: ResMut<RuntimeAssetPreviewState>,
    mut images: ResMut<Assets<Image>>,
) {
    let newest_image_path = runtime_asset_gen
        .recent
        .iter()
        .find(|entry| entry.success)
        .and_then(|entry| entry.image_file.clone());
    if newest_image_path == runtime_asset_preview.loaded_path {
        return;
    }
    runtime_asset_preview.loaded_path = newest_image_path.clone();
    runtime_asset_preview.texture_handle = None;
    runtime_asset_preview.last_error = None;
    if let Some(path) = newest_image_path {
        match fs::read(&path) {
            Ok(bytes) => {
                let extension = detect_image_extension_from_bytes(&bytes);
                match decode_preview_image(&bytes, extension) {
                    Ok(image) => {
                        runtime_asset_preview.texture_handle = Some(images.add(image));
                    }
                    Err(err) => {
                        runtime_asset_preview.last_error =
                            Some(format!("Preview decode failed: {}", err));
                    }
                }
            }
            Err(err) => {
                runtime_asset_preview.last_error =
                    Some(format!("Preview load failed ({}): {}", path.display(), err));
            }
        }
    }
}

pub fn update_region_art_system(
    hub_ui: Res<HubUiState>,
    region_transition: Res<RegionLayerTransitionState>,
    mut region_art: ResMut<RegionArtState>,
    mut images: ResMut<Assets<Image>>,
) {
    if hub_ui.screen != HubScreen::RegionView {
        return;
    }
    let region_id = match region_transition.active_payload.as_ref().map(|p| p.region_id) {
        Some(id) => id,
        None => return,
    };
    if region_art.loaded_region_id == Some(region_id) {
        return;
    }
    region_art.loaded_region_id = Some(region_id);
    region_art.texture_handle = None;
    region_art.loaded_path = None;
    region_art.status = String::new();

    let source_fragment = format!("_env_{:03}_", region_id + 1);
    let output_dir = std::path::Path::new("generated/maps/runtime_env");
    let entries = match std::fs::read_dir(output_dir) {
        Ok(e) => e,
        Err(_) => {
            region_art.status = format!("Art directory '{}' not found.", output_dir.display());
            return;
        }
    };

    let mut candidate: Option<std::path::PathBuf> = None;
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let name_str = file_name.to_string_lossy();
        if !name_str.contains(source_fragment.as_str()) {
            continue;
        }
        let ext = entry
            .path()
            .extension()
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "webp" {
            candidate = Some(entry.path());
            break;
        }
    }

    let path = match candidate {
        Some(p) => p,
        None => {
            region_art.status = format!(
                "No art file found for region {} (source id 'env_{:03}').",
                region_id,
                region_id + 1
            );
            return;
        }
    };

    match fs::read(&path) {
        Ok(bytes) => {
            let extension = detect_image_extension_from_bytes(&bytes);
            match decode_preview_image(&bytes, extension) {
                Ok(image) => {
                    region_art.texture_handle = Some(images.add(image));
                    region_art.loaded_path = Some(path);
                }
                Err(err) => {
                    region_art.status = format!("Art decode failed: {}", err);
                }
            }
        }
        Err(err) => {
            region_art.status = format!("Art read failed ({}): {}", path.display(), err);
        }
    }
}

pub fn draw_runtime_menu_background_egui_system(
    mut contexts: EguiContexts,
    hub_ui: Res<HubUiState>,
    runtime_asset_preview: Res<RuntimeAssetPreviewState>,
) {
    let gui_only_screen = matches!(
        hub_ui.screen,
        HubScreen::StartMenu
            | HubScreen::CharacterCreationFaction
            | HubScreen::CharacterCreationBackstory
    );
    if !gui_only_screen {
        return;
    }
    let preview_texture_id = runtime_asset_preview
        .texture_handle
        .as_ref()
        .map(|handle| contexts.add_image(handle.clone()));
    egui::CentralPanel::default()
        .frame(egui::Frame::none().fill(egui::Color32::from_rgb(6, 10, 14)))
        .show(contexts.ctx_mut(), |ui| {
            if let Some(texture_id) = preview_texture_id {
                let rect = ui.max_rect();
                let painter = ui.painter_at(rect);
                painter.image(
                    texture_id,
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
                painter.rect_filled(rect, 0.0, egui::Color32::from_rgba_premultiplied(6, 10, 14, 110));
            } else {
                paint_landscape_backsplash(ui, hub_ui.screen != HubScreen::StartMenu);
            }
        });
}
