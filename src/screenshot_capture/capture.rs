use bevy::app::AppExit;
use bevy::prelude::*;
use std::fs;
use std::io;
use std::path::Path;

use crate::character_select::build_faction_selection_choices;
use crate::game_core::{
    self, ActiveMission, AssignedHero, HubScreen, HubUiState, MissionData, MissionProgress,
    MissionTactics,
};
use crate::game_loop::SimulationSteps;
use crate::local_intro::{local_intro_anchor_for_region, LocalEagleEyeIntroState, LocalIntroPhase};
use crate::region_nav::{build_region_transition_payload, RegionLayerTransitionState};

use super::types::*;

pub fn configure_hub_stage_capture_fixture(world: &mut World, target_stage: HubScreen) {
    {
        let mut hub_ui = world.resource_mut::<HubUiState>();
        hub_ui.screen = target_stage;
    }

    {
        let mut overworld = world.resource_mut::<game_core::OverworldMap>();
        if !overworld.regions.is_empty() {
            let region_id = 2_usize.min(overworld.regions.len().saturating_sub(1));
            overworld.selected_region = region_id;
            overworld.current_region = region_id;
        }
    }

    let selected_faction = {
        let overworld = world.resource::<game_core::OverworldMap>();
        build_faction_selection_choices(&overworld)
            .into_iter()
            .next()
            .map(|choice| (choice.id, choice.index))
    };
    if let Some((faction_id, faction_index)) = selected_faction {
        let mut creation = world.resource_mut::<game_core::CharacterCreationState>();
        creation.selected_faction_id = Some(faction_id);
        creation.selected_faction_index = Some(faction_index);
        creation.selected_backstory_id = Some("scout-pathfinder".to_string());
        creation.is_confirmed = true;
    }

    if matches!(
        target_stage,
        HubScreen::RegionView | HubScreen::LocalEagleEyeIntro
    ) {
        let payload = {
            let overworld = world.resource::<game_core::OverworldMap>();
            let creation = world.resource::<game_core::CharacterCreationState>();
            build_region_transition_payload(&overworld, &creation).ok()
        };
        if let Some(payload) = payload {
            let mut transition = world.resource_mut::<RegionLayerTransitionState>();
            transition.active_payload = Some(payload.clone());
            transition.pending_payload = None;
            transition.pending_frames = 0;
            transition.interaction_locked = false;
            transition.status = format!(
                "Region fixture prepared: {}.",
                hub_screen_capture_name(target_stage)
            );
        }
    } else {
        let mut transition = world.resource_mut::<RegionLayerTransitionState>();
        transition.active_payload = None;
        transition.pending_payload = None;
        transition.pending_frames = 0;
        transition.interaction_locked = false;
        transition.status = "No active region context.".to_string();
    }

    if target_stage == HubScreen::LocalEagleEyeIntro {
        let region_payload = world
            .resource::<RegionLayerTransitionState>()
            .active_payload
            .clone();
        let mut local_intro = world.resource_mut::<LocalEagleEyeIntroState>();
        if let Some(payload) = region_payload {
            local_intro.source_region_id = Some(payload.region_id);
            local_intro.anchor = local_intro_anchor_for_region(payload.region_id);
            local_intro.phase = if local_intro.anchor.is_some() {
                LocalIntroPhase::HiddenInside
            } else {
                LocalIntroPhase::Idle
            };
            local_intro.phase_frames = 0;
            local_intro.intro_completed = false;
            local_intro.input_handoff_ready = false;
            local_intro.status = if local_intro.anchor.is_some() {
                format!(
                    "Local intro fixture ready for {}.",
                    hub_screen_capture_name(target_stage)
                )
            } else {
                "Local intro fixture missing anchor.".to_string()
            };
        } else {
            *local_intro = LocalEagleEyeIntroState::default();
        }
    } else {
        *world.resource_mut::<LocalEagleEyeIntroState>() = LocalEagleEyeIntroState::default();
    }
}

pub fn screenshot_capture_system(world: &mut World) {
    {
        let mut state = world.resource_mut::<ScreenshotCaptureState>();
        if state.exit_countdown_frames > 0 {
            state.exit_countdown_frames -= 1;
            if state.exit_countdown_frames == 0 {
                world.send_event(AppExit);
            }
            return;
        }
    }

    let (mode, warmup_frames, max_captures, max_attempts) = {
        let config = world.resource::<ScreenshotCaptureConfig>();
        (
            match &config.mode {
                ScreenshotMode::Single { dir } => ScreenshotMode::Single { dir: dir.clone() },
                ScreenshotMode::Sequence { dir, every } => ScreenshotMode::Sequence {
                    dir: dir.clone(),
                    every: *every,
                },
                ScreenshotMode::HubStages { dir } => ScreenshotMode::HubStages { dir: dir.clone() },
            },
            config.warmup_frames,
            config.max_captures,
            config.max_attempts,
        )
    };

    let remaining_steps = world.resource::<SimulationSteps>().0;

    let (frames_seen, captures_written) = {
        let mut state = world.resource_mut::<ScreenshotCaptureState>();
        state.frames_seen += 1;
        (state.frames_seen, state.captures_written)
    };

    if frames_seen < warmup_frames {
        return;
    }
    let is_sequence_mode = matches!(mode, ScreenshotMode::Sequence { .. });
    if is_sequence_mode && remaining_steps == Some(0) {
        world
            .resource_mut::<ScreenshotCaptureState>()
            .exit_countdown_frames = 6;
        return;
    }
    if max_captures.is_some_and(|limit| captures_written >= limit) {
        world
            .resource_mut::<ScreenshotCaptureState>()
            .exit_countdown_frames = 6;
        return;
    }

    let window = match world
        .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
        .get_single(world)
    {
        Ok(entity) => entity,
        Err(_) => return,
    };

    match &mode {
        ScreenshotMode::Single { dir } => {
            if captures_written == 0 {
                world
                    .resource_mut::<ScreenshotCaptureState>()
                    .capture_attempts += 1;
                if let Err(err) = fs::create_dir_all(dir) {
                    eprintln!("Failed to create screenshot directory '{}': {}", dir, err);
                    return;
                }
                let image_path = capture_image_path(dir, captures_written);
                let wrote = {
                    let mut screenshot_manager =
                        world.resource_mut::<bevy::render::view::screenshot::ScreenshotManager>();
                    screenshot_manager
                        .save_screenshot_to_disk(window, &image_path)
                        .map(|_| ())
                };
                if let Err(err) = wrote {
                    eprintln!("Failed to save screenshot '{}': {}", image_path, err);
                } else {
                    write_ui_frame_state(world, dir, captures_written, frames_seen);
                    let mut state = world.resource_mut::<ScreenshotCaptureState>();
                    state.captures_written = 1;
                }
            }
            let state = world.resource::<ScreenshotCaptureState>();
            if state.captures_written > 0 || state.capture_attempts >= max_attempts {
                if state.captures_written == 0 {
                    eprintln!(
                        "Single screenshot capture reached attempt limit ({}), exiting.",
                        max_attempts
                    );
                }
                world
                    .resource_mut::<ScreenshotCaptureState>()
                    .exit_countdown_frames = 6;
            }
        }
        ScreenshotMode::Sequence { dir, every } => {
            if (frames_seen - warmup_frames) % *every == 0 {
                if let Err(err) = fs::create_dir_all(dir) {
                    eprintln!("Failed to create screenshot directory '{}': {}", dir, err);
                    return;
                }
                let image_path = capture_image_path(dir, captures_written);
                let wrote = {
                    let mut screenshot_manager =
                        world.resource_mut::<bevy::render::view::screenshot::ScreenshotManager>();
                    screenshot_manager
                        .save_screenshot_to_disk(window, &image_path)
                        .map(|_| ())
                };
                if let Err(err) = wrote {
                    eprintln!("Failed to save screenshot '{}': {}", image_path, err);
                } else {
                    write_ui_frame_state(world, dir, captures_written, frames_seen);
                    world
                        .resource_mut::<ScreenshotCaptureState>()
                        .captures_written += 1;
                }
            }
        }
        ScreenshotMode::HubStages { dir } => {
            hub_stages_capture(world, dir, warmup_frames, max_attempts, frames_seen, captures_written, window);
        }
    }
}

fn hub_stages_capture(
    world: &mut World,
    dir: &str,
    warmup_frames: u32,
    max_attempts: u32,
    frames_seen: u32,
    captures_written: u32,
    window: Entity,
) {
    let stage_idx = captures_written as usize;
    if stage_idx >= HUB_STAGE_CAPTURE_SEQUENCE.len() {
        world
            .resource_mut::<ScreenshotCaptureState>()
            .exit_countdown_frames = 6;
        return;
    }
    let target_stage = HUB_STAGE_CAPTURE_SEQUENCE[stage_idx];
    let should_prepare_stage = {
        let hub_ui = world.resource::<HubUiState>();
        hub_ui.screen != target_stage
    };
    if should_prepare_stage {
        configure_hub_stage_capture_fixture(world, target_stage);
        let ready_at = frames_seen.saturating_add(warmup_frames.max(1));
        let mut state = world.resource_mut::<ScreenshotCaptureState>();
        state.stage_ready_at_frame = Some(ready_at);
        state.capture_attempts = 0;
        state.pending_capture_path = None;
        return;
    }

    if let Err(err) = fs::create_dir_all(dir) {
        eprintln!("Failed to create screenshot directory '{}': {}", dir, err);
        return;
    }

    let (pending_capture_path, capture_attempts) = {
        let state = world.resource::<ScreenshotCaptureState>();
        (
            state.pending_capture_path.clone(),
            state.capture_attempts,
        )
    };

    let stage_ready = world
        .resource::<ScreenshotCaptureState>()
        .stage_ready_at_frame
        .map_or(false, |ready_at| frames_seen >= ready_at);
    if !stage_ready {
        return;
    }

    if let Some(image_path) = pending_capture_path {
        match fs::metadata(&image_path) {
            Ok(meta) if meta.len() > 0 => {
                write_ui_frame_state(world, dir, captures_written, frames_seen);
                let mut state = world.resource_mut::<ScreenshotCaptureState>();
                state.captures_written += 1;
                state.stage_ready_at_frame = None;
                state.capture_attempts = 0;
                state.pending_capture_path = None;
                if state.captures_written as usize >= HUB_STAGE_CAPTURE_SEQUENCE.len() {
                    state.exit_countdown_frames = 6;
                }
            }
            _ => {
                let mut state = world.resource_mut::<ScreenshotCaptureState>();
                state.capture_attempts += 1;
                if state.capture_attempts > max_attempts.max(1) {
                    let stage_name = hub_screen_capture_name(target_stage);
                    eprintln!(
                        "Hub stage capture timed out for {}, skipping to next stage.",
                        stage_name
                    );
                    if let Err(err) = fs::remove_file(&image_path) {
                        if err.kind() != io::ErrorKind::NotFound {
                            eprintln!(
                                "Failed to remove timed-out screenshot '{}': {}",
                                image_path, err
                            );
                        }
                    }
                    state.captures_written += 1;
                    state.stage_ready_at_frame = None;
                    state.capture_attempts = 0;
                    state.pending_capture_path = None;
                    if state.captures_written as usize
                        >= HUB_STAGE_CAPTURE_SEQUENCE.len()
                    {
                        state.exit_countdown_frames = 6;
                    }
                }
            }
        }
        return;
    }

    let image_path = capture_image_path(dir, captures_written);
    if let Err(err) = fs::remove_file(&image_path) {
        if err.kind() != io::ErrorKind::NotFound {
            eprintln!(
                "Failed to remove previous screenshot '{}': {}",
                image_path, err
            );
            return;
        }
    }

    if capture_attempts >= max_attempts.max(1) {
        let stage_name = hub_screen_capture_name(target_stage);
        eprintln!(
            "Hub stage capture reached attempt limit for {}, skipping stage.",
            stage_name
        );
        let mut state = world.resource_mut::<ScreenshotCaptureState>();
        state.captures_written += 1;
        state.stage_ready_at_frame = None;
        state.capture_attempts = 0;
        if state.captures_written as usize >= HUB_STAGE_CAPTURE_SEQUENCE.len() {
            state.exit_countdown_frames = 6;
        }
        return;
    }

    let wrote = {
        let mut screenshot_manager =
            world.resource_mut::<bevy::render::view::screenshot::ScreenshotManager>();
        screenshot_manager
            .save_screenshot_to_disk(window, &image_path)
            .map(|_| ())
    };
    if let Err(err) = wrote {
        eprintln!("Failed to save screenshot '{}': {}", image_path, err);
        world
            .resource_mut::<ScreenshotCaptureState>()
            .capture_attempts += 1;
    } else {
        let mut state = world.resource_mut::<ScreenshotCaptureState>();
        state.pending_capture_path = Some(image_path);
    }
}

pub fn capture_image_path(dir: &str, capture_index: u32) -> String {
    Path::new(dir)
        .join(format!("frame_{:05}.png", capture_index))
        .to_string_lossy()
        .to_string()
}

pub fn write_ui_frame_state(world: &mut World, dir: &str, capture_index: u32, render_frame: u32) {
    let global_turn = world.resource::<game_core::RunState>().global_turn;
    let room_names: Vec<String> = world
        .resource::<game_core::MissionMap>()
        .rooms
        .iter()
        .map(|room| room.room_name.clone())
        .collect();
    let mut query = world.query::<(
        &MissionData,
        &MissionProgress,
        Option<&MissionTactics>,
        Option<&AssignedHero>,
        Option<&ActiveMission>,
    )>();
    let mut missions = Vec::new();
    let mut active_mission_id = None;
    let mut active_mission_name = None;

    for (data, progress, tactics, assigned, active) in query.iter(world) {
        let room_name = room_names
            .get(progress.room_index)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        let is_active = active.is_some();
        if is_active {
            active_mission_id = Some(data.id);
            active_mission_name = Some(data.mission_name.clone());
        }
        missions.push(UiMissionState {
            id: data.id,
            mission_name: data.mission_name.clone(),
            active: is_active,
            mission_active: progress.mission_active,
            result: progress.result,
            turns_remaining: progress.turns_remaining,
            reactor_integrity: progress.reactor_integrity,
            sabotage_progress: progress.sabotage_progress,
            sabotage_goal: progress.sabotage_goal,
            alert_level: progress.alert_level,
            room_index: progress.room_index,
            room_name,
            tactical_mode: tactics.map(|t| t.tactical_mode),
            command_cooldown_turns: tactics.map(|t| t.command_cooldown_turns),
            assigned_hero_id: assigned.and_then(|a| a.hero_id),
        });
    }

    missions.sort_by_key(|m| m.id);

    let state = UiFrameState {
        capture_index,
        render_frame,
        global_turn,
        active_mission_id,
        active_mission_name,
        missions,
    };

    let state_path = Path::new(dir).join(format!("frame_{:05}.json", capture_index));
    let serialized = match serde_json::to_string_pretty(&state) {
        Ok(json) => json,
        Err(err) => {
            eprintln!("Failed to serialize capture frame state: {}", err);
            return;
        }
    };
    if let Err(err) = fs::write(&state_path, serialized) {
        eprintln!(
            "Failed to write frame state '{}': {}",
            state_path.display(),
            err
        );
    }
}
