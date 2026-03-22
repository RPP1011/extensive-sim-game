use bevy::prelude::*;
use serde::Serialize;

use crate::game_core::{self, HubScreen, MissionResult};

#[derive(Resource)]
pub struct ScreenshotCaptureConfig {
    pub mode: ScreenshotMode,
    pub warmup_frames: u32,
    pub max_captures: Option<u32>,
    pub max_attempts: u32,
}

#[derive(Resource, Default)]
pub struct ScreenshotCaptureState {
    pub frames_seen: u32,
    pub captures_written: u32,
    pub capture_attempts: u32,
    pub stage_ready_at_frame: Option<u32>,
    pub exit_countdown_frames: u32,
    pub pending_capture_path: Option<String>,
}

pub enum ScreenshotMode {
    Single { dir: String },
    Sequence { dir: String, every: u32 },
    HubStages { dir: String },
}

pub const HUB_STAGE_CAPTURE_SEQUENCE: [HubScreen; 7] = [
    HubScreen::StartMenu,
    HubScreen::CharacterCreationFaction,
    HubScreen::OverworldMap,
    HubScreen::RegionView,
    HubScreen::LocalEagleEyeIntro,
    HubScreen::MissionExecution,
    HubScreen::ReplayViewer,
];

pub fn hub_screen_capture_name(screen: HubScreen) -> &'static str {
    match screen {
        HubScreen::StartMenu => "StartMenu",
        HubScreen::CharacterCreationFaction => "CharacterCreationFaction",
        HubScreen::CharacterCreationBackstory => "CharacterCreationBackstory",
        HubScreen::BackstoryCinematic => "BackstoryCinematic",
        HubScreen::GuildManagement => "GuildManagement",
        HubScreen::Overworld => "Overworld",
        HubScreen::OverworldMap => "OverworldMap",
        HubScreen::RegionView => "RegionView",
        HubScreen::LocalEagleEyeIntro => "LocalEagleEyeIntro",
        HubScreen::MissionExecution => "MissionExecution",
        HubScreen::ReplayViewer => "ReplayViewer",
        HubScreen::CampaignTraceViewer => "CampaignTraceViewer",
    }
}

#[derive(Serialize)]
pub struct UiFrameState {
    pub capture_index: u32,
    pub render_frame: u32,
    pub global_turn: u32,
    pub active_mission_id: Option<u32>,
    pub active_mission_name: Option<String>,
    pub missions: Vec<UiMissionState>,
}

#[derive(Serialize)]
pub struct UiMissionState {
    pub id: u32,
    pub mission_name: String,
    pub active: bool,
    pub mission_active: bool,
    pub result: MissionResult,
    pub turns_remaining: u32,
    pub reactor_integrity: f32,
    pub sabotage_progress: f32,
    pub sabotage_goal: f32,
    pub alert_level: f32,
    pub room_index: usize,
    pub room_name: String,
    pub tactical_mode: Option<game_core::TacticalMode>,
    pub command_cooldown_turns: Option<u32>,
    pub assigned_hero_id: Option<u32>,
}
