use bevy::prelude::*;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackstoryCinematicPhase {
    Idle,
    Loading,
    Playing,
}

#[derive(Debug, Clone)]
pub struct BackstoryCinematicBeat {
    pub index: usize,
    pub title: String,
    pub subtitle: String,
    pub source_id: String,
    pub prompt: String,
    pub image_file: Option<PathBuf>,
    pub texture_handle: Option<Handle<Image>>,
}

#[derive(Debug, Clone)]
pub struct BackstoryNarrativeResult {
    pub campaign_seed: u64,
    pub summary: String,
    pub beat_subtitles: Vec<String>,
    pub status: String,
    pub success: bool,
}

#[derive(Resource, Clone)]
pub struct BackstoryNarrativeGenState {
    pub model: String,
    pub in_flight: bool,
    pub requested_seed: Option<u64>,
    pub shared_result: Arc<Mutex<Option<BackstoryNarrativeResult>>>,
}

impl Default for BackstoryNarrativeGenState {
    fn default() -> Self {
        Self {
            model: String::new(),
            in_flight: false,
            requested_seed: None,
            shared_result: Arc::new(Mutex::new(None)),
        }
    }
}

#[derive(Resource, Clone)]
pub struct BackstoryCinematicState {
    pub initialized_for_campaign_seed: Option<u64>,
    pub phase: BackstoryCinematicPhase,
    pub narrative_summary: String,
    pub beats: Vec<BackstoryCinematicBeat>,
    pub beats_enqueued: bool,
    pub portrait_image_file: Option<PathBuf>,
    pub current_beat: usize,
    pub beat_elapsed_seconds: f32,
    pub beat_duration_seconds: f32,
    pub status: String,
    pub seen_job_ids: HashSet<u64>,
}

impl Default for BackstoryCinematicState {
    fn default() -> Self {
        Self {
            initialized_for_campaign_seed: None,
            phase: BackstoryCinematicPhase::Idle,
            narrative_summary: String::new(),
            beats: Vec::new(),
            beats_enqueued: false,
            portrait_image_file: None,
            current_beat: 0,
            beat_elapsed_seconds: 0.0,
            beat_duration_seconds: 4.5,
            status: "Backstory cinematic idle.".to_string(),
            seen_job_ids: HashSet::new(),
        }
    }
}
