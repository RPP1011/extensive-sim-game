use bevy::prelude::*;
use serde::Deserialize;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeAssetStyle {
    Concept,
    LineArt,
}

impl RuntimeAssetStyle {
    pub fn as_prompt_suffix(self) -> &'static str {
        match self {
            RuntimeAssetStyle::Concept => {
                "environment concept art, production paintover quality, broad-to-fine brushwork"
            }
            RuntimeAssetStyle::LineArt => {
                "fantasy line art illustration, clean ink contours, readable silhouettes, minimal shading, parchment-toned background"
            }
        }
    }

    pub fn as_slug(self) -> &'static str {
        match self {
            RuntimeAssetStyle::Concept => "concept",
            RuntimeAssetStyle::LineArt => "lineart",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeAssetJobKind {
    EnvironmentScene,
    CharacterPortrait,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeEnvPromptRow {
    pub id: String,
    pub title: String,
    pub prompt: String,
}

#[derive(Debug, Clone)]
pub struct RuntimeAssetJob {
    pub id: u64,
    pub source_id: String,
    pub source_title: String,
    pub prompt: String,
    pub style: RuntimeAssetStyle,
    pub kind: RuntimeAssetJobKind,
    pub reference_image_path: Option<PathBuf>,
    pub scene_tag: Option<String>,
    pub sequence_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RuntimeAssetResult {
    pub job_id: u64,
    pub source_id: String,
    pub source_title: String,
    pub prompt_file: PathBuf,
    pub image_file: Option<PathBuf>,
    pub success: bool,
    pub status: String,
    pub scene_tag: Option<String>,
    pub sequence_index: Option<usize>,
}

#[derive(Resource, Clone)]
pub struct RuntimeAssetGenState {
    pub model: String,
    pub output_dir: PathBuf,
    pub prompt_corpus_path: PathBuf,
    pub pending: VecDeque<RuntimeAssetJob>,
    pub recent: VecDeque<RuntimeAssetResult>,
    pub next_job_id: u64,
    pub in_flight_jobs: usize,
    pub max_parallel_jobs: usize,
    pub auto_seeded: bool,
    pub status: String,
    pub shared_results: Arc<Mutex<Vec<RuntimeAssetResult>>>,
}

impl Default for RuntimeAssetGenState {
    fn default() -> Self {
        Self {
            model: String::new(),
            output_dir: PathBuf::from("generated/maps/runtime_env"),
            prompt_corpus_path: PathBuf::from("scripts/ai/fantasy_env_prompt_corpus.json"),
            pending: VecDeque::new(),
            recent: VecDeque::new(),
            next_job_id: 1,
            in_flight_jobs: 0,
            max_parallel_jobs: 4,
            auto_seeded: false,
            status: "Runtime asset generation idle.".to_string(),
            shared_results: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[derive(Resource, Default, Clone)]
pub struct RuntimeAssetPreviewState {
    pub loaded_path: Option<PathBuf>,
    pub texture_handle: Option<Handle<Image>>,
    pub last_error: Option<String>,
}

/// Holds the environment art image loaded for the currently active region in RegionView.
#[derive(Resource, Default, Clone)]
pub struct RegionArtState {
    pub loaded_region_id: Option<usize>,
    pub loaded_path: Option<PathBuf>,
    pub texture_handle: Option<Handle<Image>>,
    pub status: String,
}
