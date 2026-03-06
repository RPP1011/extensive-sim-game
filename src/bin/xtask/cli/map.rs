use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

// ---------------------------------------------------------------------------
// Map subcommand tree
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
pub struct MapCommand {
    #[command(subcommand)]
    pub command: MapSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum MapSubcommand {
    Gemini(MapGeminiArgs),
    Voronoi(MapVoronoiArgs),
    EnvArt(MapEnvArtCommand),
}

#[derive(Debug, Parser)]
pub struct MapEnvArtCommand {
    #[command(subcommand)]
    pub command: MapEnvArtSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum MapEnvArtSubcommand {
    BuildIndex(MapEnvArtBuildIndexArgs),
    Query(MapEnvArtQueryArgs),
    Generate(MapEnvArtGenerateArgs),
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum EnvArtStyle {
    Concept,
    Matte,
    Illustration,
}

impl EnvArtStyle {
    pub fn as_prompt_suffix(self) -> &'static str {
        match self {
            EnvArtStyle::Concept => {
                "environment concept art, production paintover quality, broad-to-fine brushwork"
            }
            EnvArtStyle::Matte => {
                "cinematic matte painting, atmospheric perspective, ultra-detailed landscapes"
            }
            EnvArtStyle::Illustration => {
                "high-detail fantasy illustration, rich material rendering, scene-first composition"
            }
        }
    }

    pub fn as_slug(self) -> &'static str {
        match self {
            EnvArtStyle::Concept => "concept",
            EnvArtStyle::Matte => "matte",
            EnvArtStyle::Illustration => "illustration",
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Semantic term query over environment prompts using hnsw_rs")]
pub struct MapEnvArtQueryArgs {
    #[arg(long = "corpus", default_value = "scripts/ai/fantasy_env_prompt_corpus.json")]
    pub corpus: PathBuf,

    #[arg(long = "index", default_value = "generated/hnsw/env_art_index.json")]
    pub index: PathBuf,

    #[arg(long = "refresh-index", default_value_t = false)]
    pub refresh_index: bool,

    #[arg(long)]
    pub query: String,

    #[arg(long = "top-k", default_value_t = 8)]
    pub top_k: usize,
}

#[derive(Debug, Parser)]
#[command(about = "Query prompts with hnsw_rs and generate Gemini environment art batch")]
pub struct MapEnvArtGenerateArgs {
    #[arg(long = "corpus", default_value = "scripts/ai/fantasy_env_prompt_corpus.json")]
    pub corpus: PathBuf,

    #[arg(long = "index", default_value = "generated/hnsw/env_art_index.json")]
    pub index: PathBuf,

    #[arg(long = "refresh-index", default_value_t = false)]
    pub refresh_index: bool,

    #[arg(long)]
    pub query: String,

    #[arg(long = "top-k", default_value_t = 12)]
    pub top_k: usize,

    #[arg(long = "count", default_value_t = 8)]
    pub count: usize,

    #[arg(long = "model", default_value = "gemini-3-pro-image-preview")]
    pub model: String,

    #[arg(long = "style", value_enum, default_value_t = EnvArtStyle::Concept)]
    pub style: EnvArtStyle,

    #[arg(long = "out-dir", default_value = "generated/maps/fantasy_env")]
    pub out_dir: PathBuf,
}

#[derive(Debug, Parser)]
#[command(about = "Build and persist environment vector index (Gemini embeddings + corpus metadata)")]
pub struct MapEnvArtBuildIndexArgs {
    #[arg(long = "corpus", default_value = "scripts/ai/fantasy_env_prompt_corpus.json")]
    pub corpus: PathBuf,

    #[arg(long = "index", default_value = "generated/hnsw/env_art_index.json")]
    pub index: PathBuf,
}

// ---------------------------------------------------------------------------
// Capture subcommand tree
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
pub struct CaptureCommand {
    #[command(subcommand)]
    pub command: CaptureSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum CaptureSubcommand {
    Windows(CaptureWindowsArgs),
    Dedupe(CaptureDedupeArgs),
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CaptureMode {
    Single,
    Sequence,
    SafeSequence,
    HubStages,
}

impl CaptureMode {
    pub fn as_ps_value(self) -> &'static str {
        match self {
            CaptureMode::Single => "single",
            CaptureMode::Sequence => "sequence",
            CaptureMode::SafeSequence => "safe-sequence",
            CaptureMode::HubStages => "hub-stages",
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Run Windows-native screenshot capture via scripts/capture_windows.ps1")]
pub struct CaptureWindowsArgs {
    #[arg(long, value_enum, default_value_t = CaptureMode::Single)]
    pub mode: CaptureMode,

    #[arg(long = "out-dir", default_value = "generated/screenshots/windows")]
    pub out_dir: PathBuf,

    #[arg(long, default_value_t = 30)]
    pub steps: i32,

    #[arg(long, default_value_t = 1)]
    pub every: i32,

    #[arg(long = "warmup-frames", default_value_t = 3)]
    pub warmup_frames: i32,

    #[arg(long, default_value_t = false)]
    pub hub: bool,

    #[arg(long, default_value_t = false)]
    pub persist: bool,
}

#[derive(Debug, Parser)]
#[command(about = "Run screenshot dedupe via scripts/dedupe_capture.ps1")]
pub struct CaptureDedupeArgs {
    #[arg(long = "out-dir")]
    pub out_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// Map gemini / voronoi args
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(about = "Generate a map image via Gemini API")]
pub struct MapGeminiArgs {
    #[arg(long)]
    pub prompt: Option<String>,

    #[arg(long = "prompt-file")]
    pub prompt_file: Option<PathBuf>,

    #[arg(long, default_value = "gemini-3-pro-image-preview")]
    pub model: String,

    #[arg(long, default_value = "generated/maps/map.png")]
    pub out: PathBuf,

    #[arg(long = "save-text", default_value_t = false)]
    pub save_text: bool,
}

#[derive(Debug, Parser)]
#[command(about = "Generate weighted Voronoi map prompt/spec from overworld save")]
pub struct MapVoronoiArgs {
    #[arg(long, default_value = "generated/saves/campaign_autosave.json")]
    pub save: PathBuf,

    #[arg(
        long = "out-prompt",
        default_value = "generated/maps/overworld_voronoi_prompt.txt"
    )]
    pub out_prompt: PathBuf,

    #[arg(
        long = "out-spec",
        default_value = "generated/maps/overworld_voronoi_spec.json"
    )]
    pub out_spec: PathBuf,

    #[arg(long = "grid-w", default_value_t = 220)]
    pub grid_w: usize,

    #[arg(long = "grid-h", default_value_t = 140)]
    pub grid_h: usize,

    #[arg(long = "strength-scale", default_value_t = 0.22)]
    pub strength_scale: f64,

    #[arg(long = "organic-jitter", default_value_t = 0.18)]
    pub organic_jitter: f64,

    #[arg(
        long = "gemini-out",
        default_value = "generated/maps/overworld_voronoi_map.png"
    )]
    pub gemini_out: PathBuf,

    #[arg(long = "gemini-model", default_value = "gemini-3-pro-image-preview")]
    pub gemini_model: String,

    #[arg(long = "run-gemini", default_value_t = false)]
    pub run_gemini: bool,
}
