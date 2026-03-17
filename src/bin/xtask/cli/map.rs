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
    Voronoi(MapVoronoiArgs),
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
}
