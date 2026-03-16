pub mod scenario;
pub mod map;

pub use scenario::*;
pub use map::*;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(about = "Project development tasks")]
pub struct Args {
    #[command(subcommand)]
    pub command: TaskCommand,
}

#[derive(Debug, Subcommand)]
pub enum TaskCommand {
    Map(MapCommand),
    Capture(CaptureCommand),
    Ralph(RalphCommand),
    Scenario(ScenarioCommand),
    /// Build with burn-gpu and run IMPALA V6 training (auto-detects libtorch)
    TrainV6(TrainV6Args),
    Roomgen(RoomgenCommand),
}

#[derive(Debug, Parser)]
#[command(about = "IMPALA V6 training (auto-configures libtorch env, builds burn-gpu, runs training)")]
pub struct TrainV6Args {
    /// Path(s) to scenario .toml file(s) or directory(ies)
    #[arg(default_value = "dataset/scenarios/hvh")]
    pub path: Vec<PathBuf>,
    /// Output directory for checkpoints and logs
    #[arg(long, default_value = "generated/impala_v6")]
    pub output_dir: PathBuf,
    /// Resume from Burn checkpoint
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
    /// Path to embedding registry JSON
    #[arg(long)]
    pub embedding_registry: Option<PathBuf>,
    /// Number of training iterations
    #[arg(long, default_value_t = 100)]
    pub iters: usize,
    /// Episodes per scenario per iteration
    #[arg(long, default_value_t = 2)]
    pub episodes: usize,
    /// Threads for episode generation
    #[arg(long, default_value_t = 32)]
    pub threads: usize,
    /// Sims per thread during episode generation
    #[arg(long, default_value_t = 64)]
    pub sims_per_thread: usize,
    /// Training batch size
    #[arg(long, default_value_t = 512)]
    pub batch_size: usize,
    /// Training steps per iteration
    #[arg(long, default_value_t = 50)]
    pub train_steps: usize,
    /// Learning rate
    #[arg(long, default_value_t = 5e-4)]
    pub lr: f64,
    /// Exploration temperature
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f32,
    /// Step recording interval (every N ticks)
    #[arg(long, default_value_t = 3)]
    pub step_interval: u64,
    /// Entropy coefficient
    #[arg(long, default_value_t = 0.01)]
    pub entropy_coef: f32,
    /// Value loss coefficient
    #[arg(long, default_value_t = 0.5)]
    pub value_coef: f32,
    /// Enable Grokfast EMA gradient filter
    #[arg(long)]
    pub grokfast: bool,
    /// Self-play: GPU inference for enemy units too
    #[arg(long)]
    pub self_play: bool,
    /// Skip build step (assume binary is already built with burn-gpu)
    #[arg(long)]
    pub no_build: bool,
}

// ---------------------------------------------------------------------------
// Ralph subcommand tree
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(about = "Ralph agent task automation")]
pub struct RalphCommand {
    #[command(subcommand)]
    pub command: RalphSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum RalphSubcommand {
    Status(RalphStatusArgs),
}

#[derive(Debug, Parser)]
#[command(about = "Check and optionally update story status from the PRD quality gates")]
pub struct RalphStatusArgs {
    /// Path to the PRD JSON file
    #[arg(long, default_value = ".agents/tasks/prd-campaign-parties.json")]
    pub prd: PathBuf,

    /// If set, mark any in-progress story whose quality gates pass as done and
    /// write the updated JSON back to the PRD file.
    #[arg(long, default_value_t = false)]
    pub update: bool,
}

// ---------------------------------------------------------------------------
// Roomgen subcommand tree
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(about = "ML terrain generation data pipeline")]
pub struct RoomgenCommand {
    #[command(subcommand)]
    pub sub: RoomgenSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum RoomgenSubcommand {
    /// Batch-generate rooms as JSON with multi-channel grids and tactical metrics
    Export(RoomgenExportArgs),
    /// Render top-down PNG images for VLM captioning
    Render(RoomgenRenderArgs),
    /// Run HvH combat simulations on generated rooms to measure win rates
    Simulate(RoomgenSimulateArgs),
    /// Generate mission floorplans (multi-room connected layouts)
    Floorplan(RoomgenFloorplanArgs),
    /// Run solo retrieval missions: 1 hero vs N enemies, reach objective
    Retrieve(RoomgenRetrieveArgs),
}

#[derive(Debug, Parser)]
#[command(about = "Run solo retrieval missions on generated rooms")]
pub struct RoomgenRetrieveArgs {
    /// Room JSONL file
    pub rooms: PathBuf,
    /// Hero registry directory
    pub heroes: PathBuf,
    /// Number of enemies per mission
    #[arg(long, default_value_t = 20)]
    pub enemies: usize,
    /// Maximum missions to run
    #[arg(long, default_value_t = 500)]
    pub max_matches: usize,
    /// Number of parallel threads (0 = all cores)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
    /// Output JSONL
    #[arg(long, default_value = "generated/retrieval_results.jsonl")]
    pub output: PathBuf,
}

#[derive(Debug, Parser)]
#[command(about = "Generate and render mission floorplans")]
pub struct RoomgenFloorplanArgs {
    /// Number of floorplans to generate
    #[arg(long, default_value_t = 10)]
    pub count: usize,
    /// Number of rooms per floorplan
    #[arg(long, default_value_t = 5)]
    pub rooms: usize,
    /// Grid width
    #[arg(long, default_value_t = 80)]
    pub width: usize,
    /// Grid height
    #[arg(long, default_value_t = 60)]
    pub height: usize,
    /// Base RNG seed
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    /// Output directory for PNG images
    #[arg(long, default_value = "generated/floorplans")]
    pub output: PathBuf,
    /// Pixels per cell
    #[arg(long, default_value_t = 8)]
    pub pixels_per_cell: u32,
}

#[derive(Debug, Parser)]
#[command(about = "Batch-generate rooms as JSONL with grids + metrics")]
pub struct RoomgenExportArgs {
    /// Output JSONL file
    #[arg(long, default_value = "generated/rooms.jsonl")]
    pub output: PathBuf,
    /// Number of rooms to generate per room type
    #[arg(long, default_value_t = 5000)]
    pub count_per_type: usize,
    /// Base RNG seed
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    /// Use varied dimensions (±30% perturbation)
    #[arg(long, default_value_t = true)]
    pub varied: bool,
    /// Number of parallel threads (0 = all cores)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
}

#[derive(Debug, Parser)]
#[command(about = "Render top-down PNG images of rooms for VLM captioning")]
pub struct RoomgenRenderArgs {
    /// Input JSONL file (from roomgen export)
    #[arg(long, default_value = "generated/rooms.jsonl")]
    pub input: PathBuf,
    /// Output directory for PNG images
    #[arg(long, default_value = "generated/room_images")]
    pub output_dir: PathBuf,
    /// Pixels per grid cell
    #[arg(long, default_value_t = 4)]
    pub pixels_per_cell: u32,
}

#[derive(Debug, Parser)]
#[command(about = "Run HvH combat simulations on generated rooms")]
pub struct RoomgenSimulateArgs {
    /// Room JSONL file (from roomgen export)
    pub rooms: PathBuf,
    /// Hero registry: directory of .toml hero templates (e.g. assets/hero_templates)
    pub heroes: PathBuf,
    /// Maximum number of matches to run
    #[arg(long, default_value_t = 1000)]
    pub max_matches: usize,
    /// Number of parallel threads (0 = all cores)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
    /// Output JSONL results file
    #[arg(long, default_value = "generated/room_sim_results.jsonl")]
    pub output: PathBuf,
}

// ---------------------------------------------------------------------------
// Ralph subcommand tree
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(about = "Ralph agent task automation")]
pub struct RalphCommand {
    #[command(subcommand)]
    pub command: RalphSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum RalphSubcommand {
    Status(RalphStatusArgs),
}

#[derive(Debug, Parser)]
#[command(about = "Check and optionally update story status from the PRD quality gates")]
pub struct RalphStatusArgs {
    /// Path to the PRD JSON file
    #[arg(long, default_value = ".agents/tasks/prd-campaign-parties.json")]
    pub prd: PathBuf,

    /// If set, mark any in-progress story whose quality gates pass as done and
    /// write the updated JSON back to the PRD file.
    #[arg(long, default_value_t = false)]
    pub update: bool,
}

