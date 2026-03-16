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
