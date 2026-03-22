use std::path::PathBuf;

use clap::{Parser, Subcommand};

// ---------------------------------------------------------------------------
// Scenario subcommand tree
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(about = "Run deterministic scenario simulations")]
pub struct ScenarioCommand {
    #[command(subcommand)]
    pub sub: ScenarioSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum ScenarioSubcommand {
    Run(ScenarioRunArgs),
    Bench(ScenarioBenchArgs),
    Oracle(OracleArgs),
    Generate(ScenarioGenerateArgs),
}

#[derive(Debug, Parser)]
#[command(about = "Action oracle: evaluate or play scenarios with oracle guidance")]
pub struct OracleArgs {
    #[command(subcommand)]
    pub sub: OracleSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum OracleSubcommand {
    TransformerRl(TransformerRlArgs),
    MonitorTraces(MonitorTracesArgs),
    /// Run RL playtester agent population on scenarios
    Playtester(PlaytesterArgs),
}

#[derive(Debug, Parser)]
#[command(about = "Actor-critic RL with ability transformer")]
pub struct TransformerRlArgs {
    #[command(subcommand)]
    pub sub: TransformerRlSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum TransformerRlSubcommand {
    Generate(TransformerRlGenerateArgs),
    Eval(TransformerRlEvalArgs),
    /// Run IMPALA V-trace training loop using Burn V6 model (in-process, no SHM)
    ImpalaTrain(ImpalaTrainArgs),
}

#[derive(Debug, Parser)]
#[command(about = "IMPALA V-trace training with Burn V6 (in-process GPU inference + autodiff)")]
pub struct ImpalaTrainArgs {
    /// Path(s) to scenario .toml file(s) or directory(ies)
    pub path: Vec<PathBuf>,
    /// Output directory for checkpoints and logs
    #[arg(long, default_value = "generated/impala_v6")]
    pub output_dir: PathBuf,
    /// Path to embedding registry JSON
    #[arg(long)]
    pub embedding_registry: Option<PathBuf>,
    /// Resume from Burn checkpoint
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
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
    /// Exploration temperature for episode generation
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f32,
    /// Step interval for recording (every N ticks)
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
    /// Self-play: also run GPU inference for enemy units
    #[arg(long)]
    pub self_play: bool,
    /// Behavioral cloning mode: pure supervised imitation of squad AI
    #[arg(long)]
    pub bc: bool,
}

#[derive(Debug, Parser)]
#[command(about = "Generate RL episodes using transformer policy")]
pub struct TransformerRlGenerateArgs {
    /// Path(s) to scenario .toml file(s) or directory(ies)
    pub path: Vec<PathBuf>,
    /// Path to policy weights JSON (not required for --policy combined)
    #[arg(long)]
    pub weights: Option<PathBuf>,
    /// Policy type: transformer (default) or combined (squad AI)
    #[arg(long, default_value = "transformer")]
    pub policy: String,
    /// Output JSONL file
    #[arg(long, short, default_value = "generated/rl_episodes.jsonl")]
    pub output: PathBuf,
    /// Episodes per scenario
    #[arg(long, default_value_t = 10)]
    pub episodes: u32,
    /// Sampling temperature (higher = more exploration)
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f32,
    /// Number of parallel threads (0 = all cores)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
    /// Record steps every N ticks
    #[arg(long, default_value_t = 3)]
    pub step_interval: u64,
    /// Override scenario max_ticks
    #[arg(long)]
    pub max_ticks: Option<u64>,
    /// Pre-computed CLS embedding registry JSON (behavioral embeddings)
    #[arg(long)]
    pub embedding_registry: Option<PathBuf>,
    /// Enemy policy weights JSON (for self-play; omit to use default AI)
    #[arg(long)]
    pub enemy_weights: Option<PathBuf>,
    /// Enemy embedding registry JSON (for self-play)
    #[arg(long)]
    pub enemy_registry: Option<PathBuf>,
    /// Play each scenario from both sides (swap hero/enemy teams), doubling episodes
    #[arg(long)]
    pub swap_sides: bool,
    /// Use purely random actions (no model inference) for baseline measurement
    #[arg(long)]
    pub random_policy: bool,
    /// Use Burn in-process GPU inference — V5 model (no SHM, no Python server)
    #[arg(long)]
    pub burn: bool,
    /// Use Burn in-process GPU inference — V6 model (spatial cross-attn + latent interface)
    #[arg(long)]
    pub burn_v6: bool,
    /// Path to Burn checkpoint file to load weights from (for --burn or --burn-v6)
    #[arg(long)]
    pub burn_checkpoint: Option<std::path::PathBuf>,
}

#[derive(Debug, Parser)]
#[command(about = "Evaluate transformer RL policy (greedy)")]
pub struct TransformerRlEvalArgs {
    /// Path to scenario .toml file or directory
    pub path: PathBuf,
    /// Path to ability transformer weights JSON
    #[arg(long)]
    pub weights: PathBuf,
    /// Override scenario max_ticks
    #[arg(long)]
    pub max_ticks: Option<u64>,
    /// Pre-computed CLS embedding registry JSON (behavioral embeddings)
    #[arg(long)]
    pub embedding_registry: Option<PathBuf>,
    /// Enemy policy weights JSON (for self-play; omit to use default AI)
    #[arg(long)]
    pub enemy_weights: Option<PathBuf>,
    /// Enemy embedding registry JSON (for self-play)
    #[arg(long)]
    pub enemy_registry: Option<PathBuf>,
}

#[derive(Debug, Parser)]
#[command(about = "Run LOLA stream monitor on saved RL episode traces")]
pub struct MonitorTracesArgs {
    /// Path(s) to JSONL episode files or directories
    pub path: Vec<PathBuf>,
    /// Sample percentage (0.0-1.0, default 1.0 = all episodes)
    #[arg(long, default_value_t = 1.0)]
    pub sample: f32,
}

#[derive(Debug, Parser)]
#[command(about = "Generate diverse scenarios with coverage-driven constraint-based engine")]
pub struct ScenarioGenerateArgs {
    /// Output directory for generated .toml files
    #[arg(long, default_value = "scenarios/generated")]
    pub output: PathBuf,
    /// RNG seed for deterministic generation
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    /// Seed variants per base scenario (more = better trajectory diversity)
    #[arg(long, default_value_t = 3)]
    pub seed_variants: u32,
    /// Extra coverage-driven random scenarios on top of systematic strategies
    #[arg(long, default_value_t = 200)]
    pub extra_random: usize,
    /// Skip synergy pair scenarios (~700 base)
    #[arg(long, default_value_t = false)]
    pub no_synergy: bool,
    /// Skip stress archetype scenarios (~78 base)
    #[arg(long, default_value_t = false)]
    pub no_stress: bool,
    /// Skip difficulty ladder scenarios (~80 base)
    #[arg(long, default_value_t = false)]
    pub no_ladders: bool,
    /// Skip room-aware composition scenarios (~24 base)
    #[arg(long, default_value_t = false)]
    pub no_room_aware: bool,
    /// Skip team size spectrum scenarios (~38 base)
    #[arg(long, default_value_t = false)]
    pub no_sizes: bool,
    /// Print detailed coverage report
    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Debug, Parser)]
#[command(about = "Run a scenario .toml file or all *.toml files in a directory")]
pub struct ScenarioRunArgs {
    /// Path to a scenario .toml file, or a directory to run all *.toml files in
    pub path: PathBuf,
    /// Write JSON output to this file instead of stdout
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Print per-unit combat statistics table
    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Debug, Parser)]
#[command(about = "Benchmark scenario throughput (in-process, optionally parallel)")]
pub struct ScenarioBenchArgs {
    /// Path to a scenario .toml file to benchmark
    pub path: PathBuf,
    /// Number of iterations to run
    #[arg(short = 'n', long, default_value_t = 1000)]
    pub iterations: u32,
    /// Number of parallel threads (0 = use all available cores)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
    /// Profile per-phase timing breakdown (intent generation vs sim step)
    #[arg(long)]
    pub profile: bool,
}

#[derive(Debug, Parser)]
#[command(about = "Run RL playtester agent population on scenarios (random policy baseline)")]
pub struct PlaytesterArgs {
    /// Path(s) to scenario .toml file(s) or directory(ies)
    pub path: Vec<PathBuf>,
    /// Number of agents in the population
    #[arg(long, default_value_t = 8)]
    pub population: usize,
    /// Number of training iterations
    #[arg(long, default_value_t = 5)]
    pub iterations: usize,
    /// Episodes per agent per iteration
    #[arg(long, default_value_t = 4)]
    pub episodes_per_agent: usize,
    /// Base RNG seed
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    /// Step recording interval (every N ticks)
    #[arg(long, default_value_t = 5)]
    pub step_interval: u64,
    /// Override scenario max_ticks
    #[arg(long)]
    pub max_ticks: Option<u64>,
    /// Number of parallel threads (0 = all cores)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
    /// Output directory for metrics and reports
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
}
