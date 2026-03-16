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
    Eval(OracleEvalArgs),
    Play(OraclePlayArgs),
    Dataset(OracleDatasetArgs),
    CombatDataset(CombatDatasetArgs),
    Student(OracleStudentArgs),
    AbilityDataset(AbilityDatasetArgs),
    SelfPlay(SelfPlayArgs),
    RawDataset(RawDatasetArgs),
    OutcomeDataset(OutcomeDatasetArgs),
    NextstateDataset(NextstateDatasetArgs),
    TransformerPlay(TransformerPlayArgs),
    TransformerRl(TransformerRlArgs),
    OperatorDataset(OperatorDatasetArgs),
    AbilityProfile(AbilityProfileArgs),
    MonitorTraces(MonitorTracesArgs),
}

#[derive(Debug, Parser)]
#[command(about = "Generate oracle-labeled dataset with 311 raw features and 14-class actions")]
pub struct RawDatasetArgs {
    /// Path to a scenario .toml file, or a directory
    pub path: PathBuf,
    /// Output JSONL file
    #[arg(long, default_value = "generated/raw_dataset.jsonl")]
    pub output: PathBuf,
    /// Rollout depth in ticks (default 10)
    #[arg(long, default_value_t = 10)]
    pub depth: u64,
}

#[derive(Debug, Parser)]
#[command(about = "Evaluate default AI decisions against oracle recommendations")]
pub struct OracleEvalArgs {
    /// Path to a scenario .toml file, or a directory to evaluate all *.toml files in
    pub path: PathBuf,
    /// Directory to write per-scenario JSONL decision logs
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
    /// Rollout depth in ticks (default 10)
    #[arg(long, default_value_t = 10)]
    pub depth: u64,
}

#[derive(Debug, Parser)]
#[command(about = "Run scenarios with heroes using oracle top picks instead of default AI")]
pub struct OraclePlayArgs {
    /// Path to a scenario .toml file, or a directory to play all *.toml files in
    pub path: PathBuf,
    /// Rollout depth in ticks (default 10)
    #[arg(long, default_value_t = 10)]
    pub depth: u64,
}

#[derive(Debug, Parser)]
#[command(about = "Generate training dataset from oracle-played scenarios")]
pub struct OracleDatasetArgs {
    /// Path to a scenario .toml file, or a directory
    pub path: PathBuf,
    /// Output JSONL file
    #[arg(long, default_value = "generated/oracle_dataset.jsonl")]
    pub output: PathBuf,
    /// Rollout depth in ticks (default 100)
    #[arg(long, default_value_t = 100)]
    pub depth: u64,
}

#[derive(Debug, Parser)]
#[command(about = "Generate 5-class combat-only dataset (abilities excluded for frozen evaluators)")]
pub struct CombatDatasetArgs {
    /// Path to a scenario .toml file, or a directory
    pub path: PathBuf,
    /// Output JSONL file
    #[arg(long, default_value = "generated/combat_dataset.jsonl")]
    pub output: PathBuf,
    /// Rollout depth in ticks (default 100)
    #[arg(long, default_value_t = 100)]
    pub depth: u64,
}

#[derive(Debug, Parser)]
#[command(about = "Evaluate a trained student model against default AI")]
pub struct OracleStudentArgs {
    /// Path to scenario .toml file or directory
    pub path: PathBuf,
    /// Path to trained model JSON weights
    #[arg(long)]
    pub model: PathBuf,
    /// Path to frozen ability evaluator weights JSON (composes with student model)
    #[arg(long)]
    pub ability_eval: Option<PathBuf>,
}

#[derive(Debug, Parser)]
#[command(about = "Self-play: generate episodes or evaluate a policy")]
pub struct SelfPlayArgs {
    #[command(subcommand)]
    pub sub: SelfPlaySubcommand,
}

#[derive(Debug, Subcommand)]
pub enum SelfPlaySubcommand {
    Generate(SelfPlayGenerateArgs),
    Eval(SelfPlayEvalArgs),
}

#[derive(Debug, Parser)]
#[command(about = "Generate self-play episodes for REINFORCE training")]
pub struct SelfPlayGenerateArgs {
    /// Path to scenario .toml file or directory (only needed for stage=4v4)
    #[arg(default_value = "scenarios/")]
    pub path: PathBuf,
    /// Output JSONL file for episodes
    #[arg(long, default_value = "generated/self_play_episodes.jsonl")]
    pub output: PathBuf,
    /// Path to policy weights JSON (random init if not provided)
    #[arg(long)]
    pub policy: Option<PathBuf>,
    /// Episodes per scenario
    #[arg(long, default_value_t = 10)]
    pub episodes: u32,
    /// Exploration temperature
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f32,
    /// Number of parallel threads (0 = all cores)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
    /// Override scenario max_ticks (shorter = faster episodes)
    #[arg(long)]
    pub max_ticks: Option<u64>,
    /// Curriculum stage: move, kill, 2v2, 4v4 (default: 4v4 uses scenario files)
    #[arg(long, default_value = "4v4")]
    pub stage: String,
}

#[derive(Debug, Parser)]
#[command(about = "Evaluate a trained self-play policy (greedy)")]
pub struct SelfPlayEvalArgs {
    /// Path to scenario .toml file or directory
    pub path: PathBuf,
    /// Path to policy weights JSON
    #[arg(long)]
    pub policy: PathBuf,
    /// Override scenario max_ticks
    #[arg(long)]
    pub max_ticks: Option<u64>,
    /// Joint focus-target search interval in ticks (0 = disabled)
    #[arg(long, default_value_t = 0)]
    pub focus: u64,
}

#[derive(Debug, Parser)]
#[command(about = "Generate ability evaluator training dataset from oracle rollouts")]
pub struct AbilityDatasetArgs {
    /// Path to a scenario .toml file, or a directory
    pub path: PathBuf,
    /// Output JSONL file
    #[arg(long, default_value = "generated/ability_eval_dataset.jsonl")]
    pub output: PathBuf,
    /// Rollout depth in ticks (default 10)
    #[arg(long, default_value_t = 10)]
    pub depth: u64,
}

#[derive(Debug, Parser)]
#[command(about = "Generate outcome prediction dataset for entity encoder pre-training")]
pub struct OutcomeDatasetArgs {
    /// Path to a scenario .toml file, or a directory
    pub path: PathBuf,
    /// Output JSONL file
    #[arg(long, default_value = "generated/outcome_dataset.jsonl")]
    pub output: PathBuf,
    /// Sample every N ticks (controls dataset size)
    #[arg(long, default_value_t = 5)]
    pub sample_interval: u64,
    /// Use v2 format: variable-length entities + threats
    #[arg(long)]
    pub v2: bool,
}

#[derive(Debug, Parser)]
#[command(about = "Generate next-state prediction dataset for entity encoder pre-training")]
pub struct NextstateDatasetArgs {
    /// Path to a scenario .toml file, or a directory
    pub path: PathBuf,
    /// Output JSONL file
    #[arg(long, default_value = "generated/nextstate_dataset.jsonl")]
    pub output: PathBuf,
    /// Sample every N ticks (1 = every tick for max pairing flexibility)
    #[arg(long, default_value_t = 1)]
    pub sample_interval: u64,
}

#[derive(Debug, Parser)]
#[command(about = "Play scenarios using ability transformer (DSL tokens + entity encoder)")]
pub struct TransformerPlayArgs {
    /// Path to scenario .toml file or directory
    pub path: PathBuf,
    /// Path to ability transformer decision weights JSON
    #[arg(long)]
    pub weights: PathBuf,
    /// Urgency threshold for firing abilities (0.0-1.0)
    #[arg(long, default_value_t = 0.4)]
    pub urgency_threshold: f32,
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
}

#[derive(Debug, Parser)]
#[command(about = "Generate RL episodes using transformer policy")]
pub struct TransformerRlGenerateArgs {
    /// Path(s) to scenario .toml file(s) or directory(ies)
    pub path: Vec<PathBuf>,
    /// Path to policy weights JSON (not required for --policy combined)
    #[arg(long)]
    pub weights: Option<PathBuf>,
    /// Policy type: transformer (default) or combined (ability-eval + squad AI)
    #[arg(long, default_value = "transformer")]
    pub policy: String,
    /// Path to ability eval weights JSON (for --policy combined)
    #[arg(long)]
    pub ability_eval: Option<PathBuf>,
    /// Path to combat student model JSON (for --policy combined)
    #[arg(long)]
    pub student_model: Option<PathBuf>,
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
    /// GPU shared memory path (e.g. /dev/shm/impala_inf) — uses mmap batched inference
    #[arg(long)]
    pub gpu_shm: Option<String>,
    /// Concurrent sims per thread for GPU pipelining (default 1 = blocking)
    #[arg(long, default_value_t = 1)]
    pub sims_per_thread: usize,
    /// Self-play via GPU: both hero and enemy teams use the same GPU inference server
    #[arg(long)]
    pub self_play_gpu: bool,
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
#[command(about = "Generate ability operator training dataset from scenario replays")]
pub struct OperatorDatasetArgs {
    /// Path to a scenario .toml file, or a directory
    pub path: PathBuf,
    /// Output npz file
    #[arg(long, default_value = "generated/operator_dataset.npz")]
    pub output: PathBuf,
    /// Number of parallel threads (0 = all cores)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
}

#[derive(Debug, Parser)]
#[command(about = "Profile abilities via controlled sim experiments for behavioral embeddings")]
pub struct AbilityProfileArgs {
    /// Output npz file
    #[arg(long, default_value = "dataset/ability_profiles.npz")]
    pub output: PathBuf,
    /// Samples per ability (across all condition combinations)
    #[arg(long, default_value_t = 1000)]
    pub samples_per_ability: usize,
    /// Number of parallel threads (0 = all cores)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
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
    /// Path to ability evaluator weights JSON (enables interrupt-driven ability usage)
    #[arg(long)]
    pub ability_eval: Option<PathBuf>,
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
