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
    AbilityEncoderExport(AbilityEncoderExportArgs),
    SelfPlay(SelfPlayArgs),
    RawDataset(RawDatasetArgs),
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
    /// Path to frozen ability encoder JSON (enriches ability eval with 32-dim embeddings)
    #[arg(long)]
    pub ability_encoder: Option<PathBuf>,
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
    /// Path to frozen ability encoder JSON (appends 32-dim embeddings to features)
    #[arg(long)]
    pub ability_encoder: Option<PathBuf>,
}

#[derive(Debug, Parser)]
#[command(about = "Export ability properties + labels for encoder training")]
pub struct AbilityEncoderExportArgs {
    /// Output JSON file
    #[arg(long, default_value = "generated/ability_encoder_data.json")]
    pub output: PathBuf,
    /// Also include LoL heroes from assets/lol_heroes/
    #[arg(long, default_value_t = true)]
    pub include_lol: bool,
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
}
