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
    Scenario(ScenarioCommand),
    /// Build with burn-gpu and run IMPALA V6 training (auto-detects libtorch)
    TrainV6(TrainV6Args),
    Roomgen(RoomgenCommand),
    /// Model backend management (status, test)
    Model(ModelCommand),
    /// AOT content generation pipeline
    ContentGen(ContentGenCommand),
    /// ASCII art generation
    AsciiGen(AsciiGenCommand),
    /// Run headless campaign batch simulation
    CampaignBatch(CampaignBatchArgs),
    /// Run MCTS bootstrap campaigns and export BC training data
    MctsBootstrap(MctsBootstrapArgs),
    /// Generate behavioral cloning data from heuristic policy with coverage analysis
    HeuristicBc(HeuristicBcArgs),
    /// BFS state-space exploration with cluster-and-prune
    BfsExplore(BfsExploreArgs),
    /// Fuzz campaigns with randomized configs and random actions
    CampaignFuzz(CampaignFuzzArgs),
    /// Collect VAE training dataset (sweep campaigns + LLM generation + slot extraction)
    VaeDataset(VaeDatasetArgs),
    /// Extract slot vectors from DSL text using Rust parsers (ground truth)
    VaeExtractSlots(VaeExtractSlotsArgs),
    /// Build VAE training data from existing ability DSL files (no LLM needed)
    VaeGtDataset(VaeGtDatasetArgs),
    /// Analyze a BFS exploration JSONL output file
    BfsAnalyze(BfsAnalyzeArgs),
    /// Generate Sleeping King champion candidates with grammar-walked abilities
    ChampionGen(ChampionGenArgs),
    /// Generate synthetic abilities via grammar walker
    SynthAbilities {
        #[arg(long, default_value_t = 100000)] count: usize,
        #[arg(long, default_value_t = 42)] seed: u64,
        /// Emit DSL text instead of slot JSONL
        #[arg(long)] dsl: bool,
    },
    /// Run world simulation benchmark with profiling
    WorldSim(WorldSimArgs),
    /// Visualize a world sim trace file
    Visualize(VisualizeArgs),
}

#[derive(Debug, Parser)]
#[command(about = "Run world simulation benchmark with profiling")]
pub struct WorldSimArgs {
    /// Number of entities to simulate
    #[arg(long, default_value_t = 2000)]
    pub entities: usize,
    /// Number of ticks to run
    #[arg(long, default_value_t = 5000)]
    pub ticks: u64,
    /// Use parallel (rayon) tick
    #[arg(long)]
    pub parallel: bool,
    /// Number of settlements
    #[arg(long, default_value_t = 10)]
    pub settlements: usize,
    /// Number of monsters
    #[arg(long, default_value_t = 200)]
    pub monsters: usize,
    /// RNG seed
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Run for N seconds instead of N ticks (overrides --ticks)
    #[arg(long)]
    pub duration_secs: Option<u64>,
    /// Resource-rich world (10x stockpiles, varied production, more commodities)
    #[arg(long)]
    pub rich: bool,
    /// Seed for terrain generation (default: same as --seed)
    #[arg(long)]
    pub terrain_seed: Option<u64>,
    /// Number of factions (default: 6)
    #[arg(long, default_value_t = 6)]
    pub factions: usize,
    /// Enable pre-placed trade routes between nearby settlements
    #[arg(long)]
    pub trade_routes: bool,
    /// Print chronicle log after simulation
    #[arg(long)]
    pub chronicle: bool,
    /// Print a narrative world history summary after simulation
    #[arg(long)]
    pub history: bool,
    /// Warm-up mode: rich defaults, chronicle output, and summary at end
    #[arg(long)]
    pub warm: bool,
    /// Output path: serialize warmed world state as JSON for game loading
    #[arg(long)]
    pub output: Option<String>,
    /// Load world state from a previously saved JSON file (resume simulation)
    #[arg(long)]
    pub load: Option<String>,
    /// Record a trace file during simulation for later visualization
    #[arg(long)]
    pub trace: Option<String>,
    /// Export world history as a readable markdown narrative file
    #[arg(long)]
    pub export_history: Option<String>,
    /// Divine intervention: trigger events mid-simulation.
    /// Format: "bless:SETTLEMENT", "curse:REGION", "champion:SETTLEMENT", "prophecy", "plague:SETTLEMENT"
    #[arg(long)]
    pub intervene: Option<String>,
}

#[derive(Debug, Parser)]
#[command(about = "Visualize a world sim trace file")]
pub struct VisualizeArgs {
    /// Path to a world sim trace JSON file
    pub trace: String,
    /// Initial playback speed (ticks per second)
    #[arg(long, default_value_t = 100.0)]
    pub speed: f32,
    /// Target render FPS
    #[arg(long, default_value_t = 10)]
    pub fps: u32,
}

#[derive(Debug, Parser)]
#[command(about = "Analyze a BFS exploration JSONL output file and print a structured report")]
pub struct BfsAnalyzeArgs {
    /// Path to the BFS JSONL output file
    pub path: String,
}

#[derive(Debug, Parser)]
#[command(about = "Generate Sleeping King champion candidates with grammar-walked abilities")]
pub struct ChampionGenArgs {
    /// RNG seed for ability generation
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    /// Number of candidates per slot
    #[arg(long, default_value_t = 3)]
    pub candidates: usize,
}

#[derive(Debug, Parser)]
#[command(about = "Build VAE training data from ground-truth ability DSL files")]
pub struct VaeGtDatasetArgs {
    /// Number of campaigns to sweep for contexts
    #[arg(long, default_value_t = 1000)]
    pub campaigns: u64,
    /// Max ticks per campaign
    #[arg(long, default_value_t = 30_000)]
    pub max_ticks: u64,
    /// Threads
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
    /// Base RNG seed
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    /// Abilities sampled per trigger context
    #[arg(long, default_value_t = 4)]
    pub samples_per_context: usize,
    /// Output JSONL file
    #[arg(long, default_value = "generated/vae_gt_dataset.jsonl")]
    pub output: String,
}

#[derive(Debug, Parser)]
#[command(about = "Parse DSL text through Rust parsers and extract exact slot vectors")]
pub struct VaeExtractSlotsArgs {
    /// Input JSONL with {content_type, raw_dsl} records
    #[arg(long, default_value = "generated/vae_dataset_final.jsonl")]
    pub input: String,
}

#[derive(Debug, Parser)]
#[command(about = "Collect training data for the grammar-guided content VAE")]
pub struct VaeDatasetArgs {
    /// Number of campaigns to sweep
    #[arg(long, default_value_t = 100)]
    pub campaigns: u64,
    /// Max ticks per campaign
    #[arg(long, default_value_t = 30_000)]
    pub max_ticks: u64,
    /// Threads for campaign sweep
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
    /// Base RNG seed
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    /// Output directory
    #[arg(long, default_value = "generated")]
    pub output_dir: String,
    /// Skip LLM generation (sweep only, or use existing store)
    #[arg(long)]
    pub no_llm: bool,
    /// Only run campaign sweep (no generation or extraction)
    #[arg(long)]
    pub sweep_only: bool,
    /// Only run extraction on existing contexts + store
    #[arg(long)]
    pub extract_only: bool,
    /// Skip procedural item/quest generation
    #[arg(long)]
    pub no_procedural: bool,
    /// Parallel LLM workers
    #[arg(long, default_value_t = 4)]
    pub workers: usize,
    /// LLM candidates per item (best-of-N)
    #[arg(long, default_value_t = 3)]
    pub candidates: u32,
    /// Ollama server URL
    #[arg(long, default_value = "http://localhost:11434")]
    pub llm_url: String,
    /// Ollama model name
    #[arg(long, default_value = "qwen35-9b")]
    pub llm_model: String,
    /// Campaign config TOML
    #[arg(long)]
    pub config: Option<std::path::PathBuf>,
}

#[derive(Debug, Parser)]
#[command(about = "Fuzz campaigns with randomized configs and random action policies")]
pub struct CampaignFuzzArgs {
    /// Total campaigns to fuzz
    #[arg(long, default_value_t = 10_000)]
    pub campaigns: u64,
    /// Max ticks per campaign
    #[arg(long, default_value_t = 100_000)]
    pub max_ticks: u64,
    /// Number of threads (0 = all cores)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
    /// Base RNG seed
    #[arg(long, default_value_t = 0xFEED)]
    pub seed: u64,
    /// Config mutation strength (0.0-1.0)
    #[arg(long, default_value_t = 0.5)]
    pub mutation_strength: f64,
    /// Fraction of campaigns using purely random actions (0.0-1.0)
    #[arg(long, default_value_t = 0.5)]
    pub random_action_ratio: f64,
    /// Output JSONL file for findings
    #[arg(long, default_value = "generated/fuzz_findings.jsonl")]
    pub output: String,
    /// Progress report interval
    #[arg(long, default_value_t = 500)]
    pub report_interval: u64,
}

#[derive(Debug, Parser)]
#[command(about = "BFS state-space exploration: expand all actions, cluster leaves, prune to medians")]
pub struct BfsExploreArgs {
    /// Max BFS waves (0 = unlimited, run until all branches complete)
    #[arg(long, default_value_t = 0)]
    pub max_waves: u32,
    /// Ticks to advance per branch (let action play out)
    #[arg(long, default_value_t = 200)]
    pub ticks_per_branch: u64,
    /// Clusters per wave (controls width)
    #[arg(long, default_value_t = 20)]
    pub clusters: usize,
    /// Initial root states to generate
    #[arg(long, default_value_t = 50)]
    pub initial_roots: usize,
    /// Max ticks for heuristic trajectory (root generation)
    #[arg(long, default_value_t = 15000)]
    pub trajectory_ticks: u64,
    /// Root sampling interval from trajectory
    #[arg(long, default_value_t = 300)]
    pub root_interval: u64,
    /// Base RNG seed
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    /// Number of threads (0 = all cores)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
    /// Output JSONL file
    #[arg(long, default_value = "generated/bfs_explore.jsonl")]
    pub output: String,
    /// Path to campaign config TOML
    #[arg(long)]
    pub config: Option<std::path::PathBuf>,
    /// Enable LLM content generation via Ollama (requires running server)
    #[arg(long)]
    pub llm: bool,
    /// Ollama server URL (default: http://localhost:11434)
    #[arg(long, default_value = "http://localhost:11434")]
    pub llm_url: String,
    /// Ollama model name (default: qwen35-9b)
    #[arg(long, default_value = "qwen35-9b")]
    pub llm_model: String,
    /// LLM candidates per generation (best-of-N, default: 3)
    #[arg(long, default_value_t = 3)]
    pub llm_candidates: usize,
    /// Path to VAE model weights JSON (enables instant content generation)
    #[arg(long)]
    pub vae_model: Option<String>,
}

#[derive(Debug, Parser)]
#[command(about = "Generate BC training data from heuristic policy with state coverage analysis")]
pub struct HeuristicBcArgs {
    /// Total campaigns to run
    #[arg(long, default_value_t = 10_000)]
    pub campaigns: u64,
    /// Maximum ticks per campaign
    #[arg(long, default_value_t = 30_000)]
    pub max_ticks: u64,
    /// Number of threads (0 = all cores)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
    /// Base RNG seed
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    /// Progress report interval
    #[arg(long, default_value_t = 1000)]
    pub report_interval: u64,
    /// Output JSONL file
    #[arg(long, default_value = "generated/heuristic_bc.jsonl")]
    pub output: String,
    /// Record every N-th decision (1 = all)
    #[arg(long, default_value_t = 1)]
    pub sample_rate: usize,
    /// Path to campaign config TOML
    #[arg(long)]
    pub config: Option<std::path::PathBuf>,
}

#[derive(Debug, Parser)]
#[command(about = "Run headless campaigns in parallel for validation and data generation")]
pub struct CampaignBatchArgs {
    /// Target number of successful runs
    #[arg(long, default_value_t = 100_000)]
    pub target: u64,
    /// Maximum ticks per campaign before timeout
    #[arg(long, default_value_t = 50_000)]
    pub max_ticks: u64,
    /// Number of threads (0 = all cores)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
    /// Base RNG seed
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    /// Progress report interval (every N runs)
    #[arg(long, default_value_t = 1000)]
    pub report_interval: u64,
    /// Record traces for the first N campaigns (0 = disabled)
    #[arg(long, default_value_t = 0)]
    pub record_traces: u64,
    /// Snapshot interval for traces (ticks between keyframes)
    #[arg(long, default_value_t = 100)]
    pub trace_snapshot_interval: u64,
    /// Output directory for trace files
    #[arg(long, default_value = "generated/traces")]
    pub trace_output_dir: String,
    /// Path to campaign config TOML (overrides defaults)
    #[arg(long)]
    pub config: Option<std::path::PathBuf>,
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
    /// Behavioral cloning mode: pure supervised imitation of squad AI
    #[arg(long)]
    pub bc: bool,
    /// Skip build step (assume binary is already built with burn-gpu)
    #[arg(long)]
    pub no_build: bool,
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
// Model backend subcommand tree
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(about = "Model backend management")]
pub struct ModelCommand {
    #[command(subcommand)]
    pub sub: ModelSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum ModelSubcommand {
    /// Check model backend availability and hardware tier
    Status,
    /// Test model generation with a prompt
    Test(ModelTestArgs),
}

#[derive(Debug, Parser)]
#[command(about = "Test model generation")]
pub struct ModelTestArgs {
    /// Prompt to send to the model
    pub prompt: String,
    /// Path to model weights
    #[arg(long)]
    pub model_path: Option<PathBuf>,
    /// Path to inference script
    #[arg(long)]
    pub script_path: Option<PathBuf>,
    /// Python executable
    #[arg(long)]
    pub python: Option<String>,
    /// Deterministic seed
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

// ---------------------------------------------------------------------------
// Content generation subcommand tree
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(about = "AOT content generation pipeline")]
pub struct ContentGenCommand {
    #[command(subcommand)]
    pub sub: ContentGenSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum ContentGenSubcommand {
    /// Run the full 9-stage world generation pipeline
    Generate(ContentGenGenerateArgs),
    /// Inspect cached world context for a campaign seed
    Inspect(ContentGenInspectArgs),
}

#[derive(Debug, Parser)]
pub struct ContentGenGenerateArgs {
    /// Campaign seed for deterministic generation
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Output directory for cached world context
    #[arg(long, default_value = "generated/campaigns")]
    pub output_dir: PathBuf,
    /// Resume from a specific stage (skips earlier stages)
    #[arg(long)]
    pub from_stage: Option<String>,
}

#[derive(Debug, Parser)]
pub struct ContentGenInspectArgs {
    /// Campaign seed to inspect
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Directory where campaign data is cached
    #[arg(long, default_value = "generated/campaigns")]
    pub output_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// ASCII art generation subcommand tree
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(about = "ASCII art generation")]
pub struct AsciiGenCommand {
    #[command(subcommand)]
    pub sub: AsciiGenSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum AsciiGenSubcommand {
    /// Generate a single ASCII art grid
    Generate(AsciiGenGenerateArgs),
    /// Batch-export ASCII art grids as JSONL
    Export(AsciiGenExportArgs),
}

#[derive(Debug, Parser)]
pub struct AsciiGenGenerateArgs {
    /// Description of what to generate
    pub prompt: String,
    /// Art style: environment, portrait, item, ui
    #[arg(long, default_value = "environment")]
    pub style: String,
    /// Grid width in characters
    #[arg(long, default_value_t = 40)]
    pub width: usize,
    /// Grid height in characters
    #[arg(long, default_value_t = 20)]
    pub height: usize,
    /// Deterministic seed
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Optional output file (JSON)
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Parser)]
pub struct AsciiGenExportArgs {
    /// Output JSONL file
    #[arg(long, default_value = "generated/ascii_art.jsonl")]
    pub output: PathBuf,
    /// Number of grids per style
    #[arg(long, default_value_t = 100)]
    pub count_per_style: usize,
    /// Grid width
    #[arg(long, default_value_t = 20)]
    pub width: usize,
    /// Grid height
    #[arg(long, default_value_t = 10)]
    pub height: usize,
    /// Base seed
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
}

// ---------------------------------------------------------------------------
// MCTS bootstrap subcommand
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(about = "Run MCTS bootstrap campaigns and export BC training data as JSONL")]
pub struct MctsBootstrapArgs {
    /// Total campaigns to run
    #[arg(long, default_value_t = 1000)]
    pub campaigns: u64,
    /// MCTS simulations per decision point
    #[arg(long, default_value_t = 200)]
    pub simulations: u32,
    /// Maximum ticks per campaign before timeout
    #[arg(long, default_value_t = 30000)]
    pub max_ticks: u64,
    /// Rollout horizon in ticks
    #[arg(long, default_value_t = 5000)]
    pub rollout_horizon: u64,
    /// Ticks between decision points
    #[arg(long, default_value_t = 50)]
    pub decision_interval: u64,
    /// Base RNG seed
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    /// Number of threads (0 = all cores)
    #[arg(short = 'j', long, default_value_t = 0)]
    pub threads: usize,
    /// Output JSONL file
    #[arg(long, default_value = "generated/mcts_bootstrap.jsonl")]
    pub output: std::path::PathBuf,
    /// Path to campaign config TOML (overrides defaults)
    #[arg(long)]
    pub config: Option<std::path::PathBuf>,
}
