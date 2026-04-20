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
    /// Building AI dataset generation and coverage analysis
    BuildingAi(BuildingAiCommand),
    /// Compile DSL sources (`assets/sim/*.sim`) into Rust + Python artefacts.
    CompileDsl(CompileDslArgs),
}

#[derive(Debug, Parser)]
#[command(about = "Compile DSL sources to Rust + Python + schema hash")]
pub struct CompileDslArgs {
    /// Source directory holding `*.sim` files (recursively walked).
    #[arg(long, default_value = "assets/sim")]
    pub src: PathBuf,
    /// Destination root for Rust output. Files are written under
    /// `<out-rust>/events/` and `<out-rust>/schema.rs`.
    #[arg(long, default_value = "crates/engine_rules/src")]
    pub out_rust: PathBuf,
    /// Destination root for emitted physics handlers. Files are written
    /// under `<out-physics>/` (per-rule modules + `mod.rs`). Defaults into
    /// the engine crate because emitted handlers reference
    /// `engine::cascade::*` and `engine::state::SimState`; emitting them
    /// into `engine_rules` would invert the existing dep direction
    /// (`engine` depends on `engine_rules`, not the other way around).
    /// Documented in `docs/game/feature_flow.md`.
    #[arg(long, default_value = "crates/engine/src/generated/physics")]
    pub out_physics: PathBuf,
    /// Destination root for emitted mask predicates. Same dep rationale
    /// as physics — masks reference `engine::state::SimState`.
    #[arg(long, default_value = "crates/engine/src/generated/mask")]
    pub out_mask: PathBuf,
    /// Destination root for emitted scoring tables. Scoring rows are POD
    /// data shared between CPU scorer and GPU kernel, so they live in
    /// `engine_rules` with the other shared-data emissions.
    #[arg(long, default_value = "crates/engine_rules/src/scoring")]
    pub out_scoring: PathBuf,
    /// Destination root for emitted entity data (CreatureType enum,
    /// capability structs, hostility matrix). Pure data; lives in
    /// `engine_rules` next to events.
    #[arg(long, default_value = "crates/engine_rules/src/entities")]
    pub out_entity: PathBuf,
    /// Destination root for Python output. Files are written under
    /// `<out-python>/events/`.
    #[arg(long, default_value = "generated/python")]
    pub out_python: PathBuf,
    /// Compare the emitted artefacts against the committed output. Exit 0 if
    /// identical, exit 1 with a file-level diff otherwise. No files are
    /// written when this is set.
    #[arg(long)]
    pub check: bool,
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
    /// Start a WebSocket server on this port to stream TraceFrame JSON to the web visualizer
    #[arg(long)]
    pub ws: Option<u16>,
    /// Peaceful mode: no monsters, single forest settlement, zero starting gold.
    /// NPCs must gather resources and build everything from scratch.
    #[arg(long)]
    pub peaceful: bool,
    /// Open a Vulkan window and render the voxel world (requires --features app)
    #[arg(long)]
    pub render: bool,
    /// World preset: "small" = fixed 9x9x9 forest test scene (default: infinite)
    #[arg(long)]
    pub world: Option<String>,
    /// Write per-system diagnostic bench JSON to this path.
    /// Requires --features profile-systems for populated system_timings.
    #[arg(long)]
    pub bench_json: Option<String>,
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
// Building AI dataset generation subcommand tree
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(about = "Building AI dataset generation and coverage analysis")]
pub struct BuildingAiCommand {
    #[command(subcommand)]
    pub sub: BuildingAiSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum BuildingAiSubcommand {
    /// Run a single scenario TOML: load, generate world state, run oracle, validate
    Run(BuildingAiRunArgs),
    /// Generate labeled (observation, action) pairs for behavioral cloning
    Generate(BuildingAiGenerateArgs),
    /// Compute coverage matrix from an existing dataset
    Coverage(BuildingAiCoverageArgs),
    /// Generate supplemental data targeting under-represented matrix cells
    FillGaps(BuildingAiFillGapsArgs),
    /// Collect RL trajectories by running BuildingEnv episodes with oracle policy
    EnvCollect(BuildingAiEnvCollectArgs),
}

#[derive(Debug, Parser)]
#[command(about = "Run a single building AI scenario")]
pub struct BuildingAiRunArgs {
    /// Path to the scenario TOML file
    pub scenario: PathBuf,
    /// Base directory for resolving template/profile references
    #[arg(long, default_value = "building_scenarios")]
    pub base_dir: PathBuf,
    /// Run validation checks on the generated state and oracle output
    #[arg(long)]
    pub validate: bool,
    /// After oracle pipeline, run the world sim for N ticks
    #[arg(long)]
    pub sim_ticks: Option<u64>,
    /// Save final WorldState as JSON
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Print spatial sanity checks after sim
    #[arg(long)]
    pub diagnostics: bool,
}

#[derive(Debug, Parser)]
#[command(about = "Generate building AI BC dataset")]
pub struct BuildingAiGenerateArgs {
    /// Target number of (obs, action) pairs
    #[arg(long, default_value_t = 50000)]
    pub pairs: u64,
    /// Minimum pairs per active coverage matrix cell
    #[arg(long, default_value_t = 100)]
    pub min_cell: u32,
    /// Output JSONL file
    #[arg(long, default_value = "generated/building_bc.jsonl")]
    pub output: PathBuf,
    /// Coverage report output JSON file
    #[arg(long, default_value = "generated/building_coverage.json")]
    pub coverage: PathBuf,
    /// RNG seed
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

#[derive(Debug, Parser)]
#[command(about = "Compute coverage matrix from existing dataset")]
pub struct BuildingAiCoverageArgs {
    /// Path to the JSONL dataset file
    pub dataset: PathBuf,
}

#[derive(Debug, Parser)]
#[command(about = "Generate supplemental data targeting coverage gaps")]
pub struct BuildingAiFillGapsArgs {
    /// Path to the existing JSONL dataset file
    #[arg(long)]
    pub dataset: PathBuf,
    /// Minimum pairs per active coverage matrix cell
    #[arg(long, default_value_t = 100)]
    pub min_cell: u32,
    /// Output JSONL file for supplemental data
    #[arg(long, default_value = "generated/building_bc_supplement.jsonl")]
    pub output: PathBuf,
    /// RNG seed
    #[arg(long, default_value_t = 43)]
    pub seed: u64,
}

#[derive(Debug, Parser)]
#[command(about = "Collect RL trajectories from BuildingEnv with oracle policy")]
pub struct BuildingAiEnvCollectArgs {
    /// Number of episodes to run
    #[arg(long, default_value_t = 100)]
    pub episodes: usize,
    /// Curriculum level (1-4)
    #[arg(long, default_value_t = 1)]
    pub level: u8,
    /// Output NPZ file for flat trajectories
    #[arg(long, default_value = "generated/building_env_trajectories.npz")]
    pub output: PathBuf,
    /// RNG seed
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Fraction of actions that are random (epsilon-greedy exploration)
    #[arg(long, default_value_t = 0.1)]
    pub epsilon: f32,
}
