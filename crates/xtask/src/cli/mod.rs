pub mod map;

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
    /// Build with burn-gpu and run IMPALA V6 training (auto-detects libtorch)
    TrainV6(TrainV6Args),
    /// Compile DSL sources (`assets/sim/*.sim`) into Rust + Python artefacts.
    CompileDsl(CompileDslArgs),
    /// Behavioral parity harness for the Compute-Graph IR (CG) pipeline.
    /// Reads side-channel CG output (produced by `compile-dsl --cg-emit-into`)
    /// and reports whether a CG-overlaid `engine_gpu_rules` will compile.
    /// Today this is a diagnostic command — see `# Limitations` on
    /// [`CompileDslParityArgs`].
    CompileDslParity(CompileDslParityArgs),
    /// Interactive per-phase REPL: pause at each tick pipeline phase, inspect state.
    Debug(DebugArgs),
    /// Non-interactive trace: collect mask + agent-history snapshots for N ticks.
    Trace(TraceArgs),
    /// Phase timing histogram: run N ticks, dump per-phase ns samples as a table.
    Profile(ProfileArgs),
    /// Reproduction bundle: capture scenario state + causal tree to a file.
    Repro(ReproArgs),
}

/// Arguments for `debug` subcommand.
#[derive(Debug, clap::Args)]
#[command(about = "Interactive per-phase REPL (pause at each pipeline phase)")]
pub struct DebugArgs {
    /// Scenario TOML file to load (stub: path is logged; engine not invoked).
    #[arg(long, default_value = "scenarios/basic_4v4.toml")]
    pub scenario: PathBuf,
    /// Number of ticks to step through.
    #[arg(long, default_value_t = 5)]
    pub ticks: u32,
    /// Run non-interactively (auto-continue all phases; useful for CI).
    #[arg(long)]
    pub no_interactive: bool,
}

/// Arguments for `trace` subcommand.
#[derive(Debug, clap::Args)]
#[command(about = "Non-interactive mask + agent-history collection")]
pub struct TraceArgs {
    /// Scenario TOML file to load (stub: path is logged; engine not invoked).
    #[arg(long, default_value = "scenarios/basic_4v4.toml")]
    pub scenario: PathBuf,
    /// Number of ticks to collect.
    #[arg(long, default_value_t = 100)]
    pub ticks: u32,
    /// Write trace output to this file instead of stdout.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

/// Arguments for `profile` subcommand.
#[derive(Debug, clap::Args)]
#[command(about = "Phase timing histogram — run N ticks and dump per-phase ns table")]
pub struct ProfileArgs {
    /// Scenario TOML file to load (stub: path is logged; engine not invoked).
    #[arg(long, default_value = "scenarios/basic_4v4.toml")]
    pub scenario: PathBuf,
    /// Number of ticks to profile.
    #[arg(long, default_value_t = 100)]
    pub ticks: u32,
}

/// Arguments for `repro` subcommand.
#[derive(Debug, clap::Args)]
#[command(about = "Capture a reproduction bundle (snapshot + causal-tree + traces) to file")]
pub struct ReproArgs {
    /// Scenario TOML file to load (stub: path is logged; engine not invoked).
    #[arg(long, default_value = "scenarios/basic_4v4.toml")]
    pub scenario: PathBuf,
    /// Number of ticks to run before capturing.
    #[arg(long, default_value_t = 100)]
    pub ticks: u32,
    /// Write the bundle to this path.
    #[arg(long, default_value = "/tmp/repro.bundle")]
    pub output: PathBuf,
    /// Replay an existing bundle (print causal-tree dump) instead of capturing.
    #[arg(long)]
    pub replay: Option<PathBuf>,
}

#[derive(Debug, Parser)]
#[command(
    about = "IMPALA V6 training (auto-configures libtorch env, builds burn-gpu, runs training)"
)]
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

#[derive(Debug, Parser)]
#[command(about = "Compile DSL sources to Rust + Python + schema hash")]
pub struct CompileDslArgs {
    /// Source directory holding `*.sim` files (recursively walked).
    #[arg(long, default_value = "assets/sim")]
    pub src: PathBuf,
    /// Destination root for generated shared Rust output. Files are written
    /// under `<out-rust>/events/` and `<out-rust>/schema.rs`.
    #[arg(long, default_value = "crates/engine_data/src")]
    pub out_rust: PathBuf,
    /// Destination root for emitted physics handlers. Files are written
    /// under `<out-physics>/` (per-rule modules + `mod.rs`). Defaults into
    /// the engine crate because emitted handlers reference
    /// `engine::cascade::*` and `engine::state::SimState`; emitting them
    /// into `engine_rules` would invert the existing dep direction
    /// (`engine` depends on `engine_rules`, not the other way around).
    /// Documented in `docs/game/feature_flow.md`.
    #[arg(long, default_value = "crates/engine_rules/src/physics")]
    pub out_physics: PathBuf,
    /// Destination root for emitted mask predicates. Same dep rationale
    /// as physics — masks reference `engine::state::SimState`.
    #[arg(long, default_value = "crates/engine_rules/src/mask")]
    pub out_mask: PathBuf,
    /// Destination root for emitted scoring tables. Scoring rows are POD
    /// data shared between CPU scorer and GPU kernel, so they live in
    /// `engine_generated` with the other shared-data emissions.
    #[arg(long, default_value = "crates/engine_data/src/scoring")]
    pub out_scoring: PathBuf,
    /// Destination root for emitted entity data (CreatureType enum,
    /// capability structs, hostility matrix). Pure data; lives in
    /// `engine_generated` next to events.
    #[arg(long, default_value = "crates/engine_data/src/entities")]
    pub out_entity: PathBuf,
    /// Destination root for emitted config structs (per-block Rust files
    /// plus the aggregator `mod.rs`). Pure data with a TOML loader; lives
    /// in `engine_generated` alongside the other shared-data emissions.
    #[arg(long, default_value = "crates/engine_data/src/config")]
    pub out_config_rust: PathBuf,
    /// Destination directory for the authored TOML default file. The
    /// compiler writes `<out-config-toml>/default.toml`; runtime callers
    /// load it via `engine_rules::config::Config::from_toml`.
    #[arg(long, default_value = "assets/config")]
    pub out_config_toml: PathBuf,
    /// Destination root for emitted enum declarations (per-enum Rust
    /// files + aggregator `mod.rs`). Pure data, no engine dependency.
    #[arg(long, default_value = "crates/engine_data/src/enums")]
    pub out_enum: PathBuf,
    /// Destination root for emitted view modules (`@lazy` inline fns +
    /// `@materialized` fold-storage structs + aggregator `ViewRegistry`).
    /// Lives under the engine crate because materialized views hold
    /// `HashMap<(AgentId, …), …>` storage that `SimState` owns.
    #[arg(long, default_value = "crates/engine_rules/src/views")]
    pub out_views: PathBuf,
    /// Destination file for the emitted canonical serial tick pipeline
    /// (`engine_rules/src/step.rs`).
    #[arg(long, default_value = "crates/engine_rules/src/step.rs")]
    pub out_step: PathBuf,
    /// Destination file for the emitted `SerialBackend` struct.
    #[arg(long, default_value = "crates/engine_rules/src/backend.rs")]
    pub out_backend: PathBuf,
    /// Destination file for the emitted `fill_all` mask-fill function.
    #[arg(long, default_value = "crates/engine_rules/src/mask_fill.rs")]
    pub out_mask_fill: PathBuf,
    /// Destination file for the emitted `with_engine_builtins` cascade
    /// registry factory.
    #[arg(long, default_value = "crates/engine_rules/src/cascade_reg.rs")]
    pub out_cascade_reg: PathBuf,
    /// Destination root for Python output. Files are written under
    /// `<out-python>/events/`.
    #[arg(long, default_value = "generated/python")]
    pub out_python: PathBuf,
    /// Destination file for the emitted `impl engine::event::EventLike for
    /// engine_data::events::Event { ... }` block. Lives in the engine crate
    /// while engine retains its engine_data regular dep (B2-deferred).
    #[arg(long, default_value = "crates/engine/src/event/event_like_impl.rs")]
    pub out_engine_event_like_impl: PathBuf,
    /// Compare the emitted artefacts against the committed output. Exit 0 if
    /// identical, exit 1 with a file-level diff otherwise. No files are
    /// written when this is set.
    #[arg(long)]
    pub check: bool,
    /// Side-channel: when set, additionally run the Compute-Graph IR
    /// (CG) pipeline (lower → synthesize_schedule(Default) →
    /// emit_cg_program) and write its [`EmittedArtifacts`] under
    /// `<dir>/src/`. Does NOT replace the legacy emit; behavior of the
    /// other outputs is unchanged.
    ///
    /// # Limitations (Task 5.1, 2026-04-29)
    ///
    /// - The CG pipeline emits ~21 ops today (9 view_fold + 12
    ///   plumbing); 0 mask/scoring/physics/spatial. Closing those is
    ///   Task 5.5 (AST coverage).
    /// - The emitted Rust files lack `Kernel` trait impls and
    ///   cross-kernel helpers (Task 5.2).
    /// - Cross-cutting modules (BindingSources, etc.) not yet emitted
    ///   (Task 5.4).
    /// - No `Cargo.toml` or `lib.rs` is synthesised — only per-kernel
    ///   files. The companion `compile-dsl-parity` subcommand reports
    ///   which pieces are missing.
    /// - Lowering deferrals (typed `LoweringError`s from the Phase 2
    ///   driver) are printed to stderr; the side-channel still emits
    ///   the best-effort program.
    #[arg(long, value_name = "DIR")]
    pub cg_emit_into: Option<PathBuf>,
}

/// Arguments for `compile-dsl-parity` subcommand.
///
/// Behavioral-parity harness for the CG pipeline. Reads CG-emitted
/// files from `--cg-out <dir>` (produced by
/// `compile-dsl --cg-emit-into`) and reports whether a CG-overlaid
/// `engine_gpu_rules` would compile.
///
/// # Limitations (Task 5.1, 2026-04-29)
///
/// - Today this only verifies that the side-channel directory is
///   structurally inhabited (correct file count, expected extensions).
///   It does NOT yet attempt to overlay-build `engine_gpu_rules` or
///   run `parity_with_cpu` — closing that is Tasks 5.2-5.7 of the
///   reframed plan.
/// - **The command is EXPECTED to FAIL today** because the CG output
///   is missing the pieces those follow-up tasks add (`Kernel` trait
///   impls, cross-kernel helpers, Cargo.toml/lib.rs, full op
///   coverage). The non-zero exit is the diagnostic the controller
///   uses to decide what to wire next.
#[derive(Debug, clap::Args)]
#[command(about = "CG-pipeline parity harness — behavioral check on side-channel output")]
pub struct CompileDslParityArgs {
    /// Side-channel directory previously populated by
    /// `compile-dsl --cg-emit-into <dir>`. The harness expects
    /// `<dir>/src/*.{rs,wgsl}` files inside.
    #[arg(long, value_name = "DIR")]
    pub cg_out: PathBuf,
}
