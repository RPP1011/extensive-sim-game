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
