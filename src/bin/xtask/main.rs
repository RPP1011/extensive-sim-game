mod cli;
mod map;
mod env_art;
mod capture;
mod ralph;
mod scenario_cmd;
mod oracle_cmd;

use std::process::ExitCode;

use clap::Parser;
use cli::*;

fn main() -> ExitCode {
    let args = Args::parse();
    match args.command {
        TaskCommand::Map(cmd) => match cmd.command {
            MapSubcommand::Gemini(gemini) => map::run_map_gemini(gemini),
            MapSubcommand::Voronoi(voronoi) => map::run_map_voronoi(voronoi),
            MapSubcommand::EnvArt(env_art_cmd) => env_art::run_map_env_art(env_art_cmd),
        },
        TaskCommand::Capture(cmd) => match cmd.command {
            CaptureSubcommand::Windows(windows) => capture::run_capture_windows(windows),
            CaptureSubcommand::Dedupe(dedupe) => capture::run_capture_dedupe(dedupe),
        },
        TaskCommand::Ralph(cmd) => match cmd.command {
            RalphSubcommand::Status(args) => ralph::run_ralph_status(args),
        },
        TaskCommand::Scenario(cmd) => scenario_cmd::run_scenario_cmd(cmd),
    }
}
