mod cli;
mod map;
mod capture;
mod scenario_cmd;
mod oracle_cmd;
mod train_v6;
mod roomgen_cmd;
mod model_cmd;
mod content_gen_cmd;
mod ascii_gen_cmd;

use std::process::ExitCode;

use clap::Parser;
use cli::*;

fn main() -> ExitCode {
    let args = Args::parse();
    match args.command {
        TaskCommand::Map(cmd) => match cmd.command {
            MapSubcommand::Voronoi(voronoi) => map::run_map_voronoi(voronoi),
        },
        TaskCommand::Capture(cmd) => match cmd.command {
            CaptureSubcommand::Windows(windows) => capture::run_capture_windows(windows),
            CaptureSubcommand::Dedupe(dedupe) => capture::run_capture_dedupe(dedupe),
        },
        TaskCommand::Scenario(cmd) => scenario_cmd::run_scenario_cmd(cmd),
        TaskCommand::TrainV6(cmd) => train_v6::run_train_v6(cmd),
        TaskCommand::Roomgen(cmd) => roomgen_cmd::run_roomgen_cmd(cmd),
        TaskCommand::Model(cmd) => model_cmd::run_model_cmd(cmd),
        TaskCommand::ContentGen(cmd) => content_gen_cmd::run_content_gen_cmd(cmd),
        TaskCommand::AsciiGen(cmd) => ascii_gen_cmd::run_ascii_gen_cmd(cmd),
    }
}
