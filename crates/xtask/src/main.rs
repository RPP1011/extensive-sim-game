mod cli;
mod map;
mod capture;
mod train_v6;
mod compile_dsl_cmd;
mod debug_cmd;
mod trace_cmd;
mod profile_cmd;
mod repro_cmd;

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
        TaskCommand::TrainV6(cmd) => train_v6::run_train_v6(cmd),
        TaskCommand::CompileDsl(args) => compile_dsl_cmd::run_compile_dsl(args),
        TaskCommand::Debug(args) => debug_cmd::run_debug(args),
        TaskCommand::Trace(args) => trace_cmd::run_trace(args),
        TaskCommand::Profile(args) => profile_cmd::run_profile(args),
        TaskCommand::Repro(args) => repro_cmd::run_repro(args),
    }
}
