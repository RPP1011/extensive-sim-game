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
        TaskCommand::CampaignBatch(args) => {
            let campaign_config = if let Some(ref path) = args.config {
                match bevy_game::headless_campaign::config::CampaignConfig::load_from_toml(path) {
                    Ok(c) => {
                        eprintln!("Loaded campaign config from {}", path.display());
                        c
                    }
                    Err(e) => {
                        eprintln!("Error loading config: {}", e);
                        return ExitCode::from(1);
                    }
                }
            } else {
                bevy_game::headless_campaign::config::CampaignConfig::default()
            };

            let config = bevy_game::headless_campaign::batch::BatchConfig {
                target_successes: args.target,
                max_ticks: args.max_ticks,
                threads: args.threads,
                base_seed: args.seed,
                report_interval: args.report_interval,
                record_traces: args.record_traces,
                trace_snapshot_interval: args.trace_snapshot_interval,
                trace_output_dir: args.trace_output_dir,
                campaign_config,
            };
            bevy_game::headless_campaign::batch::run_batch(&config);
            ExitCode::SUCCESS
        }
    }
}
