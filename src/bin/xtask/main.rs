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
mod mcts_bootstrap_cmd;
mod vae_dataset_cmd;

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
        TaskCommand::MctsBootstrap(args) => mcts_bootstrap_cmd::run_mcts_bootstrap(args),
        TaskCommand::HeuristicBc(args) => {
            let campaign_config = if let Some(ref path) = args.config {
                match bevy_game::headless_campaign::config::CampaignConfig::load_from_toml(path) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Error loading config: {}", e);
                        return ExitCode::from(1);
                    }
                }
            } else {
                bevy_game::headless_campaign::config::CampaignConfig::default()
            };

            let config = bevy_game::headless_campaign::heuristic_bc::BcConfig {
                campaigns: args.campaigns,
                max_ticks: args.max_ticks,
                threads: args.threads,
                base_seed: args.seed,
                report_interval: args.report_interval,
                output_path: args.output,
                campaign_config,
                sample_rate: args.sample_rate,
            };
            bevy_game::headless_campaign::heuristic_bc::run_bc_generation(&config);
            ExitCode::SUCCESS
        }
        TaskCommand::BfsExplore(args) => {
            let campaign_config = if let Some(ref path) = args.config {
                match bevy_game::headless_campaign::config::CampaignConfig::load_from_toml(path) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Error loading config: {}", e);
                        return ExitCode::from(1);
                    }
                }
            } else {
                bevy_game::headless_campaign::config::CampaignConfig::default()
            };

            let llm_config = if args.llm {
                let cfg = bevy_game::headless_campaign::llm::LlmConfig {
                    base_url: args.llm_url.clone(),
                    model: args.llm_model.clone(),
                    candidates: args.llm_candidates,
                    ..Default::default()
                };
                if bevy_game::headless_campaign::llm::check_ollama(&cfg) {
                    eprintln!("LLM: connected to {} (model: {})", cfg.base_url, cfg.model);
                    Some(cfg)
                } else {
                    eprintln!("LLM: Ollama not reachable, falling back to templates");
                    None
                }
            } else {
                None
            };

            let vae_model = if let Some(ref path) = args.vae_model {
                match bevy_game::headless_campaign::vae_inference::ContentVaeWeights::load(path) {
                    Ok(w) => {
                        eprintln!("VAE: loaded from {} ({:?})", path, w);
                        Some(std::sync::Arc::new(w))
                    }
                    Err(e) => {
                        eprintln!("VAE: failed to load {}: {}", path, e);
                        None
                    }
                }
            } else {
                None
            };

            let config = bevy_game::headless_campaign::bfs_explore::BfsConfig {
                max_waves: args.max_waves,
                ticks_per_branch: args.ticks_per_branch,
                clusters_per_wave: args.clusters,
                initial_roots: args.initial_roots,
                trajectory_max_ticks: args.trajectory_ticks,
                root_sample_interval: args.root_interval,
                campaign_config,
                base_seed: args.seed,
                threads: args.threads,
                output_path: args.output,
                llm_config,
                vae_model,
            };
            bevy_game::headless_campaign::bfs_explore::run_bfs_exploration(&config);
            ExitCode::SUCCESS
        }
        TaskCommand::CampaignFuzz(args) => {
            let config = bevy_game::headless_campaign::fuzz::FuzzConfig {
                campaigns: args.campaigns,
                max_ticks: args.max_ticks,
                threads: args.threads,
                base_seed: args.seed,
                mutation_strength: args.mutation_strength,
                output_path: args.output,
                report_interval: args.report_interval,
                random_action_ratio: args.random_action_ratio,
                ..Default::default()
            };
            bevy_game::headless_campaign::fuzz::run_fuzz(&config);
            ExitCode::SUCCESS
        }
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
        TaskCommand::VaeDataset(args) => vae_dataset_cmd::run_vae_dataset(args),
        TaskCommand::VaeExtractSlots(args) => {
            bevy_game::headless_campaign::vae_dataset::extract_slots_from_jsonl(&args.input);
            ExitCode::SUCCESS
        }
        TaskCommand::VaeGtDataset(args) => {
            let config = bevy_game::headless_campaign::vae_gt_dataset::GtDatasetConfig {
                campaigns: args.campaigns,
                max_ticks: args.max_ticks,
                threads: args.threads,
                base_seed: args.seed,
                samples_per_context: args.samples_per_context,
                output_path: args.output,
            };
            bevy_game::headless_campaign::vae_gt_dataset::run_gt_pipeline(&config);
            ExitCode::SUCCESS
        }
    }
}
