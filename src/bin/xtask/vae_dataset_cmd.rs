//! CLI handler for the VAE dataset pipeline.

use std::process::ExitCode;

use bevy_game::headless_campaign::config::CampaignConfig;
use bevy_game::headless_campaign::llm::LlmConfig;
use bevy_game::headless_campaign::vae_dataset::{self, VaeDatasetConfig};

use crate::cli::VaeDatasetArgs;

pub fn run_vae_dataset(args: VaeDatasetArgs) -> ExitCode {
    let campaign_config = if let Some(ref path) = args.config {
        match CampaignConfig::load_from_toml(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error loading config: {}", e);
                return ExitCode::from(1);
            }
        }
    } else {
        CampaignConfig::default()
    };

    let llm_config = if args.no_llm {
        None
    } else {
        let cfg = LlmConfig {
            base_url: args.llm_url.clone(),
            model: args.llm_model.clone(),
            candidates: args.candidates as usize,
            ..Default::default()
        };
        Some(cfg)
    };

    let config = VaeDatasetConfig {
        campaigns: args.campaigns,
        max_ticks: args.max_ticks,
        threads: args.threads,
        base_seed: args.seed,
        campaign_config,
        contexts_path: format!("{}/vae_contexts.jsonl", args.output_dir),
        raw_path: format!("{}/vae_raw.jsonl", args.output_dir),
        dataset_path: format!("{}/vae_dataset.jsonl", args.output_dir),
        llm_config,
        llm_workers: args.workers,
        llm_candidates: args.candidates as usize,
        include_procedural: !args.no_procedural,
    };

    if args.sweep_only {
        vae_dataset::sweep_campaigns(&config);
    } else if args.extract_only {
        // Load contexts from file
        let contexts = load_contexts(&config.contexts_path);
        vae_dataset::extract_dataset(&config, &contexts);
    } else {
        // Full pipeline
        vae_dataset::run_full_pipeline(&config);
    }

    ExitCode::SUCCESS
}

fn load_contexts(path: &str) -> Vec<vae_dataset::TriggerContext> {
    match std::fs::read_to_string(path) {
        Ok(contents) => contents.lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect(),
        Err(e) => {
            eprintln!("Failed to load contexts from {}: {}", path, e);
            vec![]
        }
    }
}
