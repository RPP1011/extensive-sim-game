//! CLI handler for the `mcts-bootstrap` subcommand.

use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::collections::HashMap;

use crate::cli::MctsBootstrapArgs;

pub fn run_mcts_bootstrap(args: MctsBootstrapArgs) -> ExitCode {
    use bevy_game::headless_campaign::config::CampaignConfig;
    use bevy_game::headless_campaign::mcts::export::{export_mcts_campaign, write_samples_jsonl};
    use bevy_game::headless_campaign::mcts::MctsConfig;
    use bevy_game::headless_campaign::state::CampaignOutcome;

    // Load campaign config
    let campaign_config = if let Some(ref path) = args.config {
        match CampaignConfig::load_from_toml(path) {
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
        CampaignConfig::default()
    };

    let mcts_config = MctsConfig {
        simulations_per_move: args.simulations,
        rollout_horizon_ticks: args.rollout_horizon,
        decision_interval_ticks: args.decision_interval,
        max_campaign_ticks: args.max_ticks,
        ..Default::default()
    };

    let threads = if args.threads == 0 {
        rayon::current_num_threads()
    } else {
        args.threads
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("Failed to build rayon pool");

    eprintln!("=== MCTS Bootstrap ===");
    eprintln!("Campaigns: {}", args.campaigns);
    eprintln!("Simulations/decision: {}", args.simulations);
    eprintln!("Max ticks: {}", args.max_ticks);
    eprintln!("Rollout horizon: {}", args.rollout_horizon);
    eprintln!("Decision interval: {}", args.decision_interval);
    eprintln!("Threads: {}", threads);
    eprintln!("Output: {}", args.output.display());

    // Ensure parent directory exists
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Truncate output file if it exists
    if args.output.exists() {
        std::fs::write(&args.output, "").ok();
    }

    let completed = AtomicU64::new(0);
    let total_samples = AtomicU64::new(0);
    let victories = AtomicU64::new(0);
    let defeats = AtomicU64::new(0);
    let timeouts = AtomicU64::new(0);
    let writer_mutex = Mutex::new(args.output.clone());
    let choice_counts: Mutex<HashMap<String, u64>> = Mutex::new(HashMap::new());

    let t0 = std::time::Instant::now();

    pool.install(|| {
        use rayon::prelude::*;

        let seeds: Vec<u64> = (0..args.campaigns)
            .map(|i| args.seed.wrapping_add(i))
            .collect();

        seeds.par_iter().for_each(|&seed| {
            let (outcome, samples) =
                export_mcts_campaign(seed, &campaign_config, &mcts_config);

            let n_samples = samples.len() as u64;

            // Track starting choice from tick-0 sample
            if let Some(first) = samples.first() {
                if first.tick == 0 && !first.starting_choice_name.is_empty() {
                    let mut counts = choice_counts.lock().unwrap();
                    *counts.entry(first.starting_choice_name.clone()).or_insert(0) += 1;
                }
            }

            // Write samples under mutex
            {
                let path = writer_mutex.lock().unwrap();
                if let Err(e) = write_samples_jsonl(&samples, &path) {
                    eprintln!("Warning: failed to write samples for seed {}: {}", seed, e);
                }
            }

            total_samples.fetch_add(n_samples, Ordering::Relaxed);
            match outcome {
                CampaignOutcome::Victory => {
                    victories.fetch_add(1, Ordering::Relaxed);
                }
                CampaignOutcome::Defeat => {
                    defeats.fetch_add(1, Ordering::Relaxed);
                }
                CampaignOutcome::Timeout => {
                    timeouts.fetch_add(1, Ordering::Relaxed);
                }
            }

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 10 == 0 || done == args.campaigns {
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = done as f64 / elapsed;
                let samples_so_far = total_samples.load(Ordering::Relaxed);
                let samples_rate = samples_so_far as f64 / elapsed;
                eprintln!(
                    "[{}/{}] samples={} rate={:.1} campaigns/s  {:.0} samples/s  elapsed={:.1}s",
                    done, args.campaigns, samples_so_far, rate, samples_rate, elapsed,
                );
            }
        });
    });

    let elapsed = t0.elapsed().as_secs_f64();
    let final_samples = total_samples.load(Ordering::Relaxed);
    let final_victories = victories.load(Ordering::Relaxed);
    let final_defeats = defeats.load(Ordering::Relaxed);
    let final_timeouts = timeouts.load(Ordering::Relaxed);

    eprintln!("\n=== MCTS Bootstrap Complete ===");
    eprintln!("Total campaigns: {}", args.campaigns);
    eprintln!("Total samples: {}", final_samples);
    eprintln!(
        "Outcomes: {} victories, {} defeats, {} timeouts",
        final_victories, final_defeats, final_timeouts
    );
    eprintln!(
        "Win rate: {:.1}%",
        if args.campaigns > 0 {
            final_victories as f64 / args.campaigns as f64 * 100.0
        } else {
            0.0
        }
    );
    eprintln!(
        "Samples/campaign: {:.1}",
        if args.campaigns > 0 {
            final_samples as f64 / args.campaigns as f64
        } else {
            0.0
        }
    );
    eprintln!(
        "Rate: {:.0} campaigns/s, {:.0} samples/s",
        args.campaigns as f64 / elapsed,
        final_samples as f64 / elapsed,
    );
    eprintln!("Total time: {:.1}s", elapsed);

    // Print starting choice distribution
    let counts = choice_counts.lock().unwrap();
    if !counts.is_empty() {
        eprintln!("\nStarting choice distribution:");
        let mut sorted: Vec<_> = counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (name, count) in sorted {
            eprintln!(
                "  {}: {} ({:.1}%)",
                name,
                count,
                *count as f64 / args.campaigns as f64 * 100.0
            );
        }
    }

    eprintln!("\nOutput written to: {}", args.output.display());

    ExitCode::SUCCESS
}
