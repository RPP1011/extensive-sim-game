//! Evaluation logic for transformer RL.

use std::process::ExitCode;
use rayon::prelude::*;

use super::collect_toml_paths;
use super::transformer_rl::{Policy, run_rl_episode};

pub(crate) fn run_eval(args: crate::cli::TransformerRlEvalArgs) -> ExitCode {
    use game::ai::core::ability_transformer::tokenizer::AbilityTokenizer;
    use game::scenario::{load_scenario_file, run_scenario_to_state_with_room};

    let policy = match Policy::load(&args.weights) {
        Ok(p) => p,
        Err(e) => { eprintln!("Failed to load weights: {e}"); return ExitCode::from(1); }
    };

    let tokenizer = AbilityTokenizer::new();
    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found.");
        return ExitCode::from(1);
    }

    let scenarios: Vec<_> = paths.iter().filter_map(|p| {
        match load_scenario_file(p) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("{e}"); None }
        }
    }).collect();

    // Load embedding registry if provided
    let registry = if let Some(ref reg_path) = args.embedding_registry {
        match game::ai::core::ability_transformer::EmbeddingRegistry::from_file(
            reg_path.to_str().unwrap_or(""),
        ) {
            Ok(r) => {
                eprintln!("Loaded embedding registry: {} abilities, hash={}",
                    r.len(), r.model_hash);
                Some(r)
            }
            Err(e) => { eprintln!("Failed to load registry: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };

    // Load enemy policy for self-play eval
    let enemy_policy: Option<Policy> = if let Some(ref ew_path) = args.enemy_weights {
        match Policy::load(ew_path) {
            Ok(p) => {
                eprintln!("Loaded enemy policy from {}", ew_path.display());
                Some(p)
            }
            Err(e) => { eprintln!("Failed to load enemy policy: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };
    let enemy_registry = if let Some(ref reg_path) = args.enemy_registry {
        match game::ai::core::ability_transformer::EmbeddingRegistry::from_file(
            reg_path.to_str().unwrap_or(""),
        ) {
            Ok(r) => {
                eprintln!("Loaded enemy embedding registry: {} abilities", r.len());
                Some(r)
            }
            Err(e) => { eprintln!("Failed to load enemy registry: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };

    let policy_ref = &policy;
    let tokenizer_ref = &tokenizer;
    let registry_ref = registry.as_ref();
    let enemy_policy_ref = enemy_policy.as_ref();
    let enemy_registry_ref = enemy_registry.as_ref();
    let max_ticks_override = args.max_ticks;

    let results: Vec<(String, super::transformer_rl::RlEpisode)> = scenarios.par_iter().map(|scenario_file| {
        let cfg = &scenario_file.scenario;
        let max_ticks = max_ticks_override.unwrap_or(cfg.max_ticks);

        let (sim, squad_ai, grid_nav) = {
            let (s, ai, nav) = run_scenario_to_state_with_room(cfg);
            (s, ai, Some(nav))
        };

        let episode = run_rl_episode(
            sim, squad_ai, &cfg.name, max_ticks,
            policy_ref, tokenizer_ref, 0.01, 42, 1,
            grid_nav,
            registry_ref,
            enemy_policy_ref,
            enemy_registry_ref,
            cfg.objective.as_ref(),
            cfg.action_mask.as_deref(),
        );
        (cfg.name.clone(), episode)
    }).collect();

    let mut wins = 0u32;
    let mut losses = 0u32;
    let mut timeouts = 0u32;

    for (name, episode) in &results {
        let tag = match episode.outcome.as_str() {
            "Victory" => { wins += 1; "WIN " }
            "Defeat" => { losses += 1; "LOSS" }
            _ => { timeouts += 1; "TIME" }
        };
        println!("[{tag}] {:<30} tick={:<5} reward={:.2}", name, episode.ticks, episode.reward);
    }

    let total = wins + losses + timeouts;
    if total > 1 {
        println!("\n--- Aggregate ---");
        println!("Scenarios: {total}  Wins: {wins}  Losses: {losses}  Timeouts: {timeouts}  Win rate: {:.1}%",
            if total > 0 { wins as f64 / total as f64 * 100.0 } else { 0.0 });
    }

    ExitCode::SUCCESS
}
