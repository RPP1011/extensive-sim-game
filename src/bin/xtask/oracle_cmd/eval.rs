use std::process::ExitCode;

use super::collect_toml_paths;

pub fn run_oracle_eval(args: crate::cli::OracleEvalArgs) -> ExitCode {
    use bevy_game::ai::core::decision_log::{run_with_oracle_depth, OracleSummary};
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state};
    let depth = args.depth;

    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found in {}.", args.path.display());
        return ExitCode::from(1);
    }

    let mut summaries: Vec<OracleSummary> = Vec::new();

    for toml_path in &paths {
        let scenario_file = match load_scenario_file(toml_path) {
            Ok(f) => f,
            Err(err) => {
                eprintln!("{err}");
                continue;
            }
        };

        let log_path = args.output_dir.as_ref().map(|dir| {
            std::fs::create_dir_all(dir).ok();
            dir.join(format!(
                "{}.jsonl",
                toml_path.file_stem().unwrap().to_string_lossy()
            ))
        });

        eprintln!("Running oracle on {}...", scenario_file.scenario.name);

        let cfg = &scenario_file.scenario;
        let (sim, squad_ai) = run_scenario_to_state(cfg);
        let summary = run_with_oracle_depth(sim, squad_ai, &cfg.name, cfg.max_ticks, log_path.as_deref(), depth);

        println!(
            "{:<30} outcome={:<8} decisions={:<5} match={:.1}%  avg_delta={:.1}  max_delta={:.1}  wrong_target={} wrong_ability={} wrong_pos={}",
            summary.scenario_name,
            summary.outcome,
            summary.total_decisions,
            summary.match_rate * 100.0,
            summary.avg_score_delta,
            summary.max_score_delta,
            summary.wrong_target_count,
            summary.wrong_ability_count,
            summary.wrong_position_count,
        );

        summaries.push(summary);
    }

    if summaries.len() > 1 {
        print_eval_aggregate(&summaries);
    }

    ExitCode::SUCCESS
}

fn print_eval_aggregate(summaries: &[bevy_game::ai::core::decision_log::OracleSummary]) {
    let total_decisions: u32 = summaries.iter().map(|s| s.total_decisions).sum();
    let total_matched: u32 = summaries.iter().map(|s| s.matched_top1_count).sum();
    let overall_match = if total_decisions > 0 {
        total_matched as f64 / total_decisions as f64
    } else {
        0.0
    };
    let overall_avg_delta: f64 = if total_decisions > 0 {
        summaries
            .iter()
            .map(|s| s.avg_score_delta * s.total_decisions as f64)
            .sum::<f64>()
            / total_decisions as f64
    } else {
        0.0
    };

    let wins = summaries.iter().filter(|s| s.outcome == "Victory").count();
    let losses = summaries.iter().filter(|s| s.outcome == "Defeat").count();

    println!("\n--- Aggregate ---");
    println!("Scenarios: {} ({wins}W / {losses}L)", summaries.len());
    println!("Total decisions: {total_decisions}");
    println!("Overall oracle match: {:.1}%", overall_match * 100.0);
    println!("Overall avg score delta: {:.1}", overall_avg_delta);

    for (label, filter_outcome) in [("Win", "Victory"), ("Loss", "Defeat")] {
        let filtered: Vec<_> = summaries.iter().filter(|s| s.outcome == filter_outcome).collect();
        if !filtered.is_empty() {
            let dec: u32 = filtered.iter().map(|s| s.total_decisions).sum();
            let mat: u32 = filtered.iter().map(|s| s.matched_top1_count).sum();
            let rate = if dec > 0 { mat as f64 / dec as f64 } else { 0.0 };
            println!("  {label} scenarios match: {:.1}%", rate * 100.0);
        }
    }
}

pub fn run_oracle_play(args: crate::cli::OraclePlayArgs) -> ExitCode {
    use bevy_game::ai::core::decision_log::run_oracle_played_depth;
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state};
    let depth = args.depth;

    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found in {}.", args.path.display());
        return ExitCode::from(1);
    }

    let mut wins = 0u32;
    let mut losses = 0u32;
    let mut timeouts = 0u32;

    for toml_path in &paths {
        let scenario_file = match load_scenario_file(toml_path) {
            Ok(f) => f,
            Err(err) => {
                eprintln!("{err}");
                continue;
            }
        };

        eprintln!("Oracle playing {}...", scenario_file.scenario.name);

        let cfg = &scenario_file.scenario;
        let (sim, squad_ai) = run_scenario_to_state(cfg);
        let result = run_oracle_played_depth(sim, squad_ai, &cfg.name, cfg.max_ticks, depth);

        let tag = match result.outcome.as_str() {
            "Victory" => { wins += 1; "WIN " }
            "Defeat" => { losses += 1; "LOSS" }
            _ => { timeouts += 1; "TIME" }
        };

        println!(
            "[{tag}] {:<30} tick={:<5} heroes={} enemies={}",
            result.scenario_name, result.total_ticks, result.heroes_alive, result.enemies_alive,
        );
    }

    let total = wins + losses + timeouts;
    if total > 1 {
        println!(
            "\n--- Aggregate ---\nScenarios: {total}  Wins: {wins}  Losses: {losses}  Timeouts: {timeouts}  Win rate: {:.1}%",
            if total > 0 { wins as f64 / total as f64 * 100.0 } else { 0.0 }
        );
    }

    ExitCode::SUCCESS
}
