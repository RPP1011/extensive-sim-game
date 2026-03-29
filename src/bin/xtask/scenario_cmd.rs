use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use super::cli::{ScenarioBenchArgs, ScenarioCommand, ScenarioGenerateArgs, ScenarioRunArgs, ScenarioSubcommand};

pub fn run_scenario_cmd(cmd: ScenarioCommand) -> ExitCode {
    match cmd.sub {
        ScenarioSubcommand::Run(args) => run_scenario_run(args),
        ScenarioSubcommand::Bench(args) => run_scenario_bench(args),
        ScenarioSubcommand::Oracle(args) => super::oracle_cmd::run_oracle_cmd(args),
        ScenarioSubcommand::Generate(args) => run_scenario_generate(args),
    }
}

fn run_scenario_generate(args: ScenarioGenerateArgs) -> ExitCode {
    use bevy_game::scenario::gen::{GenConfig, generate, write_scenarios};

    let config = GenConfig {
        seed: args.seed,
        seed_variants: args.seed_variants,
        include_synergy_pairs: !args.no_synergy,
        include_stress_archetypes: !args.no_stress,
        include_difficulty_ladders: !args.no_ladders,
        include_room_aware: !args.no_room_aware,
        include_size_spectrum: !args.no_sizes,
        include_hero_vs_hero: true,
        hvh_count: 200,
        extra_random: args.extra_random,
        verbose: args.verbose,
    };

    println!("Generating scenarios (seed={}, variants={})...", config.seed, config.seed_variants);

    let scenarios = generate(&config);

    match write_scenarios(&scenarios, &args.output) {
        Ok(count) => {
            println!("Wrote {count} scenarios to {}", args.output.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::from(1)
        }
    }
}

fn print_unit_stats_table(result: &bevy_game::scenario::ScenarioResult) {
    println!(
        "  {:>5}  {:<6} {:<10} {:>10}  {:>8}  {:>8}  {:>8}  {:>8}  {:>4}  {:>5}  {:>9}",
        "Unit", "Team", "Class", "HP", "Dmg Out", "Dmg In", "Healing", "Shields", "CC", "Kills", "Abilities"
    );

    for s in &result.unit_stats {
        println!(
            "  {:>5}  {:<6} {:<10} {:>4}/{:<5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>4}  {:>5}  {:>9}",
            s.unit_id,
            s.team,
            s.template,
            s.final_hp.max(0),
            s.max_hp,
            s.damage_dealt,
            s.damage_taken,
            s.healing_done,
            s.shield_received,
            s.cc_applied_count,
            s.kills,
            s.abilities_used,
        );
    }

    // Per-ability breakdown (only for units that have ability stats).
    let has_abilities = result.unit_stats.iter().any(|s| !s.ability_stats.is_empty());
    if has_abilities {
        println!("\n  Ability Breakdown:");
        for s in &result.unit_stats {
            if s.ability_stats.is_empty() {
                continue;
            }
            println!("  Unit {} ({}):", s.unit_id, s.template);
            for ab in &s.ability_stats {
                let cc_str = if ab.cc_applied_count > 0 {
                    format!("cc={} ({}ms)", ab.cc_applied_count, ab.cc_duration_ms)
                } else {
                    "cc=0".to_string()
                };
                println!(
                    "    {:<18} x{:<3} dmg={:<6} heal={:<6} shield={:<5} {}",
                    ab.ability_name,
                    ab.times_used,
                    ab.damage_dealt,
                    ab.healing_done,
                    ab.shield_granted,
                    cc_str,
                );
            }
        }
    }
    println!();
}

fn run_scenario_run(args: ScenarioRunArgs) -> ExitCode {
    use bevy_game::scenario::{check_assertions, load_scenario_file, run_scenario, ScenarioResult};

    let path = &args.path;

    if path.is_dir() {
        // Glob all *.toml files in the directory.
        let entries = match std::fs::read_dir(path) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("Failed to read directory {}: {err}", path.display());
                return ExitCode::from(1);
            }
        };

        let mut toml_paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("toml"))
            .collect();
        toml_paths.sort();

        if toml_paths.is_empty() {
            eprintln!("No *.toml files found in {}.", path.display());
            return ExitCode::from(1);
        }

        let mut results: Vec<ScenarioResult> = Vec::new();
        let mut all_passed = true;

        for toml_path in &toml_paths {
            let scenario_file = match load_scenario_file(toml_path) {
                Ok(f) => f,
                Err(err) => {
                    eprintln!("{err}");
                    all_passed = false;
                    continue;
                }
            };

            let mut result = run_scenario(&scenario_file.scenario);
            if let Some(ref asserts) = scenario_file.assert {
                let assertion_results = check_assertions(&result, asserts);
                result.passed = assertion_results.iter().all(|a| a.passed);
                result.assertions = assertion_results;
            }

            // Print summary row.
            let status = if result.passed { "PASS" } else { "FAIL" };
            println!(
                "[{status}] {} | outcome={} tick={} heroes={} enemies={}",
                result.scenario_name,
                result.outcome,
                result.tick,
                result.final_hero_count,
                result.final_enemy_count,
            );

            for ar in &result.assertions {
                let mark = if ar.passed { "  ok" } else { "  FAIL" };
                println!(
                    "  {} {}: got '{}' expected '{}'",
                    mark, ar.name, ar.value, ar.expected
                );
            }

            if args.verbose {
                print_unit_stats_table(&result);
            }

            if !result.passed {
                all_passed = false;
            }
            results.push(result);
        }

        // Optionally write JSON output.
        if let Some(out_path) = &args.output {
            match serde_json::to_string_pretty(&results) {
                Ok(json) => {
                    if let Err(err) = fs::write(out_path, format!("{json}\n")) {
                        eprintln!("Failed to write output {}: {err}", out_path.display());
                        return ExitCode::from(1);
                    }
                }
                Err(err) => {
                    eprintln!("Failed to serialize results: {err}");
                    return ExitCode::from(1);
                }
            }
        }

        let total = results.len();
        let passed_count = results.iter().filter(|r| r.passed).count();
        println!("\n{passed_count}/{total} scenarios passed.");

        if all_passed {
            ExitCode::SUCCESS
        } else {
            std::process::exit(1);
        }
    } else {
        // Single file.
        let scenario_file = match load_scenario_file(path) {
            Ok(f) => f,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };

        let mut result = run_scenario(&scenario_file.scenario);
        if let Some(ref asserts) = scenario_file.assert {
            let assertion_results = check_assertions(&result, asserts);
            result.passed = assertion_results.iter().all(|a| a.passed);
            result.assertions = assertion_results;
        }

        if args.verbose {
            print_unit_stats_table(&result);
        }

        let json = match serde_json::to_string_pretty(&result) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("Failed to serialize result: {err}");
                return ExitCode::from(1);
            }
        };

        if let Some(out_path) = &args.output {
            if let Err(err) = fs::write(out_path, format!("{json}\n")) {
                eprintln!("Failed to write output {}: {err}", out_path.display());
                return ExitCode::from(1);
            }
        } else {
            println!("{json}");
        }

        if result.passed {
            ExitCode::SUCCESS
        } else {
            std::process::exit(1);
        }
    }
}

fn run_scenario_bench(args: ScenarioBenchArgs) -> ExitCode {
    use bevy_game::scenario::{load_scenario_file, run_scenario};
    use rayon::prelude::*;
    use std::time::Instant;

    if args.profile {
        return run_scenario_profile(args);
    }

    let scenario_file = match load_scenario_file(&args.path) {
        Ok(f) => f,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let cfg = &scenario_file.scenario;
    let unit_count = cfg.hero_count + cfg.enemy_count;
    let n = args.iterations;
    let threads = if args.threads == 0 {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    } else {
        args.threads
    };

    println!(
        "Benchmarking: {} ({} units, {} ticks max)",
        cfg.name, unit_count, cfg.max_ticks
    );
    println!("Iterations: {n}, Threads: {threads}\n");

    // --- Sequential ---
    println!("Sequential...");
    let start = Instant::now();
    let mut total_ticks_seq: u64 = 0;
    for _ in 0..n {
        let result = run_scenario(cfg);
        total_ticks_seq += result.tick;
    }
    let elapsed_seq = start.elapsed();
    let per_scenario_seq = elapsed_seq / n;
    let per_minute_seq = if elapsed_seq.as_secs_f64() > 0.0 {
        (n as f64 / elapsed_seq.as_secs_f64() * 60.0) as u64
    } else {
        0
    };
    let avg_ticks_seq = total_ticks_seq / n as u64;

    println!("  Total:         {:.3}s", elapsed_seq.as_secs_f64());
    println!("  Per scenario:  {:.3}ms", per_scenario_seq.as_secs_f64() * 1000.0);
    println!("  Throughput:    {per_minute_seq}/min");
    println!("  Avg ticks:     {avg_ticks_seq}\n");

    // --- Parallel ---
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon pool");

    println!("Parallel ({threads} threads)...");
    let start = Instant::now();
    let total_ticks_par: u64 = pool.install(|| {
        (0..n)
            .into_par_iter()
            .map(|_| {
                let result = run_scenario(cfg);
                result.tick
            })
            .sum()
    });
    let elapsed_par = start.elapsed();
    let per_scenario_par = elapsed_par / n;
    let per_minute_par = if elapsed_par.as_secs_f64() > 0.0 {
        (n as f64 / elapsed_par.as_secs_f64() * 60.0) as u64
    } else {
        0
    };
    let avg_ticks_par = total_ticks_par / n as u64;
    let speedup = elapsed_seq.as_secs_f64() / elapsed_par.as_secs_f64();

    println!("  Total:         {:.3}s", elapsed_par.as_secs_f64());
    println!("  Per scenario:  {:.3}ms (wall / n)", per_scenario_par.as_secs_f64() * 1000.0);
    println!("  Throughput:    {per_minute_par}/min");
    println!("  Avg ticks:     {avg_ticks_par}");
    println!("  Speedup:       {speedup:.2}x vs sequential");

    ExitCode::SUCCESS
}

fn run_scenario_profile(args: ScenarioBenchArgs) -> ExitCode {
    use bevy_game::ai::core::{step, FIXED_TICK_MS};
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state};
    use std::time::Instant;

    let scenario_file = match load_scenario_file(&args.path) {
        Ok(f) => f,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let cfg = &scenario_file.scenario;
    let n = args.iterations;
    println!(
        "Profiling: {} ({} heroes + {} enemies, {} ticks max)",
        cfg.name, cfg.hero_count, cfg.enemy_count, cfg.max_ticks
    );
    println!("Iterations: {n}\n");

    let mut total_intent_us = 0u64;
    let mut total_step_us = 0u64;
    let mut total_events_us = 0u64;
    let mut total_ticks = 0u64;
    let mut total_scenarios = 0u64;

    for _ in 0..n {
        let (mut sim, mut squad_state) = run_scenario_to_state(cfg);

        let mut iter_intent_us = 0u64;
        let mut iter_step_us = 0u64;
        let mut iter_events_us = 0u64;
        let mut ticks_run = 0u64;

        for _ in 0..cfg.max_ticks {
            // Phase 1: Intent generation (AI decisions)
            let t0 = Instant::now();
            let all_intents =
                bevy_game::ai::squad::generate_intents(&sim, &mut squad_state, FIXED_TICK_MS);
            iter_intent_us += t0.elapsed().as_micros() as u64;

            // Phase 2: Simulation step (physics + effects)
            let t1 = Instant::now();
            let (new_sim, _events) = step(sim, &all_intents, FIXED_TICK_MS);
            iter_step_us += t1.elapsed().as_micros() as u64;

            // Phase 3: Event processing (stats bookkeeping)
            let t2 = Instant::now();
            let _hero_dead = new_sim.units.iter().any(|u| {
                u.team == bevy_game::ai::core::Team::Hero && u.hp <= 0
            });
            let _enemy_alive = new_sim.units.iter().any(|u| {
                u.team == bevy_game::ai::core::Team::Enemy && u.hp > 0
            });
            iter_events_us += t2.elapsed().as_micros() as u64;

            sim = new_sim;
            ticks_run += 1;

            // Check for end conditions
            let all_heroes_dead = !sim.units.iter().any(|u| {
                u.team == bevy_game::ai::core::Team::Hero && u.hp > 0
            });
            let all_enemies_dead = !sim.units.iter().any(|u| {
                u.team == bevy_game::ai::core::Team::Enemy && u.hp > 0
            });
            if all_heroes_dead || all_enemies_dead {
                break;
            }
        }

        total_intent_us += iter_intent_us;
        total_step_us += iter_step_us;
        total_events_us += iter_events_us;
        total_ticks += ticks_run;
        total_scenarios += 1;
    }

    let total_us = total_intent_us + total_step_us + total_events_us;
    let avg_ticks = total_ticks / total_scenarios;

    println!("Phase breakdown ({total_scenarios} runs, avg {avg_ticks} ticks):\n");
    println!("  {:<25} {:>10} µs  {:>5.1}%",
        "Intent generation (AI)",
        total_intent_us,
        total_intent_us as f64 / total_us as f64 * 100.0);
    println!("  {:<25} {:>10} µs  {:>5.1}%",
        "Sim step (physics)",
        total_step_us,
        total_step_us as f64 / total_us as f64 * 100.0);
    println!("  {:<25} {:>10} µs  {:>5.1}%",
        "Event processing",
        total_events_us,
        total_events_us as f64 / total_us as f64 * 100.0);
    println!("  {:<25} {:>10} µs  100.0%", "TOTAL", total_us);
    println!();

    let per_tick_us = total_us as f64 / total_ticks as f64;
    let intent_per_tick = total_intent_us as f64 / total_ticks as f64;
    let step_per_tick = total_step_us as f64 / total_ticks as f64;
    println!("Per tick:");
    println!("  Total:   {per_tick_us:.1} µs/tick");
    println!("  Intent:  {intent_per_tick:.1} µs/tick");
    println!("  Step:    {step_per_tick:.1} µs/tick");

    ExitCode::SUCCESS
}
