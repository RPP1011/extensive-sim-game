use std::process::ExitCode;

use super::collect_toml_paths;

pub fn run_self_play(args: crate::cli::SelfPlayArgs) -> ExitCode {
    match args.sub {
        crate::cli::SelfPlaySubcommand::Generate(gen_args) => run_self_play_generate(gen_args),
        crate::cli::SelfPlaySubcommand::Eval(eval_args) => run_self_play_eval(eval_args),
    }
}

fn run_self_play_generate(args: crate::cli::SelfPlayGenerateArgs) -> ExitCode {
    use bevy_game::ai::core::self_play::{run_episode, write_episodes, load_policy, FEATURE_DIM, NUM_ACTIONS};
    use bevy_game::ai::core::curriculum::{self, Stage};
    use bevy_game::ai::squad::{SquadAiState, personality::Personality};
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state_with_room};
    use rayon::prelude::*;

    let stage = match Stage::from_str(&args.stage) {
        Some(s) => s,
        None => {
            eprintln!("Unknown stage '{}'. Use: move, kill, 2v2, 4v4", args.stage);
            return ExitCode::from(1);
        }
    };

    let threads = if args.threads == 0 {
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
    } else {
        args.threads
    };

    let policy = if let Some(ref path) = args.policy {
        match load_policy(path) {
            Ok(p) => p,
            Err(e) => { eprintln!("{e}"); return ExitCode::from(1); }
        }
    } else {
        eprintln!("No policy provided, using random initialization");
        random_policy(FEATURE_DIM, NUM_ACTIONS)
    };

    let max_ticks = args.max_ticks.unwrap_or(stage.max_ticks());
    let step_interval = stage.step_interval();
    let temperature = args.temperature;
    let n_episodes = args.episodes;

    eprintln!("Stage: {:?}, Episodes: {}, Max ticks: {}, Step interval: {}, Temperature: {}, Threads: {}",
        stage, n_episodes, max_ticks, step_interval, temperature, threads);
    eprintln!("Feature dim: {}, Actions: {}", FEATURE_DIM, NUM_ACTIONS);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon pool");

    let episodes: Vec<_> = if stage == Stage::Team4 {
        // Stage 4: use scenario TOML files
        let paths = collect_toml_paths(&args.path);
        if paths.is_empty() {
            eprintln!("No *.toml files found.");
            return ExitCode::from(1);
        }
        let scenarios: Vec<_> = paths.iter().filter_map(|p| load_scenario_file(p).ok()).collect();
        eprintln!("Scenario files: {}", scenarios.len());

        let tasks: Vec<(usize, u32)> = scenarios.iter().enumerate()
            .flat_map(|(si, _)| (0..n_episodes).map(move |ei| (si, ei)))
            .collect();

        pool.install(|| {
            tasks.par_iter().map(|&(si, ei)| {
                let cfg = &scenarios[si].scenario;
                let (mut sim, squad_ai, grid_nav) = run_scenario_to_state_with_room(cfg);
                // Always inject grid_nav so terrain features have signal
                if sim.grid_nav.is_none() {
                    sim.grid_nav = Some(grid_nav);
                }
                let ticks = max_ticks.min(cfg.max_ticks);
                let seed = (si as u64 * 1000 + ei as u64) ^ 0xDEADBEEF;
                run_episode(sim, squad_ai, &cfg.name, ticks, &policy, temperature, seed, step_interval)
            }).collect()
        })
    } else {
        // Curriculum stages: generate procedural scenarios
        let tasks: Vec<u32> = (0..n_episodes).collect();

        pool.install(|| {
            tasks.par_iter().map(|&ei| {
                let mut rng = (ei as u64) ^ 0xCAFEBABE;
                let (sim, name) = match stage {
                    Stage::Move => curriculum::generate_move(&mut rng),
                    Stage::Kill => curriculum::generate_kill(&mut rng),
                    Stage::Team2 => {
                        if ei % 2 == 0 {
                            curriculum::generate_team2(&mut rng)
                        } else {
                            curriculum::generate_team2_asymmetric(&mut rng)
                        }
                    }
                    Stage::Team4 => unreachable!(),
                };
                let personalities = sim.units.iter().map(|u| (u.id, Personality::default())).collect();
                let squad_ai = SquadAiState::new(&sim, personalities);
                let seed = (ei as u64 * 7919) ^ 0xDEADBEEF;
                run_episode(sim, squad_ai, &name, max_ticks, &policy, temperature, seed, step_interval)
            }).collect()
        })
    };

    let wins = episodes.iter().filter(|e| e.outcome == "Victory").count();
    let losses = episodes.iter().filter(|e| e.outcome == "Defeat").count();
    let timeouts = episodes.iter().filter(|e| e.outcome == "Timeout").count();
    let total_steps: usize = episodes.iter().map(|e| e.steps.len()).sum();

    // Compute reward stats
    let rewards: Vec<f32> = episodes.iter().map(|e| e.reward).collect();
    let mean_reward = if !rewards.is_empty() { rewards.iter().sum::<f32>() / rewards.len() as f32 } else { 0.0 };
    let step_rewards: Vec<f32> = episodes.iter()
        .flat_map(|e| e.steps.iter().map(|s| s.step_reward))
        .collect();
    let nonzero_steps = step_rewards.iter().filter(|&&r| r.abs() > 1e-6).count();

    eprintln!("\nGenerated {} episodes: {}W / {}L / {}T", episodes.len(), wins, losses, timeouts);
    eprintln!("Total decision steps: {} ({} with nonzero reward, {:.1}%)",
        total_steps, nonzero_steps,
        if total_steps > 0 { nonzero_steps as f64 / total_steps as f64 * 100.0 } else { 0.0 });
    eprintln!("Mean episode reward: {:.4}", mean_reward);
    eprintln!("Win rate: {:.1}%", if !episodes.is_empty() { wins as f64 / episodes.len() as f64 * 100.0 } else { 0.0 });

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_episodes(&episodes, &args.output).expect("Failed to write episodes");
    eprintln!("Written to: {}", args.output.display());

    ExitCode::SUCCESS
}

fn run_self_play_eval(args: crate::cli::SelfPlayEvalArgs) -> ExitCode {
    use bevy_game::ai::core::self_play::{run_episode_greedy_with_focus, load_policy};
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state_with_room};

    let policy = match load_policy(&args.policy) {
        Ok(p) => p,
        Err(e) => { eprintln!("{e}"); return ExitCode::from(1); }
    };

    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found.");
        return ExitCode::from(1);
    }

    let mut wins = 0u32;
    let mut losses = 0u32;
    let mut timeouts = 0u32;

    for toml_path in &paths {
        let scenario_file = match load_scenario_file(toml_path) {
            Ok(f) => f,
            Err(err) => { eprintln!("{err}"); continue; }
        };

        let cfg = &scenario_file.scenario;
        let (mut sim, squad_ai, grid_nav) = run_scenario_to_state_with_room(cfg);
        if sim.grid_nav.is_none() {
            sim.grid_nav = Some(grid_nav);
        }
        let ticks = args.max_ticks.unwrap_or(cfg.max_ticks);
        let ep = run_episode_greedy_with_focus(sim, squad_ai, &cfg.name, ticks, &policy, args.focus);

        let tag = match ep.outcome.as_str() {
            "Victory" => { wins += 1; "WIN " }
            "Defeat" => { losses += 1; "LOSS" }
            _ => { timeouts += 1; "TIME" }
        };
        println!("[{tag}] {:<30} tick={:<5}", cfg.name, ep.ticks);
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

fn random_policy(input_dim: usize, output_dim: usize) -> bevy_game::ai::core::self_play::PolicyWeights {
    use bevy_game::ai::core::self_play::{PolicyWeights, LayerWeights};

    let hidden = 64;
    let scale1 = (2.0 / (input_dim + hidden) as f64).sqrt() as f32;
    let scale2 = (2.0 / (hidden + output_dim) as f64).sqrt() as f32;
    let mut rng = 42u64;

    let rand_f32 = |rng: &mut u64| -> f32 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (*rng >> 33) as f32 / (1u64 << 31) as f32;
        (u - 0.5) * 2.0
    };

    let w1: Vec<Vec<f32>> = (0..input_dim).map(|_| {
        (0..hidden).map(|_| rand_f32(&mut rng) * scale1).collect()
    }).collect();
    let b1 = vec![0.0; hidden];

    let w2: Vec<Vec<f32>> = (0..hidden).map(|_| {
        (0..output_dim).map(|_| rand_f32(&mut rng) * scale2).collect()
    }).collect();
    let b2 = vec![0.0; output_dim];

    PolicyWeights {
        layers: vec![
            LayerWeights { w: w1, b: b1 },
            LayerWeights { w: w2, b: b2 },
        ],
        input_scale: Vec::new(),
    }
}

pub fn run_raw_dataset(args: crate::cli::RawDatasetArgs) -> ExitCode {
    use bevy_game::ai::core::dataset::{generate_raw_dataset, write_raw_dataset};
    use bevy_game::ai::core::self_play::{FEATURE_DIM, NUM_ACTIONS};
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state_with_room};

    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found in {}.", args.path.display());
        return ExitCode::from(1);
    }

    eprintln!("Raw dataset: {} features, {} actions, rollout depth {}",
        FEATURE_DIM, NUM_ACTIONS, args.depth);

    let mut all_samples = Vec::new();

    for toml_path in &paths {
        let scenario_file = match load_scenario_file(toml_path) {
            Ok(f) => f,
            Err(err) => { eprintln!("{err}"); continue; }
        };

        let cfg = &scenario_file.scenario;
        eprintln!("Generating from {}...", cfg.name);

        let (mut sim, squad_ai, grid_nav) = run_scenario_to_state_with_room(cfg);
        if sim.grid_nav.is_none() {
            sim.grid_nav = Some(grid_nav);
        }

        let samples = generate_raw_dataset(sim, squad_ai, &cfg.name, cfg.max_ticks, args.depth);
        eprintln!("  {} samples", samples.len());
        all_samples.extend(samples);
    }

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_raw_dataset(&all_samples, &args.output).expect("Failed to write dataset");

    eprintln!("\nTotal samples: {}", all_samples.len());
    eprintln!("Written to: {}", args.output.display());

    // Print label distribution
    let labels = [
        "AttackNearest", "AttackWeakest", "AttackFocus",
        "Ability0", "Ability1", "Ability2", "Ability3",
        "Ability4", "Ability5", "Ability6", "Ability7",
        "MoveToward", "MoveAway", "Hold",
    ];
    let mut counts = [0u32; NUM_ACTIONS];
    for s in &all_samples {
        if (s.label as usize) < NUM_ACTIONS {
            counts[s.label as usize] += 1;
        }
    }
    eprintln!("\nLabel distribution:");
    for (i, name) in labels.iter().enumerate() {
        let pct = if all_samples.is_empty() { 0.0 } else { counts[i] as f64 / all_samples.len() as f64 * 100.0 };
        eprintln!("  {:<18} {:>6} ({:.1}%)", name, counts[i], pct);
    }

    ExitCode::SUCCESS
}
