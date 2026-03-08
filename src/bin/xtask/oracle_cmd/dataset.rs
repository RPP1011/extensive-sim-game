use std::process::ExitCode;

use super::collect_toml_paths;

pub fn run_oracle_dataset(args: crate::cli::OracleDatasetArgs) -> ExitCode {
    use bevy_game::ai::core::dataset::{generate_dataset, write_dataset};
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state};

    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found in {}.", args.path.display());
        return ExitCode::from(1);
    }

    let mut all_samples = Vec::new();

    for toml_path in &paths {
        let scenario_file = match load_scenario_file(toml_path) {
            Ok(f) => f,
            Err(err) => { eprintln!("{err}"); continue; }
        };

        eprintln!("Generating dataset from {}...", scenario_file.scenario.name);

        let cfg = &scenario_file.scenario;
        let (sim, squad_ai) = run_scenario_to_state(cfg);
        let samples = generate_dataset(sim, squad_ai, &cfg.name, cfg.max_ticks, args.depth);

        eprintln!("  {} samples", samples.len());
        all_samples.extend(samples);
    }

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_dataset(&all_samples, &args.output).expect("Failed to write dataset");

    eprintln!("\nTotal samples: {}", all_samples.len());
    eprintln!("Written to: {}", args.output.display());

    // Print label distribution
    let mut counts = [0u32; 10];
    for s in &all_samples {
        counts[s.label as usize] += 1;
    }
    let labels = [
        "AttackNearest", "AttackWeakest", "DamageAbility", "HealAbility",
        "CcAbility", "DefenseAbility", "UtilityAbility", "MoveToward", "MoveAway", "Hold",
    ];
    eprintln!("\nLabel distribution:");
    for (i, name) in labels.iter().enumerate() {
        let pct = if all_samples.is_empty() { 0.0 } else { counts[i] as f64 / all_samples.len() as f64 * 100.0 };
        eprintln!("  {:<18} {:>6} ({:.1}%)", name, counts[i], pct);
    }

    ExitCode::SUCCESS
}

pub fn run_combat_dataset(args: crate::cli::CombatDatasetArgs) -> ExitCode {
    use bevy_game::ai::core::dataset::{generate_combat_dataset, write_combat_dataset, CombatActionClass};
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state};

    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found in {}.", args.path.display());
        return ExitCode::from(1);
    }

    let mut all_samples = Vec::new();

    for toml_path in &paths {
        let scenario_file = match load_scenario_file(toml_path) {
            Ok(f) => f,
            Err(err) => { eprintln!("{err}"); continue; }
        };

        eprintln!("Generating combat dataset from {}...", scenario_file.scenario.name);

        let cfg = &scenario_file.scenario;
        let (sim, squad_ai) = run_scenario_to_state(cfg);
        let samples = generate_combat_dataset(sim, squad_ai, &cfg.name, cfg.max_ticks, args.depth);

        eprintln!("  {} samples", samples.len());
        all_samples.extend(samples);
    }

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_combat_dataset(&all_samples, &args.output).expect("Failed to write dataset");

    eprintln!("\nTotal samples: {}", all_samples.len());
    eprintln!("Written to: {}", args.output.display());

    // Print label distribution
    let mut counts = [0u32; 5];
    for s in &all_samples {
        if (s.label as usize) < 5 {
            counts[s.label as usize] += 1;
        }
    }
    eprintln!("\nLabel distribution:");
    for i in 0..5 {
        let name = CombatActionClass::from_index(i).name();
        let pct = if all_samples.is_empty() { 0.0 } else { counts[i] as f64 / all_samples.len() as f64 * 100.0 };
        eprintln!("  {:<18} {:>6} ({:.1}%)", name, counts[i], pct);
    }

    ExitCode::SUCCESS
}

pub fn run_ability_dataset(args: crate::cli::AbilityDatasetArgs) -> ExitCode {
    use bevy_game::ai::core::ability_eval::{
        generate_ability_eval_dataset_with_encoder, write_ability_eval_dataset,
    };
    use bevy_game::ai::core::ability_encoding::AbilityEncoder;
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state};

    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found in {}.", args.path.display());
        return ExitCode::from(1);
    }

    // Load optional ability encoder
    let encoder = args.ability_encoder.as_ref().map(|p| {
        let data = std::fs::read_to_string(p)
            .unwrap_or_else(|e| panic!("Failed to read encoder {}: {e}", p.display()));
        let enc = AbilityEncoder::from_json(&data)
            .unwrap_or_else(|e| panic!("Failed to parse encoder: {e}"));
        eprintln!("Ability encoder loaded from {}", p.display());
        enc
    });

    let mut all_samples = Vec::new();

    for toml_path in &paths {
        let scenario_file = match load_scenario_file(toml_path) {
            Ok(f) => f,
            Err(err) => {
                eprintln!("{err}");
                continue;
            }
        };

        eprintln!(
            "Generating ability eval data from {}...",
            scenario_file.scenario.name
        );

        let cfg = &scenario_file.scenario;
        let (sim, squad_ai) = run_scenario_to_state(cfg);
        let samples = generate_ability_eval_dataset_with_encoder(
            sim,
            squad_ai,
            &cfg.name,
            cfg.max_ticks,
            args.depth,
            encoder.as_ref(),
        );

        eprintln!("  {} samples", samples.len());
        all_samples.extend(samples);
    }

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_ability_eval_dataset(&all_samples, &args.output)
        .expect("Failed to write ability eval dataset");

    eprintln!("\nTotal samples: {}", all_samples.len());
    eprintln!("Written to: {}", args.output.display());

    // Print category distribution
    let mut cat_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut urgency_sum: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for s in &all_samples {
        *cat_counts.entry(s.category.clone()).or_default() += 1;
        *urgency_sum.entry(s.category.clone()).or_default() += s.urgency as f64;
    }

    eprintln!("\nCategory distribution:");
    let mut cats: Vec<_> = cat_counts.iter().collect();
    cats.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
    for (cat, count) in &cats {
        let avg_urg = urgency_sum[*cat] / **count as f64;
        eprintln!("  {:<15} {:>6} samples  avg_urgency={:.3}", cat, count, avg_urg);
    }

    ExitCode::SUCCESS
}

pub fn run_outcome_dataset(args: crate::cli::OutcomeDatasetArgs) -> ExitCode {
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state};

    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found in {}.", args.path.display());
        return ExitCode::from(1);
    }

    if args.v2 {
        return run_outcome_dataset_v2(&paths, &args);
    }

    use bevy_game::ai::core::ability_eval::{
        generate_outcome_dataset, write_outcome_dataset,
    };

    let mut all_samples = Vec::new();
    let mut wins = 0;
    let mut losses = 0;

    for toml_path in &paths {
        let scenario_file = match load_scenario_file(toml_path) {
            Ok(f) => f,
            Err(err) => { eprintln!("{err}"); continue; }
        };

        let cfg = &scenario_file.scenario;
        let (sim, squad_ai) = run_scenario_to_state(cfg);
        let samples = generate_outcome_dataset(
            sim, squad_ai, &cfg.name, cfg.max_ticks, args.sample_interval,
        );

        if let Some(s) = samples.first() {
            if s.hero_wins > 0.5 { wins += 1; } else { losses += 1; }
        }
        eprintln!("  {} — {} samples, outcome={}", cfg.name, samples.len(),
            if samples.first().map_or(false, |s| s.hero_wins > 0.5) { "win" } else { "loss" });
        all_samples.extend(samples);
    }

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_outcome_dataset(&all_samples, &args.output)
        .expect("Failed to write outcome dataset");

    eprintln!("\nTotal: {} samples from {} scenarios ({} wins, {} losses)",
        all_samples.len(), paths.len(), wins, losses);
    eprintln!("Written to: {}", args.output.display());

    ExitCode::SUCCESS
}

fn run_outcome_dataset_v2(
    paths: &[std::path::PathBuf],
    args: &crate::cli::OutcomeDatasetArgs,
) -> ExitCode {
    use bevy_game::ai::core::ability_eval::{
        generate_outcome_dataset_v2, write_outcome_dataset_v2, OutcomeSampleV2,
    };
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state_with_room};
    use rayon::prelude::*;

    let scenarios: Vec<_> = paths.iter().filter_map(|p| {
        match load_scenario_file(p) {
            Ok(f) => Some(f),
            Err(err) => { eprintln!("{err}"); None }
        }
    }).collect();

    let n_scenarios = scenarios.len();
    eprintln!("Generating v2 outcome dataset from {} scenarios (parallel)...", n_scenarios);

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = std::fs::File::create(&args.output).expect("Failed to create output file");
    let writer = std::sync::Mutex::new(std::io::BufWriter::new(file));

    let sample_interval = args.sample_interval;
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total_samples = std::sync::atomic::AtomicUsize::new(0);
    let wins = std::sync::atomic::AtomicUsize::new(0);
    let losses = std::sync::atomic::AtomicUsize::new(0);

    scenarios.par_iter().for_each(|scenario_file| {
        let cfg = &scenario_file.scenario;
        let (mut sim, squad_ai, grid_nav) = run_scenario_to_state_with_room(cfg);
        sim.grid_nav = Some(grid_nav);
        let samples = generate_outcome_dataset_v2(
            sim, squad_ai, &cfg.name, cfg.max_ticks, sample_interval,
        );

        if let Some(s) = samples.first() {
            if s.hero_wins > 0.5 {
                wins.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                losses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let n_samples = samples.len();
        total_samples.fetch_add(n_samples, std::sync::atomic::Ordering::Relaxed);

        // Write immediately, don't buffer in memory
        {
            use std::io::Write;
            let mut w = writer.lock().unwrap();
            for sample in &samples {
                serde_json::to_writer(&mut *w, sample).unwrap();
                writeln!(&mut *w).unwrap();
            }
        }
        // Drop samples here to free memory

        let completed = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if completed % 100 == 0 || completed == n_scenarios {
            eprintln!("  [{completed}/{n_scenarios}] {} — {} samples",
                cfg.name, n_samples);
        }
    });

    let w = wins.load(std::sync::atomic::Ordering::Relaxed);
    let l = losses.load(std::sync::atomic::Ordering::Relaxed);
    let t = total_samples.load(std::sync::atomic::Ordering::Relaxed);
    eprintln!("\nTotal: {} samples (v2) from {} scenarios ({} wins, {} losses)",
        t, n_scenarios, w, l);
    eprintln!("Written to: {}", args.output.display());

    ExitCode::SUCCESS
}

pub fn run_nextstate_dataset(args: crate::cli::NextstateDatasetArgs) -> ExitCode {
    use bevy_game::ai::core::ability_eval::generate_nextstate_dataset_streaming;
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state_with_room};
    use rayon::prelude::*;

    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found in {}.", args.path.display());
        return ExitCode::from(1);
    }

    let scenarios: Vec<_> = paths.iter().filter_map(|p| {
        match load_scenario_file(p) {
            Ok(f) => Some(f),
            Err(err) => { eprintln!("{err}"); None }
        }
    }).collect();

    let n_scenarios = scenarios.len();
    eprintln!("Generating next-state dataset from {} scenarios (parallel, interval={})...",
        n_scenarios, args.sample_interval);

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = std::fs::File::create(&args.output).expect("Failed to create output file");
    let writer = std::sync::Mutex::new(std::io::BufWriter::new(file));

    let sample_interval = args.sample_interval;
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total_samples = std::sync::atomic::AtomicUsize::new(0);

    scenarios.par_iter().for_each(|scenario_file| {
        let cfg = &scenario_file.scenario;
        let (mut sim, squad_ai, grid_nav) = run_scenario_to_state_with_room(cfg);
        sim.grid_nav = Some(grid_nav);

        let n_samples = generate_nextstate_dataset_streaming(
            sim, squad_ai, &cfg.name, cfg.max_ticks, sample_interval,
            |sample| {
                use std::io::Write;
                let mut w = writer.lock().unwrap();
                serde_json::to_writer(&mut *w, &sample).unwrap();
                writeln!(&mut *w).unwrap();
            },
        );

        total_samples.fetch_add(n_samples, std::sync::atomic::Ordering::Relaxed);

        let completed = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if completed % 100 == 0 || completed == n_scenarios {
            eprintln!("  [{completed}/{n_scenarios}] {} — {} samples",
                cfg.name, n_samples);
        }
    });

    let t = total_samples.load(std::sync::atomic::Ordering::Relaxed);
    eprintln!("\nTotal: {} samples from {} scenarios", t, n_scenarios);
    eprintln!("Written to: {}", args.output.display());

    ExitCode::SUCCESS
}
