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
