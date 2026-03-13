//! Generate ability operator training dataset from scenario replays.
//!
//! Delegates to `generate_operator_dataset_streaming` in game_state.rs
//! which handles the sim replay, ability cast event capture, and
//! state snapshotting.

use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use super::collect_toml_paths;

pub fn run_operator_dataset(args: crate::cli::OperatorDatasetArgs) -> ExitCode {
    use bevy_game::ai::core::ability_eval::generate_operator_dataset_streaming;
    use bevy_game::ai::core::ability_eval::OperatorSample;
    use bevy_game::ai::core::ability_transformer::AbilityTransformerWeights;
    use bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer;
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state_with_room};
    use rayon::prelude::*;

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();
    }

    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found in {}.", args.path.display());
        return ExitCode::from(1);
    }

    let scenarios: Vec<_> = paths
        .iter()
        .filter_map(|p| match load_scenario_file(p) {
            Ok(f) => Some(f),
            Err(err) => {
                eprintln!("{err}");
                None
            }
        })
        .collect();

    let n_scenarios = scenarios.len();
    eprintln!("Generating operator dataset from {} scenarios...", n_scenarios);

    // Load ability transformer for CLS embeddings
    let transformer_weights_path = "generated/ability_transformer_weights_v5.json";
    let transformer = match std::fs::read_to_string(transformer_weights_path) {
        Ok(json) => match AbilityTransformerWeights::from_json(&json) {
            Ok(w) => {
                eprintln!("Loaded ability transformer from {transformer_weights_path}");
                Some(w)
            }
            Err(e) => {
                eprintln!("Warning: failed to load transformer weights: {e}");
                None
            }
        },
        Err(_) => {
            eprintln!("Warning: {transformer_weights_path} not found, CLS embeddings will be zeros");
            None
        }
    };
    let tokenizer = AbilityTokenizer::new();

    let all_samples: Mutex<Vec<OperatorSample>> = Mutex::new(Vec::new());
    let done = AtomicUsize::new(0);

    scenarios.par_iter().enumerate().for_each(|(scenario_idx, scenario_file)| {
        let cfg = &scenario_file.scenario;
        let (sim, squad_ai, grid_nav) = run_scenario_to_state_with_room(cfg);

        let mut local_samples = Vec::new();

        generate_operator_dataset_streaming(
            sim,
            squad_ai,
            Some(grid_nav),
            cfg.max_ticks,
            scenario_idx as u32,
            transformer.as_ref(),
            &tokenizer,
            |sample| {
                local_samples.push(sample);
            },
        );

        let n = local_samples.len();
        all_samples.lock().unwrap().extend(local_samples);

        let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
        if completed % 50 == 0 || completed == n_scenarios {
            eprintln!("  [{completed}/{n_scenarios}] {} — {n} samples", cfg.name);
        }
    });

    let samples = all_samples.into_inner().unwrap();
    let n = samples.len();
    eprintln!("\nTotal: {n} operator samples from {n_scenarios} scenarios");

    if n == 0 {
        eprintln!("No samples generated — check that scenarios have hero abilities.");
        return ExitCode::from(1);
    }

    // Write npz
    write_npz(&args.output, &samples);
    eprintln!("Written to: {}", args.output.display());

    ExitCode::SUCCESS
}

/// Write samples to npz file.
fn write_npz(path: &std::path::Path, samples: &[bevy_game::ai::core::ability_eval::OperatorSample]) {
    use ndarray::Array2;
    use ndarray_npy::NpzWriter;

    let n = samples.len();
    if n == 0 {
        return;
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = std::fs::File::create(path).expect("Failed to create output npz");
    let mut npz = NpzWriter::new(file);

    let max_entities = 7;
    let entity_dim = 23;
    let max_threats = 8;
    let max_positions = 8;

    // Determine max ability slots
    let max_abl = samples.iter().map(|s| s.n_ability_slots).max().unwrap_or(0);

    // Entity features: (N, 7*23)
    let cols = max_entities * entity_dim;
    let entity_features = build_f32_array(samples, cols, |s| &s.entity_features);
    npz.add_array("entity_features", &entity_features).unwrap();

    let entity_types = build_i32_array(samples, max_entities, |s| &s.entity_types);
    npz.add_array("entity_types", &entity_types).unwrap();

    let entity_mask = build_i32_array(samples, max_entities, |s| &s.entity_mask);
    npz.add_array("entity_mask", &entity_mask).unwrap();

    let threat_features = build_f32_array(samples, max_threats * 8, |s| &s.threat_features);
    npz.add_array("threat_features", &threat_features).unwrap();

    let threat_mask = build_i32_array(samples, max_threats, |s| &s.threat_mask);
    npz.add_array("threat_mask", &threat_mask).unwrap();

    let position_features = build_f32_array(samples, max_positions * 8, |s| &s.position_features);
    npz.add_array("position_features", &position_features).unwrap();

    let position_mask = build_i32_array(samples, max_positions, |s| &s.position_mask);
    npz.add_array("position_mask", &position_mask).unwrap();

    let abl_slot_dim = samples.first().map_or(130, |s| {
        if s.n_ability_slots > 0 { s.ability_slot_features.len() / s.n_ability_slots } else { 130 }
    });
    let abl_feat = build_f32_array(samples, max_abl * abl_slot_dim, |s| &s.ability_slot_features);
    npz.add_array("ability_slot_features", &abl_feat).unwrap();

    let abl_types = build_i32_array(samples, max_abl, |s| &s.ability_slot_types);
    npz.add_array("ability_slot_types", &abl_types).unwrap();

    let abl_mask = build_i32_array_pad(samples, max_abl, 1, |s| &s.ability_slot_mask);
    npz.add_array("ability_slot_mask", &abl_mask).unwrap();

    let cls_dim = samples.first().map_or(128, |s| s.ability_cls.len());
    let cls = build_f32_array(samples, cls_dim, |s| &s.ability_cls);
    npz.add_array("ability_cls", &cls).unwrap();

    let caster_slot: Array2<i32> = Array2::from_shape_vec(
        (n, 1),
        samples.iter().map(|s| s.caster_slot).collect(),
    ).unwrap();
    npz.add_array("caster_slot", &caster_slot).unwrap();

    let duration_norm: Array2<f32> = Array2::from_shape_vec(
        (n, 1),
        samples.iter().map(|s| s.duration_norm).collect(),
    ).unwrap();
    npz.add_array("duration_norm", &duration_norm).unwrap();

    let target_hp = build_f32_array(samples, max_entities * 3, |s| &s.target_hp);
    npz.add_array("target_hp", &target_hp).unwrap();

    let target_cc = build_f32_array(samples, max_entities, |s| &s.target_cc);
    npz.add_array("target_cc", &target_cc).unwrap();

    let target_cc_stun = build_f32_array(samples, max_entities, |s| &s.target_cc_stun);
    npz.add_array("target_cc_stun", &target_cc_stun).unwrap();

    let target_pos = build_f32_array(samples, max_entities * 2, |s| &s.target_pos);
    npz.add_array("target_pos", &target_pos).unwrap();

    let target_exists = build_f32_array(samples, max_entities, |s| &s.target_exists);
    npz.add_array("target_exists", &target_exists).unwrap();

    let ability_props = build_f32_array(samples, 80, |s| &s.ability_props);
    npz.add_array("ability_props", &ability_props).unwrap();

    let scenario_ids: Array2<i32> = Array2::from_shape_vec(
        (n, 1),
        samples.iter().map(|s| s.scenario_id as i32).collect(),
    ).unwrap();
    npz.add_array("scenario_ids", &scenario_ids).unwrap();

    npz.finish().expect("Failed to finalize npz");
}

fn build_f32_array(
    samples: &[bevy_game::ai::core::ability_eval::OperatorSample],
    cols: usize,
    extract: impl Fn(&bevy_game::ai::core::ability_eval::OperatorSample) -> &[f32],
) -> ndarray::Array2<f32> {
    let n = samples.len();
    let mut data = Vec::with_capacity(n * cols);
    for s in samples {
        let v = extract(s);
        let take = cols.min(v.len());
        data.extend_from_slice(&v[..take]);
        for _ in take..cols {
            data.push(0.0);
        }
    }
    ndarray::Array2::from_shape_vec((n, cols), data).unwrap()
}

fn build_i32_array(
    samples: &[bevy_game::ai::core::ability_eval::OperatorSample],
    cols: usize,
    extract: impl Fn(&bevy_game::ai::core::ability_eval::OperatorSample) -> &[i32],
) -> ndarray::Array2<i32> {
    build_i32_array_pad(samples, cols, 0, extract)
}

fn build_i32_array_pad(
    samples: &[bevy_game::ai::core::ability_eval::OperatorSample],
    cols: usize,
    pad_value: i32,
    extract: impl Fn(&bevy_game::ai::core::ability_eval::OperatorSample) -> &[i32],
) -> ndarray::Array2<i32> {
    let n = samples.len();
    let mut data = Vec::with_capacity(n * cols);
    for s in samples {
        let v = extract(s);
        let take = cols.min(v.len());
        data.extend_from_slice(&v[..take]);
        for _ in take..cols {
            data.push(pad_value);
        }
    }
    ndarray::Array2::from_shape_vec((n, cols), data).unwrap()
}
