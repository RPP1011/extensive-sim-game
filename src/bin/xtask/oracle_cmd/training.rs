use std::process::ExitCode;
use super::collect_toml_paths;

pub fn run_oracle_student(args: crate::cli::OracleStudentArgs) -> ExitCode {
    use bevy_game::ai::core::dataset::extract_unit_features;
    use bevy_game::ai::core::{step, is_alive, Team, FIXED_TICK_MS, UnitIntent};
    use bevy_game::ai::core::ability_eval::{AbilityEvalWeights, evaluate_abilities_with_encoder};
    use bevy_game::ai::core::ability_encoding::AbilityEncoder;
    use bevy_game::ai::squad::generate_intents;
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state};

    let model_json = match std::fs::read_to_string(&args.model) {
        Ok(s) => s,
        Err(e) => { eprintln!("Failed to read model file: {e}"); return ExitCode::from(1); }
    };

    let json_value: serde_json::Value = match serde_json::from_str(&model_json) {
        Ok(v) => v,
        Err(e) => { eprintln!("Failed to parse model JSON: {e}"); return ExitCode::from(1); }
    };
    let weights = StudentWeights::from_json(&json_value);

    // Detect model type: 5-class (combat-only) or 10-class (legacy)
    let output_dim = weights.layers.last().map(|(_, b)| b.len()).unwrap_or(10);
    let is_combat_model = output_dim == 5;

    // Load optional frozen ability evaluator weights
    let ability_weights = args.ability_eval.as_ref().map(|path| {
        let data = std::fs::read_to_string(path)
            .unwrap_or_else(|e| { eprintln!("Failed to read ability eval weights: {e}"); std::process::exit(1); });
        let json: serde_json::Value = serde_json::from_str(&data)
            .unwrap_or_else(|e| { eprintln!("Failed to parse ability eval JSON: {e}"); std::process::exit(1); });
        AbilityEvalWeights::from_json(&json)
    });

    // Load optional frozen ability encoder for embedding-enriched evaluation
    let ability_encoder = args.ability_encoder.as_ref().map(|path| {
        let data = std::fs::read_to_string(path)
            .unwrap_or_else(|e| { eprintln!("Failed to read ability encoder: {e}"); std::process::exit(1); });
        AbilityEncoder::from_json(&data)
            .unwrap_or_else(|e| { eprintln!("Failed to parse ability encoder: {e}"); std::process::exit(1); })
    });

    if is_combat_model {
        eprintln!("Student model: 5-class combat-only ({}→{})", weights.layers[0].0.len(), output_dim);
    } else {
        eprintln!("Student model: 10-class legacy ({}→{})", weights.layers[0].0.len(), output_dim);
    }
    if ability_weights.is_some() {
        eprintln!("Ability evaluators: loaded (frozen interrupt layer)");
    }
    if ability_encoder.is_some() {
        eprintln!("Ability encoder: loaded (32-dim embeddings)");
    }

    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found.");
        return ExitCode::from(1);
    }

    let mut wins = 0u32;
    let mut losses = 0u32;
    let mut timeouts = 0u32;
    let mut ability_fires = 0u64;
    let mut student_fires = 0u64;

    for toml_path in &paths {
        let scenario_file = match load_scenario_file(toml_path) {
            Ok(f) => f,
            Err(err) => { eprintln!("{err}"); continue; }
        };

        let cfg = &scenario_file.scenario;
        let (mut sim, mut squad_ai) = run_scenario_to_state(cfg);
        let hero_ids: Vec<u32> = sim.units.iter().filter(|u| u.team == Team::Hero).map(|u| u.id).collect();
        let mut outcome = "Timeout";

        for _ in 0..cfg.max_ticks {
            let mut intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

            for &uid in &hero_ids {
                if !sim.units.iter().any(|u| u.id == uid && is_alive(u)) { continue; }
                if let Some(u) = sim.units.iter().find(|u| u.id == uid) {
                    if u.casting.is_some() || u.control_remaining_ms > 0 { continue; }
                }

                // Phase 1: Check frozen ability evaluators (interrupt layer)
                if let Some(ref ab_weights) = ability_weights {
                    if let Some((action, _urgency)) = evaluate_abilities_with_encoder(
                        &sim, &squad_ai, uid, ab_weights, ability_encoder.as_ref()) {
                        intents.retain(|i| i.unit_id != uid);
                        intents.push(UnitIntent { unit_id: uid, action });
                        ability_fires += 1;
                        continue;
                    }
                }

                // Phase 2: Student model handles attack/move/hold
                let features = extract_unit_features(&sim, &squad_ai, uid);
                let action = if is_combat_model {
                    let class = student_predict_combat(&weights, &features);
                    combat_class_to_intent(class, uid, &sim)
                } else {
                    let class = student_predict(&weights, &features);
                    action_class_to_intent(class, uid, &sim)
                };
                if let Some(action) = action {
                    intents.retain(|i| i.unit_id != uid);
                    intents.push(UnitIntent { unit_id: uid, action });
                    student_fires += 1;
                }
            }

            let (new_sim, _) = step(sim, &intents, FIXED_TICK_MS);
            sim = new_sim;

            let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
            let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
            if enemies_alive == 0 { outcome = "Victory"; break; }
            if heroes_alive == 0 { outcome = "Defeat"; break; }
        }

        let tag = match outcome {
            "Victory" => { wins += 1; "WIN " }
            "Defeat" => { losses += 1; "LOSS" }
            _ => { timeouts += 1; "TIME" }
        };
        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        println!("[{tag}] {:<30} tick={:<5} heroes={} enemies={}", cfg.name, sim.tick, heroes_alive, enemies_alive);
    }

    let total = wins + losses + timeouts;
    if total > 1 {
        println!(
            "\n--- Aggregate ---\nScenarios: {total}  Wins: {wins}  Losses: {losses}  Timeouts: {timeouts}  Win rate: {:.1}%",
            if total > 0 { wins as f64 / total as f64 * 100.0 } else { 0.0 }
        );
    }
    if ability_weights.is_some() {
        let total_decisions = ability_fires + student_fires;
        let ab_pct = if total_decisions > 0 { ability_fires as f64 / total_decisions as f64 * 100.0 } else { 0.0 };
        println!("Decision split: ability_eval={ability_fires} ({ab_pct:.1}%)  student={student_fires} ({:.1}%)", 100.0 - ab_pct);
    }

    ExitCode::SUCCESS
}

/// Variable-depth MLP weights for student model inference.
/// JSON format: { "layers": [ {"w": [[...]], "b": [...]}, ... ] }
/// Also supports legacy 3-layer format with w1/b1/w2/b2/w3/b3.
#[derive(Debug)]
struct StudentWeights {
    layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>, // (weight, bias) per layer
}

impl StudentWeights {
    fn from_json(v: &serde_json::Value) -> Self {
        if let Some(layers_arr) = v.get("layers").and_then(|l| l.as_array()) {
            let layers = layers_arr.iter().map(|layer| {
                let w: Vec<Vec<f32>> = serde_json::from_value(layer["w"].clone()).unwrap();
                let b: Vec<f32> = serde_json::from_value(layer["b"].clone()).unwrap();
                (w, b)
            }).collect();
            StudentWeights { layers }
        } else {
            // Legacy 3-layer format
            let w1: Vec<Vec<f32>> = serde_json::from_value(v["w1"].clone()).unwrap();
            let b1: Vec<f32> = serde_json::from_value(v["b1"].clone()).unwrap();
            let w2: Vec<Vec<f32>> = serde_json::from_value(v["w2"].clone()).unwrap();
            let b2: Vec<f32> = serde_json::from_value(v["b2"].clone()).unwrap();
            let w3: Vec<Vec<f32>> = serde_json::from_value(v["w3"].clone()).unwrap();
            let b3: Vec<f32> = serde_json::from_value(v["b3"].clone()).unwrap();
            StudentWeights { layers: vec![(w1, b1), (w2, b2), (w3, b3)] }
        }
    }
}

fn student_predict_combat(w: &StudentWeights, features: &[f32]) -> bevy_game::ai::core::dataset::CombatActionClass {
    let mut activations: Vec<f32> = features.to_vec();

    for (layer_idx, (weights, biases)) in w.layers.iter().enumerate() {
        let is_last = layer_idx == w.layers.len() - 1;
        let next: Vec<f32> = biases.iter().enumerate().map(|(j, &b)| {
            let sum: f32 = activations.iter().enumerate().map(|(i, &x)| x * weights[i][j]).sum();
            if is_last { sum + b } else { (sum + b).max(0.0) }
        }).collect();
        activations = next;
    }

    let argmax = activations.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(4);

    bevy_game::ai::core::dataset::CombatActionClass::from_index(argmax)
}

fn combat_class_to_intent(
    class: bevy_game::ai::core::dataset::CombatActionClass,
    unit_id: u32,
    state: &bevy_game::ai::core::SimState,
) -> Option<bevy_game::ai::core::IntentAction> {
    use bevy_game::ai::core::{IntentAction, distance, is_alive, move_towards, position_at_range};
    use bevy_game::ai::core::dataset::CombatActionClass;

    let unit = state.units.iter().find(|u| u.id == unit_id)?;
    let enemies: Vec<_> = state.units.iter().filter(|u| u.team != unit.team && is_alive(u)).collect();

    let nearest_enemy = enemies.iter().min_by(|a, b| {
        distance(unit.position, a.position).partial_cmp(&distance(unit.position, b.position)).unwrap_or(std::cmp::Ordering::Equal)
    });
    let weakest_enemy = enemies.iter().min_by(|a, b| {
        let ha = a.hp as f32 / a.max_hp.max(1) as f32;
        let hb = b.hp as f32 / b.max_hp.max(1) as f32;
        ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
    });

    match class {
        CombatActionClass::AttackNearest => nearest_enemy.map(|e| IntentAction::Attack { target_id: e.id }),
        CombatActionClass::AttackWeakest => weakest_enemy.map(|e| IntentAction::Attack { target_id: e.id }),
        CombatActionClass::MoveToward => {
            nearest_enemy.map(|e| {
                let desired = position_at_range(unit.position, e.position, unit.attack_range * 0.9);
                let next = move_towards(unit.position, desired, unit.move_speed_per_sec * 0.1);
                IntentAction::MoveTo { position: next }
            })
        }
        CombatActionClass::MoveAway => {
            nearest_enemy.map(|e| {
                let away = bevy_game::ai::core::move_away(unit.position, e.position, unit.move_speed_per_sec * 0.1);
                IntentAction::MoveTo { position: away }
            })
        }
        CombatActionClass::Hold => Some(IntentAction::Hold),
    }
}

fn student_predict(w: &StudentWeights, features: &[f32]) -> bevy_game::ai::core::dataset::ActionClass {
    let mut activations: Vec<f32> = features.to_vec();

    for (layer_idx, (weights, biases)) in w.layers.iter().enumerate() {
        let is_last = layer_idx == w.layers.len() - 1;
        let next: Vec<f32> = biases.iter().enumerate().map(|(j, &b)| {
            let sum: f32 = activations.iter().enumerate().map(|(i, &x)| x * weights[i][j]).sum();
            if is_last { sum + b } else { (sum + b).max(0.0) } // ReLU on hidden, linear on output
        }).collect();
        activations = next;
    }

    let argmax = activations.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(9);

    bevy_game::ai::core::dataset::ActionClass::from_index(argmax)
}

fn action_class_to_intent(
    class: bevy_game::ai::core::dataset::ActionClass,
    unit_id: u32,
    state: &bevy_game::ai::core::SimState,
) -> Option<bevy_game::ai::core::IntentAction> {
    use bevy_game::ai::core::{IntentAction, distance, is_alive, move_towards, position_at_range};
    use bevy_game::ai::effects::AbilityTarget;

    let unit = state.units.iter().find(|u| u.id == unit_id)?;
    let enemies: Vec<_> = state.units.iter().filter(|u| u.team != unit.team && is_alive(u)).collect();
    let allies: Vec<_> = state.units.iter().filter(|u| u.team == unit.team && is_alive(u) && u.id != unit_id).collect();

    let nearest_enemy = enemies.iter().min_by(|a, b| {
        distance(unit.position, a.position).partial_cmp(&distance(unit.position, b.position)).unwrap_or(std::cmp::Ordering::Equal)
    });
    let weakest_enemy = enemies.iter().min_by(|a, b| {
        let ha = a.hp as f32 / a.max_hp.max(1) as f32;
        let hb = b.hp as f32 / b.max_hp.max(1) as f32;
        ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let weakest_ally = allies.iter().min_by(|a, b| {
        let ha = a.hp as f32 / a.max_hp.max(1) as f32;
        let hb = b.hp as f32 / b.max_hp.max(1) as f32;
        ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
    });

    use bevy_game::ai::core::dataset::ActionClass;
    match class {
        ActionClass::AttackNearest => nearest_enemy.map(|e| IntentAction::Attack { target_id: e.id }),
        ActionClass::AttackWeakest => weakest_enemy.map(|e| IntentAction::Attack { target_id: e.id }),
        ActionClass::UseDamageAbility => {
            for (idx, slot) in unit.abilities.iter().enumerate() {
                if slot.cooldown_remaining_ms == 0 && slot.def.ai_hint == "damage" {
                    if let Some(e) = nearest_enemy {
                        return Some(IntentAction::UseAbility { ability_index: idx, target: AbilityTarget::Unit(e.id) });
                    }
                }
            }
            if unit.ability_damage > 0 && unit.ability_cooldown_remaining_ms == 0 {
                return nearest_enemy.map(|e| IntentAction::CastAbility { target_id: e.id });
            }
            nearest_enemy.map(|e| IntentAction::Attack { target_id: e.id })
        }
        ActionClass::UseHealAbility => {
            let target = weakest_ally.map(|a| a.id).unwrap_or(unit_id);
            for (idx, slot) in unit.abilities.iter().enumerate() {
                if slot.cooldown_remaining_ms == 0 && slot.def.ai_hint == "heal" {
                    return Some(IntentAction::UseAbility { ability_index: idx, target: AbilityTarget::Unit(target) });
                }
            }
            if unit.heal_amount > 0 && unit.heal_cooldown_remaining_ms == 0 {
                return Some(IntentAction::CastHeal { target_id: target });
            }
            None
        }
        ActionClass::UseCcAbility => {
            for (idx, slot) in unit.abilities.iter().enumerate() {
                if slot.cooldown_remaining_ms == 0 && slot.def.ai_hint == "crowd_control" {
                    if let Some(e) = nearest_enemy {
                        return Some(IntentAction::UseAbility { ability_index: idx, target: AbilityTarget::Unit(e.id) });
                    }
                }
            }
            if unit.control_duration_ms > 0 && unit.control_cooldown_remaining_ms == 0 {
                return nearest_enemy.map(|e| IntentAction::CastControl { target_id: e.id });
            }
            None
        }
        ActionClass::UseDefenseAbility => {
            for (idx, slot) in unit.abilities.iter().enumerate() {
                if slot.cooldown_remaining_ms == 0 && slot.def.ai_hint == "defense" {
                    return Some(IntentAction::UseAbility { ability_index: idx, target: AbilityTarget::Unit(unit_id) });
                }
            }
            None
        }
        ActionClass::UseUtilityAbility => {
            for (idx, slot) in unit.abilities.iter().enumerate() {
                if slot.cooldown_remaining_ms == 0 && slot.def.ai_hint == "utility" {
                    return Some(IntentAction::UseAbility { ability_index: idx, target: AbilityTarget::None });
                }
            }
            None
        }
        ActionClass::MoveToward => {
            nearest_enemy.map(|e| {
                let desired = position_at_range(unit.position, e.position, unit.attack_range * 0.9);
                let next = move_towards(unit.position, desired, unit.move_speed_per_sec * 0.1);
                IntentAction::MoveTo { position: next }
            })
        }
        ActionClass::MoveAway => {
            nearest_enemy.map(|e| {
                let away = bevy_game::ai::core::move_away(unit.position, e.position, unit.move_speed_per_sec * 0.1);
                IntentAction::MoveTo { position: away }
            })
        }
        ActionClass::Hold => Some(IntentAction::Hold),
    }
}

pub fn run_ability_encoder_export(args: crate::cli::AbilityEncoderExportArgs) -> ExitCode {
    use std::path::PathBuf;
    use bevy_game::ai::core::ability_encoding::{
        extract_ability_properties, ability_category_label, ABILITY_PROP_DIM,
    };
    use bevy_game::ai::effects::AbilityTargeting;
    use bevy_game::mission::hero_templates::parse_hero_toml;
    use serde::Serialize;

    #[derive(Serialize)]
    struct ExportRow {
        hero_name: String,
        ability_name: String,
        category: String,
        category_index: usize,
        targeting_index: usize,
        properties: Vec<f32>,
    }

    #[derive(Serialize)]
    struct ExportData {
        prop_dim: usize,
        num_categories: usize,
        num_targeting: usize,
        abilities: Vec<ExportRow>,
    }

    let mut rows = Vec::new();

    // Load hero_templates
    let hero_dir = PathBuf::from("assets/hero_templates");
    if hero_dir.is_dir() {
        let paths = collect_toml_paths(&hero_dir);
        for path in &paths {
            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(e) => { eprintln!("  skip {}: {e}", path.display()); continue; }
            };
            let toml = match parse_hero_toml(&content) {
                Ok(t) => t,
                Err(e) => { eprintln!("  skip {}: {e}", path.display()); continue; }
            };
            for def in &toml.abilities {
                let cat = ability_category_label(def);
                let props = extract_ability_properties(def);
                let tgt_idx = match def.targeting {
                    AbilityTargeting::TargetEnemy => 0,
                    AbilityTargeting::TargetAlly => 1,
                    AbilityTargeting::SelfCast => 2,
                    AbilityTargeting::SelfAoe => 3,
                    AbilityTargeting::GroundTarget => 4,
                    AbilityTargeting::Direction => 5,
                    AbilityTargeting::Vector => 6,
                    AbilityTargeting::Global => 7,
                };
                rows.push(ExportRow {
                    hero_name: toml.hero.name.clone(),
                    ability_name: def.name.clone(),
                    category: cat.name().to_string(),
                    category_index: cat as usize,
                    targeting_index: tgt_idx,
                    properties: props.to_vec(),
                });
            }
        }
    }

    // Load lol_heroes
    if args.include_lol {
        let lol_dir = PathBuf::from("assets/lol_heroes");
        if lol_dir.is_dir() {
            let paths = collect_toml_paths(&lol_dir);
            for path in &paths {
                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(e) => { eprintln!("  skip {}: {e}", path.display()); continue; }
                };
                let toml = match parse_hero_toml(&content) {
                    Ok(t) => t,
                    Err(e) => { eprintln!("  skip {}: {e}", path.display()); continue; }
                };
                for def in &toml.abilities {
                    let cat = ability_category_label(def);
                    let props = extract_ability_properties(def);
                    let tgt_idx = match def.targeting {
                        AbilityTargeting::TargetEnemy => 0,
                        AbilityTargeting::TargetAlly => 1,
                        AbilityTargeting::SelfCast => 2,
                        AbilityTargeting::SelfAoe => 3,
                        AbilityTargeting::GroundTarget => 4,
                        AbilityTargeting::Direction => 5,
                        AbilityTargeting::Vector => 6,
                        AbilityTargeting::Global => 7,
                    };
                    rows.push(ExportRow {
                        hero_name: toml.hero.name.clone(),
                        ability_name: def.name.clone(),
                        category: cat.name().to_string(),
                        category_index: cat as usize,
                        targeting_index: tgt_idx,
                        properties: props.to_vec(),
                    });
                }
            }
        }
    }

    let data = ExportData {
        prop_dim: ABILITY_PROP_DIM,
        num_categories: 9,
        num_targeting: 8,
        abilities: rows,
    };

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let json = serde_json::to_string_pretty(&data).expect("serialize");
    std::fs::write(&args.output, &json).expect("write");

    eprintln!("Exported {} abilities ({} features each) to {}",
        data.abilities.len(), ABILITY_PROP_DIM, args.output.display());

    // Category distribution
    let mut cat_counts = [0u32; 9];
    for r in &data.abilities {
        cat_counts[r.category_index] += 1;
    }
    let cat_names = [
        "damage_unit", "damage_aoe", "cc_unit", "heal_unit", "heal_aoe",
        "defense", "utility", "summon", "obstacle",
    ];
    eprintln!("\nCategory distribution:");
    for (i, name) in cat_names.iter().enumerate() {
        eprintln!("  {:<15} {:>4}", name, cat_counts[i]);
    }

    ExitCode::SUCCESS
}
