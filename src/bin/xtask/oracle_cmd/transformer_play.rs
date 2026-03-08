use std::process::ExitCode;

use super::collect_toml_paths;

pub fn run_transformer_play(args: crate::cli::TransformerPlayArgs) -> ExitCode {
    use bevy_game::ai::core::{step, is_alive, distance, move_towards, position_at_range, Team, FIXED_TICK_MS, UnitIntent, IntentAction};
    use bevy_game::ai::core::ability_eval::extract_game_state;
    use bevy_game::ai::core::ability_transformer::AbilityTransformerWeights;
    use bevy_game::ai::effects::dsl::emit::emit_ability_dsl;
    use bevy_game::ai::effects::AbilityTarget;
    use bevy_game::ai::squad::generate_intents;
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state};

    // Load transformer weights
    let weights_json = match std::fs::read_to_string(&args.weights) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read weights: {e}");
            return ExitCode::from(1);
        }
    };
    let transformer = match AbilityTransformerWeights::from_json(&weights_json) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to parse weights: {e}");
            return ExitCode::from(1);
        }
    };

    let tokenizer = bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer::new();

    eprintln!("Transformer loaded, urgency threshold: {}", args.urgency_threshold);

    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found.");
        return ExitCode::from(1);
    }

    let mut wins = 0u32;
    let mut losses = 0u32;
    let mut timeouts = 0u32;
    let mut ability_fires = 0u64;
    let mut total_decisions = 0u64;

    for toml_path in &paths {
        let scenario_file = match load_scenario_file(toml_path) {
            Ok(f) => f,
            Err(err) => { eprintln!("{err}"); continue; }
        };

        let cfg = &scenario_file.scenario;
        let (mut sim, mut squad_ai) = run_scenario_to_state(cfg);
        let hero_ids: Vec<u32> = sim.units.iter()
            .filter(|u| u.team == Team::Hero)
            .map(|u| u.id)
            .collect();

        // Pre-tokenize abilities for each hero at fight start
        // Map: (unit_id, ability_index) -> token_ids
        let mut ability_tokens: std::collections::HashMap<(u32, usize), Vec<u32>> =
            std::collections::HashMap::new();
        for &uid in &hero_ids {
            if let Some(unit) = sim.units.iter().find(|u| u.id == uid) {
                for (idx, slot) in unit.abilities.iter().enumerate() {
                    let dsl = emit_ability_dsl(&slot.def);
                    let token_ids = tokenizer.encode_with_cls(&dsl);
                    ability_tokens.insert((uid, idx), token_ids);
                }
            }
        }

        // Pre-encode CLS embeddings (static per ability)
        let mut cls_cache: std::collections::HashMap<(u32, usize), Vec<f32>> =
            std::collections::HashMap::new();
        for ((uid, idx), tokens) in &ability_tokens {
            let cls = transformer.encode_cls(tokens);
            cls_cache.insert((*uid, *idx), cls);
        }

        let mut outcome = "Timeout";

        for _ in 0..cfg.max_ticks {
            let mut intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

            for &uid in &hero_ids {
                let unit = match sim.units.iter().find(|u| u.id == uid && is_alive(u)) {
                    Some(u) => u,
                    None => continue,
                };
                if unit.casting.is_some() || unit.control_remaining_ms > 0 {
                    continue;
                }

                total_decisions += 1;

                // Extract game state and encode entities once
                let game_state = extract_game_state(&sim, unit);
                let entities = transformer.encode_entities(&game_state);

                // Evaluate all ready abilities
                let mut best_urgency = 0.0f32;
                let mut best_action: Option<IntentAction> = None;

                let enemies: Vec<_> = sim.units.iter()
                    .filter(|u| u.team != unit.team && is_alive(u))
                    .collect();
                let allies: Vec<_> = sim.units.iter()
                    .filter(|u| u.team == unit.team && is_alive(u) && u.id != uid)
                    .collect();

                // Sort enemies by distance for target selection
                let mut enemies_by_dist: Vec<_> = enemies.iter().collect();
                enemies_by_dist.sort_by(|a, b| {
                    distance(unit.position, a.position)
                        .partial_cmp(&distance(unit.position, b.position))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Sort allies by HP% for heal targeting
                let mut allies_by_hp: Vec<_> = allies.iter().collect();
                allies_by_hp.sort_by(|a, b| {
                    let ha = a.hp as f32 / a.max_hp.max(1) as f32;
                    let hb = b.hp as f32 / b.max_hp.max(1) as f32;
                    ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
                });

                for (idx, slot) in unit.abilities.iter().enumerate() {
                    if slot.cooldown_remaining_ms > 0 {
                        continue;
                    }
                    if slot.def.resource_cost > 0 && unit.resource < slot.def.resource_cost {
                        continue;
                    }

                    let cls = match cls_cache.get(&(uid, idx)) {
                        Some(c) => c,
                        None => continue,
                    };

                    let output = transformer.predict_from_cls(cls, entities.as_ref());

                    if output.urgency > best_urgency {
                        best_urgency = output.urgency;

                        // Convert target logit to actual target
                        let target_idx = output.target_logits.iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|(i, _)| i)
                            .unwrap_or(0);

                        let hint = slot.def.ai_hint.as_str();
                        let target = match hint {
                            "heal" | "heal_unit" | "heal_aoe" => {
                                // Target wounded ally (or self)
                                if let Some(ally) = allies_by_hp.get(target_idx.min(allies_by_hp.len().saturating_sub(1))) {
                                    AbilityTarget::Unit(ally.id)
                                } else {
                                    AbilityTarget::Unit(uid) // self-heal
                                }
                            }
                            "damage" | "damage_unit" | "damage_aoe" |
                            "control" | "cc" | "crowd_control" => {
                                // Target enemy
                                if let Some(enemy) = enemies_by_dist.get(target_idx.min(enemies_by_dist.len().saturating_sub(1))) {
                                    AbilityTarget::Unit(enemy.id)
                                } else {
                                    continue; // No enemies
                                }
                            }
                            "defense" | "utility" | "buff" => {
                                AbilityTarget::Unit(uid) // self-target
                            }
                            _ => {
                                // Default: target nearest enemy for offensive, self for others
                                if let Some(enemy) = enemies_by_dist.first() {
                                    AbilityTarget::Unit(enemy.id)
                                } else {
                                    AbilityTarget::Unit(uid)
                                }
                            }
                        };

                        // Check range
                        let in_range = match &target {
                            AbilityTarget::Unit(tid) => {
                                if *tid == uid {
                                    true
                                } else if let Some(target_unit) = sim.units.iter().find(|u| u.id == *tid) {
                                    distance(unit.position, target_unit.position) <= slot.def.range + 0.5
                                } else {
                                    false
                                }
                            }
                            _ => true,
                        };

                        if in_range {
                            best_action = Some(IntentAction::UseAbility {
                                ability_index: idx,
                                target,
                            });
                        } else {
                            // Out of range: move toward target
                            if let AbilityTarget::Unit(tid) = &target {
                                if let Some(target_unit) = sim.units.iter().find(|u| u.id == *tid) {
                                    let desired = position_at_range(
                                        unit.position, target_unit.position, slot.def.range * 0.9
                                    );
                                    let next = move_towards(
                                        unit.position, desired, unit.move_speed_per_sec * 0.1
                                    );
                                    best_action = Some(IntentAction::MoveTo { position: next });
                                }
                            }
                        }
                    }
                }

                // Fire if above threshold
                if best_urgency >= args.urgency_threshold {
                    if let Some(action) = best_action {
                        intents.retain(|i| i.unit_id != uid);
                        intents.push(UnitIntent { unit_id: uid, action });
                        ability_fires += 1;
                    }
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
    let ab_pct = if total_decisions > 0 { ability_fires as f64 / total_decisions as f64 * 100.0 } else { 0.0 };
    println!("Decision split: transformer={ability_fires} ({ab_pct:.1}%)  default={} ({:.1}%)",
        total_decisions - ability_fires, 100.0 - ab_pct);

    ExitCode::SUCCESS
}
