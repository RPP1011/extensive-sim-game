use super::*;
use super::replay::{Phase3Run, generate_scripted_intents_inferred};
use super::personality::infer_personality;
use super::state::DriftTrigger;
use super::forces::DominantForce;
use crate::ai::core::FIXED_TICK_MS;
use std::collections::{HashMap, HashSet};

fn run_phase3_with_seed(seed: u64) -> Phase3Run {
    run_phase3_sample(seed, 280, FIXED_TICK_MS)
}

#[test]
fn phase3_is_deterministic() {
    let a = run_phase3_with_seed(23);
    let b = run_phase3_with_seed(23);
    assert_eq!(a.replay.event_log_hash, b.replay.event_log_hash);
    assert_eq!(a.replay.final_state_hash, b.replay.final_state_hash);
    assert_eq!(
        a.replay.per_tick_state_hashes,
        b.replay.per_tick_state_hashes
    );
}

#[test]
fn blackboard_focus_drives_target_coherence() {
    let initial = sample_phase3_party_state(23);
    let (script, history) =
        generate_scripted_intents_inferred(&initial, 120, FIXED_TICK_MS);

    let mut focused_actions = 0_u32;
    let mut total_offensive = 0_u32;
    for (tick, intents) in script.iter().enumerate() {
        let boards = &history[tick];
        for intent in intents {
            let Some(unit) = initial.units.iter().find(|u| u.id == intent.unit_id) else {
                continue;
            };
            let focus = boards.get(&unit.team).and_then(|b| b.focus_target);
            let target = match intent.action {
                crate::ai::core::IntentAction::Attack { target_id } => Some(target_id),
                crate::ai::core::IntentAction::CastAbility { target_id } => Some(target_id),
                _ => None,
            };
            if let Some(target_id) = target {
                total_offensive += 1;
                if Some(target_id) == focus {
                    focused_actions += 1;
                }
            }
        }
    }

    assert!(total_offensive > 0);
    assert!(
        (focused_actions as f32 / total_offensive as f32) >= 0.55,
        "focus ratio {} too low",
        focused_actions as f32 / total_offensive as f32
    );
}

#[test]
fn squad_mode_changes_over_time() {
    let run = run_phase3_with_seed(23);
    let mut hero_modes = run
        .board_history
        .iter()
        .filter_map(|m| m.get(&crate::ai::core::Team::Hero).map(|b| b.mode))
        .collect::<Vec<_>>();
    hero_modes.dedup();
    assert!(hero_modes.len() >= 2);
}

#[test]
fn phase3_competent_party_behavior() {
    let run = run_phase3_with_seed(23);
    assert!(run.replay.metrics.winner.is_some());
    assert_eq!(run.replay.metrics.invariant_violations, 0);
    assert!(run.replay.metrics.casts_completed + run.replay.metrics.heals_completed > 0);
}

#[test]
fn blackboard_updates_only_on_eval_cadence() {
    let run = run_phase3_with_seed(23);
    for tick in 1..run.board_history.len() {
        if tick % 5 != 0 {
            assert_eq!(run.board_history[tick], run.board_history[tick - 1]);
        }
    }
}

#[test]
fn phase3_multi_seed_metrics_bands() {
    let seeds = [23_u64, 29, 31, 37, 41, 43];
    let mut hero_wins = 0_u32;
    let mut focus_ratio_sum = 0.0_f32;

    for seed in seeds {
        let initial = sample_phase3_party_state(seed);
        let (script, history) =
            generate_scripted_intents_inferred(&initial, 200, FIXED_TICK_MS);
        let replay = crate::ai::core::run_replay(initial.clone(), &script, 200, FIXED_TICK_MS);
        if replay.metrics.winner == Some(crate::ai::core::Team::Hero) {
            hero_wins += 1;
        }
        assert_eq!(replay.metrics.invariant_violations, 0);

        let mut focused_actions = 0_u32;
        let mut offensive_actions = 0_u32;
        for (tick, intents) in script.iter().enumerate() {
            let boards = &history[tick];
            for intent in intents {
                let Some(unit) = initial.units.iter().find(|u| u.id == intent.unit_id) else {
                    continue;
                };
                let focus = boards.get(&unit.team).and_then(|b| b.focus_target);
                let target = match intent.action {
                    crate::ai::core::IntentAction::Attack { target_id } => Some(target_id),
                    crate::ai::core::IntentAction::CastAbility { target_id } => Some(target_id),
                    _ => None,
                };
                if let Some(target_id) = target {
                    offensive_actions += 1;
                    if Some(target_id) == focus {
                        focused_actions += 1;
                    }
                }
            }
        }
        if offensive_actions > 0 {
            focus_ratio_sum += focused_actions as f32 / offensive_actions as f32;
        }
    }

    let hero_win_rate = hero_wins as f32 / 6.0;
    let avg_focus_ratio = focus_ratio_sum / 6.0;
    assert!(hero_win_rate >= 0.35 && hero_win_rate <= 1.0,
        "hero win rate {} out of band", hero_win_rate);
    assert!(avg_focus_ratio >= 0.45,
        "avg focus ratio {} too low", avg_focus_ratio);
}

#[test]
fn tie_break_target_selection_is_deterministic() {
    let mut state = sample_phase3_party_state(55);
    {
        let enemy4 = state.units.iter_mut().find(|u| u.id == 4).unwrap();
        enemy4.hp = 100;
        enemy4.position = crate::ai::core::sim_vec2(8.0, 0.0);
    }
    {
        let enemy5 = state.units.iter_mut().find(|u| u.id == 5).unwrap();
        enemy5.hp = 100;
        enemy5.position = crate::ai::core::sim_vec2(8.0, 0.0);
    }

    let mut ai = SquadAiState::new_inferred(&state);
    let intents_a = generate_intents(&state, &mut ai, FIXED_TICK_MS);
    let intents_b = generate_intents(&state, &mut ai, FIXED_TICK_MS);

    let pick_targets = |intents: &[crate::ai::core::UnitIntent]| -> HashSet<u32> {
        intents
            .iter()
            .filter_map(|intent| match intent.action {
                crate::ai::core::IntentAction::Attack { target_id } => Some(target_id),
                crate::ai::core::IntentAction::CastAbility { target_id } => Some(target_id),
                _ => None,
            })
            .collect()
    };

    assert_eq!(pick_targets(&intents_a), pick_targets(&intents_b));
}

#[test]
fn no_enemies_produces_hold_for_entire_squad() {
    let mut state = sample_phase3_party_state(57);
    state.units.retain(|u| u.team == crate::ai::core::Team::Hero);
    let mut ai = SquadAiState::new_inferred(&state);
    let intents = generate_intents(&state, &mut ai, FIXED_TICK_MS);
    assert!(intents
        .iter()
        .all(|intent| matches!(intent.action, crate::ai::core::IntentAction::Hold)));
}

#[test]
fn hash_changes_with_small_parameter_mutation() {
    let baseline = run_phase3_with_seed(23);
    let mut initial = sample_phase3_party_state(23);
    let hero_tank = initial.units.iter_mut().find(|u| u.id == 1).unwrap();
    hero_tank.attack_damage += 1;
    let (script, _) = generate_scripted_intents_inferred(&initial, 280, FIXED_TICK_MS);
    let mutated = crate::ai::core::run_replay(initial, &script, 280, FIXED_TICK_MS);
    assert_ne!(baseline.replay.event_log_hash, mutated.event_log_hash);
}

#[test]
fn fuzz_invariants_hold_for_phase3_seed_sweep() {
    for seed in 70_u64..80_u64 {
        let run = run_phase3_with_seed(seed);
        assert_eq!(run.replay.metrics.invariant_violations, 0);
        assert_eq!(run.replay.metrics.dead_source_attack_intents, 0);
        assert_eq!(run.replay.per_tick_state_hashes.len(), 280);
    }
}

// --- Personality tests ---

#[test]
fn personality_inference_ranges() {
    use crate::ai::core::sim_vec2;

    // Tanky melee -- high aggression, low caution
    let mut tank = crate::ai::core::UnitState {
        id: 1, team: crate::ai::core::Team::Hero, hp: 180, max_hp: 180,
        position: sim_vec2(0.0, 0.0), move_speed_per_sec: 3.0,
        attack_damage: 15, attack_range: 1.5, attack_cooldown_ms: 1000,
        attack_cast_time_ms: 300, cooldown_remaining_ms: 0,
        ability_damage: 0, ability_range: 0.0, ability_cooldown_ms: 0,
        ability_cast_time_ms: 0, ability_cooldown_remaining_ms: 0,
        heal_amount: 0, heal_range: 0.0, heal_cooldown_ms: 0,
        heal_cast_time_ms: 0, heal_cooldown_remaining_ms: 0,
        control_range: 0.0, control_duration_ms: 0, control_cooldown_ms: 0,
        control_cast_time_ms: 0, control_cooldown_remaining_ms: 0,
        control_remaining_ms: 0, casting: None,
        abilities: Vec::new(), passives: Vec::new(),
        status_effects: Vec::new(), shield_hp: 0,
        resistance_tags: HashMap::new(),
        state_history: std::collections::VecDeque::new(),
        channeling: None, resource: 0, max_resource: 0,
        resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
    };

    let tank_p = infer_personality(&tank);
    assert!(tank_p.aggression >= 0.7, "tank aggression should be high: {}", tank_p.aggression);
    assert!(tank_p.caution <= 0.3, "tank caution should be low: {}", tank_p.caution);

    // Healer -- high compassion
    tank.heal_amount = 20;
    tank.max_hp = 80;
    tank.attack_range = 4.0;
    let healer_p = infer_personality(&tank);
    assert!(healer_p.compassion >= 0.7, "healer compassion should be high: {}", healer_p.compassion);
}

#[test]
fn drift_shifts_on_near_death() {
    let state = sample_phase3_party_state(23);
    let mut ai = SquadAiState::new_inferred(&state);
    let base_aggression = ai.personality_for(1).aggression;

    for _ in 0..5 {
        ai.apply_drift(1, DriftTrigger::SurvivedLowHp);
    }

    let after_aggression = ai.personality_for(1).aggression;
    assert!(after_aggression > base_aggression,
        "aggression should rise after near-death: before={}, after={}",
        base_aggression, after_aggression);
}

#[test]
fn force_dominant_heal_when_ally_critical() {
    let mut state = sample_phase3_party_state(23);
    let unit3 = state.units.iter_mut().find(|u| u.id == 3).unwrap();
    unit3.heal_amount = 20;
    unit3.heal_range = 5.0;
    unit3.heal_cooldown_ms = 2000;

    let unit1 = state.units.iter_mut().find(|u| u.id == 1).unwrap();
    unit1.hp = 10;

    let healer_personality = Personality {
        aggression: 0.2,
        compassion: 0.9,
        caution: 0.6,
        discipline: 0.5,
        cunning: 0.3,
        tenacity: 0.3,
        patience: 0.7,
    };

    let unit3 = state.units.iter().find(|u| u.id == 3).unwrap();
    let board = SquadBlackboard { focus_target: None, mode: FormationMode::Hold };
    let ctx = super::state::TickContext::new(&state);
    let raw = super::forces::compute_raw_forces(&state, unit3, &board, &ctx, &healer_personality);
    let weighted = super::forces::weighted_forces(&raw, &healer_personality);
    let force = super::forces::dominant_force(&weighted);

    assert_eq!(force, DominantForce::Heal,
        "heal-capable unit should prioritize healing when ally is critical");
}

#[test]
fn attack_dominates_during_approach() {
    let initial = sample_phase3_party_state(23);
    let ai = SquadAiState::new_inferred(&initial);

    let unit = initial.units.iter().find(|u| u.team == crate::ai::core::Team::Hero).unwrap();
    let p = ai.personality_for(unit.id);
    let board = SquadBlackboard { focus_target: None, mode: FormationMode::Hold };
    let ctx = super::state::TickContext::new(&initial);
    let raw = super::forces::compute_raw_forces(&initial, unit, &board, &ctx, &p);
    let weighted = super::forces::weighted_forces(&raw, &p);
    let dom = super::forces::dominant_force(&weighted);

    assert!(
        matches!(dom, DominantForce::Attack | DominantForce::Focus),
        "melee hero should want to attack/focus on approach, got {:?}",
        dom
    );
    assert!(weighted.attack > weighted.position,
        "attack ({:.2}) must beat position ({:.2}) during approach",
        weighted.attack, weighted.position);
}
