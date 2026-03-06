use super::*;
use crate::ai::core::{FIXED_TICK_MS, IntentAction, ReplayResult, SimEvent, Team, run_replay, step};
use std::collections::HashMap;

fn run_phase2_with_seed(seed: u64) -> ReplayResult {
    run_phase2_sample(seed, 260, FIXED_TICK_MS)
}

#[test]
fn phase2_is_deterministic() {
    let a = run_phase2_with_seed(17);
    let b = run_phase2_with_seed(17);
    assert_eq!(a.event_log_hash, b.event_log_hash);
    assert_eq!(a.final_state_hash, b.final_state_hash);
}

#[test]
fn healers_perform_healing() {
    let result = run_phase2_with_seed(17);
    assert!(result.metrics.heals_started > 0);
    assert!(result.metrics.heals_completed > 0);
    assert!(!result.metrics.total_healing_by_unit.is_empty());
}

#[test]
fn tanks_absorb_pressure_better_than_dps() {
    let result = run_phase2_with_seed(17);
    let taken = result
        .metrics
        .damage_taken_by_unit
        .iter()
        .copied()
        .collect::<HashMap<_, _>>();
    let hero_tank = *taken.get(&1).unwrap_or(&0);
    let hero_dps = *taken.get(&2).unwrap_or(&0);
    let enemy_tank = *taken.get(&4).unwrap_or(&0);
    let enemy_dps = *taken.get(&5).unwrap_or(&0);

    assert!(hero_tank >= hero_dps || enemy_tank >= enemy_dps);
}

#[test]
fn phase2_competent_small_party_fight() {
    let result = run_phase2_with_seed(17);
    assert_eq!(result.metrics.invariant_violations, 0);
    assert!(result.metrics.casts_completed + result.metrics.heals_completed > 0);
}

#[test]
fn phase2_regression_snapshot() {
    let result = run_phase2_with_seed(17);
    assert_eq!(result.event_log_hash, 0x0122_c661_c895_0a47);
    assert_eq!(result.final_state_hash, 0xd403_2cbd_c589_780e);
    assert_eq!(result.metrics.winner, None);
    assert_eq!(
        result.metrics.final_hp_by_unit,
        vec![(1, 36), (2, 84), (3, 51), (4, 0), (5, 0), (6, 47)]
    );
    assert_eq!(result.metrics.heals_completed, 11);
    assert_eq!(result.metrics.invariant_violations, 0);
}

#[test]
fn phase2_multi_seed_metric_bands() {
    let seeds = [11_u64, 13, 17, 19, 23, 29];
    let mut hero_wins = 0_u32;
    let mut total_ttk_ticks = 0_u32;
    let mut fights_with_death = 0_u32;
    let mut total_heals = 0_u32;

    for seed in seeds {
        let result = run_phase2_with_seed(seed);
        if result.metrics.winner == Some(Team::Hero) {
            hero_wins += 1;
        }
        if let Some(ttk) = result.metrics.tick_to_first_death {
            total_ttk_ticks += ttk as u32;
            fights_with_death += 1;
        }
        total_heals += result.metrics.heals_completed;
        assert_eq!(result.metrics.invariant_violations, 0);
    }

    let hero_win_rate = hero_wins as f32 / 6.0;
    let avg_ttk = if fights_with_death == 0 {
        0.0
    } else {
        total_ttk_ticks as f32 / fights_with_death as f32
    };

    assert!(hero_win_rate >= 0.30 && hero_win_rate <= 1.00);
    assert!(avg_ttk > 8.0 && avg_ttk < 180.0);
    assert!(total_heals >= 10);
}

#[test]
fn healer_targets_allies_only() {
    let initial = sample_phase2_party_state(31);
    let roles = default_roles();
    let mut ai = RoleAiState::new(&initial, roles);
    let mut state = initial;

    for _ in 0..140 {
        let intents = generate_intents(&state, &mut ai, FIXED_TICK_MS);
        for intent in &intents {
            if let IntentAction::CastHeal { target_id } = intent.action {
                let src = state.units.iter().find(|u| u.id == intent.unit_id).unwrap();
                let dst = state.units.iter().find(|u| u.id == target_id).unwrap();
                assert_eq!(src.team, dst.team);
            }
        }
        let (new_state, events) = step(state, &intents, FIXED_TICK_MS);
        ai.update_from_events(&events);
        state = new_state;
    }
}

#[test]
fn tank_gets_first_contact_often() {
    let seeds = [41_u64, 43, 47, 53, 59];
    let mut tank_first_hits = 0_u32;
    let mut checked = 0_u32;

    for seed in seeds {
        let initial = sample_phase2_party_state(seed);
        let roles = default_roles();
        let mut ai = RoleAiState::new(&initial, roles.clone());
        let mut state = initial;
        let mut first_source: Option<u32> = None;

        for _ in 0..90 {
            let intents = generate_intents(&state, &mut ai, FIXED_TICK_MS);
            let (new_state, events) = step(state, &intents, FIXED_TICK_MS);
            if first_source.is_none() {
                first_source = events.iter().find_map(|event| match event {
                    SimEvent::DamageApplied { source_id, .. } => Some(*source_id),
                    _ => None,
                });
            }
            ai.update_from_events(&events);
            state = new_state;
            if first_source.is_some() {
                break;
            }
        }

        if let Some(source_id) = first_source {
            checked += 1;
            if roles.get(&source_id) == Some(&Role::Tank) {
                tank_first_hits += 1;
            }
        }
    }

    assert!(checked > 0);
    let ratio = tank_first_hits as f32 / checked as f32;
    assert!((0.0..=1.0).contains(&ratio));
}

#[test]
fn dps_does_not_retarget_too_frequently() {
    let initial = sample_phase2_party_state(73);
    let roles = default_roles();
    let mut ai = RoleAiState::new(&initial, roles.clone());
    let mut state = initial;
    let dps_ids = [2_u32, 5_u32];
    let mut last_target_by_dps: HashMap<u32, u32> = HashMap::new();
    let mut retargets = 0_u32;

    for _ in 0..160 {
        let intents = generate_intents(&state, &mut ai, FIXED_TICK_MS);
        for intent in &intents {
            if !dps_ids.contains(&intent.unit_id) {
                continue;
            }
            let maybe_target = match intent.action {
                IntentAction::Attack { target_id } => Some(target_id),
                IntentAction::CastAbility { target_id } => Some(target_id),
                _ => None,
            };
            if let Some(target_id) = maybe_target {
                if let Some(last_target) = last_target_by_dps.get(&intent.unit_id) {
                    if *last_target != target_id {
                        retargets += 1;
                    }
                }
                last_target_by_dps.insert(intent.unit_id, target_id);
            }
        }
        let (new_state, events) = step(state, &intents, FIXED_TICK_MS);
        ai.update_from_events(&events);
        state = new_state;
    }

    assert!(retargets <= 24);
}

#[test]
fn no_enemy_units_results_in_hold_actions() {
    let mut state = sample_phase2_party_state(61);
    state.units.retain(|u| u.team == Team::Hero);
    let mut ai = RoleAiState::new(&state, default_roles());
    let intents = generate_intents(&state, &mut ai, FIXED_TICK_MS);
    assert!(intents
        .iter()
        .all(|i| matches!(i.action, IntentAction::Hold)));
}

#[test]
fn healer_with_no_alive_allies_does_not_cast_heal() {
    let mut state = sample_phase2_party_state(67);
    for unit in &mut state.units {
        if unit.id != 3 && unit.team == Team::Hero {
            unit.hp = 0;
        }
    }
    let mut ai = RoleAiState::new(&state, default_roles());
    let intents = generate_intents(&state, &mut ai, FIXED_TICK_MS);
    let healer_intent = intents.iter().find(|intent| intent.unit_id == 3).unwrap();
    assert!(!matches!(
        healer_intent.action,
        IntentAction::CastHeal { .. }
    ));
}

#[test]
fn out_of_range_healer_prefers_move_to_heal_target() {
    let mut state = sample_phase2_party_state(71);
    {
        let healer = state.units.iter_mut().find(|u| u.id == 3).unwrap();
        healer.position = crate::ai::core::sim_vec2(-12.0, 0.0);
    }
    {
        let tank = state.units.iter_mut().find(|u| u.id == 1).unwrap();
        tank.hp = 25;
    }
    let mut ai = RoleAiState::new(&state, default_roles());
    let intents = generate_intents(&state, &mut ai, FIXED_TICK_MS);
    let healer_intent = intents.iter().find(|intent| intent.unit_id == 3).unwrap();
    assert!(matches!(healer_intent.action, IntentAction::MoveTo { .. }));
}

#[test]
fn small_param_mutation_changes_hash() {
    let baseline = run_phase2_with_seed(17);
    let mut altered = sample_phase2_party_state(17);
    let enemy_dps = altered.units.iter_mut().find(|u| u.id == 5).unwrap();
    enemy_dps.attack_damage += 1;
    let script = generate_scripted_intents(&altered, 260, FIXED_TICK_MS, default_roles());
    let mutated = run_replay(altered, &script, 260, FIXED_TICK_MS);
    assert_ne!(baseline.event_log_hash, mutated.event_log_hash);
}

#[test]
fn fuzz_invariants_hold_across_seed_sweep() {
    for seed in 80_u64..90_u64 {
        let result = run_phase2_with_seed(seed);
        assert_eq!(result.metrics.invariant_violations, 0);
        assert_eq!(result.metrics.dead_source_attack_intents, 0);
        assert_eq!(result.per_tick_state_hashes.len(), 260);
    }
}
