use super::*;

#[test]
fn replay_hash_is_stable_for_same_seed() {
    let ticks = 120;
    let script = sample_duel_script(ticks);
    let result_a = run_replay(sample_duel_state(7), &script, ticks, FIXED_TICK_MS);
    let result_b = run_replay(sample_duel_state(7), &script, ticks, FIXED_TICK_MS);
    assert_eq!(result_a.event_log_hash, result_b.event_log_hash);
    assert_eq!(result_a.final_state_hash, result_b.final_state_hash);
    assert_eq!(result_a.per_tick_state_hashes, result_b.per_tick_state_hashes);
    assert_eq!(result_a.events, result_b.events);
}

#[test]
fn replay_hash_changes_with_different_seed() {
    let ticks = 120;
    let script = sample_duel_script(ticks);
    let result_a = run_replay(sample_duel_state(7), &script, ticks, FIXED_TICK_MS);
    let result_b = run_replay(sample_duel_state(13), &script, ticks, FIXED_TICK_MS);
    assert_ne!(result_a.event_log_hash, result_b.event_log_hash);
    assert_ne!(result_a.final_state_hash, result_b.final_state_hash);
}

#[test]
fn attack_requires_range_and_uses_movement() {
    let mut u1 = hero_unit(1, Team::Hero, (0.0, 0.0));
    u1.move_speed_per_sec = 5.0;
    u1.attack_range = 1.0; u1.attack_cooldown_ms = 500; u1.attack_cast_time_ms = 200;
    let mut u2 = hero_unit(2, Team::Enemy, (10.0, 0.0));
    u2.move_speed_per_sec = 0.0; u2.attack_damage = 0;
    let initial = make_state(vec![u1, u2], 1);
    let intent = [UnitIntent { unit_id: 1, action: IntentAction::Attack { target_id: 2 } }];
    let (_, events) = step(initial, &intent, FIXED_TICK_MS);
    assert!(events.iter().any(|e| matches!(e, SimEvent::Moved { .. })));
    assert!(events.iter().any(|e| matches!(e, SimEvent::AttackRepositioned { .. })));
    assert!(!events.iter().any(|e| matches!(e, SimEvent::CastStarted { .. })));
}

#[test]
fn metrics_include_core_verification_signals() {
    let ticks = 120;
    let script = sample_duel_script(ticks);
    let result = run_replay(sample_duel_state(7), &script, ticks, FIXED_TICK_MS);
    assert_eq!(result.metrics.ticks_elapsed, ticks);
    assert!(result.metrics.seconds_elapsed > 0.0);
    assert!(result.metrics.attack_intents > 0);
    assert!(result.metrics.casts_started > 0);
    assert!(result.metrics.casts_completed > 0);
    assert!(result.metrics.avg_cast_delay_ms > 0.0);
    assert!(!result.metrics.dps_by_unit.is_empty());
    assert_eq!(result.metrics.invariant_violations, 0);
    assert!(!result.metrics.final_hp_by_unit.is_empty());
}

#[test]
fn sample_duel_regression_snapshot() {
    let ticks = 120;
    let script = sample_duel_script(ticks);
    let result = run_replay(sample_duel_state(7), &script, ticks, FIXED_TICK_MS);
    assert_eq!(result.event_log_hash, 0xcaa9_9255_8277_ba4d);
    assert_eq!(result.final_state_hash, 0xf7f0_48a8_2dd7_b88e);
    assert_eq!(result.metrics.winner, Some(Team::Hero));
    assert_eq!(result.metrics.tick_to_first_death, Some(73));
    assert_eq!(result.metrics.final_hp_by_unit, vec![(1, 37), (2, 0)]);
    assert_eq!(result.metrics.invariant_violations, 0);
}

#[test]
fn small_param_mutation_changes_hash() {
    let ticks = 120;
    let script = sample_duel_script(ticks);
    let baseline = run_replay(sample_duel_state(7), &script, ticks, FIXED_TICK_MS);
    let mut altered = sample_duel_state(7);
    altered.units[0].attack_damage += 1;
    let mutated = run_replay(altered, &script, ticks, FIXED_TICK_MS);
    assert_ne!(baseline.event_log_hash, mutated.event_log_hash);
}

#[test]
fn deterministic_tie_break_for_identical_targets() {
    let u1 = {
        let mut u = hero_unit(1, Team::Hero, (0.0, 0.0));
        u.move_speed_per_sec = 0.0; u.attack_range = 2.0; u.attack_cooldown_ms = 0; u.attack_cast_time_ms = 0;
        u
    };
    let u2 = { let mut u = hero_unit(2, Team::Enemy, (1.0, 0.0)); u.hp = 40; u.max_hp = 40; u.move_speed_per_sec = 0.0; u.attack_damage = 0; u };
    let u3 = { let mut u = hero_unit(3, Team::Enemy, (1.0, 0.0)); u.hp = 40; u.max_hp = 40; u.move_speed_per_sec = 0.0; u.attack_damage = 0; u };
    let state = make_state(vec![u1, u2, u3], 5);
    let intent = [UnitIntent { unit_id: 1, action: IntentAction::Attack { target_id: 2 } }];
    let (_, events_a) = step(state.clone(), &intent, FIXED_TICK_MS);
    let (_, events_b) = step(state, &intent, FIXED_TICK_MS);
    assert_eq!(events_a, events_b);
}

#[test]
fn fuzz_invariants_hold_across_seed_sweep() {
    for seed in 1_u64..16_u64 {
        let ticks = 80;
        let script = sample_duel_script(ticks);
        let result = run_replay(sample_duel_state(seed), &script, ticks, FIXED_TICK_MS);
        assert_eq!(result.metrics.invariant_violations, 0);
        assert_eq!(result.per_tick_state_hashes.len(), ticks as usize);
    }
}

#[test]
fn control_cast_locks_target_actions_temporarily() {
    let mut u1 = hero_unit(1, Team::Hero, (0.0, 0.0));
    u1.control_range = 2.0; u1.control_duration_ms = 350; u1.control_cooldown_ms = 1_000;
    u1.control_cast_time_ms = 0; u1.move_speed_per_sec = 0.0;
    let mut u2 = hero_unit(2, Team::Enemy, (1.0, 0.0));
    u2.attack_range = 2.0; u2.move_speed_per_sec = 0.0;
    let mut state = make_state(vec![u1, u2], 11);
    state.units.sort_by_key(|u| u.id);
    let intents_t1 = vec![
        UnitIntent { unit_id: 1, action: IntentAction::CastControl { target_id: 2 } },
        UnitIntent { unit_id: 2, action: IntentAction::Attack { target_id: 1 } },
    ];
    let (state, events_t1) = step(state, &intents_t1, FIXED_TICK_MS);
    assert!(events_t1.iter().any(|e| matches!(e, SimEvent::ControlCastStarted { .. })));
    let intents_t2 = vec![UnitIntent { unit_id: 2, action: IntentAction::Attack { target_id: 1 } }];
    let (_, events_t2) = step(state, &intents_t2, FIXED_TICK_MS);
    assert!(events_t2.iter().any(|e| matches!(e, SimEvent::ControlApplied { target_id: 2, .. })));
    assert!(events_t2.iter().any(|e| matches!(e, SimEvent::UnitControlled { unit_id: 2, .. })));
}

#[test]
fn tag_resistance_blocks_effect() {
    use crate::ai::effects::Tags;
    let mut effect_tags: Tags = HashMap::new();
    effect_tags.insert("CROWD_CONTROL".into(), 80.0);
    let mut resistance_tags: Tags = HashMap::new();
    resistance_tags.insert("CROWD_CONTROL".into(), 90.0);
    assert!(check_tags_resisted(&effect_tags, &resistance_tags));
    let mut weak_resist: Tags = HashMap::new();
    weak_resist.insert("CROWD_CONTROL".into(), 50.0);
    assert!(!check_tags_resisted(&effect_tags, &weak_resist));
}
