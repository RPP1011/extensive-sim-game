use super::*;
use crate::ai::core::FIXED_TICK_MS;
use std::collections::HashMap;
use std::collections::HashSet;

fn run_phase5(seed: u64) -> Phase5Run {
    run_phase5_sample(seed, 320, FIXED_TICK_MS)
}

fn all_vanguard() -> HashMap<u32, PersonalityProfile> {
    (1_u32..=6_u32)
        .map(|id| (id, PersonalityProfile::vanguard()))
        .collect()
}

fn all_guardian() -> HashMap<u32, PersonalityProfile> {
    (1_u32..=6_u32)
        .map(|id| (id, PersonalityProfile::guardian()))
        .collect()
}

fn all_tactician() -> HashMap<u32, PersonalityProfile> {
    (1_u32..=6_u32)
        .map(|id| (id, PersonalityProfile::tactician()))
        .collect()
}

fn hero_aggressive_enemy_defensive() -> HashMap<u32, PersonalityProfile> {
    HashMap::from([
        (1, PersonalityProfile::vanguard()),
        (2, PersonalityProfile::vanguard()),
        (3, PersonalityProfile::vanguard()),
        (4, PersonalityProfile::guardian()),
        (5, PersonalityProfile::guardian()),
        (6, PersonalityProfile::guardian()),
    ])
}

fn hero_controller_enemy_aggressive() -> HashMap<u32, PersonalityProfile> {
    HashMap::from([
        (1, PersonalityProfile::tactician()),
        (2, PersonalityProfile::guardian()),
        (3, PersonalityProfile::tactician()),
        (4, PersonalityProfile::vanguard()),
        (5, PersonalityProfile::vanguard()),
        (6, PersonalityProfile::vanguard()),
    ])
}

fn run_profile(seed: u64, profile: HashMap<u32, PersonalityProfile>) -> Phase5Run {
    run_phase5_with_personality_overrides(seed, 320, FIXED_TICK_MS, profile)
}

#[test]
fn phase5_is_deterministic() {
    let a = run_phase5(31);
    let b = run_phase5(31);
    assert_eq!(a.replay.event_log_hash, b.replay.event_log_hash);
    assert_eq!(a.replay.final_state_hash, b.replay.final_state_hash);
}

#[test]
fn mode_state_machine_transitions_exist() {
    let run = run_phase5(31);
    let mut changes = 0_u32;
    let mut prev: Option<Vec<(u32, UnitMode)>> = None;
    for snapshot in &run.mode_history {
        if let Some(p) = &prev {
            if p != snapshot {
                changes += 1;
            }
        }
        prev = Some(snapshot.clone());
    }
    assert!(changes >= 2);
}

#[test]
fn personalities_produce_different_outcomes() {
    let base = run_phase5(31);
    let alt = run_profile(31, all_vanguard());

    assert_ne!(base.replay.event_log_hash, alt.replay.event_log_hash);
}

#[test]
fn phase5_competent_and_safe() {
    let run = run_phase5(31);
    let any_damage = run
        .replay
        .metrics
        .total_damage_by_unit
        .iter()
        .map(|(_, dmg)| *dmg)
        .sum::<i32>()
        > 0;
    assert!(any_damage);
    assert_eq!(run.replay.metrics.invariant_violations, 0);
    assert!(run.replay.metrics.casts_completed + run.replay.metrics.heals_completed > 0);
}

#[test]
fn phase5_regression_snapshot() {
    let run = run_phase5(31);
    assert_eq!(run.replay.event_log_hash, 0x8d03_5a97_0ee4_d0d0);
    assert_eq!(run.replay.final_state_hash, 0x5f4d_5cfd_8a52_119e);
    assert_eq!(run.replay.metrics.winner, Some(crate::ai::core::Team::Hero));
    assert_eq!(
        run.replay.metrics.final_hp_by_unit,
        vec![(1, 130), (2, 32), (3, 67), (4, 0), (5, 0), (6, 0)]
    );
    assert_eq!(run.replay.metrics.invariant_violations, 0);
}

#[test]
fn personality_matrix_produces_distinct_signatures() {
    let profiles = vec![
        ("vanguard", all_vanguard()),
        ("guardian", all_guardian()),
        ("tactician", all_tactician()),
        ("hero_aggr_enemy_def", hero_aggressive_enemy_defensive()),
        ("hero_ctrl_enemy_aggr", hero_controller_enemy_aggressive()),
    ];

    let mut seen = HashSet::new();
    for (_name, profile) in profiles {
        let run = run_profile(31, profile);
        assert_eq!(run.replay.metrics.invariant_violations, 0);
        seen.insert(run.replay.event_log_hash);
    }

    assert!(seen.len() >= 4);
}

#[test]
fn each_personality_preset_is_deterministic() {
    let presets = vec![
        all_vanguard(),
        all_guardian(),
        all_tactician(),
        hero_aggressive_enemy_defensive(),
        hero_controller_enemy_aggressive(),
    ];

    for preset in presets {
        let a = run_profile(37, preset.clone());
        let b = run_profile(37, preset);
        assert_eq!(a.replay.event_log_hash, b.replay.event_log_hash);
        assert_eq!(a.replay.final_state_hash, b.replay.final_state_hash);
    }
}
