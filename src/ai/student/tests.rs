use super::*;
use super::features::extract_features;
use super::model::StudentMLP;
use crate::ai::core::{sim_vec2, SimState, Team, UnitState, UnitStore};
use crate::ai::personality::PersonalityProfile;
use crate::ai::roles::Role;
use crate::ai::squad::FormationMode;
use std::collections::{HashMap, VecDeque};

fn dummy_state() -> SimState {
    let mut units = Vec::new();
    // 4 heroes
    for i in 1..=4 {
        units.push(UnitState {
            id: i,
            team: Team::Hero,
            hp: 80,
            max_hp: 100,
            position: sim_vec2(i as f32, 0.0),
            move_speed_per_sec: 4.0,
            attack_damage: 12,
            attack_range: 1.4,
            attack_cooldown_ms: 700,
            attack_cast_time_ms: 300,
            cooldown_remaining_ms: 0,
            ability_damage: 18,
            ability_range: 1.6,
            ability_cooldown_ms: 2800,
            ability_cast_time_ms: 420,
            ability_cooldown_remaining_ms: 0,
            heal_amount: if i == 3 { 15 } else { 0 },
            heal_range: if i == 3 { 5.0 } else { 0.0 },
            heal_cooldown_ms: 2100,
            heal_cast_time_ms: 380,
            heal_cooldown_remaining_ms: 0,
            control_range: if i == 1 { 3.0 } else { 0.0 },
            control_duration_ms: if i == 1 { 1200 } else { 0 },
            control_cooldown_ms: 2000,
            control_cast_time_ms: 400,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        });
    }
    // 4 enemies
    for i in 5..=8 {
        units.push(UnitState {
            id: i,
            team: Team::Enemy,
            hp: 60,
            max_hp: 100,
            position: sim_vec2(10.0 + i as f32, 0.0),
            move_speed_per_sec: 4.0,
            attack_damage: 10,
            attack_range: 1.4,
            attack_cooldown_ms: 700,
            attack_cast_time_ms: 300,
            cooldown_remaining_ms: 0,
            ability_damage: 15,
            ability_range: 1.6,
            ability_cooldown_ms: 2800,
            ability_cast_time_ms: 420,
            ability_cooldown_remaining_ms: 0,
            heal_amount: 0,
            heal_range: 0.0,
            heal_cooldown_ms: 2100,
            heal_cast_time_ms: 380,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        });
    }
    SimState {
        tick: 50,
        rng_state: 42,
        units: UnitStore::new(units),
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    }
}

#[test]
fn feature_extraction_produces_60_floats() {
    let state = dummy_state();
    let roles = HashMap::from([
        (1, Role::Tank),
        (2, Role::Dps),
        (3, Role::Healer),
        (4, Role::Dps),
        (5, Role::Dps),
        (6, Role::Dps),
        (7, Role::Dps),
        (8, Role::Dps),
    ]);
    let personalities = HashMap::from([
        (1, PersonalityProfile::guardian()),
        (2, PersonalityProfile::vanguard()),
        (3, PersonalityProfile::guardian()),
        (4, PersonalityProfile::tactician()),
    ]);
    let features = extract_features(&state, &roles, &personalities, FormationMode::Hold);
    assert_eq!(features.len(), 60);
    for (i, v) in features.iter().enumerate() {
        assert!(v.is_finite(), "feature[{i}] is not finite: {v}");
    }
}

#[test]
fn forward_produces_valid_output() {
    // Build a model with random-ish weights
    let input_dim = 60;
    let hidden1 = 128;
    let hidden2 = 64;
    let model = StudentMLP {
        input_dim,
        hidden1,
        hidden2,
        w1: vec![0.01; input_dim * hidden1],
        b1: vec![0.0; hidden1],
        w2: vec![0.01; hidden1 * hidden2],
        b2: vec![0.0; hidden2],
        w3: vec![0.01; hidden2 * 9],
        b3: vec![0.0; 9],
    };

    let input = vec![0.5_f32; 60];
    let result = model.predict(&input);

    // Personality weights should be in [0, 1]
    assert!((0.0..=1.0).contains(&result.personality.aggression));
    assert!((0.0..=1.0).contains(&result.personality.risk_tolerance));

    // Formation probs should sum to ~1
    let prob_sum: f32 = result.formation_probs.iter().sum();
    assert!((prob_sum - 1.0).abs() < 1e-5);
}

#[test]
#[ignore] // timing-sensitive — run with --release for meaningful results
fn benchmark_inference_speed() {
    let model = StudentMLP {
        input_dim: 60,
        hidden1: 128,
        hidden2: 64,
        w1: vec![0.01; 60 * 128],
        b1: vec![0.0; 128],
        w2: vec![0.01; 128 * 64],
        b2: vec![0.0; 64],
        w3: vec![0.01; 64 * 9],
        b3: vec![0.0; 9],
    };
    let input = vec![0.5_f32; 60];

    let n = 100_000;
    let start = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(model.forward_raw(std::hint::black_box(&input)));
    }
    let elapsed = start.elapsed();
    let per_call_ns = elapsed.as_nanos() / n as u128;
    eprintln!(
        "StudentMLP: {} calls in {:?} ({} ns/call, {} calls/sec)",
        n,
        elapsed,
        per_call_ns,
        1_000_000_000 / per_call_ns.max(1)
    );
    // Should be well under 1 microsecond
    assert!(per_call_ns < 5_000, "inference too slow: {per_call_ns} ns");
}
