use std::collections::{HashMap, VecDeque};

use super::*;
use super::culture::{MoraleCultureProfile, MoraleCultureRegistry, MoraleInputWeights};
use crate::sim::{SimState, SimVec2, Team, UnitState, UnitStore, sim_vec2};

fn make_unit(id: u32, team: Team, hp: i32, max_hp: i32, pos: SimVec2) -> UnitState {
    UnitState {
        id, team, hp, max_hp,
        position: pos,
        move_speed_per_sec: 3.0, attack_damage: 10,
        attack_range: 5.0, attack_cooldown_ms: 700, attack_cast_time_ms: 300,
        cooldown_remaining_ms: 0, ability_damage: 0, ability_range: 0.0,
        ability_cooldown_ms: 0, ability_cast_time_ms: 0, ability_cooldown_remaining_ms: 0,
        heal_amount: 0, heal_range: 0.0, heal_cooldown_ms: 0, heal_cast_time_ms: 0,
        heal_cooldown_remaining_ms: 0, control_range: 0.0, control_duration_ms: 0,
        control_cooldown_ms: 0, control_cast_time_ms: 0, control_cooldown_remaining_ms: 0,
        control_remaining_ms: 0, casting: None, abilities: Vec::new(), passives: Vec::new(),
        status_effects: Vec::new(), shield_hp: 0, resistance_tags: HashMap::new(),
        state_history: VecDeque::new(), channeling: None, resource: 0, max_resource: 0,
        resource_regen_per_sec: 0.0, owner_id: None, directed: false,
        armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0,
        total_healing_done: 0, total_damage_done: 0,
    }
}

fn make_state(units: Vec<UnitState>) -> SimState {
    SimState {
        tick: 0, rng_state: 42, units: UnitStore::new(units),
        projectiles: Vec::new(), passive_trigger_depth: 0,
        zones: Vec::new(), tethers: Vec::new(), grid_nav: None,
    }
}

fn test_registry() -> MoraleCultureRegistry {
    let mut reg = MoraleCultureRegistry::new();
    reg.load_from_str(culture::default_cultures_toml()).unwrap();
    reg
}

fn collectivist_profile() -> MoraleCultureProfile {
    MoraleCultureProfile {
        name: "Collectivist".to_string(),
        input_weights: MoraleInputWeights {
            self_weight: 0.15,
            allies_weight: 0.35,
            threats_weight: 0.15,
            leadership_weight: 0.25,
            situation_weight: 0.10,
        },
        cascade_susceptibility: 0.8,
        volatility: 1.0,
        routing_threshold: 0.20,
        wavering_threshold: 0.35,
        fired_up_threshold: 0.80,
    }
}

fn individualist_profile() -> MoraleCultureProfile {
    MoraleCultureProfile {
        name: "Individualist".to_string(),
        input_weights: MoraleInputWeights {
            self_weight: 0.40,
            allies_weight: 0.05,
            threats_weight: 0.30,
            leadership_weight: 0.05,
            situation_weight: 0.20,
        },
        cascade_susceptibility: 0.1,
        volatility: 0.8,
        routing_threshold: 0.15,
        wavering_threshold: 0.30,
        fired_up_threshold: 0.85,
    }
}

#[test]
fn test_morale_drops_on_low_hp() {
    let units = vec![
        make_unit(1, Team::Hero, 10, 100, sim_vec2(0.0, 0.0)),
        make_unit(2, Team::Enemy, 100, 100, sim_vec2(10.0, 0.0)),
    ];
    let state = make_state(units);
    let unit = &state.units[0];
    let inputs = compute_morale_inputs(&state, unit);

    // Self input should be very negative (10% HP → -0.8)
    assert!(inputs.self_input < -0.5);
}

#[test]
fn test_morale_positive_when_winning() {
    let units = vec![
        make_unit(1, Team::Hero, 100, 100, sim_vec2(0.0, 0.0)),
        make_unit(2, Team::Hero, 100, 100, sim_vec2(2.0, 0.0)),
        make_unit(3, Team::Enemy, 20, 100, sim_vec2(10.0, 0.0)),
    ];
    let state = make_state(units);
    let unit = &state.units[0];
    let inputs = compute_morale_inputs(&state, unit);

    // Situation should be positive (our team is healthier)
    assert!(inputs.situation_input > 0.0);
    // Threats should be positive (we outnumber them)
    assert!(inputs.threats_input > 0.0);
}

#[test]
fn test_collectivist_cascade_effect() {
    let units = vec![
        make_unit(1, Team::Hero, 80, 100, sim_vec2(0.0, 0.0)),
        make_unit(2, Team::Hero, 50, 100, sim_vec2(5.0, 0.0)),
        make_unit(3, Team::Enemy, 100, 100, sim_vec2(20.0, 0.0)),
    ];
    let state = make_state(units);

    let mut morale_state = MoraleState::default();
    morale_state.morale_by_unit.insert(1, UnitMorale {
        value: 0.5,
        level: MoraleLevel::Steady,
        culture_name: "Collectivist".to_string(),
    });
    morale_state.morale_by_unit.insert(2, UnitMorale {
        value: 0.1,
        level: MoraleLevel::Routing,
        culture_name: "Collectivist".to_string(),
    });

    let unit = &state.units[0];
    let penalty = compute_cascade_penalty(&state, unit, &morale_state);

    // Should have a negative penalty from routing neighbor
    assert!(penalty < 0.0);
}

#[test]
fn test_individualist_resists_cascade() {
    let units = vec![
        make_unit(1, Team::Hero, 80, 100, sim_vec2(0.0, 0.0)),
        make_unit(2, Team::Hero, 50, 100, sim_vec2(5.0, 0.0)),
        make_unit(3, Team::Enemy, 100, 100, sim_vec2(20.0, 0.0)),
    ];
    let state = make_state(units);

    let mut morale_state = MoraleState::default();
    morale_state.morale_by_unit.insert(1, UnitMorale {
        value: 0.5,
        level: MoraleLevel::Steady,
        culture_name: "Individualist".to_string(),
    });
    morale_state.morale_by_unit.insert(2, UnitMorale {
        value: 0.1,
        level: MoraleLevel::Routing,
        culture_name: "Individualist".to_string(),
    });

    // Run several morale updates — individualist should resist cascade
    let mut reg = MoraleCultureRegistry::new();
    reg.insert(individualist_profile());

    let _initial_value = morale_state.morale_by_unit[&1].value;
    for _ in 0..5 {
        update_morale(&state, &mut morale_state, &reg);
    }
    let after_value = morale_state.morale_by_unit[&1].value;

    // Also test collectivist with same setup
    let mut morale_state_c = MoraleState::default();
    morale_state_c.morale_by_unit.insert(1, UnitMorale {
        value: 0.5,
        level: MoraleLevel::Steady,
        culture_name: "Collectivist".to_string(),
    });
    morale_state_c.morale_by_unit.insert(2, UnitMorale {
        value: 0.1,
        level: MoraleLevel::Routing,
        culture_name: "Collectivist".to_string(),
    });

    let mut reg_c = MoraleCultureRegistry::new();
    reg_c.insert(collectivist_profile());

    for _ in 0..5 {
        update_morale(&state, &mut morale_state_c, &reg_c);
    }
    let collectivist_value = morale_state_c.morale_by_unit[&1].value;

    // Individualist should have higher morale than collectivist after cascade
    assert!(
        after_value > collectivist_value,
        "Individualist ({}) should resist cascade better than collectivist ({})",
        after_value,
        collectivist_value
    );
}

#[test]
fn test_routing_threshold() {
    let units = vec![
        make_unit(1, Team::Hero, 10, 100, sim_vec2(0.0, 0.0)),
        make_unit(2, Team::Enemy, 100, 100, sim_vec2(5.0, 0.0)),
        make_unit(3, Team::Enemy, 100, 100, sim_vec2(5.0, 5.0)),
        make_unit(4, Team::Enemy, 100, 100, sim_vec2(5.0, -5.0)),
    ];
    let state = make_state(units);

    let mut morale_state = MoraleState::default();
    morale_state.morale_by_unit.insert(1, UnitMorale {
        value: 0.25,
        level: MoraleLevel::Wavering,
        culture_name: "Mercenary".to_string(),
    });

    let mut reg = MoraleCultureRegistry::new();
    reg.load_from_str(culture::default_cultures_toml()).unwrap();

    // Badly outnumbered and low HP — morale should drop to routing
    for _ in 0..20 {
        update_morale(&state, &mut morale_state, &reg);
    }
    assert_eq!(
        morale_state.morale_by_unit[&1].level,
        MoraleLevel::Routing,
        "Unit should be routing when heavily outnumbered with low HP"
    );
}

#[test]
fn test_default_cultures_load() {
    let reg = test_registry();
    assert_eq!(reg.len(), 5);
    assert!(reg.get("Collectivist").is_some());
    assert!(reg.get("Individualist").is_some());
    assert!(reg.get("Fanatical").is_some());
    assert!(reg.get("Mercenary").is_some());
    assert!(reg.get("Disciplined").is_some());
}

#[test]
fn test_morale_state_init() {
    let units = vec![
        make_unit(1, Team::Hero, 100, 100, sim_vec2(0.0, 0.0)),
        make_unit(2, Team::Hero, 100, 100, sim_vec2(5.0, 0.0)),
        make_unit(3, Team::Enemy, 100, 100, sim_vec2(20.0, 0.0)),
    ];
    let state = make_state(units);

    let mut assignments = HashMap::new();
    assignments.insert(1, "Collectivist".to_string());
    assignments.insert(2, "Individualist".to_string());

    let morale = MoraleState::init(&state, &assignments, "Disciplined");
    assert_eq!(morale.morale_by_unit[&1].culture_name, "Collectivist");
    assert_eq!(morale.morale_by_unit[&2].culture_name, "Individualist");
    assert_eq!(morale.morale_by_unit[&3].culture_name, "Disciplined");
}
