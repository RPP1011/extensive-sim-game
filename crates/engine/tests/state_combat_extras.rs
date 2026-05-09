//! Combat-extras fields (state.md §Combat/Vitality): `shield_hp`, `armor`,
//! `magic_resist`, `attack_damage`, `attack_range`, `mana`, `max_mana`,
//! `ability_power`.
//!
//! Task B of engine-plan-state-port. Storage only; damage calc still uses
//! the step.rs constants.
//!
//! Task #233 (2026-05-08) added `ability_power` to the column set so the
//! `+ N% ability_power` ability scaling reads a real per-agent value
//! through `SimState::caster_stats`.

use engine_data::entities::CreatureType;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

#[test]
fn spawn_defaults_combat_extras() {
    let mut state = SimState::new(4, 42);
    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    // Absorb layer starts at zero — no free damage shield on spawn.
    assert_eq!(state.agent_shield_hp(a), Some(0.0));
    // Defensive stats start at zero (no reduction by default).
    assert_eq!(state.agent_armor(a), Some(0.0));
    assert_eq!(state.agent_magic_resist(a), Some(0.0));
    // Offensive defaults mirror the engine constants used by `step.rs`:
    // damage = 10.0, attack range = 2.0 (see step.rs::ATTACK_RANGE).
    assert_eq!(state.agent_attack_damage(a), Some(10.0));
    assert_eq!(state.agent_attack_range(a), Some(2.0));
    // Mana defaults to zero — abilities with mana cost are opt-in.
    assert_eq!(state.agent_mana(a), Some(0.0));
    assert_eq!(state.agent_max_mana(a), Some(0.0));
    // AbilityPower defaults to zero — matches the pre-this-slice
    // placeholder behavior so fixtures that don't set AP are bit-stable.
    assert_eq!(state.agent_ability_power(a), Some(0.0));
}

#[test]
fn set_and_read_combat_extras() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();

    state.set_agent_shield_hp(a, 50.0);
    state.set_agent_armor(a, 0.25);
    state.set_agent_magic_resist(a, 0.30);
    state.set_agent_attack_damage(a, 18.5);
    state.set_agent_attack_range(a, 5.0);
    state.set_agent_mana(a, 40.0);
    state.set_agent_max_mana(a, 100.0);
    state.set_agent_ability_power(a, 75.0);

    assert_eq!(state.agent_shield_hp(a), Some(50.0));
    assert_eq!(state.agent_armor(a), Some(0.25));
    assert_eq!(state.agent_magic_resist(a), Some(0.30));
    assert_eq!(state.agent_attack_damage(a), Some(18.5));
    assert_eq!(state.agent_attack_range(a), Some(5.0));
    assert_eq!(state.agent_mana(a), Some(40.0));
    assert_eq!(state.agent_max_mana(a), Some(100.0));
    assert_eq!(state.agent_ability_power(a), Some(75.0));
}

#[test]
fn bulk_slices_have_cap_length() {
    let state = SimState::new(8, 42);
    assert_eq!(state.hot_shield_hp().len(), 8);
    assert_eq!(state.hot_armor().len(), 8);
    assert_eq!(state.hot_magic_resist().len(), 8);
    assert_eq!(state.hot_attack_damage().len(), 8);
    assert_eq!(state.hot_attack_range().len(), 8);
    assert_eq!(state.hot_mana().len(), 8);
    assert_eq!(state.hot_max_mana().len(), 8);
    assert_eq!(state.hot_ability_power().len(), 8);
}

#[test]
fn caster_stats_projects_ability_power_from_soa() {
    // Pin the SoA → CasterStats projection: setting the AP slot via
    // `set_agent_ability_power` must surface through `caster_stats`'s
    // `ability_power` field. Pre-this-slice the field was hard-wired to
    // 0.0 (no SoA backing) — Task #233 wires it to the new
    // `hot_ability_power` column.
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    // Spawn default is 0.0 — round-trip the placeholder behavior so a
    // regression that breaks the projection (e.g. wires the wrong
    // column) surfaces immediately.
    let stats0 = state.caster_stats(a).expect("caster_stats for alive agent");
    assert_eq!(stats0.ability_power, 0.0);
    // Real value via setter — proves the SoA write reaches the
    // projection.
    state.set_agent_ability_power(a, 50.0);
    let stats1 = state.caster_stats(a).expect("caster_stats for alive agent");
    assert_eq!(stats1.ability_power, 50.0);
    // Non-AP fields keep their existing SoA-backed values (sanity).
    assert_eq!(stats1.attack_damage, state.agent_attack_damage(a).unwrap());
    assert_eq!(stats1.max_hp,        state.agent_max_hp(a).unwrap());
}
