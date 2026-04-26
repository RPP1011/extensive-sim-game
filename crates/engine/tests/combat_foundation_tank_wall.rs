//! Combat Foundation — tank-wall regression fixture.
//!
//! Named scenario: shielded "tank" units in front absorb damage before the
//! wall is broken, protecting HP behind it. Mechanics exercised:
//!   - Shield absorbs incoming damage before HP is touched.
//!   - Once shield is depleted, residual damage spills over to HP.
//!   - Multiple shields stack additively (two tanks side by side each with
//!     their own shield).
//!   - The wall "holds" for the expected number of strikes before breaking.
//!   - After the wall breaks, HP falls under continued sustained pressure.
//!
//! Uses `dispatch_effect_shield_applied` and `dispatch_effect_damage_applied`
//! directly (same pattern as `cast_handler_shield_absorb.rs`).

use engine::event::EventRing;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine_data::entities::CreatureType;
use engine_data::events::Event;
use engine_rules::physics::{dispatch_effect_damage_applied, dispatch_effect_shield_applied};
use engine_rules::views::ViewRegistry;
use glam::Vec3;

fn spawn_tank(state: &mut SimState, pos: Vec3, hp: f32) -> AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos,
            hp,
            max_hp: hp,
            ..Default::default()
        })
        .expect("spawn tank")
}

fn spawn_attacker(state: &mut SimState) -> AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 100.0,
            max_hp: 100.0,
            ..Default::default()
        })
        .expect("spawn attacker")
}

/// Tank wall holds for expected strikes: a 30-HP shield absorbs 3 strikes
/// of 10 damage each before breaking; HP is untouched during that window.
#[test]
fn shield_wall_holds_for_n_strikes_then_breaks() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(128);
    let mut views = ViewRegistry::new();

    let attacker = spawn_attacker(&mut state);
    let tank = spawn_tank(&mut state, Vec3::new(2.0, 0.0, 0.0), 100.0);

    // Apply a 30-HP shield to the tank.
    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied {
            actor: tank,
            target: tank,
            amount: 30.0,
            tick: 0,
        },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_shield_hp(tank), Some(30.0));

    // Strike 1: 10 damage → shield 20, hp 100.
    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied {
            actor: attacker,
            target: tank,
            amount: 10.0,
            tick: 1,
        },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_shield_hp(tank), Some(20.0), "after strike 1: shield 20");
    assert_eq!(state.agent_hp(tank), Some(100.0), "HP untouched in strike 1");

    // Strike 2: 10 damage → shield 10, hp 100.
    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied {
            actor: attacker,
            target: tank,
            amount: 10.0,
            tick: 2,
        },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_shield_hp(tank), Some(10.0), "after strike 2: shield 10");
    assert_eq!(state.agent_hp(tank), Some(100.0), "HP untouched in strike 2");

    // Strike 3: 10 damage — shield depleted, HP still protected (exact drain).
    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied {
            actor: attacker,
            target: tank,
            amount: 10.0,
            tick: 3,
        },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_shield_hp(tank), Some(0.0), "after strike 3: shield broken");
    assert_eq!(state.agent_hp(tank), Some(100.0), "HP still untouched when shield exactly drained");

    // Strike 4: wall is broken — damage now routes to HP.
    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied {
            actor: attacker,
            target: tank,
            amount: 10.0,
            tick: 4,
        },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_shield_hp(tank), Some(0.0), "shield remains 0 after wall break");
    assert_eq!(state.agent_hp(tank), Some(90.0), "HP drops once wall is broken");
}

/// Overflow: one hit larger than the shield bleeds remaining damage through to HP.
#[test]
fn single_hit_overflow_through_shield_to_hp() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();

    let attacker = spawn_attacker(&mut state);
    let tank = spawn_tank(&mut state, Vec3::new(2.0, 0.0, 0.0), 100.0);

    // 15-HP shield, 25-damage hit → shield wiped, 10 overflow to HP.
    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied {
            actor: tank,
            target: tank,
            amount: 15.0,
            tick: 0,
        },
        &mut state,
        &mut views,
        &mut events,
    );

    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied {
            actor: attacker,
            target: tank,
            amount: 25.0,
            tick: 1,
        },
        &mut state,
        &mut views,
        &mut events,
    );

    assert_eq!(state.agent_shield_hp(tank), Some(0.0), "shield wiped by overflow hit");
    assert_eq!(state.agent_hp(tank), Some(90.0), "10 overflow damage hit HP");
}

/// Two tanks side by side each have their own independent shield.
/// Attacking one tank only depletes that tank's shield.
#[test]
fn two_tank_wall_shields_are_independent() {
    let mut state = SimState::new(8, 42);
    let mut events = EventRing::<Event>::with_cap(128);
    let mut views = ViewRegistry::new();

    let attacker = spawn_attacker(&mut state);
    let tank1 = spawn_tank(&mut state, Vec3::new(2.0, 0.0, 0.0), 100.0);
    let tank2 = spawn_tank(&mut state, Vec3::new(4.0, 0.0, 0.0), 100.0);

    // Each tank gets a 20-HP shield.
    for &t in &[tank1, tank2] {
        dispatch_effect_shield_applied(
            &Event::EffectShieldApplied {
                actor: t,
                target: t,
                amount: 20.0,
                tick: 0,
            },
            &mut state,
            &mut views,
            &mut events,
        );
    }

    // Attacker focuses tank1 with 30 damage (20 to shield + 10 overflow).
    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied {
            actor: attacker,
            target: tank1,
            amount: 30.0,
            tick: 1,
        },
        &mut state,
        &mut views,
        &mut events,
    );

    // tank1 shield broken, hp = 90.
    assert_eq!(state.agent_shield_hp(tank1), Some(0.0), "tank1 shield depleted");
    assert_eq!(state.agent_hp(tank1), Some(90.0), "tank1 hp after overflow");

    // tank2 is completely untouched.
    assert_eq!(state.agent_shield_hp(tank2), Some(20.0), "tank2 shield intact");
    assert_eq!(state.agent_hp(tank2), Some(100.0), "tank2 hp untouched");
}

/// Sustained pressure: tank with a shield survives K more ticks than an
/// unshielded twin — the shield provides a quantifiable durability advantage.
#[test]
fn shielded_tank_outlasts_unshielded_under_sustained_fire() {
    const DAMAGE_PER_HIT: f32 = 20.0;
    const SHIELD_AMOUNT: f32 = 40.0; // absorbs 2 extra hits worth

    let mut state = SimState::new(8, 42);
    let mut events = EventRing::<Event>::with_cap(256);
    let mut views = ViewRegistry::new();

    let attacker = spawn_attacker(&mut state);
    let shielded = spawn_tank(&mut state, Vec3::new(2.0, 0.0, 0.0), 100.0);
    let unshielded = spawn_tank(&mut state, Vec3::new(4.0, 0.0, 0.0), 100.0);

    // Give shielded tank its wall.
    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied {
            actor: shielded,
            target: shielded,
            amount: SHIELD_AMOUNT,
            tick: 0,
        },
        &mut state,
        &mut views,
        &mut events,
    );

    // Apply 7 hits to both tanks simultaneously.
    for tick in 1..=7u32 {
        state.tick = tick;
        for &t in &[shielded, unshielded] {
            if state.agent_alive(t) {
                dispatch_effect_damage_applied(
                    &Event::EffectDamageApplied {
                        actor: attacker,
                        target: t,
                        amount: DAMAGE_PER_HIT,
                        tick,
                    },
                    &mut state,
                    &mut views,
                    &mut events,
                );
            }
        }
    }

    // After 7 hits (140 total damage):
    //   Unshielded: 100 hp / 20 per hit → dies after 5 hits.
    //   Shielded:   40 shield (2 hits absorbed) + 100 hp → effective 140 hp,
    //               dies after 7 hits exactly.
    assert!(!state.agent_alive(unshielded), "unshielded tank must be dead after 7 hits");
    // Shielded tank reaches 0 hp at hit 7 exactly.
    assert_eq!(state.agent_hp(shielded), Some(0.0), "shielded tank hp at 0 after 7 hits");

    // The shielded tank lived to the last hit — shield gave it durability.
    let shielded_death_tick = events
        .iter()
        .filter_map(|e| match e {
            Event::AgentDied { agent_id, tick } if *agent_id == shielded => Some(*tick),
            _ => None,
        })
        .next();
    let unshielded_death_tick = events
        .iter()
        .filter_map(|e| match e {
            Event::AgentDied { agent_id, tick } if *agent_id == unshielded => Some(*tick),
            _ => None,
        })
        .next();

    assert!(
        shielded_death_tick > unshielded_death_tick,
        "shielded tank (died tick {:?}) should outlast unshielded (died tick {:?})",
        shielded_death_tick,
        unshielded_death_tick
    );
}
