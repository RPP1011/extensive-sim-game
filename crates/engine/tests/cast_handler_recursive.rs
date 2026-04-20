//! Combat Foundation Task 18 — recursive `EffectOp::CastAbility` chains.
//!
//! Short chains (A→B→C→nothing) converge cleanly through the cascade
//! fixed-point loop and produce exactly one `EffectDamageApplied` per link.
//! The depth cap is NOT exercised here — see `cast_recursion_depth.rs` for
//! the MAX_CASCADE_ITERATIONS boundary test (which is release-only because
//! it drives the cascade to its iteration limit).
//!
//! Depth bookkeeping: each nested `EffectOp::CastAbility` increments the
//! `depth` on the emitted `AgentCast` by 1. Root casts (from action
//! dispatch or directly pushed) carry `depth = 0`. Here we push a depth=0
//! root and observe that the chain runs to completion without any
//! `CastDepthExceeded` emission.

use std::sync::Arc;

use engine::ability::{
    AbilityId, AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate, TargetSelector,
};
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos, hp: 1_000.0 }).unwrap()
}

#[test]
fn three_link_chain_fires_three_damage_events_no_depth_exceeded() {
    // Build C (leaf: one damage, no recursion), then B (damage + cast C),
    // then A (damage + cast B). Registering in order C → B → A lets B and
    // A reference earlier ids.
    let mut b = AbilityRegistryBuilder::new();
    let c_id = b.register(AbilityProgram::new_single_target(
        8.0,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [EffectOp::Damage { amount: 3.0 }],
    ));
    let b_id = b.register(AbilityProgram::new_single_target(
        8.0,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [
            EffectOp::Damage { amount: 2.0 },
            EffectOp::CastAbility { ability: c_id, selector: TargetSelector::Target },
        ],
    ));
    let a_id: AbilityId = b.register(AbilityProgram::new_single_target(
        8.0,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [
            EffectOp::Damage { amount: 1.0 },
            EffectOp::CastAbility { ability: b_id, selector: TargetSelector::Target },
        ],
    ));
    let registry = Arc::new(b.build());

    let mut cascade = CascadeRegistry::with_engine_builtins();
    cascade.register_cast_handler(registry);

    let mut state = SimState::new(8, 42);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf, Vec3::new(3.0, 0.0, 0.0));
    let mut events = EventRing::with_cap(1024);

    // Seed a root cast of A at depth 0. CastHandler will fan it out through
    // B and C via nested AgentCast events.
    events.push(Event::AgentCast {
        caster, ability: a_id, target, depth: 0, tick: 0,
    });
    cascade.run_fixed_point(&mut state, &mut events);

    // Exactly three damage emissions — one per link in A → B → C.
    let n_damage = events.iter()
        .filter(|e| matches!(e, Event::EffectDamageApplied { .. }))
        .count();
    assert_eq!(n_damage, 3, "chain should fire 3 damage events");

    // No depth-exceeded emissions — chain terminates at C (no CastAbility).
    let n_exceeded = events.iter()
        .filter(|e| matches!(e, Event::CastDepthExceeded { .. }))
        .count();
    assert_eq!(n_exceeded, 0, "short chain must not trip the depth cap");

    // Each link's depth on the emitted AgentCast events: root=0, B=1, C=2.
    let depths: Vec<u8> = events.iter().filter_map(|e| match e {
        Event::AgentCast { depth, .. } => Some(*depth),
        _ => None,
    }).collect();
    assert_eq!(depths, vec![0, 1, 2], "nested depth must increment per hop");

    // Target took 1 + 2 + 3 = 6 HP of damage over the chain.
    let expected_hp = 1_000.0 - (1.0 + 2.0 + 3.0);
    assert!(
        (state.agent_hp(target).unwrap() - expected_hp).abs() < 1e-4,
        "target hp after chain should be {expected_hp}, got {:?}",
        state.agent_hp(target),
    );
}

#[test]
fn self_targeted_recursive_link_uses_caster_selector() {
    // A casts B (with Caster selector → B targets the caster, not the
    // outer target). Damage of B therefore lands on the caster, not the
    // original target. Verifies the TargetSelector::Caster path.
    let mut b = AbilityRegistryBuilder::new();
    let b_id = b.register(AbilityProgram::new_single_target(
        8.0,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [EffectOp::Damage { amount: 5.0 }],
    ));
    let a_id = b.register(AbilityProgram::new_single_target(
        8.0,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [
            EffectOp::Damage { amount: 2.0 },
            EffectOp::CastAbility { ability: b_id, selector: TargetSelector::Caster },
        ],
    ));
    let registry = Arc::new(b.build());

    let mut cascade = CascadeRegistry::with_engine_builtins();
    cascade.register_cast_handler(registry);

    let mut state = SimState::new(4, 42);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf, Vec3::new(3.0, 0.0, 0.0));
    let mut events = EventRing::with_cap(256);

    events.push(Event::AgentCast {
        caster, ability: a_id, target, depth: 0, tick: 0,
    });
    cascade.run_fixed_point(&mut state, &mut events);

    // Caster took the 5 hp self-damage from the nested B; target took only the 2.
    assert!((state.agent_hp(caster).unwrap() - 995.0).abs() < 1e-4);
    assert!((state.agent_hp(target).unwrap() - 998.0).abs() < 1e-4);
}
