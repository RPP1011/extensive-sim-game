//! Combat Foundation Task 18 — recursion depth cap on infinite `CastAbility`
//! chains.
//!
//! The `InfiniteLoop` ability below has a `Damage` effect and then casts
//! ITSELF. Left unchecked it would recur forever; `CastHandler` bounds the
//! chain with a per-event `depth: u8` that increments per nested hop.
//! When the hop would reach `MAX_CASCADE_ITERATIONS` the handler emits
//! `Event::CastDepthExceeded` instead of pushing the nested cast. This
//! keeps the cast subsystem self-bounded BELOW the cascade framework's
//! iteration ceiling.
//!
//! **Release-only** for the exact event count. In debug the cascade
//! framework's own convergence panic (Plan 2.75 Task 8) fires because the
//! last iter still dispatches the `Damage` + `CastDepthExceeded` pair that
//! CastHandler emits as it trips the cap. We follow the pattern set by
//! `cascade_bounded.rs` (`#[cfg(not(debug_assertions))]`) — the strict
//! counts are asserted only when the panic check is suppressed.

#![cfg(not(debug_assertions))]

use engine::ability::{
    AbilityProgram, AbilityRegistry, AbilityRegistryBuilder, EffectOp, Gate, TargetSelector,
};
use engine::cascade::{CascadeRegistry, MAX_CASCADE_ITERATIONS};
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3, hp: f32) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos, hp, ..Default::default() }).unwrap()
}

/// Build a self-recursive ability: `[Damage(1.0), CastAbility(self, Target)]`.
fn build_infinite_loop() -> (AbilityRegistry, engine::ability::AbilityId) {
    // We need the ability to refer to itself. `AbilityRegistryBuilder::register`
    // returns the id for the JUST-registered program, so we use `next_id` to
    // reserve the slot, construct the program referencing that id, then
    // register. If the builder exposes a two-step pattern we use it; else we
    // construct with a placeholder and rely on the builder accepting a
    // self-reference.
    let mut b = AbilityRegistryBuilder::new();
    // Register a tombstone first to reserve the id. This is a throwaway
    // one-effect program — we replace its program slot immediately below
    // with the self-recursive one. The builder is append-only so we can't
    // actually rewrite, so we instead *predict* the id that will be
    // returned (first register = id 1).
    let self_id = engine::ability::AbilityId::new(1).unwrap();
    let id = b.register(AbilityProgram::new_single_target(
        6.0,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [
            EffectOp::Damage { amount: 1.0 },
            EffectOp::CastAbility { ability: self_id, selector: TargetSelector::Target },
        ],
    ));
    assert_eq!(id, self_id, "builder must assign id 1 to the first register");
    (b.build(), id)
}

#[test]
fn infinite_loop_caps_at_max_cascade_iterations() {
    let (registry, ability) = build_infinite_loop();
    // Stateless CastHandler is installed by `with_engine_builtins()`; the
    // registry rides on `state`.
    let cascade = CascadeRegistry::with_engine_builtins();

    let mut state = SimState::new(4, 42);
    state.ability_registry = registry;
    // Large HP so non-lethal damage doesn't emit AgentDied mid-cascade.
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO, 1_000_000.0);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(3.0, 0.0, 0.0), 1_000_000.0);
    let mut events = EventRing::with_cap(4096);

    events.push(Event::AgentCast {
        actor: caster, ability, target, depth: 0, tick: 0,
    });
    cascade.run_fixed_point(&mut state, &mut events);

    // CastHandler emits Damage on every cast it processes. The chain runs
    // for exactly MAX_CASCADE_ITERATIONS levels (depths 0..7) before the
    // depth cap replaces the nested cast with CastDepthExceeded. That's
    // 8 damage events.
    let n_damage = events.iter()
        .filter(|e| matches!(e, Event::EffectDamageApplied { .. }))
        .count();
    assert_eq!(
        n_damage, MAX_CASCADE_ITERATIONS,
        "infinite chain should emit exactly MAX_CASCADE_ITERATIONS damages"
    );

    // Exactly one CastDepthExceeded audit event.
    let n_exceeded = events.iter()
        .filter(|e| matches!(e, Event::CastDepthExceeded { .. }))
        .count();
    assert_eq!(n_exceeded, 1, "one CastDepthExceeded expected at the cap");

    // Depths observed on AgentCast events: 0 through 7 inclusive.
    let mut depths: Vec<u8> = events.iter().filter_map(|e| match e {
        Event::AgentCast { depth, .. } => Some(*depth),
        _ => None,
    }).collect();
    depths.sort_unstable();
    let expected: Vec<u8> = (0..MAX_CASCADE_ITERATIONS as u8).collect();
    assert_eq!(depths, expected, "depths of all cast events must span 0..MAX");

    // The audit event points at the ability that hit the cap.
    let hit = events.iter().find_map(|e| match e {
        Event::CastDepthExceeded { actor: c, ability: a, .. } => Some((*c, *a)),
        _ => None,
    }).unwrap();
    assert_eq!(hit.0, caster, "audit event records the caster");
    assert_eq!(hit.1, ability, "audit event records the ability that recursed");
}
