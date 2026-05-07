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
//!
//! **Currently #[ignore]'d.** `CascadeRegistry::<Event>::with_engine_builtins()`
//! was deleted along with `engine/src/generated/`. Per the comment in
//! `crates/engine/src/cascade/dispatch.rs:858-860`, the replacement is
//! compiler-emitted into `engine_rules/src/cascade.rs` as Task 11 of
//! Plan B1' — that crate does not yet exist. When it lands, swap the
//! constructor in the test body and remove the `#[ignore]`.

#![cfg(not(debug_assertions))]

use engine::ability::{
    AbilityProgram, AbilityRegistry, AbilityRegistryBuilder, EffectOp, Gate, TargetSelector,
};
use engine::cascade::{CascadeRegistry, MAX_CASCADE_ITERATIONS};
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

#[allow(dead_code)]
fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3, hp: f32) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos, hp, ..Default::default() }).unwrap()
}

/// Build a self-recursive ability: `[Damage(1.0), CastAbility(self, Target)]`.
#[allow(dead_code)]
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

// Original test body, preserved verbatim for the port. Once
// `engine_rules::with_engine_builtins` (or its successor) exists, restore the
// `#[test]` attribute on `infinite_loop_caps_at_max_cascade_iterations` below
// and replace the `with_engine_builtins()` call site with the new entry point.
//
// ```ignore
// let (registry, ability) = build_infinite_loop();
// let cascade = CascadeRegistry::<Event>::with_engine_builtins();
// let mut state = SimState::new(4, 42);
// state.ability_registry = registry;
// let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO, 1_000_000.0);
// let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(3.0, 0.0, 0.0), 1_000_000.0);
// let mut events = EventRing::<Event>::with_cap(4096);
// events.push(Event::AgentCast { actor: caster, ability, target, depth: 0, tick: 0 });
// cascade.run_fixed_point(&mut state, &mut events);
// let n_damage = events.iter().filter(|e| matches!(e, Event::EffectDamageApplied { .. })).count();
// assert_eq!(n_damage, MAX_CASCADE_ITERATIONS);
// let n_exceeded = events.iter().filter(|e| matches!(e, Event::CastDepthExceeded { .. })).count();
// assert_eq!(n_exceeded, 1);
// let mut depths: Vec<u8> = events.iter().filter_map(|e| match e {
//     Event::AgentCast { depth, .. } => Some(*depth), _ => None,
// }).collect();
// depths.sort_unstable();
// let expected: Vec<u8> = (0..MAX_CASCADE_ITERATIONS as u8).collect();
// assert_eq!(depths, expected);
// let hit = events.iter().find_map(|e| match e {
//     Event::CastDepthExceeded { actor: c, ability: a, .. } => Some((*c, *a)),
//     _ => None,
// }).unwrap();
// assert_eq!(hit.0, caster);
// assert_eq!(hit.1, ability);
// ```

#[test]
#[ignore = "needs port to compiler-emitted cascade — Plan B1' Task 11"]
fn infinite_loop_caps_at_max_cascade_iterations() {
    // Stub body: the original called `CascadeRegistry::<Event>::with_engine_builtins()`,
    // which was deleted with `engine/src/generated/`. Restore the body from
    // the doc comment above when the replacement entry point lands.
    let _ = (
        build_infinite_loop,
        spawn as fn(&mut SimState, CreatureType, Vec3, f32) -> AgentId,
        MAX_CASCADE_ITERATIONS,
    );
    // Touch the imports the original body uses so the `#[ignore]` test still
    // surfaces drift in their signatures (e.g. if `EventRing`, `Event::AgentCast`
    // or `CascadeRegistry::<Event>::new` get renamed, this stops compiling).
    let mut state = SimState::new(4, 42);
    let _ = spawn(&mut state, CreatureType::Human, Vec3::ZERO, 1.0);
    let _events = EventRing::<Event>::with_cap(16);
    let _cascade = CascadeRegistry::<Event>::new();
    let _evt = Event::AgentCast {
        actor: AgentId::new(1).unwrap(),
        ability: engine::ability::AbilityId::new(1).unwrap(),
        target: AgentId::new(1).unwrap(),
        depth: 0,
        tick: 0,
    };
    unimplemented!("port pending — see file-level docs and Plan B1' Task 11");
}
