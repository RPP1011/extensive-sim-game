//! Combat Foundation Task 15 — cooldown mask gate regression.
//!
//! `evaluate_cast_gate` (Task 9) treats `hot_cooldown_next_ready_tick` as an
//! absolute tick. The `CastHandler` sets it to `current_tick + cooldown_ticks`
//! after a cast resolves. This test pins the contract end-to-end:
//!
//! 1. Tick 0: first cast fires → `next_ready_tick = 10`.
//! 2. Ticks 1..=9: gate returns false (cooldown pending), `UtilityBackend`
//!    -style flow never re-emits the cast.
//! 3. Tick 10: gate flips true; a second cast fires and re-sets the cooldown.
//!
//! The test also drives a `step(...)` loop with a `PolicyBackend` that only
//! emits a `Cast` when the gate allows it — so the "UtilityBackend skips"
//! invariant is mirrored concretely: no cast action → no AgentCast event,
//! regardless of mask permissiveness. This is the shape the real mask
//! predicate will take once `mark_cast_valid` lands in Plan 2.

use engine::ability::{
    evaluate_cast_gate, AbilityId, AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos, hp: 100.0, ..Default::default() }).unwrap()
}

/// Backend that consults `evaluate_cast_gate` every tick and only emits a
/// `Cast` action when the gate allows it. Mirrors what the real
/// `UtilityBackend` will do once Task-9's mask predicate is integrated: the
/// policy is responsible for respecting the gate, so a false gate → no cast.
///
/// Reads the ability registry directly off `state.ability_registry` —
/// the cast-handler migration (2026-04-19) retired the per-backend
/// `Arc<AbilityRegistry>` in favour of the state-borne registry.
struct GatedCastBackend {
    caster:  AgentId,
    target:  AgentId,
    ability: AbilityId,
}

impl PolicyBackend for GatedCastBackend {
    fn evaluate(&self, state: &SimState, _m: &MaskBuffer, _target_mask: &engine::mask::TargetMask, out: &mut Vec<Action>) {
        if evaluate_cast_gate(state, &state.ability_registry, self.caster, self.ability, self.target) {
            out.push(Action {
                agent: self.caster,
                kind:  ActionKind::Micro {
                    kind:   MicroKind::Cast,
                    target: MicroTarget::Ability { id: self.ability, target: self.target },
                },
            });
        }
    }
}

#[test]
fn cooldown_blocks_recast_until_next_ready_tick() {
    // Register an ability with a 10-tick cooldown. Cast at tick 0 → next_ready=10.
    // Ticks 1..=9: gate false (cooldown pending). Tick 10: gate true, re-fires.
    let mut b = AbilityRegistryBuilder::new();
    let ability = b.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 1.0 }],
    ));
    let registry = b.build();

    let mut state = SimState::new(8, 42);
    state.ability_registry = registry;
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(2.0, 0.0, 0.0));

    // Gate passes pre-cast.
    assert!(evaluate_cast_gate(&state, &state.ability_registry, caster, ability, target));

    // Tick 0 cast.
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(512);
    // `with_engine_builtins()` registers the (now stateless) CastHandler
    // alongside every other effect handler — no explicit
    // `register_cast_handler` call needed.
    let cascade = CascadeRegistry::with_engine_builtins();

    let backend = GatedCastBackend { caster, target, ability };

    // Tick 0: expect one AgentCast, cooldown_next_ready becomes 10.
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    assert_eq!(state.agent_cooldown_next_ready(caster), Some(10));
    let n_casts_after_t0 = events.iter().filter(|e| matches!(e, Event::AgentCast { .. })).count();
    assert_eq!(n_casts_after_t0, 1);

    // Ticks 1..=9: gate MUST be false, GatedCastBackend emits nothing, no
    // new AgentCast events accumulate. The cooldown's absolute-tick design
    // means each of these iterations sees `state.tick < 10` → false.
    for expected_tick in 1..=9u32 {
        assert_eq!(state.tick, expected_tick);
        assert!(
            !evaluate_cast_gate(&state, &state.ability_registry, caster, ability, target),
            "gate must reject while state.tick={expected_tick} < 10"
        );
        step(&mut state, &mut scratch, &mut events, &backend, &cascade);
        let n_now = events.iter().filter(|e| matches!(e, Event::AgentCast { .. })).count();
        assert_eq!(
            n_now, 1,
            "no new AgentCast expected during cooldown window (tick {expected_tick})"
        );
    }

    // Tick 10: gate opens.
    assert_eq!(state.tick, 10);
    assert!(evaluate_cast_gate(&state, &state.ability_registry, caster, ability, target));
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    let n_casts_after_t10 = events.iter().filter(|e| matches!(e, Event::AgentCast { .. })).count();
    assert_eq!(n_casts_after_t10, 2, "second cast must fire at tick 10");
    // Cooldown resets: tick 10 + 10 = 20.
    assert_eq!(state.agent_cooldown_next_ready(caster), Some(20));
}

#[test]
fn zero_cooldown_ability_can_recast_every_tick() {
    // gate.cooldown_ticks = 0 → next_ready == current_tick after every cast,
    // which the gate treats as "ready now". Over 5 ticks the GatedCastBackend
    // should fire 5 casts.
    let mut b = AbilityRegistryBuilder::new();
    let ability = b.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 1.0 }],
    ));
    let registry = b.build();

    let mut state = SimState::new(8, 42);
    state.ability_registry = registry;
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(2.0, 0.0, 0.0));
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(512);
    let cascade = CascadeRegistry::with_engine_builtins();

    let backend = GatedCastBackend { caster, target, ability };

    for _ in 0..5 {
        step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    }
    let n_casts = events.iter().filter(|e| matches!(e, Event::AgentCast { .. })).count();
    assert_eq!(n_casts, 5, "zero-cooldown ability must fire every tick");
}
