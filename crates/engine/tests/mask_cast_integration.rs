//! Audit fix CRITICAL #2 — `mark_domain_hook_micros_allowed` must now consult
//! `evaluate_cast_gate` via `state.ability_registry`.
//!
//! Complements `mask_can_cast.rs` (which tests the gate predicate in
//! isolation). Each test here drives `step_full`, installs the compiled
//! `AbilityRegistry` onto `state`, and asserts the cast mask bit flips
//! appropriately for alive agents under different gate conditions. The
//! cast-handler migration (2026-04-19) retired the per-cascade
//! `Arc<AbilityRegistry>` plumbing; the registry now rides on `SimState`
//! and the stateless handler is registered by `with_engine_builtins`.

use engine::ability::{
    AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::invariant::InvariantRegistry;
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::{Action, PolicyBackend};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step_full, SimScratch};
use engine::telemetry::NullSink;
use glam::Vec3;

fn cast_bit(mask: &MaskBuffer, slot: usize) -> bool {
    let off = slot * MicroKind::ALL.len() + MicroKind::Cast as usize;
    mask.micro_kind[off]
}

/// A backend that does nothing — it never populates actions. Lets us drive
/// `step_full` once and then inspect `scratch.mask` for the cast-gate-driven
/// bit state.
struct InertBackend;
impl PolicyBackend for InertBackend {
    fn evaluate(
        &self,
        _: &SimState,
        _: &MaskBuffer,
        _: &engine::mask::TargetMask,
        _: &mut Vec<Action>,
    ) {}
}

fn setup(registry_build: impl FnOnce(&mut AbilityRegistryBuilder)) -> (
    SimState, SimScratch, EventRing, CascadeRegistry, InvariantRegistry,
) {
    let mut state = SimState::new(8, 42);
    let scratch = SimScratch::new(state.agent_cap() as usize);
    let events = EventRing::with_cap(1024);
    let mut b = AbilityRegistryBuilder::new();
    registry_build(&mut b);
    state.ability_registry = b.build();
    // `with_engine_builtins()` registers the stateless CastHandler alongside
    // the DSL-emitted effect handlers. The handler reads programs off
    // `state.ability_registry`.
    let cascade = CascadeRegistry::with_engine_builtins();
    let invariants = InvariantRegistry::new();
    (state, scratch, events, cascade, invariants)
}

fn run_one_tick(
    state: &mut SimState,
    scratch: &mut SimScratch,
    events: &mut EventRing,
    cascade: &CascadeRegistry,
    invariants: &InvariantRegistry,
) {
    step_full(
        state, scratch, events, &InertBackend, cascade,
        &mut [], invariants, &NullSink,
    );
}

#[test]
fn cast_bit_false_when_no_hostile_in_range() {
    // Single agent, no hostile — the inferred-target heuristic picks
    // None → gate fails → Cast bit must be false.
    let (mut state, mut scratch, mut events, cascade, invariants) = setup(|b| {
        let _ = b.register(AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 10.0 }],
        ));
    });
    let human = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    run_one_tick(&mut state, &mut scratch, &mut events, &cascade, &invariants);
    let slot = (human.raw() - 1) as usize;
    assert!(!cast_bit(&scratch.mask, slot),
        "no hostile in range → Cast bit must flip to false");
}

#[test]
fn cast_bit_true_when_hostile_in_range_and_all_gates_pass() {
    // Human caster with Wolf target 3m away, hostile-only gate, 5m range.
    // All gates should pass → Cast bit true.
    let (mut state, mut scratch, mut events, cascade, invariants) = setup(|b| {
        let _ = b.register(AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 10.0 }],
        ));
    });
    let caster = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    let _wolf = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(3.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    run_one_tick(&mut state, &mut scratch, &mut events, &cascade, &invariants);
    let slot = (caster.raw() - 1) as usize;
    assert!(cast_bit(&scratch.mask, slot),
        "hostile in range + all gates pass → Cast bit true");
}

#[test]
fn cast_bit_false_when_caster_stunned() {
    // Hostile in range, but caster is stunned → gate fails.
    let (mut state, mut scratch, mut events, cascade, invariants) = setup(|b| {
        let _ = b.register(AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 10.0 }],
        ));
    });
    let caster = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    let _wolf = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(3.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    // Task 143 — set an absolute expiry of state.tick+5. Mask read
    // sees `state.tick < expires_at_tick` → stunned, Cast bit flipped off.
    state.set_agent_stun_expires_at(caster, state.tick + 5);
    run_one_tick(&mut state, &mut scratch, &mut events, &cascade, &invariants);
    let slot = (caster.raw() - 1) as usize;
    assert!(!cast_bit(&scratch.mask, slot),
        "stunned caster → Cast bit must be false");
}

#[test]
fn cast_bit_permissive_when_registry_is_empty() {
    // Empty `state.ability_registry` → mask falls back to permissive for
    // Cast (legacy behaviour pre-migration, when "no cast handler
    // registered" was the trigger). `with_engine_builtins()` always
    // installs the CastHandler now, but an empty registry means
    // mask-build has no candidate ability to probe the gate with, so the
    // bit stays permissive.
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::with_engine_builtins();
    let invariants = InvariantRegistry::new();
    let caster = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    run_one_tick(&mut state, &mut scratch, &mut events, &cascade, &invariants);
    let slot = (caster.raw() - 1) as usize;
    assert!(cast_bit(&scratch.mask, slot),
        "empty state.ability_registry → Cast bit must fall back permissive");
}
