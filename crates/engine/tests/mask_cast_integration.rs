//! Audit fix CRITICAL #2 — `mark_domain_hook_micros_allowed` must now consult
//! `evaluate_cast_gate` via the registered `CastHandler`'s `AbilityRegistry`.
//!
//! Complements `mask_can_cast.rs` (which tests the gate predicate in
//! isolation). Each test here drives `step_full`, registers a cast handler
//! with a specific `AbilityRegistry`, and asserts the cast mask bit flips
//! appropriately for alive agents under different gate conditions.

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
    fn evaluate(&self, _: &SimState, _: &MaskBuffer, _: &mut Vec<Action>) {}
}

fn setup(registry_build: impl FnOnce(&mut AbilityRegistryBuilder)) -> (
    SimState, SimScratch, EventRing, CascadeRegistry, InvariantRegistry,
) {
    let state = SimState::new(8, 42);
    let scratch = SimScratch::new(state.agent_cap() as usize);
    let events = EventRing::with_cap(1024);
    let mut b = AbilityRegistryBuilder::new();
    registry_build(&mut b);
    let reg = std::sync::Arc::new(b.build());
    let mut cascade = CascadeRegistry::with_engine_builtins();
    cascade.register_cast_handler(reg);
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
        })
        .unwrap();
    let _wolf = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(3.0, 0.0, 0.0),
            hp: 100.0,
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
        })
        .unwrap();
    let _wolf = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(3.0, 0.0, 0.0),
            hp: 100.0,
        })
        .unwrap();
    // Stun for 5 ticks; tick_start decrements by 1 before the mask phase,
    // so remaining 4 at mask-build time — still stunned, still blocked.
    state.set_agent_stun_remaining(caster, 5);
    run_one_tick(&mut state, &mut scratch, &mut events, &cascade, &invariants);
    let slot = (caster.raw() - 1) as usize;
    assert!(!cast_bit(&scratch.mask, slot),
        "stunned caster → Cast bit must be false");
}

#[test]
fn cast_bit_permissive_when_no_cast_handler_registered() {
    // No cast handler → mask falls back to permissive for Cast (legacy
    // behaviour). Registering other engine builtins must not enable the
    // gate path.
    let state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::with_engine_builtins();
    let invariants = InvariantRegistry::new();
    let mut state = state;
    let caster = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
        })
        .unwrap();
    run_one_tick(&mut state, &mut scratch, &mut events, &cascade, &invariants);
    let slot = (caster.raw() - 1) as usize;
    assert!(cast_bit(&scratch.mask, slot),
        "no cast handler registered → Cast bit must fall back permissive");
}
