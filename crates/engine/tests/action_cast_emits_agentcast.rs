//! Combat Foundation Task 9 — action→event→handler round trip for casts.
//!
//! Verifies the full chain:
//! 1. A `MicroKind::Cast + MicroTarget::Ability` action pushed into `step`
//!    produces an `Event::AgentCast`.
//! 2. If a `CastHandler` is registered on the cascade for that registry,
//!    the handler fires and emits one `Effect*Applied` per `EffectOp`.
//! 3. With no cast handler registered, the `AgentCast` is emitted but no
//!    effects follow (regression guard against accidental default effects).

use std::sync::Arc;

use engine::ability::{
    AbilityProgram, AbilityRegistry, AbilityRegistryBuilder, EffectOp, Gate,
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

struct EmitOnce { caster: AgentId, kind: ActionKind }
impl PolicyBackend for EmitOnce {
    fn evaluate(&self, _state: &SimState, _m: &MaskBuffer, _target_mask: &engine::mask::TargetMask, out: &mut Vec<Action>) {
        out.push(Action { agent: self.caster, kind: self.kind });
    }
}

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos, hp: 100.0, ..Default::default() }).unwrap()
}

fn build_one_damage_ability() -> (Arc<AbilityRegistry>, engine::ability::AbilityId) {
    let mut b = AbilityRegistryBuilder::new();
    let id = b.register(AbilityProgram::new_single_target(
        6.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 25.0 }],
    ));
    (Arc::new(b.build()), id)
}

#[test]
fn cast_action_emits_agentcast_event() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();  // intentionally no CastHandler

    let a = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let b = spawn(&mut state, CreatureType::Wolf,  Vec3::new(3.0, 0.0, 0.0));

    let (_reg, ability) = build_one_damage_ability();
    let backend = EmitOnce {
        caster: a,
        kind:   ActionKind::Micro {
            kind:   MicroKind::Cast,
            target: MicroTarget::Ability { id: ability, target: b },
        },
    };

    step(&mut state, &mut scratch, &mut events, &backend, &cascade);

    let got = events.iter().any(|e| matches!(e,
        Event::AgentCast { actor, ability: ab, target, .. }
            if *actor == a && ab.raw() == ability.raw() && *target == b));
    assert!(got, "AgentCast event expected after Cast action");

    // Without a CastHandler registered, no effect event fires.
    let any_effect = events.iter().any(|e| matches!(e, Event::EffectDamageApplied { .. }));
    assert!(!any_effect, "no CastHandler → no Effect*Applied");
}

#[test]
fn cast_action_triggers_effect_damage_when_handler_registered() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);

    let (reg, ability) = build_one_damage_ability();
    let mut cascade = CascadeRegistry::new();
    cascade.register_cast_handler(reg.clone());

    let a = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let b = spawn(&mut state, CreatureType::Wolf,  Vec3::new(3.0, 0.0, 0.0));

    let backend = EmitOnce {
        caster: a,
        kind:   ActionKind::Micro {
            kind:   MicroKind::Cast,
            target: MicroTarget::Ability { id: ability, target: b },
        },
    };

    step(&mut state, &mut scratch, &mut events, &backend, &cascade);

    // Handler should have fired an EffectDamageApplied at the target with amount=25.
    let found = events.iter().any(|e| matches!(e,
        Event::EffectDamageApplied { actor, target, amount, .. }
            if *actor == a && *target == b && (*amount - 25.0).abs() < 1e-5));
    assert!(found, "EffectDamageApplied expected after CastHandler dispatch");
}

#[test]
fn cast_handler_starts_cooldown() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);

    let (reg, ability) = build_one_damage_ability();
    let mut cascade = CascadeRegistry::new();
    cascade.register_cast_handler(reg);

    let a = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let b = spawn(&mut state, CreatureType::Wolf,  Vec3::new(3.0, 0.0, 0.0));

    assert_eq!(state.agent_cooldown_next_ready(a), Some(0));

    let backend = EmitOnce {
        caster: a,
        kind:   ActionKind::Micro {
            kind:   MicroKind::Cast,
            target: MicroTarget::Ability { id: ability, target: b },
        },
    };

    step(&mut state, &mut scratch, &mut events, &backend, &cascade);

    // Cast resolved on tick 0 with cooldown_ticks=10 → next_ready_tick=10.
    assert_eq!(state.agent_cooldown_next_ready(a), Some(10));
}

#[test]
fn cast_of_unknown_ability_id_is_a_no_op() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);

    // Empty registry — every ability id is unknown.
    let reg = Arc::new(AbilityRegistry::new());
    let mut cascade = CascadeRegistry::new();
    cascade.register_cast_handler(reg);

    let a = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let b = spawn(&mut state, CreatureType::Wolf,  Vec3::new(3.0, 0.0, 0.0));
    let bogus = engine::ability::AbilityId::new(99).unwrap();

    let backend = EmitOnce {
        caster: a,
        kind:   ActionKind::Micro {
            kind:   MicroKind::Cast,
            target: MicroTarget::Ability { id: bogus, target: b },
        },
    };

    step(&mut state, &mut scratch, &mut events, &backend, &cascade);

    // Unknown id: AgentCast fired (always emitted on action), but no effect,
    // no cooldown change.
    let any_effect = events.iter().any(|e| matches!(e, Event::EffectDamageApplied { .. }));
    assert!(!any_effect);
    assert_eq!(state.agent_cooldown_next_ready(a), Some(0));
}
