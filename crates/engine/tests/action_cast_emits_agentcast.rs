//! Combat Foundation Task 9 — action→event→handler round trip for casts.
//!
//! Verifies the full chain:
//! 1. A `MicroKind::Cast + MicroTarget::Ability` action pushed into `step`
//!    produces an `Event::AgentCast`.
//! 2. With `state.ability_registry` populated, the (engine-builtin) cast
//!    handler fires and emits one `Effect*Applied` per `EffectOp`.
//! 3. With an empty registry on state, the `AgentCast` is emitted but no
//!    effects follow — the handler's `registry.get(id)` short-circuits.

use engine::ability::{
    AbilityProgram, AbilityRegistry, AbilityRegistryBuilder, EffectOp, Gate,
};
use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::AgentId;
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch}; // Plan B1' Task 11: step is unimplemented!() stub
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

fn build_one_damage_ability() -> (AbilityRegistry, engine::ability::AbilityId) {
    let mut b = AbilityRegistryBuilder::new();
    let id = b.register(AbilityProgram::new_single_target(
        6.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 25.0 }],
    ));
    (b.build(), id)
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn cast_action_emits_agentcast_event() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1024);
    // Plain `CascadeRegistry::<Event>::new()` does NOT register the cast handler,
    // so AgentCast fires with no effects even though the state's
    // ability_registry has an entry.
    let cascade = CascadeRegistry::<Event>::new();

    let a = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let b = spawn(&mut state, CreatureType::Wolf,  Vec3::new(3.0, 0.0, 0.0));

    let (reg, ability) = build_one_damage_ability();
    state.ability_registry = reg;
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

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn cast_action_triggers_effect_damage_when_handler_registered() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1024);

    let (reg, ability) = build_one_damage_ability();
    state.ability_registry = reg;
    // `with_engine_builtins()` installs the stateless CastHandler; the
    // registry on `state` drives program lookup.
    let cascade = CascadeRegistry::<Event>::with_engine_builtins();

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

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn cast_handler_starts_cooldown() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1024);

    let (reg, ability) = build_one_damage_ability();
    state.ability_registry = reg;
    let cascade = CascadeRegistry::<Event>::with_engine_builtins();

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

    // Ability-cooldowns subsystem (2026-04-22): the post-cast helper
    // writes BOTH cursors. Assertions updated accordingly.
    //   * global (GCD) cursor:  tick 0 + `combat.global_cooldown_ticks`
    //     (default 5) = 5.
    //   * local cooldown slot:  tick 0 + `ability.gate.cooldown_ticks`
    //     (= 10) = 10.
    // Previously both were fused onto the single global cursor (=10), which
    // was the shared-cursor bug the subsystem fixes.
    let gcd = state.config.combat.global_cooldown_ticks;
    assert_eq!(state.agent_cooldown_next_ready(a), Some(gcd));
    let agent_slot = (a.raw() - 1) as usize;
    assert_eq!(state.ability_cooldowns[agent_slot][0], 10);
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn cast_of_unknown_ability_id_is_a_no_op() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1024);

    // Empty registry on state — every ability id is unknown.
    state.ability_registry = AbilityRegistry::new();
    let cascade = CascadeRegistry::<Event>::with_engine_builtins();

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
