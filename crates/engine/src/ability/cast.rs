//! `CastHandler` — the single cascade handler keyed on `EventKindId::AgentCast`.
//!
//! Shape:
//! 1. Handler is a zero-sized unit struct — the ability program table lives
//!    on `SimState::ability_registry`. The handler reads it there on every
//!    cast. Moving the registry onto state retired the `Arc<AbilityRegistry>`
//!    the handler used to carry and let us fold the registration into
//!    `CascadeRegistry::register_engine_builtins` like every other effect
//!    handler.
//! 2. On each `AgentCast`, look the program up. If the ability is unknown
//!    (the state may have been mutated mid-cascade such that the id no
//!    longer resolves), silently drop — the invalidation event is a
//!    telemetry concern, not a gate one.
//! 3. Iterate `program.effects` in order and emit one `Effect*Applied` per
//!    `EffectOp`. Per-effect handlers (Tasks 10–17) pick those up and do
//!    the actual state mutation.
//! 4. Set the caster's cooldown: `next_ready = current_tick + gate.cooldown_ticks`.
//!    `gate.cooldown_ticks == 0` means "always ready" — cooldown never blocks.
//!
//! Recursive casts (`EffectOp::CastAbility`) are emitted as nested
//! `AgentCast` events on the same ring. Depth is tracked per event on
//! `Event::AgentCast.depth` (Task 18): root casts from action dispatch
//! carry `depth = 0`, each recursive hop increments by one. When the
//! handler is about to push a nested cast whose depth would reach
//! `MAX_CASCADE_ITERATIONS`, it emits `Event::CastDepthExceeded` instead
//! and skips the push. This keeps CastHandler self-bounded — the cascade
//! framework's 8-iteration ceiling never fires for cast recursion, so the
//! dev-build `cascade did not converge` panic is reserved for OTHER
//! handlers (see `cascade_bounded.rs`).
//!
//! Retiring this file in favour of a DSL `physics cast` rule is blocked on
//! the `emit_physics` compiler growing `for ... in <collection>` loops
//! and `match` over sum-type variants (`EffectOp::*`). See
//! `docs/game/compiler_progress.md` for the tracking row. Until then the
//! handler stays hand-written but stateless, which is the shape the DSL
//! emitter already supports for every other effect.

use crate::cascade::{CascadeHandler, EventKindId, Lane, MAX_CASCADE_ITERATIONS};
use crate::event::{Event, EventRing};
use crate::ids::AgentId;
use crate::state::SimState;

use super::{AbilityId, EffectOp, TargetSelector};

/// Zero-sized handler — the ability program table the handler dispatches
/// against lives on `SimState::ability_registry`. No per-handler state.
pub struct CastHandler;

impl CastHandler {
    pub const fn new() -> Self { Self }
}

impl Default for CastHandler {
    fn default() -> Self { Self::new() }
}

impl CascadeHandler for CastHandler {
    fn trigger(&self) -> EventKindId { EventKindId::AgentCast }
    fn lane(&self) -> Lane { Lane::Effect }
    fn as_any(&self) -> Option<&dyn std::any::Any> { Some(self) }

    fn handle(&self, event: &Event, state: &mut SimState, events: &mut EventRing) {
        let (caster, ability, target, depth, tick) = match *event {
            Event::AgentCast { actor, ability, target, depth, tick } =>
                (actor, ability, target, depth, tick),
            _ => return,
        };
        let prog = match state.ability_registry.get(ability) {
            Some(p) => p,
            None    => return,
        };

        // Collect cooldown first so we can use it after the effects loop —
        // we release the borrow on `prog` by dereferencing the smallvec into
        // owned `EffectOp` copies (they're `Copy`).
        let cooldown_ticks = prog.gate.cooldown_ticks;
        let effects: smallvec::SmallVec<[EffectOp; super::MAX_EFFECTS_PER_PROGRAM]> =
            prog.effects.iter().copied().collect();

        for op in effects {
            emit_effect_event(op, caster, target, ability, depth, events, tick);
        }

        // Cooldown: next_ready tick is absolute. `0` cooldown leaves
        // next_ready == tick, which the gate's `state.tick < next_ready`
        // check treats as "ready now".
        let next_ready = tick.saturating_add(cooldown_ticks);
        state.set_agent_cooldown_next_ready(caster, next_ready);
    }
}

/// Emit exactly one `Effect*Applied` event per `EffectOp`. No state mutation
/// here — the effect's own cascade handler (registered in Tasks 10–17) folds
/// the event into SoA. Keeping dispatch and effect-apply in separate handlers
/// keeps recursion bounded (the dispatch loop sees each nested cast as a
/// distinct event, so `MAX_CASCADE_ITERATIONS` works as designed).
///
/// `parent_ability` / `parent_depth` / `tick` are used only when `op` is
/// `EffectOp::CastAbility` — they let us emit a `CastDepthExceeded` audit
/// event carrying the PARENT's ability id on overflow, so the audit trail
/// points at the ability that attempted the too-deep recursion.
fn emit_effect_event(
    op:             EffectOp,
    caster:         AgentId,
    target:         AgentId,
    parent_ability: AbilityId,
    parent_depth:   u8,
    events:         &mut EventRing,
    tick:           u32,
) {
    match op {
        EffectOp::Damage { amount } => {
            events.push(Event::EffectDamageApplied { actor: caster, target, amount, tick });
        }
        EffectOp::Heal { amount } => {
            events.push(Event::EffectHealApplied { actor: caster, target, amount, tick });
        }
        EffectOp::Shield { amount } => {
            events.push(Event::EffectShieldApplied { actor: caster, target, amount, tick });
        }
        EffectOp::Stun { duration_ticks } => {
            // Task 143 — the event carries the absolute expiry tick. The
            // DSL-authored `duration_ticks` on the `EffectOp` stays (so
            // ability authors keep specifying the duration they want); we
            // compose it with `tick` here so consumers can treat the event
            // as a pure "set the expiry" directive.
            let expires_at_tick = tick.saturating_add(duration_ticks);
            events.push(Event::EffectStunApplied { actor: caster, target, expires_at_tick, tick });
        }
        EffectOp::Slow { duration_ticks, factor_q8 } => {
            let expires_at_tick = tick.saturating_add(duration_ticks);
            events.push(Event::EffectSlowApplied {
                actor: caster, target, expires_at_tick, factor_q8, tick,
            });
        }
        EffectOp::TransferGold { amount } => {
            events.push(Event::EffectGoldTransfer { from: caster, to: target, amount, tick });
        }
        EffectOp::ModifyStanding { delta } => {
            events.push(Event::EffectStandingDelta { a: caster, b: target, delta, tick });
        }
        EffectOp::CastAbility { ability, selector } => {
            let effective_target = match selector {
                TargetSelector::Target => target,
                TargetSelector::Caster => caster,
            };
            // Depth cap (Combat Foundation Task 18). Each nested cast is one
            // deeper than its parent. When the NEW depth would reach or
            // exceed the cascade framework's iteration bound we emit a
            // `CastDepthExceeded` audit event instead of pushing the nested
            // cast. This keeps the recursion self-bounded below the
            // framework's own iteration ceiling, so dev-build panics never
            // fire for cast chains.
            let new_depth = parent_depth.saturating_add(1);
            if (new_depth as usize) >= MAX_CASCADE_ITERATIONS {
                events.push(Event::CastDepthExceeded {
                    actor: caster, ability: parent_ability, tick,
                });
                return;
            }
            events.push(Event::AgentCast {
                actor: caster, ability, target: effective_target, depth: new_depth, tick,
            });
        }
    }
}
