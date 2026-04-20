//! `CastHandler` — the single cascade handler keyed on `EventKindId::AgentCast`.
//!
//! Shape:
//! 1. Handler is constructed around an `Arc<AbilityRegistry>` so multiple
//!    cascade lanes / tests can share one registry cheaply.
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
//! `AgentCast` events on the same ring. The cascade dispatch loop bounds
//! the chain at `MAX_CASCADE_ITERATIONS = 8`; Task 18's depth test pins
//! the budget. When an iterated cast would push beyond the bound, the
//! cascade truncates (in release) or panics (in debug) — the handler here
//! does not itself audit depth, that's the dispatcher's job.

use std::sync::Arc;

use crate::cascade::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
use crate::ids::AgentId;
use crate::state::SimState;

use super::{AbilityRegistry, EffectOp, TargetSelector};

pub struct CastHandler {
    registry: Arc<AbilityRegistry>,
}

impl CastHandler {
    pub fn new(registry: Arc<AbilityRegistry>) -> Self {
        Self { registry }
    }

    /// Shared handle to the registry this handler dispatches against.
    /// Useful for tests that want to assert the handler points at the
    /// registry they built.
    pub fn registry(&self) -> &Arc<AbilityRegistry> { &self.registry }
}

impl CascadeHandler for CastHandler {
    fn trigger(&self) -> EventKindId { EventKindId::AgentCast }
    fn lane(&self) -> Lane { Lane::Effect }

    fn handle(&self, event: &Event, state: &mut SimState, events: &mut EventRing) {
        let (caster, ability, target, tick) = match *event {
            Event::AgentCast { caster, ability, target, tick } => (caster, ability, target, tick),
            _ => return,
        };
        let prog = match self.registry.get(ability) {
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
            emit_effect_event(op, caster, target, events, tick);
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
fn emit_effect_event(
    op:     EffectOp,
    caster: AgentId,
    target: AgentId,
    events: &mut EventRing,
    tick:   u32,
) {
    match op {
        EffectOp::Damage { amount } => {
            events.push(Event::EffectDamageApplied { caster, target, amount, tick });
        }
        EffectOp::Heal { amount } => {
            events.push(Event::EffectHealApplied { caster, target, amount, tick });
        }
        EffectOp::Shield { amount } => {
            events.push(Event::EffectShieldApplied { caster, target, amount, tick });
        }
        EffectOp::Stun { duration_ticks } => {
            events.push(Event::EffectStunApplied { caster, target, duration_ticks, tick });
        }
        EffectOp::Slow { duration_ticks, factor_q8 } => {
            events.push(Event::EffectSlowApplied {
                caster, target, duration_ticks, factor_q8, tick,
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
            // Recursion: a nested AgentCast. The cascade dispatcher sees it on
            // the next fixed-point iteration and CastHandler runs again. Depth
            // is bounded by MAX_CASCADE_ITERATIONS; Task 18 pins the budget.
            events.push(Event::AgentCast {
                caster, ability, target: effective_target, tick,
            });
        }
    }
}
