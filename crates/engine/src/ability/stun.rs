//! Combat Foundation Task 13 — `EffectOp::Stun` handler.
//!
//! Cascade handler on `Event::EffectStunApplied`. Writes
//! `hot_stun_remaining_ticks[target]` using a "longer stun wins" rule:
//! `max(existing, duration_ticks)`. No decrement here — the unified
//! tick-start pass (`ability::expire::tick_start`, Task 3) decrements
//! the remaining counter at tick boundaries and emits `StunExpired`
//! when it transitions to zero.
//!
//! Stunned agents fail the cast-gate (`gate.rs` branch 1) and — once the
//! Plan-2 move-allowed mask predicate lands — also fail it. MVP: the
//! mask is permissive on Cast; cooldown + stun filtering happens inside
//! `evaluate_cast_gate`.

use crate::cascade::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
use crate::state::SimState;

pub struct StunHandler;

impl CascadeHandler for StunHandler {
    fn trigger(&self) -> EventKindId { EventKindId::EffectStunApplied }
    fn lane(&self) -> Lane { Lane::Effect }

    fn handle(&self, event: &Event, state: &mut SimState, _events: &mut EventRing) {
        let (target, duration_ticks) = match *event {
            Event::EffectStunApplied { target, duration_ticks, .. } => (target, duration_ticks),
            _ => return,
        };
        if !state.agent_alive(target) { return; }
        if duration_ticks == 0 { return; }

        let cur = state.agent_stun_remaining(target).unwrap_or(0);
        let new = cur.max(duration_ticks);
        state.set_agent_stun_remaining(target, new);
    }
}
