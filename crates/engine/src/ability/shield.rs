//! Combat Foundation Task 12 — `EffectOp::Shield` handler.
//!
//! Cascade handler on `Event::EffectShieldApplied`. Adds `amount` to the
//! target's `hot_shield_hp`. Stackable; no cap, no expiry — MVP shield is
//! a pure additive absorb layer consumed by `DamageHandler` (Task 10).
//!
//! Follow-ups: Ability Plan 2 adds timed shields (tick-bounded expiry) and
//! per-shield typed slots for damage-type filtering. This MVP path is kept
//! deliberately simple so the core damage-absorption invariant
//! (`overflow = max(0, damage - shield)`) has one owner.

use crate::cascade::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
use crate::state::SimState;

pub struct ShieldHandler;

impl CascadeHandler for ShieldHandler {
    fn trigger(&self) -> EventKindId { EventKindId::EffectShieldApplied }
    fn lane(&self) -> Lane { Lane::Effect }

    fn handle(&self, event: &Event, state: &mut SimState, _events: &mut EventRing) {
        let (target, amount) = match *event {
            Event::EffectShieldApplied { target, amount, .. } => (target, amount),
            _ => return,
        };
        if !state.agent_alive(target) { return; }
        if amount <= 0.0 { return; }

        let cur = state.agent_shield_hp(target).unwrap_or(0.0);
        state.set_agent_shield_hp(target, cur + amount);
    }
}
