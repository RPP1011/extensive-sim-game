//! Combat Foundation Task 11 — `EffectOp::Heal` handler.
//!
//! Cascade handler on `Event::EffectHealApplied`. Raises the target's hp by
//! `amount`, clamped to `max_hp`. Mirrors the Eat/Drink/Rest clamp semantics
//! in `step.rs::restore_need`: the clamp is silent, no overflow event, the
//! `amount` carried on the event is the REQUESTED heal. Replays reproduce
//! the post-clamp hp by running the handler again — determinism lives in
//! the state, not the event delta.
//!
//! Heal on a dead target is a no-op (can't heal the deceased via this path —
//! resurrection is a separate ability surface).

use crate::cascade::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
use crate::state::SimState;

pub struct HealHandler;

impl CascadeHandler for HealHandler {
    fn trigger(&self) -> EventKindId { EventKindId::EffectHealApplied }
    fn lane(&self) -> Lane { Lane::Effect }

    fn handle(&self, event: &Event, state: &mut SimState, _events: &mut EventRing) {
        let (target, amount) = match *event {
            Event::EffectHealApplied { target, amount, .. } => (target, amount),
            _ => return,
        };
        if !state.agent_alive(target) { return; }
        if amount <= 0.0 { return; }

        let cur_hp = state.agent_hp(target).unwrap_or(0.0);
        let max_hp = state.agent_max_hp(target).unwrap_or(cur_hp);
        let new_hp = (cur_hp + amount).min(max_hp);
        state.set_agent_hp(target, new_hp);
    }
}
