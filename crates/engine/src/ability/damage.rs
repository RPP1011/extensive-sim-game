//! Combat Foundation Task 10 — `EffectOp::Damage` handler.
//!
//! Cascade handler on `Event::EffectDamageApplied`. Applies `amount` hp damage
//! to the target with shield-first absorption:
//!
//! 1. Subtract from `hot_shield_hp` first; any overflow lands on `hot_hp`.
//! 2. If the post-damage hp reaches zero, emit `AgentDied` and `kill_agent`.
//!
//! Mirrors the direct-`Attack` death semantics in `step.rs`, so cast-delivered
//! and melee-delivered kills trigger identical replayable events.
//!
//! Registered by `CascadeRegistry::register_engine_builtins` alongside the
//! opportunity-attack handler — no separate opt-in is required once the
//! engine defaults are wired.

use crate::cascade::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
use crate::state::SimState;

pub struct DamageHandler;

impl CascadeHandler for DamageHandler {
    fn trigger(&self) -> EventKindId { EventKindId::EffectDamageApplied }
    fn lane(&self) -> Lane { Lane::Effect }

    fn handle(&self, event: &Event, state: &mut SimState, events: &mut EventRing) {
        let (target, amount, tick) = match *event {
            Event::EffectDamageApplied { target, amount, tick, .. } => (target, amount, tick),
            _ => return,
        };
        if !state.agent_alive(target) { return; }
        if amount <= 0.0 { return; }

        // Shield absorbs first; overflow hits hp.
        let shield = state.agent_shield_hp(target).unwrap_or(0.0);
        let (new_shield, overflow) = if shield >= amount {
            (shield - amount, 0.0)
        } else {
            (0.0, amount - shield)
        };
        state.set_agent_shield_hp(target, new_shield);

        if overflow > 0.0 {
            let cur_hp = state.agent_hp(target).unwrap_or(0.0);
            let new_hp = (cur_hp - overflow).max(0.0);
            state.set_agent_hp(target, new_hp);
            if new_hp <= 0.0 {
                events.push(Event::AgentDied { agent_id: target, tick });
                state.kill_agent(target);
            }
        }
    }
}
