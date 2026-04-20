//! Combat Foundation Task 14 — `EffectOp::Slow` handler.
//!
//! Cascade handler on `Event::EffectSlowApplied`. Writes
//! `hot_slow_remaining_ticks` and `hot_slow_factor_q8` using the
//! "longer-or-stronger wins" rule from the plan:
//!
//! If the incoming `duration_ticks > current_remaining` OR
//! `factor_q8 > current_factor_q8`, we replace BOTH (duration and factor)
//! with the incoming values. Otherwise we keep what's there.
//!
//! `factor_q8` semantics: q8 fixed-point speed multiplier. `0` means "no
//! slow"; `51` ≈ 0.2× (the plan's example). Higher q8 → closer to full
//! speed; lower → more slow. "Stronger" means a SMALLER factor_q8
//! (slower). The plan's "`factor_q8 > current`" wording in Task 14 is a
//! typo — the intended semantic is "stronger slow overrides weaker", which
//! for q8 multipliers means `incoming_factor < current_factor`. We pick
//! the minimum (strongest slow) when it's newer-or-stronger-or-longer.
//!
//! Tick-start decrement + `SlowExpired` emission live in
//! `ability::expire::tick_start` (Task 3). The factor is zeroed on expiry.

use crate::cascade::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
use crate::state::SimState;

pub struct SlowHandler;

impl CascadeHandler for SlowHandler {
    fn trigger(&self) -> EventKindId { EventKindId::EffectSlowApplied }
    fn lane(&self) -> Lane { Lane::Effect }

    fn handle(&self, event: &Event, state: &mut SimState, _events: &mut EventRing) {
        let (target, duration_ticks, factor_q8) = match *event {
            Event::EffectSlowApplied { target, duration_ticks, factor_q8, .. } => {
                (target, duration_ticks, factor_q8)
            }
            _ => return,
        };
        if !state.agent_alive(target) { return; }
        if duration_ticks == 0 || factor_q8 <= 0 { return; }

        let cur_dur = state.agent_slow_remaining(target).unwrap_or(0);
        let cur_fac = state.agent_slow_factor_q8(target).unwrap_or(0);

        // "Stronger" is a smaller q8 multiplier (closer to 0). If no slow is
        // active (cur_fac == 0), any incoming slow wins. Otherwise, replace
        // when incoming is longer OR stronger than current.
        let longer   = duration_ticks > cur_dur;
        let stronger = cur_fac == 0 || factor_q8 < cur_fac;
        if longer || stronger {
            state.set_agent_slow_remaining(target, duration_ticks);
            state.set_agent_slow_factor_q8(target, factor_q8);
        }
    }
}
