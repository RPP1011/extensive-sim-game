//! Tick-start timer expiry — decrements `hot_stun_remaining_ticks` and
//! `hot_slow_remaining_ticks` and emits `StunExpired` / `SlowExpired` on the
//! tick each reaches zero. `hot_cooldown_next_ready_tick` is an absolute
//! tick — no decrement, just compared against `state.tick` by the mask
//! predicate.
//!
//! Task 139 retired this module's other responsibility — the engagement
//! update pass. That logic moved to the event-driven
//! `crate::engagement::*` cascade handlers keyed on `AgentMoved` /
//! `AgentDied`, with the per-agent slot storage folded by the
//! compiler-emitted `@materialized view engaged_with`. What remains here
//! is engine-internal timer scheduling (not game rules) and stays
//! hand-written until a timestamp-based cooldown migration lands.
//!
//! The `OpportunityAttackTriggered` cascade handler that used to live in
//! this file was migrated to DSL and is emitted as
//! `crate::generated::physics::opportunity_attack::OpportunityAttackHandler`.

use crate::event::Event;
use crate::event::EventRing;
use crate::state::SimState;
use crate::step::SimScratch;

// Engagement range + engagement-slow factor moved into `assets/sim/config.sim`:
// they read off `SimState.config.combat.engagement_range` and
// `config.combat.engagement_slow_factor` respectively. Default values match
// the old `ENGAGEMENT_RANGE = 2.0` / `ENGAGEMENT_SLOW_FACTOR = 0.3` consts
// exactly, so behaviour is unchanged; TOML tuning is additive.
//
// Task 142 retired the backward-compat `pub const` shims that used to
// live here — every consumer now reads
// `engine_rules::config::Config::default()` (or `state.config.combat.*`)
// instead. Keep this file stateless timer-tick code only.

/// Back-compat re-export of the legacy tick-start entry point. The name
/// predates task 139's split (engagement moved to
/// `crate::engagement::recompute_engagement_for`); the call shape stays
/// the same so fixture tests don't need churn. New code should call
/// [`tick_start_timers`] directly.
pub fn tick_start(state: &mut SimState, scratch: &mut SimScratch, events: &mut EventRing) {
    tick_start_timers(state, scratch, events);
}

/// Decrement `hot_stun_remaining_ticks` and `hot_slow_remaining_ticks`,
/// emitting `StunExpired` / `SlowExpired` on the tick each reaches zero.
/// Called from `step_full` BEFORE `scratch.mask.reset()` so mask
/// predicates see the post-decrement state.
pub fn tick_start_timers(state: &mut SimState, scratch: &mut SimScratch, events: &mut EventRing) {
    let tick = state.tick;
    // Reuse the hoisted scratch so the per-tick allocation stays at zero.
    // `engagement_alive_ids` is no longer consumed by an engagement pass
    // (task 139), but the buffer still serves the timer walk below — it's
    // sized once in `SimScratch::new` and cleared here.
    scratch.engagement_alive_ids.clear();
    scratch.engagement_alive_ids.extend(state.agents_alive());
    for &id in &scratch.engagement_alive_ids {
        let stun = state.agent_stun_remaining(id).unwrap_or(0);
        if stun > 0 {
            let new_stun = stun - 1;
            state.set_agent_stun_remaining(id, new_stun);
            if new_stun == 0 {
                events.push(Event::StunExpired { agent_id: id, tick });
            }
        }

        let slow = state.agent_slow_remaining(id).unwrap_or(0);
        if slow > 0 {
            let new_slow = slow - 1;
            state.set_agent_slow_remaining(id, new_slow);
            if new_slow == 0 {
                events.push(Event::SlowExpired { agent_id: id, tick });
                // Clear the factor on expiry so mask/move code treats the
                // agent as unslowed. Factor is only meaningful while
                // remaining > 0.
                state.set_agent_slow_factor_q8(id, 0);
            }
        }
        // cooldown_next_ready_tick is absolute — no decrement needed.
    }
}
