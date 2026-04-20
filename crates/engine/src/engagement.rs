//! Engagement back-compat shim — post task 163.
//!
//! The hand-written cascade dispatchers and the `recompute_engagement_for` /
//! `break_engagement_on_death` bodies retired 2026-04-20 once the compiler
//! grew the four primitives the engagement rules need:
//!
//!   - `query.nearest_hostile_to` / `nearest_hostile_to_or` — spatial
//!     argmin with species-hostility filter (wraps
//!     `crate::spatial::nearest_hostile_to`).
//!   - `agents.set_engaged_with` / `clear_engaged_with` — eager writes
//!     to the `hot_engaged_with` slot so same-tick cascade handlers
//!     read-their-own-writes before the view-fold phase.
//!   - `agents.engaged_with_or` — unwrap-or-default sibling of the
//!     existing `agents.engaged_with` accessor, sidesteps the lack of
//!     `if let Some(...)` in the GPU-emittable physics subset.
//!
//! The corresponding `physics engagement_on_move` / `engagement_on_death`
//! rules live in `assets/sim/physics.sim`; the compiler emits them as
//! `generated/physics/engagement_on_move.rs` +
//! `generated/physics/engagement_on_death.rs` and wires them into the
//! per-event-kind dispatcher in `generated/physics/mod.rs`. No
//! hand-written `register_engine_builtins` entry needed.
//!
//! What remains in this module:
//!
//!   1. The `break_reason` u8 constants, which are how the DSL rule
//!      communicates the "why did this pair end" code to replayers
//!      without adding a new event kind. Engine + DSL both read these
//!      constants, so they live here rather than on the event itself.
//!   2. A thin `recompute_all_engagements` shim kept for the two
//!      fixtures that predate the event-driven pipeline
//!      (`tests/proptest_engagement.rs`,
//!      `tests/engagement_tick_start.rs`). The shim walks alive agents
//!      and emits one `AgentMoved` event each, then drains the cascade
//!      — the engagement rule fires on each, producing the same
//!      steady-state pairing the legacy hand-written loop produced,
//!      without duplicating any of the logic.
//!
//! New code should not use the shim; emit `AgentMoved` and let the
//! cascade do the rest.

use crate::cascade::CascadeRegistry;
use crate::event::{Event, EventRing};
use crate::ids::AgentId;
use crate::state::SimState;
use glam::Vec3;

/// Reason codes carried on `EngagementBroken` so replayers can tell why
/// a pairing ended without re-running the physics. Kept as a `u8` on
/// the event so the DSL schema stays trivially hashable. The
/// `engagement_on_move` / `engagement_on_death` rules in
/// `assets/sim/physics.sim` emit these numeric literals directly so
/// the DSL doesn't need a typed enum here.
pub mod break_reason {
    /// Mover switched to a new nearest hostile (the old partner is now
    /// stale).
    pub const SWITCH: u8 = 0;
    /// Mover left the engagement radius — no valid partner.
    pub const OUT_OF_RANGE: u8 = 1;
    /// Partner died; the survivor's slot is cleared so the next
    /// movement-triggered recompute starts from a clean state.
    pub const PARTNER_DIED: u8 = 2;
}

/// Back-compat shim for tests that predate the event-driven pipeline.
///
/// Walks every alive agent in slot order, emits a synthetic
/// `AgentMoved` at the agent's current position for each, and runs the
/// cascade to fixed point on the provided registry. The DSL
/// `engagement_on_move` physics rule does the actual work — the shim
/// just plays the moves that `step_full`'s movement phase would
/// normally have emitted.
///
/// Re-emits are safe: the rule compares the pre-existing pairing
/// against the recomputed one and only emits `EngagementBroken` /
/// `EngagementCommitted` on a change. A steady-state second pass
/// produces no new events.
///
/// Legacy callers: `tests/proptest_engagement.rs`,
/// `tests/engagement_tick_start.rs`. New code should emit `AgentMoved`
/// directly from the movement phase and let the cascade converge — no
/// need to reach into engagement by hand.
pub fn recompute_all_engagements(state: &mut SimState, events: &mut EventRing) {
    let registry = CascadeRegistry::with_engine_builtins();
    let tick = state.tick;
    let alive: Vec<AgentId> = state.agents_alive().collect();
    for id in alive {
        let pos = state.agent_pos(id).unwrap_or(Vec3::ZERO);
        // `from` / `location` both at the current pos means "no displacement"
        // — the rule only reads `actor`, so the other fields can match the
        // pre-cascade state exactly.
        events.push(Event::AgentMoved {
            actor: id,
            from: pos,
            location: pos,
            tick,
        });
    }
    registry.run_fixed_point(state, events);
}
