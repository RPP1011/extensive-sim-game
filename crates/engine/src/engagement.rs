//! Engagement — post task 163 + task 164.
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
//! What remains in this module is the `break_reason` u8 constants, which
//! are how the DSL rule communicates the "why did this pair end" code to
//! replayers without adding a new event kind. Engine + DSL both read
//! these constants, so they live here rather than on the event itself.

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
