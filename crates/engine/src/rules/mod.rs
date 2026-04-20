//! Hand-written game-view fns the compiler emits calls to.
//!
//! When a `mask` / `physics` / `scoring` DSL source references a callable
//! name the compiler can't resolve to a `view`, a stdlib method, or a
//! builtin, the emitter lowers the call as `crate::rules::<name>(state,
//! args...)`. Each such fn must exist hand-written in this module (or a
//! submodule).
//!
//! This module shrinks as later milestones let the DSL declare views
//! directly (see `docs/game/compiler_progress.md` row 6 — `view`
//! declarations). Once a game-view can be DSL-declared, its hand-written
//! counterpart here is deleted in the same commit.
//!
//! ## Why not put these on `engine::creature` / `engine::state`?
//!
//! These are *game views*, not engine primitives. The engine doesn't know
//! what "hostile" means — that's a game rule. Putting the fn on
//! `CreatureType::is_hostile_to` worked while the rule had one caller
//! (`evaluate_cast_gate`); the compiler-emitted mask is a second caller
//! and the DSL-emission convention points at exactly one place. The
//! legacy `CreatureType::is_hostile_to` method is kept for now because
//! `evaluate_cast_gate` still uses it directly; when the cast mask
//! migrates (milestone 4 follow-up), both callers route through here and
//! the method on `CreatureType` can be retired.

use crate::ids::AgentId;
use crate::state::SimState;

/// Pairwise hostility predicate. Delegates to the species-level default
/// table on `CreatureType`. Returns `false` when either agent lacks a
/// creature type (dead / uninitialised slot).
///
/// Compiler-emitted callers (e.g. `crate::generated::mask::mask_attack`)
/// route `is_hostile(self, target)` through this fn. Replaced by a
/// DSL-declared `view is_hostile` once milestone 6 lands.
pub fn is_hostile(state: &SimState, a: AgentId, b: AgentId) -> bool {
    let Some(ca) = state.agent_creature_type(a) else {
        return false;
    };
    let Some(cb) = state.agent_creature_type(b) else {
        return false;
    };
    ca.is_hostile_to(cb)
}
