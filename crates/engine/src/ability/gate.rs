//! `evaluate_cast_gate` — the cast-time predicate the mask-building and
//! the `CastHandler` both consult.
//!
//! Returns `true` exactly when *all* of the following hold:
//!
//! 1. **Caster alive + un-stunned.** `agent_alive(caster)` is true and
//!    `agent_stunned(caster)` is false (task 143 — stun is a synthetic
//!    boundary on `state.tick < stun_expires_at_tick`).
//! 2. **Cooldown ready.** `state.tick >= agent_cooldown_next_ready(caster)`.
//! 3. **Ability registered.** `registry.get(ability)` is `Some`.
//! 4. **Target alive + in range.** `agent_alive(target)` is true and
//!    `agent_pos(caster).distance(agent_pos(target)) <= Area.range`.
//! 5. **Hostility matches** the program's `gate.hostile_only`: when set,
//!    `CreatureType::is_hostile_to` must return `true` for both directions
//!    (hostility is symmetric in the stub; when per-pair standing replaces
//!    it this predicate will re-front).
//! 6. **Engagement lock respected.** If the caster is currently `engaged_with`
//!    somebody other than `target`, the cast is forbidden — the caster is
//!    locked in melee with their engager. Moving toward the engager (melee)
//!    is still allowed via the MoveToward path; casting at a different
//!    agent is not.
//!
//! All six predicates are short-circuited. A `false` returns early; the
//! function does no mutation. Adversarial test cases in `mask_can_cast.rs`
//! exercise each branch independently.

use super::{AbilityId, AbilityRegistry, Area};
use crate::ids::AgentId;
use crate::state::SimState;

pub fn evaluate_cast_gate(
    state:    &SimState,
    registry: &AbilityRegistry,
    caster:   AgentId,
    ability:  AbilityId,
    target:   AgentId,
) -> bool {
    // 1. Caster alive + un-stunned. Task 143: `agent_stunned` is the
    // synthetic boundary read — `state.tick < stun_expires_at_tick`.
    if !state.agent_alive(caster) { return false; }
    if state.agent_stunned(caster) { return false; }

    // 2. Cooldown ready. `next_ready_tick` is absolute; we compare to the
    //    current tick directly.
    let cooldown_ready_at = state.agent_cooldown_next_ready(caster).unwrap_or(0);
    if state.tick < cooldown_ready_at { return false; }

    // 3. Ability exists in the registry.
    let prog = match registry.get(ability) {
        Some(p) => p,
        None    => return false,
    };

    // 4. Target alive + in-range.
    if !state.agent_alive(target) { return false; }
    let caster_pos = match state.agent_pos(caster) { Some(p) => p, None => return false };
    let target_pos = match state.agent_pos(target) { Some(p) => p, None => return false };
    let dist = caster_pos.distance(target_pos);
    let range = match prog.area { Area::SingleTarget { range } => range };
    if dist > range { return false; }

    // 5. Hostile-only gate: require `CreatureType::is_hostile_to` to allow.
    if prog.gate.hostile_only {
        let ct = match state.agent_creature_type(caster) { Some(c) => c, None => return false };
        let tc = match state.agent_creature_type(target) { Some(c) => c, None => return false };
        if !ct.is_hostile_to(tc) { return false; }
    }

    // 6. Engagement lock. The caster can still cast at their current engager —
    //    that's the common "hit the thing you're fighting" case.
    if let Some(engager) = state.agent_engaged_with(caster) {
        if engager != target { return false; }
    }

    true
}
