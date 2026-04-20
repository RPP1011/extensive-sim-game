//! Unified tick-start phase. Three jobs in one pass over alive agents:
//!
//! 1. Decrement `hot_stun_remaining_ticks` and `hot_slow_remaining_ticks`,
//!    emitting `StunExpired` / `SlowExpired` on the tick each reaches zero.
//! 2. Update `hot_engaged_with` via tentative-pick-then-commit so the
//!    bidirectional invariant (`engaged_with[a] == Some(b) ⇔
//!    engaged_with[b] == Some(a)`) holds after the phase runs.
//! 3. `hot_cooldown_next_ready_tick` is an absolute tick — no decrement,
//!    just compared against `state.tick` by the mask predicate.
//!
//! Called by `step_full` BEFORE `scratch.mask.reset()` so mask predicates
//! see the post-decrement / post-engagement state.

use crate::cascade::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
use crate::ids::AgentId;
use crate::state::SimState;
use crate::step::{SimScratch, ATTACK_DAMAGE};

/// Engagement range in world-space meters. Matches `ATTACK_RANGE = 2.0` —
/// an agent is "engaged" with a hostile exactly when the hostile is within
/// melee strike distance.
pub const ENGAGEMENT_RANGE: f32 = 2.0;

/// Multiplier applied to `MoveToward` speed when an engaged agent moves
/// away from its engager (see Combat Foundation Task 4). Stored here
/// alongside `ENGAGEMENT_RANGE` so the schema-hash fingerprint can cover
/// both constants together.
pub const ENGAGEMENT_SLOW_FACTOR: f32 = 0.3;

/// Cascade handler for `OpportunityAttackTriggered`. Mirrors the normal
/// `MicroKind::Attack` damage path: applies `ATTACK_DAMAGE` to the target
/// and emits `AgentAttacked` + (on kill) `AgentDied` + `state.kill_agent`.
///
/// Registered by `CascadeRegistry::register_engine_builtins()` so it fires
/// automatically in step pipelines that use `CascadeRegistry::with_engine_builtins`.
/// Tests that want a pristine registry can opt out with `CascadeRegistry::new`.
pub struct OpportunityAttackHandler;

impl CascadeHandler for OpportunityAttackHandler {
    fn trigger(&self) -> EventKindId { EventKindId::OpportunityAttackTriggered }
    fn lane(&self) -> Lane { Lane::Effect }
    fn handle(&self, event: &Event, state: &mut SimState, events: &mut EventRing) {
        if let Event::OpportunityAttackTriggered { attacker, target, tick } = *event {
            if !state.agent_alive(target) { return; }
            // Audit fix MEDIUM #10: honour the attacker's per-agent damage.
            let damage = state.agent_attack_damage(attacker).unwrap_or(ATTACK_DAMAGE);
            let cur_hp = state.agent_hp(target).unwrap_or(0.0);
            let new_hp = (cur_hp - damage).max(0.0);
            state.set_agent_hp(target, new_hp);
            events.push(Event::AgentAttacked {
                attacker, target, damage, tick,
            });
            if new_hp <= 0.0 {
                events.push(Event::AgentDied { agent_id: target, tick });
                state.kill_agent(target);
            }
        }
    }
}

/// The unified tick-start phase. See module docs for the three jobs.
///
/// `scratch` is the per-tick buffer pool from `step_full`; the two engagement
/// scratches (`engagement_alive_ids`, `engagement_tentative`) are reused
/// across ticks instead of being heap-allocated each time.
pub fn tick_start(state: &mut SimState, scratch: &mut SimScratch, events: &mut EventRing) {
    decrement_and_expire(state, scratch, events);
    update_engagements(state, scratch);
}

fn decrement_and_expire(state: &mut SimState, scratch: &mut SimScratch, events: &mut EventRing) {
    // Snapshot the tick before we touch anything so all emitted events agree
    // on the "expired this tick" timestamp.
    let tick = state.tick;
    // Collect alive ids into the hoisted scratch so we don't risk borrow
    // conflicts with the mutating calls on `state` below — and don't
    // allocate a fresh Vec every tick.
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

fn update_engagements(state: &mut SimState, scratch: &mut SimScratch) {
    // Two-phase tentative-pick-then-commit. Each alive agent first picks the
    // nearest hostile within ENGAGEMENT_RANGE (ties broken by AgentId via
    // natural iteration order — `agents_alive()` walks slots in ascending
    // order). Then we commit only if the pick is mutual: i.e.
    // tentative[a] == Some(b) AND tentative[b] == Some(a).
    //
    // This catches the 3-agent counterexample where A picks B but B's
    // nearest hostile is C (not A) — without tentative-commit, a naive
    // "set both sides" loop would give `A.engaged=B, B.engaged=A,
    // C.engaged=B` and then overwrite with `B.engaged=C, C.engaged=B`,
    // leaving A asymmetric.
    //
    // Audit fix CRITICAL #1: consume the spatial index so the per-agent
    // hostile scan is `O(N·k)` instead of `O(N²)`.
    let cap = state.hot_engaged_with().len();
    // Hoisted tentative buffer — resized to cap and zeroed every tick.
    scratch.engagement_tentative.clear();
    scratch.engagement_tentative.resize(cap, None);
    // `engagement_alive_ids` was populated by `decrement_and_expire`
    // immediately above. Borrow it explicitly here so we can iterate
    // without freezing all of `scratch`.
    let alive = &scratch.engagement_alive_ids;
    let tentative = &mut scratch.engagement_tentative;
    let spatial = state.spatial();
    for id in alive {
        let pos = match state.agent_pos(*id) { Some(p) => p, None => continue };
        let ct = match state.agent_creature_type(*id) { Some(c) => c, None => continue };
        let mut best: Option<(AgentId, f32)> = None;
        for other in spatial.within_radius(state, pos, ENGAGEMENT_RANGE) {
            if other == *id { continue; }
            let op = match state.agent_pos(other) { Some(p) => p, None => continue };
            let oc = match state.agent_creature_type(other) { Some(c) => c, None => continue };
            if !ct.is_hostile_to(oc) { continue; }
            let d = pos.distance(op);
            // Tie-break: lower raw id wins when distances match, matching the
            // previous iteration-order-based behaviour.
            match best {
                None => best = Some((other, d)),
                Some((_, bd)) if d < bd => best = Some((other, d)),
                Some((b, bd)) if (d - bd).abs() < f32::EPSILON && other.raw() < b.raw() => {
                    best = Some((other, d));
                }
                _ => {}
            }
        }
        let slot = (id.raw() - 1) as usize;
        if slot < tentative.len() {
            tentative[slot] = best.map(|(a, _)| a);
        }
    }

    // Commit mutual-only. Re-borrow scratch buffers as immutable so
    // `state.set_agent_engaged_with` can mutate `state` simultaneously.
    let alive = &scratch.engagement_alive_ids;
    let tentative = &scratch.engagement_tentative;
    for &id in alive {
        let slot = (id.raw() - 1) as usize;
        let candidate = tentative.get(slot).copied().unwrap_or(None);
        let committed = match candidate {
            Some(other) => {
                let other_slot = (other.raw() - 1) as usize;
                let them = tentative.get(other_slot).copied().unwrap_or(None);
                if them == Some(id) { Some(other) } else { None }
            }
            None => None,
        };
        state.set_agent_engaged_with(id, committed);
    }
}
