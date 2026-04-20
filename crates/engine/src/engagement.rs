//! Event-driven engagement update — task 139.
//!
//! Retired `ability::expire::tick_start`'s tentative-commit engagement
//! pass in favour of two cascade handlers:
//!
//! - `engagement_on_move` fires on `AgentMoved`. It recomputes the mover's
//!   nearest-hostile pick and, if the pick changed, emits an
//!   `EngagementBroken` (for the stale partner, if any) followed by an
//!   `EngagementCommitted` (for the new pick, if any). The compiler-emitted
//!   `@materialized view engaged_with` folds those events into its per-agent
//!   slot map.
//!
//! - `engagement_on_death` fires on `AgentDied`. If the dead agent had a
//!   partner, it emits `EngagementBroken` for that pair so the view drops
//!   both slots in one shot.
//!
//! Simplification (picked with the task's blessing): the new pipeline only
//! ever recomputes engagement on the mover's side. The old two-pass
//! tentative-commit enforced "A.engaged=B ⇔ B.engaged=A" even when the
//! third-party's nearest hostile was someone else (see
//! `engagement_tick_start::three_agent_tentative_commit_with_dragon_closer_to_wolf`);
//! the event-driven path drops that enforcement and lets slot iteration
//! order decide who "wins" when three agents race. See
//! `engagement_tick_start::three_agent_unilateral_commit_pins_closest_pair`
//! for the pinned behaviour.
//!
//! The bidirectional invariant (`engaged(a) = Some(b) ⇒ engaged(b) =
//! Some(a)`) is preserved at all times by `recompute_engagement_for`:
//! commit writes both sides in lockstep, and when the incoming mover
//! displaces the new partner's prior pairing (three-agent case) that
//! prior pairing is broken on both sides too. Pinned by
//! `proptest_engagement::engagement_is_bidirectional`.
//!
//! Both handlers write the view slot **eagerly** (via the direct `.set()`
//! setter) in addition to emitting audit events. The eager write gives
//! same-tick read-your-own-writes for later handlers in the cascade; the
//! event emission drives the post-cascade fold so replays reconstruct the
//! view state from the event log alone.

use crate::event::{Event, EventRing};
use crate::ids::AgentId;
use crate::state::SimState;

/// Reason codes carried on `EngagementBroken` so replayers can tell why
/// a pairing ended without re-running the physics. Kept as a `u8` on
/// the event so the DSL schema stays trivially hashable.
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

/// Recompute engagement for `mover`. Called from the cascade handler on
/// `AgentMoved`; also reused by `recompute_all_engagements` (the back-compat
/// shim that the retired `tick_start` tests still lean on).
///
/// Scans hostiles within `config.combat.engagement_range` via the spatial
/// index, picks the closest with ties broken on raw id, and emits
/// `EngagementBroken` / `EngagementCommitted` as needed. Writes the view
/// eagerly so same-tick readers observe the new pairing.
pub fn recompute_engagement_for(
    state: &mut SimState,
    mover: AgentId,
    events: &mut EventRing,
) {
    if !state.agent_alive(mover) {
        // Dead mover — `AgentDied` handler owns the cleanup path. Skip.
        return;
    }
    let pos = match state.agent_pos(mover) {
        Some(p) => p,
        None => return,
    };
    let ct = match state.agent_creature_type(mover) {
        Some(c) => c,
        None => return,
    };
    let radius = state.config.combat.engagement_range;
    // Scan for nearest hostile. The spatial query returns candidates in
    // slot order, so ties at identical distances resolve on raw id.
    let spatial = state.spatial();
    let mut best: Option<(AgentId, f32)> = None;
    for other in spatial.within_radius(state, pos, radius) {
        if other == mover {
            continue;
        }
        let op = match state.agent_pos(other) {
            Some(p) => p,
            None => continue,
        };
        let oc = match state.agent_creature_type(other) {
            Some(c) => c,
            None => continue,
        };
        if !ct.is_hostile_to(oc) {
            continue;
        }
        let d = pos.distance(op);
        match best {
            None => best = Some((other, d)),
            Some((_, bd)) if d < bd => best = Some((other, d)),
            Some((b, bd)) if (d - bd).abs() < f32::EPSILON && other.raw() < b.raw() => {
                best = Some((other, d));
            }
            _ => {}
        }
    }
    let new_partner = best.map(|(a, _)| a);
    let old_partner = state.agent_engaged_with(mover);
    if old_partner == new_partner {
        return;
    }
    let tick = state.tick;
    // Break first, commit second. The view's fold_event treats these as
    // atomic state transitions; the two-step emit is for replay.
    if let Some(former) = old_partner {
        // Remove the mover's slot eagerly so the later commit's insert
        // doesn't race with a stale partner read.
        state.set_agent_engaged_with(mover, None);
        state.set_agent_engaged_with(former, None);
        let reason = if new_partner.is_some() {
            break_reason::SWITCH
        } else {
            break_reason::OUT_OF_RANGE
        };
        events.push(Event::EngagementBroken {
            actor: mover,
            former_target: former,
            reason,
            tick,
        });
    }
    if let Some(partner) = new_partner {
        // Before overwriting `partner`'s slot, check if `partner` was
        // previously paired with a third-party `stranded`. If so, break
        // that pair properly — emit the `EngagementBroken` event and
        // clear both sides — so `stranded`'s slot doesn't linger
        // pointing at `partner` after we overwrite `partner`'s slot with
        // `mover`. Without this step the bidirectional invariant
        // (`engaged(a)=Some(b) ⇒ engaged(b)=Some(a)`) breaks in the
        // three-agent case where A picks B but B was already paired
        // with C: B's slot gets rewritten to A, but C's slot keeps
        // pointing at B (stale). Pinned by `proptest_engagement::
        // engagement_is_bidirectional`.
        if let Some(stranded) = state.agent_engaged_with(partner) {
            if stranded != mover {
                state.set_agent_engaged_with(partner, None);
                state.set_agent_engaged_with(stranded, None);
                events.push(Event::EngagementBroken {
                    actor: partner,
                    former_target: stranded,
                    reason: break_reason::SWITCH,
                    tick,
                });
            }
        }
        // Eagerly commit the pairing. The view fold re-applies this on
        // the next view-fold phase; the double-insert is idempotent.
        state.set_agent_engaged_with(mover, Some(partner));
        state.set_agent_engaged_with(partner, Some(mover));
        events.push(Event::EngagementCommitted {
            actor: mover,
            target: partner,
            tick,
        });
    }
}

/// Tear down the dead agent's engagement, if any. The survivor's slot
/// clears too so their next move-triggered recompute starts fresh. Called
/// from the cascade handler on `AgentDied`.
pub fn break_engagement_on_death(
    state: &mut SimState,
    dead: AgentId,
    events: &mut EventRing,
) {
    let partner = match state.agent_engaged_with(dead) {
        Some(p) => p,
        None => return,
    };
    // Eager write: clear both slots before the view fold catches up.
    state.set_agent_engaged_with(dead, None);
    state.set_agent_engaged_with(partner, None);
    events.push(Event::EngagementBroken {
        actor: partner,
        former_target: dead,
        reason: break_reason::PARTNER_DIED,
        tick: state.tick,
    });
}

/// Back-compat shim for the retired `tick_start` engagement pass. Walks
/// every alive agent in slot order and runs `recompute_engagement_for`
/// on each. Used by tests and fixtures that predated the event-driven
/// pipeline — new code should let the cascade handler do the work.
///
/// Emits `EngagementCommitted` / `EngagementBroken` events for every
/// change, same as the per-agent handler. Writes are eager (via the
/// shared `recompute_engagement_for`), so callers that compare the
/// `hot_engaged_with()` slice after this runs observe the committed
/// state without needing to trigger the view-fold phase.
pub fn recompute_all_engagements(state: &mut SimState, events: &mut EventRing) {
    // Snapshot alive ids up front so we don't re-scan a list that's
    // mutating underneath us (dead rows don't come back mid-pass here,
    // but emitting into `events` is still a mutable borrow that races
    // with `agents_alive()`).
    let alive: Vec<AgentId> = state.agents_alive().collect();
    for id in alive {
        recompute_engagement_for(state, id, events);
    }
}

// ---------------------------------------------------------------------------
// Cascade dispatchers — installed on `CascadeRegistry` by
// `register_engine_builtins`.
// ---------------------------------------------------------------------------

/// Dispatcher installed for `EventKindId::AgentMoved`. Destructures the
/// event once and forwards `actor` to `recompute_engagement_for`. This
/// dispatcher slot is the only one for `AgentMoved`; the compiler-emitted
/// physics module currently installs no `AgentMoved` handler, so there is
/// nothing to chain — if that changes, the DSL-side register() would be
/// the first to run and get overwritten here, and this dispatcher would
/// need to chain forward to the previous slot (same pattern as
/// `dispatch_agent_died`).
pub fn dispatch_agent_moved(event: &Event, state: &mut SimState, events: &mut EventRing) {
    if let Event::AgentMoved { actor, .. } = *event {
        recompute_engagement_for(state, actor, events);
    }
}

/// Dispatcher installed for `EventKindId::AgentDied`. Destructures the
/// event once and forwards to `break_engagement_on_death`.
///
/// The DSL-emitted `chronicle_death` physics rule also needs to fire on
/// `AgentDied` (see `assets/sim/physics.sim`); because the registry's
/// `install_kind` overwrites the slot, we invoke it explicitly here
/// instead of relying on slot composition. The dispatcher shape is
/// fixed (`fn(&Event, &mut SimState, &mut EventRing)`), so this is just
/// a plain function call — no trait-object indirection.
pub fn dispatch_agent_died(event: &Event, state: &mut SimState, events: &mut EventRing) {
    // DSL-side chronicle emission runs before engagement teardown so the
    // chronicle line lands in the ring at the tick's death event
    // position rather than after the engagement_broken cascade.
    crate::generated::physics::dispatch_agent_died(event, state, events);
    if let Event::AgentDied { agent_id, .. } = *event {
        break_engagement_on_death(state, agent_id, events);
    }
}
