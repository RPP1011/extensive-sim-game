use crate::event::{EventLike, EventRing};
use crate::state::SimState;

pub mod __sealed {
    pub trait Sealed {}
}

/// Stable ordinal identifying an event variant. Dense so it indexes arrays
/// cheaply. Adding a variant appends; reordering is a schema-hash bump.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum EventKindId {
    AgentMoved           = 0,
    AgentAttacked        = 1,
    AgentDied            = 2,
    AgentFled            = 3,
    AgentAte             = 4,
    AgentDrank           = 5,
    AgentRested          = 6,
    AgentCast            = 7,
    AgentUsedItem        = 8,
    AgentHarvested       = 9,
    AgentPlacedTile      = 10,
    AgentPlacedVoxel     = 11,
    AgentHarvestedVoxel  = 12,
    AgentConversed       = 13,
    AgentSharedStory     = 14,
    AgentCommunicated    = 15,
    InformationRequested = 16,
    AgentRemembered      = 17,
    QuestPosted          = 18,
    QuestAccepted        = 19,
    BidPlaced            = 20,
    AnnounceEmitted      = 21,
    RecordMemory         = 22,
    // Slots 23-24 retired in task 143 along with StunExpired / SlowExpired.
    // Stun / slow expiry is now a synthetic boundary read off
    // `stun_expires_at_tick` / `slow_expires_at_tick`, no event emitted.
    OpportunityAttackTriggered = 25,
    // Combat Foundation Task 9 — effect fan-out + recursion-audit events.
    EffectDamageApplied  = 26,
    EffectHealApplied    = 27,
    EffectShieldApplied  = 28,
    EffectStunApplied    = 29,
    EffectSlowApplied    = 30,
    EffectGoldTransfer   = 31,
    EffectStandingDelta  = 32,
    CastDepthExceeded    = 33,
    // Task 139 — engagement transition events replacing the retired
    // `tick_start` tentative-commit loop.
    EngagementCommitted  = 34,
    EngagementBroken     = 35,
    // Task 167 — fear-spread fan-out from `AgentDied`. One emit per
    // nearby same-species kin; folded by `kin_fear` materialized view.
    FearSpread           = 36,
    // Task 169 — pack-focus fan-out from `EngagementCommitted`. One
    // emit per nearby same-species kin; folded by `pack_focus`
    // materialized view.
    PackAssist           = 37,
    // Task 178 — rally fan-out from `AgentAttacked` on a wounded
    // (alive + hp_pct < 0.5) victim. One emit per nearby same-species
    // kin; folded by `rally_boost` materialized view.
    RallyCall            = 38,
    // Slots 39-127 reserved for replayable event variants added in later tasks.
    ChronicleEntry       = 128,
}


/// Lane discipline — handlers within a lane run in registration order;
/// lanes run in the order listed here.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum Lane {
    Validation = 0,
    Effect     = 1,
    Reaction   = 2,
    Audit      = 3,
}

impl Lane {
    pub const ALL: &'static [Lane] = &[
        Lane::Validation, Lane::Effect, Lane::Reaction, Lane::Audit,
    ];
}

pub trait CascadeHandler<E: EventLike>: __sealed::Sealed + Send + Sync {
    fn trigger(&self) -> EventKindId;
    fn lane(&self) -> Lane { Lane::Effect }
    fn handle(&self, event: &E, state: &mut SimState, events: &mut EventRing<E>);

    /// Downcast hook so registries can look up the concrete handler type
    /// (e.g. `CastHandler`) to expose handler-specific state
    /// (`AbilityRegistry`). Default impl returns `None`; concrete handlers
    /// that want to be discoverable override with `Some(self)`.
    fn as_any(&self) -> Option<&dyn std::any::Any> { None }
}
