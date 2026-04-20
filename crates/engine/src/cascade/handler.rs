use crate::event::{Event, EventRing};
use crate::state::SimState;

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
    // Slots 38-127 reserved for replayable event variants added in later tasks.
    ChronicleEntry       = 128,
}

impl EventKindId {
    pub fn from_event(e: &Event) -> EventKindId {
        match e {
            Event::AgentMoved           { .. } => EventKindId::AgentMoved,
            Event::AgentAttacked        { .. } => EventKindId::AgentAttacked,
            Event::AgentDied            { .. } => EventKindId::AgentDied,
            Event::AgentFled            { .. } => EventKindId::AgentFled,
            Event::AgentAte             { .. } => EventKindId::AgentAte,
            Event::AgentDrank           { .. } => EventKindId::AgentDrank,
            Event::AgentRested          { .. } => EventKindId::AgentRested,
            Event::AgentCast            { .. } => EventKindId::AgentCast,
            Event::AgentUsedItem        { .. } => EventKindId::AgentUsedItem,
            Event::AgentHarvested       { .. } => EventKindId::AgentHarvested,
            Event::AgentPlacedTile      { .. } => EventKindId::AgentPlacedTile,
            Event::AgentPlacedVoxel     { .. } => EventKindId::AgentPlacedVoxel,
            Event::AgentHarvestedVoxel  { .. } => EventKindId::AgentHarvestedVoxel,
            Event::AgentConversed       { .. } => EventKindId::AgentConversed,
            Event::AgentSharedStory     { .. } => EventKindId::AgentSharedStory,
            Event::AgentCommunicated    { .. } => EventKindId::AgentCommunicated,
            Event::InformationRequested { .. } => EventKindId::InformationRequested,
            Event::AgentRemembered      { .. } => EventKindId::AgentRemembered,
            Event::QuestPosted          { .. } => EventKindId::QuestPosted,
            Event::QuestAccepted        { .. } => EventKindId::QuestAccepted,
            Event::BidPlaced            { .. } => EventKindId::BidPlaced,
            Event::AnnounceEmitted      { .. } => EventKindId::AnnounceEmitted,
            Event::RecordMemory         { .. } => EventKindId::RecordMemory,
            Event::OpportunityAttackTriggered { .. } => EventKindId::OpportunityAttackTriggered,
            Event::EffectDamageApplied  { .. } => EventKindId::EffectDamageApplied,
            Event::EffectHealApplied    { .. } => EventKindId::EffectHealApplied,
            Event::EffectShieldApplied  { .. } => EventKindId::EffectShieldApplied,
            Event::EffectStunApplied    { .. } => EventKindId::EffectStunApplied,
            Event::EffectSlowApplied    { .. } => EventKindId::EffectSlowApplied,
            Event::EffectGoldTransfer   { .. } => EventKindId::EffectGoldTransfer,
            Event::EffectStandingDelta  { .. } => EventKindId::EffectStandingDelta,
            Event::CastDepthExceeded    { .. } => EventKindId::CastDepthExceeded,
            Event::EngagementCommitted  { .. } => EventKindId::EngagementCommitted,
            Event::EngagementBroken     { .. } => EventKindId::EngagementBroken,
            Event::FearSpread           { .. } => EventKindId::FearSpread,
            Event::PackAssist           { .. } => EventKindId::PackAssist,
            Event::ChronicleEntry       { .. } => EventKindId::ChronicleEntry,
        }
    }
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

pub trait CascadeHandler: Send + Sync {
    fn trigger(&self) -> EventKindId;
    fn lane(&self) -> Lane { Lane::Effect }
    fn handle(&self, event: &Event, state: &mut SimState, events: &mut EventRing);

    /// Downcast hook so registries can look up the concrete handler type
    /// (e.g. `CastHandler`) to expose handler-specific state
    /// (`AbilityRegistry`). Default impl returns `None`; concrete handlers
    /// that want to be discoverable override with `Some(self)`.
    fn as_any(&self) -> Option<&dyn std::any::Any> { None }
}
