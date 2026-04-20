pub mod ring;
pub use crate::ids::EventId;
pub use ring::EventRing;

use crate::ability::AbilityId;
use crate::ids::{AgentId, QuestId};
use crate::policy::macro_kind::{QuestCategory, Resolution};
use glam::Vec3;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Event {
    // Replayable subset
    AgentMoved    { agent_id: AgentId, from: Vec3, to: Vec3, tick: u32 },
    AgentAttacked { attacker: AgentId, target: AgentId, damage: f32, tick: u32 },
    AgentDied     { agent_id: AgentId, tick: u32 },
    AgentFled     { agent_id: AgentId, from: Vec3, to: Vec3, tick: u32 },
    AgentAte      { agent_id: AgentId, delta: f32, tick: u32 },
    AgentDrank    { agent_id: AgentId, delta: f32, tick: u32 },
    AgentRested   { agent_id: AgentId, delta: f32, tick: u32 },
    // Cast dispatch (Combat Foundation Task 9). Emitted by `apply_actions`
    // when an agent issues `MicroKind::Cast` with `MicroTarget::Ability`.
    // `CastHandler` picks it up via cascade dispatch, looks the program up
    // in its `AbilityRegistry`, and emits one `Effect*Applied` per op.
    AgentCast           { caster: AgentId, ability: AbilityId, target: AgentId, tick: u32 },
    AgentUsedItem       { agent_id: AgentId, item_slot: u8, tick: u32 },
    AgentHarvested      { agent_id: AgentId, resource: u64, tick: u32 },
    AgentPlacedTile     { agent_id: AgentId, where_pos: Vec3, kind_tag: u32, tick: u32 },
    AgentPlacedVoxel    { agent_id: AgentId, where_pos: Vec3, mat_tag: u32,  tick: u32 },
    AgentHarvestedVoxel { agent_id: AgentId, where_pos: Vec3, tick: u32 },
    AgentConversed      { agent_id: AgentId, partner: AgentId, tick: u32 },
    AgentSharedStory    { agent_id: AgentId, topic: u64, tick: u32 },
    AgentCommunicated   { speaker: AgentId, recipient: AgentId, fact_ref: u64, tick: u32 },
    InformationRequested{ asker: AgentId, target: AgentId, query: u64, tick: u32 },
    AgentRemembered     { agent_id: AgentId, subject: u64, tick: u32 },
    // Event-only macros — emitted when a policy emits the corresponding
    // `MacroAction`. Domain handlers (registered later) do the actual effect.
    QuestPosted   { poster: AgentId, quest_id: QuestId, category: QuestCategory, resolution: Resolution, tick: u32 },
    QuestAccepted { acceptor: AgentId, quest_id: QuestId, tick: u32 },
    BidPlaced     { bidder: AgentId, auction_id: QuestId, amount: f32, tick: u32 },
    // Announce fan-out. `AnnounceEmitted` is a single event emitted per speaker;
    // `RecordMemory` is emitted once per recipient within the audience radius.
    // `audience_tag` matches `AnnounceAudience::tag()`: 0=Group, 1=Area, 2=Anyone.
    AnnounceEmitted { speaker: AgentId, audience_tag: u8, fact_payload: u64, tick: u32 },
    RecordMemory    { observer: AgentId, source: AgentId, fact_payload: u64, confidence: f32, tick: u32 },
    // Combat Foundation Task 3 — stun/slow expiry. Emitted by
    // `ability::expire::tick_start` on the tick a timer reaches zero.
    StunExpired { agent_id: AgentId, tick: u32 },
    SlowExpired { agent_id: AgentId, tick: u32 },
    // Combat Foundation Task 4 — opportunity attack fired by the engager
    // when an engaged agent tries to move away or flee. Applied via a
    // cascade handler (registered by the engine defaults) that reuses
    // ATTACK_DAMAGE and drops the target's hp accordingly.
    OpportunityAttackTriggered { attacker: AgentId, target: AgentId, tick: u32 },
    // Combat Foundation Tasks 9–17 — effect fan-out events emitted by
    // `CastHandler`, one per `EffectOp` carried by the cast's program.
    // Per-effect cascade handlers (registered by `register_engine_builtins`
    // in Tasks 10–17) consume these to apply the concrete state mutation.
    // Task 9 emits them; Tasks 10–17 add their handlers.
    EffectDamageApplied  { caster: AgentId, target: AgentId, amount: f32, tick: u32 },
    EffectHealApplied    { caster: AgentId, target: AgentId, amount: f32, tick: u32 },
    EffectShieldApplied  { caster: AgentId, target: AgentId, amount: f32, tick: u32 },
    EffectStunApplied    { caster: AgentId, target: AgentId, duration_ticks: u32, tick: u32 },
    EffectSlowApplied    { caster: AgentId, target: AgentId, duration_ticks: u32, factor_q8: i16, tick: u32 },
    EffectGoldTransfer   { from: AgentId, to: AgentId, amount: i64, tick: u32 },
    EffectStandingDelta  { a: AgentId, b: AgentId, delta: i16, tick: u32 },
    // Audit event emitted by `CastHandler` when a recursive `CastAbility`
    // effect would push beyond `MAX_CASCADE_ITERATIONS`. Task 18's depth
    // test pins the exact budget (8 iterations); the handler drops the
    // overflow cast silently to avoid destabilising the cascade loop.
    CastDepthExceeded    { caster: AgentId, ability: AbilityId, tick: u32 },
    // Non-replayable (chronicle / prose side-channel placeholder)
    ChronicleEntry { tick: u32, template_id: u32 },
}

impl Event {
    pub fn tick(&self) -> u32 {
        match self {
            Event::AgentMoved           { tick, .. } |
            Event::AgentAttacked        { tick, .. } |
            Event::AgentDied            { tick, .. } |
            Event::AgentFled            { tick, .. } |
            Event::AgentAte             { tick, .. } |
            Event::AgentDrank           { tick, .. } |
            Event::AgentRested          { tick, .. } |
            Event::AgentCast            { tick, .. } |
            Event::AgentUsedItem        { tick, .. } |
            Event::AgentHarvested       { tick, .. } |
            Event::AgentPlacedTile      { tick, .. } |
            Event::AgentPlacedVoxel     { tick, .. } |
            Event::AgentHarvestedVoxel  { tick, .. } |
            Event::AgentConversed       { tick, .. } |
            Event::AgentSharedStory     { tick, .. } |
            Event::AgentCommunicated    { tick, .. } |
            Event::InformationRequested { tick, .. } |
            Event::AgentRemembered      { tick, .. } |
            Event::QuestPosted          { tick, .. } |
            Event::QuestAccepted        { tick, .. } |
            Event::BidPlaced            { tick, .. } |
            Event::AnnounceEmitted      { tick, .. } |
            Event::RecordMemory         { tick, .. } |
            Event::StunExpired          { tick, .. } |
            Event::SlowExpired          { tick, .. } |
            Event::OpportunityAttackTriggered { tick, .. } |
            Event::EffectDamageApplied  { tick, .. } |
            Event::EffectHealApplied    { tick, .. } |
            Event::EffectShieldApplied  { tick, .. } |
            Event::EffectStunApplied    { tick, .. } |
            Event::EffectSlowApplied    { tick, .. } |
            Event::EffectGoldTransfer   { tick, .. } |
            Event::EffectStandingDelta  { tick, .. } |
            Event::CastDepthExceeded    { tick, .. } |
            Event::ChronicleEntry       { tick, .. } => *tick,
        }
    }
    pub fn is_replayable(&self) -> bool {
        !matches!(self, Event::ChronicleEntry { .. })
    }
}
