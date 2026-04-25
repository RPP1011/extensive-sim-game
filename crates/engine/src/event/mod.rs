//! Event vocabulary primitives — the event TYPE itself is emitted into
//! engine_data; engine declares only the trait that any event enum must
//! satisfy. See Spec B' D13.
//!
//! The `EventRing` buffer is an engine-side primitive (it's a ring of
//! events, not a vocabulary) so it stays here.
//!
//! TRANSITION (Task 1): engine still depends on engine_data via chronicle.rs
//! (deferred to Plan B2). This lets us write the bridge `impl EventLike for
//! engine_data::events::Event` here. Once Plan B2 migrates chronicle.rs, the
//! impl moves to engine_data/src/events/mod.rs and this bridge is removed.

pub mod ring;
pub use ring::EventRing;
pub use crate::ids::EventId;

// TRANSITION re-export: engine still depends on engine_data (via chronicle.rs,
// deferred to Plan B2). Generated/ files use `crate::event::Event` — this
// re-export preserves backward compatibility for those files until Task 3
// moves them to engine_rules. Drop this line in Task 3 / Plan B2.
pub use engine_data::events::Event;

/// Trait every concrete event enum implements. Engine's runtime primitives
/// (`EventRing`, `CascadeRegistry`, `CascadeHandler::handle`, view fold sites)
/// are generic over `E: EventLike`. The compiler emits `impl EventLike for
/// engine_data::Event { ... }` so the kind ordinal stays consistent across
/// regenerations.
pub trait EventLike: Sized + Clone + Send + Sync + 'static {
    fn kind(&self) -> crate::cascade::EventKindId;
    fn tick(&self) -> u32;
    fn is_replayable(&self) -> bool;
    /// Write the replayable bytes of this event into the given hasher.
    /// Called by `EventRing::replayable_sha256`. Only called when
    /// `is_replayable()` returns true. Sidecar fields (`id`, `cause`)
    /// must NOT be included.
    fn hash_replayable(&self, h: &mut sha2::Sha256);
}

// Bridge impl: engine owns the trait, engine sees engine_data via regular dep
// (retained for chronicle.rs). Orphan rule permits this. Moves to engine_data
// in Plan B2.
impl EventLike for engine_data::events::Event {
    fn kind(&self) -> crate::cascade::EventKindId {
        use crate::cascade::EventKindId;
        use engine_data::events::Event;
        match self {
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
            Event::RallyCall            { .. } => EventKindId::RallyCall,
            Event::ChronicleEntry       { .. } => EventKindId::ChronicleEntry,
        }
    }

    fn tick(&self) -> u32 { engine_data::events::Event::tick(self) }

    fn is_replayable(&self) -> bool { engine_data::events::Event::is_replayable(self) }

    fn hash_replayable(&self, h: &mut sha2::Sha256) {
        use sha2::Digest;
        use engine_data::events::Event;
        match self {
            Event::AgentMoved { actor, from, location, tick } => {
                h.update([0u8]);
                h.update(actor.raw().to_le_bytes());
                for v in [from.x, from.y, from.z, location.x, location.y, location.z] {
                    h.update(v.to_bits().to_le_bytes());
                }
                h.update(tick.to_le_bytes());
            }
            Event::AgentAttacked { actor, target, damage, tick } => {
                h.update([1u8]);
                h.update(actor.raw().to_le_bytes());
                h.update(target.raw().to_le_bytes());
                h.update(damage.to_bits().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::AgentDied { agent_id, tick } => {
                h.update([2u8]);
                h.update(agent_id.raw().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::AgentFled { agent_id, from, to, tick } => {
                h.update([3u8]);
                h.update(agent_id.raw().to_le_bytes());
                for v in [from.x, from.y, from.z, to.x, to.y, to.z] {
                    h.update(v.to_bits().to_le_bytes());
                }
                h.update(tick.to_le_bytes());
            }
            Event::AgentAte { agent_id, delta, tick } => {
                h.update([4u8]);
                h.update(agent_id.raw().to_le_bytes());
                h.update(delta.to_bits().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::AgentDrank { agent_id, delta, tick } => {
                h.update([5u8]);
                h.update(agent_id.raw().to_le_bytes());
                h.update(delta.to_bits().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::AgentRested { agent_id, delta, tick } => {
                h.update([6u8]);
                h.update(agent_id.raw().to_le_bytes());
                h.update(delta.to_bits().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::AgentCast { actor, ability, target, depth, tick } => {
                h.update([7u8]);
                h.update(actor.raw().to_le_bytes());
                h.update(ability.raw().to_le_bytes());
                h.update(target.raw().to_le_bytes());
                h.update([*depth]);
                h.update(tick.to_le_bytes());
            }
            Event::AgentUsedItem { agent_id, item_slot, tick } => {
                h.update([8u8]);
                h.update(agent_id.raw().to_le_bytes());
                h.update([*item_slot]);
                h.update(tick.to_le_bytes());
            }
            Event::AgentHarvested { agent_id, resource, tick } => {
                h.update([9u8]);
                h.update(agent_id.raw().to_le_bytes());
                h.update(resource.to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::AgentPlacedTile { actor, location, kind_tag, tick } => {
                h.update([10u8]);
                h.update(actor.raw().to_le_bytes());
                for v in [location.x, location.y, location.z] {
                    h.update(v.to_bits().to_le_bytes());
                }
                h.update(kind_tag.to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::AgentPlacedVoxel { actor, location, mat_tag, tick } => {
                h.update([11u8]);
                h.update(actor.raw().to_le_bytes());
                for v in [location.x, location.y, location.z] {
                    h.update(v.to_bits().to_le_bytes());
                }
                h.update(mat_tag.to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::AgentHarvestedVoxel { actor, location, tick } => {
                h.update([12u8]);
                h.update(actor.raw().to_le_bytes());
                for v in [location.x, location.y, location.z] {
                    h.update(v.to_bits().to_le_bytes());
                }
                h.update(tick.to_le_bytes());
            }
            Event::AgentConversed { agent_id, partner, tick } => {
                h.update([13u8]);
                h.update(agent_id.raw().to_le_bytes());
                h.update(partner.raw().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::AgentSharedStory { agent_id, topic, tick } => {
                h.update([14u8]);
                h.update(agent_id.raw().to_le_bytes());
                h.update(topic.to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::AgentCommunicated { speaker, recipient, fact_ref, tick } => {
                h.update([15u8]);
                h.update(speaker.raw().to_le_bytes());
                h.update(recipient.raw().to_le_bytes());
                h.update(fact_ref.to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::InformationRequested { asker, target, query, tick } => {
                h.update([16u8]);
                h.update(asker.raw().to_le_bytes());
                h.update(target.raw().to_le_bytes());
                h.update(query.to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::AgentRemembered { agent_id, subject, tick } => {
                h.update([17u8]);
                h.update(agent_id.raw().to_le_bytes());
                h.update(subject.to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::QuestPosted { poster, quest_id, category, resolution, tick } => {
                h.update([18u8]);
                h.update(poster.raw().to_le_bytes());
                h.update(quest_id.raw().to_le_bytes());
                h.update([*category as u8]);
                let (res_tag, min_parties) = match resolution {
                    engine_data::types::Resolution::HighestBid        => (0u8, 0u8),
                    engine_data::types::Resolution::FirstAcceptable   => (1u8, 0u8),
                    engine_data::types::Resolution::MutualAgreement   => (2u8, 0u8),
                    engine_data::types::Resolution::Coalition { min_parties } => (3u8, *min_parties),
                    engine_data::types::Resolution::Majority          => (4u8, 0u8),
                };
                h.update([res_tag, min_parties]);
                h.update(tick.to_le_bytes());
            }
            Event::QuestAccepted { acceptor, quest_id, tick } => {
                h.update([19u8]);
                h.update(acceptor.raw().to_le_bytes());
                h.update(quest_id.raw().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::BidPlaced { bidder, auction_id, amount, tick } => {
                h.update([20u8]);
                h.update(bidder.raw().to_le_bytes());
                h.update(auction_id.raw().to_le_bytes());
                h.update(amount.to_bits().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::AnnounceEmitted { speaker, audience_tag, fact_payload, tick } => {
                h.update([21u8]);
                h.update(speaker.raw().to_le_bytes());
                h.update([*audience_tag]);
                h.update(fact_payload.to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::RecordMemory { observer, source, fact_payload, confidence, tick } => {
                h.update([22u8]);
                h.update(observer.raw().to_le_bytes());
                h.update(source.raw().to_le_bytes());
                h.update(fact_payload.to_le_bytes());
                h.update(confidence.to_bits().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::OpportunityAttackTriggered { actor, target, tick } => {
                h.update([25u8]);
                h.update(actor.raw().to_le_bytes());
                h.update(target.raw().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::EffectDamageApplied { actor, target, amount, tick } => {
                h.update([26u8]);
                h.update(actor.raw().to_le_bytes());
                h.update(target.raw().to_le_bytes());
                h.update(amount.to_bits().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::EffectHealApplied { actor, target, amount, tick } => {
                h.update([27u8]);
                h.update(actor.raw().to_le_bytes());
                h.update(target.raw().to_le_bytes());
                h.update(amount.to_bits().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::EffectShieldApplied { actor, target, amount, tick } => {
                h.update([28u8]);
                h.update(actor.raw().to_le_bytes());
                h.update(target.raw().to_le_bytes());
                h.update(amount.to_bits().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::EffectStunApplied { actor, target, expires_at_tick, tick } => {
                h.update([29u8]);
                h.update(actor.raw().to_le_bytes());
                h.update(target.raw().to_le_bytes());
                h.update(expires_at_tick.to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::EffectSlowApplied { actor, target, expires_at_tick, factor_q8, tick } => {
                h.update([30u8]);
                h.update(actor.raw().to_le_bytes());
                h.update(target.raw().to_le_bytes());
                h.update(expires_at_tick.to_le_bytes());
                h.update(factor_q8.to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::EffectGoldTransfer { from, to, amount, tick } => {
                h.update([31u8]);
                h.update(from.raw().to_le_bytes());
                h.update(to.raw().to_le_bytes());
                h.update(amount.to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::EffectStandingDelta { a, b, delta, tick } => {
                h.update([32u8]);
                h.update(a.raw().to_le_bytes());
                h.update(b.raw().to_le_bytes());
                h.update(delta.to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::CastDepthExceeded { actor, ability, tick } => {
                h.update([33u8]);
                h.update(actor.raw().to_le_bytes());
                h.update(ability.raw().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::EngagementCommitted { actor, target, tick } => {
                h.update([34u8]);
                h.update(actor.raw().to_le_bytes());
                h.update(target.raw().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::EngagementBroken { actor, former_target, reason, tick } => {
                h.update([35u8]);
                h.update(actor.raw().to_le_bytes());
                h.update(former_target.raw().to_le_bytes());
                h.update([*reason]);
                h.update(tick.to_le_bytes());
            }
            Event::FearSpread { observer, dead_kin, tick } => {
                h.update([36u8]);
                h.update(observer.raw().to_le_bytes());
                h.update(dead_kin.raw().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::PackAssist { observer, target, tick } => {
                h.update([37u8]);
                h.update(observer.raw().to_le_bytes());
                h.update(target.raw().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::RallyCall { observer, wounded_kin, tick } => {
                h.update([38u8]);
                h.update(observer.raw().to_le_bytes());
                h.update(wounded_kin.raw().to_le_bytes());
                h.update(tick.to_le_bytes());
            }
            Event::ChronicleEntry { .. } => {
                // Filtered at the call site; if we reach here, the filter is broken.
                debug_assert!(false, "ChronicleEntry reached replayable hash path");
            }
        }
    }
}
