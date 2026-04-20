use super::Event;
use crate::ids::EventId;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;

/// An entry in the `EventRing`. The `event` is the replay payload; `id` and
/// `cause` form the sidecar metadata (never folded into `replayable_sha256`).
struct Entry {
    event: Event,
    id:    EventId,
    cause: Option<EventId>,
}

pub struct EventRing {
    entries:      VecDeque<Entry>,
    cap:          usize,
    current_tick: u32,
    next_seq:     u32,
    total_pushed: usize,
    /// Monotonic cursor: the index of the next event to dispatch. Updated by
    /// `CascadeRegistry::run_fixed_point` via `set_dispatched(..)`. Survives
    /// ring eviction because it's an index into `total_pushed`, not into
    /// `entries`.
    dispatched:   usize,
}

impl EventRing {
    pub fn with_cap(cap: usize) -> Self {
        debug_assert!(cap > 0, "EventRing capacity must be nonzero");
        Self {
            entries:      VecDeque::with_capacity(cap),
            cap,
            current_tick: 0,
            next_seq:     0,
            total_pushed: 0,
            dispatched:   0,
        }
    }

    /// Push a root-cause event. Returns the assigned `EventId`.
    pub fn push(&mut self, e: Event) -> EventId {
        self.push_impl(e, None)
    }

    /// Push an event caused by an earlier event. The `cause` pointer lives in
    /// the sidecar — it does NOT affect `replayable_sha256`.
    pub fn push_caused_by(&mut self, e: Event, cause: EventId) -> EventId {
        self.push_impl(e, Some(cause))
    }

    fn push_impl(&mut self, e: Event, cause: Option<EventId>) -> EventId {
        let tick = e.tick();
        if tick != self.current_tick {
            self.current_tick = tick;
            self.next_seq = 0;
        }
        if self.entries.len() == self.cap {
            self.entries.pop_front();
        }
        let id = EventId { tick, seq: self.next_seq };
        self.next_seq += 1;
        self.total_pushed += 1;
        self.entries.push_back(Entry { event: e, id, cause });
        id
    }

    /// Total number of pushes ever — even those evicted from the ring.
    pub fn total_pushed(&self) -> usize { self.total_pushed }

    /// The index of the next undispatched event. Used by the cascade dispatcher
    /// to resume scanning across multiple `run_fixed_point` calls.
    pub fn dispatched(&self) -> usize { self.dispatched }

    /// Advance the dispatched-cursor. Clamped to `total_pushed`.
    pub fn set_dispatched(&mut self, idx: usize) {
        self.dispatched = idx.min(self.total_pushed);
    }

    /// Look up an event by its monotonic push index. Returns None if the event
    /// has been evicted or if the index is out of range.
    pub fn get_pushed(&self, idx: usize) -> Option<Event> {
        let first = self.total_pushed.saturating_sub(self.entries.len());
        if idx < first { return None; }
        let local = idx - first;
        self.entries.get(local).map(|e| e.event)
    }

    /// Look up the parent event id for a given event id, if any.
    pub fn cause_of(&self, id: EventId) -> Option<EventId> {
        self.entries.iter().find(|e| e.id == id).and_then(|e| e.cause)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Event> { self.entries.iter().map(|e| &e.event) }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Stable hash over the replayable subset. Uses explicit byte-packing
    /// (via `f32::to_bits`) so the digest is stable across Rust/glam versions
    /// and is insensitive to Debug format drift. Schema-hash-load-bearing:
    /// changing the pack format or variant-tag order requires a schema-hash bump.
    ///
    /// Sidecar fields (`id`, `cause`) are INTENTIONALLY excluded from the hash
    /// so cascade fan-out annotations cannot alter replay equivalence.
    pub fn replayable_sha256(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        for entry in self.entries.iter().filter(|e| e.event.is_replayable()) {
            hash_event(&mut h, &entry.event);
        }
        h.finalize().into()
    }
}

fn hash_event(h: &mut Sha256, e: &Event) {
    match e {
        Event::AgentMoved { actor, from, location, tick } => {
            h.update([0u8]);                                   // variant tag
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
            h.update([3u8]);                                   // variant tag
            h.update(agent_id.raw().to_le_bytes());
            for v in [from.x, from.y, from.z, to.x, to.y, to.z] {
                h.update(v.to_bits().to_le_bytes());
            }
            h.update(tick.to_le_bytes());
        }
        Event::AgentAte { agent_id, delta, tick } => {
            h.update([4u8]);                                   // variant tag
            h.update(agent_id.raw().to_le_bytes());
            h.update(delta.to_bits().to_le_bytes());
            h.update(tick.to_le_bytes());
        }
        Event::AgentDrank { agent_id, delta, tick } => {
            h.update([5u8]);                                   // variant tag
            h.update(agent_id.raw().to_le_bytes());
            h.update(delta.to_bits().to_le_bytes());
            h.update(tick.to_le_bytes());
        }
        Event::AgentRested { agent_id, delta, tick } => {
            h.update([6u8]);                                   // variant tag
            h.update(agent_id.raw().to_le_bytes());
            h.update(delta.to_bits().to_le_bytes());
            h.update(tick.to_le_bytes());
        }
        Event::AgentCast { actor, ability, target, depth, tick } => {
            h.update([7u8]);                                   // variant tag
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
            // Resolution encoding: variant tag u8 + (min_parties u8 for Coalition, else 0).
            let (res_tag, min_parties) = match resolution {
                crate::policy::Resolution::HighestBid        => (0u8, 0u8),
                crate::policy::Resolution::FirstAcceptable   => (1u8, 0u8),
                crate::policy::Resolution::MutualAgreement   => (2u8, 0u8),
                crate::policy::Resolution::Coalition { min_parties } => (3u8, *min_parties),
                crate::policy::Resolution::Majority          => (4u8, 0u8),
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
        Event::ChronicleEntry { .. } => {
            // Filtered at the call site; if we reach here, the filter is broken.
            debug_assert!(false, "ChronicleEntry reached replayable hash path");
        }
    }
}
