//! Property: `EventRing::replayable_sha256()` is a pure function of
//! `(cap, sequence_of_pushed_events)` — running twice gives the same bytes.
//! Complements `tests/event_ring.rs::golden_hash_anchors_format` (single
//! pinned sequence) by covering arbitrary random sequences.
use engine::ability::AbilityId;
use engine::event::{Event, EventRing};
use engine::ids::{AgentId, QuestId};
use engine::policy::{QuestCategory, Resolution};
use glam::Vec3;
use proptest::prelude::*;

/// Strategy for a single Event with a deterministically chosen variant.
/// Covers all 23 replayable variants — the non-replayable `ChronicleEntry`
/// is excluded since it doesn't contribute to the replayable hash.
fn arb_event() -> impl Strategy<Value = Event> {
    // Variants are encoded by ordinal 0..23; we map proptest's u8 to a variant.
    (0u8..23, any::<u32>(), any::<u32>(), any::<u32>(), any::<f32>(), any::<u64>())
        .prop_map(|(tag, u_a, u_b, tick_raw, f_a, u64_a)| {
            let tick = tick_raw % 100_000; // keep ticks reasonable
            let a = AgentId::new(u_a.max(1)).unwrap();
            let b = AgentId::new(u_b.max(1)).unwrap();
            let q = QuestId::new(u_b.max(1)).unwrap();
            let p = Vec3::new(f_a, f_a * 2.0, f_a * 3.0);
            match tag {
                0  => Event::AgentMoved { actor: a, from: p, location: p, tick },
                1  => Event::AgentAttacked { actor: a, target: b, damage: f_a, tick },
                2  => Event::AgentDied { agent_id: a, tick },
                3  => Event::AgentFled { agent_id: a, from: p, to: p, tick },
                4  => Event::AgentAte { agent_id: a, delta: f_a, tick },
                5  => Event::AgentDrank { agent_id: a, delta: f_a, tick },
                6  => Event::AgentRested { agent_id: a, delta: f_a, tick },
                7  => Event::AgentCast {
                    actor: a,
                    ability: AbilityId::new(u_a.max(1)).unwrap(),
                    target: b,
                    depth: (u_a as u8),
                    tick,
                },
                8  => Event::AgentUsedItem { agent_id: a, item_slot: (u_a as u8), tick },
                9  => Event::AgentHarvested { agent_id: a, resource: u64_a, tick },
                10 => Event::AgentPlacedTile { actor: a, location: p, kind_tag: u_a, tick },
                11 => Event::AgentPlacedVoxel { actor: a, location: p, mat_tag: u_a, tick },
                12 => Event::AgentHarvestedVoxel { actor: a, location: p, tick },
                13 => Event::AgentConversed { agent_id: a, partner: b, tick },
                14 => Event::AgentSharedStory { agent_id: a, topic: u64_a, tick },
                15 => Event::AgentCommunicated { speaker: a, recipient: b, fact_ref: u64_a, tick },
                16 => Event::InformationRequested { asker: a, target: b, query: u64_a, tick },
                17 => Event::AgentRemembered { agent_id: a, subject: u64_a, tick },
                18 => Event::QuestPosted {
                    poster: a, quest_id: q,
                    category: QuestCategory::Physical, resolution: Resolution::HighestBid, tick,
                },
                19 => Event::QuestAccepted { acceptor: a, quest_id: q, tick },
                20 => Event::BidPlaced { bidder: a, auction_id: q, amount: f_a, tick },
                21 => Event::AnnounceEmitted {
                    speaker: a, audience_tag: (u_a as u8) % 3, fact_payload: u64_a, tick,
                },
                22 => Event::RecordMemory {
                    observer: a, source: b, fact_payload: u64_a, confidence: f_a, tick,
                },
                _  => unreachable!("tag in 0..23 by construction"),
            }
        })
}

fn push_all(cap: usize, events: &[Event]) -> [u8; 32] {
    let mut ring = EventRing::with_cap(cap);
    for e in events {
        ring.push(*e);
    }
    ring.replayable_sha256()
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 500,
        max_shrink_iters: 2000,
        .. ProptestConfig::default()
    })]

    /// Hashing twice with identical inputs yields identical bytes. Also
    /// covers the ring-overflow eviction path: when `events.len() > cap`,
    /// the *retained subset* of both rings must evict identically.
    #[test]
    fn replayable_hash_is_stable_across_two_runs(
        events in proptest::collection::vec(arb_event(), 1..50),
        cap in 4usize..64,
    ) {
        let h1 = push_all(cap, &events);
        let h2 = push_all(cap, &events);
        prop_assert_eq!(h1, h2, "same input sequence + same cap → same hash");
    }

}
