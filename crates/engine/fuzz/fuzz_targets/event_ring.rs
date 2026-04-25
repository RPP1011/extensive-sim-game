// crates/engine/fuzz/fuzz_targets/event_ring.rs
#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use engine::ability::AbilityId;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::{AgentId, QuestId};
use engine_data::types::{QuestCategory, Resolution};
use glam::Vec3;
use libfuzzer_sys::fuzz_target;

/// A fuzz-constructable Event. We derive Arbitrary on a small wrapper and
/// map into the real Event enum so we don't need to modify engine::Event
/// itself (which lives outside this crate).
#[derive(Arbitrary, Debug)]
struct FuzzEvent {
    tag:        u8,
    agent_raw:  u32,
    target_raw: u32,
    quest_raw:  u32,
    tick:       u32,
    f_a:        f32,
    f_b:        f32,
    u64_a:      u64,
    u_a:        u32,
    audience:   u8,
    item_slot:  u8,
    ability:    u8,
}

impl FuzzEvent {
    fn to_event(&self) -> Event {
        // Clamp all NonZeroU32 ids to >= 1.
        let a = AgentId::new(self.agent_raw.saturating_add(1)).unwrap();
        let b = AgentId::new(self.target_raw.saturating_add(1)).unwrap();
        let q = QuestId::new(self.quest_raw.saturating_add(1)).unwrap();
        let p = Vec3::new(self.f_a, self.f_b, self.f_a.mul_add(0.5, self.f_b));
        let tick = self.tick % 100_000;
        match self.tag % 23 {
            0  => Event::AgentMoved { actor: a, from: p, location: p, tick },
            1  => Event::AgentAttacked { actor: a, target: b, damage: self.f_a, tick },
            2  => Event::AgentDied { agent_id: a, tick },
            3  => Event::AgentFled { agent_id: a, from: p, to: p, tick },
            4  => Event::AgentAte { agent_id: a, delta: self.f_a, tick },
            5  => Event::AgentDrank { agent_id: a, delta: self.f_a, tick },
            6  => Event::AgentRested { agent_id: a, delta: self.f_a, tick },
            7  => Event::AgentCast {
                actor:   a,
                ability: AbilityId::new((self.ability as u32).saturating_add(1)).unwrap(),
                target:  b,
                depth:   self.item_slot,
                tick,
            },
            8  => Event::AgentUsedItem { agent_id: a, item_slot: self.item_slot, tick },
            9  => Event::AgentHarvested { agent_id: a, resource: self.u64_a, tick },
            10 => Event::AgentPlacedTile { actor: a, location: p, kind_tag: self.u_a, tick },
            11 => Event::AgentPlacedVoxel { actor: a, location: p, mat_tag: self.u_a, tick },
            12 => Event::AgentHarvestedVoxel { actor: a, location: p, tick },
            13 => Event::AgentConversed { agent_id: a, partner: b, tick },
            14 => Event::AgentSharedStory { agent_id: a, topic: self.u64_a, tick },
            15 => Event::AgentCommunicated { speaker: a, recipient: b, fact_ref: self.u64_a, tick },
            16 => Event::InformationRequested { asker: a, target: b, query: self.u64_a, tick },
            17 => Event::AgentRemembered { agent_id: a, subject: self.u64_a, tick },
            18 => Event::QuestPosted {
                poster: a, quest_id: q,
                category: QuestCategory::Physical, resolution: Resolution::HighestBid, tick,
            },
            19 => Event::QuestAccepted { acceptor: a, quest_id: q, tick },
            20 => Event::BidPlaced { bidder: a, auction_id: q, amount: self.f_a, tick },
            21 => Event::AnnounceEmitted {
                speaker: a, audience_tag: self.audience % 3,
                fact_payload: self.u64_a, tick,
            },
            22 => Event::RecordMemory {
                observer: a, source: b,
                fact_payload: self.u64_a, confidence: self.f_a, tick,
            },
            _  => unreachable!("tag % 23 in 0..23"),
        }
    }
}

/// The fuzz input is a sequence of FuzzEvents plus a ring capacity.
/// libfuzzer hands us arbitrary bytes; we decode via `arbitrary`.
#[derive(Arbitrary, Debug)]
struct FuzzInput {
    cap:    u16, // 0 rejected below
    events: Vec<FuzzEvent>,
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let Ok(input) = FuzzInput::arbitrary(&mut u) else { return; };
    // Ring capacity must be nonzero; trivial reject.
    let cap = (input.cap as usize).max(1).min(4096);
    if input.events.is_empty() { return; }

    let events: Vec<Event> = input.events.iter().map(|e| e.to_event()).collect();

    // Property: pushing the same sequence twice produces the same hash.
    let push_all = |events: &[Event]| -> [u8; 32] {
        let mut ring = EventRing::with_cap(cap);
        for e in events { ring.push(*e); }
        ring.replayable_sha256()
    };
    let h1 = push_all(&events);
    let h2 = push_all(&events);
    assert_eq!(h1, h2, "replayable hash must be stable across two identical pushes");

    // Property: iter().count() matches expected ring len (retention contract).
    let mut ring = EventRing::with_cap(cap);
    for e in &events { ring.push(*e); }
    let expected_len = events.len().min(cap);
    assert_eq!(ring.iter().count(), expected_len);
});
