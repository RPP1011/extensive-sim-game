//! Phase 6b — GPU event ring primitive.
//!
//! A fixed-capacity device-resident buffer of `EventRecord` slots plus a
//! single `atomic<u32>` tail counter. GPU kernels emit events by atomically
//! incrementing the tail and writing one record; the host drains the ring
//! once per sub-dispatch, rehydrating `engine::event::Event` variants into
//! the CPU [`EventRing`].
//!
//! ## Record layout
//!
//! Every record is the same size regardless of which event kind it encodes
//! so the WGSL side can write through a plain `array<EventRecord>` without
//! variant-tag indirection.
//!
//! ```text
//! struct EventRecord {
//!     kind:    u32,       // variant tag — see `EventKindTag`
//!     tick:    u32,       // event tick (matches Event::tick())
//!     payload: [u32; P],  // kind-specific packed words
//! }
//! ```
//!
//! `P = 8` — chosen by scanning every variant of `engine_rules::events::Event`
//! and taking the max payload-word count, plus a one-word cushion. The max
//! at time of writing is **7 words**, set by `AgentMoved` / `AgentFled`
//! (each carrying two `Vec3`s = 6 f32 words + one `AgentId` = 7 words).
//! Events with u64 fields (`AgentCommunicated::fact_ref`,
//! `EffectGoldTransfer::amount`, …) spend two words each; see the
//! per-kind encoders below. Picking 8 rather than 7 keeps the record
//! sized at `(2 + 8) * 4 = 40 B` and gives headroom before a schema
//! bump that adds a new slim field is ever forced into a wider P. The
//! capacity number is documented in [`PAYLOAD_WORDS`].
//!
//! ## Overflow
//!
//! On overflow (tail > capacity at drain time) the kernel's `atomicAdd`
//! has already run for every late emitter, but the guarded store dropped
//! records past slot `CAPACITY - 1`. [`GpuEventRing::drain`] detects this
//! via a `tail > CAPACITY` read, truncates the drain range to
//! `[0, CAPACITY)`, and sets the [`DrainOutcome::overflowed`] flag so
//! tests can assert it never fires. The simulation stays healthy —
//! overflow drops the *tail* of a tick, not any preceding work — but
//! every test in the engine treats a set flag as a hard failure.
//!
//! ## Determinism
//!
//! GPU thread scheduling interleaves atomic increments arbitrarily, so the
//! raw tail-order of emitted records is non-deterministic. [`drain`]
//! sorts the drained slice by `(tick, kind, payload-derived actor-id)`
//! before pushing into the CPU [`EventRing`], reinstating the byte-exact
//! ordering the CPU path produces. The sort is stable so equal-key
//! records (e.g. two `AgentAttacked` from the same actor on the same
//! tick) preserve their insertion order, which only differs from the
//! CPU path if the two events were also physics-concurrent there —
//! parity tests assert byte-identical `replayable_sha256` under this
//! tie-breaking scheme.
//!
//! ## Not yet: integration
//!
//! This module is a primitive. The sibling task 187 (physics WGSL
//! emitter) produces the kernel-side `gpu_emit_event_*` call sites; the
//! follow-up integration task wires a `GpuEventRing` into
//! [`crate::GpuBackend`] and replaces the CPU event path for GPU-driven
//! cascade iterations.

use std::fmt;

use bytemuck::{Pod, Zeroable};
use engine::event::{Event, EventRing};
use engine::ids::{AbilityId, AgentId, QuestId};
use engine_rules::types::{QuestCategory, Resolution};
use glam::Vec3;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Max payload words per record. The largest event variant is
/// `AgentMoved` / `AgentFled` with 7 payload words (one `AgentId` +
/// two `Vec3`s). We pad to 8 for one-word headroom before a schema
/// change forces a bump. Every kernel-side encoder writes exactly
/// this many words per record so the WGSL side can treat the buffer
/// as `array<EventRecord>` without variant-specific strides.
pub const PAYLOAD_WORDS: usize = 8;

/// Default capacity when the caller doesn't supply one. 655 360 records
/// per tick — 10× the original 65 536 envelope — at `RECORD_BYTES = 40 B`
/// this is a `25 MiB` buffer. Bumped so the perf sweep stays overflow-
/// free up to agent_cap ≈ 100 000. Adjust via [`GpuEventRing::new`] if
/// a workload pushes past this envelope; the drain logs a warning
/// before clamping.
pub const DEFAULT_CAPACITY: u32 = 655_360;

/// Task 203 — default chronicle ring capacity. The chronicle ring is
/// parallel to the main event ring but holds ONLY narrative
/// `ChronicleEntry` events. Because chronicle observability dwarfs the
/// replayable event volume (every AgentAttacked fires chronicle_attack
/// + chronicle_wound, every AgentDied fires chronicle_death, etc.) we
/// provision roughly 16× the main-ring default: 1 M records ≈ 24 MiB
/// at [`CHRONICLE_RECORD_BYTES`] = 24 B.
///
/// Overflow semantics: wrap-with-warning. The chronicle is observability
/// only — losing old entries when the ring fills mid-session is
/// acceptable (unlike the main ring, where every record is physics
/// state that must be accounted for). See [`GpuChronicleRing::drain`].
pub const DEFAULT_CHRONICLE_CAPACITY: u32 = 1_000_000;

/// Bytes per record on the wire. `(kind + tick + PAYLOAD_WORDS) * 4`.
pub const RECORD_BYTES: u64 = ((2 + PAYLOAD_WORDS) as u64) * 4;

// ---------------------------------------------------------------------------
// Kind tags
// ---------------------------------------------------------------------------

/// Stable u32 tag identifying an event variant in a GPU [`EventRecord`].
///
/// Tag values mirror the `hash_event` match arm order in
/// `crate::event::ring::hash_event` so a GPU drain that hits a given
/// tag can dispatch the same payload layout the CPU hasher uses. Tags
/// 23 and 24 are intentionally skipped (reserved for the hash-excluded
/// `ChronicleEntry` variant, which is not physics-emittable).
#[repr(u32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum EventKindTag {
    AgentMoved = 0,
    AgentAttacked = 1,
    AgentDied = 2,
    AgentFled = 3,
    AgentAte = 4,
    AgentDrank = 5,
    AgentRested = 6,
    AgentCast = 7,
    AgentUsedItem = 8,
    AgentHarvested = 9,
    AgentPlacedTile = 10,
    AgentPlacedVoxel = 11,
    AgentHarvestedVoxel = 12,
    AgentConversed = 13,
    AgentSharedStory = 14,
    AgentCommunicated = 15,
    InformationRequested = 16,
    AgentRemembered = 17,
    QuestPosted = 18,
    QuestAccepted = 19,
    BidPlaced = 20,
    AnnounceEmitted = 21,
    RecordMemory = 22,
    // 23, 24 reserved / unused (would-have-been-ChronicleEntry slots).
    OpportunityAttackTriggered = 25,
    EffectDamageApplied = 26,
    EffectHealApplied = 27,
    EffectShieldApplied = 28,
    EffectStunApplied = 29,
    EffectSlowApplied = 30,
    EffectGoldTransfer = 31,
    EffectStandingDelta = 32,
    CastDepthExceeded = 33,
    EngagementCommitted = 34,
    EngagementBroken = 35,
    FearSpread = 36,
    PackAssist = 37,
    RallyCall = 38,
}

impl EventKindTag {
    /// Parse a raw u32 tag from a GPU record. Returns `None` for
    /// the reserved / unused slots (23, 24) or out-of-range tags.
    pub fn from_u32(v: u32) -> Option<Self> {
        use EventKindTag::*;
        Some(match v {
            0 => AgentMoved,
            1 => AgentAttacked,
            2 => AgentDied,
            3 => AgentFled,
            4 => AgentAte,
            5 => AgentDrank,
            6 => AgentRested,
            7 => AgentCast,
            8 => AgentUsedItem,
            9 => AgentHarvested,
            10 => AgentPlacedTile,
            11 => AgentPlacedVoxel,
            12 => AgentHarvestedVoxel,
            13 => AgentConversed,
            14 => AgentSharedStory,
            15 => AgentCommunicated,
            16 => InformationRequested,
            17 => AgentRemembered,
            18 => QuestPosted,
            19 => QuestAccepted,
            20 => BidPlaced,
            21 => AnnounceEmitted,
            22 => RecordMemory,
            25 => OpportunityAttackTriggered,
            26 => EffectDamageApplied,
            27 => EffectHealApplied,
            28 => EffectShieldApplied,
            29 => EffectStunApplied,
            30 => EffectSlowApplied,
            31 => EffectGoldTransfer,
            32 => EffectStandingDelta,
            33 => CastDepthExceeded,
            34 => EngagementCommitted,
            35 => EngagementBroken,
            36 => FearSpread,
            37 => PackAssist,
            38 => RallyCall,
            _ => return None,
        })
    }

    pub fn raw(self) -> u32 {
        self as u32
    }
}

// ---------------------------------------------------------------------------
// GPU-POD record layout
// ---------------------------------------------------------------------------

/// On-wire record as written by the kernel and read back by [`drain`].
/// `bytemuck::Pod` requires all fields to be POD and the struct to be
/// `#[repr(C)]`; the `array<u32, N>` layout in WGSL matches the
/// `[u32; N]` here so no manual byte-packing is required at the
/// boundary.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct EventRecord {
    pub kind: u32,
    pub tick: u32,
    pub payload: [u32; PAYLOAD_WORDS],
}

impl EventRecord {
    /// Construct a record directly from its packed words. Primarily
    /// used by CPU-side tests; kernels call the per-kind encoder
    /// helpers in [`wgsl_helpers`].
    pub fn new(kind: EventKindTag, tick: u32, payload: [u32; PAYLOAD_WORDS]) -> Self {
        Self { kind: kind.raw(), tick, payload }
    }
}

// ---------------------------------------------------------------------------
// Event ↔ Record round-trip
// ---------------------------------------------------------------------------

/// Pack a CPU [`Event`] into an [`EventRecord`] using the same word
/// layout the kernel-side emitters produce. Returns `None` for
/// `ChronicleEntry`, which is not physics-emittable and not replayable
/// (no stable tag slot).
///
/// This is used both by the unit tests in this module and by the
/// event-ring parity test, which seeds the GPU buffer with CPU-packed
/// records, drains, and asserts byte-identical round-trip.
pub fn pack_event(e: &Event) -> Option<EventRecord> {
    let mut p = [0u32; PAYLOAD_WORDS];
    let (tag, tick) = match e {
        Event::AgentMoved { actor, from, location, tick } => {
            p[0] = actor.raw();
            p[1] = from.x.to_bits();
            p[2] = from.y.to_bits();
            p[3] = from.z.to_bits();
            p[4] = location.x.to_bits();
            p[5] = location.y.to_bits();
            p[6] = location.z.to_bits();
            (EventKindTag::AgentMoved, *tick)
        }
        Event::AgentAttacked { actor, target, damage, tick } => {
            p[0] = actor.raw();
            p[1] = target.raw();
            p[2] = damage.to_bits();
            (EventKindTag::AgentAttacked, *tick)
        }
        Event::AgentDied { agent_id, tick } => {
            p[0] = agent_id.raw();
            (EventKindTag::AgentDied, *tick)
        }
        Event::AgentFled { agent_id, from, to, tick } => {
            p[0] = agent_id.raw();
            p[1] = from.x.to_bits();
            p[2] = from.y.to_bits();
            p[3] = from.z.to_bits();
            p[4] = to.x.to_bits();
            p[5] = to.y.to_bits();
            p[6] = to.z.to_bits();
            (EventKindTag::AgentFled, *tick)
        }
        Event::AgentAte { agent_id, delta, tick } => {
            p[0] = agent_id.raw();
            p[1] = delta.to_bits();
            (EventKindTag::AgentAte, *tick)
        }
        Event::AgentDrank { agent_id, delta, tick } => {
            p[0] = agent_id.raw();
            p[1] = delta.to_bits();
            (EventKindTag::AgentDrank, *tick)
        }
        Event::AgentRested { agent_id, delta, tick } => {
            p[0] = agent_id.raw();
            p[1] = delta.to_bits();
            (EventKindTag::AgentRested, *tick)
        }
        Event::AgentCast { actor, ability, target, depth, tick } => {
            p[0] = actor.raw();
            p[1] = ability.raw();
            p[2] = target.raw();
            p[3] = *depth as u32;
            (EventKindTag::AgentCast, *tick)
        }
        Event::AgentUsedItem { agent_id, item_slot, tick } => {
            p[0] = agent_id.raw();
            p[1] = *item_slot as u32;
            (EventKindTag::AgentUsedItem, *tick)
        }
        Event::AgentHarvested { agent_id, resource, tick } => {
            p[0] = agent_id.raw();
            let (lo, hi) = split_u64(*resource);
            p[1] = lo;
            p[2] = hi;
            (EventKindTag::AgentHarvested, *tick)
        }
        Event::AgentPlacedTile { actor, location, kind_tag, tick } => {
            p[0] = actor.raw();
            p[1] = location.x.to_bits();
            p[2] = location.y.to_bits();
            p[3] = location.z.to_bits();
            p[4] = *kind_tag;
            (EventKindTag::AgentPlacedTile, *tick)
        }
        Event::AgentPlacedVoxel { actor, location, mat_tag, tick } => {
            p[0] = actor.raw();
            p[1] = location.x.to_bits();
            p[2] = location.y.to_bits();
            p[3] = location.z.to_bits();
            p[4] = *mat_tag;
            (EventKindTag::AgentPlacedVoxel, *tick)
        }
        Event::AgentHarvestedVoxel { actor, location, tick } => {
            p[0] = actor.raw();
            p[1] = location.x.to_bits();
            p[2] = location.y.to_bits();
            p[3] = location.z.to_bits();
            (EventKindTag::AgentHarvestedVoxel, *tick)
        }
        Event::AgentConversed { agent_id, partner, tick } => {
            p[0] = agent_id.raw();
            p[1] = partner.raw();
            (EventKindTag::AgentConversed, *tick)
        }
        Event::AgentSharedStory { agent_id, topic, tick } => {
            p[0] = agent_id.raw();
            let (lo, hi) = split_u64(*topic);
            p[1] = lo;
            p[2] = hi;
            (EventKindTag::AgentSharedStory, *tick)
        }
        Event::AgentCommunicated { speaker, recipient, fact_ref, tick } => {
            p[0] = speaker.raw();
            p[1] = recipient.raw();
            let (lo, hi) = split_u64(*fact_ref);
            p[2] = lo;
            p[3] = hi;
            (EventKindTag::AgentCommunicated, *tick)
        }
        Event::InformationRequested { asker, target, query, tick } => {
            p[0] = asker.raw();
            p[1] = target.raw();
            let (lo, hi) = split_u64(*query);
            p[2] = lo;
            p[3] = hi;
            (EventKindTag::InformationRequested, *tick)
        }
        Event::AgentRemembered { agent_id, subject, tick } => {
            p[0] = agent_id.raw();
            let (lo, hi) = split_u64(*subject);
            p[1] = lo;
            p[2] = hi;
            (EventKindTag::AgentRemembered, *tick)
        }
        Event::QuestPosted { poster, quest_id, category, resolution, tick } => {
            p[0] = poster.raw();
            p[1] = quest_id.raw();
            p[2] = *category as u32;
            let (res_tag, min_parties) = encode_resolution(*resolution);
            p[3] = res_tag as u32;
            p[4] = min_parties as u32;
            (EventKindTag::QuestPosted, *tick)
        }
        Event::QuestAccepted { acceptor, quest_id, tick } => {
            p[0] = acceptor.raw();
            p[1] = quest_id.raw();
            (EventKindTag::QuestAccepted, *tick)
        }
        Event::BidPlaced { bidder, auction_id, amount, tick } => {
            p[0] = bidder.raw();
            p[1] = auction_id.raw();
            p[2] = amount.to_bits();
            (EventKindTag::BidPlaced, *tick)
        }
        Event::AnnounceEmitted { speaker, audience_tag, fact_payload, tick } => {
            p[0] = speaker.raw();
            p[1] = *audience_tag as u32;
            let (lo, hi) = split_u64(*fact_payload);
            p[2] = lo;
            p[3] = hi;
            (EventKindTag::AnnounceEmitted, *tick)
        }
        Event::RecordMemory { observer, source, fact_payload, confidence, tick } => {
            p[0] = observer.raw();
            p[1] = source.raw();
            let (lo, hi) = split_u64(*fact_payload);
            p[2] = lo;
            p[3] = hi;
            p[4] = confidence.to_bits();
            (EventKindTag::RecordMemory, *tick)
        }
        Event::OpportunityAttackTriggered { actor, target, tick } => {
            p[0] = actor.raw();
            p[1] = target.raw();
            (EventKindTag::OpportunityAttackTriggered, *tick)
        }
        Event::EffectDamageApplied { actor, target, amount, tick } => {
            p[0] = actor.raw();
            p[1] = target.raw();
            p[2] = amount.to_bits();
            (EventKindTag::EffectDamageApplied, *tick)
        }
        Event::EffectHealApplied { actor, target, amount, tick } => {
            p[0] = actor.raw();
            p[1] = target.raw();
            p[2] = amount.to_bits();
            (EventKindTag::EffectHealApplied, *tick)
        }
        Event::EffectShieldApplied { actor, target, amount, tick } => {
            p[0] = actor.raw();
            p[1] = target.raw();
            p[2] = amount.to_bits();
            (EventKindTag::EffectShieldApplied, *tick)
        }
        Event::EffectStunApplied { actor, target, expires_at_tick, tick } => {
            p[0] = actor.raw();
            p[1] = target.raw();
            p[2] = *expires_at_tick;
            (EventKindTag::EffectStunApplied, *tick)
        }
        Event::EffectSlowApplied { actor, target, expires_at_tick, factor_q8, tick } => {
            p[0] = actor.raw();
            p[1] = target.raw();
            p[2] = *expires_at_tick;
            // `factor_q8: i16` — reinterpret the sign-bit-preserving
            // representation as u16 before widening so the bit pattern
            // round-trips through the u32 slot.
            p[3] = (*factor_q8 as u16) as u32;
            (EventKindTag::EffectSlowApplied, *tick)
        }
        Event::EffectGoldTransfer { from, to, amount, tick } => {
            p[0] = from.raw();
            p[1] = to.raw();
            let (lo, hi) = split_u64(*amount as u64); // i64 bit-reinterpret via u64
            p[2] = lo;
            p[3] = hi;
            (EventKindTag::EffectGoldTransfer, *tick)
        }
        Event::EffectStandingDelta { a, b, delta, tick } => {
            p[0] = a.raw();
            p[1] = b.raw();
            p[2] = (*delta as u16) as u32;
            (EventKindTag::EffectStandingDelta, *tick)
        }
        Event::CastDepthExceeded { actor, ability, tick } => {
            p[0] = actor.raw();
            p[1] = ability.raw();
            (EventKindTag::CastDepthExceeded, *tick)
        }
        Event::EngagementCommitted { actor, target, tick } => {
            p[0] = actor.raw();
            p[1] = target.raw();
            (EventKindTag::EngagementCommitted, *tick)
        }
        Event::EngagementBroken { actor, former_target, reason, tick } => {
            p[0] = actor.raw();
            p[1] = former_target.raw();
            p[2] = *reason as u32;
            (EventKindTag::EngagementBroken, *tick)
        }
        Event::FearSpread { observer, dead_kin, tick } => {
            p[0] = observer.raw();
            p[1] = dead_kin.raw();
            (EventKindTag::FearSpread, *tick)
        }
        Event::PackAssist { observer, target, tick } => {
            p[0] = observer.raw();
            p[1] = target.raw();
            (EventKindTag::PackAssist, *tick)
        }
        Event::RallyCall { observer, wounded_kin, tick } => {
            p[0] = observer.raw();
            p[1] = wounded_kin.raw();
            (EventKindTag::RallyCall, *tick)
        }
        Event::ChronicleEntry { .. } => {
            return None;
        }
    };
    Some(EventRecord::new(tag, tick, p))
}

/// Reverse of [`pack_event`]. Returns `None` for an unknown tag or for
/// a malformed payload (e.g. zero `AgentId` in a required slot — every
/// id field is `NonZeroU32`, so a kernel that wrote 0 indicates a
/// logic bug upstream).
pub fn unpack_record(r: &EventRecord) -> Option<Event> {
    let tag = EventKindTag::from_u32(r.kind)?;
    let p = &r.payload;
    let tick = r.tick;

    // Helpers that mirror the NonZeroU32 contract. A zero in a slot
    // that's supposed to hold an id is a kernel bug, surfaced as `None`
    // at drain time rather than silently accepted.
    let aid = |w: u32| AgentId::new(w);
    let abid = |w: u32| AbilityId::new(w);
    let qid = |w: u32| QuestId::new(w);

    Some(match tag {
        EventKindTag::AgentMoved => Event::AgentMoved {
            actor: aid(p[0])?,
            from: Vec3::new(
                f32::from_bits(p[1]),
                f32::from_bits(p[2]),
                f32::from_bits(p[3]),
            ),
            location: Vec3::new(
                f32::from_bits(p[4]),
                f32::from_bits(p[5]),
                f32::from_bits(p[6]),
            ),
            tick,
        },
        EventKindTag::AgentAttacked => Event::AgentAttacked {
            actor: aid(p[0])?,
            target: aid(p[1])?,
            damage: f32::from_bits(p[2]),
            tick,
        },
        EventKindTag::AgentDied => Event::AgentDied {
            agent_id: aid(p[0])?,
            tick,
        },
        EventKindTag::AgentFled => Event::AgentFled {
            agent_id: aid(p[0])?,
            from: Vec3::new(
                f32::from_bits(p[1]),
                f32::from_bits(p[2]),
                f32::from_bits(p[3]),
            ),
            to: Vec3::new(
                f32::from_bits(p[4]),
                f32::from_bits(p[5]),
                f32::from_bits(p[6]),
            ),
            tick,
        },
        EventKindTag::AgentAte => Event::AgentAte {
            agent_id: aid(p[0])?,
            delta: f32::from_bits(p[1]),
            tick,
        },
        EventKindTag::AgentDrank => Event::AgentDrank {
            agent_id: aid(p[0])?,
            delta: f32::from_bits(p[1]),
            tick,
        },
        EventKindTag::AgentRested => Event::AgentRested {
            agent_id: aid(p[0])?,
            delta: f32::from_bits(p[1]),
            tick,
        },
        EventKindTag::AgentCast => Event::AgentCast {
            actor: aid(p[0])?,
            ability: abid(p[1])?,
            target: aid(p[2])?,
            depth: (p[3] & 0xFF) as u8,
            tick,
        },
        EventKindTag::AgentUsedItem => Event::AgentUsedItem {
            agent_id: aid(p[0])?,
            item_slot: (p[1] & 0xFF) as u8,
            tick,
        },
        EventKindTag::AgentHarvested => Event::AgentHarvested {
            agent_id: aid(p[0])?,
            resource: join_u64(p[1], p[2]),
            tick,
        },
        EventKindTag::AgentPlacedTile => Event::AgentPlacedTile {
            actor: aid(p[0])?,
            location: Vec3::new(
                f32::from_bits(p[1]),
                f32::from_bits(p[2]),
                f32::from_bits(p[3]),
            ),
            kind_tag: p[4],
            tick,
        },
        EventKindTag::AgentPlacedVoxel => Event::AgentPlacedVoxel {
            actor: aid(p[0])?,
            location: Vec3::new(
                f32::from_bits(p[1]),
                f32::from_bits(p[2]),
                f32::from_bits(p[3]),
            ),
            mat_tag: p[4],
            tick,
        },
        EventKindTag::AgentHarvestedVoxel => Event::AgentHarvestedVoxel {
            actor: aid(p[0])?,
            location: Vec3::new(
                f32::from_bits(p[1]),
                f32::from_bits(p[2]),
                f32::from_bits(p[3]),
            ),
            tick,
        },
        EventKindTag::AgentConversed => Event::AgentConversed {
            agent_id: aid(p[0])?,
            partner: aid(p[1])?,
            tick,
        },
        EventKindTag::AgentSharedStory => Event::AgentSharedStory {
            agent_id: aid(p[0])?,
            topic: join_u64(p[1], p[2]),
            tick,
        },
        EventKindTag::AgentCommunicated => Event::AgentCommunicated {
            speaker: aid(p[0])?,
            recipient: aid(p[1])?,
            fact_ref: join_u64(p[2], p[3]),
            tick,
        },
        EventKindTag::InformationRequested => Event::InformationRequested {
            asker: aid(p[0])?,
            target: aid(p[1])?,
            query: join_u64(p[2], p[3]),
            tick,
        },
        EventKindTag::AgentRemembered => Event::AgentRemembered {
            agent_id: aid(p[0])?,
            subject: join_u64(p[1], p[2]),
            tick,
        },
        EventKindTag::QuestPosted => Event::QuestPosted {
            poster: aid(p[0])?,
            quest_id: qid(p[1])?,
            category: decode_quest_category(p[2])?,
            resolution: decode_resolution(p[3], p[4])?,
            tick,
        },
        EventKindTag::QuestAccepted => Event::QuestAccepted {
            acceptor: aid(p[0])?,
            quest_id: qid(p[1])?,
            tick,
        },
        EventKindTag::BidPlaced => Event::BidPlaced {
            bidder: aid(p[0])?,
            auction_id: qid(p[1])?,
            amount: f32::from_bits(p[2]),
            tick,
        },
        EventKindTag::AnnounceEmitted => Event::AnnounceEmitted {
            speaker: aid(p[0])?,
            audience_tag: (p[1] & 0xFF) as u8,
            fact_payload: join_u64(p[2], p[3]),
            tick,
        },
        EventKindTag::RecordMemory => Event::RecordMemory {
            observer: aid(p[0])?,
            source: aid(p[1])?,
            fact_payload: join_u64(p[2], p[3]),
            confidence: f32::from_bits(p[4]),
            tick,
        },
        EventKindTag::OpportunityAttackTriggered => Event::OpportunityAttackTriggered {
            actor: aid(p[0])?,
            target: aid(p[1])?,
            tick,
        },
        EventKindTag::EffectDamageApplied => Event::EffectDamageApplied {
            actor: aid(p[0])?,
            target: aid(p[1])?,
            amount: f32::from_bits(p[2]),
            tick,
        },
        EventKindTag::EffectHealApplied => Event::EffectHealApplied {
            actor: aid(p[0])?,
            target: aid(p[1])?,
            amount: f32::from_bits(p[2]),
            tick,
        },
        EventKindTag::EffectShieldApplied => Event::EffectShieldApplied {
            actor: aid(p[0])?,
            target: aid(p[1])?,
            amount: f32::from_bits(p[2]),
            tick,
        },
        EventKindTag::EffectStunApplied => Event::EffectStunApplied {
            actor: aid(p[0])?,
            target: aid(p[1])?,
            expires_at_tick: p[2],
            tick,
        },
        EventKindTag::EffectSlowApplied => Event::EffectSlowApplied {
            actor: aid(p[0])?,
            target: aid(p[1])?,
            expires_at_tick: p[2],
            factor_q8: (p[3] as u16) as i16,
            tick,
        },
        EventKindTag::EffectGoldTransfer => Event::EffectGoldTransfer {
            from: aid(p[0])?,
            to: aid(p[1])?,
            amount: join_u64(p[2], p[3]) as i32,
            tick,
        },
        EventKindTag::EffectStandingDelta => Event::EffectStandingDelta {
            a: aid(p[0])?,
            b: aid(p[1])?,
            delta: (p[2] as u16) as i16,
            tick,
        },
        EventKindTag::CastDepthExceeded => Event::CastDepthExceeded {
            actor: aid(p[0])?,
            ability: abid(p[1])?,
            tick,
        },
        EventKindTag::EngagementCommitted => Event::EngagementCommitted {
            actor: aid(p[0])?,
            target: aid(p[1])?,
            tick,
        },
        EventKindTag::EngagementBroken => Event::EngagementBroken {
            actor: aid(p[0])?,
            former_target: aid(p[1])?,
            reason: (p[2] & 0xFF) as u8,
            tick,
        },
        EventKindTag::FearSpread => Event::FearSpread {
            observer: aid(p[0])?,
            dead_kin: aid(p[1])?,
            tick,
        },
        EventKindTag::PackAssist => Event::PackAssist {
            observer: aid(p[0])?,
            target: aid(p[1])?,
            tick,
        },
        EventKindTag::RallyCall => Event::RallyCall {
            observer: aid(p[0])?,
            wounded_kin: aid(p[1])?,
            tick,
        },
    })
}

// ---------------------------------------------------------------------------
// Small encoding helpers
// ---------------------------------------------------------------------------

#[inline]
fn split_u64(v: u64) -> (u32, u32) {
    ((v & 0xFFFF_FFFF) as u32, (v >> 32) as u32)
}

#[inline]
fn join_u64(lo: u32, hi: u32) -> u64 {
    (lo as u64) | ((hi as u64) << 32)
}

/// Pack a `Resolution` into the `(tag, min_parties)` pair the hash
/// helper uses. Matches `EventRing::replayable_sha256`'s encoding.
fn encode_resolution(r: Resolution) -> (u8, u8) {
    match r {
        Resolution::HighestBid => (0, 0),
        Resolution::FirstAcceptable => (1, 0),
        Resolution::MutualAgreement => (2, 0),
        Resolution::Coalition { min_parties } => (3, min_parties),
        Resolution::Majority => (4, 0),
    }
}

fn decode_resolution(tag_word: u32, min_parties_word: u32) -> Option<Resolution> {
    let tag = (tag_word & 0xFF) as u8;
    let mp = (min_parties_word & 0xFF) as u8;
    Some(match tag {
        0 => Resolution::HighestBid,
        1 => Resolution::FirstAcceptable,
        2 => Resolution::MutualAgreement,
        3 => Resolution::Coalition { min_parties: mp },
        4 => Resolution::Majority,
        _ => return None,
    })
}

fn decode_quest_category(w: u32) -> Option<QuestCategory> {
    Some(match (w & 0xFF) as u8 {
        0 => QuestCategory::Physical,
        1 => QuestCategory::Political,
        2 => QuestCategory::Personal,
        3 => QuestCategory::Economic,
        4 => QuestCategory::Narrative,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// WGSL helper source (kernel side)
// ---------------------------------------------------------------------------

/// WGSL fragment the physics emitter (task 187) splices into its
/// kernel before any `gpu_emit_event_*` call sites. Declares:
///
///   * the `EventRecord` struct (matches the Rust `#[repr(C)]` layout)
///   * the `event_ring` storage array + `event_ring_tail` atomic
///   * per-kind encoder helpers that pack typed args into the record
///
/// The caller supplies `EVENT_RING_CAP` as a `const` override
/// in the host-generated shader prefix (or a `pipeline_override`
/// constant, depending on the wgpu backend) — this source uses the
/// constant verbatim so parity test and production emitter share
/// the same module text.
///
/// Binding indices are supplied relative to a base — the physics
/// kernel owns its own bind group layout, so it stitches this source
/// into its own `@group` / `@binding` prefix at emit time. See
/// [`GpuEventRing::bind_group_layout_entries`] and
/// [`GpuEventRing::bind_group_entries`] for the host-side shapes
/// matched by whichever `@binding(N)` / `@binding(N+1)` values the
/// kernel chose.
pub const EVENT_RING_WGSL: &str = r#"
// === Event ring bindings — supplied by host emitter ===
//
// The emitter prepends the following two bindings at some agreed
// base_binding:
//
//   @group(G) @binding(BASE+0) var<storage, read_write> event_ring: array<EventRecord>;
//   @group(G) @binding(BASE+1) var<storage, read_write> event_ring_tail: atomic<u32>;
//
// `EVENT_RING_CAP` and `EVENT_RING_PAYLOAD_WORDS` are declared as
// host-substituted consts in the shader prefix. Keep this block's
// expectations in sync with `EVENT_RING_WGSL` + `PAYLOAD_WORDS`.

struct EventRecord {
    kind: u32,
    tick: u32,
    payload: array<u32, EVENT_RING_PAYLOAD_WORDS>,
};

// Core append. Returns the slot index the record landed in, or
// 0xFFFFFFFFu on overflow. Every per-kind helper below calls this.
fn gpu_emit_event(kind: u32, tick: u32,
                  p0: u32, p1: u32, p2: u32, p3: u32,
                  p4: u32, p5: u32, p6: u32, p7: u32) -> u32 {
    let idx = atomicAdd(&event_ring_tail, 1u);
    if (idx >= EVENT_RING_CAP) {
        // Silent drop — the CPU drain sees tail > cap and flips the
        // overflow flag. We deliberately don't retry / spin because
        // that would serialise emitters on a full ring.
        return 0xFFFFFFFFu;
    }
    var r: EventRecord;
    r.kind = kind;
    r.tick = tick;
    r.payload[0] = p0;
    r.payload[1] = p1;
    r.payload[2] = p2;
    r.payload[3] = p3;
    r.payload[4] = p4;
    r.payload[5] = p5;
    r.payload[6] = p6;
    r.payload[7] = p7;
    event_ring[idx] = r;
    return idx;
}

// === Per-kind encoders ===
//
// Thin wrappers around `gpu_emit_event`. The kernel calls these with
// already-typed args; the helper packs them into the fixed payload
// layout the CPU drain expects. Argument names match the field names
// on `Event::<Variant>` so a reader who knows the CPU side knows
// what slot each word lives in.

fn gpu_emit_agent_moved(actor: u32, tick: u32,
                        fx: f32, fy: f32, fz: f32,
                        lx: f32, ly: f32, lz: f32) -> u32 {
    return gpu_emit_event(0u, tick, actor,
                          bitcast<u32>(fx), bitcast<u32>(fy), bitcast<u32>(fz),
                          bitcast<u32>(lx), bitcast<u32>(ly), bitcast<u32>(lz),
                          0u);
}

// NOTE: WGSL reserves `target` as a keyword, so every helper takes
// `target_id` / `partner` / `former` where the CPU struct uses
// `target`. Kernel authors can still name their own locals however
// they like; this is purely an argument-name concern.

fn gpu_emit_agent_attacked(actor: u32, target_id: u32, damage: f32, tick: u32) -> u32 {
    return gpu_emit_event(1u, tick, actor, target_id, bitcast<u32>(damage),
                          0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_agent_died(agent_id: u32, tick: u32) -> u32 {
    return gpu_emit_event(2u, tick, agent_id, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_agent_fled(agent_id: u32, tick: u32,
                       fx: f32, fy: f32, fz: f32,
                       tx: f32, ty: f32, tz: f32) -> u32 {
    return gpu_emit_event(3u, tick, agent_id,
                          bitcast<u32>(fx), bitcast<u32>(fy), bitcast<u32>(fz),
                          bitcast<u32>(tx), bitcast<u32>(ty), bitcast<u32>(tz),
                          0u);
}

fn gpu_emit_agent_ate(agent_id: u32, delta: f32, tick: u32) -> u32 {
    return gpu_emit_event(4u, tick, agent_id, bitcast<u32>(delta),
                          0u, 0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_agent_drank(agent_id: u32, delta: f32, tick: u32) -> u32 {
    return gpu_emit_event(5u, tick, agent_id, bitcast<u32>(delta),
                          0u, 0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_agent_rested(agent_id: u32, delta: f32, tick: u32) -> u32 {
    return gpu_emit_event(6u, tick, agent_id, bitcast<u32>(delta),
                          0u, 0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_agent_cast(actor: u32, ability: u32, target_id: u32, depth: u32, tick: u32) -> u32 {
    return gpu_emit_event(7u, tick, actor, ability, target_id, depth,
                          0u, 0u, 0u, 0u);
}

fn gpu_emit_opportunity_attack(actor: u32, target_id: u32, tick: u32) -> u32 {
    return gpu_emit_event(25u, tick, actor, target_id, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_effect_damage(actor: u32, target_id: u32, amount: f32, tick: u32) -> u32 {
    return gpu_emit_event(26u, tick, actor, target_id, bitcast<u32>(amount),
                          0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_effect_heal(actor: u32, target_id: u32, amount: f32, tick: u32) -> u32 {
    return gpu_emit_event(27u, tick, actor, target_id, bitcast<u32>(amount),
                          0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_effect_shield(actor: u32, target_id: u32, amount: f32, tick: u32) -> u32 {
    return gpu_emit_event(28u, tick, actor, target_id, bitcast<u32>(amount),
                          0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_effect_stun(actor: u32, target_id: u32, expires_at_tick: u32, tick: u32) -> u32 {
    return gpu_emit_event(29u, tick, actor, target_id, expires_at_tick,
                          0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_cast_depth_exceeded(actor: u32, ability: u32, tick: u32) -> u32 {
    return gpu_emit_event(33u, tick, actor, ability, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_engagement_committed(actor: u32, target_id: u32, tick: u32) -> u32 {
    return gpu_emit_event(34u, tick, actor, target_id, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_engagement_broken(actor: u32, former_target: u32, reason: u32, tick: u32) -> u32 {
    return gpu_emit_event(35u, tick, actor, former_target, reason,
                          0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_fear_spread(observer: u32, dead_kin: u32, tick: u32) -> u32 {
    return gpu_emit_event(36u, tick, observer, dead_kin, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_pack_assist(observer: u32, target_id: u32, tick: u32) -> u32 {
    return gpu_emit_event(37u, tick, observer, target_id, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn gpu_emit_rally_call(observer: u32, wounded_kin: u32, tick: u32) -> u32 {
    return gpu_emit_event(38u, tick, observer, wounded_kin, 0u, 0u, 0u, 0u, 0u, 0u);
}
"#;

/// Shader-prefix constant declarations the host emitter prepends
/// before [`EVENT_RING_WGSL`]. Returned as a formatted `String` rather
/// than a raw const so `PAYLOAD_WORDS` / capacity can vary by ring.
pub fn wgsl_prefix(capacity: u32) -> String {
    format!(
        "const EVENT_RING_CAP: u32 = {}u;\n\
         const EVENT_RING_PAYLOAD_WORDS: u32 = {}u;\n",
        capacity, PAYLOAD_WORDS
    )
}

// ---------------------------------------------------------------------------
// Drain outcome + host API
// ---------------------------------------------------------------------------

/// Result of a single [`GpuEventRing::drain`] call.
#[derive(Debug, Default, Clone, Copy)]
pub struct DrainOutcome {
    /// Number of records the kernel atomically claimed this tick —
    /// includes the records that were dropped due to overflow. A
    /// healthy run keeps this `<= capacity`.
    pub tail_raw: u32,
    /// Number of records actually read back and pushed into the CPU
    /// ring. Equal to `min(tail_raw, capacity)` after drain.
    pub drained: u32,
    /// True iff `tail_raw > capacity`. Signals a lost-event condition
    /// tests should surface as a hard failure.
    pub overflowed: bool,
}

// ---------------------------------------------------------------------------
// wgpu-backed implementation — `gpu` feature gated by parent lib.rs
// ---------------------------------------------------------------------------

/// Errors surfaced by [`GpuEventRing`] host operations.
#[derive(Debug)]
pub enum EventRingError {
    Map(String),
    Dispatch(String),
}

impl fmt::Display for EventRingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventRingError::Map(s) => write!(f, "map_async: {s}"),
            EventRingError::Dispatch(s) => write!(f, "dispatch: {s}"),
        }
    }
}

impl std::error::Error for EventRingError {}

/// Host-side handle to a GPU-resident event ring. Owns the storage
/// buffer, tail atomic, and a readback staging buffer sized to the
/// ring capacity.
pub struct GpuEventRing {
    buffer: wgpu::Buffer,
    tail_buffer: wgpu::Buffer,
    tail_readback: wgpu::Buffer,
    staging: wgpu::Buffer,
    capacity: u32,
}

impl GpuEventRing {
    /// Allocate a ring of `capacity` records. All three buffers are
    /// zero-initialised via `mapped_at_creation` so the first [`reset`]
    /// on a fresh handle is redundant but correct.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let cap = capacity.max(1);
        let buffer_bytes = (cap as u64) * RECORD_BYTES;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::event_ring::buffer"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tail_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::event_ring::tail"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tail_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::event_ring::tail_readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::event_ring::staging"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { buffer, tail_buffer, tail_readback, staging, capacity: cap }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Byte size of the GPU-resident records buffer. Does not include
    /// the 4 B tail atomic or the readback staging copy.
    pub fn buffer_bytes(&self) -> u64 {
        (self.capacity as u64) * RECORD_BYTES
    }

    /// Zero the tail atomic at tick start. Kernel `atomicAdd` starts
    /// counting from 0 again on the next dispatch. The records buffer
    /// itself is NOT cleared — old slots beyond the new tail are
    /// logically garbage but never read.
    pub fn reset(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.tail_buffer, 0, &0u32.to_le_bytes());
    }

    /// Copy tail + records out to host memory and deserialize into the
    /// CPU [`EventRing`]. Returns a [`DrainOutcome`] the caller can
    /// assert on. Events that fail to deserialize (malformed tag, zero
    /// `AgentId` where a NonZeroU32 id was expected) are silently
    /// skipped — a malformed record is strictly a kernel bug, not a
    /// simulation-state bug.
    ///
    /// Determinism: the drained slice is sorted by
    /// `(tick, kind, payload[0])` (stable sort) before push into the
    /// CPU ring. `payload[0]` is the actor / observer / agent-id for
    /// every variant, giving a natural secondary key.
    pub fn drain(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        events: &mut EventRing,
    ) -> Result<DrainOutcome, EventRingError> {
        // Copy tail into readback buffer.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine_gpu::event_ring::drain_tail_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.tail_buffer, 0, &self.tail_readback, 0, 4);
        queue.submit(Some(encoder.finish()));

        let tail_raw = read_u32(&self.tail_readback, device)?;
        let drained_count = tail_raw.min(self.capacity);
        let overflowed = tail_raw > self.capacity;

        if overflowed {
            eprintln!(
                "engine_gpu::event_ring: overflow — tail={tail_raw} exceeds capacity={} \
                 ({} records dropped)",
                self.capacity,
                tail_raw - self.capacity,
            );
        }

        if drained_count == 0 {
            return Ok(DrainOutcome {
                tail_raw,
                drained: 0,
                overflowed,
            });
        }

        // Copy records [0..drained_count) into the staging buffer and
        // map it for CPU read.
        let drained_bytes = (drained_count as u64) * RECORD_BYTES;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine_gpu::event_ring::drain_records_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &self.staging, 0, drained_bytes);
        queue.submit(Some(encoder.finish()));

        let records = read_records(&self.staging, device, drained_count as usize)?;

        // Sort for determinism. Stable sort — equal-key records keep
        // insertion order. Key: (tick, kind, payload[0]).
        let mut sorted = records;
        sorted.sort_by_key(|r| (r.tick, r.kind, r.payload[0]));

        let mut pushed = 0u32;
        for r in &sorted {
            if let Some(ev) = unpack_record(r) {
                events.push(ev);
                pushed += 1;
            }
        }

        Ok(DrainOutcome {
            tail_raw,
            drained: pushed,
            overflowed,
        })
    }

    /// Bind-group-layout entries (records + tail), offset from
    /// `base_binding`. The kernel binds them at
    /// `(base_binding + 0, base_binding + 1)` — records first, then
    /// tail — matching the expected order in [`EVENT_RING_WGSL`].
    pub fn bind_group_layout_entries(&self, base_binding: u32) -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![
            wgpu::BindGroupLayoutEntry {
                binding: base_binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: base_binding + 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]
    }

    /// Bind-group entries pointing at the records + tail buffers at
    /// `(base_binding, base_binding + 1)`.
    pub fn bind_group_entries(&self, base_binding: u32) -> Vec<wgpu::BindGroupEntry<'_>> {
        vec![
            wgpu::BindGroupEntry {
                binding: base_binding,
                resource: self.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: base_binding + 1,
                resource: self.tail_buffer.as_entire_binding(),
            },
        ]
    }

    /// Underlying wgpu handle to the records storage buffer. Kernels
    /// that build their own bind group can use this directly. Prefer
    /// [`bind_group_entries`] for the common path.
    pub fn records_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Underlying wgpu handle to the tail atomic buffer.
    pub fn tail_buffer(&self) -> &wgpu::Buffer {
        &self.tail_buffer
    }

    /// CPU-side upload helper for tests / seeding: write a batch of
    /// pre-packed records directly to the GPU buffer and set the tail
    /// to match. Simulates a kernel that atomically appended
    /// `records.len()` events. Panics if `records.len() > capacity`.
    pub fn seed_for_test(&self, queue: &wgpu::Queue, records: &[EventRecord]) {
        assert!(
            records.len() <= self.capacity as usize,
            "seed_for_test: {} records > capacity {}",
            records.len(),
            self.capacity,
        );
        if records.is_empty() {
            queue.write_buffer(&self.tail_buffer, 0, &0u32.to_le_bytes());
            return;
        }
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(records));
        let tail = records.len() as u32;
        queue.write_buffer(&self.tail_buffer, 0, &tail.to_le_bytes());
    }

    /// Same as [`seed_for_test`] but sets the tail to an arbitrary
    /// value — used by the overflow test to force `tail > capacity`.
    /// The records buffer itself is only written up to the smaller
    /// of `records.len()` and `capacity`.
    pub fn seed_with_tail_for_test(
        &self,
        queue: &wgpu::Queue,
        records: &[EventRecord],
        tail: u32,
    ) {
        let write_n = records.len().min(self.capacity as usize);
        if write_n > 0 {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&records[..write_n]));
        }
        queue.write_buffer(&self.tail_buffer, 0, &tail.to_le_bytes());
    }
}

// ---------------------------------------------------------------------------
// Task 203 — GPU chronicle ring (separate from the replayable event ring)
// ---------------------------------------------------------------------------
//
// The hot path shouldn't pay for observability. The eight chronicle
// rules (`chronicle_attack`, `_death`, `_engagement`, `_wound`, `_break`,
// `_rout`, `_flee`, `_rally`) each fire on a replayable trigger and
// push a `ChronicleEntry` event. In the prior design those entries went
// into the main `event_ring` alongside state-mutating events, so every
// chronicle emission cost an atomicAdd on the shared ring tail and
// 40 B of bandwidth per record. The CPU drain then dropped them
// (`unpack_record` returns None for kind=24), making the GPU work
// entirely wasted — and the cascade convergence detection had to scan
// past chronicle records on every sub-dispatch.
//
// The chronicle ring is a dedicated, larger buffer that only holds
// chronicle records. Physics emitter routes `emit ChronicleEntry` to
// the chronicle ring's helper instead of the main ring. The host
// drains it lazily — see [`GpuBackend::flush_chronicle`].
//
// ## Record layout
//
// Chronicle records are narrower than the main record (16 B vs. 40 B):
//
// ```text
// struct ChronicleRecord {
//     template_id: u32,
//     agent:       u32,   // 1-based AgentId
//     target:      u32,   // 1-based AgentId
//     tick:        u32,
// }
// ```
//
// ## Overflow
//
// On overflow the chronicle ring WRAPS with a warning rather than
// dropping records at the tail. Because chronicle is observability —
// not physics state — losing the OLDEST entries when the ring fills
// is acceptable; losing the NEWEST would drop the most recent
// narrative, which is exactly what a caller invoking
// `flush_chronicle` mid-session wants to see. The kernel-side emit
// path stores into `slot = idx % cap` unconditionally; the host-side
// drain detects `tail_raw > capacity` and prints a warning with the
// number of dropped-from-head records. The drain still recovers every
// record currently resident in the buffer (up to `capacity`), just in
// a rotated order that's flagged in the outcome.

/// Record layout for the chronicle ring. 16 B total, matches the WGSL
/// `ChronicleRecord` struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct ChronicleRecord {
    pub template_id: u32,
    pub agent: u32,
    pub target: u32,
    pub tick: u32,
}

impl ChronicleRecord {
    pub fn new(template_id: u32, agent: AgentId, target: AgentId, tick: u32) -> Self {
        Self {
            template_id,
            agent: agent.raw(),
            target: target.raw(),
            tick,
        }
    }
}

/// Bytes per chronicle record on the wire.
pub const CHRONICLE_RECORD_BYTES: u64 = 16;

/// WGSL fragment the physics emitter splices into the kernel. Declares:
///
///   * the `ChronicleRecord` struct matching the Rust `#[repr(C)]` layout
///   * the `chronicle_ring` storage array + `chronicle_ring_tail` atomic
///   * the `gpu_emit_chronicle_event` helper used by the physics emitter
///     for `emit ChronicleEntry` sites
///
/// `CHRONICLE_RING_CAP` is declared as a host-substituted const in the
/// shader prefix. The kernel inserts its own `@group` / `@binding`
/// declarations at whatever slot it reserves for chronicle.
///
/// Wrap-on-overflow: if the atomic tail exceeds capacity the store
/// writes to `slot = idx % cap`. Older entries at that wrapped slot
/// are overwritten — acceptable because the chronicle is
/// observability (not replay), per the task 203 contract.
pub const CHRONICLE_RING_WGSL: &str = r#"
struct ChronicleRecord {
    template_id: u32,
    agent: u32,
    target_id: u32,
    tick: u32,
};

// Append a chronicle record. `target_id` argname avoids the WGSL
// reserved word `target`. Wraps on overflow: the atomic increment
// always succeeds, but slot selection is `idx % cap` so older entries
// get overwritten rather than newer ones dropped. The caller can read
// `chronicle_ring_tail` post-dispatch to learn whether overflow
// occurred (`tail_raw > cap`).
fn gpu_emit_chronicle_event(template_id: u32, agent: u32, target_id: u32, tick: u32) -> u32 {
    let idx = atomicAdd(&chronicle_ring_tail, 1u);
    let slot = idx % CHRONICLE_RING_CAP;
    var r: ChronicleRecord;
    r.template_id = template_id;
    r.agent = agent;
    r.target_id = target_id;
    r.tick = tick;
    chronicle_ring[slot] = r;
    return idx;
}
"#;

/// Prefix constant declarations the host emitter prepends before
/// [`CHRONICLE_RING_WGSL`]. Capacity is substituted per-ring.
pub fn chronicle_wgsl_prefix(capacity: u32) -> String {
    format!("const CHRONICLE_RING_CAP: u32 = {}u;\n", capacity)
}

/// Outcome of a single [`GpuChronicleRing::drain`] call.
#[derive(Debug, Default, Clone, Copy)]
pub struct ChronicleDrainOutcome {
    /// Cumulative number of records the kernel atomically claimed
    /// since the last [`GpuChronicleRing::reset`]. Includes wrapped
    /// records that overwrote older entries.
    pub tail_raw: u32,
    /// Number of records read back and pushed into the CPU
    /// [`EventRing`]. Equal to `min(tail_raw, capacity)`.
    pub drained: u32,
    /// True iff `tail_raw > capacity` — older records were overwritten.
    /// The drain still returns what IS in the buffer, just flags that
    /// history before `tail_raw - capacity` is lost.
    pub wrapped: bool,
}

/// Host-side handle to a GPU-resident chronicle ring. Owns the
/// storage buffer, tail atomic, and a readback staging buffer sized to
/// the ring capacity.
pub struct GpuChronicleRing {
    buffer: wgpu::Buffer,
    tail_buffer: wgpu::Buffer,
    tail_readback: wgpu::Buffer,
    staging: wgpu::Buffer,
    capacity: u32,
}

impl GpuChronicleRing {
    /// Allocate a chronicle ring with `capacity` record slots. At the
    /// default 1 M capacity the records buffer is 16 MB — callers on
    /// memory-constrained adapters can shrink this via an explicit
    /// capacity, at the cost of more frequent wrap-around warnings in
    /// long-running sessions.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let cap = capacity.max(1);
        let buffer_bytes = (cap as u64) * CHRONICLE_RECORD_BYTES;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::chronicle_ring::buffer"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tail_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::chronicle_ring::tail"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tail_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::chronicle_ring::tail_readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::chronicle_ring::staging"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { buffer, tail_buffer, tail_readback, staging, capacity: cap }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Byte size of the GPU-resident records buffer.
    pub fn buffer_bytes(&self) -> u64 {
        (self.capacity as u64) * CHRONICLE_RECORD_BYTES
    }

    /// Zero the tail atomic. Kernel `atomicAdd` resets to 0; the
    /// records buffer itself is NOT cleared, but the drain only reads
    /// `[0, min(tail_raw, capacity))` so stale records are invisible.
    pub fn reset(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.tail_buffer, 0, &0u32.to_le_bytes());
    }

    /// Bind-group-layout entries (records + tail) offset from
    /// `base_binding`. Mirrors [`GpuEventRing::bind_group_layout_entries`].
    pub fn bind_group_layout_entries(&self, base_binding: u32) -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![
            wgpu::BindGroupLayoutEntry {
                binding: base_binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: base_binding + 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]
    }

    /// Bind-group entries pointing at records + tail buffers.
    pub fn bind_group_entries(&self, base_binding: u32) -> Vec<wgpu::BindGroupEntry<'_>> {
        vec![
            wgpu::BindGroupEntry {
                binding: base_binding,
                resource: self.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: base_binding + 1,
                resource: self.tail_buffer.as_entire_binding(),
            },
        ]
    }

    pub fn records_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn tail_buffer(&self) -> &wgpu::Buffer {
        &self.tail_buffer
    }

    /// Drain the chronicle ring into `events`. Every resident record
    /// becomes an `Event::ChronicleEntry` on the CPU ring. Does NOT
    /// sort — chronicle events are observability-only, never
    /// replayable, so ordering is best-effort (tail-order within a
    /// single drain, plus wrap rotation if overflow occurred).
    ///
    /// Overflow (`tail_raw > capacity`) produces a warning on stderr
    /// naming the number of lost-from-head records. The drain still
    /// pushes every currently-resident record — including the wrap
    /// survivors — into the CPU ring. After the drain, the kernel
    /// sees the tail atomic UNCHANGED (the drain does not reset),
    /// matching the lazy-drain contract: callers opt in to clearing
    /// via [`reset`] when they want a fresh window.
    pub fn drain(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        events: &mut EventRing,
    ) -> Result<ChronicleDrainOutcome, EventRingError> {
        // Copy tail into readback buffer.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine_gpu::chronicle_ring::drain_tail_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.tail_buffer, 0, &self.tail_readback, 0, 4);
        queue.submit(Some(encoder.finish()));

        let tail_raw = read_u32(&self.tail_readback, device)?;
        let drained_count = tail_raw.min(self.capacity);
        let wrapped = tail_raw > self.capacity;

        if wrapped {
            eprintln!(
                "engine_gpu::chronicle_ring: wrapped — tail={tail_raw} exceeds capacity={} \
                 ({} oldest entries overwritten; observability-only, not a replay hazard)",
                self.capacity,
                tail_raw - self.capacity,
            );
        }

        if drained_count == 0 {
            return Ok(ChronicleDrainOutcome {
                tail_raw,
                drained: 0,
                wrapped,
            });
        }

        let drained_bytes = (drained_count as u64) * CHRONICLE_RECORD_BYTES;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine_gpu::chronicle_ring::drain_records_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &self.staging, 0, drained_bytes);
        queue.submit(Some(encoder.finish()));

        let records = read_chronicle_records(&self.staging, device, drained_count as usize)?;

        let mut pushed = 0u32;
        for r in &records {
            // Malformed ids (zero) surface as a skipped record rather
            // than a panic — a kernel bug in the physics emitter would
            // produce a zero AgentId slot, but chronicle is
            // non-replayable so a silent drop is strictly better than
            // a crash.
            let Some(agent) = AgentId::new(r.agent) else { continue };
            let Some(target) = AgentId::new(r.target) else { continue };
            events.push(Event::ChronicleEntry {
                template_id: r.template_id,
                agent,
                target,
                tick: r.tick,
            });
            pushed += 1;
        }

        Ok(ChronicleDrainOutcome {
            tail_raw,
            drained: pushed,
            wrapped,
        })
    }

    /// CPU-side test helper: write a batch of pre-packed records
    /// directly to the GPU buffer and set the tail to match. Panics
    /// if `records.len() > capacity`.
    pub fn seed_for_test(&self, queue: &wgpu::Queue, records: &[ChronicleRecord]) {
        assert!(
            records.len() <= self.capacity as usize,
            "chronicle seed_for_test: {} records > capacity {}",
            records.len(),
            self.capacity,
        );
        if records.is_empty() {
            queue.write_buffer(&self.tail_buffer, 0, &0u32.to_le_bytes());
            return;
        }
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(records));
        let tail = records.len() as u32;
        queue.write_buffer(&self.tail_buffer, 0, &tail.to_le_bytes());
    }
}

fn read_chronicle_records(
    buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    n: usize,
) -> Result<Vec<ChronicleRecord>, EventRingError> {
    let byte_len = (n as u64) * CHRONICLE_RECORD_BYTES;
    let slice = buffer.slice(..byte_len);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    let map_result = rx
        .recv()
        .map_err(|e| EventRingError::Map(format!("channel closed: {e}")))?;
    map_result.map_err(|e| EventRingError::Map(format!("{e:?}")))?;
    let data = slice.get_mapped_range();
    let casted: &[ChronicleRecord] = bytemuck::cast_slice(&data);
    let out = casted[..n].to_vec();
    drop(data);
    buffer.unmap();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Small wgpu readback helpers (private)
// ---------------------------------------------------------------------------

fn read_u32(buffer: &wgpu::Buffer, device: &wgpu::Device) -> Result<u32, EventRingError> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    let map_result = rx
        .recv()
        .map_err(|e| EventRingError::Map(format!("channel closed: {e}")))?;
    map_result.map_err(|e| EventRingError::Map(format!("{e:?}")))?;
    let data = slice.get_mapped_range();
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&data[..4]);
    drop(data);
    buffer.unmap();
    Ok(u32::from_le_bytes(bytes))
}

fn read_records(
    buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    n: usize,
) -> Result<Vec<EventRecord>, EventRingError> {
    let byte_len = (n as u64) * RECORD_BYTES;
    let slice = buffer.slice(..byte_len);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    let map_result = rx
        .recv()
        .map_err(|e| EventRingError::Map(format!("channel closed: {e}")))?;
    map_result.map_err(|e| EventRingError::Map(format!("{e:?}")))?;
    let data = slice.get_mapped_range();
    let casted: &[EventRecord] = bytemuck::cast_slice(&data);
    let out = casted[..n].to_vec();
    drop(data);
    buffer.unmap();
    Ok(out)
}

// ---------------------------------------------------------------------------
// CPU-side unit tests — no GPU needed
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn aid(r: u32) -> AgentId {
        AgentId::new(r).unwrap()
    }

    fn abid(r: u32) -> AbilityId {
        AbilityId::new(r).unwrap()
    }

    fn qid(r: u32) -> QuestId {
        QuestId::new(r).unwrap()
    }

    fn roundtrip(e: Event) {
        let r = pack_event(&e).expect("pack");
        let e2 = unpack_record(&r).expect("unpack");
        assert_eq!(e, e2, "event roundtrip mismatch: {:?}", e);
    }

    #[test]
    fn roundtrip_movement_events() {
        roundtrip(Event::AgentMoved {
            actor: aid(7),
            from: Vec3::new(1.5, -2.25, 3.0),
            location: Vec3::new(1.5, 0.0, 0.5),
            tick: 100,
        });
        roundtrip(Event::AgentFled {
            agent_id: aid(42),
            from: Vec3::new(0.0, 0.0, 0.0),
            to: Vec3::new(-5.0, -10.0, 0.5),
            tick: 5,
        });
    }

    #[test]
    fn roundtrip_combat_events() {
        roundtrip(Event::AgentAttacked {
            actor: aid(1),
            target: aid(2),
            damage: 12.75,
            tick: 0,
        });
        roundtrip(Event::AgentDied {
            agent_id: aid(3),
            tick: 4,
        });
        roundtrip(Event::OpportunityAttackTriggered {
            actor: aid(5),
            target: aid(6),
            tick: 1,
        });
        roundtrip(Event::EffectDamageApplied {
            actor: aid(1),
            target: aid(2),
            amount: 7.0,
            tick: 2,
        });
        roundtrip(Event::EffectStunApplied {
            actor: aid(1),
            target: aid(2),
            expires_at_tick: 50,
            tick: 2,
        });
        roundtrip(Event::EffectSlowApplied {
            actor: aid(1),
            target: aid(2),
            expires_at_tick: 10,
            factor_q8: -128,
            tick: 3,
        });
    }

    #[test]
    fn roundtrip_u64_fields() {
        // u64 fields must split/join losslessly across the full range.
        let big: u64 = 0xDEAD_BEEF_CAFE_BABE;
        roundtrip(Event::AgentCommunicated {
            speaker: aid(1),
            recipient: aid(2),
            fact_ref: big,
            tick: 9,
        });
        roundtrip(Event::AgentSharedStory {
            agent_id: aid(3),
            topic: u64::MAX,
            tick: 9,
        });
        roundtrip(Event::EffectGoldTransfer {
            from: aid(1),
            to: aid(2),
            amount: i32::MIN,
            tick: 1,
        });
        roundtrip(Event::EffectGoldTransfer {
            from: aid(1),
            to: aid(2),
            amount: i32::MAX,
            tick: 1,
        });
    }

    #[test]
    fn roundtrip_quest_events() {
        roundtrip(Event::QuestPosted {
            poster: aid(1),
            quest_id: qid(10),
            category: QuestCategory::Narrative,
            resolution: Resolution::Coalition { min_parties: 3 },
            tick: 5,
        });
        roundtrip(Event::QuestPosted {
            poster: aid(1),
            quest_id: qid(11),
            category: QuestCategory::Economic,
            resolution: Resolution::HighestBid,
            tick: 5,
        });
        roundtrip(Event::QuestAccepted {
            acceptor: aid(2),
            quest_id: qid(10),
            tick: 6,
        });
        roundtrip(Event::BidPlaced {
            bidder: aid(1),
            auction_id: qid(10),
            amount: 99.5,
            tick: 7,
        });
    }

    #[test]
    fn roundtrip_cast_events() {
        roundtrip(Event::AgentCast {
            actor: aid(1),
            ability: abid(7),
            target: aid(2),
            depth: 3,
            tick: 1,
        });
        roundtrip(Event::CastDepthExceeded {
            actor: aid(1),
            ability: abid(7),
            tick: 1,
        });
    }

    #[test]
    fn roundtrip_all_simple_events() {
        // Spot check of every remaining variant to catch a missing arm.
        roundtrip(Event::AgentAte { agent_id: aid(1), delta: 0.5, tick: 1 });
        roundtrip(Event::AgentDrank { agent_id: aid(1), delta: 0.5, tick: 1 });
        roundtrip(Event::AgentRested { agent_id: aid(1), delta: 0.5, tick: 1 });
        roundtrip(Event::AgentUsedItem { agent_id: aid(1), item_slot: 4, tick: 1 });
        roundtrip(Event::AgentHarvested {
            agent_id: aid(1),
            resource: 0x1234_5678_9ABC_DEF0,
            tick: 1,
        });
        roundtrip(Event::AgentPlacedTile {
            actor: aid(1),
            location: Vec3::new(1.0, 2.0, 3.0),
            kind_tag: 77,
            tick: 1,
        });
        roundtrip(Event::AgentPlacedVoxel {
            actor: aid(1),
            location: Vec3::new(1.0, 2.0, 3.0),
            mat_tag: 77,
            tick: 1,
        });
        roundtrip(Event::AgentHarvestedVoxel {
            actor: aid(1),
            location: Vec3::new(1.0, 2.0, 3.0),
            tick: 1,
        });
        roundtrip(Event::AgentConversed {
            agent_id: aid(1),
            partner: aid(2),
            tick: 1,
        });
        roundtrip(Event::InformationRequested {
            asker: aid(1),
            target: aid(2),
            query: 0xFF,
            tick: 1,
        });
        roundtrip(Event::AgentRemembered {
            agent_id: aid(1),
            subject: 0xAA,
            tick: 1,
        });
        roundtrip(Event::AnnounceEmitted {
            speaker: aid(1),
            audience_tag: 2,
            fact_payload: 0xBEEF,
            tick: 1,
        });
        roundtrip(Event::RecordMemory {
            observer: aid(1),
            source: aid(2),
            fact_payload: 0xCAFE,
            confidence: 0.75,
            tick: 1,
        });
        roundtrip(Event::EffectHealApplied {
            actor: aid(1),
            target: aid(2),
            amount: 5.0,
            tick: 1,
        });
        roundtrip(Event::EffectShieldApplied {
            actor: aid(1),
            target: aid(2),
            amount: 5.0,
            tick: 1,
        });
        roundtrip(Event::EffectStandingDelta {
            a: aid(1),
            b: aid(2),
            delta: -42,
            tick: 1,
        });
        roundtrip(Event::EngagementCommitted {
            actor: aid(1),
            target: aid(2),
            tick: 1,
        });
        roundtrip(Event::EngagementBroken {
            actor: aid(1),
            former_target: aid(2),
            reason: 3,
            tick: 1,
        });
        roundtrip(Event::FearSpread {
            observer: aid(1),
            dead_kin: aid(2),
            tick: 1,
        });
        roundtrip(Event::PackAssist {
            observer: aid(1),
            target: aid(2),
            tick: 1,
        });
        roundtrip(Event::RallyCall {
            observer: aid(1),
            wounded_kin: aid(2),
            tick: 1,
        });
    }

    #[test]
    fn chronicle_entry_not_packable() {
        // `ChronicleEntry` is filtered from the replayable hash path
        // and has no stable tag slot — pack_event should return None
        // so callers can skip it without swallowing a real event.
        let ce = Event::ChronicleEntry {
            template_id: 1,
            agent: aid(1),
            target: aid(2),
            tick: 0,
        };
        assert!(pack_event(&ce).is_none());
    }

    #[test]
    fn unknown_tag_yields_none() {
        let bad = EventRecord {
            kind: 999,
            tick: 0,
            payload: [0; PAYLOAD_WORDS],
        };
        assert!(unpack_record(&bad).is_none());
    }

    #[test]
    fn zero_agent_id_is_rejected() {
        // Slot-0 of every id-carrying variant must be a valid
        // NonZeroU32. A kernel that wrote 0 is a bug — unpack returns
        // None rather than silently materialising a bogus id.
        let bad = EventRecord {
            kind: EventKindTag::AgentAttacked.raw(),
            tick: 0,
            payload: [0; PAYLOAD_WORDS],
        };
        assert!(unpack_record(&bad).is_none());
    }

    #[test]
    fn record_size_matches_constant() {
        assert_eq!(std::mem::size_of::<EventRecord>() as u64, RECORD_BYTES);
    }

    #[test]
    fn max_payload_words_accommodates_largest_variant() {
        // AgentMoved / AgentFled: 1 id + 6 f32 = 7 words. Plus the
        // PAYLOAD_WORDS cushion gives 8 — document the headroom.
        assert!(PAYLOAD_WORDS >= 7, "PAYLOAD_WORDS too small for Vec3-carrying events");
    }

    #[test]
    fn wgsl_prefix_contains_cap() {
        let s = wgsl_prefix(1234);
        assert!(s.contains("EVENT_RING_CAP: u32 = 1234u"));
        assert!(s.contains(&format!("EVENT_RING_PAYLOAD_WORDS: u32 = {}u", PAYLOAD_WORDS)));
    }

    #[test]
    fn wgsl_source_declares_core_fn() {
        assert!(EVENT_RING_WGSL.contains("fn gpu_emit_event("));
        assert!(EVENT_RING_WGSL.contains("fn gpu_emit_agent_attacked("));
        assert!(EVENT_RING_WGSL.contains("atomicAdd(&event_ring_tail"));
    }

    #[test]
    fn chronicle_record_size_is_16_bytes() {
        assert_eq!(
            std::mem::size_of::<ChronicleRecord>() as u64,
            CHRONICLE_RECORD_BYTES
        );
    }

    #[test]
    fn chronicle_wgsl_prefix_contains_cap() {
        let s = chronicle_wgsl_prefix(4096);
        assert!(s.contains("CHRONICLE_RING_CAP: u32 = 4096u"));
    }

    #[test]
    fn chronicle_wgsl_source_declares_core_fn() {
        assert!(CHRONICLE_RING_WGSL.contains("fn gpu_emit_chronicle_event("));
        assert!(CHRONICLE_RING_WGSL.contains("atomicAdd(&chronicle_ring_tail"));
        assert!(CHRONICLE_RING_WGSL.contains("struct ChronicleRecord"));
    }

    #[test]
    fn chronicle_record_layout_matches_wgsl_order() {
        // Field order on both sides: template_id, agent, target, tick.
        // A mismatch would corrupt readback; bytemuck::Pod makes the
        // layout `#[repr(C)]` and validates field types, but it can't
        // check semantic ordering vs. WGSL. This test pins the order
        // by construction.
        let r = ChronicleRecord {
            template_id: 7,
            agent: 100,
            target: 200,
            tick: 42,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&r);
        // Little-endian u32 layout:
        //   0..4   = 7
        //   4..8   = 100
        //   8..12  = 200
        //   12..16 = 42
        assert_eq!(&bytes[0..4], &7u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &100u32.to_le_bytes());
        assert_eq!(&bytes[8..12], &200u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &42u32.to_le_bytes());
    }

    #[test]
    fn default_chronicle_capacity_is_1m() {
        assert_eq!(DEFAULT_CHRONICLE_CAPACITY, 1_000_000);
        // 1 M × 16 B = 16 MB — documented overhead for the
        // observability-only ring.
        let bytes = (DEFAULT_CHRONICLE_CAPACITY as u64) * CHRONICLE_RECORD_BYTES;
        assert_eq!(bytes, 16_000_000);
    }
}
