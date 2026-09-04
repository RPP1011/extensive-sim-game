//! Shared enum vocabulary referenced by compiler-emitted event structs.
//!
//! These enums appear as field types on `engine_rules::events::Event`
//! variants (e.g. `QuestPosted { category, resolution, .. }`), and
//! `engine_rules` doesn't depend on `engine` — so the types have to live
//! here. The old locations in `engine::policy::macro_kind` and
//! `engine::channel` / `engine::creature` re-export from this module so
//! existing `use engine::policy::QuestCategory` and `use engine::CommunicationChannel`
//! call sites keep compiling.
//!
//! `CommunicationChannel`, `ChannelSet`, and `LanguageId` are defined here
//! so the compiler-emitted `Capabilities` struct can reference them without
//! inverting the `engine → engine_rules` dependency direction.

use smallvec::SmallVec;
use std::num::NonZeroU16;

/// Auction resolution policy. Matches `dsl/spec.md` §9 D1.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Resolution {
    HighestBid,
    FirstAcceptable,
    MutualAgreement,
    Coalition { min_parties: u8 },
    Majority,
}

/// Quest category — the universal coarse bucket. Domain-specific kinds
/// (Hunt, Escort, Deliver, Charter, Marriage, …) register via the compiler's
/// `QuestType` extension table; the engine only knows these five.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum QuestCategory {
    Physical = 0,
    Political = 1,
    Personal = 2,
    Economic = 3,
    Narrative = 4,
}

/// First-class communication modality. `dsl/spec.md` §9 D30 enumerates the
/// six vocabulary channels an agent can transmit / receive on. The
/// `channel_range` helper (range per channel given vocal strength) lives in
/// `engine::channel` because the range is an engine primitive, not part of
/// the compiled rule surface.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[repr(u8)]
pub enum CommunicationChannel {
    Speech = 0,
    PackSignal = 1,
    Pheromone = 2,
    Song = 3,
    Telepathy = 4,
    Testimony = 5,
}

pub type ChannelSet = SmallVec<[CommunicationChannel; 4]>;

/// Niche-optimised language id. `Option<LanguageId>` is two bytes.
/// `dsl/spec.md` §56 commits to per-capability `languages`; the concrete
/// catalogue (Common, Draconic, Elven, ...) is a later-plan concern; two
/// built-in constants seed the MVP species defaults.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[repr(transparent)]
pub struct LanguageId(NonZeroU16);

impl LanguageId {
    pub const COMMON: LanguageId = LanguageId(NonZeroU16::new(1).unwrap());
    pub const DRACONIC: LanguageId = LanguageId(NonZeroU16::new(2).unwrap());

    #[inline]
    pub fn new(raw: u16) -> Option<Self> {
        NonZeroU16::new(raw).map(Self)
    }

    #[inline]
    pub fn raw(self) -> u16 {
        self.0.get()
    }
}
