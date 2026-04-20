//! Shared enum vocabulary referenced by compiler-emitted event structs.
//!
//! Moved out of `engine::policy::macro_kind` at the milestone-2 integration
//! step. These enums appear as field types on `engine_rules::events::Event`
//! variants (e.g. `QuestPosted { category, resolution, .. }`), and
//! `engine_rules` doesn't depend on `engine` — so the types have to live
//! here. The old locations in `engine::policy::macro_kind` re-export from
//! this module so existing `use engine::policy::QuestCategory` call sites
//! keep compiling.
//!
//! When the compiler grows entity / enum declaration emission in a later
//! milestone, these definitions become compiler output.

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
    Physical  = 0,
    Political = 1,
    Personal  = 2,
    Economic  = 3,
    Narrative = 4,
}
