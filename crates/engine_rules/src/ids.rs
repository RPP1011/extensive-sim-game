//! Niche-optimised ID newtypes, shared across `engine` and `engine_rules`.
//!
//! Moved out of `engine::ids` at the milestone-2 integration step (the flip
//! that reversed the dep cycle: `engine` now depends on `engine_rules` so
//! compiler-emitted event structs can carry these types without cycling
//! back through the engine crate). `engine::ids` now re-exports from here,
//! so every existing `use engine::ids::AgentId` call site keeps compiling.
//!
//! Byte-for-byte parity with the old `engine::ids`: same `NonZeroU32` niche,
//! same `raw()`/`new()` APIs. When the compiler grows entity emission in a
//! later milestone, it will own these declarations directly.

use std::num::NonZeroU32;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
        #[repr(transparent)]
        pub struct $name(NonZeroU32);

        impl $name {
            #[inline]
            pub fn new(raw: u32) -> Option<Self> {
                NonZeroU32::new(raw).map(Self)
            }

            #[inline]
            pub fn raw(self) -> u32 {
                self.0.get()
            }
        }
    };
}

id_type!(AgentId);
id_type!(GroupId);
// `ItemId` — reserved for the item-entity subsystem plan (roadmap §Items).
id_type!(ItemId);
id_type!(QuestId);
// `AuctionId` — reserved for the auction/exchange plan (roadmap §Economy).
id_type!(AuctionId);
// `InviteId` — reserved for the invite/quest-coalition plan (roadmap §Social).
id_type!(InviteId);
// `SettlementId` — reserved for the settlement/region plan (roadmap §Regions).
id_type!(SettlementId);

/// `AbilityId` — a non-zero newtype identifying a registered `AbilityProgram`
/// inside an `AbilityRegistry`. Same niche shape as `AgentId`; `slot()`
/// returns a zero-based index for the registry's internal `Vec`.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[repr(transparent)]
pub struct AbilityId(NonZeroU32);

impl AbilityId {
    #[inline]
    pub fn new(raw: u32) -> Option<Self> {
        NonZeroU32::new(raw).map(Self)
    }

    #[inline]
    pub fn raw(self) -> u32 {
        self.0.get()
    }

    /// Zero-based slot suitable for indexing the registry's internal `Vec`.
    #[inline]
    pub fn slot(self) -> usize {
        (self.0.get() - 1) as usize
    }
}

/// Identifies an event within the `EventRing` by `(tick, seq)`. Assigned by
/// `EventRing::push` / `push_caused_by`; used as a sidecar cause pointer so
/// cascade fan-out can reconstruct causal trees without affecting the
/// replayable hash (see `EventRing::replayable_sha256`).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct EventId {
    pub tick: u32,
    pub seq:  u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_id_roundtrip_and_ordering() {
        let a = AgentId::new(5).unwrap();
        let b = AgentId::new(5).unwrap();
        let c = AgentId::new(6).unwrap();
        assert_eq!(a, b);
        assert!(a < c);
        assert_eq!(a.raw(), 5);
    }

    #[test]
    fn agent_id_zero_rejected() {
        assert!(AgentId::new(0).is_none());
    }

    #[test]
    fn size_of_option_matches_raw() {
        // NonZeroU32 niche optimisation gives us Option<AgentId> == u32 in size.
        assert_eq!(std::mem::size_of::<Option<AgentId>>(), 4);
    }

    #[test]
    fn ability_id_slot_is_zero_based() {
        let a = AbilityId::new(1).unwrap();
        assert_eq!(a.slot(), 0);
        let b = AbilityId::new(42).unwrap();
        assert_eq!(b.slot(), 41);
    }
}
