//! `AbilityId` — a non-zero newtype identifying a registered `AbilityProgram`
//! inside an `AbilityRegistry`. Mirrors the `AgentId` / `GroupId` pattern in
//! `crate::ids`: `NonZeroU32` niche so `Option<AbilityId>` packs to 4 bytes.
//!
//! IDs are monotonic and slot-stable: registering the Nth program yields
//! `AbilityId::new(N as u32)`. Lookup reads `registry[id.slot()]`.

use std::num::NonZeroU32;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[repr(transparent)]
pub struct AbilityId(NonZeroU32);

impl AbilityId {
    /// Construct from a raw id; returns `None` on zero (reserved as niche).
    #[inline]
    pub fn new(raw: u32) -> Option<Self> {
        NonZeroU32::new(raw).map(Self)
    }

    /// Raw `u32` id. `AbilityId::new(raw).unwrap().raw() == raw` for `raw > 0`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ability_id_round_trips_through_new() {
        let a = AbilityId::new(1).unwrap();
        assert_eq!(a.raw(), 1);
        assert_eq!(a.slot(), 0);
        let b = AbilityId::new(42).unwrap();
        assert_eq!(b.raw(), 42);
        assert_eq!(b.slot(), 41);
    }

    #[test]
    fn ability_id_rejects_zero() {
        assert!(AbilityId::new(0).is_none());
    }

    #[test]
    fn option_ability_id_niche_optimized() {
        // Niche optimisation: Option<AbilityId> packs to 4 bytes.
        assert_eq!(std::mem::size_of::<Option<AbilityId>>(), 4);
    }
}
