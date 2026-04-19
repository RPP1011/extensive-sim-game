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
id_type!(ItemId);
id_type!(QuestId);
id_type!(AuctionId);
id_type!(InviteId);
id_type!(SettlementId);

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
}
