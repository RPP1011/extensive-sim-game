//! ID vocabulary — owned by `engine_rules` as of milestone 2's integration
//! step. This module re-exports the compiler-output types so existing
//! `use engine::ids::AgentId` (etc.) call sites keep compiling.
//!
//! See `crates/engine_rules/src/ids.rs` for the canonical definitions.

pub use engine_rules::ids::{
    AbilityId, AgentId, AuctionId, EventId, GroupId, InviteId, ItemId, QuestId, SettlementId,
};

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
