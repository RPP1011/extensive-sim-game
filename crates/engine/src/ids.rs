//! Niche-optimised ID newtypes — canonical public API for `engine`.
//!
//! **Plan B1' Task 3 status (transitional):** The struct definitions live in
//! `engine_data::ids` for now because adding `engine_data → engine` as a
//! regular Cargo dep (to let engine_data re-export from here) would create a
//! dep cycle: `engine → engine_data → engine`. The cycle cannot be broken in
//! this task because `engine/src/event/event_like_impl.rs` (Task 2 workaround)
//! requires `engine → engine_data` until Task 5 moves it to engine_data.
//!
//! The re-export below makes `engine::ids::*` the **canonical path** for all
//! callers, even though the struct bodies live one hop away in engine_data.
//! Task 5 will move the definitions here and invert the re-export direction.
//!
//! See `engine_data/src/ids.rs` for the struct declarations.

pub use engine_data::ids::{
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

    #[test]
    fn ability_id_slot_is_zero_based() {
        let a = AbilityId::new(1).unwrap();
        assert_eq!(a.slot(), 0);
        let b = AbilityId::new(42).unwrap();
        assert_eq!(b.slot(), 41);
    }
}
