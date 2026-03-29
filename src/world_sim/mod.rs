//! Unified world simulation with order-invariant delta architecture.
//!
//! Every tick follows three phases:
//! 1. **Snapshot**: Freeze the entire world state (immutable reference)
//! 2. **Compute**: Every entity reads the snapshot, computes its delta independently
//! 3. **Merge**: All deltas combine into the next world state (commutative + associative)

pub mod delta;
pub mod state;
pub mod tick;
pub mod apply;
pub mod fidelity;
pub mod compute_high;
pub mod compute_medium;
pub mod compute_low;
pub mod bridge;
pub mod spatial;
pub mod action_context;
pub mod class_gen;
pub mod runtime;
pub mod systems;

pub use delta::{WorldDelta, MergedDeltas, merge_deltas};
pub use state::{
    WorldState, Entity, HotEntity, ColdEntity, EntityKind, GroupIndex, LocalGrid, SettlementState, RegionState, EconomyState,
    Terrain, SettlementSpecialty,
    // Campaign system types
    FactionState, DiplomaticStance, GuildState, Quest, QuestPosting, QuestType, QuestStatus,
    ChronicleEntry, ChronicleCategory, WorldEvent,
    EntityField, FactionField, RegionField, SettlementField, RelationKind, QuestDelta,
    // Tag-based action system
    ActionTags, ClassSlot, tag,
};
pub use state::tags;
pub use tick::{tick, tick_par, tick_profiled, TickProfile, ProfileAccumulator};
pub use fidelity::Fidelity;

/// Number of commodity types in the economy.
pub const NUM_COMMODITIES: usize = 8;
