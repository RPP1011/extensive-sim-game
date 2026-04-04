//! Unified world simulation with order-invariant delta architecture.
//!
//! Every tick follows three phases:
//! 1. **Snapshot**: Freeze the entire world state (immutable reference)
//! 2. **Compute**: Every entity reads the snapshot, computes its delta independently
//! 3. **Merge**: All deltas combine into the next world state (commutative + associative)

pub mod building_ai;
pub mod city_grid;
pub mod voxel;
pub mod sdf;
pub mod delta;
pub mod state;
pub mod tick;
pub mod apply;
pub mod fidelity;
pub mod compute_high;
pub mod compute_medium;
pub mod compute_low;
pub mod spatial;
pub mod action_context;
pub mod ability_gen;
pub mod ability_quality;
pub mod class_dsl;
pub mod class_gen;
pub mod grammar_space;
pub mod interior_gen;
pub mod naming;
pub mod runtime;
pub mod systems;
pub mod trace;
pub mod visualizer;

pub use delta::{WorldDelta, MergedDeltas, merge_deltas};
pub use state::{
    WorldState, Entity, HotEntity, ColdEntity, EntityKind, GroupIndex, LocalGrid, SettlementState, RegionState, EconomyState,
    Terrain, SettlementSpecialty,
    // Campaign system types
    FactionState, DiplomaticStance, GuildState, Quest, QuestPosting, QuestType, QuestStatus,
    ChronicleEntry, ChronicleCategory, WorldEvent,
    EntityField, FactionField, RegionField, SettlementField, RelationKind, QuestDelta,
    // Tag-based action system
    ActionTags, ClassSlot, Equipment, tag, EconomicIntent,
    // Resource nodes
    ResourceType, ResourceData,
};
pub use state::tags;
pub use tick::{tick, tick_par, tick_profiled, TickProfile, ProfileAccumulator};
pub use fidelity::Fidelity;

/// Number of commodity types in the economy.
pub const NUM_COMMODITIES: usize = 8;

/// Tick duration in seconds (100ms fixed tick).
pub const DT_SEC: f32 = 0.1;

/// Named commodity indices for the 8-slot stockpile/price arrays.
pub mod commodity {
    pub const FOOD: usize = 0;
    pub const IRON: usize = 1;
    pub const WOOD: usize = 2;
    pub const HERBS: usize = 3;
    pub const HIDE: usize = 4;
    pub const CRYSTAL: usize = 5;
    pub const EQUIPMENT: usize = 6;
    pub const MEDICINE: usize = 7;
}
