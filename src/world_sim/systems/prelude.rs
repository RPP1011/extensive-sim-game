//! Common imports for world sim systems.
//!
//! Usage: `use super::prelude::*;` at the top of each system file.

pub use crate::world_sim::delta::WorldDelta;
pub use crate::world_sim::state::{
    Entity, EntityKind, WorldState, WorldTeam, NpcData, EconomicIntent,
    SettlementState, RegionState, FactionState, DiplomaticStance,
    StatusEffect, StatusEffectKind,
    ActionTags, ClassSlot, PriceReport,
    EntityField, FactionField, RegionField, SettlementField,
    tags, tag,
    entity_hash, entity_hash_f32, pair_hash_f32,
};
pub use crate::world_sim::fidelity::Fidelity;
pub use crate::world_sim::commodity;
pub use crate::world_sim::{DT_SEC, NUM_COMMODITIES};
