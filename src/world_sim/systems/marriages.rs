#![allow(unused)]
//! Diplomatic marriage system — delta architecture port.
//!
//! Adventurers marry faction nobles for alliance bonuses, dowries, and heirs.
//! Marriages lock adventurer loyalty to the faction and create ongoing relation
//! bonuses, but loyalty crises arise when factions go to war with the guild.
//!
//! Original: `crates/headless_campaign/src/systems/marriages.rs`
//! Cadence: every 17 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::state::entity_hash_f32;

// NEEDS STATE: marriages: Vec<Marriage> on WorldState
//   Marriage { id, adventurer_id, faction_id, noble_name, married_tick, relation_bonus, dowry_received, produces_heir }
// NEEDS STATE: factions: Vec<FactionState> on WorldState (relationship_to_guild, diplomatic_stance, military_strength)
// NEEDS STATE: adventurer morale, loyalty, faction_id, level, status on Entity/NpcData
// NEEDS DELTA: UpdateRelation { faction_id, delta: f32 }
// NEEDS DELTA: AdjustMorale { entity_id, delta: f32 }
// NEEDS DELTA: AdjustLoyalty { entity_id, delta: f32 }
// NEEDS DELTA: SpawnEntity (for heirs)
// NEEDS DELTA: RemoveMarriage { marriage_id }

/// Maximum active marriages the guild can maintain.
const MAX_ACTIVE_MARRIAGES: usize = 3;

/// Ongoing relation bonus per tick cycle (family ties).
const ONGOING_RELATION_BONUS: f32 = 2.0;

/// Ticks after marriage before heir chance activates.
const HEIR_GESTATION_TICKS: u64 = 67;

/// Heir chance per tick cycle after gestation (10%).
const HEIR_CHANCE: f32 = 0.10;

/// Morale penalty when spouse faction is at war with guild.
const WAR_MORALE_PENALTY: f32 = 15.0;

/// Morale penalty when spouse dies (faction destroyed).
const SPOUSE_DEATH_MORALE_PENALTY: f32 = 30.0;

/// Cadence gate (ticks).
const MARRIAGE_TICK_INTERVAL: u64 = 17;

/// Compute marriage deltas: ongoing relation bonuses, loyalty crises, spouse death.
///
/// The original system mutated CampaignState directly (factions, adventurers,
/// marriages). In the delta architecture, most mutations require new delta
/// variants (UpdateRelation, AdjustMorale, RemoveMarriage, SpawnEntity).
/// This function is a structural placeholder that documents the logic and
/// emits available deltas where possible.
pub fn compute_marriages(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % MARRIAGE_TICK_INTERVAL != 0 {
        return;
    }

    // NEEDS STATE: iterate state.marriages
    // For each active marriage:
    //
    // 1. Ongoing relation bonus:
    //    out.push(WorldDelta::UpdateRelation { faction_id: marriage.faction_id, delta: ONGOING_RELATION_BONUS });
    //
    // 2. Heir generation (ticks_married >= HEIR_GESTATION_TICKS && !produces_heir):
    //    Deterministic roll: hash(tick, marriage.id) < HEIR_CHANCE
    //    out.push(WorldDelta::SpawnEntity { ... }); // high-stat heir
    //
    // 3. Loyalty crisis (faction at war with guild):
    //    Deterministic roll for desertion (30%):
    //      out.push(WorldDelta::Die { entity_id: marriage.adventurer_id }); // deserted
    //      out.push(WorldDelta::RemoveMarriage { marriage_id });
    //    Otherwise:
    //      out.push(WorldDelta::AdjustMorale { entity_id: marriage.adventurer_id, delta: -WAR_MORALE_PENALTY });
    //
    // 4. Spouse death (faction military_strength <= 0):
    //    out.push(WorldDelta::AdjustMorale { entity_id: marriage.adventurer_id, delta: -SPOUSE_DEATH_MORALE_PENALTY });
    //    out.push(WorldDelta::RemoveMarriage { marriage_id });

    // Since WorldState lacks marriage/faction storage, no real deltas can be emitted yet.
    // The economic cost of arranging marriages can be expressed as TransferGold
    // if/when the arrange_marriage action is ported:
    //
    // out.push(WorldDelta::TransferGold {
    //     from_id: guild_entity_id,
    //     to_id: faction_entity_id,
    //     amount: dowry,
    // });
}

// ---------------------------------------------------------------------------
// Deterministic RNG helper
// ---------------------------------------------------------------------------

