#![allow(unused)]
//! Funeral and memorial system — delta architecture port.
//!
//! When NPCs die, the guild can hold funerals that affect morale, bonds,
//! and create lasting memorials. Pending funerals auto-resolve as
//! SimpleFunerals after 17 ticks if not acted upon.
//!
//! Original: `crates/headless_campaign/src/systems/memorials.rs`
//! Cadence: every 10 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState};

// NEEDS STATE: pending_funerals: Vec<FuneralPending> on WorldState
//   FuneralPending { adventurer_id, name, level, death_tick }
// NEEDS STATE: memorials: Vec<Memorial> on WorldState
//   Memorial { id, adventurer_name, adventurer_id, memorial_type, created_tick, morale_bonus, description }
//   MemorialType: SimpleFuneral, HerosFuneral, Monument, NamedBuilding, LegendaryTale
// NEEDS STATE: adventurer_bonds for grief comfort bonuses
// NEEDS STATE: guild gold, reputation
// NEEDS DELTA: CreateMemorial { entity_id, memorial_type }
// NEEDS DELTA: ExpireMemorial { memorial_id }
// NEEDS DELTA: AdjustMorale { entity_id, delta }
// NEEDS DELTA: AdjustReputation { delta }

/// Cadence gate.
const MEMORIAL_TICK_INTERVAL: u64 = 10;

/// Auto-resolve pending funerals after this many ticks.
const AUTO_FUNERAL_DELAY: u64 = 17;

/// Maximum memorials kept.
const MAX_MEMORIALS: usize = 5;

/// Morale duration for temporary funeral effects (ticks).
const TEMP_MORALE_DURATION: u64 = 17;

/// Gold costs by memorial type.
const COST_HEROS_FUNERAL: f32 = 30.0;
const COST_MONUMENT: f32 = 100.0;
const COST_NAMED_BUILDING: f32 = 50.0;

/// Compute memorial deltas: detect deaths, auto-resolve funerals, apply morale.
///
/// Death detection uses entity alive status. Funeral costs can be expressed
/// via UpdateTreasury. Morale effects need per-entity AdjustMorale deltas.
pub fn compute_memorials(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % MEMORIAL_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Detect dead NPC entities ---
    let mut dead_npcs: Vec<u32> = Vec::new();
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        for entity in &state.entities[range] {
            if !entity.alive && entity.kind == EntityKind::Npc {
                dead_npcs.push(entity.id);
            }
        }
    }

    // NEEDS STATE: check if dead NPCs already have pending funerals or memorials
    // For new deaths:
    //   Add to pending_funerals (NEEDS DELTA: CreatePendingFuneral)

    // --- Auto-resolve stale pending funerals ---
    // NEEDS STATE: for each pending funeral where tick >= death_tick + AUTO_FUNERAL_DELAY:
    //   Create SimpleFuneral memorial
    //   out.push(WorldDelta::CreateMemorial { entity_id, SimpleFuneral })

    // --- Apply ongoing memorial morale effects ---
    // NEEDS STATE: for each active memorial:
    //   Permanent memorials (Monument, NamedBuilding): apply every tick
    //   Temporary memorials: expire after TEMP_MORALE_DURATION
    //
    //   For all living NPC entities:
    //     effective_bonus = base_bonus + bond_strength * 0.05
    //     per_tick = effective_bonus / MEMORIAL_TICK_INTERVAL
    //     out.push(WorldDelta::AdjustMorale { entity_id, delta: per_tick })

    // --- Funeral cost when HoldFuneral action is processed ---
    // HerosFuneral: 30 gold
    //   out.push(WorldDelta::UpdateTreasury { location_id: guild_settlement, delta: -30.0 })
    // Monument: 100 gold
    //   out.push(WorldDelta::UpdateTreasury { location_id: guild_settlement, delta: -100.0 })
    // NamedBuilding: 50 gold
    //   out.push(WorldDelta::UpdateTreasury { location_id: guild_settlement, delta: -50.0 })

    let _ = dead_npcs;
}
