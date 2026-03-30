#![allow(unused)]
//! Underground dungeon network system — every 10 ticks.
//!
//! Manages dungeon monster regeneration, loot regeneration, and threat
//! emergence from deep dungeons. Dungeon exploration is resolved when
//! friendly NPCs enter a dungeon grid, emitting Damage/TransferGold deltas.
//!
//! **Gold conservation:** Dungeon loot gold is paid from the settlement
//! treasury as a bounty reward. If the settlement cannot afford it, no gold
//! is paid.
//!
//! Ported from `crates/headless_campaign/src/systems/dungeons.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::fidelity::Fidelity;
use crate::world_sim::state::{EntityKind, WorldState, WorldTeam};

/// Cadence: runs every 10 ticks.
const DUNGEON_INTERVAL: u64 = 10;

/// Monster strength regeneration per interval.
const MONSTER_REGEN_RATE: f32 = 1.0;

/// Loot regeneration per interval.
const LOOT_REGEN_RATE: f32 = 0.1;

/// Deep dungeon depth threshold for spawning crisis threats.
const DEEP_DUNGEON_DEPTH: u32 = 4;

/// Threat multiplier for deep dungeon threats.
const DEEP_DUNGEON_THREAT_MULTIPLIER: f32 = 1.5;

/// Gold reward per loot point from dungeon exploration.
const GOLD_PER_LOOT: f32 = 3.0;

pub fn compute_dungeons(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % DUNGEON_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_dungeons_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
///
/// Handles dungeon threat pressure for this settlement and exploration on its grid.
pub fn compute_dungeons_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    _entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % DUNGEON_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // --- Dungeon monster threat pressure ---
    for region in &state.regions {
        if region.monster_density > 60.0 && settlement.treasury > -100.0 {
            let threat_drain = region.monster_density * 0.01 * DEEP_DUNGEON_THREAT_MULTIPLIER;
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement_id,
                delta: -threat_drain / state.settlements.len().max(1) as f32,
            });
        }

        if region.monster_density > 40.0 {
            if let Some(grid_id) = settlement.grid_id {
                let current_fidelity = state
                    .grid(grid_id)
                    .map(|g| g.fidelity)
                    .unwrap_or(Fidelity::Low);

                if matches!(current_fidelity, Fidelity::Background | Fidelity::Low) {
                    out.push(WorldDelta::EscalateFidelity {
                        grid_id,
                        new_fidelity: Fidelity::Medium,
                    });
                }
            }
        }
    }

    // --- Dungeon exploration by NPCs on this settlement's grid ---
    let grid_id = match settlement.grid_id {
        Some(gid) => gid,
        None => return,
    };
    let grid = match state.grid(grid_id) {
        Some(g) => g,
        None => return,
    };

    if grid.fidelity != Fidelity::High {
        return;
    }

    let friendlies: Vec<&crate::world_sim::state::Entity> = grid
        .entity_ids
        .iter()
        .filter_map(|&eid| state.entity(eid))
        .filter(|e| e.kind == EntityKind::Npc && e.alive && e.team == WorldTeam::Friendly)
        .collect();

    if friendlies.is_empty() {
        return;
    }

    let dead_monster_levels: f32 = grid
        .entity_ids
        .iter()
        .filter_map(|&eid| state.entity(eid))
        .filter(|e| e.kind == EntityKind::Monster && !e.alive)
        .map(|e| e.level as f32)
        .sum();

    if dead_monster_levels <= 0.0 {
        return;
    }

    let loot_gold = dead_monster_levels * GOLD_PER_LOOT;
    let gold_each = loot_gold / friendlies.len() as f32;

    // Dungeon loot is paid from settlement treasury as bounty reward
    let can_afford = settlement.treasury > loot_gold;

    for friendly in &friendlies {
        if gold_each > 0.0 && can_afford {
            out.push(WorldDelta::TransferGold {
                from_id: settlement_id,
                to_id: friendly.id,
                amount: gold_each,
            });
        }

        let hazard_damage = 2.0;
        out.push(WorldDelta::Damage {
            target_id: friendly.id,
            amount: hazard_damage,
            source_id: friendly.id,
        });
    }
}
