//! Signal tower system — every 7 ticks.
//!
//! Signal towers are modeled as Building entities. Operational towers boost
//! scouting (price report sharing) in their vicinity. Battles near towers
//! can damage them, and enemy entities can compromise them.
//!
//! Original: `crates/headless_campaign/src/systems/signal_towers.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, PriceReport, WorldState, WorldTeam};
use crate::world_sim::state::{entity_hash_f32};

/// Cadence: runs every 7 ticks.
const SIGNAL_TOWER_INTERVAL: u64 = 7;

/// Range (squared) at which a tower provides scouting coverage.
const TOWER_RANGE_SQ: f32 = 400.0; // 20 units

/// Chance that a battle near a tower damages it.
const BATTLE_DAMAGE_CHANCE: f32 = 0.30;


pub fn compute_signal_towers(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % SIGNAL_TOWER_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // In the delta architecture, towers are Building entities.
    // Operational buildings (alive, hp > 0) near settlements provide scouting
    // coverage by sharing price reports between NPCs.

    let buildings: Vec<(u32, (f32, f32), f32)> = state
        .entities
        .iter()
        .filter(|e| e.alive && e.kind == EntityKind::Building)
        .map(|e| (e.id, e.pos, e.hp))
        .collect();

    if buildings.is_empty() {
        return;
    }

    // --- Battle damage: hostile entities near towers can damage them ---
    for &(building_id, building_pos, building_hp) in &buildings {
        if building_hp <= 0.0 {
            continue;
        }

        // Count hostile entities near this building.
        let hostile_count = state
            .entities
            .iter()
            .filter(|e| {
                e.alive && e.team == WorldTeam::Hostile && {
                    let dx = e.pos.0 - building_pos.0;
                    let dy = e.pos.1 - building_pos.1;
                    dx * dx + dy * dy < TOWER_RANGE_SQ
                }
            })
            .count();

        if hostile_count > 0 {
            let roll = entity_hash_f32(0, state.tick, 0x70AE_DA46u64.wrapping_add(building_id as u64));
            if roll < BATTLE_DAMAGE_CHANCE {
                out.push(WorldDelta::Damage {
                    target_id: building_id,
                    amount: 20.0,
                    source_id: 0,
                });
            }
        }
    }

    // --- Scouting coverage: operational towers share price reports ---
    // Towers cause NPCs in their vicinity to receive price reports from
    // nearby settlements, simulating improved information flow.
    for &(building_id, building_pos, building_hp) in &buildings {
        if building_hp <= 0.0 {
            continue;
        }

        // Find settlements within tower range.
        let nearby_settlements: Vec<(u32, [f32; crate::world_sim::NUM_COMMODITIES])> = state
            .settlements
            .iter()
            .filter(|s| {
                let dx = s.pos.0 - building_pos.0;
                let dy = s.pos.1 - building_pos.1;
                dx * dx + dy * dy < TOWER_RANGE_SQ
            })
            .map(|s| (s.id, s.prices))
            .collect();

        if nearby_settlements.is_empty() {
            continue;
        }

        // Share reports with NPCs near the tower.
        for entity in &state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc || entity.team != WorldTeam::Friendly
            {
                continue;
            }
            let dx = entity.pos.0 - building_pos.0;
            let dy = entity.pos.1 - building_pos.1;
            if dx * dx + dy * dy > TOWER_RANGE_SQ {
                continue;
            }

            for &(settlement_id, prices) in &nearby_settlements {
                out.push(WorldDelta::SharePriceReport {
                    from_id: building_id,
                    to_id: entity.id,
                    report: PriceReport {
                        settlement_id,
                        prices,
                        tick_observed: state.tick,
                    },
                });
            }
        }
    }
}
