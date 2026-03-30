#![allow(unused)]
//! Supply line interdiction system — every 7 ticks.
//!
//! NPC entities with Trade intent traveling through hostile territory
//! suffer supply drain and potential interception. Friendly entities
//! patrolling trade routes protect nearby traders.
//!
//! Original: `crates/headless_campaign/src/systems/supply_lines.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EconomicIntent, EntityKind, WorldState, WorldTeam};
use crate::world_sim::state::{entity_hash_f32, pair_hash_f32};
use crate::world_sim::NUM_COMMODITIES;

/// Cadence: runs every 7 ticks.
const SUPPLY_LINE_TICK_INTERVAL: u64 = 7;

/// Distance (squared) at which hostile entities interdict traders.
const INTERDICTION_RANGE_SQ: f32 = 100.0; // 10 units

/// Supply drain multiplier when interdicted (extra drain on top of normal).
const INTERDICTION_DRAIN_MULT: f32 = 0.05;

/// Distance (squared) at which friendly entities protect traders.
const PATROL_RANGE_SQ: f32 = 64.0; // 8 units


pub fn compute_supply_lines(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % SUPPLY_LINE_TICK_INTERVAL != 0 {
        return;
    }

    // Collect hostile entity positions for interdiction checks.
    let hostiles: Vec<(u32, (f32, f32), f32)> = state
        .entities
        .iter()
        .filter(|e| e.alive && e.team == WorldTeam::Hostile)
        .map(|e| (e.id, e.pos, e.attack_damage))
        .collect();

    if hostiles.is_empty() {
        return;
    }

    // Collect friendly entity positions for patrol protection.
    let friendlies: Vec<(f32, f32)> = state
        .entities
        .iter()
        .filter(|e| {
            e.alive
                && e.team == WorldTeam::Friendly
                && e.kind == EntityKind::Npc
                // Patrols: NPCs that are idle near trade routes.
                && e.npc.as_ref().map_or(false, |n| {
                    matches!(n.economic_intent, EconomicIntent::Idle)
                })
        })
        .map(|e| e.pos)
        .collect();

    // Check each trading NPC for interdiction.
    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc || entity.team != WorldTeam::Friendly {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };
        let is_trading = matches!(npc.economic_intent, EconomicIntent::Trade { .. });
        if !is_trading {
            continue;
        }

        // Check if protected by a friendly patrol.
        let is_protected = friendlies.iter().any(|fp| {
            let dx = fp.0 - entity.pos.0;
            let dy = fp.1 - entity.pos.1;
            dx * dx + dy * dy < PATROL_RANGE_SQ
        });
        if is_protected {
            continue;
        }

        // Check for hostile interdiction.
        for &(hostile_id, hostile_pos, hostile_atk) in &hostiles {
            let dx = hostile_pos.0 - entity.pos.0;
            let dy = hostile_pos.1 - entity.pos.1;
            if dx * dx + dy * dy < INTERDICTION_RANGE_SQ {
                // Interdiction: drain goods from the trader.
                for c in 0..NUM_COMMODITIES {
                    let drain = npc.carried_goods[c] * INTERDICTION_DRAIN_MULT;
                    if drain > 0.001 {
                        out.push(WorldDelta::ConsumeCommodity {
                            location_id: entity.id, // entity acts as location for carried goods
                            commodity: c,
                            amount: drain,
                        });
                    }
                }

                // Also deal minor damage.
                let roll = pair_hash_f32(entity.id, hostile_id, state.tick, 0);
                if roll < 0.2 {
                    out.push(WorldDelta::Damage {
                        target_id: entity.id,
                        amount: hostile_atk * 0.3,
                        source_id: hostile_id,
                    });
                }

                break; // Only one interdiction per trader per tick.
            }
        }
    }
}
