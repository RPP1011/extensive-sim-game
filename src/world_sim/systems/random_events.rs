#![allow(unused)]
//! Random world events — fires every 17 ticks.
//!
//! Unpredictable events that affect the world state: treasure discoveries
//! (treasury boosts), bandit raids (gold/goods drain), plagues (entity
//! damage), harvest bounties (commodity production), etc.
//!
//! Original: `crates/headless_campaign/src/systems/random_events.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, StatusEffect, StatusEffectKind, WorldState, WorldTeam};
use crate::world_sim::state::{entity_hash_f32};
use crate::world_sim::NUM_COMMODITIES;

/// How often to roll for a random event (in ticks).
const EVENT_INTERVAL: u64 = 17;

/// Base probability of an event firing each roll (15%).
const BASE_CHANCE: f32 = 0.15;


pub fn compute_random_events(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % EVENT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let roll = entity_hash_f32(0, state.tick, 0x8A3D_EE01);
    if roll > BASE_CHANCE {
        return;
    }

    // Pick event type from weighted pool.
    let event_roll = entity_hash_f32(0, state.tick, 0x8A3D_EE02);

    if event_roll < 0.12 {
        // --- Treasure Discovery: gold boost to random settlement ---
        if !state.settlements.is_empty() {
            let idx = (state.tick as usize) % state.settlements.len();
            let amount = 30.0 + entity_hash_f32(0, state.tick, 0x601D_0001u64.wrapping_add(0xA)) * 70.0;
            out.push(WorldDelta::UpdateTreasury {
                location_id: state.settlements[idx].id,
                delta: amount,
            });
        }
    } else if event_roll < 0.24 {
        // --- Harvest Bounty: food production boost ---
        for settlement in &state.settlements {
            let amount = 5.0 + entity_hash_f32(settlement.id, state.tick, 0xF00D) * 10.0;
            out.push(WorldDelta::ProduceCommodity {
                location_id: settlement.id,
                commodity: crate::world_sim::commodity::FOOD, // food
                amount,
            });
        }
    } else if event_roll < 0.36 {
        // --- Bandit Raid: gold drain from settlement ---
        if !state.settlements.is_empty() {
            let idx = (state.tick as usize + 3) % state.settlements.len();
            let loss = 10.0 + entity_hash_f32(0, state.tick, 0xBA3D_1700) * 30.0;
            out.push(WorldDelta::UpdateTreasury {
                location_id: state.settlements[idx].id,
                delta: -loss,
            });
            // Drain some goods too.
            for c in 0..NUM_COMMODITIES {
                let goods_loss = state.settlements[idx].stockpile[c] * 0.05;
                if goods_loss > 0.001 {
                    out.push(WorldDelta::ConsumeCommodity {
                        location_id: state.settlements[idx].id,
                        commodity: c,
                        amount: goods_loss,
                    });
                }
            }
        }
    } else if event_roll < 0.48 {
        // --- Plague: damage all friendly NPCs ---
        for entity in &state.entities {
            if entity.alive && entity.kind == EntityKind::Npc && entity.team == WorldTeam::Friendly
            {
                out.push(WorldDelta::Damage {
                    target_id: entity.id,
                    amount: 5.0,
                    source_id: 0,
                });
                // Apply a DoT for ongoing plague effects.
                out.push(WorldDelta::ApplyStatus {
                    target_id: entity.id,
                    status: StatusEffect {
                        kind: StatusEffectKind::Dot {
                            damage_per_tick: 1.0,
                            tick_interval_ms: 2000,
                            tick_elapsed_ms: 0,
                        },
                        source_id: 0,
                        remaining_ms: 10000,
                    },
                });
            }
        }
    } else if event_roll < 0.58 {
        // --- Famine Scare: drain food from all settlements ---
        for settlement in &state.settlements {
            let drain = settlement.stockpile[0] * 0.15;
            if drain > 0.001 {
                out.push(WorldDelta::ConsumeCommodity {
                    location_id: settlement.id,
                    commodity: crate::world_sim::commodity::FOOD,
                    amount: drain,
                });
            }
        }
    } else if event_roll < 0.68 {
        // --- Faction Gift: treasury boost from friendly settlement ---
        if !state.settlements.is_empty() {
            let idx = (state.tick as usize + 7) % state.settlements.len();
            let gift = 15.0 + entity_hash_f32(0, state.tick, 0x61F7_0001) * 25.0;
            out.push(WorldDelta::UpdateTreasury {
                location_id: state.settlements[idx].id,
                delta: gift,
            });
        }
    } else if event_roll < 0.78 {
        // --- Equipment Breakage (proxy): damage a random friendly NPC ---
        let friendlies: Vec<u32> = state
            .entities
            .iter()
            .filter(|e| e.alive && e.kind == EntityKind::Npc && e.team == WorldTeam::Friendly)
            .map(|e| e.id)
            .collect();
        if !friendlies.is_empty() {
            let idx = (state.tick as usize) % friendlies.len();
            out.push(WorldDelta::Damage {
                target_id: friendlies[idx],
                amount: 3.0,
                source_id: 0,
            });
        }
    } else if event_roll < 0.88 {
        // --- Prophecy of Doom: debuff friendly NPCs (morale proxy) ---
        for entity in &state.entities {
            if entity.alive && entity.kind == EntityKind::Npc && entity.team == WorldTeam::Friendly
            {
                out.push(WorldDelta::ApplyStatus {
                    target_id: entity.id,
                    status: StatusEffect {
                        kind: StatusEffectKind::Debuff {
                            stat: "morale".to_string(),
                            factor: 0.85,
                        },
                        source_id: 0,
                        remaining_ms: 15000,
                    },
                });
            }
        }
    } else {
        // --- Mercenary Band: heal friendly NPCs (reinforcement proxy) ---
        for entity in &state.entities {
            if entity.alive
                && entity.kind == EntityKind::Npc
                && entity.team == WorldTeam::Friendly
                && entity.hp < entity.max_hp
            {
                let heal = (entity.max_hp - entity.hp) * 0.2;
                out.push(WorldDelta::Heal {
                    target_id: entity.id,
                    amount: heal,
                    source_id: 0,
                });
            }
        }
    }
}
