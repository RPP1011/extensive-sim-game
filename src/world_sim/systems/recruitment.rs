#![allow(unused)]
//! Population growth — settlements with food surplus produce new NPCs.
//!
//! Uses entity pooling: dead NPCs are recycled via Heal + Move (revive + relocate)
//! instead of spawning new entities. No array growth, no index rebuilds.
//!
//! Growth rate scales with food availability. Starving settlements don't grow.
//! New NPCs are always level 1 — they gain power through mentorship and progression.
//!
//! Cadence: every 50 ticks (~5 game-seconds).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, WorldState};

const GROWTH_INTERVAL: u64 = 50;

/// Minimum food stockpile for reproduction to happen.
const MIN_FOOD_FOR_GROWTH: f32 = 10.0;

/// Food consumed per birth.
const FOOD_PER_BIRTH: f32 = 5.0;

/// Max births per settlement per growth tick.
const MAX_BIRTHS_PER_TICK: usize = 2;

/// Minimum alive NPCs at settlement for reproduction (need at least 2).
const MIN_POP_FOR_GROWTH: usize = 2;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_recruitment(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % GROWTH_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_recruitment_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

pub fn compute_recruitment_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % GROWTH_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // Check food surplus.
    if settlement.stockpile[0] < MIN_FOOD_FOR_GROWTH {
        return;
    }

    // Count alive NPCs at this settlement.
    let alive_count = entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .count();

    if alive_count < MIN_POP_FOR_GROWTH {
        return;
    }

    // Growth chance proportional to food surplus (diminishing returns).
    let food_ratio = (settlement.stockpile[0] / (alive_count as f32 * 10.0)).min(3.0);
    let growth_chance = 0.1 * food_ratio; // ~10-30% per growth tick with surplus

    let roll = tick_hash(state.tick, settlement_id as u64 ^ 0xB177);
    if roll > growth_chance {
        return;
    }

    // Find dead NPCs anywhere in the world to recycle.
    // Prefer dead NPCs that were originally from this settlement.
    let mut births = 0;
    for entity in &state.entities {
        if births >= MAX_BIRTHS_PER_TICK { break; }
        if entity.kind != EntityKind::Npc || entity.alive { continue; }

        // Check if we can afford the food cost.
        let food_remaining = settlement.stockpile[0] - (births as f32 * FOOD_PER_BIRTH);
        if food_remaining < FOOD_PER_BIRTH { break; }

        // Deterministic: only recycle specific dead NPCs per tick to avoid
        // all settlements grabbing the same corpse.
        let entity_roll = tick_hash(state.tick, entity.id as u64 ^ settlement_id as u64);
        if entity_roll > 0.05 { continue; } // ~5% of dead NPCs eligible per settlement per tick

        // Revive at settlement position (level 1, full HP).
        out.push(WorldDelta::Heal {
            target_id: entity.id,
            amount: entity.max_hp * 2.0, // overheal to guarantee full
            source_id: 0,
        });
        out.push(WorldDelta::Move {
            entity_id: entity.id,
            force: (settlement.pos.0 - entity.pos.0, settlement.pos.1 - entity.pos.1),
        });

        // Consume food for the birth.
        out.push(WorldDelta::ConsumeCommodity {
            location_id: settlement_id,
            commodity: crate::world_sim::commodity::FOOD,
            amount: FOOD_PER_BIRTH,
        });

        births += 1;
    }
}
