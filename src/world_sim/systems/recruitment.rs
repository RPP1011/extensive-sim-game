#![allow(unused)]
//! Population growth — settlements with food surplus produce new NPCs.
//!
//! Uses entity pooling: dead NPCs are recycled via Heal + Move.
//! Growth rate scales with food availability and settlement safety.
//! New NPCs are level 1.
//!
//! Cadence: every 20 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, WorldState};
use crate::world_sim::commodity;

const GROWTH_INTERVAL: u64 = 20;

/// Minimum food stockpile for growth. Low bar — even scarce food allows slow growth.
const MIN_FOOD_FOR_GROWTH: f32 = 1.0;

/// Food consumed per birth (about 10 meals worth).
const FOOD_PER_BIRTH: f32 = 0.3;

/// Max births per settlement per growth tick.
const MAX_BIRTHS_PER_TICK: usize = 3;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_recruitment(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % GROWTH_INTERVAL != 0 || state.tick == 0 { return; }

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
    if state.tick % GROWTH_INTERVAL != 0 || state.tick == 0 { return; }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // Need food to grow.
    if settlement.stockpile[commodity::FOOD] < MIN_FOOD_FOR_GROWTH {
        return;
    }

    // Count alive NPCs. Even 1 NPC can recruit (immigration, not just birth).
    let alive_count = entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .count();

    // Growth chance: higher with more food, lower with more population (carrying capacity).
    // food_ratio > 1.0 means surplus, < 1.0 means scarce.
    let food_per_capita = settlement.stockpile[commodity::FOOD] / (alive_count as f32 + 1.0);
    let growth_chance = (food_per_capita * 0.1).clamp(0.02, 0.5);

    // Safety bonus: safe settlements attract more immigrants.
    let safety_bonus = if settlement.threat_level < 10.0 { 0.1 } else { 0.0 };

    let roll = tick_hash(state.tick, settlement_id as u64 ^ 0xB177);
    if roll > growth_chance + safety_bonus {
        return;
    }

    // Find dead NPCs to recycle. Use group_index unaffiliated range first,
    // then scan settlement's own dead.
    let mut births = 0;

    // First: recycle dead NPCs at this settlement.
    for entity in entities {
        if births >= MAX_BIRTHS_PER_TICK { break; }
        if entity.kind != EntityKind::Npc || entity.alive { continue; }

        let food_remaining = settlement.stockpile[commodity::FOOD] - (births as f32 * FOOD_PER_BIRTH);
        if food_remaining < FOOD_PER_BIRTH { break; }

        // Deterministic per-entity per-settlement per-tick.
        let entity_roll = tick_hash(state.tick, entity.id as u64 ^ settlement_id as u64 ^ 0xDEAD);
        if entity_roll > 0.15 { continue; } // ~15% eligibility

        revive_npc(entity.id, settlement, out);
        births += 1;
    }

    // Second: if we still have capacity, recycle from unaffiliated dead.
    if births < MAX_BIRTHS_PER_TICK {
        let unaffiliated = state.group_index.unaffiliated_entities();
        for entity in &state.entities[unaffiliated] {
            if births >= MAX_BIRTHS_PER_TICK { break; }
            if entity.kind != EntityKind::Npc || entity.alive { continue; }

            let food_remaining = settlement.stockpile[commodity::FOOD] - (births as f32 * FOOD_PER_BIRTH);
            if food_remaining < FOOD_PER_BIRTH { break; }

            let entity_roll = tick_hash(state.tick, entity.id as u64 ^ settlement_id as u64 ^ 0xBEEF);
            if entity_roll > 0.08 { continue; }

            revive_npc(entity.id, settlement, out);
            births += 1;
        }
    }

    // Consume food for births.
    if births > 0 {
        out.push(WorldDelta::ConsumeCommodity {
            location_id: settlement_id,
            commodity: commodity::FOOD,
            amount: FOOD_PER_BIRTH * births as f32,
        });
    }
}

fn revive_npc(
    entity_id: u32,
    settlement: &crate::world_sim::state::SettlementState,
    out: &mut Vec<WorldDelta>,
) {
    // Heal to full (revive).
    out.push(WorldDelta::Heal {
        target_id: entity_id,
        amount: 200.0, // overheal to guarantee full HP regardless of max_hp
        source_id: 0,
    });
    // Teleport to settlement.
    out.push(WorldDelta::Move {
        entity_id,
        force: (settlement.pos.0, settlement.pos.1), // absolute position as force (one-tick teleport)
    });
    // Set to producing.
    out.push(WorldDelta::SetIntent {
        entity_id,
        intent: crate::world_sim::state::EconomicIntent::Produce,
    });
}
