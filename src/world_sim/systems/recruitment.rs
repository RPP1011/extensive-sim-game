//! Population growth — settlements with food surplus produce new NPCs.
//!
//! Uses entity pooling: dead NPCs are recycled via Heal + Move.
//! Growth rate scales with food availability and settlement safety.
//! New NPCs are level 1.
//!
//! Cadence: every 10 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, WorldState};
use crate::world_sim::commodity;

const GROWTH_INTERVAL: u64 = 10;

/// Minimum food stockpile for growth. Low bar — even scarce food allows slow growth.
const MIN_FOOD_FOR_GROWTH: f32 = 0.5;

/// Food consumed per birth.
const FOOD_PER_BIRTH: f32 = 0.3;

/// Max births per settlement per growth tick.
const MAX_BIRTHS_PER_TICK: usize = 8;

/// Target population per settlement. Above this, growth slows exponentially.
const CARRYING_CAPACITY: f32 = 500.0;


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

    let alive_count = entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .count();

    // Proportional growth: number of births scales with food surplus and population headroom.
    // No binary gate — always try to birth at least 1 if food available.
    let food_surplus = settlement.stockpile[commodity::FOOD] - MIN_FOOD_FOR_GROWTH;
    let pop_ratio = alive_count as f32 / CARRYING_CAPACITY;
    // Logistic growth: fast when pop is low, slows as it approaches carrying capacity.
    let growth_factor = (1.0 - pop_ratio).max(0.05);
    // How many births food can support this tick.
    let food_births = (food_surplus / FOOD_PER_BIRTH) as usize;
    // Scale by growth factor: at 50% capacity, allow ~50% of max births.
    let target_births = ((MAX_BIRTHS_PER_TICK as f32 * growth_factor) as usize)
        .max(1) // always try at least 1
        .min(food_births)
        .min(MAX_BIRTHS_PER_TICK);

    if target_births == 0 { return; }

    // Find dead NPCs to recycle.
    let mut births = 0;

    // First: recycle dead NPCs at this settlement.
    for entity in entities {
        if births >= target_births { break; }
        if entity.kind != EntityKind::Npc || entity.alive { continue; }

        revive_npc(entity.id, settlement, out);
        births += 1;
    }

    // Second: if we still have capacity, recycle from unaffiliated dead.
    if births < target_births {
        let unaffiliated = state.group_index.unaffiliated_entities();
        for entity in &state.entities[unaffiliated] {
            if births >= target_births { break; }
            if entity.kind != EntityKind::Npc || entity.alive { continue; }

            revive_npc(entity.id, settlement, out);
            births += 1;
        }
    }

    // Consume food for births.
    if births > 0 {
        out.push(WorldDelta::ConsumeCommodity {
            settlement_id: settlement_id,
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
    out.push(WorldDelta::SetPos {
        entity_id,
        pos: settlement.pos,
    });
    // Set to producing.
    out.push(WorldDelta::SetIntent {
        entity_id,
        intent: crate::world_sim::state::EconomicIntent::Produce,
    });
}
