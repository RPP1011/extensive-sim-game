#![allow(unused)]
//! Resource gathering and crafting system — every 7 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/crafting.rs`.
//! Resource nodes in controlled regions auto-harvest materials into
//! settlement stockpiles. When sufficient materials are available,
//! the system auto-crafts by consuming commodities and producing
//! finished goods (represented as commodity transforms).
//!
//!   (id, region_id, resource_type, amount, regen_rate, max_amount)

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{ActionTags, WorldState, tags};
use crate::world_sim::NUM_COMMODITIES;

/// How often crafting ticks.
const CRAFTING_INTERVAL: u64 = 7;

/// Commodity indices used as crafting inputs.
/// By convention: 0=food, 1=ore, 2=wood, 3=hide, 4=herbs, 5=crystal, 6=obsidian, 7=luxury.
const CRAFT_INPUT_A: usize = 1; // ore
const CRAFT_INPUT_B: usize = 2; // wood
const CRAFT_OUTPUT: usize = 7;  // luxury (finished goods)

/// Minimum input stockpile to trigger crafting.
const MIN_INPUT_STOCKPILE: f32 = 5.0;

/// Amount consumed per craft batch.
const CRAFT_CONSUME_AMOUNT: f32 = 2.0;

/// Amount produced per craft batch.
const CRAFT_PRODUCE_AMOUNT: f32 = 1.0;

/// Resource regeneration rate per tick (stockpile natural growth).
const REGEN_RATE: f32 = 0.02;

/// Compute crafting deltas.
///
/// Each settlement with sufficient raw material stockpile (ore + wood)
/// consumes inputs and produces finished goods. Settlements also see
/// natural regeneration of commodity stockpiles.
pub fn compute_crafting(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CRAFTING_INTERVAL != 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_crafting_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_crafting_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    _entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % CRAFTING_INTERVAL != 0 {
        return;
    }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // --- Natural resource regeneration ---
    for c in 0..NUM_COMMODITIES {
        let regen = REGEN_RATE * (settlement.population as f32 / 100.0).max(0.1);
        out.push(WorldDelta::ProduceCommodity {
            location_id: settlement_id,
            commodity: c,
            amount: regen,
        });
    }

    // --- Auto-craft: consume raw materials, produce finished goods ---
    let has_input_a = settlement.stockpile[CRAFT_INPUT_A] >= MIN_INPUT_STOCKPILE;
    let has_input_b = settlement.stockpile[CRAFT_INPUT_B] >= MIN_INPUT_STOCKPILE;

    if has_input_a && has_input_b {
        out.push(WorldDelta::ConsumeCommodity {
            location_id: settlement_id,
            commodity: CRAFT_INPUT_A,
            amount: CRAFT_CONSUME_AMOUNT,
        });
        out.push(WorldDelta::ConsumeCommodity {
            location_id: settlement_id,
            commodity: CRAFT_INPUT_B,
            amount: CRAFT_CONSUME_AMOUNT,
        });
        out.push(WorldDelta::ProduceCommodity {
            location_id: settlement_id,
            commodity: CRAFT_OUTPUT,
            amount: CRAFT_PRODUCE_AMOUNT,
        });
        out.push(WorldDelta::UpdateTreasury {
            location_id: settlement_id,
            delta: CRAFT_PRODUCE_AMOUNT * 0.5,
        });

        // Behavior tags: NPCs at this settlement earn crafting/smithing.
        let range = state.group_index.settlement_entities(settlement_id);
        for entity in &state.entities[range] {
            if entity.alive && entity.kind == crate::world_sim::state::EntityKind::Npc {
                let mut action = ActionTags::empty();
                action.add(tags::CRAFTING, 1.0);
                action.add(tags::SMITHING, 0.5);
                let action = crate::world_sim::action_context::with_context(&action, entity, state);
                out.push(WorldDelta::AddBehaviorTags { entity_id: entity.id, tags: action.tags, count: action.count });
                break; // Tag one NPC per craft batch.
            }
        }
    }
}
