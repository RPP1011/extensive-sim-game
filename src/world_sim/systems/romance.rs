#![allow(unused)]
//! Romance system — NPC pair bonding at settlements.
//!
//! NPCs at the same settlement with complementary behavior profiles
//! develop romantic connections. Since we don't store explicit romance
//! state, affinity is computed each tick from behavior profiles and
//! proximity. Strong affinity produces morale boosts and cooperation tags.
//!
//! Cadence: every 50 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;

const ROMANCE_INTERVAL: u64 = 50;

/// Maximum pairs to evaluate per settlement (avoid O(n²) explosion).
const MAX_NPCS_PER_SETTLEMENT: usize = 32;

/// Maximum romantic pairs to process per settlement per tick.
const MAX_PAIRS: usize = 3;

/// Minimum affinity score for a romantic bond.
const AFFINITY_THRESHOLD: f32 = 0.5;


pub fn compute_romance(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % ROMANCE_INTERVAL != 0 || state.tick == 0 { return; }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_romance_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

pub fn compute_romance_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % ROMANCE_INTERVAL != 0 || state.tick == 0 { return; }

    // Find social gathering spots (Tavern/Temple/Market/Inn).
    let mut social_buildings: [(f32, f32); 16] = [(0.0, 0.0); 16];
    let mut sb_count = 0usize;
    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Building { continue; }
        if let Some(bd) = &entity.building {
            if bd.construction_progress >= 1.0
                && matches!(bd.building_type,
                    BuildingType::Inn | BuildingType::Temple
                    | BuildingType::Market | BuildingType::GuildHall)
            {
                if sb_count < 16 {
                    social_buildings[sb_count] = entity.pos;
                    sb_count += 1;
                }
            }
        }
    }

    // Collect alive NPC indices near social buildings, morale > 30.
    let mut npc_indices: [usize; 32] = [0; 32];
    let mut count = 0;

    for (idx, entity) in entities.iter().enumerate() {
        if count >= MAX_NPCS_PER_SETTLEMENT { break; }
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };
        if npc.morale < 30.0 { continue; }

        // Must be near a social gathering spot (within 15 units).
        let near_social = sb_count == 0 || (0..sb_count).any(|k| {
            let dx = entity.pos.0 - social_buildings[k].0;
            let dy = entity.pos.1 - social_buildings[k].1;
            dx * dx + dy * dy < 225.0 // 15^2
        });
        if !near_social { continue; }

        npc_indices[count] = idx;
        count += 1;
    }

    if count < 2 { return; }

    let mut pairs_found = 0;

    for i in 0..count {
        if pairs_found >= MAX_PAIRS { break; }
        for j in (i + 1)..count {
            if pairs_found >= MAX_PAIRS { break; }

            let entity_a = &entities[npc_indices[i]];
            let entity_b = &entities[npc_indices[j]];
            let id_a = entity_a.id;
            let id_b = entity_b.id;
            let npc_a = entity_a.npc.as_ref().unwrap();
            let npc_b = entity_b.npc.as_ref().unwrap();

            // Deterministic pair check: only evaluate certain pairs per tick.
            let roll = pair_hash_f32(id_a, id_b, state.tick, 0xE04A);
            if roll > 0.15 { continue; } // only 15% of pairs checked per tick

            // Compute affinity from behavior profile cosine similarity.
            let affinity = compute_shared_affinity(npc_a, npc_b);

            if affinity < AFFINITY_THRESHOLD { continue; }

            // Romantic bond effects: mutual morale boost.
            let morale_boost = (affinity - AFFINITY_THRESHOLD) * 2.0;
            out.push(WorldDelta::UpdateEntityField {
                entity_id: id_a,
                field: EntityField::Morale,
                value: morale_boost.min(3.0),
            });
            out.push(WorldDelta::UpdateEntityField {
                entity_id: id_b,
                field: EntityField::Morale,
                value: morale_boost.min(3.0),
            });

            // Cooperation tags: both get social behavior.
            let mut action = ActionTags::empty();
            action.add(tags::DIPLOMACY, 0.3);
            action.add(tags::RESILIENCE, 0.2);
            out.push(WorldDelta::AddBehaviorTags {
                entity_id: id_a,
                tags: action.tags,
                count: action.count,
            });
            out.push(WorldDelta::AddBehaviorTags {
                entity_id: id_b,
                tags: action.tags,
                count: action.count,
            });

            pairs_found += 1;
        }
    }

    // Chronicle: strong romantic bonds.
    if pairs_found > 0 && state.tick % 500 == 0 {
        let settlement_name = state.settlement(settlement_id)
            .map(|s| s.name.as_str())
            .unwrap_or("an unknown settlement");
        out.push(WorldDelta::RecordChronicle {
            entry: ChronicleEntry {
                tick: state.tick,
                category: ChronicleCategory::Narrative,
                text: format!("{} romantic bonds formed at {}",
                    pairs_found, settlement_name),
                entity_ids: vec![],
            },
        });
    }
}

/// Compute shared affinity between two NPCs based on overlapping behavior tags.
fn compute_shared_affinity(a: &NpcData, b: &NpcData) -> f32 {
    // Compare top 5 tags from each NPC.
    // Shared tags weighted by geometric mean of their values.
    let common_tags = [
        tags::COMBAT, tags::TRADE, tags::FAITH, tags::RESEARCH,
        tags::FARMING, tags::MINING, tags::LEADERSHIP, tags::DIPLOMACY,
    ];

    let mut dot = 0.0f32;
    let mut mag_a = 0.0f32;
    let mut mag_b = 0.0f32;

    for &tag in &common_tags {
        let va = a.behavior_value(tag);
        let vb = b.behavior_value(tag);
        dot += va * vb;
        mag_a += va * va;
        mag_b += vb * vb;
    }

    let denom = (mag_a.sqrt() * mag_b.sqrt()).max(1.0);
    dot / denom // cosine similarity, 0..1
}
