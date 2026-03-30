#![allow(unused)]
//! NPC migration between settlements — fires every 200 ticks.
//!
//! NPCs at settlements with high threat (> 30) or very low food (stockpile < 5)
//! consider migrating to a safer/richer settlement. They evaluate settlements
//! they know about (from price_knowledge), pick the one with lowest threat +
//! highest food, and set intent to Travel there. Only ~5% of eligible NPCs
//! migrate per cycle to keep it rare.
//!
//! Original: `crates/headless_campaign/src/systems/migration.rs`

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::naming::entity_display_name;
use crate::world_sim::state::{ChronicleCategory, ChronicleEntry, EntityKind, NpcData, SettlementState, WorldState};
use crate::world_sim::state::{entity_hash_f32, pair_hash_f32};

use super::seasons::{current_season, season_modifiers};

/// How often migration checks run (in ticks).
const MIGRATION_INTERVAL: u64 = 200;

/// Threat level above which NPCs consider fleeing.
const THREAT_THRESHOLD: f32 = 30.0;

/// Food stockpile below which NPCs consider fleeing.
const FOOD_THRESHOLD: f32 = 5.0;

/// Fraction of eligible NPCs that migrate per cycle (~5%).
const MIGRATION_CHANCE: f32 = 0.05;

/// Maximum distance an NPC will consider migrating (squared, for efficiency).
const MAX_MIGRATION_DIST_SQ: f32 = 10000.0; // 100 units


/// Compute an attractiveness score for a settlement.
/// Higher = more attractive to migrants.
fn settlement_attractiveness(
    settlement: &SettlementState,
    regions: &[crate::world_sim::state::RegionState],
) -> f32 {
    // Base score from economic health.
    let treasury_score = (settlement.treasury / 100.0).clamp(0.0, 30.0);

    // Stockpile diversity: how many commodities have non-trivial stock.
    let stockpile_score = settlement.stockpile.iter().filter(|&&s| s > 1.0).count() as f32 * 5.0;

    // Threat penalty: high regional threat reduces attractiveness.
    let threat_penalty = regions
        .iter()
        .map(|r| r.threat_level)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0)
        * 0.5;

    // Population pressure: overcrowding reduces attractiveness.
    let pop_penalty = if settlement.population > 500 {
        (settlement.population - 500) as f32 * 0.05
    } else {
        0.0
    };

    (treasury_score + stockpile_score - threat_penalty - pop_penalty).clamp(0.0, 100.0)
}

pub fn compute_migration(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % MIGRATION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.settlements.len() < 2 {
        return; // Need at least 2 settlements for migration.
    }

    use crate::world_sim::commodity;

    // For each settlement, check if conditions are bad enough to trigger migration.
    for settlement in &state.settlements {
        let high_threat = settlement.threat_level > THREAT_THRESHOLD;
        let low_food = settlement.stockpile[commodity::FOOD] < FOOD_THRESHOLD;

        if !high_threat && !low_food {
            continue; // Settlement is fine, no migration pressure.
        }

        // Find best destination: lowest threat + highest food, excluding current.
        let mut best_id: Option<u32> = None;
        let mut best_score = f32::NEG_INFINITY;
        let mut best_pos = (0.0f32, 0.0f32);

        for candidate in &state.settlements {
            if candidate.id == settlement.id {
                continue;
            }

            // Distance check.
            let dx = candidate.pos.0 - settlement.pos.0;
            let dy = candidate.pos.1 - settlement.pos.1;
            if dx * dx + dy * dy > MAX_MIGRATION_DIST_SQ {
                continue;
            }

            // Score: higher food is better, lower threat is better.
            let score = candidate.stockpile[commodity::FOOD] - candidate.threat_level;
            if score > best_score {
                best_score = score;
                best_id = Some(candidate.id);
                best_pos = candidate.pos;
            }
        }

        let target_id = match best_id {
            Some(id) => id,
            None => continue, // No viable destination.
        };

        // Iterate NPCs at this settlement.
        let range = state.group_index.settlement_entities(settlement.id);
        for entity in &state.entities[range] {
            if !entity.alive || entity.kind != EntityKind::Npc {
                continue;
            }

            let npc = match &entity.npc {
                Some(n) => n,
                None => continue,
            };

            // NPC must know about the target settlement (from price_knowledge).
            let knows_target = npc
                .price_knowledge
                .iter()
                .any(|pr| pr.settlement_id == target_id);
            if !knows_target {
                continue;
            }

            // ~5% chance per eligible NPC per cycle (deterministic hash).
            let roll = pair_hash_f32(entity.id, settlement.id, state.tick, 0xBEEF_CAFE);
            if roll >= MIGRATION_CHANCE {
                continue;
            }

            // Emit SetIntent to Travel toward the target settlement.
            out.push(WorldDelta::SetIntent {
                entity_id: entity.id,
                intent: crate::world_sim::state::EconomicIntent::Travel {
                    destination: best_pos,
                },
            });

            // Chronicle: notable NPCs (level >= 20) fleeing.
            if entity.level >= 20 {
                let npc_name = entity_display_name(entity);
                let target_name = state.settlement(target_id)
                    .map(|s| s.name.as_str())
                    .unwrap_or("unknown");
                let reason = if high_threat { "rising threat" } else { "famine" };
                out.push(WorldDelta::RecordChronicle {
                    entry: ChronicleEntry {
                        tick: state.tick,
                        category: ChronicleCategory::Narrative,
                        text: format!("{} fled {} due to {}, heading for {}",
                            npc_name, settlement.name, reason, target_name),
                        entity_ids: vec![entity.id],
                    },
                });
            }
        }
    }
}
