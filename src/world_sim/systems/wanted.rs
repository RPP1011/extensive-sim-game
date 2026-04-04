//! Wanted poster system — every 10 ticks.
//!
//! Settlements issue bounties for hostile entities (monsters or enemy faction
//! NPCs) near their grid. When an NPC kills a wanted target, they receive a
//! gold reward via TransferGold. Bounties are tracked via RecordEvent and the
//! existing quest_board/world_events.
//!
//! **Gold conservation:** Bounty payouts are funded by the nearest settlement
//! treasury. If the settlement cannot afford the bounty, no gold is paid.
//!
//! Original: `crates/headless_campaign/src/systems/wanted.rs`
//! Cadence: every 10 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{
    ChronicleCategory, DiplomaticStance, EntityKind, SettlementField,
    WorldEvent, WorldState, WorldTeam,
};
use crate::world_sim::state::pair_hash_f32;

/// Cadence gate.
const WANTED_TICK_INTERVAL: u64 = 10;

/// Range (squared) within which a settlement can identify wanted targets.
const WANTED_RANGE_SQ: f32 = 900.0; // 30 units

/// Base bounty gold per monster level.
const MONSTER_BOUNTY_PER_LEVEL: f32 = 15.0;

/// Base bounty gold for hostile faction NPCs.
const HOSTILE_NPC_BOUNTY_BASE: f32 = 40.0;

/// Minimum threat level for a settlement to issue wanted posters.
const MIN_THREAT_FOR_WANTED: f32 = 0.15;

/// Max wanted events emitted per settlement per tick (prevent spam).
const MAX_WANTED_PER_SETTLEMENT: usize = 3;

/// Bounty reward multiplier for high-level targets.
const HIGH_LEVEL_BONUS_THRESHOLD: u32 = 5;
const HIGH_LEVEL_BONUS_MULT: f32 = 1.5;


pub fn compute_wanted(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % WANTED_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Build hostile faction set.
    let hostile_faction_ids: Vec<u32> = state
        .factions
        .iter()
        .filter(|f| {
            matches!(
                f.diplomatic_stance,
                DiplomaticStance::Hostile | DiplomaticStance::AtWar
            )
        })
        .map(|f| f.id)
        .collect();

    // --- Phase 1: Issue wanted posters for hostile entities near settlements ---
    for settlement in &state.settlements {
        if settlement.threat_level < MIN_THREAT_FOR_WANTED {
            continue;
        }

        let mut posted_count = 0usize;

        for entity in &state.entities {
            if posted_count >= MAX_WANTED_PER_SETTLEMENT {
                break;
            }
            if !entity.alive {
                continue;
            }

            // Check range to settlement.
            let dx = entity.pos.0 - settlement.pos.0;
            let dy = entity.pos.1 - settlement.pos.1;
            if dx * dx + dy * dy > WANTED_RANGE_SQ {
                continue;
            }

            // Determine if this entity is a valid wanted target.
            let (is_wanted, bounty_gold) = match entity.kind {
                EntityKind::Monster => {
                    let base = MONSTER_BOUNTY_PER_LEVEL * entity.level.max(1) as f32;
                    let mult = if entity.level >= HIGH_LEVEL_BONUS_THRESHOLD {
                        HIGH_LEVEL_BONUS_MULT
                    } else {
                        1.0
                    };
                    (true, base * mult)
                }
                EntityKind::Npc => {
                    // Hostile faction NPCs are wanted.
                    let is_hostile_npc = entity
                        .npc
                        .as_ref()
                        .and_then(|n| n.faction_id)
                        .map(|fid| hostile_faction_ids.contains(&fid))
                        .unwrap_or(false);
                    if is_hostile_npc {
                        let base = HOSTILE_NPC_BOUNTY_BASE + entity.level as f32 * 10.0;
                        (true, base)
                    } else {
                        (false, 0.0)
                    }
                }
                _ => (false, 0.0),
            };

            if !is_wanted {
                continue;
            }

            // Use deterministic roll to gate poster issuance (not every hostile
            // gets a poster every tick).
            let roll = pair_hash_f32(settlement.id, entity.id, state.tick, 0);
            if roll > 0.4 {
                continue;
            }

            // Record the wanted poster as a world event.
            out.push(WorldDelta::RecordEvent {
                event: WorldEvent::Generic {
                    category: ChronicleCategory::Quest,
                    text: format!(
                        "WANTED: {} (entity {}, level {}) near {} — bounty {} gold",
                        if entity.kind == EntityKind::Monster {
                            "monster"
                        } else {
                            "hostile agent"
                        },
                        entity.id,
                        entity.level,
                        settlement.name,
                        bounty_gold as u32
                    ),
                },
            });

            posted_count += 1;
        }
    }

    // --- Phase 2: Reward NPCs near dead wanted targets ---
    // When a hostile entity has died on a grid near a settlement, reward
    // friendly NPCs on the same grid as bounty collectors.
    // Bounties are funded by the nearest settlement treasury.
    for grid in &state.fidelity_zones {
        // Find dead hostiles on this grid.
        let dead_hostiles: Vec<(u32, u32)> = grid
            .entity_ids
            .iter()
            .filter_map(|&eid| state.entity(eid))
            .filter(|e| !e.alive && (e.kind == EntityKind::Monster || {
                e.npc
                    .as_ref()
                    .and_then(|n| n.faction_id)
                    .map(|fid| hostile_faction_ids.contains(&fid))
                    .unwrap_or(false)
            }))
            .map(|e| (e.id, e.level))
            .collect();

        if dead_hostiles.is_empty() {
            continue;
        }

        // Find friendly NPCs on this grid who can collect the bounty.
        let collectors: Vec<u32> = grid
            .entity_ids
            .iter()
            .filter_map(|&eid| state.entity(eid))
            .filter(|e| e.alive && e.kind == EntityKind::Npc && e.team == WorldTeam::Friendly)
            .map(|e| e.id)
            .collect();

        if collectors.is_empty() {
            continue;
        }

        // Find the nearest settlement to fund the bounty.
        let grid_center = grid
            .entity_ids
            .first()
            .and_then(|&eid| state.entity(eid))
            .map(|e| e.pos)
            .unwrap_or((0.0, 0.0));

        let nearest_settlement = state.settlements.iter().min_by(|a, b| {
            let da = (a.pos.0 - grid_center.0).powi(2) + (a.pos.1 - grid_center.1).powi(2);
            let db = (b.pos.0 - grid_center.0).powi(2) + (b.pos.1 - grid_center.1).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        let funding_settlement = match nearest_settlement {
            Some(s) => s,
            None => continue,
        };

        for (_dead_id, dead_level) in &dead_hostiles {
            let bounty = MONSTER_BOUNTY_PER_LEVEL * (*dead_level).max(1) as f32;

            // Settlement must be able to afford the bounty
            if funding_settlement.treasury <= bounty {
                continue;
            }

            let share = bounty / collectors.len() as f32;

            for &collector_id in &collectors {
                out.push(WorldDelta::TransferGold {
                    from_entity: funding_settlement.id,
                    to_entity: collector_id,
                    amount: share,
                });
            }
        }
    }

    // --- Phase 3: High-threat settlements reduce threat when wanted targets die ---
    for settlement in &state.settlements {
        if settlement.threat_level < MIN_THREAT_FOR_WANTED {
            continue;
        }

        // Count dead hostiles near this settlement as a proxy for bounty fulfillment.
        let dead_nearby = state
            .entities
            .iter()
            .filter(|e| {
                !e.alive
                    && (e.kind == EntityKind::Monster
                        || e.npc
                            .as_ref()
                            .and_then(|n| n.faction_id)
                            .map(|fid| hostile_faction_ids.contains(&fid))
                            .unwrap_or(false))
            })
            .filter(|e| {
                let dx = e.pos.0 - settlement.pos.0;
                let dy = e.pos.1 - settlement.pos.1;
                dx * dx + dy * dy <= WANTED_RANGE_SQ
            })
            .count();

        if dead_nearby > 0 {
            // Each dead hostile reduces settlement threat slightly.
            let threat_reduction = -(dead_nearby as f32 * 0.02).min(settlement.threat_level);
            out.push(WorldDelta::UpdateSettlementField {
                settlement_id: settlement.id,
                field: SettlementField::ThreatLevel,
                value: threat_reduction,
            });
        }
    }
}
