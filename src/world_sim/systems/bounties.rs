#![allow(unused)]
//! Bounty board system — every 10 ticks.
//!
//! Tracks high-value targets (elite monsters, faction leaders) via RecordEvent.
//! Bounty completion rewards gold via TransferGold. High-threat regions
//! generate implicit bounty pressure via UpdateTreasury funding.
//!
//! **Gold conservation:** Bounty rewards are paid from the nearest settlement
//! treasury. If the settlement cannot afford it, no gold is paid.
//!
//! Ported from `crates/headless_campaign/src/systems/bounties.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{
    ChronicleCategory, DiplomaticStance, EntityField, EntityKind, FactionField,
    WorldEvent, WorldState, WorldTeam,
};
use crate::world_sim::state::pair_hash_f32;

/// How often the bounty system ticks (in world sim ticks).
const BOUNTY_INTERVAL: u64 = 10;

/// Monster level threshold for "high-value" bounty target.
const HIGH_VALUE_LEVEL_THRESHOLD: u32 = 5;

/// Gold reward per level for high-value monster kills.
const HV_BOUNTY_PER_LEVEL: f32 = 25.0;

/// Gold reward for killing a hostile faction NPC (scales with level).
const FACTION_NPC_BOUNTY_BASE: f32 = 50.0;

/// XP reward for bounty completion.
const BOUNTY_XP_REWARD: u32 = 15;

/// Threat threshold for regions to generate bounty funding.
const REGION_THREAT_FUNDING_THRESHOLD: f32 = 50.0;

/// Funding per unit of threat above threshold.
const THREAT_FUNDING_RATE: f32 = 0.05;

/// Maximum bounty events per tick (prevent log spam).
const MAX_BOUNTY_EVENTS_PER_TICK: usize = 5;


pub fn compute_bounties(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % BOUNTY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

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

    let mut bounty_events = 0usize;

    // --- Phase 1: Auto-complete bounties for dead high-value targets on grids ---
    for grid in &state.grids {
        let dead_hv_targets: Vec<&crate::world_sim::state::Entity> = grid
            .entity_ids
            .iter()
            .filter_map(|&eid| state.entity(eid))
            .filter(|e| !e.alive && is_high_value_target(e, &hostile_faction_ids))
            .collect();

        if dead_hv_targets.is_empty() {
            continue;
        }

        let friendlies: Vec<&crate::world_sim::state::Entity> = grid
            .entity_ids
            .iter()
            .filter_map(|&eid| state.entity(eid))
            .filter(|e| e.kind == EntityKind::Npc && e.alive && e.team == WorldTeam::Friendly)
            .collect();

        if friendlies.is_empty() {
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

        // Bounty reward per dead high-value target, split among friendlies.
        for target in &dead_hv_targets {
            let bounty_gold = compute_bounty_reward(target, &hostile_faction_ids);

            // Settlement must be able to afford the bounty
            if funding_settlement.treasury <= bounty_gold {
                continue;
            }

            let gold_each = bounty_gold / friendlies.len() as f32;

            for friendly in &friendlies {
                if gold_each > 0.0 {
                    out.push(WorldDelta::TransferGold {
                        from_id: funding_settlement.id,
                        to_id: friendly.id,
                        amount: gold_each,
                    });
                }

            }

            // Record bounty completion event.
            if bounty_events < MAX_BOUNTY_EVENTS_PER_TICK {
                out.push(WorldDelta::RecordEvent {
                    event: WorldEvent::Generic {
                        category: ChronicleCategory::Quest,
                        text: format!(
                            "BOUNTY COMPLETED: {} (entity {}, level {}) slain — {} gold distributed",
                            if target.kind == EntityKind::Monster {
                                "elite monster"
                            } else {
                                "high-value target"
                            },
                            target.id,
                            target.level,
                            bounty_gold as u32
                        ),
                    },
                });
                bounty_events += 1;
            }

            // If the target was a hostile faction NPC, weaken their faction.
            if let Some(fid) = target
                .npc
                .as_ref()
                .and_then(|n| n.faction_id)
                .filter(|fid| hostile_faction_ids.contains(fid))
            {
                out.push(WorldDelta::UpdateFaction {
                    faction_id: fid,
                    field: FactionField::MilitaryStrength,
                    value: -(target.level as f32 * 2.0),
                });
            }
        }
    }

    // --- Phase 2: Post new bounty notices for high-value targets near settlements ---
    for settlement in &state.settlements {
        for entity in &state.entities {
            if bounty_events >= MAX_BOUNTY_EVENTS_PER_TICK {
                break;
            }
            if !entity.alive || !is_high_value_target(entity, &hostile_faction_ids) {
                continue;
            }

            let dx = entity.pos.0 - settlement.pos.0;
            let dy = entity.pos.1 - settlement.pos.1;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq > 900.0 {
                continue; // Beyond 30-unit range.
            }

            // Stagger posting: not every target gets posted every tick.
            let roll = pair_hash_f32(settlement.id, entity.id, state.tick, 0);
            if roll > 0.25 {
                continue;
            }

            let bounty_gold = compute_bounty_reward(entity, &hostile_faction_ids);

            out.push(WorldDelta::RecordEvent {
                event: WorldEvent::Generic {
                    category: ChronicleCategory::Quest,
                    text: format!(
                        "BOUNTY POSTED at {}: {} (entity {}, level {}) — {} gold reward",
                        settlement.name,
                        if entity.kind == EntityKind::Monster {
                            "elite monster"
                        } else {
                            "high-value target"
                        },
                        entity.id,
                        entity.level,
                        bounty_gold as u32
                    ),
                },
            });
            bounty_events += 1;
        }
    }

    // --- Phase 3: High-threat regions generate implicit bounty funding ---
    for region in &state.regions {
        if region.threat_level > REGION_THREAT_FUNDING_THRESHOLD {
            // Fund nearest settlement treasury as bounty incentive.
            if let Some(settlement) = state.settlements.first() {
                let funding = region.threat_level * THREAT_FUNDING_RATE;
                out.push(WorldDelta::UpdateTreasury {
                    location_id: settlement.id,
                    delta: funding,
                });
            }
        }
    }
}

/// Check if an entity qualifies as a high-value bounty target.
fn is_high_value_target(
    entity: &crate::world_sim::state::Entity,
    hostile_faction_ids: &[u32],
) -> bool {
    match entity.kind {
        EntityKind::Monster => entity.level >= HIGH_VALUE_LEVEL_THRESHOLD,
        EntityKind::Npc => entity
            .npc
            .as_ref()
            .and_then(|n| n.faction_id)
            .map(|fid| hostile_faction_ids.contains(&fid))
            .unwrap_or(false)
            && entity.level >= 3,
        _ => false,
    }
}

/// Compute gold reward for a bounty target.
fn compute_bounty_reward(
    target: &crate::world_sim::state::Entity,
    hostile_faction_ids: &[u32],
) -> f32 {
    match target.kind {
        EntityKind::Monster => HV_BOUNTY_PER_LEVEL * target.level.max(1) as f32,
        EntityKind::Npc => FACTION_NPC_BOUNTY_BASE + target.level as f32 * 15.0,
        _ => 0.0,
    }
}
