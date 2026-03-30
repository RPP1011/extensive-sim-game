#![allow(unused)]
//! Caravan and trade route system — every tick.
//!
//! NPC entities with Trade intent move between settlements, transferring
//! goods on arrival. This replaces the original caravan struct with
//! entity-based movement and TransferGoods/TransferGold deltas.
//!
//! Original: `crates/headless_campaign/src/systems/caravans.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EconomicIntent, Entity, EntityKind, WorldState, WorldTeam};
use crate::world_sim::NUM_COMMODITIES;

/// Cadence: runs every tick.
const CARAVAN_TICK_INTERVAL: u64 = 1;

/// Base caravan speed (force magnitude per tick).
const CARAVAN_SPEED: f32 = 0.05;

/// Distance (squared) at which a caravan is considered "arrived".
const ARRIVAL_DIST_SQ: f32 = 4.0;

/// Distance (squared) at which hostile entities can raid caravans.
const RAID_RANGE_SQ: f32 = 9.0;

pub fn compute_caravans(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CARAVAN_TICK_INTERVAL != 0 {
        return;
    }

    // Move trading NPCs toward their destination settlement.
    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        let dest_settlement_id = match &npc.economic_intent {
            EconomicIntent::Trade {
                destination_settlement_id,
            } => *destination_settlement_id,
            _ => continue,
        };

        let dest_pos = match state.settlement(dest_settlement_id) {
            Some(s) => s.pos,
            None => continue,
        };

        let dx = dest_pos.0 - entity.pos.0;
        let dy = dest_pos.1 - entity.pos.1;
        let dist_sq = dx * dx + dy * dy;

        if dist_sq < ARRIVAL_DIST_SQ {
            // Arrived: transfer carried goods to settlement.
            for c in 0..NUM_COMMODITIES {
                if npc.carried_goods[c] > 0.001 {
                    out.push(WorldDelta::TransferGoods {
                        from_id: entity.id,
                        to_id: dest_settlement_id,
                        commodity: c,
                        amount: npc.carried_goods[c],
                    });
                }
            }

            // Deliver gold to settlement treasury.
            if npc.gold > 0.001 {
                out.push(WorldDelta::TransferGold {
                    from_id: entity.id,
                    to_id: dest_settlement_id,
                    amount: npc.gold * 0.1, // 10% commission
                });
            }
        } else {
            // Move toward destination.
            let dist = dist_sq.sqrt();
            let nx = dx / dist;
            let ny = dy / dist;
            let speed = entity.move_speed * CARAVAN_SPEED;
            out.push(WorldDelta::Move {
                entity_id: entity.id,
                force: (nx * speed, ny * speed),
            });
        }
    }

    // --- Raid check: hostile entities near trading NPCs ---
    check_caravan_raids(state, out);
}

/// Check if hostile entities are close enough to raid trading NPCs.
fn check_caravan_raids(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Collect trading NPC positions.
    let traders: Vec<(u32, (f32, f32))> = state
        .entities
        .iter()
        .filter(|e| {
            e.alive
                && e.kind == EntityKind::Npc
                && e.team == WorldTeam::Friendly
                && e.npc.as_ref().map_or(false, |n| {
                    matches!(n.economic_intent, EconomicIntent::Trade { .. })
                })
        })
        .map(|e| (e.id, e.pos))
        .collect();

    if traders.is_empty() {
        return;
    }

    // Check hostile entities near traders.
    for &(trader_id, trader_pos) in &traders {
        for entity in &state.entities {
            if !entity.alive || entity.team != WorldTeam::Hostile {
                continue;
            }
            let dx = entity.pos.0 - trader_pos.0;
            let dy = entity.pos.1 - trader_pos.1;
            if dx * dx + dy * dy < RAID_RANGE_SQ {
                // Raid: damage the trading NPC.
                out.push(WorldDelta::Damage {
                    target_id: trader_id,
                    amount: entity.attack_damage * 0.5,
                    source_id: entity.id,
                });
                break; // Only one raid per trader per tick.
            }
        }
    }
}
