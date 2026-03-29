#![allow(unused)]
//! Price discovery — NPCs learn local prices and share knowledge via gossip.
//!
//! 1. NPCs at a settlement automatically learn its current prices (free).
//! 2. NPCs at the same settlement gossip: 5% exchange foreign price info per tick.
//! 3. Stale reports pruned to cap of 30 per NPC.
//!
//! This drives trade decisions — NPCs with knowledge of price differentials
//! will initiate trade runs.
//!
//! Cadence: every 10 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;

const PRICE_DISCOVERY_INTERVAL: u64 = 10;
const MAX_REPORTS_PER_NPC: usize = 30;
const GOSSIP_RATE: f32 = 0.05;

fn tick_hash(tick: u64, salt: u64) -> u32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    (x >> 33) as u32
}

pub fn compute_price_discovery(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % PRICE_DISCOVERY_INTERVAL != 0 || state.tick == 0 { return; }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_price_discovery_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

pub fn compute_price_discovery_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % PRICE_DISCOVERY_INTERVAL != 0 || state.tick == 0 { return; }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // Phase 1: Every NPC at this settlement learns current local prices.
    // Emit SharePriceReport to self — this updates their price_knowledge.
    let report = PriceReport {
        settlement_id,
        prices: settlement.prices,
        tick_observed: state.tick,
    };

    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        // Check if NPC already has a recent report for this settlement.
        let already_current = npc.price_knowledge.iter().any(|r|
            r.settlement_id == settlement_id && r.tick_observed >= state.tick.saturating_sub(PRICE_DISCOVERY_INTERVAL)
        );
        if already_current { continue; }

        out.push(WorldDelta::SharePriceReport {
            from_id: entity.id,
            to_id: entity.id,
            report: report.clone(),
        });
    }

    // Phase 2: Gossip — NPCs at the same settlement share foreign price knowledge.
    // For ~5% of NPC pairs, share reports.
    // Since we can't mutate during compute, we emit SharePriceReport deltas
    // between random pairs. The apply phase handles dedup.

    // Collect alive NPC IDs at this settlement (stack allocated).
    let mut npc_ids = [0u32; 256];
    let mut count = 0usize;
    for entity in entities {
        if entity.alive && entity.kind == EntityKind::Npc && entity.npc.is_some() && count < 256 {
            npc_ids[count] = entity.id;
            count += 1;
        }
    }

    if count < 2 { return; }

    // Number of gossip exchanges this tick.
    let exchanges = ((count as f32) * GOSSIP_RATE).ceil() as usize;

    for i in 0..exchanges {
        let a_idx = (tick_hash(state.tick, settlement_id as u64 * 31 + i as u64) as usize) % count;
        let b_idx = (tick_hash(state.tick, settlement_id as u64 * 37 + i as u64 + 1000) as usize) % count;
        if a_idx == b_idx { continue; }

        let a_id = npc_ids[a_idx];
        let b_id = npc_ids[b_idx];

        // A shares their best foreign report with B.
        if let Some(a_entity) = state.entity(a_id) {
            if let Some(a_npc) = &a_entity.npc {
                // Find A's most recent foreign report.
                if let Some(best_report) = a_npc.price_knowledge.iter()
                    .filter(|r| r.settlement_id != settlement_id)
                    .max_by_key(|r| r.tick_observed)
                {
                    out.push(WorldDelta::SharePriceReport {
                        from_id: a_id,
                        to_id: b_id,
                        report: best_report.clone(),
                    });
                }
            }
        }

        // B shares with A.
        if let Some(b_entity) = state.entity(b_id) {
            if let Some(b_npc) = &b_entity.npc {
                if let Some(best_report) = b_npc.price_knowledge.iter()
                    .filter(|r| r.settlement_id != settlement_id)
                    .max_by_key(|r| r.tick_observed)
                {
                    out.push(WorldDelta::SharePriceReport {
                        from_id: b_id,
                        to_id: a_id,
                        report: best_report.clone(),
                    });
                }
            }
        }
    }
}
