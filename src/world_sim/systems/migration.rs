#![allow(unused)]
//! NPC migration between settlements — fires every 7 ticks.
//!
//! NPCs migrate from unattractive settlements (high threat, low treasury,
//! depleted stockpiles) to more attractive ones. Migration is expressed as
//! Move deltas that shift NPC entities toward their new home settlement,
//! plus economic deltas for population-dependent effects.
//!
//! Original: `crates/headless_campaign/src/systems/migration.rs`
//!
//! NEEDS STATE: `home_settlement_id` on NpcData (already exists).
//! NEEDS STATE: `population` on SettlementState (already exists).
//! NEEDS DELTA: UpdatePopulation { settlement_id, delta } — for population tracking.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, NpcData, SettlementState, WorldState};

use super::seasons::{current_season, season_modifiers};

/// How often migration checks run (in ticks).
const MIGRATION_INTERVAL: u64 = 7;

/// Minimum settlement population to consider out-migration.
const MIN_POP_FOR_MIGRATION: u32 = 50;

/// Attractiveness score threshold below which NPCs want to leave.
const LEAVE_THRESHOLD: f32 = 30.0;

/// Attractiveness score threshold above which a settlement attracts immigrants.
const ATTRACT_THRESHOLD: f32 = 60.0;

/// Maximum distance an NPC will consider migrating (squared, for efficiency).
const MAX_MIGRATION_DIST_SQ: f32 = 10000.0; // 100 units

/// Deterministic hash for pseudo-random decisions from immutable state.
fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

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

    let season = current_season(state.tick);
    let recruit_mod = season_modifiers(season).recruit_chance;

    // Score all settlements.
    let scores: Vec<(u32, f32, (f32, f32))> = state
        .settlements
        .iter()
        .map(|s| (s.id, settlement_attractiveness(s, &state.regions), s.pos))
        .collect();

    // Find the most attractive settlement as a migration target.
    let best_dest = scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let best_dest = match best_dest {
        Some(d) => d,
        None => return,
    };

    // Check each NPC: if their home settlement is unattractive, consider migrating.
    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }

        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        let home_id = match npc.home_settlement_id {
            Some(id) => id,
            None => continue,
        };

        // Skip if already at the best destination.
        if home_id == best_dest.0 {
            continue;
        }

        // Look up home settlement attractiveness.
        let home_score = scores
            .iter()
            .find(|(id, _, _)| *id == home_id)
            .map(|(_, score, _)| *score)
            .unwrap_or(50.0);

        // Only migrate if home is unattractive and destination is attractive.
        if home_score >= LEAVE_THRESHOLD || best_dest.1 < ATTRACT_THRESHOLD {
            continue;
        }

        // Check distance — don't migrate across the whole world.
        let dx = best_dest.2 .0 - entity.pos.0;
        let dy = best_dest.2 .1 - entity.pos.1;
        if dx * dx + dy * dy > MAX_MIGRATION_DIST_SQ {
            continue;
        }

        // Probabilistic: not every NPC migrates every tick.
        let roll = tick_hash(state.tick, entity.id as u64 ^ home_id as u64 ^ 0xBEEF_CAFE);
        let migration_chance = 0.02 * recruit_mod; // ~2% per check, modified by season
        if roll >= migration_chance {
            continue;
        }

        // Emit a Move delta to push the NPC toward the destination settlement.
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > 0.5 {
            let speed = entity.move_speed * 0.1;
            let nx = dx / dist;
            let ny = dy / dist;
            out.push(WorldDelta::Move {
                entity_id: entity.id,
                force: (nx * speed, ny * speed),
            });
        }

        // Economic effect: migrant brings some gold to the destination.
        if npc.gold > 1.0 {
            let transfer = npc.gold * 0.1; // Migrants spend money on relocation.
            out.push(WorldDelta::UpdateTreasury {
                location_id: best_dest.0,
                delta: transfer,
            });
        }
    }
}
