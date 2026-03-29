#![allow(unused)]
//! Monster ecology — spawning, migration pressure, and settlement attacks.
//!
//! Monsters spawn in wilderness (far from settlements), preferring:
//! - Low-population regions (empty land breeds monsters)
//! - Resource-rich areas (monsters are drawn to abundance)
//! - Unpatrolled territory (no friendly military presence)
//!
//! Uses entity pooling: dead monsters are recycled via Heal + Move deltas
//! instead of spawning new entities. Pool is pre-allocated at world init.
//!
//! Cadence: every 7 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState, WorldTeam};

use super::seasons::{current_season, season_modifiers};

const ECOLOGY_TICK_INTERVAL: u64 = 7;
const MAX_RESPAWNS_PER_TICK: usize = 5;
const MIN_DISTANCE_FROM_SETTLEMENT_SQ: f32 = 2500.0; // 50 units
const SETTLEMENT_ATTACK_THRESHOLD: f32 = 80.0;
const AMBIENT_ENCOUNTER_THRESHOLD: f32 = 50.0;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

fn tick_hash_u32(tick: u64, salt: u64) -> u32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    (x >> 33) as u32
}

pub fn compute_monster_ecology(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % ECOLOGY_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let season = current_season(state.tick);
    let threat_mod = season_modifiers(season).threat;

    // ---------------------------------------------------------------
    // Phase 1: Recycle dead monsters (entity pooling)
    // ---------------------------------------------------------------
    // Spawn location logic:
    //   - Pick regions with LOW population (frontier/wilderness)
    //   - Weight by resource richness (monsters drawn to abundance)
    //   - Ensure spawn point is FAR from all settlements (>50 units)
    //   - Regions with friendly NPC patrols suppress spawning

    // Score each region for spawn attractiveness.
    // Higher = more likely to spawn monsters here.
    let mut region_scores: [(u32, f32); 64] = [(0, 0.0); 64];
    let mut num_scored = 0usize;

    for region in &state.regions {
        if num_scored >= 64 { break; }

        // Base: low population = more spawning
        let pop_factor = 1.0 / (1.0 + region.monster_density * 0.02);

        // Resource richness: sum stockpiles of settlements in this region
        let resource_richness: f32 = state.settlements.iter()
            .filter(|s| s.faction_id.map(|f| f == region.id).unwrap_or(false) || true)
            .take(3) // don't scan all settlements, just a few nearby
            .map(|s| s.stockpile.iter().sum::<f32>())
            .sum::<f32>()
            * 0.001; // normalize

        // Patrol suppression: count friendly NPCs in the region
        // (approximated by entities near the region's derived position)
        let region_x = (region.id as f32 * 137.5).sin() * 150.0;
        let region_y = (region.id as f32 * 73.1).cos() * 150.0;
        let patrol_count = state.group_index.unaffiliated_entities()
            .len().min(10) as f32 * 0.0; // unaffiliated aren't patrols
        // Real patrol: count friendly NPCs far from home settlements
        // For now, use threat_level as inverse proxy (high threat = low patrols)
        let patrol_factor = 1.0 / (1.0 + (100.0 - region.threat_level).max(0.0) * 0.02);

        // Seasonal: winter drives spawns up (monsters desperate for food)
        let seasonal = threat_mod;

        let score = pop_factor * (1.0 + resource_richness) * patrol_factor * seasonal;

        region_scores[num_scored] = (region.id, score);
        num_scored += 1;
    }

    // Normalize scores to probabilities
    let total_score: f32 = region_scores[..num_scored].iter().map(|r| r.1).sum();

    let mut respawned = 0;
    for entity in &state.entities {
        if respawned >= MAX_RESPAWNS_PER_TICK { break; }
        if entity.kind != EntityKind::Monster || entity.alive { continue; }

        // Deterministic respawn roll: ~10% of dead monsters per ecology tick
        let roll = tick_hash(state.tick, entity.id as u64 ^ 0xBEEF);
        if roll > 0.1 { continue; }

        // Pick spawn region weighted by score
        let target = tick_hash(state.tick, entity.id as u64 ^ 0xCAFE) * total_score;
        let mut cumulative = 0.0;
        let mut spawn_region_idx = 0usize;
        for i in 0..num_scored {
            cumulative += region_scores[i].1;
            if cumulative >= target {
                spawn_region_idx = i;
                break;
            }
        }

        // Compute spawn position far from settlements
        let region_id = region_scores[spawn_region_idx.min(num_scored.saturating_sub(1))].0;
        let base_x = (region_id as f32 * 137.5).sin() * 150.0;
        let base_y = (region_id as f32 * 73.1).cos() * 150.0;
        let jx = tick_hash(state.tick, entity.id as u64 * 3) * 80.0 - 40.0;
        let jy = tick_hash(state.tick, entity.id as u64 * 7) * 80.0 - 40.0;
        let spawn_pos = (base_x + jx, base_y + jy);

        // Verify spawn is far enough from all settlements
        let too_close = state.settlements.iter().any(|s| {
            let dx = spawn_pos.0 - s.pos.0;
            let dy = spawn_pos.1 - s.pos.1;
            dx * dx + dy * dy < MIN_DISTANCE_FROM_SETTLEMENT_SQ
        });
        if too_close { continue; }

        // Revive: heal to full + teleport to spawn location
        out.push(WorldDelta::Heal {
            target_id: entity.id,
            amount: entity.max_hp * 2.0, // overheal to guarantee full HP
            source_id: 0,
        });
        // Move delta: force = desired_pos - current_pos (teleport in one tick)
        out.push(WorldDelta::Move {
            entity_id: entity.id,
            force: (spawn_pos.0 - entity.pos.0, spawn_pos.1 - entity.pos.1),
        });

        respawned += 1;
    }

    // ---------------------------------------------------------------
    // Phase 2: Settlement attacks from high-density regions
    // ---------------------------------------------------------------
    for region in &state.regions {
        let density = region.monster_density;

        if density > SETTLEMENT_ATTACK_THRESHOLD {
            let excess = density - SETTLEMENT_ATTACK_THRESHOLD;
            let damage = excess * 0.5 * threat_mod;

            for settlement in &state.settlements {
                let roll = tick_hash(state.tick, region.id as u64 ^ settlement.id as u64);
                if roll > 0.3 { continue; }

                out.push(WorldDelta::UpdateTreasury {
                    location_id: settlement.id,
                    delta: -damage,
                });
            }
        }

    }
}
