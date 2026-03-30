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
use crate::world_sim::state::{EntityKind, WorldState};
use crate::world_sim::state::{entity_hash_f32, pair_hash_f32};

use super::seasons::{current_season, season_modifiers};

const ECOLOGY_TICK_INTERVAL: u64 = 7;
const MAX_RESPAWNS_PER_TICK: usize = 5;
const MIN_DISTANCE_FROM_SETTLEMENT_SQ: f32 = 2500.0; // 50 units
const SETTLEMENT_ATTACK_THRESHOLD: f32 = 80.0;
const AMBIENT_ENCOUNTER_THRESHOLD: f32 = 50.0;



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

        // Patrol suppression: use threat_level as inverse proxy (high threat = low patrols)
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
        let roll = entity_hash_f32(entity.id, state.tick, 0xBEEF);
        if roll > 0.1 { continue; }

        // Pick spawn region weighted by score
        let target = entity_hash_f32(entity.id, state.tick, 0xCAFE) * total_score;
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
        let jx = entity_hash_f32(entity.id, state.tick, 3) * 80.0 - 40.0;
        let jy = entity_hash_f32(entity.id, state.tick, 7) * 80.0 - 40.0;
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
                if settlement.treasury <= -100.0 { continue; }
                let roll = pair_hash_f32(region.id, settlement.id, state.tick, 0);
                if roll > 0.3 { continue; }

                out.push(WorldDelta::UpdateTreasury {
                    settlement_id: settlement.id,
                    delta: -damage,
                });
            }
        }
    }

    // ---------------------------------------------------------------
    // Phase 3: Migration — monsters move toward settlements with food
    // ---------------------------------------------------------------
    // Alive monsters drift toward the nearest settlement with high stockpile
    // (drawn to abundance). Rate: small force every ecology tick.
    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Monster { continue; }

        // Only migrate 20% of monsters per tick (stagger).
        let roll = entity_hash_f32(entity.id, state.tick, 0xD1F7);
        if roll > 0.2 { continue; }

        // Find nearest settlement with food.
        let mut best_target: Option<(f32, f32)> = None;
        let mut best_score = 0.0f32;
        for s in &state.settlements {
            let dx = s.pos.0 - entity.pos.0;
            let dy = s.pos.1 - entity.pos.1;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq < 100.0 { continue; } // already near settlement
            let food_draw = s.stockpile[0] * 0.01; // food attracts
            let score = food_draw / (1.0 + dist_sq * 0.0001);
            if score > best_score {
                best_score = score;
                best_target = Some(s.pos);
            }
        }

        if let Some(target) = best_target {
            let dx = target.0 - entity.pos.0;
            let dy = target.1 - entity.pos.1;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > 5.0 {
                // Slow drift toward food source.
                let speed = 0.3;
                out.push(WorldDelta::Move {
                    entity_id: entity.id,
                    force: (dx / dist * speed, dy / dist * speed),
                });
            }
        }
    }

    // ---------------------------------------------------------------
    // Phase 4: Reproduction — monsters near each other can breed
    // ---------------------------------------------------------------
    // When 3+ alive monsters are within 30 units of each other, and population
    // is below cap, spawn a new monster (recycle dead entity pool).
    if state.tick % (ECOLOGY_TICK_INTERVAL * 10) == 0 { // every 70 ticks
        // Count alive monsters and find clusters.
        let alive_monsters: Vec<(u32, (f32, f32), u32)> = state.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Monster)
            .map(|e| (e.id, e.pos, e.level))
            .collect();

        let alive_count = alive_monsters.len();
        let max_monsters = 300; // population cap

        if alive_count < max_monsters {
            // Find clusters of 3+ monsters within 30 units.
            let mut bred = 0;
            for &(mid, mpos, _mlevel) in &alive_monsters {
                if bred >= 2 { break; } // max 2 births per cycle

                let neighbors = alive_monsters.iter()
                    .filter(|&&(oid, opos, _)| {
                        if oid == mid { return false; }
                        let dx = opos.0 - mpos.0;
                        let dy = opos.1 - mpos.1;
                        dx * dx + dy * dy < 900.0 // 30^2
                    })
                    .count();

                if neighbors < 2 { continue; }

                // Deterministic breeding check (~5% per cluster per cycle).
                let breed_roll = entity_hash_f32(mid, state.tick, 0xBABE);
                if breed_roll > 0.05 { continue; }

                // Recycle a dead monster: find one and revive near the cluster.
                for entity in &state.entities {
                    if entity.kind != EntityKind::Monster || entity.alive { continue; }

                    let jx = entity_hash_f32(entity.id, state.tick, 0xB1) * 20.0 - 10.0;
                    let jy = entity_hash_f32(entity.id, state.tick, 0xB2) * 20.0 - 10.0;
                    let baby_pos = (mpos.0 + jx, mpos.1 + jy);

                    out.push(WorldDelta::Heal {
                        target_id: entity.id,
                        amount: entity.max_hp * 2.0,
                        source_id: 0,
                    });
                    out.push(WorldDelta::Move {
                        entity_id: entity.id,
                        force: (baby_pos.0 - entity.pos.0, baby_pos.1 - entity.pos.1),
                    });
                    bred += 1;
                    break;
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Phase 5: Den formation — high monster density creates dens
    // ---------------------------------------------------------------
    // When 5+ monsters cluster in a small area for extended time, the region's
    // monster_density increases permanently — creating a "den" that attracts
    // more respawns. This makes some areas naturally more dangerous.
    if state.tick % (ECOLOGY_TICK_INTERVAL * 5) == 0 {
        for region in &state.regions {
            // Count monsters loosely associated with this region (by position hash).
            let region_x = (region.id as f32 * 137.5).sin() * 150.0;
            let region_y = (region.id as f32 * 73.1).cos() * 150.0;

            let nearby = state.entities.iter()
                .filter(|e| e.alive && e.kind == EntityKind::Monster)
                .filter(|e| {
                    let dx = e.pos.0 - region_x;
                    let dy = e.pos.1 - region_y;
                    dx * dx + dy * dy < 10000.0 // 100 unit radius
                })
                .count();

            if nearby >= 5 {
                // Den forming: increase region monster density.
                out.push(WorldDelta::UpdateRegion {
                    region_id: region.id,
                    field: crate::world_sim::state::RegionField::MonsterDensity,
                    value: 2.0,
                });
            } else if nearby <= 1 && region.monster_density > 10.0 {
                // Den dispersing: slowly reduce density.
                out.push(WorldDelta::UpdateRegion {
                    region_id: region.id,
                    field: crate::world_sim::state::RegionField::MonsterDensity,
                    value: -1.0,
                });
            }
        }
    }

    // ---------------------------------------------------------------
    // Phase 6: Sea monster patrol — high-level monsters near ocean
    // edges circle and hunt NPCs at sea.
    // ---------------------------------------------------------------
    if state.tick % (ECOLOGY_TICK_INTERVAL * 3) == 0 {
        // Find coastal settlement positions for patrol targets.
        let coastal_pos: Vec<(f32, f32)> = state.settlements.iter()
            .filter(|_s| state.regions.iter().any(|r| r.is_coastal))
            .map(|s| s.pos)
            .take(5)
            .collect();

        for entity in &state.entities {
            if !entity.alive || entity.kind != EntityKind::Monster { continue; }
            // Sea monsters are level 20+ with high HP (5× base).
            if entity.level < 20 || entity.max_hp < 300.0 { continue; }

            // Patrol: circle around a coastal settlement.
            if let Some(&target_pos) = coastal_pos.first() {
                let dx = target_pos.0 - entity.pos.0;
                let dy = target_pos.1 - entity.pos.1;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist > 80.0 {
                    // Too far — drift toward coast.
                    let speed = 0.5;
                    out.push(WorldDelta::Move {
                        entity_id: entity.id,
                        force: (dx / dist * speed, dy / dist * speed),
                    });
                } else if dist < 30.0 {
                    // Too close to shore — drift away.
                    let speed = 0.3;
                    out.push(WorldDelta::Move {
                        entity_id: entity.id,
                        force: (-dx / dist * speed, -dy / dist * speed),
                    });
                } else {
                    // Patrol circle: perpendicular drift.
                    let perp_x = -dy / dist * 0.2;
                    let perp_y = dx / dist * 0.2;
                    out.push(WorldDelta::Move {
                        entity_id: entity.id,
                        force: (perp_x, perp_y),
                    });
                }
            }
        }
    }
}
