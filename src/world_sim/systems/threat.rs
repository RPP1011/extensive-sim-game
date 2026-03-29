#![allow(unused)]
//! Regional threat tracking — fires every 50 ticks.
//!
//! Threat level per region increases with monster density and decreases
//! when friendly NPC entities are present (patrols). High threat triggers
//! fidelity escalation so combat encounters run at higher fidelity.
//!
//! Original: `crates/headless_campaign/src/systems/threat.rs`
//!
//! NEEDS STATE: `patrol_presence` or derive from NPC entity positions per region.
//! NEEDS DELTA: UpdateThreat { region_id, delta } — to adjust threat_level.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::fidelity::Fidelity;
use crate::world_sim::state::{EntityKind, WorldState, WorldTeam};

use super::seasons::{current_season, season_modifiers};

/// How often threat is recalculated (in ticks).
const THREAT_INTERVAL: u64 = 50;

/// Threat increase per tick from monster density above this value.
const DENSITY_THREAT_THRESHOLD: f32 = 30.0;

/// Threat decrease per friendly NPC in the region (patrols reduce threat).
const PATROL_THREAT_REDUCTION: f32 = 2.0;

/// Threat level that triggers fidelity escalation to Medium.
const ESCALATE_MEDIUM_THRESHOLD: f32 = 50.0;

/// Threat level that triggers fidelity escalation to High.
const ESCALATE_HIGH_THRESHOLD: f32 = 80.0;

/// Natural threat decay per interval (threat decays toward baseline).
const THREAT_DECAY_RATE: f32 = 1.0;

pub fn compute_threat(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % THREAT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let season = current_season(state.tick);
    let threat_mod = season_modifiers(season).threat;

    for region in &state.regions {
        // --- Compute threat delta from monster density ---
        let density_pressure = if region.monster_density > DENSITY_THREAT_THRESHOLD {
            (region.monster_density - DENSITY_THREAT_THRESHOLD) * 0.1 * threat_mod
        } else {
            0.0
        };

        // --- Count friendly NPC presence in region as patrol ---
        // Without region_id on entities, we approximate by counting NPCs
        // near any settlement. This is a simplification.
        let patrol_count = state
            .entities
            .iter()
            .filter(|e| e.alive && e.kind == EntityKind::Npc && e.team == WorldTeam::Friendly)
            .count() as f32;

        // Scale patrol effect by number of regions (so total patrol spread matters).
        let regions_count = state.regions.len().max(1) as f32;
        let patrol_reduction = (patrol_count / regions_count) * PATROL_THREAT_REDUCTION;

        // --- Natural decay ---
        let decay = THREAT_DECAY_RATE;

        // --- Net threat change ---
        let net_change = density_pressure - patrol_reduction - decay;

        // Clamp projected threat to [0, 100].
        let projected = (region.threat_level + net_change).clamp(0.0, 100.0);

        // Only emit a delta if threat actually changed meaningfully.
        // Since we can't directly update threat_level (NEEDS DELTA: UpdateThreat),
        // we express the consequence: fidelity escalation.

        // --- Fidelity escalation based on threat ---
        // Find grids associated with this region's settlements.
        for settlement in &state.settlements {
            // Heuristic: match settlements to regions.
            // With proper region_id on settlements this would be exact.
            let grid_id = match settlement.grid_id {
                Some(gid) => gid,
                None => continue,
            };

            let current_fidelity = state
                .grid(grid_id)
                .map(|g| g.fidelity)
                .unwrap_or(Fidelity::Low);

            // Escalate fidelity based on projected threat.
            let desired_fidelity = if projected >= ESCALATE_HIGH_THRESHOLD {
                Fidelity::High
            } else if projected >= ESCALATE_MEDIUM_THRESHOLD {
                Fidelity::Medium
            } else {
                Fidelity::Low
            };

            if fidelity_rank(desired_fidelity) > fidelity_rank(current_fidelity) {
                out.push(WorldDelta::EscalateFidelity {
                    grid_id,
                    new_fidelity: desired_fidelity,
                });
            }
        }

        // When monsters are present and threat is high, move hostile entities
        // toward settlements to create pressure.
        if projected > ESCALATE_MEDIUM_THRESHOLD {
            for entity in &state.entities {
                if !entity.alive || entity.kind != EntityKind::Monster {
                    continue;
                }
                // Find nearest settlement and push monsters toward it.
                if let Some(nearest) = state.settlements.iter().min_by(|a, b| {
                    let da = dist_sq(entity.pos, a.pos);
                    let db = dist_sq(entity.pos, b.pos);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    let dx = nearest.pos.0 - entity.pos.0;
                    let dy = nearest.pos.1 - entity.pos.1;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > 2.0 {
                        // Small additional force from threat pressure.
                        let pressure = (projected / 100.0) * 0.02;
                        out.push(WorldDelta::Move {
                            entity_id: entity.id,
                            force: (dx / dist * pressure, dy / dist * pressure),
                        });
                    }
                }
            }
        }
    }
}

fn dist_sq(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

/// Map fidelity to a numeric rank so we can compare levels.
fn fidelity_rank(f: Fidelity) -> u8 {
    match f {
        Fidelity::Background => 0,
        Fidelity::Low => 1,
        Fidelity::Medium => 2,
        Fidelity::High => 3,
    }
}
