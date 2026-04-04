//! Regional threat tracking — fires every 50 ticks.
//!
//! Threat level per region increases with monster density and decreases
//! when friendly NPC entities are present (patrols). High threat triggers
//! fidelity escalation so combat encounters run at higher fidelity.
//!
//! Original: `crates/headless_campaign/src/systems/threat.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::fidelity::Fidelity;
use crate::world_sim::state::{EntityKind, RegionField, SettlementField, ChronicleEntry, ChronicleCategory, WorldState, WorldTeam};

use super::seasons::{current_season, season_modifiers};

/// How often threat is recalculated (in ticks).
const THREAT_INTERVAL: u64 = 50;

/// Monster density above which threat increases (density is 0.0-1.0).
const DENSITY_THREAT_THRESHOLD: f32 = 0.3;

/// Threat decrease per friendly NPC in the region (patrols reduce threat).
const PATROL_THREAT_REDUCTION: f32 = 0.0002;

/// Threat level that triggers fidelity escalation to Medium.
const ESCALATE_MEDIUM_THRESHOLD: f32 = 50.0;

/// Threat level that triggers fidelity escalation to High.
const ESCALATE_HIGH_THRESHOLD: f32 = 80.0;

/// Natural threat decay per interval.
const THREAT_DECAY_RATE: f32 = 0.1;

pub fn compute_threat(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % THREAT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let season = current_season(state.tick);
    let threat_mod = season_modifiers(season).threat;

    for region in &state.regions {
        // --- Compute threat delta from monster density ---
        let density_pressure = if region.monster_density > DENSITY_THREAT_THRESHOLD {
            (region.monster_density - DENSITY_THREAT_THRESHOLD) * 1.0 * threat_mod
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

        // Emit threat level update.
        let threat_delta = projected - region.threat_level;
        if threat_delta.abs() > 0.01 {
            out.push(WorldDelta::UpdateRegion {
                region_id: region.id,
                field: RegionField::ThreatLevel,
                value: threat_delta,
            });
        }

        // Chronicle milestone: threat crossing critical thresholds.
        if region.threat_level < 80.0 && projected >= 80.0 {
            out.push(WorldDelta::RecordChronicle {
                entry: ChronicleEntry {
                    tick: state.tick,
                    category: ChronicleCategory::Crisis,
                    text: format!("Threat level in {} reached critical levels!", region.name),
                    entity_ids: vec![],
                },
            });
        }
        if region.threat_level >= 50.0 && projected < 20.0 {
            out.push(WorldDelta::RecordChronicle {
                entry: ChronicleEntry {
                    tick: state.tick,
                    category: ChronicleCategory::Achievement,
                    text: format!("Threat in {} has been pacified.", region.name),
                    entity_ids: vec![],
                },
            });
        }

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
                .fidelity_zone(grid_id)
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

        // Monster threat pressure movement is handled by move_target set in
        // the monster_ecology post-apply phase and advance_movement().
    }

    // --- Propagate regional threat to settlements ---
    // Settlements inherit threat from their region's monster density.
    for settlement in &state.settlements {
        // Find region for this settlement (match by faction_id or nearest).
        let regional_threat = state.regions.iter()
            .filter(|r| r.faction_id == settlement.faction_id)
            .map(|r| r.threat_level)
            .fold(0.0f32, f32::max);

        // Settlement threat = fraction of regional threat (0.0-1.0 range).
        let target_threat = (regional_threat / 100.0).clamp(0.0, 1.0);
        let current = settlement.threat_level;
        let delta = (target_threat - current) * 0.1; // drift toward regional threat
        if delta.abs() > 0.001 {
            out.push(WorldDelta::UpdateSettlementField {
                settlement_id: settlement.id,
                field: SettlementField::ThreatLevel,
                value: delta,
            });
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
