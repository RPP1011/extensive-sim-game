#![allow(unused)]
//! Civil war system — every 7 ticks.
//!
//! When a faction's regions have high unrest (>70) and its military_strength
//! is low (<30), civil war conditions emerge. During civil war (signaled by
//! escalation_level >= 3), the faction bleeds treasury and military_strength
//! each interval. Resolution occurs when military_strength drops to near zero
//! (leadership collapses) or unrest subsides.
//!
//! Uses only existing WorldState types and WorldDelta variants:
//! - UpdateFaction for military_strength, treasury, coup_risk, escalation_level
//! - UpdateRegion for unrest and control
//! - RecordEvent / RecordChronicle for narrative
//!
//! Ported from `crates/headless_campaign/src/systems/civil_war.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{
    ChronicleCategory, ChronicleEntry, FactionField, RegionField, WorldEvent,
    WorldState,
};
use crate::world_sim::state::entity_hash_f32;

/// Cadence: every 7 ticks.
const CIVIL_WAR_INTERVAL: u64 = 7;

/// Average regional unrest threshold to trigger civil war onset.
const UNREST_TRIGGER: f32 = 0.70;

/// Military strength below which civil war can ignite.
const MILITARY_WEAKNESS_TRIGGER: f32 = 30.0;

/// Escalation level that marks active civil war.
const CIVIL_WAR_ESCALATION: u32 = 3;

/// Treasury drain per interval during civil war.
const TREASURY_DRAIN: f32 = 5.0;

/// Military strength attrition per interval during civil war.
const MILITARY_ATTRITION: f32 = 2.0;

/// Unrest increase per interval in regions of a faction at civil war.
const REGION_UNREST_INCREASE: f32 = 0.04;

/// Control decrease per interval in regions of a faction at civil war.
const REGION_CONTROL_DECREASE: f32 = 0.03;

/// Coup risk increase during civil war per interval.
const COUP_RISK_INCREASE: f32 = 0.05;

/// Chance per interval to trigger civil war when conditions are met.
const IGNITION_CHANCE: f32 = 0.08;


pub fn compute_civil_war(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CIVIL_WAR_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    compute_trigger_civil_wars(state, out);
    compute_active_civil_wars(state, out);
}

// ---------------------------------------------------------------------------
// Trigger logic — check if any faction should enter civil war
// ---------------------------------------------------------------------------

fn compute_trigger_civil_wars(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for faction in &state.factions {
        // Skip factions already in civil war (escalation >= CIVIL_WAR_ESCALATION).
        if faction.escalation_level >= CIVIL_WAR_ESCALATION {
            continue;
        }

        // Skip factions that are strong enough to suppress dissent.
        if faction.military_strength >= MILITARY_WEAKNESS_TRIGGER {
            continue;
        }

        // Compute average unrest across faction regions.
        let faction_regions: Vec<_> = state
            .regions
            .iter()
            .filter(|r| r.faction_id == Some(faction.id))
            .collect();

        if faction_regions.is_empty() {
            continue;
        }

        let avg_unrest: f32 =
            faction_regions.iter().map(|r| r.unrest).sum::<f32>() / faction_regions.len() as f32;

        if avg_unrest < UNREST_TRIGGER {
            continue;
        }

        // Probabilistic ignition.
        let roll = entity_hash_f32(faction.id, state.tick, 100 as u64);
        if roll >= IGNITION_CHANCE {
            continue;
        }

        // Civil war ignites: set escalation to CIVIL_WAR_ESCALATION.
        // escalation_level is u32 stored as f32 delta, so we set it to target - current.
        let escalation_delta = CIVIL_WAR_ESCALATION as f32 - faction.escalation_level as f32;
        if escalation_delta > 0.0 {
            out.push(WorldDelta::UpdateFaction {
                faction_id: faction.id,
                field: FactionField::EscalationLevel,
                value: escalation_delta,
            });
        }

        // Military strength drops as the faction splits.
        let strength_loss = faction.military_strength * 0.3;
        out.push(WorldDelta::UpdateFaction {
            faction_id: faction.id,
            field: FactionField::MilitaryStrength,
            value: -strength_loss,
        });

        // Spike coup risk.
        out.push(WorldDelta::UpdateFaction {
            faction_id: faction.id,
            field: FactionField::CoupRisk,
            value: 0.2,
        });

        out.push(WorldDelta::RecordChronicle {
            entry: ChronicleEntry {
                tick: state.tick,
                category: ChronicleCategory::Crisis,
                text: format!(
                    "Civil war erupts in {}! Military splits as regional unrest boils over.",
                    faction.name
                ),
                entity_ids: vec![],
            },
        });

        out.push(WorldDelta::RecordEvent {
            event: WorldEvent::Generic {
                category: ChronicleCategory::Crisis,
                text: format!("Civil war in {}", faction.name),
            },
        });
    }
}

// ---------------------------------------------------------------------------
// Active civil war processing
// ---------------------------------------------------------------------------

fn compute_active_civil_wars(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for faction in &state.factions {
        // Only process factions at civil war escalation level.
        if faction.escalation_level < CIVIL_WAR_ESCALATION {
            continue;
        }

        let fi = faction.id;

        // --- Ongoing attrition ---
        // Treasury drain from fighting.
        out.push(WorldDelta::UpdateFaction {
            faction_id: fi,
            field: FactionField::Treasury,
            value: -TREASURY_DRAIN,
        });

        // Military strength attrition from internal conflict.
        let attrition_roll = entity_hash_f32(fi, state.tick, 200 as u64);
        let attrition = MILITARY_ATTRITION + attrition_roll * 1.5;
        out.push(WorldDelta::UpdateFaction {
            faction_id: fi,
            field: FactionField::MilitaryStrength,
            value: -attrition,
        });

        // Coup risk continues to climb during civil war.
        out.push(WorldDelta::UpdateFaction {
            faction_id: fi,
            field: FactionField::CoupRisk,
            value: COUP_RISK_INCREASE,
        });

        // --- Apply unrest to faction regions ---
        for region in &state.regions {
            if region.faction_id == Some(fi) {
                out.push(WorldDelta::UpdateRegion {
                    region_id: region.id,
                    field: RegionField::Unrest,
                    value: REGION_UNREST_INCREASE,
                });
                out.push(WorldDelta::UpdateRegion {
                    region_id: region.id,
                    field: RegionField::Control,
                    value: -REGION_CONTROL_DECREASE,
                });
            }
        }

        // --- Resolution check ---
        // Loyalist victory: military strength still above 10 and unrest starts dropping.
        // We check if the faction has been in civil war long enough for resolution.
        let faction_regions: Vec<_> = state
            .regions
            .iter()
            .filter(|r| r.faction_id == Some(fi))
            .collect();

        let avg_unrest = if !faction_regions.is_empty() {
            faction_regions.iter().map(|r| r.unrest).sum::<f32>() / faction_regions.len() as f32
        } else {
            0.0
        };

        let projected_strength = (faction.military_strength - attrition).max(0.0);

        if projected_strength <= 5.0 {
            // Faction collapses — civil war resolves with regime change.
            resolve_collapse(state, faction, out);
        } else if avg_unrest < 0.30 && faction.military_strength > 15.0 {
            // Unrest has subsided and faction is still viable — loyalist victory.
            resolve_loyalist_win(state, faction, out);
        }
    }
}

// ---------------------------------------------------------------------------
// Resolution helpers
// ---------------------------------------------------------------------------

fn resolve_collapse(
    state: &WorldState,
    faction: &crate::world_sim::state::FactionState,
    out: &mut Vec<WorldDelta>,
) {
    let fi = faction.id;

    // Reset escalation to 0 (end civil war).
    let esc_delta = -(faction.escalation_level as f32);
    out.push(WorldDelta::UpdateFaction {
        faction_id: fi,
        field: FactionField::EscalationLevel,
        value: esc_delta,
    });

    // Surviving strength is minimal.
    let remaining = (faction.military_strength * 0.2).max(1.0) - faction.military_strength;
    out.push(WorldDelta::UpdateFaction {
        faction_id: fi,
        field: FactionField::MilitaryStrength,
        value: remaining,
    });

    // Relations reset — new leadership is uncertain about the guild.
    let rel_delta = -faction.relationship_to_guild; // Reset to 0.
    out.push(WorldDelta::UpdateFaction {
        faction_id: fi,
        field: FactionField::RelationshipToGuild,
        value: rel_delta,
    });

    // Reset coup risk.
    out.push(WorldDelta::UpdateFaction {
        faction_id: fi,
        field: FactionField::CoupRisk,
        value: -faction.coup_risk,
    });

    // Reduce unrest in regions (war is over, but at great cost).
    for region in &state.regions {
        if region.faction_id == Some(fi) {
            out.push(WorldDelta::UpdateRegion {
                region_id: region.id,
                field: RegionField::Unrest,
                value: -0.20,
            });
        }
    }

    out.push(WorldDelta::RecordChronicle {
        entry: ChronicleEntry {
            tick: state.tick,
            category: ChronicleCategory::Crisis,
            text: format!(
                "Civil war in {} ends in collapse. New regime emerges from the ashes.",
                faction.name
            ),
            entity_ids: vec![],
        },
    });
}

fn resolve_loyalist_win(
    state: &WorldState,
    faction: &crate::world_sim::state::FactionState,
    out: &mut Vec<WorldDelta>,
) {
    let fi = faction.id;

    // Reset escalation to 0 (end civil war).
    let esc_delta = -(faction.escalation_level as f32);
    out.push(WorldDelta::UpdateFaction {
        faction_id: fi,
        field: FactionField::EscalationLevel,
        value: esc_delta,
    });

    // Loyalists retain 80% of current strength.
    let strength_loss = faction.military_strength * 0.2;
    out.push(WorldDelta::UpdateFaction {
        faction_id: fi,
        field: FactionField::MilitaryStrength,
        value: -strength_loss,
    });

    // Coup risk resets.
    out.push(WorldDelta::UpdateFaction {
        faction_id: fi,
        field: FactionField::CoupRisk,
        value: -faction.coup_risk,
    });

    // Relationship boost if guild was supportive (positive relationship).
    if faction.relationship_to_guild > 0.0 {
        out.push(WorldDelta::UpdateFaction {
            faction_id: fi,
            field: FactionField::RelationshipToGuild,
            value: 10.0,
        });
    }

    // Reduce unrest in faction regions.
    for region in &state.regions {
        if region.faction_id == Some(fi) {
            out.push(WorldDelta::UpdateRegion {
                region_id: region.id,
                field: RegionField::Unrest,
                value: -0.15,
            });
        }
    }

    out.push(WorldDelta::RecordChronicle {
        entry: ChronicleEntry {
            tick: state.tick,
            category: ChronicleCategory::Crisis,
            text: format!(
                "Loyalists prevail in {}. Civil war ends with the faction weakened but intact.",
                faction.name
            ),
            entity_ids: vec![],
        },
    });
}
