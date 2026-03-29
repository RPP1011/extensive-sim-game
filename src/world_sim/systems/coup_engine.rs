#![allow(unused)]
//! Coup engine system — every 10 ticks.
//!
//! Tracks running coup-risk scores for each faction based on internal
//! instability factors. When accumulated risk crosses 0.7 and a random roll
//! succeeds, a coup is triggered: the faction's diplomatic_stance resets,
//! military_strength is reduced, and regions gain unrest.
//!
//! Uses only existing WorldState types and WorldDelta variants:
//! - UpdateFaction for coup_risk, military_strength, relationship_to_guild,
//!   escalation_level
//! - UpdateRegion for unrest and control
//! - RecordEvent / RecordChronicle for narrative
//!
//! Ported from `crates/headless_campaign/src/systems/coup_engine.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{
    ChronicleCategory, ChronicleEntry, FactionField, RegionField, WorldEvent, WorldState,
};

/// How often the coup engine evaluates risk (in ticks).
const COUP_CHECK_INTERVAL: u64 = 10;

/// Risk threshold above which a coup can fire.
const COUP_RISK_THRESHOLD: f32 = 0.7;

/// Risk decay per interval for factions below threshold (stabilization).
const RISK_DECAY: f32 = 0.02;

/// Deterministic hash for pseudo-random decisions.
#[inline]
fn deterministic_roll(tick: u64, faction_id: u32, salt: u32) -> f32 {
    let mut h = tick
        .wrapping_mul(6364136223846793005)
        .wrapping_add(faction_id as u64)
        .wrapping_mul(2862933555777941757)
        .wrapping_add(salt as u64);
    h = h
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (h >> 33) as f32 / (1u64 << 31) as f32
}

pub fn compute_coup_engine(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % COUP_CHECK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for faction in &state.factions {
        let fi = faction.id;

        // --- Update coup risk based on instability factors ---
        let risk_delta = compute_risk_delta(faction, state);
        out.push(WorldDelta::UpdateFaction {
            faction_id: fi,
            field: FactionField::CoupRisk,
            value: risk_delta,
        });

        let projected_risk = (faction.coup_risk + risk_delta).clamp(0.0, 1.0);

        // --- Attempt coup if risk exceeds threshold ---
        if projected_risk > COUP_RISK_THRESHOLD {
            let roll = deterministic_roll(state.tick, fi, 0);
            let coup_chance = (projected_risk - COUP_RISK_THRESHOLD).min(0.3);
            if roll < coup_chance {
                emit_coup_deltas(state, faction, projected_risk, out);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Risk computation
// ---------------------------------------------------------------------------

/// Compute the per-interval change to coup_risk.
///
/// Five factors drive risk upward:
///   1. Leader unpopularity (low/negative relationship_to_guild)
///   2. Military weakness relative to max
///   3. Treasury deficit
///   4. High regional unrest
///   5. Active civil war (escalation_level >= 3)
///
/// Factions with low risk get a small decay toward stability.
fn compute_risk_delta(
    faction: &crate::world_sim::state::FactionState,
    state: &WorldState,
) -> f32 {
    let mut delta = 0.0_f32;

    // 1. Leader unpopularity: negative relationship drives risk up.
    if faction.relationship_to_guild < -20.0 {
        let unpopularity =
            ((-faction.relationship_to_guild - 20.0) / 80.0).clamp(0.0, 1.0);
        delta += unpopularity * 0.04;
    }

    // 2. Military weakness: low strength relative to max.
    let strength_ratio = if faction.max_military_strength > 0.0 {
        faction.military_strength / faction.max_military_strength
    } else {
        0.5
    };
    if strength_ratio < 0.3 {
        delta += (0.3 - strength_ratio) * 0.10;
    }

    // 3. Treasury deficit: low treasury increases risk.
    if faction.treasury < 20.0 {
        let deficit = ((20.0 - faction.treasury) / 20.0).clamp(0.0, 1.0);
        delta += deficit * 0.03;
    }

    // 4. Regional unrest: average unrest across owned regions.
    let faction_regions: Vec<_> = state
        .regions
        .iter()
        .filter(|r| r.faction_id == Some(faction.id))
        .collect();

    if !faction_regions.is_empty() {
        let avg_unrest =
            faction_regions.iter().map(|r| r.unrest).sum::<f32>() / faction_regions.len() as f32;
        if avg_unrest > 0.5 {
            delta += (avg_unrest - 0.5) * 0.06;
        }
    }

    // 5. Active civil war (escalation_level >= 3) accelerates risk.
    if faction.escalation_level >= 3 {
        delta += 0.05;
    }

    // Stability decay when below threshold.
    if faction.coup_risk < COUP_RISK_THRESHOLD && delta <= 0.0 {
        delta -= RISK_DECAY;
    }

    delta
}

// ---------------------------------------------------------------------------
// Coup execution
// ---------------------------------------------------------------------------

/// Emit deltas for a coup attempt. Success probability scales with risk.
fn emit_coup_deltas(
    state: &WorldState,
    faction: &crate::world_sim::state::FactionState,
    risk: f32,
    out: &mut Vec<WorldDelta>,
) {
    let fi = faction.id;
    let success_roll = deterministic_roll(state.tick, fi, 1);
    let success = success_roll < 0.6 + (risk - COUP_RISK_THRESHOLD) * 0.5;

    if success {
        // --- Successful coup ---

        // Relationship resets: new leadership is unpredictable.
        let new_relation = if faction.relationship_to_guild > 30.0 {
            -15.0
        } else {
            0.0
        };
        let rel_delta = new_relation - faction.relationship_to_guild;
        out.push(WorldDelta::UpdateFaction {
            faction_id: fi,
            field: FactionField::RelationshipToGuild,
            value: rel_delta,
        });

        // Military strength reduced by 20%.
        let strength_loss = faction.military_strength * 0.20;
        out.push(WorldDelta::UpdateFaction {
            faction_id: fi,
            field: FactionField::MilitaryStrength,
            value: -strength_loss,
        });

        // Reset coup risk to 0.
        out.push(WorldDelta::UpdateFaction {
            faction_id: fi,
            field: FactionField::CoupRisk,
            value: -faction.coup_risk,
        });

        // Escalation resets (new regime starts fresh).
        if faction.escalation_level > 0 {
            out.push(WorldDelta::UpdateFaction {
                faction_id: fi,
                field: FactionField::EscalationLevel,
                value: -(faction.escalation_level as f32),
            });
        }

        // Regional instability: +15% unrest, -20% control.
        for region in &state.regions {
            if region.faction_id == Some(fi) {
                out.push(WorldDelta::UpdateRegion {
                    region_id: region.id,
                    field: RegionField::Unrest,
                    value: 0.15,
                });
                out.push(WorldDelta::UpdateRegion {
                    region_id: region.id,
                    field: RegionField::Control,
                    value: -0.20,
                });
            }
        }

        out.push(WorldDelta::RecordChronicle {
            entry: ChronicleEntry {
                tick: state.tick,
                category: ChronicleCategory::Crisis,
                text: format!(
                    "Coup in {}! Leadership overthrown. New regime seizes power.",
                    faction.name
                ),
                entity_ids: vec![],
            },
        });

        out.push(WorldDelta::RecordEvent {
            event: WorldEvent::Generic {
                category: ChronicleCategory::Crisis,
                text: format!("Successful coup in {}", faction.name),
            },
        });
    } else {
        // --- Failed coup ---

        // Crackdown: relations worsen, small military loss.
        out.push(WorldDelta::UpdateFaction {
            faction_id: fi,
            field: FactionField::RelationshipToGuild,
            value: -10.0,
        });

        let strength_loss = faction.military_strength * 0.10;
        out.push(WorldDelta::UpdateFaction {
            faction_id: fi,
            field: FactionField::MilitaryStrength,
            value: -strength_loss,
        });

        // Partially reset coup risk (crackdown suppresses some dissent).
        let risk_reduction = faction.coup_risk * 0.5;
        out.push(WorldDelta::UpdateFaction {
            faction_id: fi,
            field: FactionField::CoupRisk,
            value: -risk_reduction,
        });

        // Crackdown unrest in regions.
        for region in &state.regions {
            if region.faction_id == Some(fi) {
                out.push(WorldDelta::UpdateRegion {
                    region_id: region.id,
                    field: RegionField::Unrest,
                    value: 0.08,
                });
            }
        }

        out.push(WorldDelta::RecordEvent {
            event: WorldEvent::Generic {
                category: ChronicleCategory::Crisis,
                text: format!("Failed coup attempt in {}", faction.name),
            },
        });
    }
}
