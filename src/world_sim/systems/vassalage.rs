#![allow(unused)]
//! Vassalage system — every 17 ticks.
//!
//! Weak factions can become vassals of stronger ones. Vassals pay tribute
//! to their lord and receive military protection. Autonomy drifts over time:
//! high autonomy leads to rebellion, low autonomy leads to absorption.
//!
//! Ported from `crates/headless_campaign/src/systems/vassalage.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: factions: Vec<FactionState> on WorldState
//   FactionState { id, name, military_strength, territory_size, relationship_to_guild }
// NEEDS STATE: vassal_relations: Vec<VassalRelation> on WorldState
//   VassalRelation { vassal_id: u32, lord_id: u32, tribute_rate: f32,
//                    autonomy: f32, started_tick: u64 }
// NEEDS STATE: diplomacy: DiplomacyState { guild_faction_id, relations: Vec<Vec<i32>> }
// NEEDS STATE: guild: GuildState { gold }

// NEEDS DELTA: CreateVassalRelation { vassal_id: u32, lord_id: u32, tribute_rate: f32,
//              autonomy: f32 }
// NEEDS DELTA: RemoveVassalRelation { vassal_id: u32, lord_id: u32 }
// NEEDS DELTA: AdjustAutonomy { vassal_id: u32, lord_id: u32, delta: f32 }
// NEEDS DELTA: AdjustRelationship { faction_id: u32, delta: f32 }
// NEEDS DELTA: AdjustDiplomacyRelation { faction_a: u32, faction_b: u32, delta: i32 }
// NEEDS DELTA: AdjustMilitaryStrength { faction_id: u32, delta: f32 }
// NEEDS DELTA: SetMilitaryStrength { faction_id: u32, value: f32 }
// NEEDS DELTA: AdjustTerritorySize { faction_id: u32, delta: i32 }
// NEEDS DELTA: SetTerritorySize { faction_id: u32, value: u32 }

/// Cadence: every 17 ticks.
const VASSALAGE_INTERVAL: u64 = 17;

/// Deterministic hash for pseudo-random decisions.
#[inline]
fn deterministic_roll(tick: u64, a: u32, b: u32, salt: u32) -> f32 {
    let mut h = tick
        .wrapping_mul(6364136223846793005)
        .wrapping_add(a as u64)
        .wrapping_mul(2862933555777941757)
        .wrapping_add(b as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(salt as u64);
    h = h
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (h >> 33) as f32 / (1u64 << 31) as f32
}

pub fn compute_vassalage(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % VASSALAGE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Once state.factions, state.vassal_relations, state.diplomacy, state.guild
    // exist, enable this.

    /*
    let guild_id = state.diplomacy.guild_faction_id;

    // --- Phase 1: Auto-vassalage for weak NPC factions ---
    compute_auto_vassalage(state, guild_id, out);

    // --- Phase 2: Tribute collection ---
    compute_tribute(state, guild_id, out);

    // --- Phase 3: Autonomy drift and rebellion ---
    compute_autonomy(state, guild_id, out);

    // --- Phase 4: Absorption of very low-autonomy vassals ---
    compute_absorption(state, out);
    */
}

/*
// ---------------------------------------------------------------------------
// Auto-vassalage
// ---------------------------------------------------------------------------

/// Weak factions (strength < 20) with a strong neighbor (strength > 60)
/// become vassals with 30% chance.
fn compute_auto_vassalage(
    state: &WorldState,
    guild_id: u32,
    out: &mut Vec<WorldDelta>,
) {
    let already_vassal: Vec<u32> = state
        .vassal_relations
        .iter()
        .map(|v| v.vassal_id)
        .collect();

    for faction in &state.factions {
        let fid = faction.id;

        // Skip guild, existing vassals, and lords.
        if fid == guild_id || already_vassal.contains(&fid) {
            continue;
        }
        if state.vassal_relations.iter().any(|v| v.lord_id == fid) {
            continue;
        }

        if faction.military_strength >= 20.0 {
            continue;
        }

        // Find strongest eligible neighbor.
        let best_lord = state
            .factions
            .iter()
            .filter(|f| {
                f.id != fid
                    && f.id != guild_id
                    && f.military_strength > 60.0
                    && !already_vassal.contains(&f.id)
            })
            .max_by(|a, b| a.military_strength.partial_cmp(&b.military_strength).unwrap());

        if let Some(lord) = best_lord {
            let roll = deterministic_roll(state.tick, fid, lord.id, 0);
            if roll < 0.30 {
                let tribute_roll = deterministic_roll(state.tick, fid, lord.id, 1);
                let tribute_rate = 0.10 + tribute_roll * 0.20; // 10-30%

                out.push(WorldDelta::CreateVassalRelation {
                    vassal_id: fid,
                    lord_id: lord.id,
                    tribute_rate,
                    autonomy: 50.0,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tribute collection
// ---------------------------------------------------------------------------

fn compute_tribute(
    state: &WorldState,
    guild_id: u32,
    out: &mut Vec<WorldDelta>,
) {
    for rel in &state.vassal_relations {
        let vassal = match state.factions.iter().find(|f| f.id == rel.vassal_id) {
            Some(f) => f,
            None => continue,
        };

        let base_income = vassal.territory_size as f32 * 5.0;
        let tribute = base_income * rel.tribute_rate;

        if rel.lord_id == guild_id {
            // Guild receives tribute via TransferGold.
            out.push(WorldDelta::TransferGold {
                from_id: rel.vassal_id,
                to_id: guild_id,
                amount: tribute,
            });
        } else if rel.vassal_id == guild_id {
            // Guild pays tribute.
            out.push(WorldDelta::TransferGold {
                from_id: guild_id,
                to_id: rel.lord_id,
                amount: tribute,
            });
        } else {
            // NPC-to-NPC: lord gains strength.
            out.push(WorldDelta::AdjustMilitaryStrength {
                faction_id: rel.lord_id,
                delta: tribute * 0.1,
            });
        }

        // Lord provides military protection: small strength boost to vassal.
        let lord_strength = state
            .factions
            .iter()
            .find(|f| f.id == rel.lord_id)
            .map(|f| f.military_strength)
            .unwrap_or(0.0);
        out.push(WorldDelta::AdjustMilitaryStrength {
            faction_id: rel.vassal_id,
            delta: lord_strength * 0.02,
        });
    }
}

// ---------------------------------------------------------------------------
// Autonomy drift and rebellion
// ---------------------------------------------------------------------------

fn compute_autonomy(
    state: &WorldState,
    guild_id: u32,
    out: &mut Vec<WorldDelta>,
) {
    for rel in &state.vassal_relations {
        let lord_strength = state
            .factions
            .iter()
            .find(|f| f.id == rel.lord_id)
            .map(|f| f.military_strength)
            .unwrap_or(0.0);
        let vassal_strength = state
            .factions
            .iter()
            .find(|f| f.id == rel.vassal_id)
            .map(|f| f.military_strength)
            .unwrap_or(0.0);

        // Strength ratio drives autonomy drift.
        let ratio = if lord_strength > 0.0 {
            vassal_strength / lord_strength
        } else {
            1.0
        };

        let mut autonomy_delta = if ratio > 0.5 {
            3.0
        } else if ratio < 0.2 {
            -2.0
        } else {
            1.0
        };

        // High tribute rate increases autonomy (resentment).
        if rel.tribute_rate > 0.25 {
            autonomy_delta += 2.0;
        }

        out.push(WorldDelta::AdjustAutonomy {
            vassal_id: rel.vassal_id,
            lord_id: rel.lord_id,
            delta: autonomy_delta,
        });

        // Rebellion check: autonomy > 80, 20% chance.
        if rel.autonomy > 80.0 {
            let roll = deterministic_roll(state.tick, rel.vassal_id, rel.lord_id, 100);
            if roll < 0.20 {
                // Remove vassal relation.
                out.push(WorldDelta::RemoveVassalRelation {
                    vassal_id: rel.vassal_id,
                    lord_id: rel.lord_id,
                });

                // Vassal goes hostile toward lord.
                if rel.lord_id == guild_id {
                    out.push(WorldDelta::AdjustRelationship {
                        faction_id: rel.vassal_id,
                        delta: -20.0,
                    });
                }

                // Update diplomacy matrix.
                out.push(WorldDelta::AdjustDiplomacyRelation {
                    faction_a: rel.vassal_id,
                    faction_b: rel.lord_id,
                    delta: -30,
                });
                out.push(WorldDelta::AdjustDiplomacyRelation {
                    faction_a: rel.lord_id,
                    faction_b: rel.vassal_id,
                    delta: -20,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Absorption of very low-autonomy vassals
// ---------------------------------------------------------------------------

fn compute_absorption(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for rel in &state.vassal_relations {
        if rel.autonomy >= 20.0 {
            continue;
        }

        let roll = deterministic_roll(state.tick, rel.vassal_id, rel.lord_id, 200);
        if roll >= 0.15 {
            continue;
        }

        let vassal = match state.factions.iter().find(|f| f.id == rel.vassal_id) {
            Some(f) => f,
            None => continue,
        };

        // Transfer territory and partial strength to lord.
        out.push(WorldDelta::AdjustTerritorySize {
            faction_id: rel.lord_id,
            delta: vassal.territory_size as i32,
        });
        out.push(WorldDelta::AdjustMilitaryStrength {
            faction_id: rel.lord_id,
            delta: vassal.military_strength * 0.5,
        });

        // Zero out absorbed faction.
        out.push(WorldDelta::SetMilitaryStrength {
            faction_id: rel.vassal_id,
            value: 0.0,
        });
        out.push(WorldDelta::SetTerritorySize {
            faction_id: rel.vassal_id,
            value: 0,
        });

        // Remove vassal relation.
        out.push(WorldDelta::RemoveVassalRelation {
            vassal_id: rel.vassal_id,
            lord_id: rel.lord_id,
        });
    }
}
*/
