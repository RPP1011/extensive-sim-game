#![allow(unused)]
//! Coup engine system — every 10 ticks.
//!
//! Tracks running coup-risk scores for each faction based on internal
//! instability factors. When accumulated risk crosses 0.7 and a random roll
//! succeeds, a coup is triggered: the leader is replaced, faction relations
//! reset, and the faction suffers temporary instability.
//!
//! Ported from `crates/headless_campaign/src/systems/coup_engine.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: factions: Vec<FactionState> on WorldState
//   FactionState { id, relationship_to_guild, military_strength, max_military_strength,
//                  diplomatic_stance, coalition_member, at_war_with, coup_cooldown,
//                  coup_risk, recent_actions }
// NEEDS STATE: civil_wars: Vec<CivilWarState> on WorldState
// NEEDS STATE: guild_faction_id: u32 on WorldState
// NEEDS STATE: regions with owner_faction_id, unrest, control fields

// NEEDS DELTA: SetCoupRisk { faction_id: u32, value: f32 }
// NEEDS DELTA: SetCoupCooldown { faction_id: u32, ticks: u32 }
// NEEDS DELTA: SetRelationship { faction_id: u32, value: f32 }
// NEEDS DELTA: AdjustRelationship { faction_id: u32, delta: f32 }
// NEEDS DELTA: SetDiplomaticStance { faction_id: u32, stance: DiplomaticStance }
// NEEDS DELTA: ClearWars { faction_id: u32 }
// NEEDS DELTA: SetCoalitionMember { faction_id: u32, value: bool }
// NEEDS DELTA: AdjustMilitaryStrength { faction_id: u32, factor: f32 }
// NEEDS DELTA: AdjustRegionUnrest { region_id: u32, delta: f32 }
// NEEDS DELTA: AdjustRegionControl { region_id: u32, factor: f32 }

/// How often the coup engine evaluates risk (in ticks).
const COUP_CHECK_INTERVAL: u64 = 10;

/// Risk threshold above which a coup can fire.
const COUP_RISK_THRESHOLD: f32 = 0.7;

/// Ticks of instability after a coup (-20% effectiveness).
const COUP_INSTABILITY_DURATION: u32 = 1000;

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

    // Once state.factions, state.civil_wars, state.regions exist, enable this.

    /*
    let guild_faction_id = state.guild_faction_id;

    for faction in &state.factions {
        let fi = faction.id;

        // Skip the guild faction.
        if fi == guild_faction_id {
            continue;
        }

        // Tick down cooldown -- skip factions on cooldown.
        if faction.coup_cooldown > 0 {
            let decrement = COUP_CHECK_INTERVAL as u32;
            out.push(WorldDelta::SetCoupCooldown {
                faction_id: fi,
                ticks: faction.coup_cooldown.saturating_sub(decrement),
            });
            continue;
        }

        // Skip factions in active civil war (separate system handles that).
        if state.civil_wars.iter().any(|cw| cw.faction_id == fi) {
            continue;
        }

        // Compute coup risk factors (each contributes 0.0-0.2, max total 1.0).
        let risk = compute_coup_risk(faction, state);

        out.push(WorldDelta::SetCoupRisk {
            faction_id: fi,
            value: risk,
        });

        // Attempt coup if risk exceeds threshold.
        if risk > COUP_RISK_THRESHOLD {
            let roll = deterministic_roll(state.tick, fi, 0);
            let coup_chance = (risk - COUP_RISK_THRESHOLD).min(0.3);
            if roll < coup_chance {
                emit_coup_deltas(state, fi, risk, out);
            }
        }
    }
    */
}

/*
/// Compute the aggregate coup-risk score for a faction.
///
/// Five factors, each contributing 0.0-0.2:
///   1. Leader unpopularity (low relationship_to_guild as proxy)
///   2. Military concentration (extreme strength ratios)
///   3. Treasury deficit (low military strength as proxy)
///   4. Recent defeats (ceasefires, exhaustion)
///   5. Vassal resentment (high unrest in owned regions)
fn compute_coup_risk(faction: &FactionState, state: &WorldState) -> f32 {
    let mut risk = 0.0_f32;

    // 1. Leader unpopularity
    let unpopularity = ((-faction.relationship_to_guild - 20.0) / 80.0).clamp(0.0, 1.0);
    risk += unpopularity * 0.2;

    // 2. Military concentration
    let strength_ratio = if faction.max_military_strength > 0.0 {
        faction.military_strength / faction.max_military_strength
    } else {
        0.5
    };
    let concentration = if strength_ratio > 0.9 {
        (strength_ratio - 0.9) * 10.0
    } else if strength_ratio < 0.3 {
        (0.3 - strength_ratio) / 0.3
    } else {
        0.0
    };
    risk += concentration.clamp(0.0, 1.0) * 0.2;

    // 3. Treasury deficit
    let deficit = ((30.0 - faction.military_strength) / 30.0).clamp(0.0, 1.0);
    risk += deficit * 0.2;

    // 4. Recent defeats
    let recent_defeats = faction
        .recent_actions
        .iter()
        .filter(|a| {
            a.tick + 2000 >= state.tick
                && (a.action.contains("ceasefire")
                    || a.action.contains("exhausted")
                    || a.action.contains("Captured region"))
        })
        .count();
    let defeat_factor = (recent_defeats as f32 / 3.0).clamp(0.0, 1.0);
    risk += defeat_factor * 0.2;

    // 5. Vassal resentment -- average unrest across owned regions
    let faction_regions: Vec<_> = state
        .regions
        .iter()
        .filter(|r| r.faction_id == Some(faction.id))
        .collect();
    let avg_unrest = if !faction_regions.is_empty() {
        faction_regions.iter().map(|r| r.threat_level).sum::<f32>()
            / faction_regions.len() as f32
    } else {
        0.0
    };
    let resentment = (avg_unrest / 100.0).clamp(0.0, 1.0);
    risk += resentment * 0.2;

    risk.clamp(0.0, 1.0)
}

/// Emit deltas for a coup attempt (success or failure).
fn emit_coup_deltas(
    state: &WorldState,
    faction_id: u32,
    risk: f32,
    out: &mut Vec<WorldDelta>,
) {
    let success_roll = deterministic_roll(state.tick, faction_id, 1);
    let success = success_roll < 0.6 + (risk - COUP_RISK_THRESHOLD) * 0.5;

    if success {
        let faction = state.factions.iter().find(|f| f.id == faction_id).unwrap();
        let old_relation = faction.relationship_to_guild;

        // Leader replaced -- relations reset to 0.
        let new_relation = if old_relation > 30.0 { -15.0 } else { 0.0 };
        out.push(WorldDelta::SetRelationship {
            faction_id,
            value: new_relation,
        });

        // Diplomatic stance resets to neutral, wars cleared.
        out.push(WorldDelta::SetDiplomaticStance {
            faction_id,
            stance: DiplomaticStance::Neutral,
        });
        out.push(WorldDelta::SetCoalitionMember {
            faction_id,
            value: false,
        });
        out.push(WorldDelta::ClearWars { faction_id });

        // Military strength reduced by 20%.
        out.push(WorldDelta::AdjustMilitaryStrength {
            faction_id,
            factor: 0.8,
        });

        // Cooldown to prevent rapid successive coups.
        out.push(WorldDelta::SetCoupCooldown {
            faction_id,
            ticks: COUP_INSTABILITY_DURATION,
        });

        // Reset coup risk after successful coup.
        out.push(WorldDelta::SetCoupRisk {
            faction_id,
            value: 0.0,
        });

        // Apply unrest to faction regions (+15 unrest, 0.8x control).
        for region in &state.regions {
            if region.faction_id == Some(faction_id) {
                out.push(WorldDelta::AdjustRegionUnrest {
                    region_id: region.id,
                    delta: 15.0,
                });
                out.push(WorldDelta::AdjustRegionControl {
                    region_id: region.id,
                    factor: 0.8,
                });
            }
        }
    } else {
        // Failed coup -- faction cracks down.
        out.push(WorldDelta::AdjustRelationship {
            faction_id,
            delta: -10.0,
        });
        out.push(WorldDelta::AdjustMilitaryStrength {
            faction_id,
            factor: 0.9,
        });
        out.push(WorldDelta::SetCoupCooldown {
            faction_id,
            ticks: COUP_INSTABILITY_DURATION / 2,
        });
        out.push(WorldDelta::SetCoupRisk {
            faction_id,
            value: risk * 0.5,
        });

        // Unrest from crackdown.
        for region in &state.regions {
            if region.faction_id == Some(faction_id) {
                out.push(WorldDelta::AdjustRegionUnrest {
                    region_id: region.id,
                    delta: 8.0,
                });
            }
        }
    }
}
*/
