#![allow(unused)]
//! Civil war and coup system — internal faction instability.
//!
//! Every 7 ticks, checks whether factions should erupt into civil war.
//! During a civil war the faction splits strength between loyalists and rebels,
//! cannot declare new external wars, and regions gain unrest. The guild can
//! intervene by supporting either side.
//!
//! Ported from `crates/headless_campaign/src/systems/civil_war.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: factions: Vec<FactionState> on WorldState
//   FactionState { id, name, military_strength, max_military_strength,
//                  diplomatic_stance, relationship_to_guild, coalition_member,
//                  at_war_with, recent_actions }
// NEEDS STATE: civil_wars: Vec<CivilWarState> on WorldState
//   CivilWarState { faction_id: u32, rebel_strength: f32, loyalist_strength: f32,
//                   started_tick: u64, rebel_leader_name: String,
//                   cause: CivilWarCause, guild_supported_side: Option<bool> }
// NEEDS STATE: CivilWarCause enum { SuccessionCrisis, MilitaryDefeat, Corruption,
//              ForeignInfluence, ReligiousSplit }
// NEEDS STATE: DiplomaticStance enum
// NEEDS STATE: guild_faction_id: u32 on WorldState
// NEEDS STATE: diplomacy.relations: Vec<Vec<i32>> on WorldState
// NEEDS STATE: regions with owner_faction_id, unrest fields

// NEEDS DELTA: StartCivilWar { faction_id: u32, rebel_strength: f32,
//              loyalist_strength: f32, rebel_leader_name: String, cause: String }
// NEEDS DELTA: AdjustMilitaryStrength { faction_id: u32, delta: f32 }
// NEEDS DELTA: SetMilitaryStrength { faction_id: u32, value: f32 }
// NEEDS DELTA: AdjustRegionUnrest { region_id: u32, delta: f32 }
// NEEDS DELTA: AdjustCivilWarStrength { faction_id: u32, side: Side, delta: f32 }
// NEEDS DELTA: ResolveCivilWar { faction_id: u32, rebels_won: bool }
// NEEDS DELTA: AdjustRelationship { faction_id: u32, delta: f32 }
// NEEDS DELTA: SetRelationship { faction_id: u32, value: f32 }
// NEEDS DELTA: SetDiplomaticStance { faction_id: u32, stance: DiplomaticStance }
// NEEDS DELTA: EndWar { faction_a: u32, faction_b: u32 }
// NEEDS DELTA: CreateFaction { name: String, military_strength: f32, ... }
// NEEDS DELTA: ChangeRegionOwner { region_id: u32, new_owner: u32 }
// NEEDS DELTA: DeclareWar { attacker_id: u32, defender_id: u32 }

/// Cadence: every 7 ticks.
const CIVIL_WAR_INTERVAL: u64 = 7;

/// Deterministic hash for pseudo-random decisions.
#[inline]
fn deterministic_roll(tick: u64, faction_id: u32, salt: u32) -> f32 {
    let mut h = tick.wrapping_mul(6364136223846793005)
        .wrapping_add(faction_id as u64)
        .wrapping_mul(2862933555777941757)
        .wrapping_add(salt as u64);
    h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (h >> 33) as f32 / (1u64 << 31) as f32
}

pub fn compute_civil_war(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CIVIL_WAR_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Trigger new civil wars, then tick active ones.
    compute_trigger_civil_wars(state, out);
    compute_active_civil_wars(state, out);
}

// ---------------------------------------------------------------------------
// Trigger logic
// ---------------------------------------------------------------------------

fn compute_trigger_civil_wars(state: &WorldState, out: &mut Vec<WorldDelta>) {
    /*
    let guild_faction_id = state.guild_faction_id;

    for faction in &state.factions {
        let fi = faction.id;

        // Skip guild faction and factions already in civil war
        if fi == guild_faction_id { continue; }
        if state.civil_wars.iter().any(|cw| cw.faction_id == fi) { continue; }

        // Determine cause (if any condition is met)
        let cause = determine_civil_war_cause(faction, state);
        let cause = match cause {
            Some(c) => c,
            None => continue,
        };

        // 5% chance per check when conditions met
        let roll = deterministic_roll(state.tick, fi, 100);
        if roll > 0.05 { continue; }

        // Pick a rebel leader name deterministically
        let rebel_names = [
            "General Voss", "Commander Reth", "Lord Kaine", "Marshal Dren",
            "Captain Sura", "Warlord Thane", "Prefect Mala", "Admiral Bron",
        ];
        let name_roll = deterministic_roll(state.tick, fi, 101);
        let name_idx = (name_roll * rebel_names.len() as f32) as usize % rebel_names.len();
        let rebel_leader = rebel_names[name_idx].to_string();

        // Split faction strength: rebels get 30-50%
        let rebel_fraction_roll = deterministic_roll(state.tick, fi, 102);
        let rebel_fraction = 0.3 + rebel_fraction_roll * 0.2;
        let total_strength = faction.military_strength;
        let rebel_strength = total_strength * rebel_fraction;
        let loyalist_strength = total_strength - rebel_strength;

        // Faction's effective military strength drops to loyalist portion
        out.push(WorldDelta::SetMilitaryStrength {
            faction_id: fi,
            value: loyalist_strength,
        });

        out.push(WorldDelta::StartCivilWar {
            faction_id: fi,
            rebel_strength,
            loyalist_strength,
            rebel_leader_name: rebel_leader,
            cause: format!("{:?}", cause),
        });
    }
    */
}

/*
fn determine_civil_war_cause(faction: &FactionState, state: &WorldState) -> Option<String> {
    // Succession crisis: current strength < 50% of max
    let strength_ratio = faction.military_strength / faction.max_military_strength;
    if strength_ratio < 0.5 {
        // Check for recent defeats
        let recent_defeats = faction.recent_actions.iter()
            .filter(|a| a.tick + 2000 >= state.tick
                && (a.action.contains("ceasefire") || a.action.contains("exhausted")))
            .count();
        if recent_defeats > 0 {
            return Some("MilitaryDefeat".to_string());
        }
        return Some("SuccessionCrisis".to_string());
    }

    // Corruption: hostile to guild but strong
    if faction.relationship_to_guild < -50.0 && faction.military_strength > 60.0 {
        return Some("Corruption".to_string());
    }

    // High unrest across faction regions
    let faction_regions: Vec<_> = state.regions.iter()
        .filter(|r| r.faction_id == Some(faction.id))
        .collect();

    if !faction_regions.is_empty() {
        let avg_unrest: f32 = faction_regions.iter()
            .map(|r| r.threat_level) // Using threat_level as proxy for unrest
            .sum::<f32>() / faction_regions.len() as f32;
        if avg_unrest > 60.0 {
            let causes = ["ForeignInfluence", "ReligiousSplit", "Corruption"];
            let idx = (faction.id as usize + state.tick as usize) % causes.len();
            return Some(causes[idx].to_string());
        }
    }

    None
}
*/

// ---------------------------------------------------------------------------
// Active civil war processing
// ---------------------------------------------------------------------------

fn compute_active_civil_wars(state: &WorldState, out: &mut Vec<WorldDelta>) {
    /*
    for cw in &state.civil_wars {
        let fi = cw.faction_id;
        let elapsed = state.tick.saturating_sub(cw.started_tick);

        // Apply unrest to faction regions (+4 per 7-tick check)
        for region in &state.regions {
            if region.faction_id == Some(fi) {
                out.push(WorldDelta::AdjustRegionUnrest {
                    region_id: region.id,
                    delta: 4.0,
                });
            }
        }

        // Strength attrition: both sides lose strength from fighting
        let attrition_roll = deterministic_roll(state.tick, fi, 200);
        let attrition = 1.0 + attrition_roll * 2.0;

        out.push(WorldDelta::AdjustCivilWarStrength {
            faction_id: fi,
            side: Side::Loyalist,
            delta: -attrition,
        });
        out.push(WorldDelta::AdjustCivilWarStrength {
            faction_id: fi,
            side: Side::Rebel,
            delta: -attrition,
        });

        let loyalist = (cw.loyalist_strength - attrition).max(0.0);
        let rebel = (cw.rebel_strength - attrition).max(0.0);

        // Check resolution conditions
        if loyalist > 0.0 && rebel > 0.0 && loyalist >= rebel * 2.0 {
            // Loyalist victory
            resolve_loyalist_win(state, cw, out);
        } else if rebel > 0.0 && loyalist > 0.0 && rebel >= loyalist * 2.0 {
            // Rebel victory
            resolve_rebel_win(state, cw, out);
        } else if elapsed >= 5000 {
            // Stalemate timeout
            resolve_stalemate(state, cw, out);
        }
    }
    */
}

/*
fn resolve_loyalist_win(
    state: &WorldState,
    cw: &CivilWarState,
    out: &mut Vec<WorldDelta>,
) {
    let fi = cw.faction_id;
    let surviving_strength = cw.loyalist_strength * 0.8;

    out.push(WorldDelta::SetMilitaryStrength {
        faction_id: fi,
        value: surviving_strength,
    });

    // Guild support consequences
    if let Some(supported_rebels) = cw.guild_supported_side {
        if supported_rebels {
            // Supported loser (rebels) -> -30 relation
            out.push(WorldDelta::AdjustRelationship {
                faction_id: fi,
                delta: -30.0,
            });
        } else {
            // Supported winner (loyalists) -> +20 relation
            out.push(WorldDelta::AdjustRelationship {
                faction_id: fi,
                delta: 20.0,
            });
        }
    }

    // Reduce unrest in faction regions
    for region in &state.regions {
        if region.faction_id == Some(fi) {
            out.push(WorldDelta::AdjustRegionUnrest {
                region_id: region.id,
                delta: -10.0,
            });
        }
    }

    out.push(WorldDelta::ResolveCivilWar {
        faction_id: fi,
        rebels_won: false,
    });
}

fn resolve_rebel_win(
    state: &WorldState,
    cw: &CivilWarState,
    out: &mut Vec<WorldDelta>,
) {
    let fi = cw.faction_id;
    let surviving_strength = cw.rebel_strength * 0.8;

    out.push(WorldDelta::SetMilitaryStrength {
        faction_id: fi,
        value: surviving_strength,
    });

    // Guild support consequences
    if let Some(supported_rebels) = cw.guild_supported_side {
        if supported_rebels {
            out.push(WorldDelta::SetRelationship { faction_id: fi, value: 20.0 });
        } else {
            out.push(WorldDelta::SetRelationship { faction_id: fi, value: -30.0 });
        }
    } else {
        out.push(WorldDelta::SetRelationship { faction_id: fi, value: 0.0 });
    }

    // New leadership resets stance to neutral, clears wars
    out.push(WorldDelta::SetDiplomaticStance {
        faction_id: fi,
        stance: DiplomaticStance::Neutral,
    });

    // End all wars for this faction
    for other in &state.factions {
        if other.id != fi && other.at_war_with.contains(&fi) {
            out.push(WorldDelta::EndWar {
                faction_a: fi,
                faction_b: other.id,
            });
        }
    }

    out.push(WorldDelta::ResolveCivilWar {
        faction_id: fi,
        rebels_won: true,
    });
}

fn resolve_stalemate(
    state: &WorldState,
    cw: &CivilWarState,
    out: &mut Vec<WorldDelta>,
) {
    let fi = cw.faction_id;

    // If room for new factions (< 6), faction splits
    if state.factions.len() < 6 {
        // Split territory: give rebels roughly half the regions
        let faction_region_ids: Vec<u32> = state.regions.iter()
            .filter(|r| r.faction_id == Some(fi))
            .map(|r| r.id)
            .collect();

        let split_count = faction_region_ids.len() / 2;
        // New faction gets first half of regions
        // (CreateFaction delta would include the new faction's properties)
        out.push(WorldDelta::CreateFaction {
            name: format!("{} Separatists", /* faction name */ "Faction"),
            military_strength: cw.rebel_strength * 0.7,
        });

        for (i, &region_id) in faction_region_ids.iter().enumerate() {
            if i < split_count {
                // new_faction_id would be assigned by apply phase
                out.push(WorldDelta::ChangeRegionOwner {
                    region_id,
                    new_owner: u32::MAX, // Sentinel: apply phase assigns actual ID
                });
            }
        }

        // Original faction retains loyalist strength
        out.push(WorldDelta::SetMilitaryStrength {
            faction_id: fi,
            value: cw.loyalist_strength * 0.7,
        });

        // New factions start at war with each other
        // (handled by apply phase when processing CreateFaction)
    } else {
        // Can't split — loyalists win by default
        out.push(WorldDelta::SetMilitaryStrength {
            faction_id: fi,
            value: (cw.loyalist_strength + cw.rebel_strength) * 0.4,
        });
    }

    // Guild support consequences for stalemate
    if let Some(supported_rebels) = cw.guild_supported_side {
        if supported_rebels {
            out.push(WorldDelta::AdjustRelationship {
                faction_id: fi,
                delta: -15.0,
            });
        }
        // If faction split, rebels (new faction) dislike guild
        // (handled by apply phase)
    }

    out.push(WorldDelta::ResolveCivilWar {
        faction_id: fi,
        rebels_won: false, // stalemate — neither side fully won
    });
}
*/
