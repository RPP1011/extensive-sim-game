#![allow(unused)]
//! Faction AI — every 600 ticks (~60s).
//!
//! Each faction evaluates its situation and picks from multiple possible
//! actions based on scoring. Maps mutations to WorldDelta variants.
//!
//! Ported from `crates/headless_campaign/src/systems/faction_ai.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: factions: Vec<FactionState> on WorldState
// NEEDS STATE: FactionState { id, name, military_strength, max_military_strength,
//              diplomatic_stance, relationship_to_guild, coalition_member,
//              at_war_with, territory_size, recent_actions }
// NEEDS STATE: DiplomaticStance enum { AtWar, Hostile, Neutral, Friendly, Coalition }
// NEEDS STATE: guild_faction_id: usize on WorldState (or diplomacy sub-struct)
// NEEDS STATE: faction_ai_config on WorldState (decision_interval_ticks,
//              attack_power_fraction, territory_capture_control, hostile_strength_gain,
//              war_declaration_threshold, war_declaration_penalty,
//              neutral_control_gain, friendly_relationship_gain, max_recent_actions)

// NEEDS DELTA: SetDiplomaticStance { faction_id: u32, stance: DiplomaticStance }
// NEEDS DELTA: AdjustMilitaryStrength { faction_id: u32, delta: f32 }
// NEEDS DELTA: AdjustRelationship { faction_id: u32, delta: f32 }
// NEEDS DELTA: AdjustRegionControl { region_id: u32, delta: f32 }
// NEEDS DELTA: AdjustRegionUnrest { region_id: u32, delta: f32 }
// NEEDS DELTA: ChangeRegionOwner { region_id: u32, new_owner: u32 }
// NEEDS DELTA: DeclareWar { attacker_id: u32, defender_id: u32 }
// NEEDS DELTA: EndWar { faction_a: u32, faction_b: u32 }
// NEEDS DELTA: RecordFactionAction { faction_id: u32, tick: u64, action: String }

/// Cadence: every 600 ticks.
const FACTION_AI_INTERVAL: u64 = 600;

/// Default config values (mirrors FactionAiConfig defaults from campaign).
const ATTACK_POWER_FRACTION: f32 = 0.05;
const TERRITORY_CAPTURE_CONTROL: f32 = 30.0;
const HOSTILE_STRENGTH_GAIN: f32 = 2.0;
const WAR_DECLARATION_THRESHOLD: f32 = 60.0;
const WAR_DECLARATION_PENALTY: f32 = 20.0;
const NEUTRAL_CONTROL_GAIN: f32 = 2.0;
const FRIENDLY_RELATIONSHIP_GAIN: f32 = 1.0;

/// Deterministic hash for pseudo-random decisions (no mutable RNG needed).
/// Returns a float in [0, 1).
#[inline]
fn deterministic_roll(tick: u64, faction_id: u32, salt: u32) -> f32 {
    let mut h = tick.wrapping_mul(6364136223846793005)
        .wrapping_add(faction_id as u64)
        .wrapping_mul(2862933555777941757)
        .wrapping_add(salt as u64);
    h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (h >> 33) as f32 / (1u64 << 31) as f32
}

pub fn compute_faction_ai(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % FACTION_AI_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Faction-level decisions ---
    // Since WorldState doesn't yet have `factions`, this is structured to show
    // what deltas would be emitted once the state fields exist.
    //
    // The logic below is commented-out pseudocode that mirrors the original
    // system, rewritten to push deltas instead of mutating state.

    /*
    let guild_faction_id = state.guild_faction_id;

    for faction in &state.factions {
        let fi = faction.id;
        let strength = faction.military_strength;
        let max_strength = faction.max_military_strength;

        // --- Per-stance behavior ---
        match faction.diplomatic_stance {
            DiplomaticStance::AtWar => {
                // War costs strength — if too weak, forced ceasefire
                if strength < 20.0 {
                    out.push(WorldDelta::SetDiplomaticStance {
                        faction_id: fi,
                        stance: DiplomaticStance::Hostile,
                    });
                    out.push(WorldDelta::EndWar {
                        faction_a: fi,
                        faction_b: guild_faction_id,
                    });
                } else {
                    // Attack costs
                    let attack_cost = 3.0 + strength * 0.03;
                    out.push(WorldDelta::AdjustMilitaryStrength {
                        faction_id: fi,
                        delta: -attack_cost,
                    });

                    // Find weakest guild region to attack
                    if let Some(region) = state.regions.iter()
                        .filter(|r| r.faction_id == Some(guild_faction_id)
                                     && r.threat_level > 0.0)
                        .min_by(|a, b| a.threat_level.partial_cmp(&b.threat_level).unwrap())
                    {
                        let attack_power = strength * ATTACK_POWER_FRACTION;
                        out.push(WorldDelta::AdjustRegionControl {
                            region_id: region.id,
                            delta: -attack_power,
                        });
                        out.push(WorldDelta::AdjustRegionUnrest {
                            region_id: region.id,
                            delta: attack_power * 0.5,
                        });

                        // Conquest check: if region control would drop to 0
                        // (apply phase handles clamping; we push the ownership
                        //  change delta conditionally based on current control)
                        if region.threat_level - attack_power <= 0.0 {
                            out.push(WorldDelta::ChangeRegionOwner {
                                region_id: region.id,
                                new_owner: fi,
                            });
                            out.push(WorldDelta::AdjustMilitaryStrength {
                                faction_id: fi,
                                delta: 5.0,
                            });
                        }
                    }
                }
            }

            DiplomaticStance::Hostile => {
                out.push(WorldDelta::AdjustMilitaryStrength {
                    faction_id: fi,
                    delta: HOSTILE_STRENGTH_GAIN,
                });

                // May declare war if strong enough and relations bad
                if strength > WAR_DECLARATION_THRESHOLD
                    && faction.relationship_to_guild < -20.0
                {
                    out.push(WorldDelta::SetDiplomaticStance {
                        faction_id: fi,
                        stance: DiplomaticStance::AtWar,
                    });
                    out.push(WorldDelta::DeclareWar {
                        attacker_id: fi,
                        defender_id: guild_faction_id,
                    });
                    out.push(WorldDelta::AdjustRelationship {
                        faction_id: fi,
                        delta: -WAR_DECLARATION_PENALTY,
                    });
                }
            }

            DiplomaticStance::Neutral => {
                // Defend owned territory
                for region in &state.regions {
                    if region.faction_id == Some(fi) {
                        out.push(WorldDelta::AdjustRegionControl {
                            region_id: region.id,
                            delta: NEUTRAL_CONTROL_GAIN,
                        });
                    }
                }

                // Drift based on relationship
                if faction.relationship_to_guild > 40.0 {
                    out.push(WorldDelta::SetDiplomaticStance {
                        faction_id: fi,
                        stance: DiplomaticStance::Friendly,
                    });
                } else if faction.relationship_to_guild < -30.0 {
                    out.push(WorldDelta::SetDiplomaticStance {
                        faction_id: fi,
                        stance: DiplomaticStance::Hostile,
                    });
                }
            }

            DiplomaticStance::Friendly => {
                // Defend own territory
                for region in &state.regions {
                    if region.faction_id == Some(fi) {
                        out.push(WorldDelta::AdjustRegionControl {
                            region_id: region.id,
                            delta: NEUTRAL_CONTROL_GAIN * 0.5,
                        });
                    }
                }

                // Send military aid if guild under pressure
                let guild_under_pressure = state.regions.iter()
                    .any(|r| r.faction_id == Some(guild_faction_id)
                             && r.threat_level < 40.0);

                if guild_under_pressure && strength > 30.0 {
                    if let Some(region) = state.regions.iter()
                        .find(|r| r.faction_id == Some(guild_faction_id)
                                   && r.threat_level < 40.0)
                    {
                        let aid = strength * 0.03;
                        out.push(WorldDelta::AdjustRegionControl {
                            region_id: region.id,
                            delta: aid,
                        });
                        out.push(WorldDelta::AdjustMilitaryStrength {
                            faction_id: fi,
                            delta: -aid * 0.5,
                        });
                    }
                }

                // Relationship improvement
                out.push(WorldDelta::AdjustRelationship {
                    faction_id: fi,
                    delta: FRIENDLY_RELATIONSHIP_GAIN * 0.5,
                });

                // Drift to neutral if relations cool
                if faction.relationship_to_guild < 20.0 {
                    out.push(WorldDelta::SetDiplomaticStance {
                        faction_id: fi,
                        stance: DiplomaticStance::Neutral,
                    });
                }
            }

            DiplomaticStance::Coalition => {
                // Defend guild + own territory
                for region in &state.regions {
                    if region.faction_id == Some(guild_faction_id)
                        || region.faction_id == Some(fi)
                    {
                        out.push(WorldDelta::AdjustRegionControl {
                            region_id: region.id,
                            delta: NEUTRAL_CONTROL_GAIN,
                        });
                        out.push(WorldDelta::AdjustRegionUnrest {
                            region_id: region.id,
                            delta: -0.5,
                        });
                    }
                }

                // Counter-attack factions at war with guild
                for other in &state.factions {
                    if other.id == fi { continue; }
                    if other.diplomatic_stance == DiplomaticStance::AtWar {
                        if let Some(region) = state.regions.iter()
                            .find(|r| r.faction_id == Some(other.id)
                                       && r.threat_level > 10.0)
                        {
                            let attack = strength * 0.05;
                            out.push(WorldDelta::AdjustRegionControl {
                                region_id: region.id,
                                delta: -attack,
                            });
                            out.push(WorldDelta::AdjustMilitaryStrength {
                                faction_id: fi,
                                delta: -attack * 0.3,
                            });
                        }
                    }
                }
            }
        }

        // --- Natural strength regeneration ---
        let regen = (max_strength - strength).max(0.0) * 0.02;
        out.push(WorldDelta::AdjustMilitaryStrength {
            faction_id: fi,
            delta: regen,
        });

        // --- Relationship drift ---
        match faction.diplomatic_stance {
            DiplomaticStance::AtWar => {
                out.push(WorldDelta::AdjustRelationship {
                    faction_id: fi,
                    delta: -3.0,
                });
            }
            DiplomaticStance::Hostile => {
                out.push(WorldDelta::AdjustRelationship {
                    faction_id: fi,
                    delta: -1.0,
                });
            }
            DiplomaticStance::Coalition | DiplomaticStance::Friendly => {
                out.push(WorldDelta::AdjustRelationship {
                    faction_id: fi,
                    delta: 0.5,
                });
            }
            _ => {}
        }
    }

    // --- Faction-to-faction wars ---
    for faction in &state.factions {
        let fi = faction.id;
        for &target in &faction.at_war_with {
            if target == guild_faction_id { continue; }

            let attacker_strength = faction.military_strength;
            if attacker_strength < 10.0 { continue; }

            let defender_strength = state.factions.iter()
                .find(|f| f.id == target)
                .map(|f| f.military_strength)
                .unwrap_or(0.0);

            if let Some(region) = state.regions.iter()
                .find(|r| r.faction_id == Some(target) && r.threat_level > 5.0)
            {
                let attack = attacker_strength * 0.04;
                let defense = defender_strength * 0.02;
                let net = (attack - defense).max(0.0);

                out.push(WorldDelta::AdjustRegionControl {
                    region_id: region.id,
                    delta: -net,
                });
                out.push(WorldDelta::AdjustMilitaryStrength {
                    faction_id: fi,
                    delta: -attack * 0.3,
                });

                // Conquest check
                if region.threat_level - net <= 0.0 {
                    out.push(WorldDelta::ChangeRegionOwner {
                        region_id: region.id,
                        new_owner: fi,
                    });
                }
            }
        }
    }
    */
}
