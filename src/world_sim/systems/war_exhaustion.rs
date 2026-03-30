//! War exhaustion system — every 3 ticks.
//!
//! Prolonged wars drain morale, gold, and public support. Factions accumulate
//! exhaustion from time at war, casualties, and gold spent. At threshold levels
//! (25/50/75/90) escalating penalties apply: combat debuffs, morale collapse,
//! forced peace-seeking, and eventual faction collapse with auto-ceasefire.
//!
//! Guild war exhaustion tracks the guild's own involvement in wars and penalizes
//! adventurer morale and recruitment difficulty.
//!
//! Ported from `crates/headless_campaign/src/systems/war_exhaustion.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

//   FactionState { id, diplomatic_stance, at_war_with, military_strength,
//                  relationship_to_guild, name }
//   WarExhaustionEntry { faction_id: u32, exhaustion_level: f32,
//                        war_start_tick: u64, casualties: u32, gold_spent: f32 }


/// Cadence: every 3 ticks.
const WAR_EXHAUSTION_INTERVAL: u64 = 3;

/// Base exhaustion gain per tick while at war.
const BASE_EXHAUSTION_PER_TICK: f32 = 0.5;
/// Extra exhaustion per casualty suffered.
const EXHAUSTION_PER_CASUALTY: f32 = 2.0;
/// Extra exhaustion per gold spent on war.
const EXHAUSTION_PER_GOLD: f32 = 0.1;
/// Exhaustion decay per tick while at peace.
const PEACE_DECAY_PER_TICK: f32 = 1.0;

/// Morale penalty applied to adventurers per 25 exhaustion (guild wars only).
const GUILD_MORALE_PENALTY_PER_25: f32 = 0.5;

pub fn compute_war_exhaustion(state: &WorldState, _out: &mut Vec<WorldDelta>) {
    if state.tick % WAR_EXHAUSTION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Once faction and war_exhaustion state fields exist, this system will
    // read them immutably and push adjustment deltas.

    /*
    let guild_faction_id = state.guild_faction_id;

    // --- Process each faction ---
    for faction in &state.factions {
        let fi = faction.id;
        let at_war = faction.diplomatic_stance == DiplomaticStance::AtWar
            || !faction.at_war_with.is_empty();

        let entry = state.war_exhaustion.iter().find(|e| e.faction_id == fi);

        if at_war {
            if entry.is_none() {
                // New war — create exhaustion entry
                out.push(WorldDelta::CreateWarExhaustion {
                    faction_id: fi,
                    war_start_tick: state.tick,
                });
            }

            let prev_level = entry.map(|e| e.exhaustion_level).unwrap_or(0.0);
            let casualties = entry.map(|e| e.casualties).unwrap_or(0);
            let gold_spent = entry.map(|e| e.gold_spent).unwrap_or(0.0);

            // Per-tick accumulation with diminishing marginal contribution
            let exhaustion_gain = BASE_EXHAUSTION_PER_TICK
                + casualties as f32 * EXHAUSTION_PER_CASUALTY * 0.01
                + gold_spent * EXHAUSTION_PER_GOLD * 0.001;

            out.push(WorldDelta::AdjustWarExhaustion {
                faction_id: fi,
                delta: exhaustion_gain,
            });

            let new_level = (prev_level + exhaustion_gain).min(100.0);

            // --- Threshold effects ---

            // 25: Minor combat penalty (-10% military strength)
            if prev_level < 25.0 && new_level >= 25.0 {
                out.push(WorldDelta::SetMilitaryStrength {
                    faction_id: fi,
                    value: faction.military_strength * 0.90,
                });
            }

            // 50: Morale crisis (-20% strength, +10 unrest in owned regions)
            if prev_level < 50.0 && new_level >= 50.0 {
                out.push(WorldDelta::SetMilitaryStrength {
                    faction_id: fi,
                    value: faction.military_strength * 0.80,
                });
                for region in &state.regions {
                    if region.faction_id == Some(fi) {
                        out.push(WorldDelta::AdjustRegionUnrest {
                            region_id: region.id,
                            delta: 10.0,
                        });
                    }
                }
            }

            // 75: Faction seeks peace — improve relationship to make ceasefire likely
            if prev_level < 75.0 && new_level >= 75.0 {
                out.push(WorldDelta::AdjustRelationship {
                    faction_id: fi,
                    delta: 15.0,
                });
            }

            // 90: Collapse — auto-ceasefire, massive strength loss, regions may defect
            if prev_level < 90.0 && new_level >= 90.0 {
                out.push(WorldDelta::SetDiplomaticStance {
                    faction_id: fi,
                    stance: DiplomaticStance::Hostile,
                });
                out.push(WorldDelta::EndWar {
                    faction_a: fi,
                    faction_b: guild_faction_id,
                });

                // Massive strength loss (retain 30%)
                out.push(WorldDelta::SetMilitaryStrength {
                    faction_id: fi,
                    value: faction.military_strength * 0.30,
                });

                // High-unrest regions defect to guild
                for region in &state.regions {
                    if region.faction_id == Some(fi) {
                        // Using threat_level as proxy for unrest
                        if region.threat_level > 50.0 {
                            out.push(WorldDelta::ChangeRegionOwner {
                                region_id: region.id,
                                new_owner: guild_faction_id,
                            });
                            out.push(WorldDelta::AdjustRegionUnrest {
                                region_id: region.id,
                                delta: -20.0,
                            });
                        }
                    }
                }

                // End all faction-to-faction wars
                for other in &state.factions {
                    if other.id != fi && other.at_war_with.contains(&fi) {
                        out.push(WorldDelta::EndWar {
                            faction_a: fi,
                            faction_b: other.id,
                        });
                    }
                }
            }
        } else if let Some(entry) = entry {
            // At peace — decay exhaustion
            if entry.exhaustion_level > 0.0 {
                out.push(WorldDelta::AdjustWarExhaustion {
                    faction_id: fi,
                    delta: -PEACE_DECAY_PER_TICK,
                });
            }
            // Remove entry if fully recovered (apply phase handles this)
            if entry.exhaustion_level - PEACE_DECAY_PER_TICK <= 0.0 {
                out.push(WorldDelta::RemoveWarExhaustion { faction_id: fi });
            }
        }
    }

    // --- Guild war exhaustion ---
    let guild_at_war = state.factions.iter()
        .any(|f| f.diplomatic_stance == DiplomaticStance::AtWar);

    let guild_entry = state.war_exhaustion.iter()
        .find(|e| e.faction_id == guild_faction_id);

    if guild_at_war {
        if guild_entry.is_none() {
            out.push(WorldDelta::CreateWarExhaustion {
                faction_id: guild_faction_id,
                war_start_tick: state.tick,
            });
        }

        let prev_level = guild_entry.map(|e| e.exhaustion_level).unwrap_or(0.0);
        let casualties = guild_entry.map(|e| e.casualties).unwrap_or(0);
        let gold_spent = guild_entry.map(|e| e.gold_spent).unwrap_or(0.0);

        let exhaustion_gain = BASE_EXHAUSTION_PER_TICK
            + casualties as f32 * EXHAUSTION_PER_CASUALTY * 0.01
            + gold_spent * EXHAUSTION_PER_GOLD * 0.001;

        out.push(WorldDelta::AdjustWarExhaustion {
            faction_id: guild_faction_id,
            delta: exhaustion_gain,
        });

        let new_level = (prev_level + exhaustion_gain).min(100.0);

        // Apply adventurer morale penalty based on guild exhaustion
        let morale_penalty = (new_level / 25.0).floor() * GUILD_MORALE_PENALTY_PER_25;
        if morale_penalty > 0.0 {
            // Penalize all guild-aligned entities (NPCs without faction)
            for entity in &state.entities {
                if let Some(ref npc) = entity.npc {
                    // Guild adventurers are those without a specific faction assignment
                    // In world_sim this maps to entities on team Friendly
                    if entity.team == crate::world_sim::state::WorldTeam::Friendly
                        && entity.alive
                    {
                        out.push(WorldDelta::AdjustMorale {
                            entity_id: entity.id,
                            delta: -morale_penalty,
                        });
                    }
                }
            }
        }
    } else if let Some(entry) = guild_entry {
        if entry.exhaustion_level > 0.0 {
            out.push(WorldDelta::AdjustWarExhaustion {
                faction_id: guild_faction_id,
                delta: -PEACE_DECAY_PER_TICK,
            });
        }
        if entry.exhaustion_level - PEACE_DECAY_PER_TICK <= 0.0 {
            out.push(WorldDelta::RemoveWarExhaustion {
                faction_id: guild_faction_id,
            });
        }
    }
    */
}
