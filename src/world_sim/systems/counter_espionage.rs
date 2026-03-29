#![allow(unused)]
//! Counter-espionage system — every 7 ticks.
//!
//! Factions plant agents in the guild. The guild must detect and neutralize
//! them using counter-intelligence. Undetected agents steal gold, leak intel,
//! sabotage equipment, and boost enemy combat effectiveness.
//!
//! Ported from `crates/headless_campaign/src/systems/counter_espionage.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: factions: Vec<FactionState> on WorldState
//   FactionState { id, diplomatic_stance, relationship_to_guild }
//   DiplomaticStance enum (Hostile, AtWar, ...)
// NEEDS STATE: enemy_agents: Vec<EnemyAgentState> on WorldState
//   EnemyAgentState { id, faction_id, infiltration_level, detected, planted_tick, damage_done }
// NEEDS STATE: guild: GuildState on WorldState
//   GuildState { gold, reputation, inventory: Vec<Item> }
// NEEDS STATE: counter_intel_level: f32 on WorldState
// NEEDS STATE: guild_buildings: GuildBuildings on WorldState
//   GuildBuildings { war_room: u32 }
// NEEDS STATE: adventurers (for idle rogues on guard duty)
// NEEDS STATE: spies: Vec<SpyState> on WorldState (for turning double agents)

// NEEDS DELTA: PlantEnemyAgent { faction_id: u32, infiltration_level: f32 }
// NEEDS DELTA: GrowInfiltration { agent_id: u32, delta: f32 }
// NEEDS DELTA: SiphonGold { agent_id: u32, amount: f32 }
// NEEDS DELTA: AdjustRelationship { faction_id: u32, delta: f32 }
// NEEDS DELTA: DestroyInventoryItem { item_index: usize }
// NEEDS DELTA: AdjustGuildReputation { delta: f32 }
// NEEDS DELTA: DetectEnemyAgent { agent_id: u32 }
// NEEDS DELTA: RemoveEnemyAgent { agent_id: u32 }
// NEEDS DELTA: AdjustCounterIntel { delta: f32 }
// NEEDS DELTA: PlantSpy { target_faction_id: u32, cover: f32 }

/// Cadence: runs every 7 ticks.
const COUNTER_ESPIONAGE_INTERVAL: u64 = 7;

/// Chance (0-1) per hostile faction per tick to plant an agent.
const PLANT_CHANCE: f32 = 0.03;

/// Gold siphoned per tick per agent (scaled by infiltration_level / 100).
const SIPHON_GOLD_PER_TICK: f32 = 0.5;

/// Chance (0-1) per tick for a sabotage event (scaled by infiltration_level / 100).
const SABOTAGE_CHANCE: f32 = 0.02;

/// Base detection chance per tick: counter_intel_level / 100 * 0.10.
const BASE_DETECTION_RATE: f32 = 0.10;

/// Aggression boost per agent leaking intel.
const INTEL_LEAK_AGGRESSION: f32 = 3.0;

/// Combat effectiveness bonus percentage enemies get per agent leaking info.
const COMBAT_LEAK_BONUS_PCT: f32 = 5.0;

/// Infiltration growth per tick for an undetected agent.
const INFILTRATION_GROWTH: f32 = 2.0;

/// Counter-intel boost from War Room (per tier).
const WAR_ROOM_BOOST: f32 = 5.0;

/// Counter-intel boost per idle rogue adventurer on guard duty.
const ROGUE_GUARD_BOOST: f32 = 3.0;

/// Deterministic hash for pseudo-random decisions.
#[inline]
fn deterministic_roll(tick: u64, agent_id: u32, salt: u32) -> f32 {
    let mut h = tick
        .wrapping_mul(6364136223846793005)
        .wrapping_add(agent_id as u64)
        .wrapping_mul(2862933555777941757)
        .wrapping_add(salt as u64);
    h = h
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (h >> 33) as f32 / (1u64 << 31) as f32
}

pub fn compute_counter_espionage(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % COUNTER_ESPIONAGE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Once state.factions, state.enemy_agents, state.guild,
    // state.counter_intel_level, state.guild_buildings, state.adventurers exist,
    // enable this.

    /*
    // --- Phase 1: Hostile factions attempt to plant agents ---
    for faction in &state.factions {
        let is_hostile = matches!(
            faction.diplomatic_stance,
            DiplomaticStance::Hostile | DiplomaticStance::AtWar
        );
        if !is_hostile {
            continue;
        }

        let roll = deterministic_roll(state.tick, faction.id, 0);
        if roll < PLANT_CHANCE {
            out.push(WorldDelta::PlantEnemyAgent {
                faction_id: faction.id,
                infiltration_level: 5.0,
            });
        }
    }

    if state.enemy_agents.is_empty() {
        return;
    }

    // --- Compute effective counter-intel level ---
    let war_room_bonus = state.guild_buildings.war_room as f32 * WAR_ROOM_BOOST;

    let rogue_guard_count = state
        .adventurers
        .iter()
        .filter(|a| {
            a.status == AdventurerStatus::Idle
                && a.class_tags.iter().any(|t| t.to_lowercase().contains("rogue"))
        })
        .count();
    let rogue_bonus = rogue_guard_count as f32 * ROGUE_GUARD_BOOST;

    let effective_intel = (state.counter_intel_level + war_room_bonus + rogue_bonus).min(100.0);

    // --- Phase 2: Agent effects + detection ---
    for agent in &state.enemy_agents {
        if agent.detected {
            continue;
        }

        let inf_scale = agent.infiltration_level / 100.0;

        // Gold siphoning -- map to TransferGold (guild -> faction entity).
        let gold_siphoned = (SIPHON_GOLD_PER_TICK * inf_scale).min(state.guild.gold);
        if gold_siphoned > 0.0 {
            out.push(WorldDelta::SiphonGold {
                agent_id: agent.id,
                amount: gold_siphoned,
            });
        }

        // Intel leak: boost faction aggression.
        out.push(WorldDelta::AdjustRelationship {
            faction_id: agent.faction_id,
            delta: -INTEL_LEAK_AGGRESSION,
        });

        // Grow infiltration.
        out.push(WorldDelta::GrowInfiltration {
            agent_id: agent.id,
            delta: INFILTRATION_GROWTH,
        });

        // Sabotage check.
        let sab_roll = deterministic_roll(state.tick, agent.id, 1);
        if sab_roll < SABOTAGE_CHANCE * inf_scale {
            let sab_type_roll = deterministic_roll(state.tick, agent.id, 2);
            if !state.guild.inventory.is_empty() && sab_type_roll < 0.5 {
                let item_idx = (deterministic_roll(state.tick, agent.id, 3)
                    * state.guild.inventory.len() as f32) as usize
                    % state.guild.inventory.len();
                out.push(WorldDelta::DestroyInventoryItem {
                    item_index: item_idx,
                });
            } else {
                out.push(WorldDelta::AdjustGuildReputation { delta: -3.0 });
            }
        }

        // Detection check.
        let detect_chance = (effective_intel / 100.0) * BASE_DETECTION_RATE;
        let detect_roll = deterministic_roll(state.tick, agent.id, 4);
        if detect_roll < detect_chance {
            out.push(WorldDelta::DetectEnemyAgent {
                agent_id: agent.id,
            });
        }
    }
    */
}
