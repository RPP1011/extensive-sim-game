//! Counter-espionage system — every 200 ticks.
//!
//! Factions plant agents in the guild. The guild must detect and neutralize
//! them using counter-intelligence. Undetected agents steal gold, leak intel,
//! sabotage equipment, and boost enemy combat effectiveness.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Cadence: runs every 200 ticks.
const COUNTER_ESPIONAGE_INTERVAL: u64 = 200;

/// Chance (0–1) per hostile faction per tick to plant an agent.
const PLANT_CHANCE: f32 = 0.03;

/// Gold siphoned per tick per agent (scaled by infiltration_level / 100).
const SIPHON_GOLD_PER_TICK: f32 = 0.5;

/// Chance (0–1) per tick for a sabotage event (scaled by infiltration_level / 100).
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

pub fn tick_counter_espionage(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % COUNTER_ESPIONAGE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Hostile factions attempt to plant agents ---
    let hostile_faction_ids: Vec<usize> = state
        .factions
        .iter()
        .filter(|f| {
            matches!(
                f.diplomatic_stance,
                DiplomaticStance::Hostile | DiplomaticStance::AtWar
            )
        })
        .map(|f| f.id)
        .collect();

    for &fid in &hostile_faction_ids {
        let roll = lcg_f32(&mut state.rng);
        if roll < PLANT_CHANCE {
            let agent_id = state.next_enemy_agent_id;
            state.next_enemy_agent_id += 1;
            state.enemy_agents.push(EnemyAgent {
                id: agent_id,
                faction_id: fid,
                infiltration_level: 5.0, // starts low
                detected: false,
                planted_tick: state.tick,
                damage_done: 0.0,
            });
        }
    }

    if state.enemy_agents.is_empty() {
        return;
    }

    // --- Compute effective counter-intel level ---
    let war_room_tier = state.guild_buildings.war_room;
    let war_room_bonus = war_room_tier as f32 * WAR_ROOM_BOOST;

    let rogue_guard_count = state
        .adventurers
        .iter()
        .filter(|a| {
            a.status == AdventurerStatus::Idle
                && a.archetype.to_lowercase().contains("rogue")
        })
        .count();
    let rogue_bonus = rogue_guard_count as f32 * ROGUE_GUARD_BOOST;

    let effective_intel = (state.counter_intel_level + war_room_bonus + rogue_bonus).min(100.0);

    // --- Phase 2: Agent effects + detection ---
    // Collect updates to avoid borrow conflicts.
    struct AgentUpdate {
        idx: usize,
        detected: bool,
        gold_siphoned: f32,
        sabotaged: bool,
        faction_id: usize,
        agent_id: u32,
    }

    let mut updates: Vec<AgentUpdate> = Vec::new();

    for (idx, agent) in state.enemy_agents.iter().enumerate() {
        if agent.detected {
            continue; // already detected, waiting for choice resolution
        }

        let inf_scale = agent.infiltration_level / 100.0;

        // Gold siphoning
        let gold_siphoned = SIPHON_GOLD_PER_TICK * inf_scale;

        // Sabotage check
        let sab_roll = lcg_f32(&mut state.rng);
        let sabotaged = sab_roll < SABOTAGE_CHANCE * inf_scale;

        // Detection check
        let detect_chance = (effective_intel / 100.0) * BASE_DETECTION_RATE;
        let detect_roll = lcg_f32(&mut state.rng);
        let detected = detect_roll < detect_chance;

        updates.push(AgentUpdate {
            idx,
            detected,
            gold_siphoned,
            sabotaged,
            faction_id: agent.faction_id,
            agent_id: agent.id,
        });
    }

    // Apply updates
    for update in &updates {
        let agent = &mut state.enemy_agents[update.idx];

        // Grow infiltration
        agent.infiltration_level = (agent.infiltration_level + INFILTRATION_GROWTH).min(100.0);

        // Siphon gold
        let siphoned = update.gold_siphoned.min(state.guild.gold);
        state.guild.gold -= siphoned;
        agent.damage_done += siphoned;

        // Intel leak: boost faction aggression
        if let Some(faction) = state.factions.iter_mut().find(|f| f.id == update.faction_id) {
            faction.relationship_to_guild =
                (faction.relationship_to_guild - INTEL_LEAK_AGGRESSION).max(-100.0);
        }

        // Sabotage
        if update.sabotaged {
            // Damage a random piece of equipment or reduce reputation
            let sab_roll = lcg_f32(&mut state.rng);
            if !state.guild.inventory.is_empty() && sab_roll < 0.5 {
                // Destroy a random inventory item
                let item_idx = (lcg_next(&mut state.rng) as usize) % state.guild.inventory.len();
                let item_name = state.guild.inventory[item_idx].name.clone();
                state.guild.inventory.remove(item_idx);
                events.push(WorldEvent::SabotagePrevented {
                    agent_id: update.agent_id,
                    description: format!("Enemy agent sabotaged equipment: {} destroyed", item_name),
                    prevented: false,
                });
            } else {
                // Damage guild reputation
                state.guild.reputation = (state.guild.reputation - 3.0).max(0.0);
                events.push(WorldEvent::SabotagePrevented {
                    agent_id: update.agent_id,
                    description: "Enemy agent spread damaging rumors about the guild".into(),
                    prevented: false,
                });
            }
            agent.damage_done += 5.0;
        }

        // Leak combat info
        events.push(WorldEvent::GuildIntelLeaked {
            faction_id: update.faction_id,
            combat_bonus_pct: COMBAT_LEAK_BONUS_PCT * (agent.infiltration_level / 100.0),
        });

        // Detection
        if update.detected {
            agent.detected = true;
            events.push(WorldEvent::EnemyAgentDetected {
                agent_id: update.agent_id,
                faction_id: update.faction_id,
                infiltration_level: agent.infiltration_level,
            });

            // Generate a choice event for the player to decide what to do
            let choice_id = state.next_event_id;
            state.next_event_id += 1;

            let faction_name = state
                .factions
                .iter()
                .find(|f| f.id == update.faction_id)
                .map(|f| f.name.clone())
                .unwrap_or_else(|| format!("Faction {}", update.faction_id));

            let choice = ChoiceEvent {
                id: choice_id,
                source: ChoiceSource::CounterEspionage,
                prompt: format!(
                    "An enemy agent from {} has been detected in the guild! Infiltration level: {:.0}%. What should we do?",
                    faction_name, agent.infiltration_level
                ),
                options: vec![
                    ChoiceOption {
                        label: "Capture".into(),
                        description: "Capture the agent for interrogation. Gain intel about the faction.".into(),
                        effects: vec![
                            ChoiceEffect::Narrative(format!(
                                "The {} spy was captured and interrogated, revealing enemy plans.",
                                faction_name
                            )),
                        ],
                    },
                    ChoiceOption {
                        label: "Expel".into(),
                        description: "Expel the agent and send a warning to their faction.".into(),
                        effects: vec![
                            ChoiceEffect::Narrative(format!(
                                "The {} spy was expelled with a stern warning.",
                                faction_name
                            )),
                        ],
                    },
                    ChoiceOption {
                        label: "Turn Double Agent".into(),
                        description: "Attempt to turn the agent. Risky but could provide ongoing intelligence.".into(),
                        effects: vec![
                            ChoiceEffect::Narrative(format!(
                                "An attempt was made to turn the {} spy into a double agent.",
                                faction_name
                            )),
                        ],
                    },
                ],
                default_option: 0, // capture by default
                deadline_ms: Some(state.elapsed_ms + 5000), // 50 ticks to decide
                created_at_ms: state.elapsed_ms,
            };

            events.push(WorldEvent::ChoicePresented {
                choice_id,
                prompt: choice.prompt.clone(),
                num_options: 3,
            });

            state.pending_choices.push(choice);
        }
    }
}

/// Handle the resolution of a counter-espionage choice.
/// Called from the choice resolution system when a CounterEspionage choice is resolved.
pub fn resolve_counter_espionage_choice(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    option_index: usize,
    agent_id_hint: Option<u32>,
) {
    // Find the first detected, unresolved enemy agent
    let agent_idx = if let Some(aid) = agent_id_hint {
        state.enemy_agents.iter().position(|a| a.id == aid && a.detected)
    } else {
        state.enemy_agents.iter().position(|a| a.detected)
    };

    let agent_idx = match agent_idx {
        Some(idx) => idx,
        None => return, // no detected agent to resolve
    };

    let agent = state.enemy_agents[agent_idx].clone();

    match option_index {
        0 => {
            // Capture — remove agent, gain intel
            state.enemy_agents.remove(agent_idx);
            // Boost counter-intel from captured intel
            state.counter_intel_level = (state.counter_intel_level + 5.0).min(100.0);
            events.push(WorldEvent::EnemyAgentCaptured {
                agent_id: agent.id,
                faction_id: agent.faction_id,
                intel_gained: agent.infiltration_level * 0.5,
            });
        }
        1 => {
            // Expel — remove agent, warn faction (small relation improvement)
            state.enemy_agents.remove(agent_idx);
            if let Some(faction) = state.factions.iter_mut().find(|f| f.id == agent.faction_id) {
                // Warning makes them less likely to try again temporarily
                faction.relationship_to_guild =
                    (faction.relationship_to_guild + 2.0).min(100.0);
            }
            events.push(WorldEvent::EnemyAgentExpelled {
                agent_id: agent.id,
                faction_id: agent.faction_id,
            });
        }
        2 => {
            // Turn double agent — risky
            let roll = lcg_f32(&mut state.rng);
            if roll < 0.4 {
                // Success! Convert to a guild spy in that faction
                state.enemy_agents.remove(agent_idx);
                let spy_id = state.next_spy_id;
                state.next_spy_id += 1;
                // Create a virtual spy (no adventurer attached, uses id 0)
                state.spies.push(Spy {
                    id: spy_id,
                    adventurer_id: 0, // virtual double agent
                    target_faction_id: agent.faction_id,
                    cover: 80.0, // good cover since they're already embedded
                    intel_gathered: 0.0,
                    planted_tick: state.tick,
                });
                state.counter_intel_level = (state.counter_intel_level + 10.0).min(100.0);
                events.push(WorldEvent::EnemyAgentCaptured {
                    agent_id: agent.id,
                    faction_id: agent.faction_id,
                    intel_gained: agent.infiltration_level,
                });
            } else {
                // Failed — agent escapes and faction becomes more aggressive
                state.enemy_agents.remove(agent_idx);
                if let Some(faction) = state.factions.iter_mut().find(|f| f.id == agent.faction_id)
                {
                    faction.relationship_to_guild =
                        (faction.relationship_to_guild - 10.0).max(-100.0);
                }
                events.push(WorldEvent::EnemyAgentExpelled {
                    agent_id: agent.id,
                    faction_id: agent.faction_id,
                });
            }
        }
        _ => {
            // Invalid option — default to capture
            state.enemy_agents.remove(agent_idx);
            events.push(WorldEvent::EnemyAgentCaptured {
                agent_id: agent.id,
                faction_id: agent.faction_id,
                intel_gained: 0.0,
            });
        }
    }
}
