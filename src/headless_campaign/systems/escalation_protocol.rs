//! Escalation protocol — every 400 ticks (~40s game time).
//!
//! Enemy factions dispatch elite response squads when the guild destroys
//! too many of their patrols, creating an arms-race dynamic. Patrol losses
//! decay over time, and escalation de-escalates if the guild stops attacking.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often (in ticks) the escalation system evaluates.
const ESCALATION_INTERVAL: u64 = 13;

/// Patrol losses decay by 1 per this many ticks of no new attacks.
const LOSS_DECAY_INTERVAL: u64 = 33;

/// If the guild stops attacking for this many ticks, escalation drops by 1.
const DE_ESCALATION_THRESHOLD: u64 = 67;

/// Maximum escalation level.
const MAX_ESCALATION: u32 = 5;

/// Evaluate escalation levels per faction every 400 ticks.
pub fn tick_escalation_protocol(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % ESCALATION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let n_factions = state.factions.len();
    if n_factions == 0 {
        return;
    }

    let current_tick = state.tick;

    for fi in 0..n_factions {
        // Only hostile or at-war factions participate in escalation
        let stance = state.factions[fi].diplomatic_stance;
        if !matches!(stance, DiplomaticStance::Hostile | DiplomaticStance::AtWar) {
            // Friendly/neutral/coalition factions reset escalation
            if state.factions[fi].escalation_level > 0 {
                state.factions[fi].escalation_level = 0;
                state.factions[fi].patrol_losses = 0;
                state.factions[fi].escalation_cooldown = 0;
            }
            continue;
        }

        // --- Patrol loss decay ---
        // If guild hasn't attacked recently, losses decay by 1 per LOSS_DECAY_INTERVAL
        let ticks_since_last_loss = current_tick.saturating_sub(state.factions[fi].last_patrol_loss_tick);
        if state.factions[fi].patrol_losses > 0
            && ticks_since_last_loss >= LOSS_DECAY_INTERVAL
        {
            // Decay proportional to how long it's been
            let decay = (ticks_since_last_loss / LOSS_DECAY_INTERVAL) as u32;
            let decay = decay.min(state.factions[fi].patrol_losses);
            if decay > 0 {
                state.factions[fi].patrol_losses =
                    state.factions[fi].patrol_losses.saturating_sub(decay);
            }
        }

        // --- De-escalation ---
        // If guild stops attacking for DE_ESCALATION_THRESHOLD ticks, drop level by 1
        if state.factions[fi].escalation_level > 0
            && ticks_since_last_loss >= DE_ESCALATION_THRESHOLD
            && state.factions[fi].escalation_cooldown == 0
        {
            state.factions[fi].escalation_level -= 1;
            // Set cooldown to prevent rapid de-escalation
            state.factions[fi].escalation_cooldown = (DE_ESCALATION_THRESHOLD / ESCALATION_INTERVAL) as u32;
            let new_level = state.factions[fi].escalation_level;
            events.push(WorldEvent::EscalationDecrease {
                faction_id: fi,
                new_level,
            });
            continue; // Don't escalate and de-escalate in same tick
        }

        // Tick down cooldown
        if state.factions[fi].escalation_cooldown > 0 {
            state.factions[fi].escalation_cooldown -= 1;
        }

        // --- Escalation evaluation ---
        let losses = state.factions[fi].patrol_losses;
        let current_level = state.factions[fi].escalation_level;

        let target_level = if losses >= 20 {
            5
        } else if losses >= 15 {
            4
        } else if losses >= 10 {
            3
        } else if losses >= 6 {
            2
        } else if losses >= 3 {
            1
        } else {
            0
        };

        let target_level = target_level.min(MAX_ESCALATION);

        // Only escalate upward (de-escalation handled above)
        if target_level > current_level && state.factions[fi].escalation_cooldown == 0 {
            state.factions[fi].escalation_level = target_level;

            events.push(WorldEvent::EscalationIncreased {
                faction_id: fi,
                new_level: target_level,
            });

            // Apply escalation effects based on new level
            apply_escalation_effects(state, fi, target_level, events);
        }
    }
}

/// Apply effects when a faction reaches a new escalation level.
fn apply_escalation_effects(
    state: &mut CampaignState,
    fi: usize,
    level: u32,
    events: &mut Vec<WorldEvent>,
) {
    let base_strength = state.factions[fi].military_strength;

    match level {
        1 => {
            // Level 1: Faction increases patrol size by 20%
            let boost = base_strength * 0.20;
            state.factions[fi].military_strength += boost;
            state.factions[fi].max_military_strength =
                (state.factions[fi].max_military_strength + boost).max(state.factions[fi].military_strength);
        }
        2 => {
            // Level 2: Faction sends scouts to track guild parties
            // Increase unrest in guild-controlled regions (scouts cause tension)
            let guild_faction_id = state.diplomacy.guild_faction_id;
            for region in &mut state.overworld.regions {
                if region.owner_faction_id == guild_faction_id {
                    region.unrest = (region.unrest + 3.0).min(100.0);
                }
            }
            // Small military boost from mobilized scouts
            state.factions[fi].military_strength += 5.0;
        }
        3 => {
            // Level 3: Elite response squad deployed (2x normal stats)
            let squad_power = base_strength * 0.3;
            state.factions[fi].military_strength += squad_power;
            state.factions[fi].max_military_strength =
                (state.factions[fi].max_military_strength + squad_power * 0.5).max(state.factions[fi].military_strength);

            // Increase global threat
            state.overworld.global_threat_level =
                (state.overworld.global_threat_level + 5.0).min(100.0);

            events.push(WorldEvent::EliteSquadDispatched {
                faction_id: fi,
                squad_power: squad_power * 2.0, // 2x normal stats
            });
        }
        4 => {
            // Level 4: Coordinated multi-squad pincer attack
            let squad_power = base_strength * 0.4;
            state.factions[fi].military_strength += squad_power;

            // Attack multiple guild regions simultaneously
            let guild_faction_id = state.diplomacy.guild_faction_id;
            let attack_power = squad_power * 0.15;
            for region in &mut state.overworld.regions {
                if region.owner_faction_id == guild_faction_id {
                    region.control = (region.control - attack_power).max(0.0);
                    region.unrest = (region.unrest + 5.0).min(100.0);
                }
            }

            state.overworld.global_threat_level =
                (state.overworld.global_threat_level + 8.0).min(100.0);

            events.push(WorldEvent::EliteSquadDispatched {
                faction_id: fi,
                squad_power: squad_power * 2.0,
            });
        }
        5 => {
            // Level 5: Faction champion dispatched (nemesis-tier enemy)
            let champion_power = base_strength * 0.6;
            state.factions[fi].military_strength += champion_power;
            state.factions[fi].max_military_strength =
                (state.factions[fi].max_military_strength + champion_power).max(state.factions[fi].military_strength);

            // Massive threat increase
            state.overworld.global_threat_level =
                (state.overworld.global_threat_level + 12.0).min(100.0);

            // Worsen relations further
            state.factions[fi].relationship_to_guild =
                (state.factions[fi].relationship_to_guild - 15.0).max(-100.0);

            // Attack guild territories hard
            let guild_faction_id = state.diplomacy.guild_faction_id;
            let attack_power = champion_power * 0.2;
            for region in &mut state.overworld.regions {
                if region.owner_faction_id == guild_faction_id {
                    region.control = (region.control - attack_power).max(0.0);
                    region.unrest = (region.unrest + 8.0).min(100.0);
                }
            }

            events.push(WorldEvent::EliteSquadDispatched {
                faction_id: fi,
                squad_power: champion_power * 3.0, // Nemesis-tier
            });
        }
        _ => {}
    }
}

/// Called by battle/quest systems when a guild party destroys a faction patrol.
/// Increments patrol_losses and updates the last-loss tick.
pub fn record_patrol_loss(state: &mut CampaignState, faction_id: usize) {
    if faction_id >= state.factions.len() {
        return;
    }
    state.factions[faction_id].patrol_losses += 1;
    state.factions[faction_id].last_patrol_loss_tick = state.tick;
}
