//! Prisoner system — every 200 ticks.
//!
//! Manages captured enemy prisoners: escape attempts, upkeep costs,
//! and captured adventurer ransom events. Capture occurs during battle
//! resolution in quest_lifecycle.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// Prisoner upkeep cost per prisoner per tick (when the system fires).
const PRISONER_UPKEEP: f32 = 1.0;
/// Base escape chance per tick (5%).
const BASE_ESCAPE_CHANCE: f32 = 0.05;
/// Escape chance increase per 200-tick cycle held.
const ESCAPE_CHANCE_PER_TICK: f32 = 0.02;
/// Reputation penalty on prisoner escape.
const ESCAPE_REPUTATION_PENALTY: f32 = 5.0;
/// Chance per defeated enemy unit to be captured (15%).
const CAPTURE_CHANCE: f32 = 0.15;

pub fn tick_prisoners(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 7 != 0 || state.tick == 0 {
        return;
    }

    // --- Prisoner upkeep ---
    let upkeep = state.prisoners.len() as f32 * PRISONER_UPKEEP;
    if upkeep > 0.0 {
        state.guild.gold = (state.guild.gold - upkeep).max(0.0);
    }

    // --- Escape attempts ---
    let mut escaped_indices = Vec::new();
    for (i, prisoner) in state.prisoners.iter().enumerate() {
        let ticks_held = state.tick.saturating_sub(prisoner.captured_tick);
        let cycles_held = ticks_held / 200;
        // Base 5% + 2% per cycle held, capped at 80%
        let effective_chance = (prisoner.escape_chance + cycles_held as f32 * ESCAPE_CHANCE_PER_TICK)
            .min(0.8);

        let roll = lcg_f32(&mut state.rng);
        if roll < effective_chance {
            escaped_indices.push(i);
        }
    }

    // Process escapes in reverse order to keep indices valid.
    for &i in escaped_indices.iter().rev() {
        let prisoner = state.prisoners.remove(i);
        state.guild.reputation = (state.guild.reputation - ESCAPE_REPUTATION_PENALTY).max(0.0);
        events.push(WorldEvent::PrisonerEscaped {
            prisoner_id: prisoner.id,
            prisoner_name: prisoner.name.clone(),
            faction_id: prisoner.faction_id,
        });
    }

    // --- Captured adventurer ransom events ---
    // Factions holding our adventurers may offer to ransom them back.
    if !state.captured_adventurers.is_empty() {
        let mut ransom_offered = Vec::new();
        for &adv_id in &state.captured_adventurers {
            let roll = lcg_f32(&mut state.rng);
            // 10% chance per cycle that a faction offers ransom
            if roll < 0.10 {
                // Find which faction holds them (look up the adventurer's faction_id)
                if let Some(adv) = state.adventurers.iter().find(|a| a.id == adv_id) {
                    if let Some(faction_id) = adv.faction_id {
                        let ransom_cost = 50.0 + adv.stats.attack * 2.0 + adv.level as f32 * 5.0;
                        ransom_offered.push((adv_id, faction_id, ransom_cost));
                    }
                }
            }
        }

        for (adv_id, faction_id, ransom_cost) in ransom_offered {
            let adv_name = state
                .adventurers
                .iter()
                .find(|a| a.id == adv_id)
                .map(|a| a.name.clone())
                .unwrap_or_else(|| format!("Adventurer {}", adv_id));

            let choice_id = state.next_event_id;
            state.next_event_id += 1;

            let faction_name = state
                .factions
                .get(faction_id)
                .map(|f| f.name.clone())
                .unwrap_or_else(|| format!("Faction {}", faction_id));

            let choice = ChoiceEvent {
                id: choice_id,
                source: ChoiceSource::FactionDemand,
                prompt: format!(
                    "{} offers to return {} for {:.0} gold.",
                    faction_name, adv_name, ransom_cost
                ),
                options: vec![
                    ChoiceOption {
                        label: format!("Pay {:.0}g ransom", ransom_cost),
                        description: format!("Pay {:.0} gold to get {} back.", ransom_cost, adv_name),
                        effects: vec![ChoiceEffect::Gold(-ransom_cost)],
                    },
                    ChoiceOption {
                        label: "Refuse".into(),
                        description: "Refuse the ransom demand.".into(),
                        effects: vec![ChoiceEffect::Narrative(
                            "You refuse the ransom. Your adventurer remains captive.".into(),
                        )],
                    },
                ],
                default_option: 1,
                deadline_ms: Some(state.elapsed_ms + 67 * CAMPAIGN_TURN_SECS as u64 * 1000),
                created_at_ms: state.elapsed_ms,
            };

            events.push(WorldEvent::ChoicePresented {
                choice_id,
                prompt: choice.prompt.clone(),
                num_options: 2,
            });
            state.pending_choices.push(choice);
        }
    }
}

/// Called from quest_lifecycle when a battle ends in Victory.
/// Rolls capture chance for each defeated enemy unit.
/// `enemy_count` is estimated from enemy_strength.
pub fn try_capture_prisoners(
    state: &mut CampaignState,
    battle_quest_id: u32,
    enemy_strength: f32,
    events: &mut Vec<WorldEvent>,
) {
    // Estimate number of enemy units from strength (1 unit per 20 strength, min 1)
    let enemy_count = ((enemy_strength / 20.0).ceil() as u32).max(1);

    for _ in 0..enemy_count {
        let roll = lcg_f32(&mut state.rng);
        if roll < CAPTURE_CHANCE {
            let prisoner_id = state.next_prisoner_id;
            state.next_prisoner_id += 1;

            // Determine faction from the quest source
            let faction_id = state
                .active_quests
                .iter()
                .find(|q| q.id == battle_quest_id)
                .and_then(|q| q.request.source_faction_id)
                .unwrap_or(0);

            let strength = 10.0 + lcg_f32(&mut state.rng) * enemy_strength.min(50.0);
            let ransom_value = strength * 3.0 + 20.0;

            let names = [
                "Grulk", "Vorash", "Skarn", "Threga", "Moldrik",
                "Kezzik", "Yanthra", "Bolvek", "Drenna", "Ushnak",
            ];
            let name_idx = (lcg_next(&mut state.rng) as usize) % names.len();
            let name = format!("{} the Captive", names[name_idx]);

            let prisoner = Prisoner {
                id: prisoner_id,
                name: name.clone(),
                faction_id,
                strength,
                captured_tick: state.tick,
                ransom_value,
                escape_chance: BASE_ESCAPE_CHANCE,
                loyalty_shift: 0.0,
            };

            events.push(WorldEvent::PrisonerCaptured {
                prisoner_id,
                prisoner_name: name,
                faction_id,
            });

            state.prisoners.push(prisoner);
        }
    }
}

/// Called from quest_lifecycle when a battle ends in Defeat.
/// Each party member has a chance of being captured by the enemy faction.
pub fn try_capture_adventurers(
    state: &mut CampaignState,
    party_id: u32,
    quest_faction_id: Option<usize>,
    events: &mut Vec<WorldEvent>,
) {
    let faction_id = quest_faction_id.unwrap_or(0);
    let member_ids: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| a.party_id == Some(party_id) && a.status != AdventurerStatus::Dead)
        .map(|a| a.id)
        .collect();

    for mid in member_ids {
        let roll = lcg_f32(&mut state.rng);
        // 15% chance each surviving member gets captured instead of returning
        if roll < CAPTURE_CHANCE {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                adv.faction_id = Some(faction_id);
                // Don't mark as Dead — they're captured, not killed
                // Remove from party
                adv.party_id = None;
                adv.status = AdventurerStatus::Idle; // effectively "held"
            }
            state.captured_adventurers.push(mid);

            let adv_name = state
                .adventurers
                .iter()
                .find(|a| a.id == mid)
                .map(|a| a.name.clone())
                .unwrap_or_default();

            events.push(WorldEvent::AdventurerCaptured {
                adventurer_id: mid,
                adventurer_name: adv_name,
                faction_id,
            });
        }
    }
}
