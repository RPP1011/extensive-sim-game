//! Rival guild AI system.
//!
//! An AI-controlled competing guild that takes quests, recruits adventurers,
//! and competes for faction favor — creating urgency and strategic tension.
//!
//! Activates at tick 2000 (grace period) and ticks every 200 ticks.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Name bank for rival guilds (~20 options).
const RIVAL_NAMES: &[&str] = &[
    "The Iron Company",
    "Shadow Syndicate",
    "The Crimson Order",
    "Silver Talons",
    "The Obsidian Pact",
    "Warden Brotherhood",
    "The Ashen Collective",
    "Stormwatch Guild",
    "The Gilded Fang",
    "Nightfall Compact",
    "The Bronze Accord",
    "Duskblade Consortium",
    "The Ember Covenant",
    "Frostpeak Company",
    "The Jade Enclave",
    "Ironroot Alliance",
    "The Scarlet Vow",
    "Thornwatch Society",
    "The Violet Circle",
    "Wyrmguard Legion",
];

/// Grace period: rival guild activates at this tick.
const ACTIVATION_TICK: u64 = 2000;

/// Rival guild ticks every this many ticks.
const TICK_CADENCE: u64 = 200;

/// Run the rival guild AI for one tick.
///
/// Does nothing before tick 2000 (grace period). On tick 2000, initializes
/// the rival with stats scaled to the player's current position.
/// Every 200 ticks after activation, the rival takes actions.
pub fn tick_rival_guild(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    // Before activation tick, do nothing.
    if state.tick < ACTIVATION_TICK {
        return;
    }

    // Initialize on activation tick.
    if state.tick == ACTIVATION_TICK {
        initialize_rival(state, events);
        return;
    }

    // Only tick on cadence.
    if !state.rival_guild.active || state.tick % TICK_CADENCE != 0 {
        return;
    }

    // Run rival AI behaviors in order.
    rival_quest_competition(state, events);
    rival_reputation_growth(state, events);
    rival_faction_influence(state, events);
    rival_power_scaling(state);
    rival_sabotage(state, events);
}

/// Initialize the rival guild with stats scaled to the player's current position.
fn initialize_rival(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let name_idx = (lcg_next(&mut state.rng) as usize) % RIVAL_NAMES.len();
    let name = RIVAL_NAMES[name_idx].to_string();

    let personality_roll = lcg_next(&mut state.rng) % 3;
    let personality = match personality_roll {
        0 => RivalPersonality::Aggressive,
        1 => RivalPersonality::Diplomatic,
        _ => RivalPersonality::Opportunistic,
    };

    // Scale starting stats to player's current position so the rival is competitive.
    let player_rep = state.guild.reputation;
    let player_gold = state.guild.gold;
    let adventurer_count = state.adventurers.len() as u32;
    let avg_level = if state.adventurers.is_empty() {
        1.0
    } else {
        state.adventurers.iter().map(|a| a.level as f32).sum::<f32>()
            / state.adventurers.len() as f32
    };

    // Rival starts at 60-80% of player's position.
    let scale = 0.6 + lcg_f32(&mut state.rng) * 0.2;

    let faction_count = state.factions.len();
    let faction_relations = vec![30.0 + lcg_f32(&mut state.rng) * 10.0; faction_count];

    state.rival_guild = RivalGuildState {
        name: name.clone(),
        gold: player_gold * scale,
        reputation: (player_rep * scale).clamp(5.0, 95.0),
        adventurer_count: (adventurer_count as f32 * scale).max(1.0) as u32,
        power_level: avg_level * adventurer_count as f32 * scale,
        quests_completed: (state.completed_quests.len() as f32 * scale * 0.5) as u32,
        faction_relations,
        active: true,
        personality,
    };

    events.push(WorldEvent::CampaignMilestone {
        description: format!(
            "A rival guild has emerged: {} ({}). They compete for quests and faction favor.",
            name,
            match personality {
                RivalPersonality::Aggressive => "aggressive",
                RivalPersonality::Diplomatic => "diplomatic",
                RivalPersonality::Opportunistic => "opportunistic",
            }
        ),
    });
}

/// Quest competition: rival "takes" quests from the board.
///
/// Probability of stealing each quest is proportional to rival reputation.
/// Personality affects which quest types the rival prefers.
fn rival_quest_competition(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    if state.request_board.is_empty() {
        return;
    }

    let rival_rep = state.rival_guild.reputation;
    let personality = state.rival_guild.personality;

    // Base steal chance: 5-20% per quest depending on rival reputation.
    let base_chance = 0.05 + (rival_rep / 100.0) * 0.15;

    // Collect indices of quests to remove (iterate in reverse later).
    let mut stolen_indices = Vec::new();

    // Copy rng out to avoid borrow conflict.
    let mut rng = state.rng;

    for (i, quest) in state.request_board.iter().enumerate() {
        // Personality preference multiplier.
        let pref_mult = match personality {
            RivalPersonality::Aggressive => match quest.quest_type {
                QuestType::Combat => 1.8,
                QuestType::Rescue => 1.3,
                _ => 0.6,
            },
            RivalPersonality::Diplomatic => match quest.quest_type {
                QuestType::Diplomatic => 1.8,
                QuestType::Escort => 1.4,
                QuestType::Gather => 1.2,
                _ => 0.5,
            },
            RivalPersonality::Opportunistic => {
                // Take quest types the player does least.
                // Approximate by checking what's NOT on the active quest list.
                let player_has_type = state
                    .active_quests
                    .iter()
                    .any(|aq| aq.request.quest_type == quest.quest_type);
                if player_has_type { 0.4 } else { 1.6 }
            }
        };

        let chance = (base_chance * pref_mult).clamp(0.0, 0.5);
        let roll = lcg_f32(&mut rng);
        if roll < chance {
            stolen_indices.push(i);
        }
    }

    state.rng = rng;

    // Remove stolen quests in reverse order to preserve indices.
    for &i in stolen_indices.iter().rev() {
        let quest = state.request_board.remove(i);
        state.rival_guild.quests_completed += 1;

        // Rival gains gold and reputation from the quest.
        state.rival_guild.gold += quest.reward.gold * 0.8;
        state.rival_guild.reputation = (state.rival_guild.reputation
            + quest.reward.reputation * 0.5)
            .clamp(0.0, 100.0);

        // Rival gains faction relation if quest has a source faction.
        if let Some(fid) = quest.source_faction_id {
            if fid < state.rival_guild.faction_relations.len() {
                state.rival_guild.faction_relations[fid] =
                    (state.rival_guild.faction_relations[fid] + quest.reward.relation_change * 0.6)
                        .clamp(0.0, 100.0);
            }
        }

        events.push(WorldEvent::CampaignMilestone {
            description: format!(
                "{} completed a {:?} quest before you could take it.",
                state.rival_guild.name, quest.quest_type,
            ),
        });
    }
}

/// Reputation growth: rival gains reputation from completed quests.
///
/// If rival reputation exceeds player reputation, factions start favoring them.
fn rival_reputation_growth(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Passive reputation growth based on adventurer count and power.
    let growth = 0.1 + state.rival_guild.adventurer_count as f32 * 0.05;
    state.rival_guild.reputation = (state.rival_guild.reputation + growth).clamp(0.0, 100.0);

    // Rival recruits occasionally.
    let mut rng = state.rng;
    let recruit_roll = lcg_f32(&mut rng);
    state.rng = rng;

    if recruit_roll < 0.15 {
        state.rival_guild.adventurer_count += 1;
        state.rival_guild.power_level += 2.0 + lcg_f32(&mut state.rng) * 3.0;
    }

    // Warn when rival overtakes player reputation.
    let player_rep = state.guild.reputation;
    let rival_rep = state.rival_guild.reputation;

    if rival_rep > player_rep && (rival_rep - player_rep).abs() > 5.0 {
        // Only emit warning occasionally (every 1000 ticks).
        if state.tick % 1000 == 0 {
            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{} has a higher reputation ({:.0}) than your guild ({:.0}). \
                     Factions may start favoring them.",
                    state.rival_guild.name, rival_rep, player_rep,
                ),
            });
        }
    }
}

/// Faction influence: rival builds faction relations.
///
/// When rival has better relations with a faction, that faction's quest rewards
/// to the player decrease.
fn rival_faction_influence(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let personality = state.rival_guild.personality;

    // Diplomatic rivals build faction relations faster.
    let relation_growth = match personality {
        RivalPersonality::Diplomatic => 1.5,
        RivalPersonality::Aggressive => 0.3,
        RivalPersonality::Opportunistic => 0.8,
    };

    let mut rng = state.rng;

    for i in 0..state.rival_guild.faction_relations.len() {
        let growth = relation_growth * (0.5 + lcg_f32(&mut rng) * 1.0);
        state.rival_guild.faction_relations[i] =
            (state.rival_guild.faction_relations[i] + growth).clamp(0.0, 100.0);

        // If rival has significantly better relations, penalize player quest rewards.
        if i < state.factions.len() {
            let player_rel = state.factions[i].relationship_to_guild;
            let rival_rel = state.rival_guild.faction_relations[i];

            if rival_rel > player_rel + 15.0 {
                // Reduce faction's relation to player slightly (competitive pressure).
                state.factions[i].relationship_to_guild =
                    (state.factions[i].relationship_to_guild - 0.3).max(0.0);
            }
        }
    }

    state.rng = rng;

    // Emit event if rival dominates a faction.
    for i in 0..state.rival_guild.faction_relations.len().min(state.factions.len()) {
        let rival_rel = state.rival_guild.faction_relations[i];
        let player_rel = state.factions[i].relationship_to_guild;

        if rival_rel > 70.0 && rival_rel > player_rel + 20.0 && state.tick % 2000 == 0 {
            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{} has strong influence over the {} faction. \
                     Your standing with them is suffering.",
                    state.rival_guild.name, state.factions[i].name,
                ),
            });
        }
    }
}

/// Power scaling: rival power grows with completed quests.
fn rival_power_scaling(state: &mut CampaignState) {
    // Power grows logarithmically with quests completed.
    let quest_factor = (state.rival_guild.quests_completed as f32 + 1.0).ln();
    state.rival_guild.power_level += quest_factor * 0.2;
}

/// Sabotage events: rival occasionally interferes with the player.
///
/// Can raise prices, steal recruits, or spread rumors (reputation penalty).
/// May also trigger ChoiceEvents for the player to respond.
fn rival_sabotage(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let mut rng = state.rng;
    let sabotage_roll = lcg_f32(&mut rng);
    state.rng = rng;

    // Base 10% chance per tick cycle, higher for aggressive rivals.
    let sabotage_chance = match state.rival_guild.personality {
        RivalPersonality::Aggressive => 0.15,
        RivalPersonality::Diplomatic => 0.04,
        RivalPersonality::Opportunistic => 0.10,
    };

    if sabotage_roll >= sabotage_chance {
        return;
    }

    let action_roll = lcg_next(&mut state.rng) % 4;

    match action_roll {
        0 => {
            // Raise prices: reduce player gold.
            let loss = 5.0 + lcg_f32(&mut state.rng) * 15.0;
            state.guild.gold = (state.guild.gold - loss).max(0.0);
            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{} cornered the supply market, costing you {:.0} gold.",
                    state.rival_guild.name, loss,
                ),
            });
        }
        1 => {
            // Steal a recruit: reduce rival adventurer chance for player.
            state.rival_guild.adventurer_count += 1;
            state.rival_guild.power_level += 3.0;
            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{} poached a promising adventurer before you could recruit them.",
                    state.rival_guild.name,
                ),
            });
        }
        2 => {
            // Spread rumors: reputation penalty.
            let rep_loss = 1.0 + lcg_f32(&mut state.rng) * 3.0;
            state.guild.reputation = (state.guild.reputation - rep_loss).max(0.0);
            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{} spread rumors about your guild, costing you {:.1} reputation.",
                    state.rival_guild.name, rep_loss,
                ),
            });
        }
        _ => {
            // Provocation: create a choice event for the player.
            let choice_id = state.next_event_id;
            state.next_event_id += 1;

            let rival_name = state.rival_guild.name.clone();
            state.pending_choices.push(ChoiceEvent {
                id: choice_id,
                source: ChoiceSource::WorldEvent,
                prompt: format!(
                    "{} has challenged your guild publicly. How do you respond?",
                    rival_name,
                ),
                options: vec![
                    ChoiceOption {
                        label: "Ignore the provocation".into(),
                        description: "Take the high road. Lose a small amount of reputation."
                            .into(),
                        effects: vec![ChoiceEffect::Reputation(-2.0)],
                    },
                    ChoiceOption {
                        label: "Accept the challenge".into(),
                        description:
                            "Spend gold to mount a public counter-display. Gain reputation."
                                .into(),
                        effects: vec![
                            ChoiceEffect::Gold(-30.0),
                            ChoiceEffect::Reputation(5.0),
                        ],
                    },
                    ChoiceOption {
                        label: "Undermine them quietly".into(),
                        description:
                            "Spend gold on intelligence. Weaken the rival's next move.".into(),
                        effects: vec![ChoiceEffect::Gold(-20.0)],
                    },
                ],
                default_option: 0,
                deadline_ms: Some(state.elapsed_ms + 5000),
                created_at_ms: state.elapsed_ms,
            });

            events.push(WorldEvent::ChoicePresented {
                choice_id,
                prompt: format!("{} has challenged your guild publicly.", rival_name),
                num_options: 3,
            });
        }
    }
}
