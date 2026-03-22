//! Choice event generation — every 500 ticks (~50s).
//!
//! Generates branching decisions based on game state:
//! - Quest approach choices (combat quests get stealth/assault/negotiate)
//! - Progression picks (choose 1 of 3 unlocks at milestones)
//! - NPC encounter deals (merchant offers, faction proposals)
//! - Random world events (opportunities, crises)

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often to check for new choice events (ticks).
const CHOICE_CHECK_INTERVAL: u64 = 500;

/// Maximum simultaneous pending choices.
const MAX_PENDING_CHOICES: usize = 3;

/// Default deadline for choices (ms). ~60s to decide.
const DEFAULT_DEADLINE_MS: u64 = 60_000;

pub fn tick_choices(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % CHOICE_CHECK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.pending_choices.len() >= MAX_PENDING_CHOICES {
        return;
    }

    // Roll for which kind of choice to generate
    let roll = lcg_f32(&mut state.rng);

    if roll < 0.3 {
        try_generate_quest_branch(state, events);
    } else if roll < 0.5 {
        try_generate_npc_encounter(state, events);
    } else if roll < 0.7 {
        try_generate_world_event(state, events);
    }
    // 30% chance of no choice this interval — keeps them from flooding
}

/// Quest branch: a combat quest gets approach options that affect difficulty/reward.
fn try_generate_quest_branch(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Find a preparing quest that hasn't had a branch yet
    let quest = state.active_quests.iter().find(|q| {
        q.status == ActiveQuestStatus::Preparing
            && matches!(q.request.quest_type, QuestType::Combat | QuestType::Rescue)
            && !state.pending_choices.iter().any(|c| matches!(&c.source, ChoiceSource::QuestBranch { quest_id } if *quest_id == q.id))
    });

    let quest = match quest {
        Some(q) => q,
        None => return,
    };

    let quest_id = quest.id;
    let threat = quest.request.threat_level;
    let choice_id = state.next_event_id;
    state.next_event_id += 1;

    let choice = ChoiceEvent {
        id: choice_id,
        source: ChoiceSource::QuestBranch { quest_id },
        prompt: format!(
            "How should the party approach the {:?} quest (threat {:.0})?",
            quest.request.quest_type, threat
        ),
        options: vec![
            ChoiceOption {
                label: "Stealth Approach".into(),
                description: "Reduce threat by 30% but receive 20% less gold reward.".into(),
                effects: vec![
                    ChoiceEffect::ModifyQuestThreat { quest_id, multiplier: 0.7 },
                    ChoiceEffect::ModifyQuestReward { quest_id, gold_bonus: -threat * 0.4, rep_bonus: 0.0 },
                ],
            },
            ChoiceOption {
                label: "Direct Assault".into(),
                description: "Standard approach. No modifiers.".into(),
                effects: vec![
                    ChoiceEffect::Narrative("The party marches in directly.".into()),
                ],
            },
            ChoiceOption {
                label: "Negotiate First".into(),
                description: "Costs 20g for a diplomat. If it works, +50% gold reward. If not, +20% threat.".into(),
                effects: if lcg_f32(&mut state.rng) > 0.5 {
                    // Negotiation succeeds
                    vec![
                        ChoiceEffect::Gold(-20.0),
                        ChoiceEffect::ModifyQuestReward { quest_id, gold_bonus: threat * 1.0, rep_bonus: 2.0 },
                        ChoiceEffect::Narrative("Negotiation succeeded — favorable terms!".into()),
                    ]
                } else {
                    // Negotiation fails
                    vec![
                        ChoiceEffect::Gold(-20.0),
                        ChoiceEffect::ModifyQuestThreat { quest_id, multiplier: 1.2 },
                        ChoiceEffect::Narrative("Negotiation failed — they're on alert now.".into()),
                    ]
                },
            },
        ],
        default_option: 1, // Direct Assault = least investment
        deadline_ms: Some(state.elapsed_ms + DEFAULT_DEADLINE_MS),
        created_at_ms: state.elapsed_ms,
    };

    events.push(WorldEvent::ChoicePresented {
        choice_id,
        prompt: choice.prompt.clone(),
        num_options: choice.options.len(),
    });

    state.pending_choices.push(choice);
}

/// NPC encounter: a merchant or informant offers a deal.
fn try_generate_npc_encounter(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    if state.npc_relationships.is_empty() {
        return;
    }

    let npc_idx = (lcg_next(&mut state.rng) as usize) % state.npc_relationships.len();
    let npc = &state.npc_relationships[npc_idx];
    let npc_id = npc.npc_id;
    let npc_name = npc.npc_name.clone();

    let choice_id = state.next_event_id;
    state.next_event_id += 1;

    let choice = ChoiceEvent {
        id: choice_id,
        source: ChoiceSource::NpcEncounter { npc_id },
        prompt: format!("{} approaches with an offer.", npc_name),
        options: vec![
            ChoiceOption {
                label: "Accept Deal".into(),
                description: format!("Pay 30g. {} provides supplies and improves relations.", npc_name),
                effects: vec![
                    ChoiceEffect::Gold(-30.0),
                    ChoiceEffect::Supplies(25.0),
                    ChoiceEffect::FactionRelation {
                        faction_id: 0,
                        delta: 5.0,
                    },
                ],
            },
            ChoiceOption {
                label: "Decline Politely".into(),
                description: "No cost, no benefit. Relations unchanged.".into(),
                effects: vec![
                    ChoiceEffect::Narrative(format!("{} nods and departs.", npc_name)),
                ],
            },
            ChoiceOption {
                label: "Counter-Offer".into(),
                description: "Pay 15g for a smaller package. Slight relationship boost.".into(),
                effects: vec![
                    ChoiceEffect::Gold(-15.0),
                    ChoiceEffect::Supplies(10.0),
                    ChoiceEffect::FactionRelation {
                        faction_id: 0,
                        delta: 2.0,
                    },
                ],
            },
        ],
        default_option: 1, // Decline = zero investment
        deadline_ms: Some(state.elapsed_ms + DEFAULT_DEADLINE_MS),
        created_at_ms: state.elapsed_ms,
    };

    events.push(WorldEvent::ChoicePresented {
        choice_id,
        prompt: choice.prompt.clone(),
        num_options: choice.options.len(),
    });

    state.pending_choices.push(choice);
}

/// Random world event: opportunity or crisis that requires a response.
fn try_generate_world_event(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let choice_id = state.next_event_id;
    state.next_event_id += 1;

    let event_type = lcg_next(&mut state.rng) % 3;

    let choice = match event_type {
        0 => {
            // Wandering adventurer
            let level = 2 + (lcg_next(&mut state.rng) % 3) as u32;
            ChoiceEvent {
                id: choice_id,
                source: ChoiceSource::WorldEvent,
                prompt: format!("A level {} wandering adventurer offers to join the guild.", level),
                options: vec![
                    ChoiceOption {
                        label: "Hire (40g)".into(),
                        description: "A capable recruit joins immediately.".into(),
                        effects: vec![
                            ChoiceEffect::Gold(-40.0),
                            ChoiceEffect::AddAdventurer(Adventurer {
                                id: 0, // Will be reassigned
                                name: "Wanderer".into(),
                                archetype: "rogue".into(),
                                level,
                                xp: 0,
                                stats: AdventurerStats {
                                    max_hp: 55.0 + level as f32 * 5.0,
                                    attack: 14.0 + level as f32 * 2.0,
                                    defense: 7.0 + level as f32 * 1.5,
                                    speed: 13.0,
                                    ability_power: 5.0,
                                },
                                equipment: Equipment::default(),
                                traits: Vec::new(),
                                status: AdventurerStatus::Idle,
                                loyalty: 45.0,
                                stress: 20.0,
                                fatigue: 15.0,
                                injury: 0.0,
                                resolve: 50.0,
                                morale: 55.0,
                                party_id: None,
                                guild_relationship: 30.0,
                            }),
                        ],
                    },
                    ChoiceOption {
                        label: "Decline".into(),
                        description: "The wanderer moves on.".into(),
                        effects: vec![
                            ChoiceEffect::Narrative("The wanderer tips their hat and walks away.".into()),
                        ],
                    },
                ],
                default_option: 1, // Decline = no investment
                deadline_ms: Some(state.elapsed_ms + 30_000), // 30s to decide
                created_at_ms: state.elapsed_ms,
            }
        }
        1 => {
            // Regional crisis
            ChoiceEvent {
                id: choice_id,
                source: ChoiceSource::WorldEvent,
                prompt: "A nearby village reports bandit attacks. They request guild assistance.".into(),
                options: vec![
                    ChoiceOption {
                        label: "Send Aid (20g)".into(),
                        description: "Costs gold but greatly improves reputation.".into(),
                        effects: vec![
                            ChoiceEffect::Gold(-20.0),
                            ChoiceEffect::Reputation(8.0),
                        ],
                    },
                    ChoiceOption {
                        label: "Offer Advice".into(),
                        description: "No cost. Small reputation gain.".into(),
                        effects: vec![
                            ChoiceEffect::Reputation(2.0),
                        ],
                    },
                    ChoiceOption {
                        label: "Ignore".into(),
                        description: "Save resources but lose standing.".into(),
                        effects: vec![
                            ChoiceEffect::Reputation(-3.0),
                        ],
                    },
                ],
                default_option: 2, // Ignore = least investment
                deadline_ms: Some(state.elapsed_ms + DEFAULT_DEADLINE_MS),
                created_at_ms: state.elapsed_ms,
            }
        }
        _ => {
            // Trade caravan opportunity
            ChoiceEvent {
                id: choice_id,
                source: ChoiceSource::WorldEvent,
                prompt: "A trade caravan passes through. They offer bulk supplies at a discount.".into(),
                options: vec![
                    ChoiceOption {
                        label: "Buy Large (50g)".into(),
                        description: "50g for 80 supplies — great value.".into(),
                        effects: vec![
                            ChoiceEffect::Gold(-50.0),
                            ChoiceEffect::Supplies(80.0),
                        ],
                    },
                    ChoiceOption {
                        label: "Buy Small (20g)".into(),
                        description: "20g for 25 supplies — fair price.".into(),
                        effects: vec![
                            ChoiceEffect::Gold(-20.0),
                            ChoiceEffect::Supplies(25.0),
                        ],
                    },
                    ChoiceOption {
                        label: "Pass".into(),
                        description: "The caravan continues on its way.".into(),
                        effects: vec![
                            ChoiceEffect::Narrative("The caravan rumbles past without stopping.".into()),
                        ],
                    },
                ],
                default_option: 2, // Pass = no investment
                deadline_ms: Some(state.elapsed_ms + 45_000),
                created_at_ms: state.elapsed_ms,
            }
        }
    };

    events.push(WorldEvent::ChoicePresented {
        choice_id,
        prompt: choice.prompt.clone(),
        num_options: choice.options.len(),
    });

    state.pending_choices.push(choice);
}
