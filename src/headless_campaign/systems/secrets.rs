//! Adventurer secret past system.
//!
//! Some adventurers have hidden histories that are revealed over time,
//! creating dramatic plot twists. Fires every 500 ticks (~50s game time).
//!
//! - 15% of new recruits get a random secret on assignment
//! - Suspicion grows +2 per tick when adventurer acts inconsistently
//! - Reveal triggers at suspicion > 80, or by random event
//! - Reveal effects vary by secret type (faction bonuses, betrayals, quests)

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often to tick the secrets system (in ticks).
const SECRETS_INTERVAL: u64 = 500;

/// Chance that a new recruit gets a secret past (15%).
const SECRET_ASSIGNMENT_CHANCE: f32 = 0.15;

/// Suspicion growth per tick when acting inconsistently.
const SUSPICION_GROWTH: f32 = 2.0;

/// Suspicion threshold that triggers reveal.
const REVEAL_THRESHOLD: f32 = 80.0;

/// Random reveal chance per tick (low — about 2%).
const RANDOM_REVEAL_CHANCE: f32 = 0.02;

/// Tick secrets every 500 ticks: assign secrets to new recruits,
/// grow suspicion, and trigger reveals.
pub fn tick_secrets(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % SECRETS_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let num_adventurers = state.adventurers.len();
    for i in 0..num_adventurers {
        if state.adventurers[i].status == AdventurerStatus::Dead {
            continue;
        }

        // --- Secret assignment: 15% of adventurers without a secret ---
        if state.adventurers[i].secret_past.is_none() {
            let roll = lcg_f32(&mut state.rng);
            if roll < SECRET_ASSIGNMENT_CHANCE {
                let secret_type = random_secret_type(&mut state.rng);
                state.adventurers[i].secret_past = Some(SecretPast {
                    secret_type,
                    revealed: false,
                    reveal_tick: None,
                    suspicion: 0.0,
                });
            }
            continue;
        }

        // --- Already revealed: skip ---
        let already_revealed = state.adventurers[i]
            .secret_past
            .as_ref()
            .map_or(true, |s| s.revealed);
        if already_revealed {
            continue;
        }

        // --- Grow suspicion based on inconsistent behavior ---
        let adv_id = state.adventurers[i].id;
        let secret_type = state.adventurers[i]
            .secret_past
            .as_ref()
            .unwrap()
            .secret_type;

        let suspicion_bump = compute_suspicion_bump(
            &state.adventurers[i],
            secret_type,
            &state.active_quests,
            &state.factions,
        );

        if let Some(ref mut secret) = state.adventurers[i].secret_past {
            secret.suspicion = (secret.suspicion + suspicion_bump).min(100.0);
        }

        let suspicion = state.adventurers[i]
            .secret_past
            .as_ref()
            .unwrap()
            .suspicion;

        // Emit suspicion rising event at key thresholds
        if suspicion >= 40.0 && (suspicion - suspicion_bump) < 40.0 {
            events.push(WorldEvent::SuspicionRising {
                adventurer_id: adv_id,
            });
        }
        if suspicion >= 60.0 && (suspicion - suspicion_bump) < 60.0 {
            events.push(WorldEvent::SuspicionRising {
                adventurer_id: adv_id,
            });
        }

        // --- Check reveal trigger ---
        let random_roll = lcg_f32(&mut state.rng);
        let should_reveal = suspicion > REVEAL_THRESHOLD || random_roll < RANDOM_REVEAL_CHANCE;

        if should_reveal {
            apply_reveal(state, i, events);
        }
    }
}

/// Pick a random SecretType from the LCG.
fn random_secret_type(rng: &mut u64) -> SecretType {
    let roll = lcg_next(rng) % 8;
    match roll {
        0 => SecretType::ExiledNoble,
        1 => SecretType::FormerAssassin,
        2 => SecretType::RunawayHeir,
        3 => SecretType::CursedBloodline,
        4 => SecretType::DeepCoverSpy,
        5 => SecretType::FallenPaladin,
        6 => SecretType::WitnessToCrime,
        _ => SecretType::HiddenMage,
    }
}

/// Compute how much suspicion grows this tick based on secret type and context.
fn compute_suspicion_bump(
    adv: &Adventurer,
    secret_type: SecretType,
    active_quests: &[ActiveQuest],
    factions: &[FactionState],
) -> f32 {
    let base = SUSPICION_GROWTH;

    // Check if adventurer is on a quest (by party assignment)
    let on_quest = adv.party_id.is_some()
        && active_quests
            .iter()
            .any(|q| q.dispatched_party_id == adv.party_id);

    match secret_type {
        SecretType::FormerAssassin => {
            // Suspicion rises during diplomatic/escort quests
            if on_quest {
                let on_diplomatic = active_quests.iter().any(|q| {
                    q.dispatched_party_id == adv.party_id
                        && matches!(q.request.quest_type, QuestType::Diplomatic | QuestType::Escort)
                });
                if on_diplomatic {
                    return base * 2.0;
                }
            }
            base * 0.5
        }
        SecretType::ExiledNoble => {
            // Suspicion rises when near their home faction
            if let Some(faction_id) = adv.faction_id {
                if faction_id < factions.len() {
                    return base * 1.5;
                }
            }
            base * 0.5
        }
        SecretType::DeepCoverSpy => {
            // Suspicion rises steadily — spies are hard to hide long-term
            base * 1.0
        }
        SecretType::CursedBloodline => {
            // Suspicion rises when stressed (curse leaks through)
            if adv.stress > 60.0 {
                return base * 2.0;
            }
            base * 0.3
        }
        SecretType::HiddenMage => {
            // Suspicion rises during combat quests (magic slips out)
            if on_quest {
                let on_combat = active_quests.iter().any(|q| {
                    q.dispatched_party_id == adv.party_id
                        && matches!(q.request.quest_type, QuestType::Combat | QuestType::Rescue)
                });
                if on_combat {
                    return base * 1.5;
                }
            }
            base * 0.5
        }
        SecretType::WitnessToCrime => {
            // Suspicion rises when faction with grudge is nearby
            base * 0.8
        }
        SecretType::RunawayHeir | SecretType::FallenPaladin => {
            // Moderate steady growth
            base * 0.7
        }
    }
}

/// Apply reveal effects for adventurer at index `idx`.
fn apply_reveal(state: &mut CampaignState, idx: usize, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;
    let adv_id = state.adventurers[idx].id;
    let secret_type = state.adventurers[idx]
        .secret_past
        .as_ref()
        .unwrap()
        .secret_type;

    // Mark as revealed
    if let Some(ref mut secret) = state.adventurers[idx].secret_past {
        secret.revealed = true;
        secret.reveal_tick = Some(tick);
    }

    let description = match secret_type {
        SecretType::ExiledNoble => {
            // +20 faction relation, noble quest chain potential
            if let Some(faction_id) = state.adventurers[idx].faction_id {
                if let Some(f) = state.factions.get_mut(faction_id) {
                    f.relation = (f.relation + 20.0).min(100.0);
                }
            } else if !state.factions.is_empty() {
                // Assign to first faction as connection
                let fi = (lcg_next(&mut state.rng) as usize) % state.factions.len();
                state.factions[fi].relation = (state.factions[fi].relation + 20.0).min(100.0);
            }
            state.adventurers[idx].loyalty = (state.adventurers[idx].loyalty + 10.0).min(100.0);
            format!(
                "{} is revealed to be an exiled noble! Faction relations improve.",
                state.adventurers[idx].name
            )
        }
        SecretType::FormerAssassin => {
            // +15 combat effectiveness, but a wanted poster appears
            state.adventurers[idx].stats.attack += 15.0;
            state.adventurers[idx].morale =
                (state.adventurers[idx].morale - 10.0).max(0.0);
            format!(
                "{} is exposed as a former assassin! Combat skills recognized, but enemies take notice.",
                state.adventurers[idx].name
            )
        }
        SecretType::DeepCoverSpy => {
            // Betrayal event — may steal gold and flee, or turn loyal
            let loyalty_roll = lcg_f32(&mut state.rng);
            if loyalty_roll < 0.4 {
                // Betrayal: steal gold and desert
                let stolen = (state.guild.gold * 0.1).min(500.0);
                state.guild.gold -= stolen;
                state.adventurers[idx].status = AdventurerStatus::Dead;
                events.push(WorldEvent::AdventurerDeserted {
                    adventurer_id: adv_id,
                    reason: "Revealed as a deep cover spy — fled with stolen gold".into(),
                });
                format!(
                    "{} was a deep cover spy and has betrayed the guild, stealing {:.0} gold!",
                    state.adventurers[idx].name, stolen
                )
            } else {
                // Turned loyal — bonus loyalty
                state.adventurers[idx].loyalty =
                    (state.adventurers[idx].loyalty + 25.0).min(100.0);
                format!(
                    "{} was a spy but has chosen genuine loyalty to the guild!",
                    state.adventurers[idx].name
                )
            }
        }
        SecretType::CursedBloodline => {
            // Powerful but -morale to party, cure quest potential
            state.adventurers[idx].stats.attack += 10.0;
            state.adventurers[idx].stats.defense += 10.0;
            state.adventurers[idx].stress =
                (state.adventurers[idx].stress + 20.0).min(100.0);
            // Reduce morale for party members
            if let Some(party_id) = state.adventurers[idx].party_id {
                for adv in &mut state.adventurers {
                    if adv.party_id == Some(party_id) && adv.id != adv_id {
                        adv.morale = (adv.morale - 10.0).max(0.0);
                    }
                }
            }
            format!(
                "{}'s cursed bloodline manifests! Powerful but unsettling to allies.",
                state.adventurers[idx].name
            )
        }
        SecretType::RunawayHeir => {
            // Inheritance quest potential, reputation boost
            state.guild.reputation += 10.0;
            state.adventurers[idx].loyalty =
                (state.adventurers[idx].loyalty + 15.0).min(100.0);
            format!(
                "{} is revealed as a runaway heir! An inheritance beckons.",
                state.adventurers[idx].name
            )
        }
        SecretType::FallenPaladin => {
            // Lost powers hint at restoration
            state.adventurers[idx].resolve =
                (state.adventurers[idx].resolve + 20.0).min(100.0);
            state.adventurers[idx].morale =
                (state.adventurers[idx].morale + 10.0).min(100.0);
            format!(
                "{}'s past as a fallen paladin comes to light. A path to redemption opens.",
                state.adventurers[idx].name
            )
        }
        SecretType::WitnessToCrime => {
            // A faction wants them silenced — danger + intrigue
            state.adventurers[idx].stress =
                (state.adventurers[idx].stress + 15.0).min(100.0);
            if !state.factions.is_empty() {
                let fi = (lcg_next(&mut state.rng) as usize) % state.factions.len();
                state.factions[fi].relation = (state.factions[fi].relation - 15.0).max(-100.0);
            }
            format!(
                "{} witnessed a terrible crime — a powerful faction wants them silenced!",
                state.adventurers[idx].name
            )
        }
        SecretType::HiddenMage => {
            // Concealed magical ability revealed — stat boost
            state.adventurers[idx].stats.attack += 12.0;
            state.adventurers[idx].stats.speed += 5.0;
            format!(
                "{} has been hiding magical abilities! Their true power is unleashed.",
                state.adventurers[idx].name
            )
        }
    };

    events.push(WorldEvent::SecretRevealed {
        adventurer_id: adv_id,
        secret_type,
        description: description.clone(),
    });

    // Generate a choice event for the player to react
    let choice_id = state.next_event_id;
    state.next_event_id += 1;

    let choice = ChoiceEvent {
        id: choice_id,
        source: ChoiceSource::RandomEvent,
        prompt: format!("Secret revealed: {}", description),
        options: vec![
            ChoiceOption {
                label: "Accept and integrate".into(),
                description: format!(
                    "Accept {}'s past and keep them in the guild",
                    state.adventurers[idx].name
                ),
                effects: vec![ChoiceEffect::Narrative(format!(
                    "Accept {}'s past and keep them in the guild",
                    state.adventurers[idx].name
                ))],
            },
            ChoiceOption {
                label: "Exile from the guild".into(),
                description: format!(
                    "Remove {} from the guild",
                    state.adventurers[idx].name
                ),
                effects: vec![ChoiceEffect::Narrative(format!(
                    "Remove {} from the guild",
                    state.adventurers[idx].name
                ))],
            },
            ChoiceOption {
                label: "Exploit the secret".into(),
                description: format!(
                    "Use {}'s secret for the guild's advantage",
                    state.adventurers[idx].name
                ),
                effects: vec![ChoiceEffect::Narrative(format!(
                    "Use {}'s secret for the guild's advantage",
                    state.adventurers[idx].name
                ))],
            },
        ],
        default_option: 0,
        deadline_ms: Some(tick + 1000),
        created_at_ms: state.elapsed_ms,
    };
    state.pending_choices.push(choice);
}
