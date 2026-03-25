//! Political intrigue system — fires every 500 ticks.
//!
//! Generates court intrigues within factions experiencing instability
//! (civil war, low strength, recent leadership changes). The guild can
//! choose to support claimants, expose scandals, exploit chaos, or stay
//! neutral via choice events.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often to check for new intrigues and resolve existing ones.
const INTRIGUE_INTERVAL: u64 = 17;

/// Base chance per qualifying faction per tick of spawning an intrigue.
const BASE_INTRIGUE_CHANCE: f32 = 0.05;

/// Minimum ticks before an intrigue resolves.
const MIN_RESOLUTION_TICKS: u64 = 33;

/// Maximum ticks before an intrigue resolves.
const MAX_RESOLUTION_TICKS: u64 = 67;

/// Maximum concurrent active (unresolved) intrigues.
const MAX_ACTIVE_INTRIGUES: usize = 5;

pub fn tick_intrigue(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % INTRIGUE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Resolve matured intrigues ---
    resolve_intrigues(state, events);

    // --- Generate new intrigues ---
    generate_intrigues(state, events);
}

// ---------------------------------------------------------------------------
// Intrigue generation
// ---------------------------------------------------------------------------

/// A faction qualifies for intrigue if it has internal instability:
/// - At war (civil unrest)
/// - Low military strength (< 40% of max)
/// - Recent hostile stance changes
fn faction_qualifies(faction: &FactionState) -> bool {
    let strength_ratio = if faction.max_military_strength > 0.0 {
        faction.military_strength / faction.max_military_strength
    } else {
        1.0
    };

    let at_war = !faction.at_war_with.is_empty();
    let low_strength = strength_ratio < 0.4;
    let hostile = faction.diplomatic_stance == DiplomaticStance::Hostile
        || faction.diplomatic_stance == DiplomaticStance::AtWar;

    at_war || low_strength || hostile
}

fn generate_intrigues(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let active_count = state.intrigues.iter().filter(|i| !i.resolved).count();
    if active_count >= MAX_ACTIVE_INTRIGUES {
        return;
    }

    let n_factions = state.factions.len();
    if n_factions == 0 {
        return;
    }

    // Collect qualifying faction IDs first to avoid borrow issues.
    let qualifying: Vec<usize> = (0..n_factions)
        .filter(|&fi| {
            let qualifies = faction_qualifies(&state.factions[fi]);
            // Don't spawn multiple active intrigues in the same faction.
            let already_active = state
                .intrigues
                .iter()
                .any(|i| !i.resolved && i.faction_id == fi);
            qualifies && !already_active
        })
        .collect();

    for fi in qualifying {
        let roll = lcg_f32(&mut state.rng);
        if roll >= BASE_INTRIGUE_CHANCE {
            continue;
        }

        // Pick intrigue type based on faction situation.
        let intrigue_type = pick_intrigue_type(state, fi);

        // Generate participants.
        let participants = generate_participants(state, &intrigue_type);

        // Resolution tick: random between MIN and MAX.
        let duration_range = MAX_RESOLUTION_TICKS - MIN_RESOLUTION_TICKS;
        let duration =
            MIN_RESOLUTION_TICKS + (lcg_next(&mut state.rng) as u64 % (duration_range + 1));
        let resolution_tick = state.tick + duration;

        let id = state.next_intrigue_id;
        state.next_intrigue_id += 1;

        let faction_name = state.factions[fi].name.clone();
        let type_label = intrigue_type.label().to_string();

        let intrigue = PoliticalIntrigue {
            id,
            faction_id: fi,
            intrigue_type,
            participants: participants.clone(),
            guild_involvement: 0.0,
            resolution_tick,
            resolved: false,
        };
        state.intrigues.push(intrigue);

        // Emit event.
        let description = format!(
            "{} in the {} faction involving {}",
            type_label,
            faction_name,
            participants.join(", ")
        );
        events.push(WorldEvent::IntrigueStarted {
            intrigue_id: id,
            faction_id: fi,
            intrigue_type: type_label.clone(),
            description: description.clone(),
        });

        // Present choice to the guild.
        let choice = build_intrigue_choice(state, id, fi, &intrigue_type, &participants);
        events.push(WorldEvent::ChoicePresented {
            choice_id: choice.id,
            prompt: choice.prompt.clone(),
            num_options: choice.options.len(),
        });
        state.pending_choices.push(choice);
    }
}

fn pick_intrigue_type(state: &mut CampaignState, faction_id: usize) -> IntrigueType {
    let faction = &state.factions[faction_id];
    let strength_ratio = if faction.max_military_strength > 0.0 {
        faction.military_strength / faction.max_military_strength
    } else {
        1.0
    };

    // Build weighted pool based on situation.
    let mut pool: Vec<(IntrigueType, u32)> = Vec::new();

    if strength_ratio < 0.3 {
        pool.push((IntrigueType::PowerGrab, 3));
        pool.push((IntrigueType::SuccessionDispute, 3));
    } else {
        pool.push((IntrigueType::PowerGrab, 1));
        pool.push((IntrigueType::SuccessionDispute, 1));
    }

    if !faction.at_war_with.is_empty() {
        pool.push((IntrigueType::SecretAlliance, 2));
        pool.push((IntrigueType::Assassination, 2));
    } else {
        pool.push((IntrigueType::SecretAlliance, 1));
        pool.push((IntrigueType::Assassination, 1));
    }

    pool.push((IntrigueType::NobleRivalry, 2));
    pool.push((IntrigueType::CourtScandal, 2));

    let total: u32 = pool.iter().map(|(_, w)| w).sum();
    let pick = lcg_next(&mut state.rng) % total;
    let mut cumulative = 0u32;
    for (itype, w) in &pool {
        cumulative += w;
        if pick < cumulative {
            return *itype;
        }
    }
    IntrigueType::NobleRivalry
}

fn generate_participants(state: &mut CampaignState, intrigue_type: &IntrigueType) -> Vec<String> {
    let first_names = [
        "Lord Aldric",
        "Lady Elara",
        "Duke Varen",
        "Countess Miriel",
        "Baron Thorne",
        "Marchioness Sable",
        "Viscount Caelum",
        "Dame Isolde",
        "Sir Roderick",
        "Archduke Fenwick",
    ];
    let count = match intrigue_type {
        IntrigueType::SuccessionDispute => 2,
        IntrigueType::NobleRivalry => 2,
        IntrigueType::CourtScandal => 1,
        IntrigueType::PowerGrab => 1,
        IntrigueType::SecretAlliance => 2,
        IntrigueType::Assassination => 2,
    };

    let mut names = Vec::with_capacity(count);
    for _ in 0..count {
        let idx = (lcg_next(&mut state.rng) as usize) % first_names.len();
        let name = first_names[idx].to_string();
        if !names.contains(&name) {
            names.push(name);
        } else {
            // Avoid duplicate — pick next available.
            for offset in 1..first_names.len() {
                let alt = first_names[(idx + offset) % first_names.len()].to_string();
                if !names.contains(&alt) {
                    names.push(alt);
                    break;
                }
            }
        }
    }
    names
}

// ---------------------------------------------------------------------------
// Choice building
// ---------------------------------------------------------------------------

fn build_intrigue_choice(
    state: &mut CampaignState,
    intrigue_id: u32,
    faction_id: usize,
    intrigue_type: &IntrigueType,
    participants: &[String],
) -> ChoiceEvent {
    let choice_id = state.next_event_id;
    state.next_event_id += 1;

    let faction_name = state.factions[faction_id].name.clone();
    let type_label = intrigue_type.label();

    let prompt = match intrigue_type {
        IntrigueType::SuccessionDispute => format!(
            "A succession dispute has erupted in {}. {} and {} both claim the right to lead. How does the guild respond?",
            faction_name,
            participants.first().unwrap_or(&"Unknown".to_string()),
            participants.get(1).unwrap_or(&"Unknown".to_string()),
        ),
        IntrigueType::NobleRivalry => format!(
            "Two noble houses in {} are feuding. {} and {} compete for court influence. The guild could intervene.",
            faction_name,
            participants.first().unwrap_or(&"Unknown".to_string()),
            participants.get(1).unwrap_or(&"Unknown".to_string()),
        ),
        IntrigueType::CourtScandal => format!(
            "A court scandal rocks {}! {} is implicated in corruption. The guild has evidence that could be leveraged.",
            faction_name,
            participants.first().unwrap_or(&"Unknown".to_string()),
        ),
        IntrigueType::PowerGrab => format!(
            "{} is attempting to seize power in the weakened {} faction. The guild could tip the balance.",
            participants.first().unwrap_or(&"Unknown".to_string()),
            faction_name,
        ),
        IntrigueType::SecretAlliance => format!(
            "Whispers of a secret alliance within {} between {} and {}. The guild has learned of the pact.",
            faction_name,
            participants.first().unwrap_or(&"Unknown".to_string()),
            participants.get(1).unwrap_or(&"Unknown".to_string()),
        ),
        IntrigueType::Assassination => format!(
            "An assassination plot in {} targets {}. {} may be behind it. The guild can intervene.",
            faction_name,
            participants.first().unwrap_or(&"Unknown".to_string()),
            participants.get(1).unwrap_or(&"Unknown".to_string()),
        ),
    };

    let options = match intrigue_type {
        IntrigueType::SuccessionDispute => vec![
            ChoiceOption {
                label: format!("Support {}", participants.first().unwrap_or(&"claimant A".to_string())),
                description: "Back the first claimant. If they win, gain faction favor and trade benefits.".into(),
                effects: vec![
                    ChoiceEffect::FactionRelation { faction_id, delta: 10.0 },
                    ChoiceEffect::Reputation(5.0),
                    ChoiceEffect::Gold(-30.0),
                ],
            },
            ChoiceOption {
                label: format!("Support {}", participants.get(1).unwrap_or(&"claimant B".to_string())),
                description: "Back the second claimant. A riskier bet with potentially greater reward.".into(),
                effects: vec![
                    ChoiceEffect::FactionRelation { faction_id, delta: 10.0 },
                    ChoiceEffect::Gold(-30.0),
                ],
            },
            ChoiceOption {
                label: "Exploit the chaos".into(),
                description: "Use the confusion to acquire territory and resources, but damage reputation.".into(),
                effects: vec![
                    ChoiceEffect::Gold(80.0),
                    ChoiceEffect::Supplies(30.0),
                    ChoiceEffect::Reputation(-10.0),
                    ChoiceEffect::FactionRelation { faction_id, delta: -10.0 },
                ],
            },
            ChoiceOption {
                label: "Stay neutral".into(),
                description: "The guild takes no side. Safe but no benefit.".into(),
                effects: vec![],
            },
        ],
        IntrigueType::CourtScandal => vec![
            ChoiceOption {
                label: "Expose the scandal".into(),
                description: "Publicly reveal the corruption. Gain reputation and faction gratitude.".into(),
                effects: vec![
                    ChoiceEffect::Reputation(15.0),
                    ChoiceEffect::FactionRelation { faction_id, delta: 15.0 },
                ],
            },
            ChoiceOption {
                label: "Blackmail the implicated".into(),
                description: "Use evidence for leverage. Gold now, enemies later.".into(),
                effects: vec![
                    ChoiceEffect::Gold(100.0),
                    ChoiceEffect::Reputation(-15.0),
                    ChoiceEffect::FactionRelation { faction_id, delta: -5.0 },
                ],
            },
            ChoiceOption {
                label: "Stay neutral".into(),
                description: "Not the guild's business.".into(),
                effects: vec![],
            },
        ],
        _ => vec![
            ChoiceOption {
                label: format!("Support {}", participants.first().unwrap_or(&"the faction".to_string())),
                description: format!("Intervene in the {}. Costs gold but builds faction relations.", type_label),
                effects: vec![
                    ChoiceEffect::FactionRelation { faction_id, delta: 10.0 },
                    ChoiceEffect::Gold(-40.0),
                    ChoiceEffect::Reputation(5.0),
                ],
            },
            ChoiceOption {
                label: "Exploit the chaos".into(),
                description: "Take advantage of the disorder for material gain.".into(),
                effects: vec![
                    ChoiceEffect::Gold(60.0),
                    ChoiceEffect::Reputation(-10.0),
                    ChoiceEffect::FactionRelation { faction_id, delta: -10.0 },
                ],
            },
            ChoiceOption {
                label: "Stay neutral".into(),
                description: "The guild does not get involved.".into(),
                effects: vec![],
            },
        ],
    };

    // Default option is always the last one (stay neutral).
    let default_option = options.len() - 1;

    // Deadline: ~167 turns (~500s game time) to decide.
    let deadline_ms = Some(state.elapsed_ms + 167 * CAMPAIGN_TURN_SECS as u64 * 1000);

    ChoiceEvent {
        id: choice_id,
        source: ChoiceSource::PoliticalIntrigue { intrigue_id },
        prompt,
        options,
        default_option,
        deadline_ms,
        created_at_ms: state.elapsed_ms,
    }
}

// ---------------------------------------------------------------------------
// Intrigue resolution
// ---------------------------------------------------------------------------

fn resolve_intrigues(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let current_tick = state.tick;

    // Collect intrigues ready to resolve.
    let to_resolve: Vec<usize> = state
        .intrigues
        .iter()
        .enumerate()
        .filter(|(_, i)| !i.resolved && current_tick >= i.resolution_tick)
        .map(|(idx, _)| idx)
        .collect();

    for idx in to_resolve {
        state.intrigues[idx].resolved = true;

        let intrigue = state.intrigues[idx].clone();
        let faction_id = intrigue.faction_id;

        let outcome = if intrigue.guild_involvement > 20.0 {
            // Guild-backed winner — favorable outcome.
            if faction_id < state.factions.len() {
                state.factions[faction_id].relationship_to_guild =
                    (state.factions[faction_id].relationship_to_guild + 20.0).min(100.0);
            }
            state.guild.gold += 50.0;
            format!(
                "The guild's support paid off. {} resolved favorably. +20 faction relation, +50 gold trade bonus.",
                intrigue.intrigue_type.label()
            )
        } else if intrigue.guild_involvement < -20.0 {
            // Guild exploited — bad outcome if caught.
            let caught = lcg_f32(&mut state.rng) < 0.4;
            if caught {
                if faction_id < state.factions.len() {
                    state.factions[faction_id].relationship_to_guild =
                        (state.factions[faction_id].relationship_to_guild - 20.0).max(-100.0);
                }
                format!(
                    "The guild's exploitation was discovered. {} resolved with hostility. -20 faction relation.",
                    intrigue.intrigue_type.label()
                )
            } else {
                format!(
                    "The guild profited from the chaos undetected. {} resolved.",
                    intrigue.intrigue_type.label()
                )
            }
        } else if intrigue.guild_involvement.abs() > 5.0 {
            // Guild moderately involved — mixed results.
            // Coin flip on whether the backed side won.
            let backed_won = lcg_f32(&mut state.rng) < 0.5;
            if backed_won {
                if faction_id < state.factions.len() {
                    state.factions[faction_id].relationship_to_guild =
                        (state.factions[faction_id].relationship_to_guild + 10.0).min(100.0);
                }
                format!(
                    "The guild's ally prevailed. {} resolved. +10 faction relation.",
                    intrigue.intrigue_type.label()
                )
            } else {
                if faction_id < state.factions.len() {
                    state.factions[faction_id].relationship_to_guild =
                        (state.factions[faction_id].relationship_to_guild - 10.0).max(-100.0);
                }
                format!(
                    "The guild's ally lost. {} resolved unfavorably. -10 faction relation.",
                    intrigue.intrigue_type.label()
                )
            }
        } else {
            // Neutral — no effect.
            format!(
                "The guild stayed out of it. {} resolved without guild influence.",
                intrigue.intrigue_type.label()
            )
        };

        events.push(WorldEvent::IntrigueResolved {
            intrigue_id: intrigue.id,
            faction_id,
            outcome,
        });
    }
}
