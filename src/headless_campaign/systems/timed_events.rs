//! Time-limited events system — special opportunities and threats that appear
//! briefly and require quick decision-making.
//!
//! Fires every 200 ticks (~20s game time). Events have short deadlines
//! (200–500 ticks) and reward faster responses with bonuses.
//! Max 2 active timed events at once. 10% of events are traps.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often to check for new timed events (in ticks).
const EVENT_INTERVAL: u64 = 200;

/// Base probability of an event firing each roll (5%).
const BASE_CHANCE: f32 = 0.05;

/// Maximum concurrent active timed events.
const MAX_ACTIVE: usize = 2;

/// Probability that a generated event is actually a trap.
const TRAP_CHANCE: f32 = 0.10;

/// Tick the timed events system.
///
/// - Expires events past their deadline
/// - Generates new events (5% base chance per roll, scaled by season/phase)
/// - Max 2 active at once
pub fn tick_timed_events(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % EVENT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Expire past-deadline events ---
    let mut expired_names = Vec::new();
    for te in &mut state.timed_events {
        if !te.responded && state.tick >= te.deadline_tick {
            expired_names.push(te.name.clone());
        }
    }
    for name in &expired_names {
        events.push(WorldEvent::TimedEventExpired {
            name: name.clone(),
        });
    }
    state
        .timed_events
        .retain(|te| te.responded || state.tick < te.deadline_tick);

    // --- Count active (non-responded) events ---
    let active_count = state
        .timed_events
        .iter()
        .filter(|te| !te.responded)
        .count();
    if active_count >= MAX_ACTIVE {
        return;
    }

    // --- Roll for new event ---
    let season_mult = match state.overworld.season {
        Season::Spring => 1.2,
        Season::Summer => 1.0,
        Season::Autumn => 1.3,
        Season::Winter => 0.7,
    };
    let phase_mult = 0.8 + state.overworld.campaign_progress.clamp(0.0, 1.0) * 0.4;
    let chance = BASE_CHANCE * season_mult * phase_mult;

    let roll = lcg_f32(&mut state.rng);
    if roll > chance {
        return;
    }

    // --- Pick event type ---
    let event_type = pick_event_type(&mut state.rng);

    // --- Determine duration (200-500 ticks) ---
    let duration = 200 + (lcg_next(&mut state.rng) % 301) as u64;
    let deadline_tick = state.tick + duration;

    // --- Determine difficulty (0.2 - 1.0) ---
    let difficulty = 0.2 + lcg_f32(&mut state.rng) * 0.8;

    // --- Check if trap (10% chance) ---
    let is_trap = lcg_f32(&mut state.rng) < TRAP_CHANCE;

    // --- Build event ---
    let id = state.next_event_id;
    state.next_event_id += 1;

    let (name, description, requires_party, reward) =
        build_event_details(&event_type, difficulty, is_trap, &mut state.rng);

    let timed_event = TimedEvent {
        id,
        name: name.clone(),
        event_type,
        description: description.clone(),
        reward,
        deadline_tick,
        responded: false,
        requires_party,
        difficulty,
        is_trap,
    };

    state.timed_events.push(timed_event);

    events.push(WorldEvent::TimedEventAppeared {
        event_id: id,
        name,
        description,
        deadline_tick,
    });
}

/// Apply a player's response to a timed event.
///
/// Returns `(success_message, world_events)`.
/// Called from `apply_action` in step.rs.
pub fn respond_to_timed_event(
    state: &mut CampaignState,
    event_id: u32,
    party_id: Option<u32>,
    events: &mut Vec<WorldEvent>,
) -> Result<String, String> {
    let te_idx = state
        .timed_events
        .iter()
        .position(|te| te.id == event_id)
        .ok_or_else(|| format!("Timed event {} not found", event_id))?;

    let te = &state.timed_events[te_idx];

    if te.responded {
        return Err(format!("Timed event {} already responded to", event_id));
    }
    if state.tick >= te.deadline_tick {
        return Err(format!("Timed event {} has expired", event_id));
    }

    // Check party requirement
    if te.requires_party {
        match party_id {
            Some(pid) => {
                let has_idle_party = state.parties.iter().any(|p| {
                    p.id == pid
                        && matches!(
                            p.status,
                            PartyStatus::Idle | PartyStatus::Returning
                        )
                });
                // Also accept if there are idle adventurers (they can form a party)
                let has_idle_adventurers = state
                    .adventurers
                    .iter()
                    .any(|a| a.status == AdventurerStatus::Idle);
                if !has_idle_party && !has_idle_adventurers {
                    return Err("No available party or idle adventurers for this event".into());
                }
            }
            None => {
                // Check if there are any idle adventurers
                let has_idle = state
                    .adventurers
                    .iter()
                    .any(|a| a.status == AdventurerStatus::Idle);
                if !has_idle {
                    return Err("This event requires a party but no adventurers are idle".into());
                }
            }
        }
    }

    // Clone needed fields before mutating
    let te_name = state.timed_events[te_idx].name.clone();
    let te_is_trap = state.timed_events[te_idx].is_trap;
    let te_difficulty = state.timed_events[te_idx].difficulty;
    let te_deadline = state.timed_events[te_idx].deadline_tick;
    let te_reward_gold = state.timed_events[te_idx].reward.gold;
    let te_reward_reputation = state.timed_events[te_idx].reward.reputation;
    let te_reward_resource_type = state.timed_events[te_idx].reward.resource_type.clone();
    let te_reward_resource_amount = state.timed_events[te_idx].reward.resource_amount;
    let te_reward_special = state.timed_events[te_idx].reward.special.clone();

    // Mark as responded
    state.timed_events[te_idx].responded = true;

    // --- Handle trap ---
    if te_is_trap {
        // Trap! Penalty instead of reward
        let gold_penalty = te_reward_gold * 0.5; // lose half the promised gold
        state.guild.gold = (state.guild.gold - gold_penalty).max(0.0);
        state.guild.reputation = (state.guild.reputation - 5.0).max(0.0);

        // Injure a random adventurer if party was sent
        if te_difficulty > 0.5 {
            let alive: Vec<u32> = state
                .adventurers
                .iter()
                .filter(|a| a.status != AdventurerStatus::Dead)
                .map(|a| a.id)
                .collect();
            if !alive.is_empty() {
                let victim_idx = (lcg_next(&mut state.rng) as usize) % alive.len();
                let victim_id = alive[victim_idx];
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == victim_id) {
                    adv.injury = (adv.injury + 20.0 * te_difficulty).min(100.0);
                    adv.stress = (adv.stress + 15.0).min(100.0);
                }
            }
        }

        events.push(WorldEvent::TimedEventTrap {
            event_id,
            name: te_name.clone(),
            gold_lost: gold_penalty,
        });

        return Ok(format!(
            "It was a trap! Lost {:.0} gold responding to '{}'.",
            gold_penalty, te_name
        ));
    }

    // --- Calculate speed bonus ---
    // Faster response = bigger bonus (up to 50% extra)
    let ticks_remaining = te_deadline.saturating_sub(state.tick);
    let total_duration = te_deadline.saturating_sub(te_deadline.saturating_sub(500));
    let speed_ratio = if total_duration > 0 {
        ticks_remaining as f32 / total_duration as f32
    } else {
        0.5
    };
    let speed_bonus = 1.0 + speed_ratio * 0.5; // 1.0 to 1.5x

    // --- Apply rewards ---
    let gold_reward = te_reward_gold * speed_bonus;
    state.guild.gold += gold_reward;

    let rep_reward = te_reward_reputation * speed_bonus;
    state.guild.reputation = (state.guild.reputation + rep_reward).min(100.0);

    if let Some(ref res_type) = te_reward_resource_type {
        let res_amount = te_reward_resource_amount * speed_bonus;
        match res_type.as_str() {
            "supplies" => state.guild.supplies += res_amount,
            _ => {
                // Generic resource
                let rt = match res_type.as_str() {
                    "iron" => ResourceType::Iron,
                    "herbs" => ResourceType::Herbs,
                    "wood" => ResourceType::Wood,
                    "crystal" => ResourceType::Crystal,
                    "hide" => ResourceType::Hide,
                    _ => ResourceType::Iron, // fallback
                };
                *state.resources.entry(rt).or_insert(0.0) += res_amount;
            }
        }
    }

    // Handle special rewards
    if let Some(ref special) = te_reward_special {
        match special.as_str() {
            "morale_boost" => {
                for adv in &mut state.adventurers {
                    if adv.status != AdventurerStatus::Dead {
                        adv.morale = (adv.morale + 15.0).min(100.0);
                    }
                }
            }
            "xp_boost" => {
                for adv in &mut state.adventurers {
                    if adv.status != AdventurerStatus::Dead {
                        adv.xp += 50;
                    }
                }
            }
            "stress_relief" => {
                for adv in &mut state.adventurers {
                    if adv.status != AdventurerStatus::Dead {
                        adv.stress = (adv.stress - 20.0).max(0.0);
                    }
                }
            }
            _ => {}
        }
    }

    events.push(WorldEvent::TimedEventResponded {
        event_id,
        name: te_name.clone(),
        gold_gained: gold_reward,
        reputation_gained: rep_reward,
        speed_bonus,
    });

    Ok(format!(
        "Responded to '{}': +{:.0} gold, +{:.1} rep (speed bonus: {:.0}%)",
        te_name,
        gold_reward,
        rep_reward,
        (speed_bonus - 1.0) * 100.0
    ))
}

// ---------------------------------------------------------------------------
// Event generation helpers
// ---------------------------------------------------------------------------

fn pick_event_type(rng: &mut u64) -> TimedEventType {
    let roll = lcg_next(rng) % 8;
    match roll {
        0 => TimedEventType::RareCreatureSighting,
        1 => TimedEventType::MeteorShower,
        2 => TimedEventType::Eclipse,
        3 => TimedEventType::TradeWinds,
        4 => TimedEventType::FactionSummit,
        5 => TimedEventType::AncientPortal,
        6 => TimedEventType::ProphecyAlignment,
        _ => TimedEventType::HarvestMoon,
    }
}

fn build_event_details(
    event_type: &TimedEventType,
    difficulty: f32,
    is_trap: bool,
    rng: &mut u64,
) -> (String, String, bool, TimedEventReward) {
    let base_gold = 30.0 + difficulty * 70.0; // 30-100
    let base_rep = 2.0 + difficulty * 8.0; // 2-10

    match event_type {
        TimedEventType::RareCreatureSighting => {
            let creatures = [
                "Phoenix", "Golden Stag", "Crystal Wyrm", "Shadow Fox", "Moonbeam Elk",
            ];
            let creature = creatures[(lcg_next(rng) as usize) % creatures.len()];
            let name = format!("{} Sighting", creature);
            let desc = if is_trap {
                format!(
                    "A {} has been spotted nearby! Hunters say it's worth a fortune. \
                     Act quickly before it moves on!",
                    creature
                )
            } else {
                format!(
                    "A rare {} has been spotted in the wild! Send a party to capture it \
                     for a generous bounty.",
                    creature
                )
            };
            (
                name,
                desc,
                true, // requires party
                TimedEventReward {
                    gold: base_gold * 1.5,
                    reputation: base_rep,
                    resource_type: None,
                    resource_amount: 0.0,
                    special: None,
                },
            )
        }

        TimedEventType::MeteorShower => {
            let resources = ["iron", "crystal", "herbs"];
            let res = resources[(lcg_next(rng) as usize) % resources.len()];
            let amount = 10.0 + (lcg_next(rng) % 21) as f32;
            (
                "Meteor Shower".into(),
                format!(
                    "A meteor shower has deposited rare {} across the region! \
                     Gather them before scavengers arrive.",
                    res
                ),
                false,
                TimedEventReward {
                    gold: base_gold * 0.5,
                    reputation: base_rep * 0.5,
                    resource_type: Some(res.to_string()),
                    resource_amount: amount,
                    special: None,
                },
            )
        }

        TimedEventType::Eclipse => (
            "Eclipse".into(),
            "A rare eclipse amplifies magical energy! Your mages can harness \
             this power for extraordinary results."
                .into(),
            false,
            TimedEventReward {
                gold: base_gold * 0.3,
                reputation: base_rep * 0.5,
                resource_type: None,
                resource_amount: 0.0,
                special: Some("xp_boost".into()),
            },
        ),

        TimedEventType::TradeWinds => (
            "Trade Winds".into(),
            "Favorable trade winds have opened a temporary trade route! \
             Merchants offer exceptional deals for a limited time."
                .into(),
            false,
            TimedEventReward {
                gold: base_gold * 2.0,
                reputation: base_rep * 0.3,
                resource_type: Some("supplies".into()),
                resource_amount: 20.0 + (lcg_next(rng) % 31) as f32,
                special: None,
            },
        ),

        TimedEventType::FactionSummit => {
            let rep_bonus = base_rep * 2.0;
            (
                "Faction Summit".into(),
                "A rare diplomatic summit has been called! Attending could \
                 dramatically improve relations with multiple factions."
                    .into(),
                false,
                TimedEventReward {
                    gold: 0.0,
                    reputation: rep_bonus,
                    resource_type: None,
                    resource_amount: 0.0,
                    special: Some("morale_boost".into()),
                },
            )
        }

        TimedEventType::AncientPortal => (
            "Ancient Portal".into(),
            "An ancient portal has flickered to life! It leads to a dungeon \
             shortcut that bypasses the most dangerous sections."
                .into(),
            true, // requires party
            TimedEventReward {
                gold: base_gold * 1.2,
                reputation: base_rep * 1.5,
                resource_type: None,
                resource_amount: 0.0,
                special: Some("xp_boost".into()),
            },
        ),

        TimedEventType::ProphecyAlignment => (
            "Prophecy Alignment".into(),
            "The stars align with an ancient prophecy! Seers report perfect \
             clarity — visions of the future are 100% accurate right now."
                .into(),
            false,
            TimedEventReward {
                gold: base_gold * 0.5,
                reputation: base_rep * 1.0,
                resource_type: None,
                resource_amount: 0.0,
                special: Some("stress_relief".into()),
            },
        ),

        TimedEventType::HarvestMoon => {
            let amount = 25.0 + (lcg_next(rng) % 26) as f32;
            (
                "Harvest Moon".into(),
                "The Harvest Moon rises, doubling yields of crops and herbs! \
                 Gather resources before the moon wanes."
                    .into(),
                false,
                TimedEventReward {
                    gold: base_gold * 0.4,
                    reputation: base_rep * 0.3,
                    resource_type: Some("herbs".into()),
                    resource_amount: amount,
                    special: None,
                },
            )
        }
    }
}
