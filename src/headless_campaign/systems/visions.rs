//! Prophetic visions system — high-level or magical adventurers receive
//! foreshadowing of crises, betrayals, treasure, and doom.
//!
//! Fires every 500 ticks. Eligible adventurers (level >= 7 or archetypes
//! mage/healer/shaman) have a 10% chance of receiving a vision. Accuracy
//! scales with level. Fulfilled visions grant morale and a history tag.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often to roll for visions (in ticks).
const VISION_INTERVAL: u64 = 500;

/// Base probability per eligible adventurer per roll.
const VISION_CHANCE: f32 = 0.10;

/// Maximum active (unfulfilled) visions at any time.
const MAX_ACTIVE_VISIONS: usize = 3;

/// Ticks before an unfulfilled vision fades away.
const VISION_FADE_TICKS: u64 = 2000;

/// Roll for prophetic visions and check fulfillment of existing visions.
pub fn tick_visions(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    // --- Fulfillment check (every tick) ---
    check_fulfillment(state, events);

    // --- Fade expired visions (every tick) ---
    fade_expired_visions(state);

    // --- Generate new visions (cadenced) ---
    if state.tick % VISION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let active_count = state
        .visions
        .iter()
        .filter(|v| !v.fulfilled && !v.faded)
        .count();
    if active_count >= MAX_ACTIVE_VISIONS {
        return;
    }

    // Collect eligible adventurer IDs + metadata.
    let eligible: Vec<(u32, String, u32, String)> = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .filter(|a| {
            a.level >= 7
                || matches!(
                    a.archetype.as_str(),
                    "mage" | "healer" | "shaman"
                )
        })
        .map(|a| (a.id, a.name.clone(), a.level, a.archetype.clone()))
        .collect();

    for (adv_id, adv_name, adv_level, _archetype) in &eligible {
        // Re-check cap each iteration (a previous adventurer may have generated one).
        let active_now = state
            .visions
            .iter()
            .filter(|v| !v.fulfilled && !v.faded)
            .count();
        if active_now >= MAX_ACTIVE_VISIONS {
            break;
        }

        let roll = lcg_f32(&mut state.rng);
        if roll >= VISION_CHANCE {
            continue;
        }

        // Pick vision type weighted by game state.
        let vision_type = pick_vision_type(state);

        // Accuracy: 50% base + 5% per level above 7, capped at 100%.
        let accuracy = (0.50 + 0.05 * (*adv_level as f32 - 7.0).max(0.0)).min(1.0);

        let text = generate_vision_text(&vision_type, state, &adv_name);

        let id = state.next_vision_id;
        state.next_vision_id += 1;

        state.visions.push(Vision {
            id,
            adventurer_id: *adv_id,
            vision_type: vision_type.clone(),
            text: text.clone(),
            accuracy,
            tick: state.tick,
            fulfilled: false,
            faded: false,
        });

        events.push(WorldEvent::VisionReceived {
            adventurer_id: *adv_id,
            text: text.clone(),
            vision_type: format!("{:?}", vision_type),
        });
    }
}

// ---------------------------------------------------------------------------
// Vision type selection
// ---------------------------------------------------------------------------

fn pick_vision_type(state: &mut CampaignState) -> VisionType {
    let mut pool: Vec<(VisionType, f32)> = Vec::new();

    // Active crisis → CrisisWarning more likely.
    let crisis_weight = if !state.overworld.active_crises.is_empty() {
        20.0
    } else {
        5.0
    };
    pool.push((VisionType::CrisisWarning, crisis_weight));

    // Low-loyalty adventurer → BetrayalForecast more likely.
    let has_low_loyalty = state
        .adventurers
        .iter()
        .any(|a| a.status != AdventurerStatus::Dead && a.loyalty < 30.0);
    let betrayal_weight = if has_low_loyalty { 15.0 } else { 3.0 };
    pool.push((VisionType::BetrayalForecast, betrayal_weight));

    // Always possible.
    pool.push((VisionType::TreasureReveal, 8.0));
    pool.push((VisionType::DeathOmen, 6.0));

    // High campaign progress → VictoryGlimpse.
    let victory_weight = if state.overworld.campaign_progress > 0.6 {
        12.0
    } else {
        3.0
    };
    pool.push((VisionType::VictoryGlimpse, victory_weight));

    // Hostile faction → FactionDoom.
    let has_hostile = state
        .factions
        .iter()
        .any(|f| f.relationship_to_guild < -20.0);
    let faction_weight = if has_hostile { 14.0 } else { 4.0 };
    pool.push((VisionType::FactionDoom, faction_weight));

    // Weighted random selection.
    let total: f32 = pool.iter().map(|(_, w)| w).sum();
    let pick = lcg_f32(&mut state.rng) * total;
    let mut cumulative = 0.0;
    for (vt, w) in &pool {
        cumulative += w;
        if pick < cumulative {
            return vt.clone();
        }
    }
    pool.last().unwrap().0.clone()
}

// ---------------------------------------------------------------------------
// Vision text generation from templates
// ---------------------------------------------------------------------------

fn generate_vision_text(
    vision_type: &VisionType,
    state: &mut CampaignState,
    adventurer_name: &str,
) -> String {
    match vision_type {
        VisionType::CrisisWarning => {
            if let Some(crisis) = state.overworld.active_crises.first() {
                let crisis_name = crisis_display_name(crisis);
                let templates = [
                    format!(
                        "{} dreams of a dark tide rising — the {} threatens to consume all.",
                        adventurer_name, crisis_name
                    ),
                    format!(
                        "In a trance, {} sees the land shattered by the {}.",
                        adventurer_name, crisis_name
                    ),
                    format!(
                        "{} wakes screaming, having witnessed the {} engulfing the guild.",
                        adventurer_name, crisis_name
                    ),
                ];
                let idx = (lcg_next(&mut state.rng) as usize) % templates.len();
                templates[idx].clone()
            } else {
                format!(
                    "{} senses a nameless dread gathering on the horizon.",
                    adventurer_name
                )
            }
        }
        VisionType::BetrayalForecast => {
            let traitor = state
                .adventurers
                .iter()
                .filter(|a| a.status != AdventurerStatus::Dead && a.loyalty < 30.0)
                .min_by(|a, b| {
                    a.loyalty
                        .partial_cmp(&b.loyalty)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|a| a.name.clone());
            if let Some(name) = traitor {
                let templates = [
                    format!(
                        "{} sees a shadowy figure leaving the guild at midnight — it looks like {}.",
                        adventurer_name, name
                    ),
                    format!(
                        "In {}'s vision, {} turns away from the guild banner, eyes full of doubt.",
                        adventurer_name, name
                    ),
                ];
                let idx = (lcg_next(&mut state.rng) as usize) % templates.len();
                templates[idx].clone()
            } else {
                format!(
                    "{} feels a vague unease about loyalty within the guild.",
                    adventurer_name
                )
            }
        }
        VisionType::TreasureReveal => {
            if let Some(loc) = state.overworld.locations.first() {
                let loc_name = loc.name.clone();
                format!(
                    "{} dreams of a golden hoard glittering beneath {}.",
                    adventurer_name, loc_name
                )
            } else {
                format!(
                    "{} dreams of buried treasure in an unmarked ruin.",
                    adventurer_name
                )
            }
        }
        VisionType::DeathOmen => {
            let target = state
                .adventurers
                .iter()
                .filter(|a| a.status != AdventurerStatus::Dead && a.injury > 40.0)
                .max_by(|a, b| {
                    a.injury
                        .partial_cmp(&b.injury)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|a| a.name.clone());
            if let Some(name) = target {
                format!(
                    "{} sees {} walking toward a pale gate, unable to turn back.",
                    adventurer_name, name
                )
            } else {
                format!(
                    "{} dreams of an empty seat at the guild table.",
                    adventurer_name
                )
            }
        }
        VisionType::VictoryGlimpse => {
            format!(
                "{} glimpses a future where the guild banner flies triumphant over the land.",
                adventurer_name
            )
        }
        VisionType::FactionDoom => {
            let hostile = state
                .factions
                .iter()
                .filter(|f| f.relationship_to_guild < -20.0)
                .min_by(|a, b| {
                    a.relationship_to_guild
                        .partial_cmp(&b.relationship_to_guild)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|f| f.name.clone());
            if let Some(name) = hostile {
                format!(
                    "{} foresees the {} marching against the guild under a blood-red sky.",
                    adventurer_name, name
                )
            } else {
                format!(
                    "{} senses distant drums of war — a faction gathers its strength.",
                    adventurer_name
                )
            }
        }
    }
}

fn crisis_display_name(crisis: &ActiveCrisis) -> String {
    match crisis {
        ActiveCrisis::SleepingKing { .. } => "Sleeping King".to_string(),
        ActiveCrisis::Breach { .. } => "Breach".to_string(),
        ActiveCrisis::Corruption { .. } => "Corruption".to_string(),
        ActiveCrisis::Unifier { .. } => "Unifier".to_string(),
        ActiveCrisis::Decline { .. } => "Decline".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Fulfillment checks
// ---------------------------------------------------------------------------

fn check_fulfillment(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect fulfillment results first, then apply changes.
    let mut fulfilled_indices: Vec<usize> = Vec::new();

    for (i, vision) in state.visions.iter().enumerate() {
        if vision.fulfilled || vision.faded {
            continue;
        }

        let is_fulfilled = match &vision.vision_type {
            VisionType::CrisisWarning => {
                // Fulfilled if an active crisis exists.
                !state.overworld.active_crises.is_empty()
            }
            VisionType::BetrayalForecast => {
                // Fulfilled if any living adventurer has very low loyalty (< 15)
                // or has deserted (status Dead with low loyalty at time of vision).
                state
                    .adventurers
                    .iter()
                    .any(|a| a.loyalty < 15.0 && a.status != AdventurerStatus::Dead)
                    || state
                        .adventurers
                        .iter()
                        .any(|a| a.status == AdventurerStatus::Dead && a.loyalty < 20.0)
            }
            VisionType::TreasureReveal => {
                // Fulfilled if guild gold increased significantly since vision.
                state.guild.gold > 200.0
            }
            VisionType::DeathOmen => {
                // Fulfilled if any adventurer died since the vision.
                state
                    .adventurers
                    .iter()
                    .any(|a| a.status == AdventurerStatus::Dead)
            }
            VisionType::VictoryGlimpse => {
                // Fulfilled if campaign progress > 0.8.
                state.overworld.campaign_progress > 0.8
            }
            VisionType::FactionDoom => {
                // Fulfilled if a hostile faction declared war.
                state.factions.iter().any(|f| {
                    f.relationship_to_guild < -50.0
                        && !f.at_war_with.is_empty()
                })
            }
        };

        if is_fulfilled {
            fulfilled_indices.push(i);
        }
    }

    // Apply fulfillment.
    for idx in fulfilled_indices {
        let vision = &mut state.visions[idx];
        vision.fulfilled = true;

        let adv_id = vision.adventurer_id;
        let text = vision.text.clone();
        let vision_type = vision.vision_type.clone();

        // Grant morale + history tag to the adventurer.
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
            adv.morale = (adv.morale + 10.0).min(100.0);
            *adv.history_tags.entry("prophetic".to_string()).or_insert(0) += 1;
        }

        events.push(WorldEvent::VisionFulfilled {
            adventurer_id: adv_id,
            text,
            vision_type: format!("{:?}", vision_type),
        });
    }
}

// ---------------------------------------------------------------------------
// Expiry
// ---------------------------------------------------------------------------

fn fade_expired_visions(state: &mut CampaignState) {
    for vision in &mut state.visions {
        if !vision.fulfilled && !vision.faded && state.tick >= vision.tick + VISION_FADE_TICKS {
            vision.faded = true;
        }
    }
}
