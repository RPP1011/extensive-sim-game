//! Food and rations system — every 100 ticks.
//!
//! Parties consume food proportional to their size.
//! Food sources: quality_food (feast) → rations (hardtack) → forage (wilderness).
//! Meal quality affects morale, fatigue, and combat effectiveness.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often the food system ticks (every 100 campaign ticks).
const FOOD_TICK_INTERVAL: u64 = 3;

/// Food consumption per party member per food tick.
const FOOD_PER_MEMBER: f32 = 1.0;

/// Base foraging yield per member per tick (before modifiers).
const BASE_FORAGE_YIELD: f32 = 0.4;

/// Determine the meal quality for a party based on food source and party traits.
fn determine_meal_quality(
    food_supply: &FoodSupply,
    party_member_ids: &[u32],
    adventurers: &[Adventurer],
    food_needed: f32,
    foraged: bool,
) -> MealQuality {
    let has_cook = party_member_ids.iter().any(|&mid| {
        adventurers
            .iter()
            .find(|a| a.id == mid)
            .map(|a| a.traits.iter().any(|t| t == "Cooking"))
            .unwrap_or(false)
    });

    let base_quality = if food_supply.quality_food >= food_needed {
        MealQuality::Feast
    } else if food_supply.rations >= food_needed {
        MealQuality::Standard
    } else if food_supply.rations > 0.0 || foraged {
        MealQuality::Hardtack
    } else {
        MealQuality::Starving
    };

    // Cooking hobby bonus: upgrade one tier
    if has_cook {
        match base_quality {
            MealQuality::Starving => MealQuality::Hardtack,
            MealQuality::Hardtack => MealQuality::Standard,
            MealQuality::Standard => MealQuality::Feast,
            MealQuality::Feast => MealQuality::Feast, // already max
        }
    } else {
        base_quality
    }
}

/// Season modifier for foraging yield.
fn season_forage_modifier(season: Season) -> f32 {
    match season {
        Season::Spring => 1.2,
        Season::Summer => 1.0,
        Season::Autumn => 0.8,
        Season::Winter => 0.3,
    }
}

/// Check if a party is in a wilderness region (near a Wilderness location).
fn party_in_wilderness(party_pos: (f32, f32), locations: &[Location]) -> bool {
    locations.iter().any(|loc| {
        loc.location_type == LocationType::Wilderness && {
            let dx = party_pos.0 - loc.position.0;
            let dy = party_pos.1 - loc.position.1;
            (dx * dx + dy * dy).sqrt() < 50.0
        }
    })
}

pub fn tick_food(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % FOOD_TICK_INTERVAL != 0 {
        return;
    }

    let season = state.overworld.season;
    let forage_season_mod = season_forage_modifier(season);

    // Collect party info to avoid borrow issues
    let party_info: Vec<(u32, Vec<u32>, (f32, f32), PartyStatus)> = state
        .parties
        .iter()
        .map(|p| (p.id, p.member_ids.clone(), p.position, p.status))
        .collect();

    for (party_id, member_ids, position, status) in &party_info {
        // Only active (non-idle) parties consume food
        if matches!(status, PartyStatus::Idle) {
            continue;
        }

        let member_count = member_ids.len() as f32;
        let food_needed = FOOD_PER_MEMBER * member_count;

        // Check for herbalism trait in party
        let has_herbalist = member_ids.iter().any(|&mid| {
            state
                .adventurers
                .iter()
                .find(|a| a.id == mid)
                .map(|a| a.traits.iter().any(|t| t == "Herbalism"))
                .unwrap_or(false)
        });

        // Determine food source priority: quality_food > rations > forage
        let mut remaining = food_needed;
        let mut used_quality = false;

        // 1. Quality food first
        if state.guild.food_supply.quality_food > 0.0 {
            let consume = remaining.min(state.guild.food_supply.quality_food);
            state.guild.food_supply.quality_food -= consume;
            remaining -= consume;
            if consume > 0.0 {
                used_quality = true;
            }
        }

        // 2. Rations
        if remaining > 0.0 && state.guild.food_supply.rations > 0.0 {
            let consume = remaining.min(state.guild.food_supply.rations);
            state.guild.food_supply.rations -= consume;
            remaining -= consume;
        }

        // 3. Forage if in wilderness
        let mut foraged = false;
        if remaining > 0.0 && party_in_wilderness(*position, &state.overworld.locations) {
            let herbalism_bonus = if has_herbalist { 1.5 } else { 1.0 };
            let forage_yield = BASE_FORAGE_YIELD
                * member_count
                * forage_season_mod
                * herbalism_bonus
                * state.guild.food_supply.foraging_efficiency;
            let gathered = remaining.min(forage_yield);
            remaining -= gathered;
            if gathered > 0.0 {
                foraged = true;
            }
        }

        // Determine meal quality
        let meal = determine_meal_quality(
            &state.guild.food_supply,
            member_ids,
            &state.adventurers,
            food_needed,
            foraged,
        );

        // If we couldn't source enough food, the party effectively used what it had
        // but the meal quality reflects the shortfall.
        let meal = if remaining >= food_needed {
            // No food at all
            MealQuality::Starving
        } else if remaining > food_needed * 0.5 {
            // Very little food
            match meal {
                MealQuality::Feast | MealQuality::Standard => MealQuality::Hardtack,
                other => other,
            }
        } else if !used_quality && !foraged && state.guild.food_supply.rations <= 0.0 {
            MealQuality::Starving
        } else {
            meal
        };

        // Update party food_level based on meal quality
        if let Some(party) = state.parties.iter_mut().find(|p| p.id == *party_id) {
            match meal {
                MealQuality::Starving => {
                    party.food_level = (party.food_level - 15.0).max(0.0);
                }
                MealQuality::Hardtack => {
                    // Neutral — slight decay
                    party.food_level = (party.food_level - 2.0).max(0.0);
                }
                MealQuality::Standard => {
                    party.food_level = (party.food_level + 5.0).min(100.0);
                }
                MealQuality::Feast => {
                    party.food_level = (party.food_level + 10.0).min(100.0);
                }
            }
        }

        // Apply meal quality effects to adventurers
        for &mid in member_ids {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                if adv.status == AdventurerStatus::Dead {
                    continue;
                }
                match meal {
                    MealQuality::Starving => {
                        adv.morale = (adv.morale - 3.0).max(0.0);
                        adv.fatigue = (adv.fatigue + 2.0).min(100.0);
                    }
                    MealQuality::Hardtack => {
                        // Neutral — no morale change
                    }
                    MealQuality::Standard => {
                        adv.morale = (adv.morale + 0.5).min(100.0);
                    }
                    MealQuality::Feast => {
                        adv.morale = (adv.morale + 2.0).min(100.0);
                    }
                }
            }
        }

        // Emit starvation warning
        if matches!(meal, MealQuality::Starving) {
            events.push(WorldEvent::PartyStarving {
                party_id: *party_id,
            });
        }
    }
}
