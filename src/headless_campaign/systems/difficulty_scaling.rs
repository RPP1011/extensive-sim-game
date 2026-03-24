//! Dynamic difficulty scaling — every 300 ticks (~30s game time).
//!
//! Monitors guild power and adjusts pressure to keep tension consistent.
//! If the guild is dominant, escalation events trigger (harder enemies,
//! trade disruptions, nemesis threats). If struggling, relief events appear
//! (faction aid, supply caravans, recruit arrivals).

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often (in ticks) the difficulty system evaluates and adjusts.
const SCALING_INTERVAL: u64 = 300;

/// Rate at which current_pressure moves toward target_pressure per tick.
const PRESSURE_SMOOTHING: f32 = 2.0;

/// Power rating threshold for "dominant" (above this triggers escalation).
const DOMINANT_THRESHOLD: f32 = 70.0;

/// Power rating threshold for "struggling" (below this triggers relief).
const STRUGGLING_THRESHOLD: f32 = 30.0;

/// Evaluate guild power and adjust difficulty pressure every 300 ticks.
pub fn tick_difficulty_scaling(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % SCALING_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let diff = &mut state.difficulty_scaling;

    // --- Compute guild power rating (0-100) ---
    let adventurer_count = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .count() as f32;

    let avg_level = if adventurer_count > 0.0 {
        state
            .adventurers
            .iter()
            .filter(|a| a.status != AdventurerStatus::Dead)
            .map(|a| a.level as f32)
            .sum::<f32>()
            / adventurer_count
    } else {
        1.0
    };

    // Manpower score: adventurer_count * avg_level, normalized to ~0-30
    // 8 adventurers at level 5 = 40, capped contribution at 30
    let manpower_score = (adventurer_count * avg_level / 40.0 * 30.0).min(30.0);

    // Gold score: 0-20 based on liquid gold
    let gold_score = (state.guild.gold / 500.0 * 20.0).min(20.0);

    // Reputation score: 0-15
    let rep_score = state.guild.reputation / 100.0 * 15.0;

    // Territory control: fraction of regions owned by guild
    let guild_regions = state
        .overworld
        .regions
        .iter()
        .filter(|r| r.owner_faction_id == state.diplomacy.guild_faction_id)
        .count() as f32;
    let total_regions = state.overworld.regions.len().max(1) as f32;
    let territory_score = (guild_regions / total_regions) * 20.0;

    // Building investment: sum of building tiers (0-18 max) normalized to 0-15
    let building_total = state.guild_buildings.training_grounds as f32
        + state.guild_buildings.watchtower as f32
        + state.guild_buildings.trade_post as f32
        + state.guild_buildings.barracks as f32
        + state.guild_buildings.infirmary as f32
        + state.guild_buildings.war_room as f32;
    let building_score = (building_total / 18.0 * 15.0).min(15.0);

    let power_rating = (manpower_score + gold_score + rep_score + territory_score + building_score)
        .clamp(0.0, 100.0);
    diff.guild_power_rating = power_rating;

    // --- Compute target pressure ---
    // Neutral pressure is 50. Dominant guilds face higher pressure, struggling ones get relief.
    let mut target = 50.0_f32;

    if power_rating > DOMINANT_THRESHOLD {
        // Scale escalation with how far above the threshold
        let excess = (power_rating - DOMINANT_THRESHOLD) / (100.0 - DOMINANT_THRESHOLD);
        target += excess * 30.0; // Up to +30 pressure
    } else if power_rating < STRUGGLING_THRESHOLD {
        let deficit = (STRUGGLING_THRESHOLD - power_rating) / STRUGGLING_THRESHOLD;
        target -= deficit * 30.0; // Down to -30 pressure
    }

    // Consecutive wins/losses amplify the effect
    if diff.consecutive_wins >= 3 {
        target += (diff.consecutive_wins - 2) as f32 * 3.0;
    }
    if diff.consecutive_losses >= 2 {
        target -= (diff.consecutive_losses - 1) as f32 * 4.0;
    }

    diff.target_pressure = target.clamp(0.0, 100.0);

    // --- Smooth pressure adjustment ---
    let old_pressure = diff.current_pressure;
    if diff.current_pressure < diff.target_pressure {
        diff.current_pressure =
            (diff.current_pressure + PRESSURE_SMOOTHING).min(diff.target_pressure);
    } else if diff.current_pressure > diff.target_pressure {
        diff.current_pressure =
            (diff.current_pressure - PRESSURE_SMOOTHING).max(diff.target_pressure);
    }

    // --- Emit pressure milestone events ---
    let milestones = [25.0, 50.0, 75.0];
    for &milestone in &milestones {
        let crossed_up = old_pressure < milestone && diff.current_pressure >= milestone;
        let crossed_down = old_pressure > milestone && diff.current_pressure <= milestone;
        if crossed_up || crossed_down {
            events.push(WorldEvent::PressureChanged {
                old: old_pressure,
                new: diff.current_pressure,
            });
        }
    }

    // --- Apply escalation or relief ---
    if power_rating > DOMINANT_THRESHOLD && diff.current_pressure > 60.0 {
        apply_escalation(state, events);
    } else if power_rating < STRUGGLING_THRESHOLD && diff.current_pressure < 40.0 {
        apply_relief(state, events);
    }
}

/// Record a quest/battle win for difficulty tracking.
pub fn record_win(state: &mut CampaignState) {
    state.difficulty_scaling.consecutive_wins += 1;
    state.difficulty_scaling.consecutive_losses = 0;
}

/// Record a quest/battle loss for difficulty tracking.
pub fn record_loss(state: &mut CampaignState) {
    state.difficulty_scaling.consecutive_losses += 1;
    state.difficulty_scaling.consecutive_wins = 0;
}

// ---------------------------------------------------------------------------
// Escalation events (guild is dominant)
// ---------------------------------------------------------------------------

fn apply_escalation(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let roll = lcg_next(&mut state.rng) % 4;
    state.difficulty_scaling.scaling_events_triggered += 1;

    match roll {
        0 => {
            // Stronger faction attacks — boost hostility and unrest
            state.overworld.global_threat_level =
                (state.overworld.global_threat_level + 5.0).min(100.0);
            for faction in &mut state.factions {
                if faction.relationship_to_guild < 0.0 {
                    faction.relationship_to_guild =
                        (faction.relationship_to_guild - 10.0).max(-100.0);
                }
            }
            events.push(WorldEvent::DifficultyEscalation {
                description: "Hostile factions coordinate an offensive against the guild's holdings. Threat level rises."
                    .to_string(),
            });
        }
        1 => {
            // Trade disruption — lose gold and increase supply costs
            let gold_loss = (state.guild.gold * 0.1).min(80.0);
            state.guild.gold = (state.guild.gold - gold_loss).max(0.0);
            state.guild.market_prices.supply_multiplier =
                (state.guild.market_prices.supply_multiplier + 0.15).min(3.0);
            events.push(WorldEvent::DifficultyEscalation {
                description: format!(
                    "Trade routes are disrupted! Lost {:.0} gold and supply prices rise.",
                    gold_loss
                ),
            });
            if gold_loss > 0.0 {
                events.push(WorldEvent::GoldChanged {
                    amount: -gold_loss,
                    reason: "Trade disruption (difficulty scaling)".to_string(),
                });
            }
        }
        2 => {
            // Regional unrest spike
            for region in &mut state.overworld.regions {
                if region.owner_faction_id == state.diplomacy.guild_faction_id {
                    region.unrest = (region.unrest + 10.0).min(100.0);
                }
            }
            state.overworld.global_threat_level =
                (state.overworld.global_threat_level + 5.0).min(100.0);
            events.push(WorldEvent::DifficultyEscalation {
                description: "Unrest rises across guild-controlled territories. The populace grows restless."
                    .to_string(),
            });
        }
        _ => {
            // Crisis acceleration — quest threat levels increase
            for req in &mut state.request_board {
                req.threat_level = (req.threat_level + 8.0).min(100.0);
            }
            state.overworld.global_threat_level =
                (state.overworld.global_threat_level + 5.0).min(100.0);
            events.push(WorldEvent::DifficultyEscalation {
                description: "Reports grow dire — new threats emerge and existing dangers intensify."
                    .to_string(),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Relief events (guild is struggling)
// ---------------------------------------------------------------------------

fn apply_relief(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let roll = lcg_next(&mut state.rng) % 4;
    state.difficulty_scaling.scaling_events_triggered += 1;

    match roll {
        0 => {
            // Friendly faction aid — gold and supplies
            let gold_gift = 30.0 + (lcg_next(&mut state.rng) % 41) as f32; // 30-70
            let supply_gift = 15.0 + (lcg_next(&mut state.rng) % 26) as f32; // 15-40
            state.guild.gold += gold_gift;
            state.guild.supplies += supply_gift;
            events.push(WorldEvent::DifficultyRelief {
                description: format!(
                    "A sympathetic faction sends aid: +{:.0} gold, +{:.0} supplies.",
                    gold_gift, supply_gift
                ),
            });
            events.push(WorldEvent::GoldChanged {
                amount: gold_gift,
                reason: "Faction aid (difficulty relief)".to_string(),
            });
            events.push(WorldEvent::SupplyChanged {
                amount: supply_gift,
                reason: "Faction aid (difficulty relief)".to_string(),
            });
        }
        1 => {
            // Ceasefire — hostile factions ease off
            state.overworld.global_threat_level =
                (state.overworld.global_threat_level - 5.0).max(0.0);
            for faction in &mut state.factions {
                if faction.relationship_to_guild < -20.0 {
                    faction.relationship_to_guild =
                        (faction.relationship_to_guild + 15.0).min(0.0);
                }
            }
            events.push(WorldEvent::DifficultyRelief {
                description:
                    "Hostile factions pull back temporarily. Threat level decreases."
                        .to_string(),
            });
        }
        2 => {
            // Recruit arrives — free adventurer
            let alive_count = state
                .adventurers
                .iter()
                .filter(|a| a.status != AdventurerStatus::Dead)
                .count();

            if alive_count < 12 {
                let archetypes = ["ranger", "knight", "rogue", "cleric", "mage"];
                let idx = (lcg_next(&mut state.rng) as usize) % archetypes.len();
                let archetype = archetypes[idx];

                let names = [
                    "Theron", "Lyssa", "Gareth", "Ivy", "Orin", "Freya", "Dain", "Maren",
                ];
                let name_idx = (lcg_next(&mut state.rng) as usize) % names.len();
                let name = format!("{} the Volunteer", names[name_idx]);

                let id = state
                    .adventurers
                    .iter()
                    .map(|a| a.id)
                    .max()
                    .unwrap_or(0)
                    + 1;

                let avg_level = state
                    .adventurers
                    .iter()
                    .filter(|a| a.status != AdventurerStatus::Dead)
                    .map(|a| a.level)
                    .max()
                    .unwrap_or(1)
                    .saturating_sub(1)
                    .max(1);

                let (hp, atk, def, spd, ap) = match archetype {
                    "knight" => (90.0, 10.0, 14.0, 6.0, 3.0),
                    "ranger" => (60.0, 13.0, 6.0, 11.0, 5.0),
                    "mage" => (45.0, 5.0, 4.0, 8.0, 16.0),
                    "cleric" => (55.0, 4.0, 8.0, 7.0, 14.0),
                    _ => (55.0, 14.0, 5.0, 12.0, 4.0), // rogue
                };

                let adventurer = Adventurer {
                    id,
                    name: name.clone(),
                    archetype: archetype.to_string(),
                    level: avg_level,
                    xp: 0,
                    stats: AdventurerStats {
                        max_hp: hp,
                        attack: atk,
                        defense: def,
                        speed: spd,
                        ability_power: ap,
                    },
                    equipment: Equipment::default(),
                    traits: vec!["volunteer".to_string()],
                    status: AdventurerStatus::Idle,
                    loyalty: 60.0,
                    stress: 10.0,
                    fatigue: 0.0,
                    injury: 0.0,
                    resolve: 60.0,
                    morale: 70.0,
                    party_id: None,
                    guild_relationship: 50.0,
                    leadership_role: None,
                    is_player_character: false,
                    faction_id: None,
                    rallying_to: None,
                    tier_status: Default::default(),
                    history_tags: Default::default(),
            backstory: None,
            deeds: Vec::new(),
            hobbies: Vec::new(),
            disease_status: crate::headless_campaign::state::DiseaseStatus::Healthy,
                };

                state.adventurers.push(adventurer);
                events.push(WorldEvent::DifficultyRelief {
                    description: format!(
                        "A {} volunteer ({}) joins the guild, inspired by your cause.",
                        archetype, name
                    ),
                });
            } else {
                // Guild is full, give supplies instead
                let amount = 25.0;
                state.guild.supplies += amount;
                events.push(WorldEvent::DifficultyRelief {
                    description: format!(
                        "Sympathizers deliver emergency supplies. +{:.0} supplies.",
                        amount
                    ),
                });
                events.push(WorldEvent::SupplyChanged {
                    amount,
                    reason: "Emergency delivery (difficulty relief)".to_string(),
                });
            }
        }
        _ => {
            // Gold gift — direct financial aid
            let amount = 40.0 + (lcg_next(&mut state.rng) % 61) as f32; // 40-100
            state.guild.gold += amount;
            state.overworld.global_threat_level =
                (state.overworld.global_threat_level - 5.0).max(0.0);
            events.push(WorldEvent::DifficultyRelief {
                description: format!(
                    "Anonymous benefactors donate {:.0} gold to the guild. Threat eases.",
                    amount
                ),
            });
            events.push(WorldEvent::GoldChanged {
                amount,
                reason: "Anonymous donation (difficulty relief)".to_string(),
            });
        }
    }
}
