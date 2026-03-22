//! Adventurer recruitment — every 3000 ticks (~5 minutes).
//!
//! New adventurers become available for hire based on guild reputation.
//! The guild can grow from 4 starting adventurers to 8-12 over a campaign.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Check interval: every 3000 ticks (~5 minutes game time).
const RECRUIT_INTERVAL: u64 = 3000;

/// Base gold cost to recruit.
const RECRUIT_COST: f32 = 40.0;

/// Max adventurers the guild can have.
const MAX_ADVENTURERS: usize = 12;

pub fn tick_recruitment(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % RECRUIT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let alive_count = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .count();

    if alive_count >= MAX_ADVENTURERS {
        return;
    }

    // Recruitment chance scales with reputation
    let recruit_chance = (state.guild.reputation / 100.0).clamp(0.2, 0.8);
    let roll = lcg_f32(&mut state.rng);
    if roll > recruit_chance {
        return;
    }

    // Auto-recruit if we can afford it
    if state.guild.gold < RECRUIT_COST {
        return;
    }

    state.guild.gold -= RECRUIT_COST;

    // Generate a random adventurer
    let archetypes = ["ranger", "knight", "mage", "cleric", "rogue"];
    let archetype_idx = (lcg_next(&mut state.rng) as usize) % archetypes.len();
    let archetype = archetypes[archetype_idx];

    let level = 1 + (lcg_next(&mut state.rng) % 3) as u32; // level 1-3

    let names = [
        "Alaric", "Brynn", "Cira", "Daven", "Elara", "Finn", "Greta", "Holt",
        "Ivy", "Jorik", "Kessa", "Lorn", "Maren", "Nyx", "Orin", "Petra",
    ];
    let name_idx = (lcg_next(&mut state.rng) as usize) % names.len();
    let name = format!("{} the {}", names[name_idx], archetype.chars().next().unwrap().to_uppercase().to_string() + &archetype[1..]);

    let id = state
        .adventurers
        .iter()
        .map(|a| a.id)
        .max()
        .unwrap_or(0)
        + 1;

    let (hp, atk, def, spd, ap) = match archetype {
        "knight" => (110.0, 12.0, 18.0, 7.0, 4.0),
        "ranger" => (75.0, 16.0, 8.0, 13.0, 7.0),
        "mage" => (55.0, 6.0, 5.0, 9.0, 22.0),
        "cleric" => (65.0, 5.0, 10.0, 8.0, 18.0),
        "rogue" => (65.0, 18.0, 6.0, 15.0, 6.0),
        _ => (70.0, 10.0, 10.0, 10.0, 10.0),
    };

    let adventurer = Adventurer {
        id,
        name: name.clone(),
        archetype: archetype.into(),
        level,
        xp: 0,
        stats: AdventurerStats {
            max_hp: hp + level as f32 * 5.0,
            attack: atk + level as f32 * 2.0,
            defense: def + level as f32 * 1.5,
            speed: spd,
            ability_power: ap + level as f32 * 2.0,
        },
        equipment: Equipment::default(),
        traits: Vec::new(),
        status: AdventurerStatus::Idle,
        loyalty: 50.0 + lcg_f32(&mut state.rng) * 30.0,
        stress: lcg_f32(&mut state.rng) * 15.0,
        fatigue: lcg_f32(&mut state.rng) * 10.0,
        injury: 0.0,
        resolve: 40.0 + lcg_f32(&mut state.rng) * 30.0,
        morale: 60.0 + lcg_f32(&mut state.rng) * 30.0,
        party_id: None,
        guild_relationship: 30.0 + lcg_f32(&mut state.rng) * 20.0,
    };

    events.push(WorldEvent::AdventurerRecovered {
        adventurer_id: id,
    }); // Reusing event type for "new recruit arrived"

    state.adventurers.push(adventurer);
}
