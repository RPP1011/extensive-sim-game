//! Adventurer recruitment — every 3000 ticks (~5 minutes).
//!
//! New adventurers become available for hire based on guild reputation.
//! The guild can grow from 4 starting adventurers to 8-12 over a campaign.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

pub fn tick_recruitment(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let cfg = &state.config.recruitment;
    if state.tick % cfg.interval_ticks != 0 || state.tick == 0 {
        return;
    }

    let alive_count = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .count();

    if alive_count >= cfg.max_adventurers {
        return;
    }

    // Recruitment chance scales with reputation
    let recruit_chance = (state.guild.reputation / 100.0).clamp(cfg.min_recruit_chance, cfg.max_recruit_chance);
    let roll = lcg_f32(&mut state.rng);
    if roll > recruit_chance {
        return;
    }

    // Auto-recruit if we can afford it
    if state.guild.gold < cfg.recruit_cost {
        return;
    }

    state.guild.gold = (state.guild.gold - cfg.recruit_cost).max(0.0);

    // Generate a random adventurer from expanded archetype pool
    let archetypes = [
        "ranger", "knight", "mage", "cleric", "rogue",
        "paladin", "berserker", "necromancer", "bard", "druid",
        "warlock", "monk", "assassin", "guardian", "shaman",
        "artificer", "tank",
    ];
    let archetype_idx = (lcg_next(&mut state.rng) as usize) % archetypes.len();
    let archetype = archetypes[archetype_idx];

    let level_range = cfg.max_level - cfg.min_level + 1;
    let level = cfg.min_level + (lcg_next(&mut state.rng) % level_range) as u32;

    let names = [
        "Alaric", "Brynn", "Cira", "Daven", "Elara", "Finn", "Greta", "Holt",
        "Ivy", "Jorik", "Kessa", "Lorn", "Maren", "Nyx", "Orin", "Petra",
        "Quinlan", "Rhea", "Soren", "Thalia", "Ulric", "Vera", "Wren", "Xara",
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

    //                     HP    ATK   DEF   SPD   AP
    let (hp, atk, def, spd, ap) = match archetype {
        "knight"     => (110.0, 12.0, 18.0,  7.0,  4.0),
        "ranger"     => ( 75.0, 16.0,  8.0, 13.0,  7.0),
        "mage"       => ( 55.0,  6.0,  5.0,  9.0, 22.0),
        "cleric"     => ( 65.0,  5.0, 10.0,  8.0, 18.0),
        "rogue"      => ( 65.0, 18.0,  6.0, 15.0,  6.0),
        "paladin"    => (100.0, 10.0, 15.0,  6.0, 10.0),
        "berserker"  => ( 95.0, 22.0,  5.0, 10.0,  3.0),
        "necromancer" => ( 50.0,  8.0,  4.0,  7.0, 24.0),
        "bard"       => ( 60.0,  7.0,  7.0, 11.0, 15.0),
        "druid"      => ( 70.0,  8.0,  9.0,  9.0, 16.0),
        "warlock"    => ( 55.0, 10.0,  5.0,  8.0, 20.0),
        "monk"       => ( 75.0, 14.0, 10.0, 16.0,  8.0),
        "assassin"   => ( 60.0, 20.0,  4.0, 17.0,  5.0),
        "guardian"   => (120.0,  8.0, 20.0,  5.0,  3.0),
        "shaman"     => ( 65.0,  7.0,  8.0,  8.0, 18.0),
        "artificer"  => ( 60.0,  9.0,  7.0, 10.0, 16.0),
        "tank"       => (130.0,  6.0, 22.0,  4.0,  2.0),
        _            => ( 70.0, 10.0, 10.0, 10.0, 10.0),
    };

    let adventurer = Adventurer {
        id,
        name: name.clone(),
        archetype: archetype.into(),
        level,
        xp: 0,
        stats: AdventurerStats {
            max_hp: hp + level as f32 * state.config.recruitment.hp_per_level,
            attack: atk + level as f32 * state.config.recruitment.attack_per_level,
            defense: def + level as f32 * state.config.recruitment.defense_per_level,
            speed: spd,
            ability_power: ap + level as f32 * state.config.recruitment.ability_power_per_level,
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
        leadership_role: None,
        is_player_character: false,
        faction_id: None,
        rallying_to: None,
                    tier_status: Default::default(),
                    history_tags: Default::default(),
    };

    events.push(WorldEvent::AdventurerRecovered {
        adventurer_id: id,
    }); // Reusing event type for "new recruit arrived"

    state.adventurers.push(adventurer);
}
