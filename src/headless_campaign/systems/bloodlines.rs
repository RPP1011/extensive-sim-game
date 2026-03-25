//! Adventurer legacy/bloodline system — fires every 1000 ticks.
//!
//! When adventurers retire or die heroically (level >= 8), a bloodline is
//! established. Descendants of that bloodline may appear as recruits carrying
//! inherited traits, stat bonuses, and the founder's archetype bias.
//!
//! Bloodline prestige grows when active members accomplish noteworthy feats,
//! improving future descendant quality. A maximum of 5 active bloodlines
//! prevents runaway compounding.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often to check bloodline events (in ticks).
const BLOODLINE_INTERVAL: u64 = 1000;

/// Minimum adventurer level to establish a bloodline on retirement/death.
const MIN_BLOODLINE_LEVEL: u32 = 8;

/// Maximum concurrent active bloodlines.
const MAX_BLOODLINES: usize = 5;

/// Fraction of parent history tags inherited by descendants.
const TRAIT_INHERITANCE_RATE: f32 = 0.5;

/// Per-bloodline chance to spawn a descendant each tick interval.
const DESCENDANT_SPAWN_CHANCE: f32 = 0.10;

/// Base prestige gained per generation of active membership.
const BASE_PRESTIGE_PER_GENERATION: f32 = 5.0;

/// Check for new bloodlines and spawn descendants every `BLOODLINE_INTERVAL` ticks.
pub fn tick_bloodlines(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % BLOODLINE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Establish new bloodlines from recent retirements/heroic deaths ---
    establish_bloodlines(state, events);

    // --- Phase 2: Grow prestige for bloodlines with active members ---
    grow_prestige(state, events);

    // --- Phase 3: Spawn descendants ---
    spawn_descendants(state, events);
}

/// Create bloodlines from adventurers who recently retired or died heroically.
fn establish_bloodlines(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    if state.bloodlines.len() >= MAX_BLOODLINES {
        return;
    }

    // Check recently retired adventurers (retired since last bloodline tick)
    let last_check = state.tick.saturating_sub(BLOODLINE_INTERVAL);
    let mut new_founders: Vec<(String, String, u32, std::collections::HashMap<String, u32>)> =
        Vec::new();

    // Retired adventurers eligible for bloodline founding
    for retired in &state.retired_adventurers {
        if retired.retired_at_tick > last_check
            && retired.level >= MIN_BLOODLINE_LEVEL
            && !state.bloodlines.iter().any(|b| b.founder_name == retired.name)
        {
            // Look up history tags from the adventurer record (may still be in list as Dead)
            let tags = state
                .adventurers
                .iter()
                .find(|a| a.id == retired.id)
                .map(|a| a.history_tags.clone())
                .unwrap_or_default();
            new_founders.push((
                retired.name.clone(),
                retired.archetype.clone(),
                retired.level,
                tags,
            ));
        }
    }

    // Dead adventurers who died heroically (level >= MIN and died in battle = "last stand")
    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead
            && adv.level >= MIN_BLOODLINE_LEVEL
            && adv.history_tags.get("last_stand").copied().unwrap_or(0) > 0
            && !state.bloodlines.iter().any(|b| b.founder_name == adv.name)
            && !new_founders.iter().any(|(name, _, _, _)| name == &adv.name)
            // Only consider deaths in the current interval
            && !state.retired_adventurers.iter().any(|r| r.id == adv.id)
        {
            new_founders.push((
                adv.name.clone(),
                adv.archetype.clone(),
                adv.level,
                adv.history_tags.clone(),
            ));
        }
    }

    for (name, archetype, level, tags) in new_founders {
        if state.bloodlines.len() >= MAX_BLOODLINES {
            break;
        }

        // Inherit 50% of history tags
        let inherited_traits = inherit_tags(&tags, &mut state.rng);
        let stat_bonus = level as f32 * 0.5;

        let id = state
            .bloodlines
            .iter()
            .map(|b| b.id)
            .max()
            .unwrap_or(0)
            + 1;

        let bloodline = Bloodline {
            id,
            founder_name: name.clone(),
            founder_archetype: archetype.clone(),
            traits: inherited_traits,
            generations: 1,
            active_member_id: None,
            stat_bonus,
            prestige: 1.0,
            established_tick: state.tick,
        };

        events.push(WorldEvent::BloodlineEstablished {
            bloodline_id: id,
            founder_name: name,
            archetype,
            stat_bonus,
        });

        state.bloodlines.push(bloodline);
    }
}

/// Grow prestige for bloodlines whose active member has accomplished things.
fn grow_prestige(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    for bloodline in &mut state.bloodlines {
        let member_id = match bloodline.active_member_id {
            Some(id) => id,
            None => continue,
        };

        let member = match state.adventurers.iter().find(|a| a.id == member_id) {
            Some(a) => a,
            None => {
                // Member no longer exists — clear the link
                bloodline.active_member_id = None;
                continue;
            }
        };

        if member.status == AdventurerStatus::Dead {
            bloodline.active_member_id = None;
            continue;
        }

        // Prestige grows based on member level and deeds
        let deed_count = member.deeds.len() as f32;
        let level_factor = member.level as f32 / 10.0;
        let prestige_gain = BASE_PRESTIGE_PER_GENERATION * level_factor * (1.0 + deed_count * 0.1);

        if prestige_gain > 0.5 {
            bloodline.prestige += prestige_gain;
            events.push(WorldEvent::BloodlinePrestigeGrown {
                bloodline_id: bloodline.id,
                founder_name: bloodline.founder_name.clone(),
                new_prestige: bloodline.prestige,
            });
        }
    }
}

/// Attempt to spawn descendants from bloodlines without active members.
fn spawn_descendants(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let alive_count = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .count();

    let max_adventurers = state.config.recruitment.max_adventurers;
    if alive_count >= max_adventurers {
        return;
    }

    // Collect bloodline info for spawning (avoid borrow issues)
    let spawn_candidates: Vec<(u32, String, String, Vec<String>, f32, f32, u32)> = state
        .bloodlines
        .iter()
        .filter(|b| b.active_member_id.is_none())
        .map(|b| {
            (
                b.id,
                b.founder_name.clone(),
                b.founder_archetype.clone(),
                b.traits.clone(),
                b.stat_bonus,
                b.prestige,
                b.generations,
            )
        })
        .collect();

    for (bl_id, founder_name, archetype, traits, stat_bonus, prestige, generations) in
        spawn_candidates
    {
        let roll = lcg_f32(&mut state.rng);
        if roll >= DESCENDANT_SPAWN_CHANCE {
            continue;
        }

        // Re-check capacity (might have spawned one already this tick)
        let current_alive = state
            .adventurers
            .iter()
            .filter(|a| a.status != AdventurerStatus::Dead)
            .count();
        if current_alive >= max_adventurers {
            break;
        }

        let adv_id = state
            .adventurers
            .iter()
            .map(|a| a.id)
            .max()
            .unwrap_or(0)
            + 1;

        // 70% chance to inherit founder archetype, 30% random
        let chosen_archetype = if lcg_f32(&mut state.rng) < 0.7 {
            archetype.clone()
        } else {
            let archetypes = [
                "ranger", "knight", "mage", "cleric", "rogue", "paladin", "berserker",
                "necromancer", "bard", "druid", "warlock", "monk", "assassin", "guardian",
                "shaman", "artificer", "tank",
            ];
            let idx = (lcg_next(&mut state.rng) as usize) % archetypes.len();
            archetypes[idx].to_string()
        };

        // Base stats for archetype
        let (hp, atk, def, spd, ap) = base_stats_for_archetype(&chosen_archetype);

        // Prestige-scaled bonus: higher prestige = better recruits
        let prestige_mult = 1.0 + (prestige / 50.0).min(1.0);
        let effective_bonus = stat_bonus * prestige_mult;

        // Descendant starts at level 1-3
        let level = 1 + (lcg_next(&mut state.rng) % 3);
        let cfg = &state.config.recruitment;

        let descendant_name = format!(
            "{} {}",
            descendant_given_name(&mut state.rng),
            founder_name.split_whitespace().next().unwrap_or("Unknown")
        );

        let mut history_tags: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        for tag in &traits {
            history_tags.insert(tag.clone(), 1);
        }

        let adventurer = Adventurer {
            id: adv_id,
            name: descendant_name.clone(),
            archetype: chosen_archetype.clone(),
            level,
            xp: 0,
            stats: AdventurerStats {
                max_hp: hp + level as f32 * cfg.hp_per_level + effective_bonus,
                attack: atk + level as f32 * cfg.attack_per_level + effective_bonus * 0.3,
                defense: def + level as f32 * cfg.defense_per_level + effective_bonus * 0.3,
                speed: spd + effective_bonus * 0.1,
                ability_power: ap
                    + level as f32 * cfg.ability_power_per_level
                    + effective_bonus * 0.3,
            },
            equipment: Equipment::default(),
            traits: traits.clone(),
            status: AdventurerStatus::Idle,
            loyalty: 60.0 + lcg_f32(&mut state.rng) * 25.0,
            stress: lcg_f32(&mut state.rng) * 10.0,
            fatigue: lcg_f32(&mut state.rng) * 5.0,
            injury: 0.0,
            resolve: 50.0 + lcg_f32(&mut state.rng) * 25.0,
            morale: 65.0 + lcg_f32(&mut state.rng) * 25.0,
            party_id: None,
            guild_relationship: 40.0 + lcg_f32(&mut state.rng) * 20.0,
            leadership_role: None,
            is_player_character: false,
            faction_id: None,
            rallying_to: None,
            tier_status: Default::default(),
            history_tags,
            backstory: None,
            deeds: Vec::new(),
            hobbies: Vec::new(),
            disease_status: DiseaseStatus::Healthy,
            mood_state: MoodState::default(),
            fears: Vec::new(),
            personal_goal: None,
            journal: Vec::new(),
            equipped_items: Vec::new(),
            nicknames: Vec::new(),
            secret_past: None,
            wounds: Vec::new(),
            potion_dependency: 0.0,
            withdrawal_severity: 0.0,
            ticks_since_last_potion: 0,
            total_potions_consumed: 0,
        };

        state.adventurers.push(adventurer);

        // Link the descendant as the active member
        if let Some(bl) = state.bloodlines.iter_mut().find(|b| b.id == bl_id) {
            bl.active_member_id = Some(adv_id);
            bl.generations += 1;
        }

        events.push(WorldEvent::DescendantAppeared {
            bloodline_id: bl_id,
            adventurer_id: adv_id,
            name: descendant_name,
            founder_name: founder_name.clone(),
            generation: generations + 1,
        });
    }
}

/// Inherit approximately 50% of history tags from the parent.
fn inherit_tags(
    tags: &std::collections::HashMap<String, u32>,
    rng: &mut u64,
) -> Vec<String> {
    let mut inherited = Vec::new();
    for key in tags.keys() {
        if lcg_f32(rng) < TRAIT_INHERITANCE_RATE {
            inherited.push(key.clone());
        }
    }
    inherited
}

/// Base stats lookup by archetype (matches recruitment.rs).
fn base_stats_for_archetype(archetype: &str) -> (f32, f32, f32, f32, f32) {
    match archetype {
        "knight" => (110.0, 12.0, 18.0, 7.0, 4.0),
        "ranger" => (75.0, 16.0, 8.0, 13.0, 7.0),
        "mage" => (55.0, 6.0, 5.0, 9.0, 22.0),
        "cleric" => (65.0, 5.0, 10.0, 8.0, 18.0),
        "rogue" => (65.0, 18.0, 6.0, 15.0, 6.0),
        "paladin" => (100.0, 10.0, 15.0, 6.0, 10.0),
        "berserker" => (95.0, 22.0, 5.0, 10.0, 3.0),
        "necromancer" => (50.0, 8.0, 4.0, 7.0, 24.0),
        "bard" => (60.0, 7.0, 7.0, 11.0, 15.0),
        "druid" => (70.0, 8.0, 9.0, 9.0, 16.0),
        "warlock" => (55.0, 10.0, 5.0, 8.0, 20.0),
        "monk" => (75.0, 14.0, 10.0, 16.0, 8.0),
        "assassin" => (60.0, 20.0, 4.0, 17.0, 5.0),
        "guardian" => (120.0, 8.0, 20.0, 5.0, 3.0),
        "shaman" => (65.0, 7.0, 8.0, 8.0, 18.0),
        "artificer" => (60.0, 9.0, 7.0, 10.0, 16.0),
        "tank" => (130.0, 6.0, 22.0, 4.0, 2.0),
        _ => (70.0, 10.0, 10.0, 10.0, 10.0),
    }
}

/// Generate a descendant's given name from a fixed pool.
fn descendant_given_name(rng: &mut u64) -> String {
    let names = [
        "Aethan", "Britta", "Corwin", "Dalla", "Erren", "Freya", "Gareth", "Hilde",
        "Idris", "Jorun", "Kael", "Liora", "Magnus", "Nessa", "Osric", "Priya",
        "Riven", "Sigrid", "Torbin", "Ula", "Voss", "Wynne", "Xander", "Yara",
    ];
    let idx = (lcg_next(rng) as usize) % names.len();
    names[idx].to_string()
}
