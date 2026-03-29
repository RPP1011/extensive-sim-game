//! Legacy weapon system — every 500 ticks.
//!
//! Weapons that grow alongside their wielder, gaining abilities and lore over
//! time. A legacy weapon never breaks (exempt from durability) and becomes a
//! guild artifact when its wielder dies, eligible for inheritance.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::{
    AdventurerStatus, CampaignState, LegacyWeapon, lcg_f32, lcg_next,
};

/// How often to tick legacy weapons (in ticks).
const LEGACY_WEAPON_INTERVAL: u64 = 17;

/// Chance for an adventurer reaching level 5+ with equipped weapon to forge a legacy weapon.
const CREATION_CHANCE: f32 = 0.05;

/// XP thresholds for weapon levels 1–5.
const LEVEL_THRESHOLDS: [u32; 5] = [10, 25, 50, 100, 200];

/// Abilities unlocked at each weapon level (one per level).
const LEVEL_ABILITIES: [&str; 5] = [
    "Keen",      // +10% crit
    "Sturdy",    // no degradation
    "Bane",      // +20% vs specific monster
    "Vampiric",  // +5% lifesteal
    "Radiant",   // +5 morale aura
];

/// First-name templates for weapon naming at level 3.
const WEAPON_FIRST_NAMES: &[&str] = &[
    "Dawn", "Dusk", "Storm", "Iron", "Shadow", "Flame", "Frost", "Thunder",
    "Ruin", "Glory", "Fury", "Grace", "Wrath", "Hope", "Blight", "Star",
];

/// Suffix templates for weapon naming at level 3.
const WEAPON_SUFFIXES: &[&str] = &[
    "bringer", "fang", "edge", "heart", "bane", "song", "caller", "strike",
    "cleaver", "keeper", "warden", "reaver", "bloom", "thorn", "shard", "crest",
];

/// Tick the legacy weapon system. Called every `LEGACY_WEAPON_INTERVAL` ticks.
///
/// 1. Try to create new legacy weapons for eligible adventurers.
/// 2. Award XP to existing weapons based on wielder activity.
/// 3. Level up weapons that cross XP thresholds.
/// 4. Handle inheritance when wielders die.
pub fn tick_legacy_weapons(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % LEGACY_WEAPON_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Create new legacy weapons ---
    let adv_count = state.adventurers.len();
    for i in 0..adv_count {
        let adv = &state.adventurers[i];
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        // Requires level >= 5 and at least one equipped weapon
        if adv.level < 5 {
            continue;
        }
        let has_weapon = adv.equipment.weapon.is_some();
        if !has_weapon {
            continue;
        }
        // Skip if adventurer already wields a legacy weapon
        let adv_id = adv.id;
        let already_has = state
            .legacy_weapons
            .iter()
            .any(|w| w.wielder_id == Some(adv_id));
        if already_has {
            continue;
        }
        let roll = lcg_f32(&mut state.rng);
        if roll < CREATION_CHANCE {
            let weapon_id = state.next_legacy_weapon_id;
            state.next_legacy_weapon_id += 1;
            let base_attack = state.adventurers[i].stats.attack * 0.1;
            let weapon = LegacyWeapon {
                id: weapon_id,
                name: format!("{}'s Weapon", state.adventurers[i].name),
                wielder_id: Some(adv_id),
                base_attack,
                bonus_attack: 0.0,
                kills: 0,
                battles_survived: 0,
                xp: 0,
                level: 1,
                abilities: Vec::new(),
                created_tick: state.tick,
            };
            state.legacy_weapons.push(weapon);
            events.push(WorldEvent::LegacyWeaponCreated {
                weapon_id,
                wielder_id: adv_id,
                name: format!("{}'s Weapon", state.adventurers[i].name),
            });
        }
    }

    // --- Phase 2: Award XP and track wielder activity ---
    let weapon_count = state.legacy_weapons.len();
    for w in 0..weapon_count {
        let wielder_id = match state.legacy_weapons[w].wielder_id {
            Some(id) => id,
            None => continue, // guild artifact, no wielder
        };

        // Find wielder
        let adv_idx = match state.adventurers.iter().position(|a| a.id == wielder_id) {
            Some(idx) => idx,
            None => continue,
        };

        let adv = &state.adventurers[adv_idx];

        // Check if wielder died — transfer to guild artifacts
        if adv.status == AdventurerStatus::Dead {
            state.legacy_weapons[w].wielder_id = None;
            let weapon_name = state.legacy_weapons[w].name.clone();
            let weapon_id = state.legacy_weapons[w].id;
            events.push(WorldEvent::LegacyWeaponInherited {
                weapon_id,
                old_wielder_id: wielder_id,
                new_wielder_id: None,
                weapon_name,
            });
            continue;
        }

        // Award XP based on activity
        let mut xp_gain: u32 = 0;

        // +1 per battle survived (check if adventurer is fighting or just finished)
        if adv.status == AdventurerStatus::Fighting
            || adv.status == AdventurerStatus::OnMission
        {
            xp_gain += 1;
            state.legacy_weapons[w].battles_survived += 1;
        }

        // +2 for nemesis fight: check if any active battle involves this adventurer's party
        // and a nemesis is present
        if let Some(party_id) = adv.party_id {
            let in_nemesis_fight = state.active_battles.iter().any(|b| {
                b.party_id == party_id
                    && state.nemeses.iter().any(|n| {
                        !n.defeated && n.region_id.is_some()
                    })
            });
            if in_nemesis_fight {
                xp_gain += 2;
            }
        }

        // +3 for boss kill: high threat quests (threat_level >= 0.8) that were won
        if let Some(party_id) = adv.party_id {
            let boss_victory = state.active_battles.iter().any(|b| {
                b.party_id == party_id
                    && b.enemy_strength >= 80.0
                    && b.status == crate::state::BattleStatus::Victory
            });
            if boss_victory {
                xp_gain += 3;
                state.legacy_weapons[w].kills += 1;
            }
        }

        if xp_gain > 0 {
            state.legacy_weapons[w].xp += xp_gain;
        }

        // --- Phase 3: Level up ---
        let current_level = state.legacy_weapons[w].level;
        if current_level >= 5 {
            continue; // max level
        }
        let threshold = LEVEL_THRESHOLDS[(current_level - 1).min(4) as usize];
        if state.legacy_weapons[w].xp >= threshold && current_level < 5 {
            state.legacy_weapons[w].level += 1;
            let new_level = state.legacy_weapons[w].level;
            let ability_idx = (new_level - 1).min(4) as usize;

            // Choose ability based on weapon history
            let ability = choose_ability(state, w, ability_idx);
            state.legacy_weapons[w].abilities.push(ability.clone());

            // Bonus attack per level
            state.legacy_weapons[w].bonus_attack +=
                state.legacy_weapons[w].base_attack * 0.1;

            let weapon_id = state.legacy_weapons[w].id;
            let weapon_name = state.legacy_weapons[w].name.clone();

            events.push(WorldEvent::LegacyWeaponLevelUp {
                weapon_id,
                new_level,
                ability: ability.clone(),
            });

            // At level 3, weapon gets a proper name
            if new_level == 3 {
                let first_idx =
                    (lcg_next(&mut state.rng) as usize) % WEAPON_FIRST_NAMES.len();
                let suffix_idx =
                    (lcg_next(&mut state.rng) as usize) % WEAPON_SUFFIXES.len();
                let proper_name = format!(
                    "{}{}",
                    WEAPON_FIRST_NAMES[first_idx], WEAPON_SUFFIXES[suffix_idx]
                );
                let old_name = weapon_name;
                state.legacy_weapons[w].name = proper_name.clone();
                events.push(WorldEvent::LegacyWeaponNamed {
                    weapon_id,
                    old_name,
                    new_name: proper_name,
                });
            }
        }
    }

    // --- Phase 4: Inheritance — pass unowned weapons to eligible adventurers ---
    let weapon_count = state.legacy_weapons.len();
    for w in 0..weapon_count {
        if state.legacy_weapons[w].wielder_id.is_some() {
            continue; // already wielded
        }
        // Find a living adventurer with no legacy weapon, level >= 3, with a weapon equipped
        let candidate = state.adventurers.iter().find(|a| {
            a.status != AdventurerStatus::Dead
                && a.level >= 3
                && a.equipment.weapon.is_some()
                && !state
                    .legacy_weapons
                    .iter()
                    .any(|lw| lw.wielder_id == Some(a.id))
        });
        if let Some(new_wielder) = candidate {
            let new_wielder_id = new_wielder.id;
            state.legacy_weapons[w].wielder_id = Some(new_wielder_id);
            let weapon_name = state.legacy_weapons[w].name.clone();
            let weapon_id = state.legacy_weapons[w].id;
            events.push(WorldEvent::LegacyWeaponInherited {
                weapon_id,
                old_wielder_id: 0, // guild artifact, no previous wielder
                new_wielder_id: Some(new_wielder_id),
                weapon_name,
            });
        }
    }
}

/// Choose an ability for a weapon level-up based on weapon history.
/// Uses deterministic RNG seeded by weapon state.
fn choose_ability(state: &mut CampaignState, weapon_idx: usize, _ability_tier: usize) -> String {
    let weapon = &state.legacy_weapons[weapon_idx];

    // Prefer abilities not already on this weapon
    let mut candidates: Vec<&str> = LEVEL_ABILITIES
        .iter()
        .copied()
        .filter(|a| !weapon.abilities.iter().any(|existing| existing == a))
        .collect();

    if candidates.is_empty() {
        candidates = LEVEL_ABILITIES.to_vec();
    }

    // Bias selection based on weapon history
    // High kills → Bane, high battles → Sturdy, etc.
    let idx = if weapon.kills > 5 && candidates.contains(&"Bane") {
        candidates.iter().position(|a| *a == "Bane").unwrap()
    } else if weapon.battles_survived > 20 && candidates.contains(&"Sturdy") {
        candidates.iter().position(|a| *a == "Sturdy").unwrap()
    } else {
        (lcg_next(&mut state.rng) as usize) % candidates.len()
    };

    candidates[idx].to_string()
}
