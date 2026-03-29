//! Legacy artifact system — every 500 ticks.
//!
//! Artifacts are created when legendary adventurers die or retire, or when
//! high-threat quests are completed. Unequipped artifacts in the guild vault
//! emit reminder events for potential wielders.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::{
    AdventurerStatus, Artifact, ArtifactEffect, ArtifactStats, CampaignState,
    lcg_f32, lcg_next,
};

// ---------------------------------------------------------------------------
// Artifact name generation
// ---------------------------------------------------------------------------

const WEAPON_TITLES: &[&str] = &["Blade", "Edge", "Fang", "Wrath", "Bane"];
const ARMOR_TITLES: &[&str] = &["Aegis", "Bulwark", "Ward", "Bastion", "Shell"];
const ACCESSORY_TITLES: &[&str] = &["Sigil", "Relic", "Token", "Charm", "Talisman"];

/// Generate a name for an artifact based on the adventurer and slot.
fn generate_artifact_name(adventurer_name: &str, slot: &str, rng: &mut u64) -> String {
    let titles = match slot {
        "Weapon" => WEAPON_TITLES,
        "Offhand" | "Chest" | "Boots" => ARMOR_TITLES,
        _ => ACCESSORY_TITLES,
    };
    let idx = lcg_next(rng) as usize % titles.len();
    let title = titles[idx];

    // Two naming patterns: "{Name}'s {Title}" or "The {Title} of {Name}"
    if lcg_f32(rng) < 0.5 {
        format!("{}'s {}", adventurer_name, title)
    } else {
        format!("The {} of {}", title, adventurer_name)
    }
}

/// Generate an origin deed description from context.
fn generate_deed(adventurer_name: &str, cause: &str, rng: &mut u64) -> String {
    let templates = [
        format!("{} fell in glorious battle", adventurer_name),
        format!("{} gave their life defending the guild", adventurer_name),
        format!("{}'s legend lives on through their gear", adventurer_name),
        format!("{} perished: {}", adventurer_name, cause),
    ];
    let idx = lcg_next(rng) as usize % templates.len();
    templates[idx].clone()
}

// ---------------------------------------------------------------------------
// Artifact effect selection
// ---------------------------------------------------------------------------

/// Choose a special effect based on the adventurer's archetype.
fn effect_for_archetype(archetype: &str, rng: &mut u64) -> ArtifactEffect {
    let lower = archetype.to_lowercase();
    let roll = lcg_f32(rng);

    if matches!(lower.as_str(), "knight" | "guardian" | "tank" | "paladin" | "sentinel") {
        if roll < 0.4 {
            ArtifactEffect::CombatPowerBoost(0.08 + lcg_f32(rng) * 0.07)
        } else if roll < 0.7 {
            ArtifactEffect::MoraleAura(5.0 + lcg_f32(rng) * 5.0)
        } else {
            ArtifactEffect::ThreatReduction(3.0 + lcg_f32(rng) * 4.0)
        }
    } else if matches!(lower.as_str(), "ranger" | "assassin" | "rogue" | "hunter" | "duelist") {
        if roll < 0.4 {
            ArtifactEffect::CombatPowerBoost(0.10 + lcg_f32(rng) * 0.10)
        } else if roll < 0.7 {
            ArtifactEffect::XpMultiplier(0.10 + lcg_f32(rng) * 0.10)
        } else {
            ArtifactEffect::ThreatReduction(4.0 + lcg_f32(rng) * 6.0)
        }
    } else if matches!(lower.as_str(), "mage" | "warlock" | "necromancer" | "sorcerer" | "elementalist" | "wizard") {
        if roll < 0.4 {
            ArtifactEffect::XpMultiplier(0.12 + lcg_f32(rng) * 0.08)
        } else if roll < 0.7 {
            ArtifactEffect::FactionInfluence(5.0 + lcg_f32(rng) * 5.0)
        } else {
            ArtifactEffect::CombatPowerBoost(0.06 + lcg_f32(rng) * 0.09)
        }
    } else if matches!(lower.as_str(), "cleric" | "healer" | "priest" | "druid" | "shaman") {
        if roll < 0.4 {
            ArtifactEffect::HealingBoost(0.15 + lcg_f32(rng) * 0.10)
        } else if roll < 0.7 {
            ArtifactEffect::MoraleAura(8.0 + lcg_f32(rng) * 7.0)
        } else {
            ArtifactEffect::FactionInfluence(4.0 + lcg_f32(rng) * 6.0)
        }
    } else {
        // Generic archetype
        match (lcg_next(rng) % 6) as u8 {
            0 => ArtifactEffect::MoraleAura(5.0 + lcg_f32(rng) * 5.0),
            1 => ArtifactEffect::CombatPowerBoost(0.05 + lcg_f32(rng) * 0.10),
            2 => ArtifactEffect::XpMultiplier(0.08 + lcg_f32(rng) * 0.12),
            3 => ArtifactEffect::FactionInfluence(3.0 + lcg_f32(rng) * 7.0),
            4 => ArtifactEffect::ThreatReduction(3.0 + lcg_f32(rng) * 5.0),
            _ => ArtifactEffect::HealingBoost(0.10 + lcg_f32(rng) * 0.15),
        }
    }
}

/// Compute artifact stat bonuses based on adventurer level and slot.
fn compute_artifact_stats(level: u32, slot: &str, rng: &mut u64) -> ArtifactStats {
    let power = level as f32 * 2.0 + lcg_f32(rng) * 5.0;
    match slot {
        "Weapon" => ArtifactStats {
            attack: power * 1.5,
            defense: 0.0,
            hp: 0.0,
            speed: power * 0.3,
        },
        "Offhand" => ArtifactStats {
            attack: 0.0,
            defense: power * 1.2,
            hp: power * 1.0,
            speed: 0.0,
        },
        "Chest" => ArtifactStats {
            attack: 0.0,
            defense: power * 1.0,
            hp: power * 2.0,
            speed: 0.0,
        },
        "Boots" => ArtifactStats {
            attack: 0.0,
            defense: power * 0.5,
            hp: 0.0,
            speed: power * 1.5,
        },
        _ => ArtifactStats {
            attack: power * 0.5,
            defense: power * 0.3,
            hp: power * 0.5,
            speed: power * 0.3,
        },
    }
}

/// Pick the best equipment slot name from an adventurer's gear.
fn best_equipment_slot(equipment: &crate::state::Equipment) -> &'static str {
    // Prefer weapon, then chest, then other slots
    if equipment.weapon.is_some() {
        "Weapon"
    } else if equipment.chest.is_some() {
        "Chest"
    } else if equipment.offhand.is_some() {
        "Offhand"
    } else if equipment.boots.is_some() {
        "Boots"
    } else {
        "Accessory"
    }
}

// ---------------------------------------------------------------------------
// Public API: artifact creation triggers
// ---------------------------------------------------------------------------

/// Attempt to create an artifact from an adventurer who died.
///
/// Requires level >= 8, 30% chance. Called from quest_lifecycle when
/// an adventurer dies.
pub fn try_create_death_artifact(
    state: &mut CampaignState,
    adventurer_id: u32,
    cause: &str,
    events: &mut Vec<WorldEvent>,
) {
    let adv = match state.adventurers.iter().find(|a| a.id == adventurer_id) {
        Some(a) => a,
        None => return,
    };

    if adv.level < 8 {
        return;
    }

    // 30% chance
    if lcg_f32(&mut state.rng) > 0.30 {
        return;
    }

    let name = adv.name.clone();
    let archetype = adv.archetype.clone();
    let level = adv.level;
    let equipment = adv.equipment.clone();
    let slot = best_equipment_slot(&equipment);

    let artifact_name = generate_artifact_name(&name, slot, &mut state.rng);
    let deed = generate_deed(&name, cause, &mut state.rng);
    let stat_bonuses = compute_artifact_stats(level, slot, &mut state.rng);
    let special_effect = effect_for_archetype(&archetype, &mut state.rng);

    let artifact_id = state.next_artifact_id;
    state.next_artifact_id += 1;

    let artifact = Artifact {
        id: artifact_id,
        name: artifact_name.clone(),
        origin_adventurer_name: name.clone(),
        origin_deed: deed.clone(),
        slot: slot.to_string(),
        stat_bonuses,
        special_effect,
        created_tick: state.tick,
        equipped_by: None,
    };

    state.artifacts.push(artifact);

    events.push(WorldEvent::ArtifactCreated {
        name: artifact_name,
        origin: format!("{}: {}", name, deed),
    });
}

/// Attempt to create an artifact from a quest completion (ancient artifact find).
///
/// Requires threat > 80, 10% chance.
pub fn try_create_quest_artifact(
    state: &mut CampaignState,
    threat_level: f32,
    member_ids: &[u32],
    events: &mut Vec<WorldEvent>,
) {
    if threat_level <= 80.0 {
        return;
    }

    // 10% chance
    if lcg_f32(&mut state.rng) > 0.10 {
        return;
    }

    // Use the first alive member for context
    let (adventurer_name, archetype, level) = state
        .adventurers
        .iter()
        .find(|a| member_ids.contains(&a.id) && a.status != AdventurerStatus::Dead)
        .map(|a| (a.name.clone(), a.archetype.clone(), a.level))
        .unwrap_or_else(|| ("Unknown".to_string(), "warrior".to_string(), 5));

    // Ancient artifacts are always accessories
    let slot = "Accessory";
    let artifact_name = {
        let ancient_names = ["Ancient", "Forgotten", "Primordial", "Lost", "Sealed"];
        let objects = ["Relic", "Talisman", "Amulet", "Sigil", "Crown"];
        let name_idx = lcg_next(&mut state.rng) as usize % ancient_names.len();
        let obj_idx = lcg_next(&mut state.rng) as usize % objects.len();
        format!("{} {}", ancient_names[name_idx], objects[obj_idx])
    };

    let deed = format!("Discovered during a perilous quest (threat {})", threat_level as u32);
    let stat_bonuses = compute_artifact_stats(level, slot, &mut state.rng);
    let special_effect = effect_for_archetype(&archetype, &mut state.rng);

    let artifact_id = state.next_artifact_id;
    state.next_artifact_id += 1;

    let artifact = Artifact {
        id: artifact_id,
        name: artifact_name.clone(),
        origin_adventurer_name: adventurer_name.clone(),
        origin_deed: deed,
        slot: slot.to_string(),
        stat_bonuses,
        special_effect,
        created_tick: state.tick,
        equipped_by: None,
    };

    state.artifacts.push(artifact);

    events.push(WorldEvent::ArtifactCreated {
        name: artifact_name,
        origin: format!("Found by {}'s party", adventurer_name),
    });
}

// ---------------------------------------------------------------------------
// Equip action
// ---------------------------------------------------------------------------

/// Equip an artifact on an adventurer. Applies stat bonuses.
pub fn equip_artifact(
    state: &mut CampaignState,
    artifact_id: u32,
    adventurer_id: u32,
    events: &mut Vec<WorldEvent>,
) -> Result<String, String> {
    // Validate artifact exists and is unequipped
    let artifact_idx = state.artifacts.iter().position(|a| a.id == artifact_id)
        .ok_or_else(|| format!("Artifact {} not found", artifact_id))?;

    if state.artifacts[artifact_idx].equipped_by.is_some() {
        return Err(format!("Artifact {} is already equipped", artifact_id));
    }

    // Validate adventurer exists and is idle
    let adv = state.adventurers.iter().find(|a| a.id == adventurer_id)
        .ok_or_else(|| format!("Adventurer {} not found", adventurer_id))?;

    if adv.status == AdventurerStatus::Dead {
        return Err("Cannot equip artifact on dead adventurer".into());
    }

    // Apply stat bonuses
    let bonuses = state.artifacts[artifact_idx].stat_bonuses.clone();
    let artifact_name = state.artifacts[artifact_idx].name.clone();

    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adventurer_id) {
        adv.stats.attack += bonuses.attack;
        adv.stats.defense += bonuses.defense;
        adv.stats.max_hp += bonuses.hp;
        adv.stats.speed += bonuses.speed;
    }

    state.artifacts[artifact_idx].equipped_by = Some(adventurer_id);

    events.push(WorldEvent::ArtifactEquipped {
        name: artifact_name.clone(),
        adventurer_id,
    });

    Ok(format!("Equipped {} on adventurer {}", artifact_name, adventurer_id))
}

// ---------------------------------------------------------------------------
// Periodic tick — every 500 ticks
// ---------------------------------------------------------------------------

/// Periodic artifact system tick. Emits vault reminders for unequipped artifacts.
pub fn tick_artifacts(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 17 != 0 {
        return;
    }

    // Clean up artifacts equipped by dead adventurers
    let dead_ids: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| a.status == AdventurerStatus::Dead)
        .map(|a| a.id)
        .collect();

    for artifact in &mut state.artifacts {
        if let Some(equipped_id) = artifact.equipped_by {
            if dead_ids.contains(&equipped_id) {
                artifact.equipped_by = None;
            }
        }
    }

    // Check for unequipped artifacts that a strong adventurer could use
    let unequipped_artifacts: Vec<(u32, String)> = state
        .artifacts
        .iter()
        .filter(|a| a.equipped_by.is_none())
        .map(|a| (a.id, a.name.clone()))
        .collect();

    if unequipped_artifacts.is_empty() {
        return;
    }

    // Find strong idle adventurers (level >= 5)
    let strong_idle: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| {
            a.status == AdventurerStatus::Idle && a.level >= 5
        })
        .map(|a| a.id)
        .collect();

    if !strong_idle.is_empty() && !unequipped_artifacts.is_empty() {
        // Emit a chronicle event about the vault
        let artifact_names: Vec<String> = unequipped_artifacts.iter().map(|(_, n)| n.clone()).collect();
        events.push(WorldEvent::CampaignMilestone {
            description: format!(
                "The guild vault holds {} unclaimed artifact{}: {}",
                artifact_names.len(),
                if artifact_names.len() == 1 { "" } else { "s" },
                artifact_names.join(", ")
            ),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_name_generation_deterministic() {
        let mut rng1 = 42u64;
        let mut rng2 = 42u64;
        let name1 = generate_artifact_name("Kael", "Weapon", &mut rng1);
        let name2 = generate_artifact_name("Kael", "Weapon", &mut rng2);
        assert_eq!(name1, name2);
        assert!(!name1.is_empty());
    }

    #[test]
    fn effect_for_archetype_deterministic() {
        let mut rng1 = 99u64;
        let mut rng2 = 99u64;
        let e1 = format!("{:?}", effect_for_archetype("knight", &mut rng1));
        let e2 = format!("{:?}", effect_for_archetype("knight", &mut rng2));
        assert_eq!(e1, e2);
    }

    #[test]
    fn artifact_stats_scale_with_level() {
        let mut rng1 = 100u64;
        let mut rng2 = 100u64;
        let low = compute_artifact_stats(3, "Weapon", &mut rng1);
        let high = compute_artifact_stats(15, "Weapon", &mut rng2);
        assert!(high.attack > low.attack, "Higher level should give more attack");
    }
}
