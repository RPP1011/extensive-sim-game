//! Loot generation and equipment system.
//!
//! Grammar-walked loot generator that creates equipment drops on quest victory.
//! Quality scales with threat level, item type biased by quest type and archetype.

use crate::headless_campaign::state::*;

// ---------------------------------------------------------------------------
// Name tables
// ---------------------------------------------------------------------------

const PREFIXES_COMMON: &[&str] = &["Rusty", "Worn", "Simple", "Crude"];
const PREFIXES_UNCOMMON: &[&str] = &["Iron", "Steel", "Sturdy", "Tempered"];
const PREFIXES_RARE: &[&str] = &["Enchanted", "Ancient", "Blessed", "Masterwork"];
const PREFIXES_EPIC: &[&str] = &["Shadowforged", "Runescribed", "Dragon", "Mythril"];

const WEAPON_NAMES: &[&str] = &["Sword", "Axe", "Mace", "Spear", "Dagger", "Staff", "Bow"];
const OFFHAND_NAMES: &[&str] = &["Shield", "Buckler", "Tome", "Orb", "Lantern"];
const CHEST_NAMES: &[&str] = &["Plate", "Chainmail", "Robe", "Leather Vest", "Brigandine"];
const BOOTS_NAMES: &[&str] = &["Boots", "Greaves", "Sandals", "Treads", "Sabatons"];
const ACCESSORY_NAMES: &[&str] = &["Ring", "Cloak", "Amulet", "Gauntlets", "Circlet"];

// ---------------------------------------------------------------------------
// Slot weights by quest type
// ---------------------------------------------------------------------------

/// Returns (weapon, offhand, chest, boots, accessory) weight tuples and drop chance.
fn quest_type_loot_profile(quest_type: QuestType) -> (f32, [f32; 5]) {
    match quest_type {
        QuestType::Combat => (0.70, [0.35, 0.15, 0.30, 0.10, 0.10]),
        QuestType::Exploration => (0.50, [0.10, 0.10, 0.10, 0.20, 0.50]),
        QuestType::Rescue => (0.30, [0.10, 0.25, 0.10, 0.35, 0.20]),
        QuestType::Gather => (0.20, [0.10, 0.10, 0.10, 0.10, 0.60]),
        QuestType::Diplomatic => (0.15, [0.05, 0.05, 0.05, 0.15, 0.70]),
        QuestType::Escort => (0.40, [0.20, 0.20, 0.20, 0.20, 0.20]),
    }
}

/// Bias slot weights by archetype. Returns additive adjustments.
fn archetype_slot_bias(archetype: &str) -> [f32; 5] {
    let lower = archetype.to_lowercase();
    if matches!(
        lower.as_str(),
        "knight" | "guardian" | "tank" | "paladin" | "sentinel"
    ) {
        // Bias toward armor and shields
        [0.0, 0.15, 0.20, 0.0, -0.10]
    } else if matches!(
        lower.as_str(),
        "ranger" | "assassin" | "rogue" | "hunter" | "duelist"
    ) {
        // Bias toward weapons and boots
        [0.20, 0.0, -0.10, 0.15, 0.0]
    } else if matches!(
        lower.as_str(),
        "mage" | "warlock" | "necromancer" | "sorcerer" | "elementalist" | "wizard"
    ) {
        // Bias toward accessories and rings
        [-0.10, 0.0, -0.10, 0.0, 0.30]
    } else if matches!(
        lower.as_str(),
        "cleric" | "healer" | "priest" | "druid" | "shaman"
    ) {
        // Bias toward accessories
        [-0.10, 0.10, 0.0, 0.0, 0.20]
    } else {
        [0.0; 5]
    }
}

// ---------------------------------------------------------------------------
// Quality calculation
// ---------------------------------------------------------------------------

fn quality_from_threat(threat: f32, rng: &mut u64) -> f32 {
    let base = if threat < 20.0 {
        0.1 + (threat / 20.0) * 0.2
    } else if threat < 50.0 {
        0.3 + ((threat - 20.0) / 30.0) * 0.3
    } else if threat < 80.0 {
        0.6 + ((threat - 50.0) / 30.0) * 0.2
    } else {
        0.8 + ((threat - 80.0) / 20.0).min(1.0) * 0.2
    };
    // Add some variance (±0.05)
    let variance = (lcg_f32(rng) - 0.5) * 0.10;
    (base + variance).clamp(0.05, 1.0)
}

// ---------------------------------------------------------------------------
// Stat bonus calculation
// ---------------------------------------------------------------------------

fn compute_stat_bonuses(slot: EquipmentSlot, quality: f32) -> StatBonuses {
    match slot {
        EquipmentSlot::Weapon => StatBonuses {
            hp_bonus: 0.0,
            attack_bonus: quality * 25.0,
            defense_bonus: 0.0,
            speed_bonus: 0.0,
        },
        EquipmentSlot::Offhand => StatBonuses {
            hp_bonus: quality * 20.0,
            attack_bonus: 0.0,
            defense_bonus: quality * 25.0,
            speed_bonus: 0.0,
        },
        EquipmentSlot::Chest => StatBonuses {
            hp_bonus: quality * 50.0,
            attack_bonus: 0.0,
            defense_bonus: quality * 25.0,
            speed_bonus: 0.0,
        },
        EquipmentSlot::Boots => StatBonuses {
            hp_bonus: 0.0,
            attack_bonus: 0.0,
            defense_bonus: quality * 10.0,
            speed_bonus: quality * 10.0,
        },
        EquipmentSlot::Accessory => StatBonuses {
            hp_bonus: quality * 15.0,
            attack_bonus: quality * 10.0,
            defense_bonus: quality * 5.0,
            speed_bonus: quality * 5.0,
        },
    }
}

fn total_bonus(b: &StatBonuses) -> f32 {
    b.hp_bonus + b.attack_bonus + b.defense_bonus + b.speed_bonus
}

// ---------------------------------------------------------------------------
// Name generation
// ---------------------------------------------------------------------------

fn pick_from(list: &[&str], rng: &mut u64) -> String {
    let idx = lcg_next(rng) as usize % list.len();
    list[idx].to_string()
}

fn generate_name(slot: EquipmentSlot, quality: f32, rng: &mut u64) -> String {
    let prefix = if quality < 0.3 {
        pick_from(PREFIXES_COMMON, rng)
    } else if quality < 0.6 {
        pick_from(PREFIXES_UNCOMMON, rng)
    } else if quality < 0.8 {
        pick_from(PREFIXES_RARE, rng)
    } else {
        pick_from(PREFIXES_EPIC, rng)
    };

    let base = match slot {
        EquipmentSlot::Weapon => pick_from(WEAPON_NAMES, rng),
        EquipmentSlot::Offhand => pick_from(OFFHAND_NAMES, rng),
        EquipmentSlot::Chest => pick_from(CHEST_NAMES, rng),
        EquipmentSlot::Boots => pick_from(BOOTS_NAMES, rng),
        EquipmentSlot::Accessory => pick_from(ACCESSORY_NAMES, rng),
    };

    format!("{} {}", prefix, base)
}

// ---------------------------------------------------------------------------
// Slot selection
// ---------------------------------------------------------------------------

fn select_slot(weights: &[f32; 5], rng: &mut u64) -> EquipmentSlot {
    let total: f32 = weights.iter().map(|w| w.max(0.0)).sum();
    if total <= 0.0 {
        return EquipmentSlot::Accessory;
    }
    let mut roll = lcg_f32(rng) * total;
    let slots = [
        EquipmentSlot::Weapon,
        EquipmentSlot::Offhand,
        EquipmentSlot::Chest,
        EquipmentSlot::Boots,
        EquipmentSlot::Accessory,
    ];
    for (i, &w) in weights.iter().enumerate() {
        let clamped = w.max(0.0);
        if roll < clamped {
            return slots[i];
        }
        roll -= clamped;
    }
    EquipmentSlot::Accessory
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate a loot item from a quest victory.
///
/// Returns `None` if the RNG roll fails the drop chance check.
pub fn generate_loot(
    quest_type: QuestType,
    threat_level: f32,
    adventurer_archetype: &str,
    rng: &mut u64,
) -> Option<InventoryItem> {
    let (drop_chance, base_weights) = quest_type_loot_profile(quest_type);

    // Drop chance check
    if lcg_f32(rng) > drop_chance {
        return None;
    }

    // Combine base weights with archetype bias
    let bias = archetype_slot_bias(adventurer_archetype);
    let mut weights = [0.0f32; 5];
    for i in 0..5 {
        weights[i] = (base_weights[i] + bias[i]).max(0.01);
    }

    let slot = select_slot(&weights, rng);
    let quality = quality_from_threat(threat_level, rng);
    let name = generate_name(slot, quality, rng);
    let stat_bonuses = compute_stat_bonuses(slot, quality);

    // Use a hash of rng state as item ID (unique enough for campaign scope)
    let id = lcg_next(rng);

    Some(InventoryItem {
        id,
        name,
        slot,
        quality,
        stat_bonuses,
        durability: 100.0,
        appraised: false,
    })
}

/// Equip an item on an adventurer, adding its stat bonuses.
///
/// If the slot is already occupied, keeps the item with higher total bonuses.
/// Returns `true` if the new item was equipped, `false` if the old one was kept.
pub fn try_equip_item(
    adventurer: &mut Adventurer,
    item: &InventoryItem,
    inventory: &[InventoryItem],
) -> bool {
    let slot_ref = match item.slot {
        EquipmentSlot::Weapon => &mut adventurer.equipment.weapon,
        EquipmentSlot::Offhand => &mut adventurer.equipment.offhand,
        EquipmentSlot::Chest => &mut adventurer.equipment.chest,
        EquipmentSlot::Boots => &mut adventurer.equipment.boots,
        EquipmentSlot::Accessory => &mut adventurer.equipment.accessory,
    };

    if let Some(existing_id) = *slot_ref {
        // Compare with existing item
        if let Some(existing) = inventory.iter().find(|i| i.id == existing_id) {
            if total_bonus(&existing.stat_bonuses) >= total_bonus(&item.stat_bonuses) {
                return false; // Keep the existing item
            }
            // Remove old item's stat bonuses
            unapply_bonuses(&mut adventurer.stats, &existing.stat_bonuses);
        }
    }

    // Equip the new item
    *slot_ref = Some(item.id);
    apply_bonuses(&mut adventurer.stats, &item.stat_bonuses);
    true
}

/// Add stat bonuses from an item to adventurer stats.
fn apply_bonuses(stats: &mut AdventurerStats, bonuses: &StatBonuses) {
    stats.max_hp += bonuses.hp_bonus;
    stats.attack += bonuses.attack_bonus;
    stats.defense += bonuses.defense_bonus;
    stats.speed += bonuses.speed_bonus;
}

/// Remove stat bonuses from an item from adventurer stats.
fn unapply_bonuses(stats: &mut AdventurerStats, bonuses: &StatBonuses) {
    stats.max_hp -= bonuses.hp_bonus;
    stats.attack -= bonuses.attack_bonus;
    stats.defense -= bonuses.defense_bonus;
    stats.speed -= bonuses.speed_bonus;
}

/// Process loot for a completed quest victory.
///
/// Generates loot for the best-matching party member and auto-equips it.
/// The item is always added to guild inventory regardless of equip outcome.
pub fn process_quest_loot(
    state: &mut CampaignState,
    quest_type: QuestType,
    threat_level: f32,
    potential_loot: bool,
    member_ids: &[u32],
) {
    if !potential_loot || member_ids.is_empty() {
        return;
    }

    // Pick the first alive member's archetype for biasing
    let archetype = state
        .adventurers
        .iter()
        .find(|a| member_ids.contains(&a.id) && a.status != AdventurerStatus::Dead)
        .map(|a| a.archetype.clone())
        .unwrap_or_default();

    let item = match generate_loot(quest_type, threat_level, &archetype, &mut state.rng) {
        Some(item) => item,
        None => return,
    };

    // Add to guild inventory
    state.guild.inventory.push(item.clone());

    // Find the best adventurer to equip this item:
    // prefer the archetype-matching member, then any alive member
    let best_member_id = find_best_recipient(state, &item, member_ids);

    if let Some(mid) = best_member_id {
        // Need to clone inventory for borrow checker (items are small)
        let inv_snapshot: Vec<InventoryItem> = state.guild.inventory.clone();
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
            try_equip_item(adv, &item, &inv_snapshot);
        }
    }
}

/// Find the best party member to receive an item based on archetype affinity.
fn find_best_recipient(
    state: &CampaignState,
    item: &InventoryItem,
    member_ids: &[u32],
) -> Option<u32> {
    let alive_members: Vec<&Adventurer> = state
        .adventurers
        .iter()
        .filter(|a| member_ids.contains(&a.id) && a.status != AdventurerStatus::Dead)
        .collect();

    if alive_members.is_empty() {
        return None;
    }

    // Score each member: archetype match bonus + empty slot bonus
    let mut best_id = alive_members[0].id;
    let mut best_score = -1.0f32;

    for adv in &alive_members {
        let mut score = 0.0f32;

        // Archetype affinity
        let bias = archetype_slot_bias(&adv.archetype);
        let slot_idx = match item.slot {
            EquipmentSlot::Weapon => 0,
            EquipmentSlot::Offhand => 1,
            EquipmentSlot::Chest => 2,
            EquipmentSlot::Boots => 3,
            EquipmentSlot::Accessory => 4,
        };
        score += bias[slot_idx];

        // Empty slot bonus
        let slot_occupied = match item.slot {
            EquipmentSlot::Weapon => adv.equipment.weapon.is_some(),
            EquipmentSlot::Offhand => adv.equipment.offhand.is_some(),
            EquipmentSlot::Chest => adv.equipment.chest.is_some(),
            EquipmentSlot::Boots => adv.equipment.boots.is_some(),
            EquipmentSlot::Accessory => adv.equipment.accessory.is_some(),
        };
        if !slot_occupied {
            score += 1.0;
        }

        if score > best_score {
            best_score = score;
            best_id = adv.id;
        }
    }

    Some(best_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_loot_deterministic() {
        let mut rng = 12345u64;
        let item = generate_loot(QuestType::Combat, 50.0, "knight", &mut rng);
        assert!(item.is_some(), "Combat at threat 50 should usually drop loot");
        let item = item.unwrap();
        assert!(!item.name.is_empty());
        assert!(item.quality >= 0.05 && item.quality <= 1.0);

        // Same seed should give same result
        let mut rng2 = 12345u64;
        let item2 = generate_loot(QuestType::Combat, 50.0, "knight", &mut rng2).unwrap();
        assert_eq!(item.name, item2.name);
        assert_eq!(item.quality, item2.quality);
    }

    #[test]
    fn test_quality_scales_with_threat() {
        // Average quality over many rolls should increase with threat
        let mut low_sum = 0.0f32;
        let mut high_sum = 0.0f32;
        let n = 100;
        let mut rng = 42u64;
        for _ in 0..n {
            low_sum += quality_from_threat(10.0, &mut rng);
            high_sum += quality_from_threat(90.0, &mut rng);
        }
        assert!(
            high_sum / n as f32 > low_sum / n as f32,
            "High threat should produce higher quality on average"
        );
    }

    #[test]
    fn test_low_drop_chance_quests() {
        // Diplomatic quests (15% chance) should rarely drop
        let mut drops = 0;
        let mut rng = 999u64;
        for _ in 0..100 {
            if generate_loot(QuestType::Diplomatic, 30.0, "mage", &mut rng).is_some() {
                drops += 1;
            }
        }
        assert!(drops < 40, "Diplomatic quests should have low drop rate, got {}", drops);
    }

    #[test]
    fn test_stat_bonuses_weapon_vs_chest() {
        let weapon_bonus = compute_stat_bonuses(EquipmentSlot::Weapon, 0.5);
        let chest_bonus = compute_stat_bonuses(EquipmentSlot::Chest, 0.5);

        assert!(weapon_bonus.attack_bonus > 0.0, "Weapons should have attack bonus");
        assert_eq!(weapon_bonus.defense_bonus, 0.0, "Weapons should not have defense bonus");
        assert!(chest_bonus.defense_bonus > 0.0, "Chest should have defense bonus");
        assert!(chest_bonus.hp_bonus > 0.0, "Chest should have HP bonus");
    }

    #[test]
    fn test_equip_replaces_worse_item() {
        let mut adv = Adventurer {
            id: 1,
            name: "Test".into(),
            archetype: "knight".into(),
            level: 1,
            xp: 0,
            stats: AdventurerStats {
                max_hp: 100.0,
                attack: 10.0,
                defense: 10.0,
                speed: 5.0,
                ability_power: 0.0,
            },
            equipment: Equipment::default(),
            traits: vec![],
            status: AdventurerStatus::Idle,
            loyalty: 80.0,
            stress: 0.0,
            fatigue: 0.0,
            injury: 0.0,
            resolve: 50.0,
            morale: 80.0,
            party_id: None,
            guild_relationship: 50.0,
            leadership_role: None,
            is_player_character: false,
            faction_id: None,
            rallying_to: None,
            tier_status: crate::headless_campaign::unit_tiers::UnitTierStatus::default(),
            history_tags: std::collections::HashMap::new(),
        };

        let weak_item = InventoryItem {
            id: 100,
            name: "Rusty Sword".into(),
            slot: EquipmentSlot::Weapon,
            quality: 0.2,
            stat_bonuses: compute_stat_bonuses(EquipmentSlot::Weapon, 0.2),
            durability: 100.0,
            appraised: false,
        };

        let strong_item = InventoryItem {
            id: 200,
            name: "Dragon Sword".into(),
            slot: EquipmentSlot::Weapon,
            quality: 0.9,
            stat_bonuses: compute_stat_bonuses(EquipmentSlot::Weapon, 0.9),
            durability: 100.0,
            appraised: false,
        };

        // Equip weak item first
        let inv = vec![weak_item.clone()];
        assert!(try_equip_item(&mut adv, &weak_item, &inv));
        assert_eq!(adv.equipment.weapon, Some(100));
        let attack_with_weak = adv.stats.attack;
        assert!(attack_with_weak > 10.0, "Attack should increase with weapon");

        // Replace with strong item
        let inv = vec![weak_item.clone(), strong_item.clone()];
        assert!(try_equip_item(&mut adv, &strong_item, &inv));
        assert_eq!(adv.equipment.weapon, Some(200));
        assert!(
            adv.stats.attack > attack_with_weak,
            "Attack should increase with better weapon"
        );

        // Try to replace with weak item — should fail
        let inv = vec![weak_item.clone(), strong_item.clone()];
        assert!(!try_equip_item(&mut adv, &weak_item, &inv));
        assert_eq!(adv.equipment.weapon, Some(200));
    }
}
