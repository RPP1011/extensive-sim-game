//! Seasonal traveling merchant system — fires every 500 ticks.
//!
//! NPC merchant caravans arrive at the guild seasonally with unique wares,
//! special deals, and rare items. Merchants stay for 500 ticks then depart.
//! Return visits improve with guild reputation, and rare legendary dealers
//! appear with 5% chance.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often to check for merchant arrivals/departures (in ticks).
const MERCHANT_INTERVAL: u64 = 17;

/// How long a merchant stays at the guild (in ticks).
const MERCHANT_STAY_DURATION: u64 = 17;

/// Chance of a rare legendary merchant appearing (0.0–1.0).
const RARE_MERCHANT_CHANCE: f32 = 0.05;

/// Reputation threshold for return visits with better items.
const RETURN_VISIT_THRESHOLD: f32 = 30.0;

/// Reputation gained per purchase.
const REPUTATION_PER_PURCHASE: f32 = 5.0;

/// Check for merchant arrivals, departures, and inventory generation.
pub fn tick_traveling_merchants(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % MERCHANT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Depart merchants whose stay has expired ---
    let mut departed = Vec::new();
    state.traveling_merchants.retain(|m| {
        if state.tick >= m.departure_tick {
            departed.push((m.id, m.name.clone()));
            false
        } else {
            true
        }
    });
    for (id, name) in departed {
        events.push(WorldEvent::MerchantDeparted {
            merchant_id: id,
            name,
        });
    }

    // --- Roll for new arrivals (1-2 merchants per season change boundary) ---
    let num_arrivals = 1 + (lcg_next(&mut state.rng) % 2) as usize; // 1 or 2

    for _ in 0..num_arrivals {
        let merchant_id = state.next_event_id;
        state.next_event_id += 1;

        // Check for rare legendary merchant
        let is_rare = lcg_f32(&mut state.rng) < RARE_MERCHANT_CHANCE;

        // Check if this is a return visit from a previous merchant
        let is_return_visit = lcg_f32(&mut state.rng) < 0.3; // 30% chance of return visit concept

        let specialty = if is_rare {
            MerchantSpecialty::ExoticGoods
        } else {
            pick_specialty(&mut state.rng)
        };

        let name = generate_merchant_name(&mut state.rng, &specialty);

        // Base reputation starts at 0 for new merchants, higher for returns
        let reputation = if is_return_visit { 35.0 } else { 0.0 };

        // Generate inventory based on specialty, guild reputation, and return status
        let guild_rep = state.guild.reputation;
        let inventory = generate_inventory(
            &mut state.rng,
            &specialty,
            guild_rep,
            reputation,
            is_rare,
        );

        let merchant = TravelingMerchant {
            id: merchant_id,
            name: name.clone(),
            specialty: specialty.clone(),
            inventory,
            arrival_tick: state.tick,
            departure_tick: state.tick + MERCHANT_STAY_DURATION,
            visited: false,
            reputation_with_guild: reputation,
        };

        state.traveling_merchants.push(merchant);

        if is_rare {
            events.push(WorldEvent::RareMerchantSpotted {
                merchant_id,
                name: name.clone(),
                specialty: format!("{:?}", specialty),
            });
        }

        events.push(WorldEvent::MerchantArrived {
            merchant_id,
            name,
            specialty: format!("{:?}", specialty),
            num_items: state
                .traveling_merchants
                .last()
                .map(|m| m.inventory.len())
                .unwrap_or(0),
        });
    }
}

/// Process a purchase from a traveling merchant.
/// Returns (success_message, cost) or error string.
pub fn buy_from_merchant(
    state: &mut CampaignState,
    merchant_id: u32,
    item_index: usize,
    events: &mut Vec<WorldEvent>,
) -> Result<(String, f32), String> {
    // Find merchant and validate
    let merchant_idx = state
        .traveling_merchants
        .iter()
        .position(|m| m.id == merchant_id)
        .ok_or_else(|| format!("Merchant {} not found", merchant_id))?;

    if item_index >= state.traveling_merchants[merchant_idx].inventory.len() {
        return Err(format!("Item index {} out of range", item_index));
    }

    // Apply guild reputation discount: Legendary guild (rep >= 80) gets 15% off
    let discount = if state.guild.reputation >= 80.0 {
        0.85
    } else if state.guild.reputation >= 60.0 {
        0.92
    } else {
        1.0
    };

    let item = &state.traveling_merchants[merchant_idx].inventory[item_index];
    let price = item.price * discount;

    if state.guild.gold < price {
        return Err("Not enough gold".into());
    }

    let item_name = item.name.clone();
    let item_type = item.item_type.clone();
    let quality = item.quality;
    let special_effect = item.special_effect.clone();

    // Deduct gold
    state.guild.gold = (state.guild.gold - price).max(0.0);

    // Remove item from merchant inventory
    state.traveling_merchants[merchant_idx]
        .inventory
        .remove(item_index);

    // Mark as visited and increase reputation
    state.traveling_merchants[merchant_idx].visited = true;
    state.traveling_merchants[merchant_idx].reputation_with_guild += REPUTATION_PER_PURCHASE;

    // Add item to guild inventory
    let inv_item_id = state.next_event_id;
    state.next_event_id += 1;

    let slot = match item_type.as_str() {
        "weapon" => EquipmentSlot::Weapon,
        "armor" | "chest" => EquipmentSlot::Chest,
        "boots" => EquipmentSlot::Boots,
        "shield" | "offhand" => EquipmentSlot::Offhand,
        _ => EquipmentSlot::Accessory,
    };

    let stat_bonuses = quality_to_stat_bonuses(quality, &item_type);

    state.guild.inventory.push(InventoryItem {
        id: inv_item_id,
        name: item_name.clone(),
        slot,
        quality,
        stat_bonuses,
        durability: 100.0,
        appraised: false,
    });

    events.push(WorldEvent::MerchantPurchase {
        merchant_id,
        item_name: item_name.clone(),
        price,
    });

    events.push(WorldEvent::GoldChanged {
        amount: -price,
        reason: format!("Purchased {} from merchant", item_name),
    });

    Ok((
        format!("Bought {} for {:.0} gold", item_name, price),
        price,
    ))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn pick_specialty(rng: &mut u64) -> MerchantSpecialty {
    match lcg_next(rng) % 6 {
        0 => MerchantSpecialty::WeaponSmith,
        1 => MerchantSpecialty::ArmorMerchant,
        2 => MerchantSpecialty::PotionBrewer,
        3 => MerchantSpecialty::ExoticGoods,
        4 => MerchantSpecialty::BookDealer,
        _ => MerchantSpecialty::AnimalTrader,
    }
}

fn generate_merchant_name(rng: &mut u64, specialty: &MerchantSpecialty) -> String {
    let first_names = [
        "Aldric", "Brenna", "Corvus", "Dalla", "Eldrin", "Faye",
        "Gareth", "Hilda", "Izar", "Jorik", "Katla", "Lysander",
    ];
    let titles = match specialty {
        MerchantSpecialty::WeaponSmith => &[
            "the Forgemaster", "Steel-Singer", "Blade Merchant",
        ][..],
        MerchantSpecialty::ArmorMerchant => &[
            "the Ironclad", "Shield-Bearer", "Armor Peddler",
        ],
        MerchantSpecialty::PotionBrewer => &[
            "the Alchemist", "Elixir Maker", "Herb Trader",
        ],
        MerchantSpecialty::ExoticGoods => &[
            "the Wanderer", "Far-Traveler", "Relic Hunter",
        ],
        MerchantSpecialty::BookDealer => &[
            "the Scholar", "Tome Keeper", "Lore Merchant",
        ],
        MerchantSpecialty::AnimalTrader => &[
            "the Beastmaster", "Creature Dealer", "Beast Wrangler",
        ],
    };

    let name_idx = (lcg_next(rng) as usize) % first_names.len();
    let title_idx = (lcg_next(rng) as usize) % titles.len();

    format!("{} {}", first_names[name_idx], titles[title_idx])
}

fn generate_inventory(
    rng: &mut u64,
    specialty: &MerchantSpecialty,
    guild_reputation: f32,
    merchant_reputation: f32,
    is_rare: bool,
) -> Vec<MerchantItem> {
    let base_count = if is_rare { 2 } else { 3 + (lcg_next(rng) % 3) as usize }; // 3-5 normal, 2 rare
    let quality_bonus = if merchant_reputation > RETURN_VISIT_THRESHOLD {
        15.0 // Return visitors bring better goods
    } else {
        0.0
    };

    let mut items = Vec::new();

    for _ in 0..base_count {
        let item = match specialty {
            MerchantSpecialty::WeaponSmith => generate_weapon_item(rng, guild_reputation, quality_bonus, is_rare),
            MerchantSpecialty::ArmorMerchant => generate_armor_item(rng, guild_reputation, quality_bonus, is_rare),
            MerchantSpecialty::PotionBrewer => generate_potion_item(rng, guild_reputation, quality_bonus, is_rare),
            MerchantSpecialty::ExoticGoods => generate_exotic_item(rng, guild_reputation, quality_bonus, is_rare),
            MerchantSpecialty::BookDealer => generate_book_item(rng, guild_reputation, quality_bonus, is_rare),
            MerchantSpecialty::AnimalTrader => generate_companion_item(rng, guild_reputation, quality_bonus, is_rare),
        };
        items.push(item);
    }

    items
}

fn base_quality(rng: &mut u64, guild_rep: f32, quality_bonus: f32, is_rare: bool) -> f32 {
    let base = 40.0 + (lcg_next(rng) % 31) as f32; // 40-70
    let rep_bonus = guild_rep * 0.2; // up to +20 at 100 rep
    let rare_bonus = if is_rare { 25.0 } else { 0.0 };
    (base + rep_bonus + quality_bonus + rare_bonus).min(100.0)
}

fn base_price(rng: &mut u64, quality: f32, is_rare: bool) -> f32 {
    let base = 20.0 + (lcg_next(rng) % 31) as f32; // 20-50
    let quality_mult = 1.0 + quality / 100.0; // 1.0-2.0
    let rare_mult = if is_rare { 2.5 } else { 1.0 };
    base * quality_mult * rare_mult
}

fn generate_weapon_item(rng: &mut u64, guild_rep: f32, quality_bonus: f32, is_rare: bool) -> MerchantItem {
    let weapons = if is_rare {
        &["Legendary Flamebrand", "Voidsteel Greatsword", "Astral Glaive"][..]
    } else {
        &["Fine Steel Sword", "War Axe", "Enchanted Spear", "Composite Bow", "Runic Dagger"]
    };
    let idx = (lcg_next(rng) as usize) % weapons.len();
    let quality = base_quality(rng, guild_rep, quality_bonus, is_rare);
    let price = base_price(rng, quality, is_rare);
    let effect = if is_rare {
        Some(format!("+{:.0}% critical strike chance", quality * 0.3))
    } else if quality > 70.0 {
        Some(format!("+{:.0} attack power", quality * 0.1))
    } else {
        None
    };

    MerchantItem {
        name: weapons[idx].to_string(),
        item_type: "weapon".to_string(),
        price,
        quality,
        special_effect: effect,
    }
}

fn generate_armor_item(rng: &mut u64, guild_rep: f32, quality_bonus: f32, is_rare: bool) -> MerchantItem {
    let armors = if is_rare {
        &["Dragonscale Plate", "Ethereal Ward", "Titan's Bulwark"][..]
    } else {
        &["Reinforced Chainmail", "Leather Cuirass", "Iron Greaves", "Padded Vest", "Steel Helm"]
    };
    let idx = (lcg_next(rng) as usize) % armors.len();
    let quality = base_quality(rng, guild_rep, quality_bonus, is_rare);
    let price = base_price(rng, quality, is_rare);
    let item_type = if armors[idx].contains("Greaves") || armors[idx].contains("Boots") {
        "boots"
    } else {
        "armor"
    };
    let effect = if is_rare {
        Some(format!("+{:.0}% damage reduction", quality * 0.2))
    } else {
        None
    };

    MerchantItem {
        name: armors[idx].to_string(),
        item_type: item_type.to_string(),
        price,
        quality,
        special_effect: effect,
    }
}

fn generate_potion_item(rng: &mut u64, guild_rep: f32, quality_bonus: f32, is_rare: bool) -> MerchantItem {
    let potions = if is_rare {
        &["Elixir of Immortality", "Philosopher's Draught", "Dragon's Blood Tonic"][..]
    } else {
        &["Health Potion", "Stamina Elixir", "Antidote Vial", "Focus Tonic", "Courage Brew"]
    };
    let idx = (lcg_next(rng) as usize) % potions.len();
    let quality = base_quality(rng, guild_rep, quality_bonus, is_rare);
    let price = base_price(rng, quality, is_rare) * 0.7; // Potions are cheaper
    let effect = if is_rare {
        Some("Fully restores all conditions".to_string())
    } else {
        match idx {
            0 => Some(format!("Restores {:.0} injury", quality * 0.5)),
            1 => Some(format!("Reduces {:.0} fatigue", quality * 0.4)),
            _ => None,
        }
    };

    MerchantItem {
        name: potions[idx].to_string(),
        item_type: "consumable".to_string(),
        price,
        quality,
        special_effect: effect,
    }
}

fn generate_exotic_item(rng: &mut u64, guild_rep: f32, quality_bonus: f32, is_rare: bool) -> MerchantItem {
    let exotics = if is_rare {
        &["Crown of the Forgotten King", "Void Shard Pendant", "Chronos Hourglass"][..]
    } else {
        &["Crystal Amulet", "Foreign Talisman", "Enchanted Ring", "Mystic Orb", "Ancient Relic"]
    };
    let idx = (lcg_next(rng) as usize) % exotics.len();
    let quality = base_quality(rng, guild_rep, quality_bonus, is_rare);
    let price = base_price(rng, quality, is_rare) * 1.5; // Exotics cost more
    let effect = if is_rare {
        Some(format!("+{:.0} to all stats", quality * 0.15))
    } else {
        Some(format!("+{:.0} ability power", quality * 0.1))
    };

    MerchantItem {
        name: exotics[idx].to_string(),
        item_type: "accessory".to_string(),
        price,
        quality,
        special_effect: effect,
    }
}

fn generate_book_item(rng: &mut u64, guild_rep: f32, quality_bonus: f32, is_rare: bool) -> MerchantItem {
    let books = if is_rare {
        &["Tome of Forbidden Arts", "Codex of the Ancients", "Grimoire of Transcendence"][..]
    } else {
        &["Tactics Manual", "Herbalism Guide", "Bestiary Volume", "History of the Realm", "Meditation Scroll"]
    };
    let idx = (lcg_next(rng) as usize) % books.len();
    let quality = base_quality(rng, guild_rep, quality_bonus, is_rare);
    let price = base_price(rng, quality, is_rare) * 0.8;
    let effect = if is_rare {
        Some("+50 XP to reader".to_string())
    } else {
        match idx {
            0 => Some("+20 XP to reader".to_string()),
            1 => Some("Unlocks herbal remedies".to_string()),
            _ => None,
        }
    };

    MerchantItem {
        name: books[idx].to_string(),
        item_type: "consumable".to_string(),
        price,
        quality,
        special_effect: effect,
    }
}

fn generate_companion_item(rng: &mut u64, guild_rep: f32, quality_bonus: f32, is_rare: bool) -> MerchantItem {
    let companions = if is_rare {
        &["War Griffin", "Shadow Panther", "Frost Drake"][..]
    } else {
        &["Pack Mule", "Scout Hawk", "Guard Dog", "Messenger Pigeon", "War Horse"]
    };
    let idx = (lcg_next(rng) as usize) % companions.len();
    let quality = base_quality(rng, guild_rep, quality_bonus, is_rare);
    let price = base_price(rng, quality, is_rare) * 1.2;
    let effect = if is_rare {
        Some(format!("+{:.0} party speed, +{:.0} combat strength", quality * 0.2, quality * 0.3))
    } else {
        match idx {
            0 => Some("+10 party supply capacity".to_string()),
            1 => Some("+5 scouting range".to_string()),
            4 => Some("+15% travel speed".to_string()),
            _ => None,
        }
    };

    MerchantItem {
        name: companions[idx].to_string(),
        item_type: "accessory".to_string(),
        price,
        quality,
        special_effect: effect,
    }
}

fn quality_to_stat_bonuses(quality: f32, item_type: &str) -> StatBonuses {
    let scale = quality / 100.0;
    match item_type {
        "weapon" => StatBonuses {
            attack_bonus: 5.0 * scale,
            speed_bonus: 2.0 * scale,
            ..Default::default()
        },
        "armor" | "chest" => StatBonuses {
            defense_bonus: 6.0 * scale,
            hp_bonus: 10.0 * scale,
            ..Default::default()
        },
        "boots" => StatBonuses {
            speed_bonus: 4.0 * scale,
            defense_bonus: 2.0 * scale,
            ..Default::default()
        },
        "shield" | "offhand" => StatBonuses {
            defense_bonus: 4.0 * scale,
            hp_bonus: 5.0 * scale,
            ..Default::default()
        },
        _ => StatBonuses {
            attack_bonus: 2.0 * scale,
            defense_bonus: 2.0 * scale,
            hp_bonus: 5.0 * scale,
            speed_bonus: 1.0 * scale,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::state::CampaignState;

    #[test]
    fn merchant_arrives_and_departs() {
        let mut state = CampaignState::default_test_campaign(42);
        state.tick = 499; // Will become 500 on next step
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        // Advance to tick 500
        state.tick = 500;
        tick_traveling_merchants(&mut state, &mut deltas, &mut events);

        assert!(
            !state.traveling_merchants.is_empty(),
            "Should have merchants after tick 500"
        );
        let arrival_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, WorldEvent::MerchantArrived { .. }))
            .collect();
        assert!(
            !arrival_events.is_empty(),
            "Should emit MerchantArrived events"
        );

        // Fast-forward to departure
        let depart_tick = state.traveling_merchants[0].departure_tick;
        state.tick = depart_tick;
        events.clear();
        tick_traveling_merchants(&mut state, &mut deltas, &mut events);

        let departure_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, WorldEvent::MerchantDeparted { .. }))
            .collect();
        assert!(
            !departure_events.is_empty(),
            "Should emit MerchantDeparted events"
        );
    }

    #[test]
    fn buy_from_merchant_works() {
        let mut state = CampaignState::default_test_campaign(42);
        state.guild.gold = 500.0;

        // Manually add a merchant with inventory
        state.traveling_merchants.push(TravelingMerchant {
            id: 1,
            name: "Test Merchant".to_string(),
            specialty: MerchantSpecialty::WeaponSmith,
            inventory: vec![MerchantItem {
                name: "Test Sword".to_string(),
                item_type: "weapon".to_string(),
                price: 50.0,
                quality: 70.0,
                special_effect: None,
            }],
            arrival_tick: 0,
            departure_tick: 1000,
            visited: false,
            reputation_with_guild: 0.0,
        });

        let mut events = Vec::new();
        let result = buy_from_merchant(&mut state, 1, 0, &mut events);
        assert!(result.is_ok());
        let (msg, cost) = result.unwrap();
        assert!(msg.contains("Test Sword"));
        assert!(cost > 0.0);
        assert!(state.guild.gold < 500.0);
        assert!(!state.guild.inventory.is_empty());
        assert!(state.traveling_merchants[0].visited);
        assert!(state.traveling_merchants[0].reputation_with_guild > 0.0);
    }

    #[test]
    fn buy_fails_with_insufficient_gold() {
        let mut state = CampaignState::default_test_campaign(42);
        state.guild.gold = 5.0;

        state.traveling_merchants.push(TravelingMerchant {
            id: 1,
            name: "Test Merchant".to_string(),
            specialty: MerchantSpecialty::WeaponSmith,
            inventory: vec![MerchantItem {
                name: "Expensive Sword".to_string(),
                item_type: "weapon".to_string(),
                price: 100.0,
                quality: 90.0,
                special_effect: None,
            }],
            arrival_tick: 0,
            departure_tick: 1000,
            visited: false,
            reputation_with_guild: 0.0,
        });

        let mut events = Vec::new();
        let result = buy_from_merchant(&mut state, 1, 0, &mut events);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Not enough gold"));
    }

    #[test]
    fn inventory_generation_deterministic() {
        let mut rng1 = 12345u64;
        let mut rng2 = 12345u64;
        let inv1 = generate_inventory(&mut rng1, &MerchantSpecialty::WeaponSmith, 50.0, 0.0, false);
        let inv2 = generate_inventory(&mut rng2, &MerchantSpecialty::WeaponSmith, 50.0, 0.0, false);
        assert_eq!(inv1.len(), inv2.len());
        for (a, b) in inv1.iter().zip(inv2.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.price, b.price);
            assert_eq!(a.quality, b.quality);
        }
    }
}
