//! Auction house system — periodic item auctions where rival guilds and factions
//! compete for rare gear, driving up prices but offering unique items.
//!
//! Fires every 500 ticks. New auctions spawn every 2000 ticks with 2-4 items.
//! AI bidders (rival guild, factions) bid based on gold reserves and need.
//! Auctions last 500 ticks before resolution.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often the auction system ticks (in campaign ticks).
const AUCTION_TICK_INTERVAL: u64 = 500;

/// How often a new auction is created (in campaign ticks).
const NEW_AUCTION_INTERVAL: u64 = 2000;

/// How long an auction lasts before resolution (in campaign ticks).
const AUCTION_DURATION: u64 = 500;

/// Chance of an artifact-quality item per auction item slot.
const ARTIFACT_CHANCE: f32 = 0.10;

/// Tick the auction house system.
///
/// Every `AUCTION_TICK_INTERVAL` ticks:
/// - AI bidders place bids on active auctions
/// - Expired auctions are resolved (highest bidder wins)
/// - New auctions are created every `NEW_AUCTION_INTERVAL` ticks
pub fn tick_auction(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % AUCTION_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Resolve expired auctions ---
    resolve_auctions(state, events);

    // --- AI bidding on active auctions ---
    ai_bidding(state, events);

    // --- Create new auction every NEW_AUCTION_INTERVAL ticks ---
    if state.tick % NEW_AUCTION_INTERVAL == 0 {
        create_new_auction(state, events);
    }
}

// ---------------------------------------------------------------------------
// Auction creation
// ---------------------------------------------------------------------------

/// Special traits that can appear on artifact-quality items.
const SPECIAL_TRAITS: &[&str] = &[
    "Lifestealing",
    "Flameburst",
    "Frostward",
    "Swiftcast",
    "Ironwall",
    "Shadowmeld",
    "Thunderstrike",
    "Regeneration",
    "Voidtouch",
    "Soulbound",
];

/// Auction item name prefixes by quality tier.
const AUCTION_PREFIXES_HIGH: &[&str] = &[
    "Masterwork", "Blessed", "Ancient", "Enchanted", "Runescribed",
];
const AUCTION_PREFIXES_ARTIFACT: &[&str] = &[
    "Legendary", "Mythic", "Celestial", "Primordial", "Godforged",
];

const AUCTION_WEAPON_NAMES: &[&str] = &[
    "Greatsword", "Warhammer", "Longbow", "Battle Staff", "Twin Daggers",
];
const AUCTION_ARMOR_NAMES: &[&str] = &[
    "Plate Armor", "Warding Robe", "Scale Mail", "Shadow Leather",
];
const AUCTION_OFFHAND_NAMES: &[&str] = &[
    "Tower Shield", "Arcane Focus", "War Banner", "Crystal Orb",
];
const AUCTION_BOOTS_NAMES: &[&str] = &[
    "Winged Boots", "Iron Treads", "Phantom Steps", "War Sabatons",
];
const AUCTION_ACCESSORY_NAMES: &[&str] = &[
    "Signet Ring", "War Cloak", "Crown Fragment", "Dragon Amulet",
];

fn pick_from(list: &[&str], rng: &mut u64) -> String {
    let idx = lcg_next(rng) as usize % list.len();
    list[idx].to_string()
}

fn generate_auction_item(rng: &mut u64, base_quality: f32) -> AuctionItem {
    // Slot selection
    let slot_roll = lcg_next(rng) % 5;
    let (slot_name, item_names) = match slot_roll {
        0 => ("Weapon", AUCTION_WEAPON_NAMES),
        1 => ("Offhand", AUCTION_OFFHAND_NAMES),
        2 => ("Chest", AUCTION_ARMOR_NAMES),
        3 => ("Boots", AUCTION_BOOTS_NAMES),
        _ => ("Accessory", AUCTION_ACCESSORY_NAMES),
    };

    let base_name = pick_from(item_names, rng);

    // Artifact check
    let is_artifact = lcg_f32(rng) < ARTIFACT_CHANCE;

    let (quality, prefix, special_trait) = if is_artifact {
        let q = (base_quality + 0.3).clamp(0.85, 1.0);
        let p = pick_from(AUCTION_PREFIXES_ARTIFACT, rng);
        let trait_name = pick_from(SPECIAL_TRAITS, rng);
        (q, p, Some(trait_name))
    } else {
        let q = (base_quality + lcg_f32(rng) * 0.15).clamp(0.5, 0.95);
        let p = pick_from(AUCTION_PREFIXES_HIGH, rng);
        (q, p, None)
    };

    let name = format!("{} {}", prefix, base_name);

    // Stat bonus scales with quality
    let stat_bonus = quality * 20.0 + if special_trait.is_some() { 10.0 } else { 0.0 };

    // Starting price based on quality
    let starting_price = 30.0 + quality * 120.0 + if special_trait.is_some() { 50.0 } else { 0.0 };

    // Generate a unique ID from rng
    let id = lcg_next(rng);

    AuctionItem {
        id,
        name,
        slot: slot_name.to_string(),
        quality,
        stat_bonus,
        special_trait,
        current_bid: starting_price,
        bidder: AuctionBidder::None,
        auction_end_tick: 0, // filled by caller
    }
}

fn create_new_auction(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Generate 2-4 items
    let item_count = 2 + (lcg_next(&mut state.rng) % 3) as usize; // 2, 3, or 4

    // Base quality scales with campaign progress
    let base_quality = 0.5 + state.overworld.campaign_progress * 0.3;

    let end_tick = state.tick + AUCTION_DURATION;
    let mut item_names = Vec::new();

    for _ in 0..item_count {
        let mut item = generate_auction_item(&mut state.rng, base_quality);
        item.auction_end_tick = end_tick;
        item_names.push(item.name.clone());
        state.auction_house.items.push(item);
    }

    state.auction_house.last_auction_tick = state.tick;

    events.push(WorldEvent::AuctionStarted {
        items: item_names,
    });
}

// ---------------------------------------------------------------------------
// AI bidding
// ---------------------------------------------------------------------------

fn ai_bidding(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect active auction item indices
    let active_indices: Vec<usize> = state
        .auction_house
        .items
        .iter()
        .enumerate()
        .filter(|(_, item)| item.auction_end_tick > state.tick)
        .map(|(i, _)| i)
        .collect();

    if active_indices.is_empty() {
        return;
    }

    // Rival guild bidding
    let rival_gold = state.rival_guild.gold;
    let rival_active = state.rival_guild.active;

    for &idx in &active_indices {
        // Rival guild bids if active and has gold
        if rival_active && rival_gold > 0.0 {
            let item = &state.auction_house.items[idx];
            // Rival bids if they can afford 10-20% above current bid
            let bid_increase = 1.0 + 0.10 + lcg_f32(&mut state.rng) * 0.10;
            let bid_amount = item.current_bid * bid_increase;

            // Only bid if not already the highest bidder and can afford it
            let is_rival_bidder = matches!(item.bidder, AuctionBidder::RivalGuild);
            if !is_rival_bidder && bid_amount <= rival_gold {
                // 40% chance the rival bids on any given item
                if lcg_f32(&mut state.rng) < 0.40 {
                    let item = &mut state.auction_house.items[idx];
                    item.current_bid = bid_amount;
                    item.bidder = AuctionBidder::RivalGuild;

                    events.push(WorldEvent::AuctionBidPlaced {
                        item: item.name.clone(),
                        amount: bid_amount,
                    });
                }
            }
        }

        // Faction bidding
        let faction_count = state.factions.len();
        for fid in 0..faction_count {
            let faction_strength = state.factions[fid].military_strength;
            // Factions bid based on military strength (proxy for resources)
            let faction_budget = faction_strength * 2.0;

            let item = &state.auction_house.items[idx];
            let is_faction_bidder = matches!(item.bidder, AuctionBidder::Faction(id) if id == fid);
            if is_faction_bidder {
                continue;
            }

            let bid_increase = 1.0 + 0.10 + lcg_f32(&mut state.rng) * 0.10;
            let bid_amount = item.current_bid * bid_increase;

            if bid_amount <= faction_budget {
                // 20% chance per faction per item
                if lcg_f32(&mut state.rng) < 0.20 {
                    let item = &mut state.auction_house.items[idx];
                    item.current_bid = bid_amount;
                    item.bidder = AuctionBidder::Faction(fid);

                    events.push(WorldEvent::AuctionBidPlaced {
                        item: item.name.clone(),
                        amount: bid_amount,
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Auction resolution
// ---------------------------------------------------------------------------

fn resolve_auctions(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;

    // Partition items into expired and active
    let mut expired = Vec::new();
    let mut remaining = Vec::new();

    for item in std::mem::take(&mut state.auction_house.items) {
        if item.auction_end_tick <= tick {
            expired.push(item);
        } else {
            remaining.push(item);
        }
    }

    state.auction_house.items = remaining;

    for item in expired {
        match item.bidder {
            AuctionBidder::Guild => {
                // Player won — deduct gold and add to inventory
                state.guild.gold = (state.guild.gold - item.current_bid).max(0.0);
                state.auction_house.total_spent += item.current_bid;
                state.auction_house.total_won += 1;

                // Convert to InventoryItem
                let slot = match item.slot.as_str() {
                    "Weapon" => EquipmentSlot::Weapon,
                    "Offhand" => EquipmentSlot::Offhand,
                    "Chest" => EquipmentSlot::Chest,
                    "Boots" => EquipmentSlot::Boots,
                    _ => EquipmentSlot::Accessory,
                };

                let inv_item = InventoryItem {
                    id: item.id,
                    name: item.name.clone(),
                    slot,
                    quality: item.quality,
                    stat_bonuses: auction_stat_bonuses(&item),
                    durability: 100.0,
                };

                state.guild.inventory.push(inv_item);

                events.push(WorldEvent::AuctionWon {
                    item: item.name.clone(),
                    winner: "Guild".to_string(),
                    price: item.current_bid,
                });
                events.push(WorldEvent::GoldChanged {
                    amount: -item.current_bid,
                    reason: format!("Auction won: {}", item.name),
                });
            }
            AuctionBidder::RivalGuild => {
                // Rival won — deduct from rival gold
                state.rival_guild.gold = (state.rival_guild.gold - item.current_bid).max(0.0);
                // Rival gains power from the item
                state.rival_guild.power_level += item.stat_bonus * 0.5;

                events.push(WorldEvent::AuctionLost {
                    item: item.name.clone(),
                    winner: state.rival_guild.name.clone(),
                });
            }
            AuctionBidder::Faction(fid) => {
                // Faction won — boost their military strength slightly
                if let Some(faction) = state.factions.iter_mut().find(|f| f.id == fid) {
                    faction.military_strength += item.stat_bonus * 0.3;
                    let winner_name = faction.name.clone();
                    events.push(WorldEvent::AuctionLost {
                        item: item.name.clone(),
                        winner: winner_name,
                    });
                } else {
                    events.push(WorldEvent::AuctionLost {
                        item: item.name.clone(),
                        winner: format!("Faction {}", fid),
                    });
                }
            }
            AuctionBidder::None => {
                // No bidder — item goes unsold (no event needed)
            }
        }
    }
}

/// Compute stat bonuses for an auction item based on slot and quality.
fn auction_stat_bonuses(item: &AuctionItem) -> StatBonuses {
    let q = item.quality;
    let trait_mult = if item.special_trait.is_some() { 1.3 } else { 1.0 };

    match item.slot.as_str() {
        "Weapon" => StatBonuses {
            hp_bonus: 0.0,
            attack_bonus: q * 30.0 * trait_mult,
            defense_bonus: 0.0,
            speed_bonus: 0.0,
        },
        "Offhand" => StatBonuses {
            hp_bonus: q * 25.0 * trait_mult,
            attack_bonus: 0.0,
            defense_bonus: q * 30.0 * trait_mult,
            speed_bonus: 0.0,
        },
        "Chest" => StatBonuses {
            hp_bonus: q * 60.0 * trait_mult,
            attack_bonus: 0.0,
            defense_bonus: q * 30.0 * trait_mult,
            speed_bonus: 0.0,
        },
        "Boots" => StatBonuses {
            hp_bonus: 0.0,
            attack_bonus: 0.0,
            defense_bonus: q * 15.0 * trait_mult,
            speed_bonus: q * 15.0 * trait_mult,
        },
        _ => StatBonuses {
            // Accessory
            hp_bonus: q * 20.0 * trait_mult,
            attack_bonus: q * 15.0 * trait_mult,
            defense_bonus: q * 10.0 * trait_mult,
            speed_bonus: q * 5.0 * trait_mult,
        },
    }
}
