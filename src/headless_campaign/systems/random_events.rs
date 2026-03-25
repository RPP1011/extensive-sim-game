//! Random world events — fires every 500 ticks (~50s game time).
//!
//! Unpredictable events create situations the player must adapt to,
//! improving BFS state space diversity. Each event modifies state directly
//! and emits a `WorldEvent::RandomEvent` for logging. Some events also
//! create `ChoiceEvent`s for player decisions.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often to roll for a random event (in ticks).
const EVENT_INTERVAL: u64 = 500;

/// Base probability of an event firing each roll.
const BASE_CHANCE: f32 = 0.15;

/// Roll for and apply a random world event every `EVENT_INTERVAL` ticks.
pub fn tick_random_events(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % EVENT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Roll to see if an event fires at all.
    let roll = lcg_f32(&mut state.rng);
    if roll > BASE_CHANCE {
        return;
    }

    // Build weighted event pool based on game state.
    let candidates = build_candidate_pool(state);
    if candidates.is_empty() {
        return;
    }

    // Compute total weight.
    let total_weight: f32 = candidates.iter().map(|(_, w)| w).sum();
    let pick = lcg_f32(&mut state.rng) * total_weight;
    let mut cumulative = 0.0;
    let mut chosen = &candidates[0].0;
    for (evt, w) in &candidates {
        cumulative += w;
        if pick < cumulative {
            chosen = evt;
            break;
        }
    }

    apply_event(chosen.clone(), state, events);
}

// ---------------------------------------------------------------------------
// Event types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum RandomEventKind {
    // Positive
    WanderingMerchant,
    TreasureDiscovery,
    RefugeeInflux,
    HarvestBounty,
    FactionGift,
    // Negative
    Plague,
    BanditRaid,
    Desertion,
    EquipmentBreakage,
    FamineScare,
    // Neutral/Strategic
    RumorOfDungeon,
    FactionFeud,
    AncientRuinDiscovered,
    MercenaryBand,
    ProphecyOfDoom,
}

// ---------------------------------------------------------------------------
// Candidate pool construction
// ---------------------------------------------------------------------------

/// Build a list of (event, weight) pairs filtered by game-state eligibility.
/// Positive events are weighted more early game, negative more late game.
fn build_candidate_pool(state: &CampaignState) -> Vec<(RandomEventKind, f32)> {
    let progress = state.overworld.campaign_progress.clamp(0.0, 1.0);
    // Early game: positive_mult ~1.5, late game: ~0.5
    let positive_mult = 1.5 - progress;
    // Early game: negative_mult ~0.5, late game: ~1.5
    let negative_mult = 0.5 + progress;

    let has_alive_adventurers = state
        .adventurers
        .iter()
        .any(|a| a.status != AdventurerStatus::Dead);

    let has_low_loyalty = state
        .adventurers
        .iter()
        .any(|a| a.status != AdventurerStatus::Dead && a.loyalty < 20.0);

    let has_equipped_items = state.adventurers.iter().any(|a| {
        a.status != AdventurerStatus::Dead
            && (a.equipment.weapon.is_some()
                || a.equipment.offhand.is_some()
                || a.equipment.chest.is_some()
                || a.equipment.boots.is_some()
                || a.equipment.accessory.is_some())
    });

    let has_friendly_faction = state
        .factions
        .iter()
        .any(|f| f.relationship_to_guild > 30.0);

    let has_multiple_factions = state.factions.len() >= 2;

    let mut pool = Vec::new();

    // --- Positive ---
    pool.push((RandomEventKind::WanderingMerchant, 10.0 * positive_mult));
    pool.push((RandomEventKind::TreasureDiscovery, 12.0 * positive_mult));
    pool.push((RandomEventKind::RefugeeInflux, 6.0 * positive_mult));
    pool.push((RandomEventKind::HarvestBounty, 10.0 * positive_mult));

    if has_friendly_faction {
        pool.push((RandomEventKind::FactionGift, 8.0 * positive_mult));
    }

    // --- Negative ---
    if has_alive_adventurers {
        pool.push((RandomEventKind::Plague, 8.0 * negative_mult));
    }
    pool.push((RandomEventKind::BanditRaid, 10.0 * negative_mult));

    if has_low_loyalty {
        pool.push((RandomEventKind::Desertion, 6.0 * negative_mult));
    }
    if has_equipped_items {
        pool.push((RandomEventKind::EquipmentBreakage, 7.0 * negative_mult));
    }
    pool.push((RandomEventKind::FamineScare, 8.0 * negative_mult));

    // --- Neutral/Strategic ---
    pool.push((RandomEventKind::RumorOfDungeon, 10.0));
    if has_multiple_factions {
        pool.push((RandomEventKind::FactionFeud, 7.0));
    }
    pool.push((RandomEventKind::AncientRuinDiscovered, 6.0));
    pool.push((RandomEventKind::MercenaryBand, 8.0));
    pool.push((RandomEventKind::ProphecyOfDoom, 7.0));

    pool
}

// ---------------------------------------------------------------------------
// Event application
// ---------------------------------------------------------------------------

fn apply_event(
    kind: RandomEventKind,
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
) {
    match kind {
        RandomEventKind::WanderingMerchant => apply_wandering_merchant(state, events),
        RandomEventKind::TreasureDiscovery => apply_treasure_discovery(state, events),
        RandomEventKind::RefugeeInflux => apply_refugee_influx(state, events),
        RandomEventKind::HarvestBounty => apply_harvest_bounty(state, events),
        RandomEventKind::FactionGift => apply_faction_gift(state, events),
        RandomEventKind::Plague => apply_plague(state, events),
        RandomEventKind::BanditRaid => apply_bandit_raid(state, events),
        RandomEventKind::Desertion => apply_desertion(state, events),
        RandomEventKind::EquipmentBreakage => apply_equipment_breakage(state, events),
        RandomEventKind::FamineScare => apply_famine_scare(state, events),
        RandomEventKind::RumorOfDungeon => apply_rumor_of_dungeon(state, events),
        RandomEventKind::FactionFeud => apply_faction_feud(state, events),
        RandomEventKind::AncientRuinDiscovered => apply_ancient_ruin(state, events),
        RandomEventKind::MercenaryBand => apply_mercenary_band(state, events),
        RandomEventKind::ProphecyOfDoom => apply_prophecy_of_doom(state, events),
    }
}

// ---------------------------------------------------------------------------
// Positive events
// ---------------------------------------------------------------------------

fn apply_wandering_merchant(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Offer a rare item at discount via ChoiceEvent.
    let item_roll = lcg_next(&mut state.rng) % 5;
    let (item_name, slot, quality, cost) = match item_roll {
        0 => ("Fine Steel Sword", EquipmentSlot::Weapon, 75.0, 30.0),
        1 => ("Reinforced Shield", EquipmentSlot::Offhand, 70.0, 25.0),
        2 => ("Enchanted Chainmail", EquipmentSlot::Chest, 80.0, 40.0),
        3 => ("Swiftfoot Boots", EquipmentSlot::Boots, 65.0, 20.0),
        _ => ("Lucky Charm", EquipmentSlot::Accessory, 60.0, 15.0),
    };

    let item_id = state.next_event_id;
    state.next_event_id += 1;

    let choice_id = state.next_event_id;
    state.next_event_id += 1;

    state.pending_choices.push(ChoiceEvent {
        id: choice_id,
        source: ChoiceSource::WorldEvent,
        prompt: format!(
            "A wandering merchant offers a {} (quality {:.0}) for {} gold.",
            item_name, quality, cost
        ),
        options: vec![
            ChoiceOption {
                label: format!("Buy {} (-{} gold)", item_name, cost),
                description: format!("Purchase the {} for a discounted price.", item_name),
                effects: vec![
                    ChoiceEffect::Gold(-cost),
                    ChoiceEffect::AddItem(InventoryItem {
                        id: item_id,
                        name: item_name.to_string(),
                        slot,
                        quality,
                        stat_bonuses: Default::default(),
                    durability: 100.0,
                    }),
                ],
            },
            ChoiceOption {
                label: "Decline".to_string(),
                description: "Let the merchant pass.".to_string(),
                effects: vec![ChoiceEffect::Narrative(
                    "The merchant tips his hat and moves on.".to_string(),
                )],
            },
        ],
        default_option: 1,
        deadline_ms: Some(state.elapsed_ms + 30_000),
        created_at_ms: state.elapsed_ms,
    });

    events.push(WorldEvent::RandomEvent {
        name: "Wandering Merchant".to_string(),
        description: format!(
            "A traveling merchant offers a {} at a discount.",
            item_name
        ),
    });
    events.push(WorldEvent::ChoicePresented {
        choice_id,
        prompt: format!("A merchant offers a {} for {} gold.", item_name, cost),
        num_options: 2,
    });
}

fn apply_treasure_discovery(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let amount = 50.0 + (lcg_next(&mut state.rng) % 151) as f32; // 50-200
    state.guild.gold += amount;

    events.push(WorldEvent::RandomEvent {
        name: "Treasure Discovery".to_string(),
        description: format!(
            "An adventurer stumbles upon a hidden gold cache! +{:.0} gold.",
            amount
        ),
    });
    events.push(WorldEvent::GoldChanged {
        amount,
        reason: "Treasure discovery".to_string(),
    });
}

fn apply_refugee_influx(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let alive_count = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .count();

    // Cap at a reasonable number.
    if alive_count >= 12 {
        // Downgrade to a small supply bonus instead.
        state.guild.supplies += 15.0;
        events.push(WorldEvent::RandomEvent {
            name: "Refugee Influx".to_string(),
            description: "Refugees arrive but the guild is full. They leave supplies as thanks. +15 supplies.".to_string(),
        });
        events.push(WorldEvent::SupplyChanged {
            amount: 15.0,
            reason: "Refugee gratitude".to_string(),
        });
        return;
    }

    let archetypes = ["ranger", "knight", "rogue", "cleric", "mage"];
    let idx = (lcg_next(&mut state.rng) as usize) % archetypes.len();
    let archetype = archetypes[idx];

    let names = [
        "Cadoc", "Elin", "Rowan", "Tarya", "Milo", "Brin", "Sela", "Kael",
    ];
    let name_idx = (lcg_next(&mut state.rng) as usize) % names.len();
    let name = format!("{} the Refugee", names[name_idx]);

    let id = state
        .adventurers
        .iter()
        .map(|a| a.id)
        .max()
        .unwrap_or(0)
        + 1;

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
        level: 1,
        xp: 0,
        stats: AdventurerStats {
            max_hp: hp,
            attack: atk,
            defense: def,
            speed: spd,
            ability_power: ap,
        },
        equipment: Equipment::default(),
        traits: vec!["refugee".to_string()],
        status: AdventurerStatus::Idle,
        loyalty: 40.0, // starts low — they're grateful but cautious
        stress: 30.0,
        fatigue: 20.0,
        injury: 0.0,
        resolve: 50.0,
        morale: 60.0,
        party_id: None,
        guild_relationship: 30.0,
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

            mood_state: crate::headless_campaign::state::MoodState::default(),

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
            behavior_ledger: BehaviorLedger::default(),
            classes: Vec::new(),
    };

    state.adventurers.push(adventurer);

    events.push(WorldEvent::RandomEvent {
        name: "Refugee Influx".to_string(),
        description: format!(
            "A refugee {} ({}) joins the guild, seeking shelter.",
            name, archetype
        ),
    });
}

fn apply_harvest_bounty(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let amount = 30.0 + (lcg_next(&mut state.rng) % 21) as f32; // 30-50
    state.guild.supplies += amount;

    events.push(WorldEvent::RandomEvent {
        name: "Harvest Bounty".to_string(),
        description: format!(
            "A bountiful harvest in the region provides extra supplies! +{:.0} supplies.",
            amount
        ),
    });
    events.push(WorldEvent::SupplyChanged {
        amount,
        reason: "Harvest bounty".to_string(),
    });
}

fn apply_faction_gift(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Find the friendliest faction.
    let best = state
        .factions
        .iter()
        .filter(|f| f.relationship_to_guild > 30.0)
        .max_by(|a, b| {
            a.relationship_to_guild
                .partial_cmp(&b.relationship_to_guild)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

    let (faction_name, faction_id) = match best {
        Some(f) => (f.name.clone(), f.id),
        None => return, // no friendly faction — skip
    };

    let gold_gift = 20.0 + (lcg_next(&mut state.rng) % 31) as f32; // 20-50
    let supply_gift = 10.0 + (lcg_next(&mut state.rng) % 21) as f32; // 10-30
    state.guild.gold += gold_gift;
    state.guild.supplies += supply_gift;

    events.push(WorldEvent::RandomEvent {
        name: "Faction Gift".to_string(),
        description: format!(
            "The {} sends a gift of {:.0} gold and {:.0} supplies as a sign of goodwill.",
            faction_name, gold_gift, supply_gift
        ),
    });
    events.push(WorldEvent::GoldChanged {
        amount: gold_gift,
        reason: format!("Gift from {}", faction_name),
    });
    events.push(WorldEvent::SupplyChanged {
        amount: supply_gift,
        reason: format!("Gift from {}", faction_name),
    });
    // Slightly boost the relationship.
    if let Some(f) = state.factions.iter_mut().find(|f| f.id == faction_id) {
        f.relationship_to_guild = (f.relationship_to_guild + 3.0).min(100.0);
    }
}

// ---------------------------------------------------------------------------
// Negative events
// ---------------------------------------------------------------------------

fn apply_plague(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let mut affected = 0u32;
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        adv.fatigue = (adv.fatigue + 20.0).min(100.0);
        adv.injury = (adv.injury + 10.0).min(100.0);
        affected += 1;
    }

    events.push(WorldEvent::RandomEvent {
        name: "Plague".to_string(),
        description: format!(
            "A plague sweeps through the guild! {} adventurers gain +20 fatigue and +10 injury.",
            affected
        ),
    });
}

fn apply_bandit_raid(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let gold_loss = 20.0 + (lcg_next(&mut state.rng) % 31) as f32; // 20-50
    let actual_loss = gold_loss.min(state.guild.gold);
    state.guild.gold = (state.guild.gold - actual_loss).max(0.0);

    // Increase unrest in a random region.
    if !state.overworld.regions.is_empty() {
        let region_idx =
            (lcg_next(&mut state.rng) as usize) % state.overworld.regions.len();
        state.overworld.regions[region_idx].unrest =
            (state.overworld.regions[region_idx].unrest + 15.0).min(100.0);
    }

    events.push(WorldEvent::RandomEvent {
        name: "Bandit Raid".to_string(),
        description: format!(
            "Bandits raid the guild! Lost {:.0} gold, regional unrest rises.",
            actual_loss
        ),
    });
    if actual_loss > 0.0 {
        events.push(WorldEvent::GoldChanged {
            amount: -actual_loss,
            reason: "Bandit raid".to_string(),
        });
    }
}

fn apply_desertion(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Find lowest-loyalty living adventurer with loyalty < 20.
    let deserter = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead && a.loyalty < 20.0)
        .min_by(|a, b| {
            a.loyalty
                .partial_cmp(&b.loyalty)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|a| (a.id, a.name.clone()));

    let (id, name) = match deserter {
        Some(d) => d,
        None => return, // no eligible deserter
    };

    // Mark as dead (deserted = permanently gone).
    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == id) {
        adv.status = AdventurerStatus::Dead;
    }

    events.push(WorldEvent::RandomEvent {
        name: "Desertion".to_string(),
        description: format!(
            "{} has deserted the guild due to low loyalty!",
            name
        ),
    });
    events.push(WorldEvent::AdventurerDeserted {
        adventurer_id: id,
        reason: "Random event: low loyalty desertion".to_string(),
    });
}

fn apply_equipment_breakage(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect all equipped item IDs across adventurers.
    let mut equipped_items: Vec<(u32, u32)> = Vec::new(); // (adventurer_id, item_id)
    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        for opt_id in [
            adv.equipment.weapon,
            adv.equipment.offhand,
            adv.equipment.chest,
            adv.equipment.boots,
            adv.equipment.accessory,
        ] {
            if let Some(item_id) = opt_id {
                equipped_items.push((adv.id, item_id));
            }
        }
    }

    if equipped_items.is_empty() {
        return;
    }

    let idx = (lcg_next(&mut state.rng) as usize) % equipped_items.len();
    let (_adv_id, item_id) = equipped_items[idx];

    // Find the item in inventory and reduce quality.
    let mut item_name = String::new();
    for item in &mut state.guild.inventory {
        if item.id == item_id {
            let reduction = 15.0 + (lcg_next(&mut state.rng) % 16) as f32; // 15-30
            item.quality = (item.quality - reduction).max(0.0);
            item_name = item.name.clone();
            break;
        }
    }

    if item_name.is_empty() {
        item_name = format!("item #{}", item_id);
    }

    events.push(WorldEvent::RandomEvent {
        name: "Equipment Breakage".to_string(),
        description: format!(
            "A piece of equipment ({}) has degraded from wear and tear!",
            item_name
        ),
    });
}

fn apply_famine_scare(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Drain supplies immediately — simulates the "doubled drain for 500 ticks"
    // as a lump cost since we don't track temporary modifiers.
    let drain = (state.guild.supplies * 0.3).min(50.0).max(10.0);
    state.guild.supplies = (state.guild.supplies - drain).max(0.0);

    // Also hit morale.
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        adv.morale = (adv.morale - 8.0).max(0.0);
    }

    events.push(WorldEvent::RandomEvent {
        name: "Famine Scare".to_string(),
        description: format!(
            "A famine scare causes panic! Lost {:.0} supplies, morale drops across the guild.",
            drain
        ),
    });
    events.push(WorldEvent::SupplyChanged {
        amount: -drain,
        reason: "Famine scare".to_string(),
    });
}

// ---------------------------------------------------------------------------
// Neutral / Strategic events
// ---------------------------------------------------------------------------

fn apply_rumor_of_dungeon(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Reveal an unscouted location (if any).
    let unscouted: Vec<usize> = state
        .overworld
        .locations
        .iter()
        .filter(|l| !l.scouted)
        .map(|l| l.id)
        .collect();

    if let Some(&loc_id) = unscouted.first() {
        if let Some(loc) = state
            .overworld
            .locations
            .iter_mut()
            .find(|l| l.id == loc_id)
        {
            loc.scouted = true;
            events.push(WorldEvent::RandomEvent {
                name: "Rumor of Dungeon".to_string(),
                description: format!(
                    "Rumors reveal the location of {} — a high-reward opportunity!",
                    loc.name
                ),
            });
            events.push(WorldEvent::ScoutReport {
                location_id: loc_id,
                threat_level: loc.threat_level,
            });
            return;
        }
    }

    // All locations scouted — just flavor text.
    events.push(WorldEvent::RandomEvent {
        name: "Rumor of Dungeon".to_string(),
        description: "Adventurers hear rumors of treasure, but all known locations are already mapped.".to_string(),
    });
}

fn apply_faction_feud(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    if state.factions.len() < 2 {
        return;
    }

    let idx_a = (lcg_next(&mut state.rng) as usize) % state.factions.len();
    let mut idx_b = (lcg_next(&mut state.rng) as usize) % state.factions.len();
    if idx_b == idx_a {
        idx_b = (idx_a + 1) % state.factions.len();
    }

    let id_a = state.factions[idx_a].id;
    let id_b = state.factions[idx_b].id;
    let name_a = state.factions[idx_a].name.clone();
    let name_b = state.factions[idx_b].name.clone();

    // Drop faction-to-faction relation.
    if id_a < state.diplomacy.relations.len()
        && id_b < state.diplomacy.relations[id_a].len()
    {
        state.diplomacy.relations[id_a][id_b] =
            (state.diplomacy.relations[id_a][id_b] - 20).max(-100);
        state.diplomacy.relations[id_b][id_a] =
            (state.diplomacy.relations[id_b][id_a] - 20).max(-100);
    }

    events.push(WorldEvent::RandomEvent {
        name: "Faction Feud".to_string(),
        description: format!(
            "Tensions rise between {} and {}! Their relations deteriorate.",
            name_a, name_b
        ),
    });
}

fn apply_ancient_ruin(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let loc_id = state.overworld.locations.len();
    let x = (lcg_next(&mut state.rng) % 100) as f32;
    let y = (lcg_next(&mut state.rng) % 100) as f32;
    let threat = 40.0 + (lcg_next(&mut state.rng) % 41) as f32; // 40-80

    let ruin_names = [
        "Sunken Citadel",
        "Forgotten Crypt",
        "Shattered Observatory",
        "Cursed Hollow",
        "Titan's Remnant",
        "Voidtouched Spire",
    ];
    let name_idx = (lcg_next(&mut state.rng) as usize) % ruin_names.len();
    let name = ruin_names[name_idx].to_string();

    state.overworld.locations.push(Location {
        id: loc_id,
        name: name.clone(),
        position: (x, y),
        location_type: LocationType::Ruin,
        threat_level: threat,
        resource_availability: 60.0 + (lcg_next(&mut state.rng) % 31) as f32,
        faction_owner: None,
        scouted: true,
    });

    events.push(WorldEvent::RandomEvent {
        name: "Ancient Ruin Discovered".to_string(),
        description: format!(
            "Explorers discover {} at ({:.0}, {:.0}), threat level {:.0}!",
            name, x, y, threat
        ),
    });
}

fn apply_mercenary_band(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let count = 2 + (lcg_next(&mut state.rng) % 2) as u32; // 2-3
    let cost_each = 40.0 + (lcg_next(&mut state.rng) % 21) as f32; // 40-60
    let total_cost = cost_each * count as f32;

    let choice_id = state.next_event_id;
    state.next_event_id += 1;

    // Create a choice: hire mercenaries or decline.
    let mut hire_effects = vec![ChoiceEffect::Gold(-total_cost)];
    // Add reputation for taking in mercenaries.
    hire_effects.push(ChoiceEffect::Reputation(5.0));

    state.pending_choices.push(ChoiceEvent {
        id: choice_id,
        source: ChoiceSource::WorldEvent,
        prompt: format!(
            "A band of {} mercenaries offers their services for {:.0} gold each ({:.0} total).",
            count, cost_each, total_cost
        ),
        options: vec![
            ChoiceOption {
                label: format!("Hire {} mercenaries (-{:.0} gold)", count, total_cost),
                description: "Bolster your forces with experienced fighters.".to_string(),
                effects: hire_effects,
            },
            ChoiceOption {
                label: "Decline".to_string(),
                description: "The mercenaries move on to find other work.".to_string(),
                effects: vec![ChoiceEffect::Narrative(
                    "The mercenaries depart without incident.".to_string(),
                )],
            },
        ],
        default_option: 1,
        deadline_ms: Some(state.elapsed_ms + 25_000),
        created_at_ms: state.elapsed_ms,
    });

    events.push(WorldEvent::RandomEvent {
        name: "Mercenary Band".to_string(),
        description: format!(
            "A band of {} mercenaries offers to join for {:.0} gold total.",
            count, total_cost
        ),
    });
    events.push(WorldEvent::ChoicePresented {
        choice_id,
        prompt: format!("{} mercenaries available for hire.", count),
        num_options: 2,
    });
}

fn apply_prophecy_of_doom(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Morale drops but threat awareness (resolve) increases.
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        adv.morale = (adv.morale - 12.0).max(0.0);
        adv.resolve = (adv.resolve + 10.0).min(100.0);
    }

    // Slightly increase global threat awareness.
    state.overworld.global_threat_level =
        (state.overworld.global_threat_level + 5.0).min(100.0);

    events.push(WorldEvent::RandomEvent {
        name: "Prophecy of Doom".to_string(),
        description: "A seer foretells disaster! Morale drops, but resolve hardens.".to_string(),
    });
}
