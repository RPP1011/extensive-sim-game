//! NPC Economic Agent system.
//!
//! Gives NPCs individual wealth, class-driven income, and a utility function
//! so they autonomously settle, form parties, and migrate. Settlement patterns
//! emerge as side effects of rational self-interest.
//!
//! The wilderness is lethal — safety is the primary driver of party formation
//! and settlement. Level is the dominant economic variable: a level-40 NPC and
//! a level-10 NPC are not in the same world.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{
    AdventurerStatus, CampaignState, LocationType,
    COMMODITY_FOOD, COMMODITY_IRON, COMMODITY_WOOD, COMMODITY_HERBS,
    COMMODITY_HIDE, COMMODITY_CRYSTAL, COMMODITY_EQUIPMENT, COMMODITY_MEDICINE,
};
use super::class_system::effective_noncombat_stats;

// ---------------------------------------------------------------------------
// Core power functions
// ---------------------------------------------------------------------------

/// Compute the effective level of an NPC: base class level + resource bonus
/// from wealth and equipment. The resource bonus is capped at half the base
/// level so wealth augments capability but never eclipses it.
pub fn effective_level(adv: &crate::headless_campaign::state::Adventurer) -> f32 {
    let base = adv.level as f32;
    let resource_value = adv.gold.max(0.0) + equipment_value(adv);
    let threshold = adv
        .home_location_id
        .map(|_| {
            // Use the config threshold if available; fallback to 100.0.
            // In practice this is called from tick_npc_economy which has access
            // to config, but we keep a sane default here.
            100.0_f32
        })
        .unwrap_or(100.0);
    let raw_bonus = (resource_value / threshold).max(0.0).sqrt().floor();
    let cap = (base / 2.0).floor();
    base + raw_bonus.min(cap)
}

/// Compute effective level using an explicit resource threshold from config.
pub fn effective_level_with_config(
    adv: &crate::headless_campaign::state::Adventurer,
    resource_threshold: f32,
) -> f32 {
    let base = adv.level as f32;
    let resource_value = adv.gold.max(0.0) + equipment_value(adv);
    let raw_bonus = (resource_value / resource_threshold).max(0.0).sqrt().floor();
    let cap = (base / 2.0).floor();
    base + raw_bonus.min(cap)
}

/// Power rating: effective_level². This is the fundamental economic unit.
/// A level-40 NPC has power_rating 1600; a level-10 has 100 (16× difference).
pub fn power_rating(effective_level: f32) -> f32 {
    effective_level * effective_level
}

/// Estimate gold value of equipped items. Simple sum of item tiers.
fn equipment_value(adv: &crate::headless_campaign::state::Adventurer) -> f32 {
    // Each equipped item contributes roughly its tier × 50 gold equivalent.
    // This is deliberately simple — exact item economics come later.
    let item_count = adv.equipped_items.len() as f32;
    // Average item contributes ~50 gold equivalent per tier.
    // Without detailed tier data, use item count × 25 as a baseline.
    item_count * 25.0
}

/// Compute the combat-relevant effective level for an NPC (only meaningful
/// if they have combat classes). Uses attack + defense + ability_power as proxy.
pub fn combat_effective_level(adv: &crate::headless_campaign::state::Adventurer) -> f32 {
    let has_combat_stats = adv.stats.attack > 5.0 || adv.stats.defense > 5.0;
    if has_combat_stats {
        effective_level(adv)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Phase 6: Counterleveling — adversity accelerates class growth
// ---------------------------------------------------------------------------

/// Compute the adversity multiplier for an NPC's behavior ledger entries.
/// Uses logarithmic scaling: `1.0 + ln(1.0 + raw_adversity)`, capped at ~2.0x.
/// This is called by the class system's XP processing to accelerate growth
/// during hard times.
pub fn adversity_multiplier(
    adv: &crate::headless_campaign::state::Adventurer,
    location_threat: f32,
    demand: &[f32; 8],
) -> f32 {
    let stress_bonus = (adv.stress / 100.0) * 0.5;
    let scarcity_bonus = demand.iter().cloned().fold(0.0_f32, f32::max) * 0.3;
    let danger_bonus = (location_threat / 100.0) * 0.4;
    // Loss bonus: approximate from recent wounds and fears.
    let loss_bonus = (adv.fears.len() as f32) * 0.2
        + (adv.wounds.len() as f32) * 0.1;

    let raw = stress_bonus + scarcity_bonus + danger_bonus + loss_bonus;
    1.0 + (1.0 + raw).ln() - 1.0_f32.ln() // = 1.0 + ln(1 + raw)
}

/// Increment combat behavior for all residents of a settlement that is under
/// attack. Called when a battle occurs at a settlement location.
pub fn settlement_defense_behavior(
    state: &mut CampaignState,
    location_id: usize,
    threat_severity: f32,
) {
    let resident_ids: Vec<u32> = state
        .overworld
        .locations
        .iter()
        .find(|l| l.id == location_id)
        .map(|l| l.resident_ids.clone())
        .unwrap_or_default();

    let behavior_amount = (threat_severity / 10.0).max(1.0);
    for adv_id in &resident_ids {
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *adv_id) {
            if adv.status == AdventurerStatus::Dead {
                continue;
            }
            // All residents get combat behavior from witnessing/participating in defense.
            adv.behavior_ledger.melee_combat += behavior_amount * 0.5;
            adv.behavior_ledger.damage_absorbed += behavior_amount * 0.3;
            adv.behavior_ledger.allies_supported += behavior_amount * 0.2;
            // Recent window too.
            adv.behavior_ledger.recent_melee_combat += behavior_amount * 0.5;
            adv.behavior_ledger.recent_damage_absorbed += behavior_amount * 0.3;
            adv.behavior_ledger.recent_allies_supported += behavior_amount * 0.2;
        }
    }
}

/// Apply frontier ambient exposure: settlements near high-threat regions get
/// slow combat behavior ticks for residents. Called every decision interval.
pub fn frontier_exposure(state: &mut CampaignState) {
    for loc in &state.overworld.locations {
        if loc.threat_level < 20.0 {
            continue; // Only meaningful threat generates exposure.
        }
        let exposure = loc.threat_level / 500.0; // Very slow — 0.04 at threat 20, 0.2 at threat 100.
        let resident_ids = loc.resident_ids.clone();
        for adv_id in &resident_ids {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *adv_id) {
                if adv.status == AdventurerStatus::Dead {
                    continue;
                }
                adv.behavior_ledger.melee_combat += exposure * 0.3;
                adv.behavior_ledger.damage_absorbed += exposure * 0.2;
                adv.behavior_ledger.recent_melee_combat += exposure * 0.3;
                adv.behavior_ledger.recent_damage_absorbed += exposure * 0.2;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main tick entry point
// ---------------------------------------------------------------------------

/// Top-level NPC economy tick. Dispatches subsystems at their cadences.
pub fn tick_npc_economy(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    // One-time initialization: assign home locations on first tick.
    if state.tick == 1 {
        initialize_npc_locations(state);
    }

    let cfg = &state.config.npc_economy;
    let interval = cfg.decision_interval_ticks;

    // Every-tick subsystems — commodity loop
    tick_production(state);
    tick_consumption(state);
    tick_gold_flow(state);  // NPC-to-NPC commodity transactions
    tick_services(state);   // NPC-to-NPC service transactions (healing, repair)
    tick_local_prices(state);
    apply_monster_damage_to_settlements(state, events);
    apply_travel_danger(state, events);

    // Cadenced subsystems (every decision_interval_ticks)
    if state.tick % interval == 0 && state.tick > 0 {
        update_service_demand(state);
        update_location_safety(state);
        frontier_exposure(state);
        tick_information(state);
        tick_trade_decisions(state, events);
        tick_treasury_spending(state);
        check_bandit_transition(state, events);
        check_death_spiral_floor(state);
        run_npc_decisions(state, events);
        run_party_formation(state, events);
        manage_adventuring_parties(state, events);
        manage_patronage(state, events);
    }
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/// Assign NPCs to home settlements on first tick. Seed 2-3 NPCs per settlement
/// to avoid a massive initial diaspora from one location.
fn initialize_npc_locations(state: &mut CampaignState) {
    // Find all settlement locations.
    let settlement_ids: Vec<usize> = state
        .overworld
        .locations
        .iter()
        .filter(|l| l.location_type == crate::headless_campaign::state::LocationType::Settlement)
        .map(|l| l.id)
        .collect();

    if settlement_ids.is_empty() {
        return;
    }

    // Assign each alive adventurer without a home to a settlement.
    // Distribute round-robin across settlements for seeding.
    let mut slot = 0usize;
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        if adv.home_location_id.is_some() {
            continue;
        }
        let loc_id = settlement_ids[slot % settlement_ids.len()];
        adv.home_location_id = Some(loc_id);
        adv.economic_intent = crate::headless_campaign::state::EconomicIntent::Working;
        slot += 1;
    }

    // Populate resident_ids on each location from adventurers' home assignments.
    for loc in &mut state.overworld.locations {
        loc.resident_ids.clear();
    }
    let assignments: Vec<(u32, usize)> = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .filter_map(|a| a.home_location_id.map(|lid| (a.id, lid)))
        .collect();
    for (adv_id, loc_id) in assignments {
        if let Some(loc) = state.overworld.locations.iter_mut().find(|l| l.id == loc_id) {
            loc.resident_ids.push(adv_id);
        }
    }
}

// ---------------------------------------------------------------------------
// Subsystem: Service demand
// ---------------------------------------------------------------------------

/// Recompute demand per service channel at each settlement location.
/// Uses hyperbolic saturation: demand = base / (base + supply × saturation_rate).
fn update_service_demand(state: &mut CampaignState) {
    let cfg = state.config.npc_economy.clone();

    for loc in &mut state.overworld.locations {
        // Find the region this location belongs to (approximate by faction).
        let region_pop = state
            .overworld
            .regions
            .iter()
            .find(|r| Some(r.owner_faction_id) == loc.faction_owner.map(|f| f))
            .map(|r| r.population as f32)
            .unwrap_or(100.0);

        let region_threat = loc.threat_level;
        let resource_richness = loc.resource_availability / 100.0;

        // Compute supply per channel from residents.
        let mut supply = [0.0_f32; 8];
        for &adv_id in &loc.resident_ids {
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == adv_id) {
                if adv.status == AdventurerStatus::Dead {
                    continue;
                }
                let stats = effective_noncombat_stats(adv);
                let lvl_mult = effective_level_with_config(adv, cfg.resource_level_threshold) / 10.0;
                supply[0] += stats.0 * lvl_mult; // diplomacy
                supply[1] += stats.1 * lvl_mult; // commerce
                supply[2] += stats.2 * lvl_mult; // crafting
                supply[3] += stats.3 * lvl_mult; // medicine
                supply[4] += stats.4 * lvl_mult; // scholarship
                supply[5] += stats.5 * lvl_mult; // stealth
                supply[6] += stats.6 * lvl_mult; // leadership
                // combat: use attack+defense as proxy
                let combat_stat = (adv.stats.attack + adv.stats.defense + adv.stats.ability_power) / 30.0;
                supply[7] += combat_stat * lvl_mult;
            }
        }

        // Compute base demand per channel.
        let pop_base = region_pop * cfg.population_to_base_demand;
        let mut base = [pop_base; 8];

        // Channel-specific base demand adjustments.
        base[1] += resource_richness * 0.3; // commerce from resource richness
        base[2] += resource_richness * 0.2; // crafting from resource richness
        base[3] += cfg.base_medicine_demand;
        base[7] += cfg.base_combat_demand + region_threat * cfg.threat_to_combat_demand;

        // Hyperbolic saturation: demand = base / (base + supply × saturation_rate)
        let sat = cfg.demand_saturation_rate;
        for i in 0..8 {
            let b = base[i].max(0.001); // avoid division by zero
            loc.service_demand[i] = b / (b + supply[i] * sat);
        }
    }
}

// ---------------------------------------------------------------------------
// Subsystem: Location safety
// ---------------------------------------------------------------------------

/// Recompute safety_level and min_viable_threat per settlement.
/// Safety ceiling is defined by the strongest combat-class resident.
fn update_location_safety(state: &mut CampaignState) {
    let cfg = state.config.npc_economy.clone();

    for loc in &mut state.overworld.locations {
        let mut max_eff_level = 0.0_f32;
        for &adv_id in &loc.resident_ids {
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == adv_id) {
                if adv.status == AdventurerStatus::Dead {
                    continue;
                }
                let cel = combat_effective_level(adv);
                if cel > max_eff_level {
                    max_eff_level = cel;
                }
            }
        }
        loc.safety_level = max_eff_level;
        loc.min_viable_threat = if max_eff_level > 0.0 {
            max_eff_level.powf(1.5) / cfg.threat_scale
        } else {
            0.0
        };
    }
}

// ---------------------------------------------------------------------------
// Subsystem: NPC income
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Commodity: Production
// ---------------------------------------------------------------------------

/// Map NPC class tags to what commodity they produce. Returns (commodity_index, base_rate).
/// An NPC can produce multiple commodities if they have tags spanning clusters.
fn tags_to_production(adv: &crate::headless_campaign::state::Adventurer) -> Vec<(usize, f32)> {
    let mut production = Vec::new();

    // Collect all tags from all classes.
    let mut all_tags: Vec<&str> = Vec::new();
    for class in &adv.classes {
        // Use class name heuristics since tags aren't stored on ClassInstance.
        // Map class names to production commodities.
        let name = class.class_name.to_lowercase();
        if name.contains("farmer") || name.contains("cook") || name.contains("innkeeper")
            || name.contains("herbalist") || name.contains("stablehand")
        {
            all_tags.push("agriculture");
        }
        if name.contains("miner") {
            all_tags.push("mining");
        }
        if name.contains("hunter") || name.contains("ranger") || name.contains("scout")
            || name.contains("beast") || name.contains("warden")
        {
            all_tags.push("hunting");
            all_tags.push("nature");
        }
        if name.contains("blacksmith") || name.contains("artisan") || name.contains("arcane_blacksmith")
            || name.contains("crafter") || name.contains("forged")
        {
            all_tags.push("crafting");
        }
        if name.contains("healer") || name.contains("doctor") || name.contains("plague")
            || name.contains("cleric") || name.contains("alchemist")
        {
            all_tags.push("medicine");
        }
        if name.contains("mage") || name.contains("sorcerer") || name.contains("scholar")
            || name.contains("enchanter") || name.contains("archmage") || name.contains("oracle")
        {
            all_tags.push("arcane");
        }
        if name.contains("merchant") || name.contains("diplomat") || name.contains("noble")
            || name.contains("trader") || name.contains("peddler")
        {
            all_tags.push("trade");
        }
        if name.contains("warrior") || name.contains("knight") || name.contains("guardian")
            || name.contains("militia") || name.contains("berserker") || name.contains("commander")
            || name.contains("paladin")
        {
            all_tags.push("combat");
        }
        if name.contains("builder") || name.contains("engineer") {
            all_tags.push("construction");
        }
        // Laborer is a general producer — contributes food + wood + basic crafting
        if name.contains("laborer") || name.contains("errand") {
            all_tags.push("labor");
            all_tags.push("crafting"); // basic repair and tool-making
        }
    }

    all_tags.sort();
    all_tags.dedup();

    // Map tags → commodities.
    if all_tags.contains(&"agriculture") || all_tags.contains(&"labor") {
        production.push((COMMODITY_FOOD, 1.0));
    }
    if all_tags.contains(&"mining") {
        production.push((COMMODITY_IRON, 0.3));
    }
    if all_tags.contains(&"construction") || all_tags.contains(&"labor") {
        production.push((COMMODITY_WOOD, 0.3));
    }
    if all_tags.contains(&"nature") {
        production.push((COMMODITY_HERBS, 0.3));
    }
    if all_tags.contains(&"hunting") {
        production.push((COMMODITY_HIDE, 0.3));
    }
    if all_tags.contains(&"arcane") {
        production.push((COMMODITY_CRYSTAL, 0.2));
    }
    // Processors: consume raw → produce processed
    if all_tags.contains(&"crafting") {
        production.push((COMMODITY_EQUIPMENT, 0.2)); // will also consume iron+wood
    }
    if all_tags.contains(&"medicine") {
        production.push((COMMODITY_MEDICINE, 0.3)); // will also consume herbs
    }

    // Everyone forages some food (survival baseline). NPCs with no specific production
    // tags get a higher rate since food is their main contribution.
    if !production.iter().any(|(c, _)| *c == COMMODITY_FOOD) {
        production.push((COMMODITY_FOOD, 0.3)); // baseline foraging for non-farmers
    }

    // Fallback: NPCs with absolutely no production tags still forage
    if production.is_empty() {
        production.push((COMMODITY_FOOD, 0.3));
    }

    production
}

/// Get terrain modifier for a commodity at a location.
/// All commodities producible everywhere at 0.2x minimum (survival foraging).
/// Terrain boosts matching commodities to 1.5-2.0x.
fn terrain_commodity_modifier(loc: &crate::headless_campaign::state::Location, commodity: usize) -> f32 {
    // Use location properties to infer terrain type.
    // High threat + low resource → harsh terrain (desert/mountain)
    // Low threat + high resource → fertile terrain (plains/forest)
    let resource = loc.resource_availability / 100.0; // 0.0-1.0
    let threat = loc.threat_level / 100.0;

    // Base survival rate for all commodities
    let base = 0.2;

    match commodity {
        COMMODITY_FOOD => {
            // Food production favors low-threat, high-resource areas (fertile plains)
            base + resource * 1.5 * (1.0 - threat * 0.5)
        }
        COMMODITY_IRON => {
            // Iron favors high-threat areas (mountains)
            base + threat * 1.5
        }
        COMMODITY_WOOD => {
            // Wood favors moderate resource areas (forests)
            base + resource * 1.0
        }
        COMMODITY_HERBS => {
            // Herbs favor high-resource areas (swamps, forests)
            base + resource * 1.2 * (1.0 - threat * 0.3)
        }
        COMMODITY_HIDE => {
            // Hide available where there's wildlife (moderate everything)
            base + resource * 0.8
        }
        COMMODITY_CRYSTAL => {
            // Crystal favors extreme terrain (high threat)
            base + threat * 1.2
        }
        _ => 1.0, // processed goods not terrain-dependent
    }
}

/// NPCs produce commodities into their settlement's stockpile.
fn tick_production(state: &mut CampaignState) {
    // Snapshot location data for terrain modifiers.
    struct LocInfo {
        id: usize,
        resource_availability: f32,
        threat_level: f32,
        location_type: LocationType,
    }
    let loc_infos: Vec<LocInfo> = state.overworld.locations.iter().map(|l| LocInfo {
        id: l.id,
        resource_availability: l.resource_availability,
        threat_level: l.threat_level,
        location_type: l.location_type,
    }).collect();

    // Collect production per location.
    let mut production_by_loc: std::collections::HashMap<usize, [f32; 8]> = std::collections::HashMap::new();

    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        let loc_id = match adv.home_location_id {
            Some(id) => id,
            None => continue,
        };
        if !matches!(adv.economic_intent, crate::headless_campaign::state::EconomicIntent::Working) {
            continue;
        }

        let loc_info = match loc_infos.iter().find(|l| l.id == loc_id) {
            Some(l) => l,
            None => continue,
        };

        // Only produce at settlements
        if loc_info.location_type != LocationType::Settlement {
            continue;
        }

        let productions = tags_to_production(adv);
        let level_mult = effective_level_with_config(adv, state.config.npc_economy.resource_level_threshold) / 10.0;

        let entry = production_by_loc.entry(loc_id).or_insert([0.0; 8]);

        for (commodity, base_rate) in productions {
            if commodity == COMMODITY_EQUIPMENT || commodity == COMMODITY_MEDICINE {
                // Processed goods: check if raw materials are available (will be consumed)
                // For now, produce at base rate — consumption of inputs handled separately
                entry[commodity] += base_rate * level_mult;
            } else {
                // Raw extraction: terrain-modified
                let terrain_mod = terrain_commodity_modifier(
                    // Reconstruct a minimal Location for the modifier
                    &crate::headless_campaign::state::Location {
                        id: loc_info.id,
                        name: String::new(),
                        position: (0.0, 0.0),
                        location_type: loc_info.location_type,
                        threat_level: loc_info.threat_level,
                        resource_availability: loc_info.resource_availability,
                        faction_owner: None,
                        scouted: false,
                        resident_ids: Vec::new(),
                        service_demand: [0.0; 8],
                        cost_of_living: 1.0,
                        safety_level: 0.0,
                        min_viable_threat: 0.0,
                        treasury: 0.0,
                        tax_rate: 0.15,
                        stockpile: Default::default(),
                        local_prices: crate::headless_campaign::state::BASE_PRICES,
                    },
                    commodity,
                );
                entry[commodity] += base_rate * level_mult * terrain_mod;
            }
        }
    }

    // Apply production to stockpiles.
    for (loc_id, amounts) in &production_by_loc {
        if let Some(loc) = state.overworld.locations.iter_mut().find(|l| l.id == *loc_id) {
            loc.stockpile.food += amounts[COMMODITY_FOOD];
            loc.stockpile.iron += amounts[COMMODITY_IRON];
            loc.stockpile.wood += amounts[COMMODITY_WOOD];
            loc.stockpile.herbs += amounts[COMMODITY_HERBS];
            loc.stockpile.hide += amounts[COMMODITY_HIDE];
            loc.stockpile.crystal += amounts[COMMODITY_CRYSTAL];

            // Processed goods consume raw materials
            if amounts[COMMODITY_EQUIPMENT] > 0.0 {
                let iron_needed = amounts[COMMODITY_EQUIPMENT] * 2.0;
                let wood_needed = amounts[COMMODITY_EQUIPMENT] * 1.0;
                let iron_avail = loc.stockpile.iron.min(iron_needed);
                let wood_avail = loc.stockpile.wood.min(wood_needed);
                let actual = (iron_avail / 2.0).min(wood_avail / 1.0).min(amounts[COMMODITY_EQUIPMENT]);
                if actual > 0.0 {
                    loc.stockpile.iron -= actual * 2.0;
                    loc.stockpile.wood -= actual * 1.0;
                    loc.stockpile.equipment += actual;
                }
            }
            if amounts[COMMODITY_MEDICINE] > 0.0 {
                let herbs_needed = amounts[COMMODITY_MEDICINE] * 2.0;
                let herbs_avail = loc.stockpile.herbs.min(herbs_needed);
                let actual = (herbs_avail / 2.0).min(amounts[COMMODITY_MEDICINE]);
                if actual > 0.0 {
                    loc.stockpile.herbs -= actual * 2.0;
                    loc.stockpile.medicine += actual;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Commodity: Consumption
// ---------------------------------------------------------------------------

/// Every NPC consumes food. Combat NPCs consume equipment. Injured NPCs consume medicine.
/// Deducts from settlement stockpile. Applies shortage effects.
fn tick_consumption(state: &mut CampaignState) {
    // Collect consumption per location.
    struct ConsumptionEntry {
        food_needed: f32,
        equipment_needed: f32,
        medicine_needed: f32,
    }
    let mut consumption_by_loc: std::collections::HashMap<usize, ConsumptionEntry> = std::collections::HashMap::new();

    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        let loc_id = match adv.home_location_id {
            Some(id) => id,
            None => continue,
        };

        let entry = consumption_by_loc.entry(loc_id).or_insert(ConsumptionEntry {
            food_needed: 0.0,
            equipment_needed: 0.0,
            medicine_needed: 0.0,
        });

        entry.food_needed += 0.2; // 1 meal per ~15 ticks (~45 seconds game time)
        if adv.stats.attack > 10.0 || adv.stats.defense > 10.0 {
            entry.equipment_needed += 0.05;
        }
        if adv.injury > 0.0 {
            entry.medicine_needed += 0.1;
        }
    }

    // Deduct from stockpiles and track shortages per location.
    let mut food_shortage_locs: Vec<usize> = Vec::new();
    let mut equip_shortage_locs: Vec<usize> = Vec::new();
    let mut med_shortage_locs: Vec<usize> = Vec::new();

    for (loc_id, needs) in &consumption_by_loc {
        if let Some(loc) = state.overworld.locations.iter_mut().find(|l| l.id == *loc_id) {
            // Food
            let food_consumed = loc.stockpile.food.min(needs.food_needed);
            loc.stockpile.food -= food_consumed;
            if food_consumed < needs.food_needed * 0.5 {
                food_shortage_locs.push(*loc_id);
            }

            // Equipment
            let equip_consumed = loc.stockpile.equipment.min(needs.equipment_needed);
            loc.stockpile.equipment -= equip_consumed;
            if equip_consumed < needs.equipment_needed * 0.5 {
                equip_shortage_locs.push(*loc_id);
            }

            // Medicine
            let med_consumed = loc.stockpile.medicine.min(needs.medicine_needed);
            loc.stockpile.medicine -= med_consumed;
            if med_consumed < needs.medicine_needed * 0.5 {
                med_shortage_locs.push(*loc_id);
            }
        }
    }

    // Apply shortage effects to NPCs.
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        let loc_id = match adv.home_location_id {
            Some(id) => id,
            None => continue,
        };

        let is_starving = food_shortage_locs.contains(&loc_id);
        if is_starving {
            adv.injury = (adv.injury + 2.0).min(100.0);
            adv.stress = (adv.stress + 3.0).min(100.0);
            adv.fatigue = (adv.fatigue + 1.0).min(100.0);

            // Death from starvation: when injury maxed out, chance per tick.
            if adv.injury >= 95.0 {
                let roll = crate::headless_campaign::state::lcg_f32(&mut state.rng);
                if roll < 0.02 { // 2% per tick when critically starving
                    adv.status = AdventurerStatus::Dead;
                }
            }
        }
        if equip_shortage_locs.contains(&loc_id) && (adv.stats.attack > 10.0 || adv.stats.defense > 10.0) {
            adv.stress = (adv.stress + 1.0).min(100.0);
        }
        // Healing only happens when NOT starving.
        if !is_starving && adv.injury > 0.0 {
            if med_shortage_locs.contains(&loc_id) {
                adv.injury = (adv.injury - 0.01).max(0.0); // slow heal without medicine
            } else {
                adv.injury = (adv.injury - 0.1).max(0.0); // normal heal with medicine
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Commodity: Local Prices
// ---------------------------------------------------------------------------

/// Recompute local prices at each settlement from stockpile/consumption ratios.
fn tick_local_prices(state: &mut CampaignState) {
    let price_halflife = 50.0_f32;

    for loc in &mut state.overworld.locations {
        if loc.location_type != LocationType::Settlement {
            continue;
        }
        let pop = loc.resident_ids.len().max(1) as f32;

        // Consumption rates for price computation
        let consumption = [
            pop * 1.0,    // food: everyone eats
            pop * 0.05,   // iron: indirect (blacksmiths consume)
            pop * 0.03,   // wood: indirect
            pop * 0.04,   // herbs: indirect (healers consume)
            pop * 0.02,   // hide: indirect
            pop * 0.01,   // crystal: indirect
            pop * 0.05,   // equipment: combat NPCs
            pop * 0.1,    // medicine: injured NPCs
        ];

        let stockpile_arr = [
            loc.stockpile.food,
            loc.stockpile.iron,
            loc.stockpile.wood,
            loc.stockpile.herbs,
            loc.stockpile.hide,
            loc.stockpile.crystal,
            loc.stockpile.equipment,
            loc.stockpile.medicine,
        ];

        for i in 0..8 {
            let ticks_of_supply = stockpile_arr[i] / consumption[i].max(0.01);
            loc.local_prices[i] = crate::headless_campaign::state::BASE_PRICES[i]
                / (1.0 + ticks_of_supply / price_halflife);
        }
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Commodity: Gold Flow (NPC-to-NPC transactions)
// ---------------------------------------------------------------------------

/// Zero-sum gold transfer: consumers pay for goods, that gold goes to producers.
/// Total gold in the system is conserved (minus settlement tax).
fn tick_gold_flow(state: &mut CampaignState) {
    let tax_rate = 0.05_f32;

    // Per-settlement: total consumer spending = total producer earnings.
    // We compute total spending, distribute to producers proportionally, tax a slice.

    struct NpcEcon {
        adv_idx: usize,
        loc_id: usize,
        spending: f32,          // what this NPC owes for food/equipment/medicine
        production_value: f32,  // value of what this NPC produced
    }

    let loc_prices: Vec<(usize, [f32; 8])> = state.overworld.locations.iter()
        .map(|l| (l.id, l.local_prices))
        .collect();

    let mut npc_econs: Vec<NpcEcon> = Vec::new();

    for (idx, adv) in state.adventurers.iter().enumerate() {
        if adv.status == AdventurerStatus::Dead { continue; }
        let loc_id = match adv.home_location_id {
            Some(id) => id,
            None => continue,
        };

        let prices = match loc_prices.iter().find(|(id, _)| *id == loc_id) {
            Some((_, p)) => p,
            None => continue,
        };

        // Spending: what this NPC consumes valued at local prices
        let food_cost = 0.2 * prices[COMMODITY_FOOD];
        let equip_cost = if adv.stats.attack > 10.0 || adv.stats.defense > 10.0 {
            0.05 * prices[COMMODITY_EQUIPMENT]
        } else { 0.0 };
        let med_cost = if adv.injury > 0.0 {
            0.1 * prices[COMMODITY_MEDICINE]
        } else { 0.0 };
        let spending = food_cost + equip_cost + med_cost;

        // Production value: what this NPC produced valued at local prices
        let mut production_value = 0.0_f32;
        if matches!(adv.economic_intent, crate::headless_campaign::state::EconomicIntent::Working) {
            let productions = tags_to_production(adv);
            let level_mult = effective_level_with_config(adv, state.config.npc_economy.resource_level_threshold) / 10.0;
            for (commodity, base_rate) in &productions {
                production_value += base_rate * level_mult * prices[*commodity];
            }
        }

        npc_econs.push(NpcEcon { adv_idx: idx, loc_id, spending, production_value });
    }

    // Per-settlement zero-sum distribution.
    // Total spending at a settlement = pool of gold available to producers.
    // Each producer gets a share proportional to their production_value.
    let mut loc_total_spending: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
    let mut loc_total_production: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();

    for e in &npc_econs {
        *loc_total_spending.entry(e.loc_id).or_insert(0.0) += e.spending;
        *loc_total_production.entry(e.loc_id).or_insert(0.0) += e.production_value;
    }

    // Two passes: first collect actual spending (capped by gold on hand),
    // then distribute to producers.

    // Pass 1: consumers pay what they can afford.
    let mut loc_actual_pool: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
    for e in &npc_econs {
        let adv = &state.adventurers[e.adv_idx];
        let can_pay = adv.gold.max(0.0);
        let actual_spend = e.spending.min(can_pay);
        *loc_actual_pool.entry(e.loc_id).or_insert(0.0) += actual_spend;
    }

    // Pass 2: deduct from consumers and distribute to producers.
    let mut treasury_gains: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
    for e in &npc_econs {
        let total_prod = *loc_total_production.get(&e.loc_id).unwrap_or(&0.0);
        let actual_pool = *loc_actual_pool.get(&e.loc_id).unwrap_or(&0.0);

        let adv = &mut state.adventurers[e.adv_idx];

        // Consumer: pay
        let can_pay = adv.gold.max(0.0);
        let actual_spend = e.spending.min(can_pay);
        adv.gold -= actual_spend;

        // Producer: earn proportional share of the actual gold pool (what consumers actually paid)
        let earned = if total_prod > 0.01 && actual_pool > 0.01 {
            let share = e.production_value / total_prod;
            share * actual_pool * (1.0 - tax_rate)
        } else {
            0.0
        };
        adv.gold += earned;
        adv.ticks_since_income = if earned > 0.01 { 0 } else { adv.ticks_since_income + 1 };

        // Tax
        if total_prod > 0.01 && actual_pool > 0.01 {
            let share = e.production_value / total_prod;
            *treasury_gains.entry(e.loc_id).or_insert(0.0) += share * actual_pool * tax_rate;
        }
    }

    for (loc_id, tax) in &treasury_gains {
        if let Some(loc) = state.overworld.locations.iter_mut().find(|l| l.id == *loc_id) {
            loc.treasury += tax;
        }
    }
}

// ---------------------------------------------------------------------------
// Information System
// ---------------------------------------------------------------------------

const INFO_STALE_TICKS: u32 = 200;

/// Propagate price information between NPCs.
/// 1. NPCs at a settlement learn current local prices (free).
/// 2. Scout/Navigator/Ranger NPCs become runners — they travel and share info.
/// 3. Local gossip: 5% of NPCs at same settlement share knowledge each cycle.
fn tick_information(state: &mut CampaignState) {
    use crate::headless_campaign::state::PriceReport;

    let tick = state.tick as u32;

    // Snapshot current local prices and stockpiles per settlement.
    struct LocSnapshot {
        id: usize,
        prices: [f32; 8],
        stockpiles: [f32; 8],
    }
    let snapshots: Vec<LocSnapshot> = state.overworld.locations.iter().map(|l| LocSnapshot {
        id: l.id,
        prices: l.local_prices,
        stockpiles: [
            l.stockpile.food, l.stockpile.iron, l.stockpile.wood, l.stockpile.herbs,
            l.stockpile.hide, l.stockpile.crystal, l.stockpile.equipment, l.stockpile.medicine,
        ],
    }).collect();

    // 1. Every NPC at a settlement learns current local prices there.
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead { continue; }
        let loc_id = match adv.home_location_id {
            Some(id) => id,
            None => continue,
        };
        if let Some(snap) = snapshots.iter().find(|s| s.id == loc_id) {
            // Update or insert price report for this location.
            if let Some(report) = adv.price_knowledge.iter_mut().find(|r| r.location_id == loc_id) {
                report.prices = snap.prices;
                report.stockpiles = snap.stockpiles;
                report.reported_tick = tick;
            } else {
                adv.price_knowledge.push(PriceReport {
                    location_id: loc_id,
                    prices: snap.prices,
                    stockpiles: snap.stockpiles,
                    reported_tick: tick,
                });
            }
        }
    }

    // 2. Local gossip: NPCs at the same settlement share foreign price knowledge.
    // For each settlement, 5% of NPCs exchange info with a random neighbor.
    let gossip_rate = 0.05;
    let settlement_ids: Vec<usize> = state.overworld.locations.iter()
        .filter(|l| l.location_type == LocationType::Settlement)
        .map(|l| l.id)
        .collect();

    for &loc_id in &settlement_ids {
        let resident_idxs: Vec<usize> = state.adventurers.iter().enumerate()
            .filter(|(_, a)| a.status != AdventurerStatus::Dead && a.home_location_id == Some(loc_id))
            .map(|(i, _)| i)
            .collect();

        if resident_idxs.len() < 2 { continue; }

        // Number of gossip exchanges this cycle
        let exchanges = ((resident_idxs.len() as f32) * gossip_rate).ceil() as usize;
        for _ in 0..exchanges {
            let idx_a = (crate::headless_campaign::state::lcg_next(&mut state.rng) as usize) % resident_idxs.len();
            let idx_b = (crate::headless_campaign::state::lcg_next(&mut state.rng) as usize) % resident_idxs.len();
            if idx_a == idx_b { continue; }

            let a_idx = resident_idxs[idx_a];
            let b_idx = resident_idxs[idx_b];

            // Collect reports from A that B doesn't have (or has staler).
            let a_reports: Vec<PriceReport> = state.adventurers[a_idx].price_knowledge.clone();
            let b_knowledge = &mut state.adventurers[b_idx].price_knowledge;

            for report in &a_reports {
                if let Some(existing) = b_knowledge.iter_mut().find(|r| r.location_id == report.location_id) {
                    if report.reported_tick > existing.reported_tick {
                        *existing = report.clone();
                    }
                } else {
                    b_knowledge.push(report.clone());
                }
            }

            // And vice versa
            let b_reports: Vec<PriceReport> = state.adventurers[b_idx].price_knowledge.clone();
            let a_knowledge = &mut state.adventurers[a_idx].price_knowledge;

            for report in &b_reports {
                if let Some(existing) = a_knowledge.iter_mut().find(|r| r.location_id == report.location_id) {
                    if report.reported_tick > existing.reported_tick {
                        *existing = report.clone();
                    }
                } else {
                    a_knowledge.push(report.clone());
                }
            }
        }
    }

    // 3. Prune stale reports (older than INFO_STALE_TICKS * 3 — keep them but they're unreliable)
    // We don't delete — we let the staleness_discount in trade decisions handle it.
    // But cap the knowledge list to prevent unbounded growth.
    for adv in &mut state.adventurers {
        // Keep at most 30 reports per NPC.
        if adv.price_knowledge.len() > 30 {
            adv.price_knowledge.sort_by(|a, b| b.reported_tick.cmp(&a.reported_tick));
            adv.price_knowledge.truncate(30);
        }
    }
}

// ---------------------------------------------------------------------------
// Trade Decisions (Merchant NPCs)
// ---------------------------------------------------------------------------

/// Merchant NPCs evaluate price differentials from their knowledge and initiate
/// trade runs. They buy at origin (deducting from stockpile, paying gold),
/// carry goods, and sell at destination on arrival.
fn tick_trade_decisions(state: &mut CampaignState, _events: &mut Vec<WorldEvent>) {
    use crate::headless_campaign::state::{EconomicIntent, PartyPurpose};

    let tick = state.tick as u32;

    // Find Merchant NPCs who are Working and have price knowledge of other settlements.
    struct TradeOpp {
        adv_idx: usize,
        dest_loc_id: usize,
        commodity: usize,
        expected_profit: f32,
        buy_amount: f32,
        buy_cost: f32,
    }

    let mut opportunities: Vec<TradeOpp> = Vec::new();

    for (idx, adv) in state.adventurers.iter().enumerate() {
        if adv.status == AdventurerStatus::Dead { continue; }
        if !matches!(adv.economic_intent, EconomicIntent::Working) { continue; }
        let loc_id = match adv.home_location_id {
            Some(id) => id,
            None => continue,
        };

        // Is this NPC a merchant? Check class names for trade-related classes.
        let is_merchant = adv.classes.iter().any(|c| {
            let name = c.class_name.to_lowercase();
            name.contains("merchant") || name.contains("peddler") || name.contains("trader")
                || name.contains("trade") || name.contains("gilded")
        });
        if !is_merchant { continue; }

        // Get local prices.
        let local_prices = state.overworld.locations.iter()
            .find(|l| l.id == loc_id)
            .map(|l| l.local_prices)
            .unwrap_or(crate::headless_campaign::state::BASE_PRICES);

        // Commerce stat determines carry capacity.
        let stats = effective_noncombat_stats(adv);
        let commerce = stats.1;
        let carry_capacity = 10.0 + adv.level as f32 + commerce * 2.0;

        // Evaluate each known settlement for trade opportunities.
        let mut best_profit = 0.0_f32;
        let mut best_opp: Option<TradeOpp> = None;

        for report in &adv.price_knowledge {
            if report.location_id == loc_id { continue; } // same settlement
            let staleness = (tick.saturating_sub(report.reported_tick)) as f32 / INFO_STALE_TICKS as f32;
            if staleness > 2.0 { continue; } // too stale
            let confidence = (1.0 - staleness * 0.5).max(0.1);

            for c in 0..8 {
                let local_buy_price = local_prices[c];
                let remote_sell_price = report.prices[c];
                let margin = remote_sell_price - local_buy_price;
                if margin <= 0.0 { continue; }

                let units = carry_capacity.min(
                    // Don't buy more than we can afford
                    if local_buy_price > 0.01 { adv.gold / local_buy_price } else { carry_capacity }
                );
                if units < 1.0 { continue; }

                let profit = margin * units * confidence;
                if profit > best_profit {
                    best_profit = profit;
                    best_opp = Some(TradeOpp {
                        adv_idx: idx,
                        dest_loc_id: report.location_id,
                        commodity: c,
                        expected_profit: profit,
                        buy_amount: units,
                        buy_cost: units * local_buy_price,
                    });
                }
            }
        }

        if let Some(opp) = best_opp {
            // Only trade if profit exceeds a threshold (worth the trip).
            if opp.expected_profit > 5.0 {
                opportunities.push(opp);
            }
        }
    }

    // Execute trade decisions: buy goods and set intent to travel.
    for opp in &opportunities {
        let adv = &mut state.adventurers[opp.adv_idx];
        let loc_id = adv.home_location_id.unwrap_or(0);

        // Buy from local stockpile.
        if let Some(loc) = state.overworld.locations.iter_mut().find(|l| l.id == loc_id) {
            let stockpile_arr = [
                &mut loc.stockpile.food, &mut loc.stockpile.iron, &mut loc.stockpile.wood,
                &mut loc.stockpile.herbs, &mut loc.stockpile.hide, &mut loc.stockpile.crystal,
                &mut loc.stockpile.equipment, &mut loc.stockpile.medicine,
            ];
            let available = *stockpile_arr[opp.commodity];
            let actual_buy = opp.buy_amount.min(available);
            if actual_buy < 1.0 { continue; }

            *stockpile_arr[opp.commodity] -= actual_buy;
            let cost = actual_buy * state.overworld.locations.iter()
                .find(|l| l.id == loc_id)
                .map(|l| l.local_prices[opp.commodity])
                .unwrap_or(1.0);

            // Pay for goods (gold to local producers via treasury as proxy).
            let adv = &mut state.adventurers[opp.adv_idx];
            let actual_cost = cost.min(adv.gold.max(0.0));
            adv.gold -= actual_cost;
            adv.carried_goods[opp.commodity] += actual_buy;

            // Set intent to travel to destination.
            adv.economic_intent = EconomicIntent::SeekingParty {
                purpose: PartyPurpose::TradeRun {
                    from: loc_id,
                    to: opp.dest_loc_id,
                },
            };
        }
    }

    // Handle merchants arriving at destinations: sell carried goods.
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead { continue; }
        // Check if merchant has arrived (Traveling intent but at their destination).
        // This is simplified — in practice, the party arrival system should trigger this.
        // For now, check if any carried goods and they're Working at a location.
        if !matches!(adv.economic_intent, EconomicIntent::Working) { continue; }
        let loc_id = match adv.home_location_id {
            Some(id) => id,
            None => continue,
        };

        // Sell any carried goods at local price.
        let mut total_sold = 0.0_f32;
        for c in 0..8 {
            if adv.carried_goods[c] > 0.01 {
                let amount = adv.carried_goods[c];
                // Add to local stockpile.
                if let Some(loc) = state.overworld.locations.iter_mut().find(|l| l.id == loc_id) {
                    let price = loc.local_prices[c];
                    match c {
                        0 => loc.stockpile.food += amount,
                        1 => loc.stockpile.iron += amount,
                        2 => loc.stockpile.wood += amount,
                        3 => loc.stockpile.herbs += amount,
                        4 => loc.stockpile.hide += amount,
                        5 => loc.stockpile.crystal += amount,
                        6 => loc.stockpile.equipment += amount,
                        7 => loc.stockpile.medicine += amount,
                        _ => {}
                    }
                    // Gold from consumers at destination (simplified — taken from treasury).
                    let revenue = amount * price;
                    let available_treasury = loc.treasury;
                    let actual_revenue = revenue.min(available_treasury);
                    loc.treasury -= actual_revenue;
                    total_sold += actual_revenue;
                }
                adv.carried_goods[c] = 0.0;
            }
        }
        adv.gold += total_sold;
    }
}

// ---------------------------------------------------------------------------
// Service Transactions
// ---------------------------------------------------------------------------

/// NPC-to-NPC service payments: injured NPCs pay healers, NPCs with degraded
/// gear pay blacksmiths. Gold flows directly between NPCs.
fn tick_services(state: &mut CampaignState) {
    // Per-settlement: find providers and consumers of services.
    let settlement_ids: Vec<usize> = state.overworld.locations.iter()
        .filter(|l| l.location_type == LocationType::Settlement)
        .map(|l| l.id)
        .collect();

    for &loc_id in &settlement_ids {
        // Find healers and injured NPCs at this settlement.
        let mut healer_idxs: Vec<usize> = Vec::new();
        let mut patient_idxs: Vec<usize> = Vec::new();

        for (idx, adv) in state.adventurers.iter().enumerate() {
            if adv.status == AdventurerStatus::Dead { continue; }
            if adv.home_location_id != Some(loc_id) { continue; }

            let is_healer = adv.classes.iter().any(|c| {
                let name = c.class_name.to_lowercase();
                name.contains("healer") || name.contains("doctor") || name.contains("cleric")
                    || name.contains("herbalist") || name.contains("blessed")
            });
            if is_healer {
                healer_idxs.push(idx);
            }
            if adv.injury > 10.0 {
                patient_idxs.push(idx);
            }
        }

        // Match patients to healers. Each healer treats one patient per cycle.
        for (i, &healer_idx) in healer_idxs.iter().enumerate() {
            if i >= patient_idxs.len() { break; }
            let patient_idx = patient_idxs[i];
            if healer_idx == patient_idx { continue; }

            let healer_level = state.adventurers[healer_idx].level as f32;
            let patient_injury = state.adventurers[patient_idx].injury;
            let service_price = (patient_injury / 100.0) * healer_level * 0.5;

            // Patient pays (capped by what they can afford).
            let patient = &state.adventurers[patient_idx];
            let payment = service_price.min(patient.gold.max(0.0));
            if payment < 0.01 { continue; }

            // Transfer gold and heal.
            state.adventurers[patient_idx].gold -= payment;
            state.adventurers[healer_idx].gold += payment;
            state.adventurers[patient_idx].injury = (state.adventurers[patient_idx].injury - healer_level * 0.5).max(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Settlement Threats — connect monster ecology to NPC economy
// ---------------------------------------------------------------------------

/// When monster populations in a region are high enough to attack (population > 80),
/// the monster_ecology system emits MonsterAttack events and reduces region control.
/// This function translates that into real damage to settlements: stockpile raiding,
/// NPC injuries/deaths, and combat behavior generation for defenders.
///
/// Also applies faction war damage to settlements when factions attack.
fn apply_monster_damage_to_settlements(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Read actual monster populations per region.
    // Settlements in regions with high monster populations get attacked.
    struct RegionThreat {
        region_id: usize,
        total_population: f32,
        max_aggression: f32,
    }
    let mut region_threats: Vec<RegionThreat> = Vec::new();
    for pop in &state.monster_populations {
        if let Some(rt) = region_threats.iter_mut().find(|r| r.region_id == pop.region_id) {
            rt.total_population += pop.population;
            rt.max_aggression = rt.max_aggression.max(pop.aggression);
        } else {
            region_threats.push(RegionThreat {
                region_id: pop.region_id,
                total_population: pop.population,
                max_aggression: pop.aggression,
            });
        }
    }

    // Also factor in faction hostility: factions at war increase danger.
    let guild_fid = state.diplomacy.guild_faction_id;
    for faction in &state.factions {
        if faction.at_war_with.contains(&guild_fid) || matches!(faction.diplomatic_stance, crate::headless_campaign::state::DiplomaticStance::AtWar) {
            // Hostile faction's territory has extra threat.
            for region in &state.overworld.regions {
                if region.owner_faction_id == faction.id {
                    if let Some(rt) = region_threats.iter_mut().find(|r| r.region_id == region.id) {
                        rt.total_population += faction.military_strength * 0.5;
                        rt.max_aggression = rt.max_aggression.max(80.0);
                    } else {
                        region_threats.push(RegionThreat {
                            region_id: region.id,
                            total_population: faction.military_strength * 0.5,
                            max_aggression: 80.0,
                        });
                    }
                }
            }
        }
    }

    // Update settlement threat_level as a DESCRIPTIVE value from actual hostile actors.
    for loc in &mut state.overworld.locations {
        if loc.location_type != LocationType::Settlement { continue; }
        // Find which region this settlement is in (approximate by faction owner).
        let region_threat = loc.faction_owner
            .and_then(|fid| {
                state.overworld.regions.iter()
                    .find(|r| r.owner_faction_id == fid)
                    .map(|r| r.id)
            })
            .and_then(|rid| region_threats.iter().find(|rt| rt.region_id == rid))
            .map(|rt| rt.total_population * 0.5 + rt.max_aggression * 0.3)
            .unwrap_or(0.0);
        // Blend: 80% from hostile actors, 20% from base terrain danger.
        loc.threat_level = region_threat * 0.8 + loc.threat_level * 0.2;
    }

    // Now apply actual attacks to settlements in threatened regions.
    struct SettlementSnap {
        loc_id: usize,
        region_threat_pop: f32,
        region_aggression: f32,
        safety: f32,
        pop: usize,
    }
    let snaps: Vec<SettlementSnap> = state.overworld.locations.iter()
        .filter(|l| l.location_type == LocationType::Settlement && !l.resident_ids.is_empty())
        .filter_map(|l| {
            let rt = l.faction_owner
                .and_then(|fid| state.overworld.regions.iter().find(|r| r.owner_faction_id == fid).map(|r| r.id))
                .and_then(|rid| region_threats.iter().find(|rt| rt.region_id == rid));
            rt.map(|rt| SettlementSnap {
                loc_id: l.id,
                region_threat_pop: rt.total_population,
                region_aggression: rt.max_aggression,
                safety: l.safety_level,
                pop: l.resident_ids.len(),
            })
        })
        .collect();

    let mut killed: Vec<u32> = Vec::new();

    for snap in &snaps {
        // Attacks only happen when monster population is high AND aggressive.
        if snap.region_threat_pop < 40.0 || snap.region_aggression < 30.0 { continue; }

        // Attack probability from actual hostile actor pressure.
        let pressure = (snap.region_threat_pop / 100.0) * (snap.region_aggression / 100.0);
        let attack_chance = pressure * 0.05; // max ~5% per tick at full pressure
        let roll = crate::headless_campaign::state::lcg_f32(&mut state.rng);
        if roll >= attack_chance { continue; }

        let severity = pressure; // 0.0-1.0

        // Defense check: safety_level vs monster threat.
        let effective_threat = snap.region_threat_pop * 0.5;
        let defense_ratio = if effective_threat > 0.0 { snap.safety / effective_threat } else { 1.0 };

        if defense_ratio >= 1.0 {
            // Defended — minor losses, defenders gain combat XP.
            settlement_defense_behavior(state, snap.loc_id, severity * 30.0);
            if let Some(loc) = state.overworld.locations.iter_mut().find(|l| l.id == snap.loc_id) {
                loc.stockpile.food = (loc.stockpile.food - severity * 5.0).max(0.0);
            }
        } else if defense_ratio >= 0.3 {
            // Partial defense.
            let damage_mult = 1.0 - defense_ratio;
            if let Some(loc) = state.overworld.locations.iter_mut().find(|l| l.id == snap.loc_id) {
                let raid = severity * 20.0 * damage_mult;
                loc.stockpile.food = (loc.stockpile.food - raid * 2.0).max(0.0);
                loc.stockpile.iron = (loc.stockpile.iron - raid).max(0.0);
                loc.stockpile.wood = (loc.stockpile.wood - raid).max(0.0);
                loc.stockpile.equipment = (loc.stockpile.equipment - raid * 0.5).max(0.0);

                let resident_ids = loc.resident_ids.clone();
                let injury_count = ((snap.pop as f32) * damage_mult * 0.15).ceil() as usize;
                for _ in 0..injury_count.min(resident_ids.len()) {
                    let idx = (crate::headless_campaign::state::lcg_next(&mut state.rng) as usize) % resident_ids.len();
                    let adv_id = resident_ids[idx];
                    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
                        if adv.status != AdventurerStatus::Dead {
                            adv.injury = (adv.injury + severity * 20.0).min(100.0);
                            adv.stress = (adv.stress + severity * 15.0).min(100.0);
                        }
                    }
                }
            }
            settlement_defense_behavior(state, snap.loc_id, severity * 50.0);
        } else {
            // Overwhelmed.
            if let Some(loc) = state.overworld.locations.iter_mut().find(|l| l.id == snap.loc_id) {
                let raid = severity * 40.0;
                loc.stockpile.food = (loc.stockpile.food - raid * 3.0).max(0.0);
                loc.stockpile.iron = (loc.stockpile.iron - raid * 2.0).max(0.0);
                loc.stockpile.wood = (loc.stockpile.wood - raid * 2.0).max(0.0);
                loc.stockpile.herbs = (loc.stockpile.herbs - raid).max(0.0);
                loc.stockpile.equipment = (loc.stockpile.equipment - raid).max(0.0);
                loc.stockpile.medicine = (loc.stockpile.medicine - raid).max(0.0);
                loc.treasury = (loc.treasury - raid).max(0.0);

                let resident_ids = loc.resident_ids.clone();
                let casualty_count = ((snap.pop as f32) * severity * 0.1).ceil() as usize;
                for _ in 0..casualty_count.min(resident_ids.len()) {
                    let idx = (crate::headless_campaign::state::lcg_next(&mut state.rng) as usize) % resident_ids.len();
                    let adv_id = resident_ids[idx];
                    if killed.contains(&adv_id) { continue; }
                    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
                        if adv.status == AdventurerStatus::Dead { continue; }
                        adv.injury = (adv.injury + severity * 30.0).min(100.0);
                        adv.stress = 100.0;
                        if adv.injury > 80.0 {
                            let death_roll = crate::headless_campaign::state::lcg_f32(&mut state.rng);
                            if death_roll < 0.15 * severity {
                                adv.status = AdventurerStatus::Dead;
                                killed.push(adv_id);
                            }
                        }
                    }
                }
            }
            settlement_defense_behavior(state, snap.loc_id, severity * 80.0);
        }
    }

    // Clean up dead NPCs from resident lists.
    if !killed.is_empty() {
        for loc in &mut state.overworld.locations {
            loc.resident_ids.retain(|id| !killed.contains(id));
        }
    }
}

// ---------------------------------------------------------------------------
// Treasury Spending (recirculation)
// ---------------------------------------------------------------------------

/// Settlement treasuries spend on bounties, emergency food, and infrastructure.
/// This recirculates gold that was drained via transaction tax.
fn tick_treasury_spending(state: &mut CampaignState) {
    for loc in &mut state.overworld.locations {
        if loc.location_type != LocationType::Settlement { continue; }
        if loc.treasury < 10.0 { continue; }

        let budget = loc.treasury * 0.2; // spend up to 20% per cycle

        // Priority 1: Emergency food purchase — if food < 20 ticks supply.
        let pop = loc.resident_ids.len() as f32;
        let food_ticks = if pop > 0.0 { loc.stockpile.food / (pop * 0.2) } else { 999.0 };
        if food_ticks < 20.0 && budget > 5.0 {
            // Buy food at premium — inject gold back into NPC economy
            // by adding food to stockpile (as if purchased from outside).
            let food_to_buy = (budget / 2.0).min(pop * 10.0);
            loc.stockpile.food += food_to_buy;
            loc.treasury -= food_to_buy * 0.5; // pay half the budget for emergency food
        }

        // Priority 2: Infrastructure (simplified — just improve resource_availability)
        if loc.treasury > 50.0 {
            let infra_spend = (loc.treasury * 0.05).min(20.0);
            loc.resource_availability = (loc.resource_availability + infra_spend * 0.01).min(100.0);
            loc.treasury -= infra_spend;
        }
    }
}

// Legacy income/expense (to be removed)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
/// Apply per-tick service income to working NPCs. Guild receives tax.
fn apply_npc_income(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let cfg = state.config.npc_economy.clone();
    let exp = cfg.demand_income_exponent;
    let base_rate = cfg.service_income_per_stat_point;
    let tax_rate = cfg.guild_tax_rate;
    let variance = cfg.income_variance;

    // Snapshot location demands (avoid borrow issues).
    let location_demands: Vec<(usize, [f32; 8])> = state
        .overworld
        .locations
        .iter()
        .map(|l| (l.id, l.service_demand))
        .collect();

    // Per-location tax accumulator (location_id → tax collected).
    let mut loc_tax: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
    let rng = &mut state.rng;

    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        if !matches!(adv.economic_intent, crate::headless_campaign::state::EconomicIntent::Working) {
            adv.ticks_since_income += 1;
            continue;
        }
        let loc_id = match adv.home_location_id {
            Some(id) => id,
            None => {
                adv.ticks_since_income += 1;
                continue;
            }
        };
        let demand = match location_demands.iter().find(|(id, _)| *id == loc_id) {
            Some((_, d)) => d,
            None => {
                adv.ticks_since_income += 1;
                continue;
            }
        };

        let stats = effective_noncombat_stats(adv);
        let eff_lvl = effective_level_with_config(adv, cfg.resource_level_threshold);
        let lvl_mult = eff_lvl / 10.0;

        let stat_arr = [stats.0, stats.1, stats.2, stats.3, stats.4, stats.5, stats.6];
        let mut raw_income = 0.0_f32;
        for (i, &s) in stat_arr.iter().enumerate() {
            raw_income += s * lvl_mult * demand[i].powf(exp) * base_rate;
        }
        // Combat income channel (index 7).
        let combat_stat = (adv.stats.attack + adv.stats.defense + adv.stats.ability_power) / 30.0;
        raw_income += combat_stat * lvl_mult * demand[7].powf(exp) * cfg.base_combat_income_rate;

        // Apply variance: ±income_variance via deterministic RNG.
        let roll = crate::headless_campaign::state::lcg_f32(rng);
        let var_mult = 1.0 - variance + roll * 2.0 * variance; // [1-v, 1+v]
        let income = (raw_income * var_mult).max(0.0);

        let tax = income * tax_rate;
        adv.gold += income - tax;
        // Tax goes to settlement treasury, not guild.
        *loc_tax.entry(loc_id).or_insert(0.0) += tax;
        adv.ticks_since_income = 0;
    }

    // Deposit collected taxes into settlement treasuries.
    for (loc_id, tax) in loc_tax {
        if let Some(loc) = state.overworld.locations.iter_mut().find(|l| l.id == loc_id) {
            loc.treasury += tax;
        }
    }
}

// ---------------------------------------------------------------------------
// Subsystem: NPC expenses
// ---------------------------------------------------------------------------

#[allow(dead_code)]
/// Deduct living costs from NPCs at settlements. Bankrupt NPCs gain stress.
fn apply_npc_expenses(state: &mut CampaignState) {
    let cfg = state.config.npc_economy.clone();

    // Snapshot cost_of_living per location.
    let location_costs: Vec<(usize, f32)> = state
        .overworld
        .locations
        .iter()
        .map(|l| (l.id, l.cost_of_living))
        .collect();

    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        let loc_id = match adv.home_location_id {
            Some(id) => id,
            None => continue, // In transit — no living costs
        };
        let col = location_costs
            .iter()
            .find(|(id, _)| *id == loc_id)
            .map(|(_, c)| *c)
            .unwrap_or(1.0);

        let eff_lvl = effective_level_with_config(adv, cfg.resource_level_threshold);
        let base = cfg.base_living_cost_per_tick * col;
        let upkeep = cfg.equipment_upkeep_per_tick * (eff_lvl / 10.0);
        let medical = if adv.injury > 0.0 {
            (adv.injury / 100.0) * cfg.injury_cost_multiplier * cfg.base_living_cost_per_tick
        } else {
            0.0
        };

        let expenses = base + upkeep + medical;
        adv.gold -= expenses;

        // Bankruptcy stress: going broke is stressful.
        if adv.gold < 0.0 {
            adv.stress = (adv.stress + 0.5).min(100.0);
        }
    }

    // Update cost_of_living per location based on resident count.
    let pop_scale = cfg.cost_pop_scale;
    for loc in &mut state.overworld.locations {
        let resident_count = loc.resident_ids.len() as f32;
        loc.cost_of_living = 1.0 + (resident_count / pop_scale).powf(1.2);
    }
}

// ---------------------------------------------------------------------------
// Subsystem: Travel danger
// ---------------------------------------------------------------------------

/// Apply per-tick injury/death checks to NPCs in traveling autonomous parties.
/// Level-gated: below viability floor = near-certain casualties.
fn apply_travel_danger(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let cfg = state.config.npc_economy.clone();

    // Collect autonomous traveling parties.
    let traveling_parties: Vec<(u32, Vec<u32>, (f32, f32))> = state
        .parties
        .iter()
        .filter(|p| p.autonomous && matches!(p.status, crate::headless_campaign::state::PartyStatus::Traveling))
        .map(|p| (p.id, p.member_ids.clone(), p.position))
        .collect();

    for (party_id, member_ids, position) in &traveling_parties {
        // Estimate regional threat at party's current position.
        let region_threat = state
            .overworld
            .regions
            .iter()
            .map(|r| r.threat_level)
            .fold(0.0_f32, |acc, t| acc.max(t))
            * 0.5; // rough average — proper spatial lookup would use position

        // Find party's max combat effective level.
        let party_max = member_ids
            .iter()
            .filter_map(|id| state.adventurers.iter().find(|a| a.id == *id))
            .map(|a| combat_effective_level(a))
            .fold(0.0_f32, f32::max);

        let floor = region_threat * cfg.viability_floor_fraction;
        let injury_chance = if party_max < floor {
            cfg.below_floor_injury_chance
        } else if party_max < region_threat {
            cfg.attrition_injury_rate * (region_threat / party_max.max(1.0))
        } else {
            cfg.base_travel_injury_chance
        };

        let injury_chance = injury_chance.clamp(0.0, cfg.below_floor_injury_chance);

        // Roll for injury on a random member.
        let roll = crate::headless_campaign::state::lcg_f32(&mut state.rng);
        if roll < injury_chance && !member_ids.is_empty() {
            let target_idx = (crate::headless_campaign::state::lcg_f32(&mut state.rng) * member_ids.len() as f32) as usize;
            let target_idx = target_idx.min(member_ids.len() - 1);
            let target_id = member_ids[target_idx];

            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == target_id) {
                let severity = cfg.travel_injury_base + region_threat * 0.2;
                adv.injury = (adv.injury + severity).min(100.0);
                adv.stress = (adv.stress + 5.0).min(100.0);
                adv.fatigue = (adv.fatigue + 3.0).min(100.0);

                events.push(WorldEvent::TravelEncounter {
                    party_id: *party_id,
                    adventurer_id: target_id,
                    injury: severity,
                    threat: region_threat,
                });

                // Death check.
                if adv.injury > 90.0 {
                    let death_roll = crate::headless_campaign::state::lcg_f32(&mut state.rng);
                    if death_roll < cfg.travel_death_chance {
                        adv.status = AdventurerStatus::Dead;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Subsystem stubs (implemented in later phases)
// ---------------------------------------------------------------------------

/// Phase 3: NPC utility-based decision making.
/// Each NPC evaluates: stay and work, relocate, or seek adventuring.
fn run_npc_decisions(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    use crate::headless_campaign::state::{EconomicIntent, PartyPurpose, GoalType};

    let cfg = state.config.npc_economy.clone();

    // Snapshot location data to avoid borrow conflicts.
    struct LocSnap {
        id: usize,
        demand: [f32; 8],
        safety: f32,
        threat: f32,
        cost_of_living: f32,
        resident_count: usize,
    }
    let loc_snaps: Vec<LocSnap> = state
        .overworld
        .locations
        .iter()
        .filter(|l| l.location_type == crate::headless_campaign::state::LocationType::Settlement)
        .map(|l| LocSnap {
            id: l.id,
            demand: l.service_demand,
            safety: l.safety_level,
            threat: l.threat_level,
            cost_of_living: l.cost_of_living,
            resident_count: l.resident_ids.len(),
        })
        .collect();

    if loc_snaps.is_empty() {
        return;
    }

    // Collect NPC decisions to apply after iteration.
    struct Decision {
        adv_idx: usize,
        new_intent: EconomicIntent,
        new_home: Option<usize>,       // Some(loc_id) if relocating
        stagnating_class: Option<String>,
    }
    let mut decisions = Vec::new();

    for (adv_idx, adv) in state.adventurers.iter().enumerate() {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        // Only re-evaluate Idle or Working NPCs (not already traveling/seeking/adventuring).
        match &adv.economic_intent {
            EconomicIntent::Idle | EconomicIntent::Working => {}
            _ => continue,
        }

        let stats = effective_noncombat_stats(adv);
        let eff_lvl = effective_level_with_config(adv, cfg.resource_level_threshold);
        let lvl_mult = eff_lvl / 10.0;
        let stat_arr = [stats.0, stats.1, stats.2, stats.3, stats.4, stats.5, stats.6];
        let combat_stat = (adv.stats.attack + adv.stats.defense + adv.stats.ability_power) / 30.0;

        // Risk aversion from personality: high stress/fear = risk averse.
        let risk_aversion = 1.0 + (adv.stress / 100.0) * 0.5
            + (adv.fears.len() as f32) * 0.2
            - (adv.resolve / 100.0) * 0.3;
        let risk_aversion = risk_aversion.clamp(0.3, 3.0);

        // --- Check class stagnation (combat classes only) ---
        let mut stagnating_class: Option<String> = None;
        for class in &adv.classes {
            let is_combat = class.stat_growth_diplomacy == 0.0
                && class.stat_growth_commerce == 0.0
                && class.stat_growth_crafting == 0.0
                && class.stat_growth_medicine == 0.0
                && class.stat_growth_scholarship == 0.0
                && class.stat_growth_stealth == 0.0
                && class.stat_growth_leadership == 0.0;
            if is_combat && class.stagnation_ticks > 100 {
                stagnating_class = Some(class.class_name.clone());
                break;
            }
        }

        // --- Utility: Stay and work at current location ---
        let current_loc = adv.home_location_id;
        let stay_utility = if let Some(loc_id) = current_loc {
            if let Some(snap) = loc_snaps.iter().find(|s| s.id == loc_id) {
                let income = estimate_income(&stat_arr, combat_stat, lvl_mult, &snap.demand, &cfg);
                let safety = if snap.safety > 0.0 { (snap.safety / snap.threat.max(1.0)).min(1.0) } else { 0.1 };
                let growth = if stagnating_class.is_some() { 0.0 } else { 0.5 }; // non-combat classes grow fine
                income + 2.0 * safety + 0.5 * growth
            } else {
                0.0
            }
        } else {
            0.0
        };

        // --- Utility: Relocate to best alternative settlement ---
        let mut best_reloc_utility = f32::NEG_INFINITY;
        let mut best_reloc_loc = None;
        for snap in &loc_snaps {
            if Some(snap.id) == current_loc {
                continue;
            }
            let income = estimate_income(&stat_arr, combat_stat, lvl_mult, &snap.demand, &cfg);
            let safety = if snap.safety > 0.0 { (snap.safety / snap.threat.max(1.0)).min(1.0) } else { 0.1 };
            // Travel risk: estimate route danger (simplified — use max regional threat).
            let route_threat = snap.threat * 0.5; // simplified estimation
            let death_risk = if eff_lvl > route_threat * cfg.viability_floor_fraction {
                0.05
            } else {
                0.8
            };
            let travel_penalty = death_risk * risk_aversion * 3.0;
            let u = income + 2.0 * safety - travel_penalty;
            if u > best_reloc_utility {
                best_reloc_utility = u;
                best_reloc_loc = Some(snap.id);
            }
        }

        // --- Utility: Found a new settlement ---
        // Attractive when current settlement is overcrowded (high cost of living)
        // and there are viable locations nearby.
        let founding_utility = if let Some(loc_id) = current_loc {
            if let Some(snap) = loc_snaps.iter().find(|s| s.id == loc_id) {
                if snap.cost_of_living > 2.0 && adv.gold < 0.0 {
                    // Overcrowded and broke — founding is very attractive.
                    // New settlement: low cost of living, high demand, but unsafe initially.
                    let new_income = 0.5; // high demand (sole provider)
                    let new_safety = 0.2; // dangerous until established
                    let death_risk = 0.1 * risk_aversion;
                    new_income + 2.0 * new_safety - death_risk * 3.0 + 1.0 // bonus for desperation
                } else if snap.cost_of_living > 3.0 {
                    // Very overcrowded even if not broke
                    0.5
                } else {
                    f32::NEG_INFINITY
                }
            } else {
                f32::NEG_INFINITY
            }
        } else {
            f32::NEG_INFINITY
        };

        // --- Utility: Seek adventuring party (combat classes with stagnation) ---
        let adventure_utility = if stagnating_class.is_some() {
            let growth = 1.0; // adventuring provides combat XP
            let income = 0.3; // quest income roughly comparable but volatile
            let safety = 0.1; // dangerous
            let death_risk = 0.3 * risk_aversion;
            // Goal bonus
            let goal_bonus = match &adv.personal_goal {
                Some(g) => match &g.goal_type {
                    GoalType::ReachLevel { .. } => 0.5,
                    GoalType::DefeatNemesis { .. } => 1.0,
                    GoalType::ExploreAllRegions => 0.5,
                    GoalType::EarnTitle => 0.3,
                    _ => 0.0,
                },
                None => 0.0,
            };
            income + 2.0 * safety + 1.5 * growth + goal_bonus - death_risk * 3.0
        } else {
            f32::NEG_INFINITY // non-stagnating NPCs don't consider adventuring
        };

        // --- Pick best option ---
        let mut best = stay_utility;
        let mut choice = EconomicIntent::Working;
        let mut new_home = None;

        // Require significant improvement to relocate (hysteresis).
        if best_reloc_utility > best * 1.3 {
            if let Some(loc_id) = best_reloc_loc {
                best = best_reloc_utility;
                choice = EconomicIntent::SeekingParty {
                    purpose: PartyPurpose::Relocation { destination: loc_id },
                };
                new_home = Some(loc_id);
            }
        }
        if founding_utility > best * 1.1 {
            best = founding_utility;
            // Pick a position offset from current settlement for the new site.
            let base_pos = current_loc
                .and_then(|lid| loc_snaps.iter().find(|s| s.id == lid))
                .map(|_| {
                    // Generate a position near but not on top of existing settlements
                    let offset_x = (crate::headless_campaign::state::lcg_f32(&mut state.rng) - 0.5) * 40.0;
                    let offset_y = (crate::headless_campaign::state::lcg_f32(&mut state.rng) - 0.5) * 40.0;
                    // Get current location position
                    state.overworld.locations.iter()
                        .find(|l| Some(l.id) == current_loc)
                        .map(|l| (l.position.0 + offset_x, l.position.1 + offset_y))
                        .unwrap_or((offset_x, offset_y))
                })
                .unwrap_or((0.0, 0.0));
            choice = EconomicIntent::SeekingParty {
                purpose: PartyPurpose::Founding { target_position: base_pos },
            };
            new_home = None;
        }
        if adventure_utility > best * 1.2 {
            best = adventure_utility;
            choice = EconomicIntent::SeekingParty {
                purpose: PartyPurpose::Adventuring,
            };
            new_home = None;
        }
        let _ = best; // suppress unused warning

        // Only record if intent changed.
        let intent_changed = !matches!(
            (&adv.economic_intent, &choice),
            (EconomicIntent::Working, EconomicIntent::Working)
                | (EconomicIntent::Idle, EconomicIntent::Working)
        );
        if intent_changed || new_home.is_some() {
            decisions.push(Decision {
                adv_idx,
                new_intent: choice,
                new_home,
                stagnating_class: stagnating_class.clone(),
            });
        }

        // Emit stagnation event.
        if let Some(class_name) = &stagnating_class {
            // Only emit periodically (every ~10 decision cycles).
            if state.tick % (cfg.decision_interval_ticks * 10) == 0 {
                events.push(WorldEvent::NpcClassStagnating {
                    adventurer_id: adv.id,
                    class_name: class_name.clone(),
                    ticks_stagnant: adv.classes.iter()
                        .find(|c| &c.class_name == class_name)
                        .map(|c| c.stagnation_ticks)
                        .unwrap_or(0),
                });
            }
        }
    }

    // Apply decisions.
    for dec in decisions {
        let adv = &mut state.adventurers[dec.adv_idx];
        let old_home = adv.home_location_id;
        adv.economic_intent = dec.new_intent;

        if let Some(new_loc) = dec.new_home {
            // Will actually relocate once they join a travel party (Phase 4).
            // For now, emit the intent event.
            events.push(WorldEvent::NpcRelocating {
                adventurer_id: adv.id,
                from: old_home,
                to: new_loc,
            });
        }
    }
}

/// Estimate service income at a location given NPC stats and demand.
fn estimate_income(
    stat_arr: &[f32; 7],
    combat_stat: f32,
    lvl_mult: f32,
    demand: &[f32; 8],
    cfg: &crate::headless_campaign::config::NpcEconomyConfig,
) -> f32 {
    let exp = cfg.demand_income_exponent;
    let base_rate = cfg.service_income_per_stat_point;
    let mut income = 0.0_f32;
    for (i, &s) in stat_arr.iter().enumerate() {
        income += s * lvl_mult * demand[i].powf(exp) * base_rate;
    }
    income += combat_stat * lvl_mult * demand[7].powf(exp) * cfg.base_combat_income_rate;
    income
}

/// Phase 4: Autonomous party formation (adventuring, caravan, travel).
/// Groups SeekingParty NPCs at the same location by purpose and forms viable parties.
fn run_party_formation(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    use crate::headless_campaign::state::{
        EconomicIntent, PartyPurpose, AutonomousPartyType, PartyStatus,
    };

    let cfg = state.config.npc_economy.clone();

    // Collect seekers grouped by location.
    struct Seeker {
        adv_idx: usize,
        adv_id: u32,
        purpose: PartyPurpose,
        combat_eff_level: f32,
        position: (f32, f32),
    }
    let mut seekers_by_loc: std::collections::HashMap<usize, Vec<Seeker>> =
        std::collections::HashMap::new();

    for (idx, adv) in state.adventurers.iter().enumerate() {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        if let EconomicIntent::SeekingParty { purpose } = &adv.economic_intent {
            if let Some(loc_id) = adv.home_location_id {
                let pos = state
                    .overworld
                    .locations
                    .iter()
                    .find(|l| l.id == loc_id)
                    .map(|l| l.position)
                    .unwrap_or((0.0, 0.0));
                seekers_by_loc.entry(loc_id).or_default().push(Seeker {
                    adv_idx: idx,
                    adv_id: adv.id,
                    purpose: purpose.clone(),
                    combat_eff_level: combat_effective_level(adv),
                    position: pos,
                });
            }
        }
    }

    // Generate a new party ID.
    let mut next_party_id = state
        .parties
        .iter()
        .map(|p| p.id)
        .max()
        .unwrap_or(10000)
        + 1;

    let mut new_parties: Vec<crate::headless_campaign::state::Party> = Vec::new();
    let mut formed_adv_idxs: Vec<usize> = Vec::new();
    let mut party_events: Vec<WorldEvent> = Vec::new();

    for (_loc_id, seekers) in &seekers_by_loc {
        // --- 1. Adventuring parties ---
        let adventure_seekers: Vec<&Seeker> = seekers
            .iter()
            .filter(|s| matches!(s.purpose, PartyPurpose::Adventuring))
            .collect();

        if adventure_seekers.len() >= 2 {
            let max_combat = adventure_seekers
                .iter()
                .map(|s| s.combat_eff_level)
                .fold(0.0_f32, f32::max);
            // Only form if the party has meaningful combat capability.
            if max_combat > 5.0 {
                let members: Vec<&Seeker> = adventure_seekers
                    .iter()
                    .take(cfg.max_autonomous_party_size)
                    .copied()
                    .collect();
                let member_ids: Vec<u32> = members.iter().map(|s| s.adv_id).collect();
                let pos = members[0].position;

                new_parties.push(crate::headless_campaign::state::Party {
                    id: next_party_id,
                    member_ids: member_ids.clone(),
                    position: pos,
                    destination: None, // Quest system will assign destination
                    speed: 1.0,
                    status: PartyStatus::Idle,
                    supply_level: 80.0,
                    morale: 70.0,
                    quest_id: None,
                    food_level: 80.0,
                    autonomous: true,
                    party_type: AutonomousPartyType::Adventuring,
                });
                party_events.push(WorldEvent::AdventuringPartyFormed {
                    party_id: next_party_id,
                    member_ids: member_ids.clone(),
                    region: String::from("frontier"),
                });
                for m in &members {
                    formed_adv_idxs.push(m.adv_idx);
                }
                next_party_id += 1;
            }
        }

        // --- 2. Founding parties ---
        let founders: Vec<&Seeker> = seekers
            .iter()
            .filter(|s| matches!(s.purpose, PartyPurpose::Founding { .. }))
            .filter(|s| !formed_adv_idxs.contains(&s.adv_idx))
            .collect();

        if founders.len() >= 3 {
            // Founding expeditions are desperate — lower combat bar than other parties.
            // Even level-1 NPCs can found if there are enough of them.
            let max_combat = founders.iter().map(|s| s.combat_eff_level).fold(0.0_f32, f32::max);
            if max_combat > 0.5 || founders.len() >= 10 {
                let members: Vec<&Seeker> = founders.iter()
                    .take(cfg.max_autonomous_party_size)
                    .copied()
                    .collect();
                let member_ids: Vec<u32> = members.iter().map(|s| s.adv_id).collect();
                let pos = members[0].position;
                // Use the first founder's target position.
                let target_pos = match &members[0].purpose {
                    PartyPurpose::Founding { target_position } => *target_position,
                    _ => (pos.0 + 20.0, pos.1 + 20.0),
                };

                new_parties.push(crate::headless_campaign::state::Party {
                    id: next_party_id,
                    member_ids: member_ids.clone(),
                    position: pos,
                    destination: Some(target_pos),
                    speed: 0.8, // slower — carrying supplies
                    status: PartyStatus::Traveling,
                    supply_level: 50.0,
                    morale: 60.0,
                    quest_id: None,
                    food_level: 50.0,
                    autonomous: true,
                    party_type: AutonomousPartyType::Travel, // will become settlement on arrival
                });
                party_events.push(WorldEvent::TravelPartyFormed {
                    party_id: next_party_id,
                    member_ids: member_ids.clone(),
                    destination: 0, // new settlement — doesn't exist yet
                });
                for m in &members {
                    formed_adv_idxs.push(m.adv_idx);
                }
                next_party_id += 1;
            }
        }

        // --- 3. Travel / relocation parties ---
        // Group relocation seekers by destination.
        let mut reloc_by_dest: std::collections::HashMap<usize, Vec<&Seeker>> =
            std::collections::HashMap::new();
        for s in seekers.iter() {
            if let PartyPurpose::Relocation { destination } = &s.purpose {
                if !formed_adv_idxs.contains(&s.adv_idx) {
                    reloc_by_dest.entry(*destination).or_default().push(s);
                }
            }
        }
        for (dest_id, group) in &reloc_by_dest {
            if group.len() < 2 {
                continue;
            }
            let max_combat = group
                .iter()
                .map(|s| s.combat_eff_level)
                .fold(0.0_f32, f32::max);
            // Need at least some combat capability for travel.
            if max_combat < 3.0 {
                continue;
            }
            let dest_pos = state
                .overworld
                .locations
                .iter()
                .find(|l| l.id == *dest_id)
                .map(|l| l.position)
                .unwrap_or((0.0, 0.0));
            let member_ids: Vec<u32> = group.iter().map(|s| s.adv_id).collect();
            let pos = group[0].position;

            new_parties.push(crate::headless_campaign::state::Party {
                id: next_party_id,
                member_ids: member_ids.clone(),
                position: pos,
                destination: Some(dest_pos),
                speed: 1.0,
                status: PartyStatus::Traveling,
                supply_level: 60.0,
                morale: 60.0,
                quest_id: None,
                food_level: 60.0,
                autonomous: true,
                party_type: AutonomousPartyType::Travel,
            });
            party_events.push(WorldEvent::TravelPartyFormed {
                party_id: next_party_id,
                member_ids: member_ids.clone(),
                destination: *dest_id,
            });
            for m in group {
                formed_adv_idxs.push(m.adv_idx);
            }
            next_party_id += 1;
        }
    }

    // Apply: update adventurer intents for those who formed parties.
    for idx in &formed_adv_idxs {
        let adv = &mut state.adventurers[*idx];
        match adv.economic_intent {
            EconomicIntent::SeekingParty { purpose: PartyPurpose::Adventuring } => {
                adv.economic_intent = EconomicIntent::Adventuring;
            }
            EconomicIntent::SeekingParty { purpose: PartyPurpose::Relocation { .. } } => {
                adv.economic_intent = EconomicIntent::Traveling;
            }
            _ => {
                adv.economic_intent = EconomicIntent::Traveling;
            }
        }
    }

    // Handle timeout: NPCs seeking for too long give up.
    for adv in &mut state.adventurers {
        if let EconomicIntent::SeekingParty { .. } = &adv.economic_intent {
            // If they've been seeking for more than patience threshold, give up.
            if adv.ticks_since_income > cfg.party_seek_patience_ticks {
                adv.economic_intent = EconomicIntent::Working;
                adv.stress = (adv.stress + 5.0).min(100.0);
            }
        }
    }

    // Add new parties to state.
    state.parties.extend(new_parties);
    events.extend(party_events);
}

/// Phase 4: Manage existing adventuring parties (attrition, rotation, disbanding).
/// Members with high injury/stress leave. Parties with too few members disband.
fn manage_adventuring_parties(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    use crate::headless_campaign::state::{EconomicIntent, AutonomousPartyType, PartyPurpose};

    let mut parties_to_disband: Vec<u32> = Vec::new();
    let mut members_to_leave: Vec<(u32, u32)> = Vec::new(); // (party_id, adv_id)

    for party in &state.parties {
        if party.party_type != AutonomousPartyType::Adventuring {
            continue;
        }

        for &member_id in &party.member_ids {
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == member_id) {
                // Leave if too injured or stressed.
                if adv.injury > 60.0 || adv.stress > 80.0 || adv.status == AdventurerStatus::Dead {
                    members_to_leave.push((party.id, member_id));
                }
            }
        }

        // Disband if too few viable members.
        let viable_count = party
            .member_ids
            .iter()
            .filter(|id| {
                state
                    .adventurers
                    .iter()
                    .find(|a| a.id == **id)
                    .map(|a| a.status != AdventurerStatus::Dead && a.injury <= 60.0)
                    .unwrap_or(false)
            })
            .count();
        if viable_count < 2 {
            parties_to_disband.push(party.id);
        }
    }

    // Remove leaving members from their parties.
    for (party_id, adv_id) in &members_to_leave {
        if let Some(party) = state.parties.iter_mut().find(|p| p.id == *party_id) {
            party.member_ids.retain(|id| id != adv_id);
        }
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *adv_id) {
            adv.economic_intent = EconomicIntent::Idle;
            // Settle at nearest settlement if possible.
            if adv.home_location_id.is_none() {
                if let Some(loc) = state.overworld.locations.iter().find(|l| {
                    l.location_type == crate::headless_campaign::state::LocationType::Settlement
                }) {
                    adv.home_location_id = Some(loc.id);
                }
            }
        }
    }

    // Disband depleted parties.
    for party_id in &parties_to_disband {
        // Set all remaining members back to idle.
        if let Some(party) = state.parties.iter().find(|p| p.id == *party_id) {
            let member_ids = party.member_ids.clone();
            for mid in &member_ids {
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *mid) {
                    adv.economic_intent = EconomicIntent::Idle;
                    if adv.home_location_id.is_none() {
                        if let Some(loc) = state.overworld.locations.iter().find(|l| {
                            l.location_type
                                == crate::headless_campaign::state::LocationType::Settlement
                        }) {
                            adv.home_location_id = Some(loc.id);
                        }
                    }
                }
            }
        }
        events.push(WorldEvent::AdventuringPartyDisbanded {
            party_id: *party_id,
            reason: "Too few viable members".to_string(),
        });
    }
    state
        .parties
        .retain(|p| !parties_to_disband.contains(&p.id));

    // Clean up travel parties that have arrived.
    // Detect arrival by: destination is None (cleared by tick_travel on arrival),
    // OR status is Idle, OR status is OnMission with no quest.
    let mut arrived_travel: Vec<u32> = Vec::new();
    for party in &state.parties {
        if party.party_type != AutonomousPartyType::Travel {
            continue;
        }
        let arrived = party.destination.is_none()
            || party.status == crate::headless_campaign::state::PartyStatus::Idle;
        if arrived {
            arrived_travel.push(party.id);
        }
    }

    let mut new_settlements: Vec<(u32, (f32, f32), Vec<u32>)> = Vec::new(); // (party_id, position, members)
    for party_id in &arrived_travel {
        if let Some(party) = state.parties.iter().find(|p| p.id == *party_id) {
            let member_ids = party.member_ids.clone();
            let pos = party.position;

            // Check if any member was seeking founding (indicates this was a founding party).
            let is_founding = member_ids.iter().any(|mid| {
                state.adventurers.iter().find(|a| a.id == *mid)
                    .map(|a| matches!(&a.economic_intent,
                        EconomicIntent::Traveling | EconomicIntent::SeekingParty {
                            purpose: PartyPurpose::Founding { .. }
                        }))
                    .unwrap_or(false)
            });

            if is_founding && member_ids.len() >= 3 {
                new_settlements.push((*party_id, pos, member_ids.clone()));
            } else {
                // Normal relocation — assign to nearest settlement.
                for mid in &member_ids {
                    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *mid) {
                        adv.economic_intent = EconomicIntent::Working;
                    }
                }
            }
        }
    }

    // Create new settlements from founding parties.
    for (party_id, pos, member_ids) in &new_settlements {
        let new_loc_id = state.overworld.locations.iter().map(|l| l.id).max().unwrap_or(0) + 1;
        let settlement_name = format!("New Settlement {}", new_loc_id);

        // Determine threat from nearby regions (simplified — use average).
        let avg_threat = if state.overworld.regions.is_empty() {
            20.0
        } else {
            state.overworld.regions.iter().map(|r| r.threat_level).sum::<f32>()
                / state.overworld.regions.len() as f32
        };

        state.overworld.locations.push(crate::headless_campaign::state::Location {
            id: new_loc_id,
            name: settlement_name.clone(),
            position: *pos,
            location_type: crate::headless_campaign::state::LocationType::Settlement,
            threat_level: avg_threat,
            resource_availability: 50.0,
            faction_owner: Some(0), // guild faction
            scouted: true,
            resident_ids: member_ids.clone(),
            service_demand: [1.0; 8], // high initial demand — everything is needed
            cost_of_living: 1.0,      // cheap — brand new, no crowding
            safety_level: 0.0,        // will be computed next tick
            min_viable_threat: 0.0,
            treasury: 0.0,
            tax_rate: 0.10, // low tax to attract more settlers
            stockpile: {
                let pop = member_ids.len() as f32;
                crate::headless_campaign::state::CommodityStockpile {
                    food: 100.0 * pop,   // ~500 ticks of food per person at 0.2 consumption
                    iron: 10.0 * pop,
                    wood: 15.0 * pop,
                    herbs: 8.0 * pop,
                    hide: 5.0 * pop,
                    crystal: 3.0 * pop,
                    equipment: 1.0 * pop,
                    medicine: 5.0 * pop,
                }
            },
            local_prices: crate::headless_campaign::state::BASE_PRICES,
        });

        // Assign all founding members to the new settlement.
        for mid in member_ids {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *mid) {
                // Remove from old settlement's resident list.
                if let Some(old_loc_id) = adv.home_location_id {
                    if let Some(old_loc) = state.overworld.locations.iter_mut().find(|l| l.id == old_loc_id) {
                        old_loc.resident_ids.retain(|id| *id != adv.id);
                    }
                }
                adv.home_location_id = Some(new_loc_id);
                adv.economic_intent = EconomicIntent::Working;
                adv.stress = (adv.stress - 20.0).max(0.0); // relief from escaping overcrowding
            }
        }

        events.push(WorldEvent::NpcRelocating {
            adventurer_id: member_ids[0],
            from: None,
            to: new_loc_id,
        });
    }

    state.parties.retain(|p| !arrived_travel.contains(&p.id));
    state
        .parties
        .retain(|p| !arrived_travel.contains(&p.id));
}

/// Phase 5: Patronage contract management.
/// High-level NPCs offer contracts to recruit specialists. Handles offers,
/// acceptance, expiry, and affordability checks.
fn manage_patronage(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    use crate::headless_campaign::state::PatronageContract;

    let tick = state.tick as u32;

    // --- Phase A: Expire and enforce existing contracts ---
    let mut expired = Vec::new();
    for (i, contract) in state.patronage_contracts.iter().enumerate() {
        let elapsed = tick.saturating_sub(contract.started_tick);
        if elapsed >= contract.duration_ticks {
            expired.push(i);
            continue;
        }
        // Patron can't afford it anymore?
        if let Some(patron) = state.adventurers.iter().find(|a| a.id == contract.patron_id) {
            if patron.gold < contract.income_guarantee * 10.0 {
                // Running low — will expire soon naturally via bankruptcy.
                // Don't force-expire yet, let the expense system drain them.
            }
        }
    }
    // Remove expired contracts (reverse order to preserve indices).
    for &i in expired.iter().rev() {
        state.patronage_contracts.remove(i);
    }

    // --- Phase B: Apply contract benefits to clients ---
    for contract in &state.patronage_contracts {
        // Client gets income guarantee.
        if let Some(client) = state.adventurers.iter_mut().find(|a| a.id == contract.client_id) {
            // Income guarantee is a floor — if they're already earning more, no effect.
            // The guarantee is applied by ensuring gold doesn't drop below a threshold.
            if client.gold < 0.0 && contract.income_guarantee > 0.0 {
                client.gold += contract.income_guarantee;
            }
        }
        // Patron pays the cost.
        if let Some(patron) = state.adventurers.iter_mut().find(|a| a.id == contract.patron_id) {
            patron.gold -= contract.income_guarantee * 0.1; // Per-tick cost to patron
        }
    }

    // --- Phase C: New contract offers (every 5 decision intervals) ---
    let cfg = &state.config.npc_economy;
    if state.tick % (cfg.decision_interval_ticks * 5) != 0 {
        return;
    }

    // Find wealthy NPCs with leadership/authority potential who could be patrons.
    struct PatronCandidate {
        adv_idx: usize,
        adv_id: u32,
        loc_id: usize,
        gold: f32,
        leadership: f32,
    }
    let mut patron_candidates = Vec::new();
    for (idx, adv) in state.adventurers.iter().enumerate() {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        let stats = effective_noncombat_stats(adv);
        let leadership = stats.6; // leadership stat
        let has_authority = leadership > 3.0 || adv.gold > 200.0;
        if has_authority {
            if let Some(loc_id) = adv.home_location_id {
                patron_candidates.push(PatronCandidate {
                    adv_idx: idx,
                    adv_id: adv.id,
                    loc_id,
                    gold: adv.gold,
                    leadership,
                });
            }
        }
    }

    // For each patron, check if their settlement is missing a service.
    let loc_demands: Vec<(usize, [f32; 8])> = state
        .overworld
        .locations
        .iter()
        .map(|l| (l.id, l.service_demand))
        .collect();

    for candidate in &patron_candidates {
        // Don't offer if already patronizing too many.
        let existing_contracts = state
            .patronage_contracts
            .iter()
            .filter(|c| c.patron_id == candidate.adv_id)
            .count();
        if existing_contracts >= 3 {
            continue;
        }
        // Can't afford a new contract if gold < 100.
        if candidate.gold < 100.0 {
            continue;
        }

        let demand = loc_demands
            .iter()
            .find(|(id, _)| *id == candidate.loc_id)
            .map(|(_, d)| d);
        let demand = match demand {
            Some(d) => d,
            None => continue,
        };

        // Find the highest-demand channel.
        let max_demand_channel = demand
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, d)| (i, *d));

        if let Some((_, max_d)) = max_demand_channel {
            if max_d < 0.6 {
                continue; // Settlement is well-served enough.
            }

            // Look for a specialist at another settlement who could fill this gap.
            // For now, just create the contract if a suitable NPC exists at the same location
            // who isn't already under contract.
            let already_contracted: Vec<u32> = state
                .patronage_contracts
                .iter()
                .map(|c| c.client_id)
                .collect();

            // Find an NPC at this location who isn't the patron and isn't contracted.
            for adv in &state.adventurers {
                if adv.id == candidate.adv_id
                    || adv.status == AdventurerStatus::Dead
                    || already_contracted.contains(&adv.id)
                {
                    continue;
                }
                if adv.home_location_id != Some(candidate.loc_id) {
                    continue;
                }
                // Offer a patronage contract.
                let income_guarantee = 0.5; // modest guarantee
                state.patronage_contracts.push(PatronageContract {
                    patron_id: candidate.adv_id,
                    client_id: adv.id,
                    location_id: candidate.loc_id,
                    income_guarantee,
                    housing_provided: candidate.gold > 300.0,
                    duration_ticks: 1000,
                    started_tick: tick,
                });
                break; // One contract per patron per cycle.
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 7: Destabilizing forces
// ---------------------------------------------------------------------------

/// Check for combat NPCs who've failed economically and transition them to bandits.
/// Bandits increase regional threat, which disrupts trade and attracts bounty hunters.
fn check_bandit_transition(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Bandits: combat NPCs with gold < 0 for extended periods + low loyalty + high stress.
    let mut new_bandits = Vec::new();
    for (idx, adv) in state.adventurers.iter().enumerate() {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        let has_combat = adv.stats.attack > 10.0 || adv.stats.defense > 10.0;
        if !has_combat {
            continue;
        }
        // Economic failure: broke, stressed, disloyal.
        if adv.gold < -10.0 && adv.stress > 70.0 && adv.loyalty < 30.0 {
            // Small random chance per decision cycle to turn bandit.
            let roll = crate::headless_campaign::state::lcg_f32(&mut state.rng);
            if roll < 0.05 {
                new_bandits.push(idx);
            }
        }
    }

    for &idx in &new_bandits {
        let adv = &mut state.adventurers[idx];
        // Bandit NPC: leaves their settlement, increases regional threat.
        let old_home = adv.home_location_id;

        // Remove from settlement.
        if let Some(loc_id) = old_home {
            if let Some(loc) = state.overworld.locations.iter_mut().find(|l| l.id == loc_id) {
                loc.resident_ids.retain(|id| *id != adv.id);
            }
        }
        adv.home_location_id = None;
        adv.economic_intent = crate::headless_campaign::state::EconomicIntent::Idle;
        adv.loyalty = 0.0;

        // Increase regional threat (bandits ARE threat).
        // Find the region the NPC was in and increase unrest + threat.
        for region in &mut state.overworld.regions {
            // Simplified: affect all regions (bandits roam).
            region.unrest = (region.unrest + 2.0).min(100.0);
        }

        events.push(WorldEvent::AdventurerDeserted {
            adventurer_id: adv.id,
            reason: "Turned bandit due to economic failure".to_string(),
        });
    }
}

/// Death spiral floor: when combat NPCs are critically low, increase recruitment
/// of higher-level replacements to prevent permanent collapse.
fn check_death_spiral_floor(state: &mut CampaignState) {
    let cfg = &state.config.npc_economy;
    let region_count = state.overworld.regions.len().max(1);
    let min_combat = cfg.min_combat_npcs_per_region * region_count;

    let combat_count = state
        .adventurers
        .iter()
        .filter(|a| {
            a.status != AdventurerStatus::Dead
                && (a.stats.attack > 10.0 || a.stats.defense > 10.0)
        })
        .count();

    if combat_count < min_combat {
        // Increase recruitment pressure: bump reputation slightly and reduce
        // recruitment cost. This simulates external fighters being attracted
        // by the power vacuum.
        state.guild.reputation = (state.guild.reputation + 0.5).min(100.0);
        // Also add a small gold subsidy to enable recruitment.
        state.guild.gold += 5.0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal test adventurer (Adventurer doesn't derive Default).
    fn test_adventurer(id: u32, level: u32, gold: f32) -> crate::headless_campaign::state::Adventurer {
        crate::headless_campaign::state::Adventurer {
            id,
            name: format!("Test-{id}"),
            archetype: "knight".to_string(),
            level,
            xp: 0,
            stats: crate::headless_campaign::state::AdventurerStats {
                max_hp: 100.0 + level as f32 * 5.0,
                attack: 10.0 + level as f32 * 2.0,
                defense: 8.0 + level as f32 * 1.5,
                speed: 5.0,
                ability_power: 5.0,
            },
            equipment: Default::default(),
            traits: Vec::new(),
            status: AdventurerStatus::Idle,
            loyalty: 70.0,
            stress: 20.0,
            fatigue: 10.0,
            injury: 0.0,
            resolve: 50.0,
            morale: 60.0,
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
            disease_status: Default::default(),
            mood_state: Default::default(),
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
            behavior_ledger: Default::default(),
            classes: Vec::new(),
            skill_state: Default::default(),
            gold,
            home_location_id: None,
            economic_intent: Default::default(),
            ticks_since_income: 0,
        }
    }

    #[test]
    fn test_effective_level_basic() {
        let adv = test_adventurer(1, 20, 0.0);
        let eff = effective_level_with_config(&adv, 100.0);
        assert!((eff - 20.0).abs() < 0.01, "base level should be 20, got {eff}");
    }

    #[test]
    fn test_effective_level_with_wealth() {
        let adv = test_adventurer(1, 20, 400.0);
        let eff = effective_level_with_config(&adv, 100.0);
        // sqrt(400/100) = 2, capped at 20/2 = 10, so +2.
        assert!((eff - 22.0).abs() < 0.01, "expected 22, got {eff}");
    }

    #[test]
    fn test_effective_level_cap() {
        let adv = test_adventurer(1, 10, 100000.0);
        let eff = effective_level_with_config(&adv, 100.0);
        // sqrt(100000/100) = ~31.6, but cap = 10/2 = 5. So effective = 15.
        assert!((eff - 15.0).abs() < 0.01, "expected 15 (capped), got {eff}");
    }

    #[test]
    fn test_power_rating_quadratic() {
        let pr10 = power_rating(10.0);
        let pr40 = power_rating(40.0);
        assert!((pr10 - 100.0).abs() < 0.01);
        assert!((pr40 - 1600.0).abs() < 0.01);
        assert!(pr40 / pr10 > 15.0, "level 40 should be ~16x level 10");
    }

    #[test]
    fn test_adversity_multiplier_bounds() {
        let adv = test_adventurer(1, 10, 0.0);
        let low = adversity_multiplier(&adv, 0.0, &[0.0; 8]);
        assert!((low - 1.0).abs() < 0.01, "no adversity = 1.0x, got {low}");

        let mut adv_stressed = test_adventurer(1, 10, 0.0);
        adv_stressed.stress = 100.0;
        let demand = [1.0; 8];
        let high = adversity_multiplier(&adv_stressed, 100.0, &demand);
        assert!(high < 2.5, "adversity should cap around 2.0x, got {high}");
        assert!(high > 1.5, "high adversity should boost, got {high}");
    }

    #[test]
    fn test_income_scales_with_level() {
        let cfg = crate::headless_campaign::config::NpcEconomyConfig::default();
        let demand = [0.5; 8];

        let low_stats = [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let low_income = estimate_income(&low_stats, 0.0, 1.0, &demand, &cfg);

        let high_stats = [0.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let high_income = estimate_income(&high_stats, 0.0, 4.0, &demand, &cfg);

        assert!(
            high_income > low_income * 10.0,
            "level 40 should earn >>10x: low={low_income}, high={high_income}"
        );
    }

    #[test]
    fn test_hyperbolic_demand_no_cliff() {
        let base = 0.3_f32;
        let supply = 100.0_f32;
        let sat_rate = 0.02_f32;
        let demand = base / (base + supply * sat_rate);
        assert!(demand > 0.0, "demand should never cliff to 0, got {demand}");
        assert!(demand < 0.2, "high supply should reduce demand, got {demand}");
    }

    #[test]
    fn test_initialize_npc_locations() {
        let mut state = CampaignState::default_test_campaign(42);
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        // Before init, NPCs should have no home.
        assert!(state.adventurers.iter().all(|a| a.home_location_id.is_none()));

        // Simulate tick 1 to trigger initialization.
        state.tick = 1;
        tick_npc_economy(&mut state, &mut deltas, &mut events);

        // After init, alive NPCs should have homes.
        let alive_with_home = state
            .adventurers
            .iter()
            .filter(|a| a.status != AdventurerStatus::Dead && a.home_location_id.is_some())
            .count();
        let alive_total = state
            .adventurers
            .iter()
            .filter(|a| a.status != AdventurerStatus::Dead)
            .count();
        assert_eq!(
            alive_with_home, alive_total,
            "all alive NPCs should have homes after init"
        );

        // Locations should have residents.
        let total_residents: usize = state
            .overworld
            .locations
            .iter()
            .map(|l| l.resident_ids.len())
            .sum();
        assert!(total_residents > 0, "locations should have residents");
    }
}
