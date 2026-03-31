//! Spatial work state machine — universal work loop for NPC production.
//!
//! NPCs with a `work_building_id` cycle through:
//!   Idle -> TravelingToWork -> Working -> Idle
//!
//! When a work cycle completes, the produced commodity goes directly into the
//! worker NPC's inventory (they keep what they produce). Wages are paid from the
//! settlement treasury to the NPC at the same time.
//!
//! `compute_work` reads immutable `&WorldState` and emits movement/behavior deltas.
//! `advance_work_states` takes `&mut WorldState` and handles state transitions
//! (called post-apply in runtime.rs since it needs mutable access).

use std::collections::HashMap;
use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;
use crate::world_sim::commodity;
use super::resource_nodes::find_nearest_resource;

/// Base work ticks for farming before skill scaling.
const FARM_WORK_TICKS: u16 = 20;

/// Food produced per farming work cycle.
const FARM_OUTPUT_AMOUNT: f32 = 1.0;

/// Base wage paid per work cycle completion. Paid from settlement treasury to NPC.
const BASE_WAGE: f32 = 1.0;

/// Arrival distance threshold — NPC is "at" the target when closer than this.
const ARRIVAL_DIST: f32 = 3.0;

/// Compute movement and behavior-tag deltas for NPCs in the work loop.
///
/// This function does NOT mutate `WorkState` — that happens in `advance_work_states`.
/// It only emits `Move` deltas (for traveling/carrying) and `AddBehaviorTags`
/// (while working).
pub fn compute_work(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };
        // Only process NPCs with a work assignment.
        if npc.work_building_id.is_none() {
            continue;
        }

        match &npc.work_state {
            WorkState::Idle => {
                // Transitions handled in advance_work_states.
            }
            WorkState::TravelingToWork { .. } => {
                // Movement handled via entity.move_target (set in advance_work_states).
                // Arrival transition handled in advance_work_states.
            }
            WorkState::Working { building_id, ticks_remaining } => {
                if *ticks_remaining > 0 {
                    // Emit behavior tags based on building type.
                    let mut action = ActionTags::empty();
                    if let Some(work_entity) = state.entity(*building_id) {
                        if let Some(bld) = &work_entity.building {
                            match bld.building_type {
                                BuildingType::Farm => {
                                    action.add(tags::FARMING, 1.0);
                                    action.add(tags::LABOR, 0.5);
                                }
                                BuildingType::Mine => {
                                    action.add(tags::MINING, 1.0);
                                    action.add(tags::LABOR, 0.5);
                                }
                                BuildingType::Sawmill => {
                                    action.add(tags::WOODWORK, 1.0);
                                    action.add(tags::LABOR, 0.5);
                                }
                                BuildingType::Forge => {
                                    action.add(tags::SMITHING, 1.0);
                                    action.add(tags::CRAFTING, 0.5);
                                }
                                BuildingType::Apothecary => {
                                    action.add(tags::ALCHEMY, 1.0);
                                    action.add(tags::MEDICINE, 0.5);
                                }
                                _ => {
                                    action.add(tags::LABOR, 1.0);
                                }
                            }
                        }
                    }
                    if action.count > 0 {
                        out.push(WorldDelta::AddBehaviorTags {
                            entity_id: entity.id,
                            tags: action.tags,
                            count: action.count,
                        });
                    }
                }
                // Completion transition handled in advance_work_states.
            }
            WorkState::CarryingToStorage { .. } => {
                // Movement handled via entity.move_target (set in advance_work_states).
                // Deposit transition handled in advance_work_states.
            }
        }
    }
}

/// Advance NPC work state machine transitions.
///
/// Called post-apply in runtime.rs since it needs `&mut WorldState`.
/// Handles: Idle->Traveling, Traveling->Working (on arrival),
/// Working->Carrying (on completion), Carrying->Idle (on deposit).
pub fn advance_work_states(state: &mut WorldState) {
    // Pre-compute storage buildings per settlement for O(1) lookup.
    // Each entry: (building_id, pos, storage_capacity).
    let mut storage_by_settlement: HashMap<u32, Vec<(u32, (f32, f32), f32)>> = HashMap::new();
    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Building { continue; }
        let bd = match &entity.building { Some(b) => b, None => continue };
        if bd.construction_progress < 1.0 { continue; }
        if let Some(sid) = bd.settlement_id {
            let is_storage_type = matches!(bd.building_type,
                BuildingType::Warehouse | BuildingType::Inn | BuildingType::Market);
            if is_storage_type || bd.storage_capacity > 0.0 {
                storage_by_settlement.entry(sid)
                    .or_default()
                    .push((entity.id, entity.pos, bd.storage_capacity));
            }
        }
    }

    // Collect commodity production events to apply after the loop
    // (can't borrow settlements while iterating entities mutably).
    let mut deposits: Vec<(u32, usize, f32)> = Vec::new();
    // Collect deposits to building storage: (building_id, commodity, amount).
    let mut building_deposits: Vec<(u32, usize, f32)> = Vec::new();
    // Collect wage payments: (entity_idx, settlement_id, amount).
    let mut wages: Vec<(usize, u32, f32)> = Vec::new();
    // Collect item spawns from Forge production (applied after entity loop).
    let mut item_spawns: Vec<(Entity,)> = Vec::new();
    // Collect resource harvests: (resource_entity_index, amount_to_deduct).
    let mut resource_harvests: Vec<(usize, f32)> = Vec::new();
    // Collect esteem boosts for high-quality production.
    let mut esteem_boosts: Vec<usize> = Vec::new();

    // Pre-compute nearest resource per building position + type for O(1) lookup.
    // Key: (building_id), Value: (resource_entity_idx, distance_sq).
    // This avoids repeated linear scans of entities for each worker.
    let mut resource_cache: HashMap<u32, Option<usize>> = HashMap::new();

    let entity_count = state.entities.len();
    for i in 0..entity_count {
        let entity = &state.entities[i];
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }

        // Extract fields we need before taking mutable borrow.
        let entity_id = entity.id;
        let entity_pos = entity.pos;
        let (work_bid, work_state_clone, home_sid) = {
            let npc = match &entity.npc {
                Some(n) => n,
                None => continue,
            };
            let bid = match npc.work_building_id {
                Some(id) => id,
                None => continue,
            };
            (bid, npc.work_state.clone(), npc.home_settlement_id)
        };

        match work_state_clone {
            WorkState::Idle => {
                // Only start work during the "work" phase of the commute cycle.
                // Work phase: tick % 200 < 120 (60% of time working, 40% resting).
                if state.tick % 200 >= 120 { continue; }

                // Find work building position and start traveling.
                let target = state.entity(work_bid).map(|e| e.pos);
                if let Some(target_pos) = target {
                    state.entities[i].move_target = Some(target_pos);
                    let npc = state.entities[i].npc.as_mut().unwrap();
                    npc.work_state = WorkState::TravelingToWork { target_pos };
                }
            }
            WorkState::TravelingToWork { target_pos: _ } => {
                if state.entities[i].move_target.is_none() {
                    // Movement system cleared move_target — we've arrived.
                    let work_ticks = compute_work_ticks(state, work_bid, i);
                    let npc = state.entities[i].npc.as_mut().unwrap();
                    npc.work_state = WorkState::Working {
                        building_id: work_bid,
                        ticks_remaining: work_ticks,
                    };
                }
            }
            WorkState::Working { building_id, ticks_remaining } => {
                if ticks_remaining == 0 {
                    // Check if this is a Forge — Forge produces item entities, not commodities.
                    let is_forge = state.entity(building_id)
                        .and_then(|e| e.building.as_ref())
                        .map(|b| b.building_type == BuildingType::Forge)
                        .unwrap_or(false);

                    if is_forge {
                        // Forge production: spawn an item entity.
                        let smithing_skill = state.entities[i].npc.as_ref()
                            .map(|n| n.behavior_value(tags::SMITHING))
                            .unwrap_or(0.0);
                        let tick = state.tick;
                        let item_id = state.next_entity_id();
                        let item_entity = craft_item(
                            item_id, entity_id, entity_pos, smithing_skill,
                            home_sid, tick,
                        );
                        item_spawns.push((item_entity,));

                        // Pay wage for crafting.
                        if let Some(sid) = home_sid {
                            wages.push((i, sid, BASE_WAGE * 2.0)); // smithing pays more
                        }

                        let npc = state.entities[i].npc.as_mut().unwrap();
                        npc.work_state = WorkState::Idle;
                    } else {
                        // Non-forge production: commodity goes directly into NPC inventory.
                        let (commodity, base_amount) = output_for_building(state, building_id);

                        // Look up building type and required resource.
                        let btype = state.entity(building_id)
                            .and_then(|e| e.building.as_ref())
                            .map(|b| b.building_type);
                        let req_resource = btype.and_then(required_resource_type);

                        // Check if a matching resource node is nearby.
                        // Production fails (0 output) if the building requires a resource
                        // but none is available within range.
                        let resource_idx = if let Some(rtype) = req_resource {
                            let bld_pos = state.entity(building_id)
                                .map(|e| e.pos)
                                .unwrap_or(entity_pos);
                            let cached = resource_cache.entry(building_id).or_insert_with(|| {
                                find_nearest_resource(state, bld_pos, rtype, RESOURCE_SEARCH_RADIUS)
                                    .map(|(idx, _)| idx)
                            });
                            *cached
                        } else {
                            None // No resource requirement (e.g. fallback buildings).
                        };

                        // If a resource is required but none found, production fails.
                        let production_allowed = req_resource.is_none() || resource_idx.is_some();

                        if production_allowed {
                            // Apply building specialization bonus when worker's primary
                            // class matches the building's specialization tag.
                            let mut amount = {
                                let spec = state.entity(building_id)
                                    .and_then(|e| e.building.as_ref())
                                    .and_then(|b| {
                                        b.specialization_tag.map(|tag| (tag, b.specialization_strength))
                                    });
                                if let Some((spec_tag, spec_str)) = spec {
                                    let worker_primary = state.entities[i].npc.as_ref()
                                        .and_then(|n| n.classes.iter().max_by_key(|c| c.level))
                                        .map(|c| c.class_name_hash);
                                    if worker_primary == Some(spec_tag) {
                                        base_amount * (1.0 + spec_str * 0.5)
                                    } else {
                                        base_amount
                                    }
                                } else {
                                    base_amount
                                }
                            };

                            // Class-level production quality: best matching class ×
                            // level × focus. Higher-level workers in relevant classes
                            // produce more, modulated by need satisfaction.
                            let building_tags = state.entity(building_id)
                                .and_then(|e| e.building.as_ref())
                                .map(|b| b.building_type.production_tags())
                                .unwrap_or(&[]);
                            if let Some(npc) = state.entities[i].npc.as_ref() {
                                let mut best_effective_level = 0.0f32;
                                for class in &npc.classes {
                                    let rel = crate::world_sim::class_gen::class_building_relevance(
                                        class.class_name_hash, building_tags,
                                    );
                                    if rel > 0.1 {
                                        let eff = class.level as f32 * rel;
                                        if eff > best_effective_level {
                                            best_effective_level = eff;
                                        }
                                    }
                                }
                                let focus = npc.focus();
                                let quality_mult = (1.0 + best_effective_level * 0.03) * focus;
                                amount *= quality_mult;

                                // Esteem boost on high-quality production.
                                if quality_mult > 1.3 {
                                    esteem_boosts.push(i);
                                }

                                // Passive ability production bonus (Phase F).
                                amount *= npc.passive_effects.production_mult;
                            }

                            // Deduct from the resource node.
                            if let Some(ridx) = resource_idx {
                                resource_harvests.push((ridx, amount));
                            }

                            // Deposit commodity directly into worker's inventory.
                            if let Some(inv) = &mut state.entities[i].inventory {
                                inv.deposit(commodity, amount);
                            }

                            // Update price belief from direct production (Phase E).
                            if commodity < crate::world_sim::NUM_COMMODITIES {
                                if let Some(npc) = &mut state.entities[i].npc {
                                    // Value based on scarcity: less produced → higher perceived value.
                                    let scarcity_value = 1.0 / (amount + 0.1);
                                    npc.price_beliefs[commodity].update_direct(scarcity_value, state.tick);
                                }
                            }

                            // Pay wage for completed work cycle.
                            if let Some(sid) = home_sid {
                                wages.push((i, sid, BASE_WAGE));
                            }
                        }
                        // If production failed (no resource), NPC still returns to idle
                        // but produces nothing and earns no wage.

                        let npc = state.entities[i].npc.as_mut().unwrap();
                        npc.work_state = WorkState::Idle;
                    }
                } else {
                    // Decrement ticks remaining.
                    let npc = state.entities[i].npc.as_mut().unwrap();
                    npc.work_state = WorkState::Working {
                        building_id,
                        ticks_remaining: ticks_remaining - 1,
                    };
                }
            }
            WorkState::CarryingToStorage { commodity, amount, target_pos: _ } => {
                if state.entities[i].move_target.is_none() {
                    // Arrived at storage — deposit into nearest storage building.
                    let storage_bid = home_sid
                        .and_then(|sid| storage_by_settlement.get(&sid))
                        .and_then(|buildings| {
                            buildings.iter()
                                .filter(|b| b.2 > 0.0) // has storage capacity
                                .min_by(|a, b| {
                                    let da = (a.1.0 - entity_pos.0).powi(2) + (a.1.1 - entity_pos.1).powi(2);
                                    let db = (b.1.0 - entity_pos.0).powi(2) + (b.1.1 - entity_pos.1).powi(2);
                                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|b| b.0)
                        });
                    if let Some(bid) = storage_bid {
                        building_deposits.push((bid, commodity as usize, amount));
                    } else {
                        let deposit_sid = home_sid.unwrap_or(0);
                        deposits.push((deposit_sid, commodity as usize, amount));
                    }

                    // Pay wage for completed delivery.
                    if let Some(sid) = home_sid {
                        wages.push((i, sid, BASE_WAGE));
                    }

                    let npc = state.entities[i].npc.as_mut().unwrap();
                    npc.work_state = WorkState::Idle;
                }
            }
        }
    }

    // Apply commodity deposits to settlement stockpiles.
    for (settlement_id, commodity, amount) in deposits {
        if let Some(settlement) = state.settlement_mut(settlement_id) {
            if commodity < settlement.stockpile.len() {
                settlement.stockpile[commodity] += amount;
            }
        }
    }

    // Pay wages: settlement treasury → NPC gold.
    for (entity_idx, settlement_id, wage) in wages {
        let treasury = state.settlement(settlement_id)
            .map(|s| s.treasury).unwrap_or(0.0);
        // Only pay if settlement can afford it.
        let paid = wage.min(treasury.max(0.0));
        if paid > 0.0 {
            if let Some(settlement) = state.settlement_mut(settlement_id) {
                settlement.treasury -= paid;
            }
            if let Some(npc) = state.entities[entity_idx].npc.as_mut() {
                npc.gold = (npc.gold + paid).min(10000.0); // cap NPC gold
                // Update rolling average income rate.
                npc.income_rate = npc.income_rate * 0.9 + paid * 0.1;
            }
        }
    }

    // Apply deposits to building storage.
    for (building_id, commodity, amount) in building_deposits {
        if let Some(bld_entity) = state.entity_mut(building_id) {
            if let Some(bld) = &mut bld_entity.building {
                bld.deposit(commodity, amount);
            }
        }
    }

    // Spawn crafted item entities.
    for (item,) in item_spawns {
        state.entities.push(item);
    }

    // Deduct harvested amounts from resource nodes.
    for (resource_idx, amount) in resource_harvests {
        if resource_idx < state.entities.len() {
            if let Some(res) = &mut state.entities[resource_idx].resource {
                res.remaining = (res.remaining - amount).max(0.0);
            }
        }
    }

    // Apply esteem boosts for high-quality production.
    for entity_idx in esteem_boosts {
        if let Some(npc) = &mut state.entities[entity_idx].npc {
            npc.needs.esteem = (npc.needs.esteem + 1.0).min(100.0);
        }
    }
}

/// Craft an item entity from Forge production.
/// Quality and rarity based on crafter's SMITHING skill.
/// Slot determined by deterministic hash from tick + crafter ID.
fn craft_item(
    item_id: u32,
    crafter_id: u32,
    pos: (f32, f32),
    smithing_skill: f32,
    settlement_id: Option<u32>,
    tick: u64,
) -> Entity {
    let next_id = item_id;

    // Determine slot from hash.
    let h = entity_hash(crafter_id, tick, 0xC8AF);
    let slot = match h % 3 {
        0 => ItemSlot::Weapon,
        1 => ItemSlot::Armor,
        _ => ItemSlot::Accessory,
    };

    // Quality from skill (base 1.0 + skill contribution, capped at 20).
    let quality = (1.0 + smithing_skill * 0.005).min(20.0);
    let rarity = ItemRarity::from_skill(smithing_skill);

    // Generate name.
    let name = generate_item_name(slot, rarity, h as u64);

    Entity::new_item(next_id, pos, ItemData {
        slot,
        rarity,
        quality,
        durability: 100.0,
        max_durability: 100.0,
        owner_id: None,
        settlement_id,
        name,
        crafter_id: Some(crafter_id),
        crafted_tick: tick,
        history: vec![ItemEvent {
            tick,
            kind: ItemEventKind::Crafted { crafter_name: format!("Crafter #{}", crafter_id) },
        }],
        is_legendary: false,
        is_relic: false,
        relic_bonus: None,
    })
}

/// Generate a simple item name based on slot, rarity, and hash.
fn generate_item_name(slot: ItemSlot, rarity: ItemRarity, h: u64) -> String {
    let prefix = match rarity {
        ItemRarity::Common => "",
        ItemRarity::Uncommon => "Fine ",
        ItemRarity::Rare => "Superior ",
        ItemRarity::Epic => "Masterwork ",
        ItemRarity::Legendary => "Legendary ",
    };

    let base = match slot {
        ItemSlot::Weapon => {
            const WEAPONS: [&str; 6] = ["Sword", "Axe", "Mace", "Spear", "Dagger", "Halberd"];
            WEAPONS[(h >> 40) as usize % WEAPONS.len()]
        }
        ItemSlot::Armor => {
            const ARMORS: [&str; 5] = ["Chainmail", "Platemail", "Leather Armor", "Brigandine", "Scale Armor"];
            ARMORS[(h >> 40) as usize % ARMORS.len()]
        }
        ItemSlot::Accessory => {
            const ACCESSORIES: [&str; 5] = ["Boots", "Cloak", "Ring", "Amulet", "Belt"];
            ACCESSORIES[(h >> 40) as usize % ACCESSORIES.len()]
        }
    };

    let material = {
        const MATERIALS: [&str; 5] = ["Iron", "Steel", "Mithril", "Adamant", "Orichalcum"];
        let idx = match rarity {
            ItemRarity::Common => 0,
            ItemRarity::Uncommon => 1,
            ItemRarity::Rare => 2,
            ItemRarity::Epic => 3,
            ItemRarity::Legendary => 4,
        };
        MATERIALS[idx]
    };

    format!("{}{} {}", prefix, material, base)
}

/// Compute skill-scaled work ticks for an NPC at a given building.
fn compute_work_ticks(state: &WorldState, building_id: u32, entity_idx: usize) -> u16 {
    let building_type = state.entity(building_id)
        .and_then(|e| e.building.as_ref())
        .map(|b| b.building_type);

    let base_ticks = match building_type {
        Some(BuildingType::Farm) => FARM_WORK_TICKS,
        Some(BuildingType::Mine) => 30,
        Some(BuildingType::Sawmill) => 25,
        Some(BuildingType::Forge) => 35,
        Some(BuildingType::Apothecary) => 30,
        _ => 20,
    };

    // Skill scaling: effective_ticks = base / (1.0 + skill_value * 0.01)
    let skill_tag = match building_type {
        Some(BuildingType::Farm) => tags::FARMING,
        Some(BuildingType::Mine) => tags::MINING,
        Some(BuildingType::Sawmill) => tags::WOODWORK,
        Some(BuildingType::Forge) => tags::SMITHING,
        Some(BuildingType::Apothecary) => tags::ALCHEMY,
        _ => tags::LABOR,
    };

    let skill_value = state.entities[entity_idx]
        .npc.as_ref()
        .map(|n| n.behavior_value(skill_tag))
        .unwrap_or(0.0);

    let scaled = (base_ticks as f32) / (1.0 + skill_value * 0.01);
    (scaled as u16).max(1)
}

/// Determine the output commodity and amount for a building's production.
fn output_for_building(state: &WorldState, building_id: u32) -> (usize, f32) {
    let building_type = state.entity(building_id)
        .and_then(|e| e.building.as_ref())
        .map(|b| b.building_type);

    match building_type {
        Some(BuildingType::Farm) => (commodity::FOOD, FARM_OUTPUT_AMOUNT),
        Some(BuildingType::Mine) => (commodity::IRON, 1.0),
        Some(BuildingType::Sawmill) => (commodity::WOOD, 1.0),
        Some(BuildingType::Forge) => (commodity::EQUIPMENT, 1.0),
        Some(BuildingType::Apothecary) => (commodity::MEDICINE, 1.0),
        _ => (commodity::FOOD, 0.5), // fallback
    }
}

/// Map building type to the resource type it harvests from.
/// Returns `None` for buildings that don't harvest physical resources (e.g. Forge).
fn required_resource_type(building_type: BuildingType) -> Option<ResourceType> {
    match building_type {
        BuildingType::Farm => Some(ResourceType::BerryBush),
        BuildingType::Mine => Some(ResourceType::OreVein),
        BuildingType::Sawmill => Some(ResourceType::Tree),
        BuildingType::Apothecary => Some(ResourceType::HerbPatch),
        _ => None,
    }
}

/// Maximum distance (world units) to search for a matching resource node.
const RESOURCE_SEARCH_RADIUS: f32 = 50.0;

// ---------------------------------------------------------------------------
// Physical eating — NPCs seek food when hungry
// ---------------------------------------------------------------------------

/// How much hunger is replenished per meal.
const MEAL_HUNGER_RESTORE: f32 = 60.0;

/// Food consumed from settlement stockpile per meal.
const FOOD_PER_MEAL: f32 = 0.1;

/// Gold cost per meal. Paid by NPC to settlement treasury.
const MEAL_GOLD_COST: f32 = 0.5;

/// Hunger threshold below which NPCs interrupt work to eat.
const HUNGER_THRESHOLD: f32 = 70.0;

/// Process NPC eating. Called post-apply from runtime.
///
/// NPCs with hunger below threshold walk toward the nearest Inn or Market
/// building at their settlement (falling back to settlement center if none
/// exists). On arrival, consume food from stockpile and replenish hunger.
/// If no food available, record Starved event.
pub fn advance_eating(state: &mut WorldState) {
    // Collect settlement food availability.
    let settlement_food: Vec<(u32, f32, (f32, f32))> = state.settlements.iter()
        .map(|s| (s.id, s.stockpile[commodity::FOOD], s.pos))
        .collect();

    // Eating: NPCs consume food from their own inventory.
    // If they have food → eat immediately (no travel).
    // If no food but have gold + nearby Inn/Market → buy then eat.
    // If nothing → goal_eval handles pushing Gather(FOOD) goals.
    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &mut entity.npc { Some(n) => n, None => continue };

        if npc.needs.hunger >= HUNGER_THRESHOLD { continue; }
        if matches!(npc.economic_intent, EconomicIntent::Adventuring { .. }) { continue; }

        // Option 1: Eat from own inventory
        let has_food = entity.inventory.as_ref()
            .map(|inv| inv.commodities[commodity::FOOD] >= FOOD_PER_MEAL)
            .unwrap_or(false);

        if has_food {
            if let Some(inv) = &mut entity.inventory {
                inv.commodities[commodity::FOOD] -= FOOD_PER_MEAL;
            }
            npc.needs.hunger = (npc.needs.hunger + MEAL_HUNGER_RESTORE).min(100.0);
            continue;
        }

        // Option 2: Buy food from settlement (if gold available and settlement has food)
        let sid = match npc.home_settlement_id { Some(id) => id, None => continue };
        let settlement_has_food = settlement_food.iter()
            .find(|(id, _, _)| *id == sid)
            .map(|(_, food, _)| *food >= FOOD_PER_MEAL)
            .unwrap_or(false);

        if settlement_has_food && npc.gold >= MEAL_GOLD_COST {
            npc.gold -= MEAL_GOLD_COST;
            // Deduct from settlement stockpile, add gold to treasury
            let si = sid as usize;
            if si < state.settlement_index.len() {
                let idx = state.settlement_index[si] as usize;
                if idx < state.settlements.len() {
                    state.settlements[idx].stockpile[commodity::FOOD] -= FOOD_PER_MEAL;
                    state.settlements[idx].treasury += MEAL_GOLD_COST;
                }
            }
            // Food goes into inventory then consumed immediately
            npc.needs.hunger = (npc.needs.hunger + MEAL_HUNGER_RESTORE).min(100.0);
            continue;
        }

        // No food and no gold — goal_eval will push Gather(FOOD) goal
    }
}

/// Sync settlement stockpiles ↔ treasury building inventories.
///
/// Called post-apply from runtime. The treasury building's inventory is the
/// physical source of truth. Settlement.stockpile/treasury are caches for
/// fast access by abstract systems.
///
/// Two-way sync:
/// 1. If abstract systems changed settlement.stockpile (delta-based), push
///    the delta into the treasury building's inventory.
/// 2. Copy treasury inventory → settlement.stockpile/treasury for cache.
pub fn sync_stockpiles_from_buildings(state: &mut WorldState) {
    // Only sync every 5 ticks.
    if state.tick % 5 != 0 { return; }

    // Collect (settlement_idx, treasury_entity_id, old_stockpile, old_treasury_gold).
    let sync_targets: Vec<(usize, u32, [f32; 8], f32)> = state.settlements.iter()
        .enumerate()
        .filter_map(|(si, s)| {
            s.treasury_building_id.map(|tid| (si, tid, s.stockpile, s.treasury))
        })
        .collect();

    for (si, tid, old_stockpile, old_gold) in &sync_targets {
        let eidx = state.entity_idx(*tid);
        if let Some(idx) = eidx {
            if let Some(inv) = &mut state.entities[idx].inventory {
                // Push any deltas from abstract settlement changes into treasury inventory.
                let current_stockpile = state.settlements[*si].stockpile;
                for c in 0..8 {
                    let diff = current_stockpile[c] - old_stockpile[c];
                    if diff > 0.0 {
                        inv.deposit(c, diff);
                    } else if diff < 0.0 {
                        inv.withdraw(c, -diff);
                    }
                }
                // Same for gold.
                let gold_diff = state.settlements[*si].treasury - old_gold;
                if gold_diff > 0.0 {
                    inv.gold += gold_diff;
                } else if gold_diff < 0.0 {
                    inv.gold = (inv.gold + gold_diff).max(0.0);
                }
            }
        }

        // Now copy treasury inventory back to settlement cache.
        if let Some(idx) = eidx {
            if let Some(inv) = &state.entities[idx].inventory {
                state.settlements[*si].stockpile = inv.commodities;
                state.settlements[*si].treasury = inv.gold;
            }
        }
    }
}
