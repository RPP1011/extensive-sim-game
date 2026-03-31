//! Spatial work state machine — universal work loop for NPC production.
//!
//! NPCs with a `work_building_id` cycle through:
//!   Idle -> TravelingToWork -> Working -> CarryingToStorage -> Idle
//!
//! `compute_work` reads immutable `&WorldState` and emits movement/behavior deltas.
//! `advance_work_states` takes `&mut WorldState` and handles state transitions
//! (called post-apply in runtime.rs since it needs mutable access).

use std::collections::HashMap;
use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;
use crate::world_sim::commodity;
use crate::world_sim::DT_SEC;

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
            WorkState::TravelingToWork { target_pos } => {
                let dx = target_pos.0 - entity.pos.0;
                let dy = target_pos.1 - entity.pos.1;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist >= ARRIVAL_DIST {
                    // Move toward work building.
                    let speed = entity.move_speed * DT_SEC;
                    out.push(WorldDelta::Move {
                        entity_id: entity.id,
                        force: (dx / dist * speed, dy / dist * speed),
                    });
                }
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
            WorkState::CarryingToStorage { commodity: _, amount: _, target_pos } => {
                let dx = target_pos.0 - entity.pos.0;
                let dy = target_pos.1 - entity.pos.1;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist >= ARRIVAL_DIST {
                    // Move toward storage (home settlement).
                    let speed = entity.move_speed * DT_SEC;
                    out.push(WorldDelta::Move {
                        entity_id: entity.id,
                        force: (dx / dist * speed, dy / dist * speed),
                    });
                }
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
                    let npc = state.entities[i].npc.as_mut().unwrap();
                    npc.work_state = WorkState::TravelingToWork { target_pos };
                }
            }
            WorkState::TravelingToWork { target_pos } => {
                let dx = target_pos.0 - entity_pos.0;
                let dy = target_pos.1 - entity_pos.1;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < ARRIVAL_DIST {
                    // Arrived — look up building type to determine work ticks.
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
                        // Non-forge production: commodity output.
                        let (commodity, base_amount) = output_for_building(state, building_id);

                        // Apply building specialization bonus when worker's primary
                        // class matches the building's specialization tag.
                        let amount = {
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

                        let storage_pos = home_sid
                            .and_then(|sid| storage_by_settlement.get(&sid))
                            .and_then(|buildings| {
                                buildings.iter()
                                    .min_by(|a, b| {
                                        let da = (a.1.0 - entity_pos.0).powi(2) + (a.1.1 - entity_pos.1).powi(2);
                                        let db = (b.1.0 - entity_pos.0).powi(2) + (b.1.1 - entity_pos.1).powi(2);
                                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                    .map(|b| b.1)
                            })
                            .or_else(|| {
                                home_sid
                                    .and_then(|sid| state.settlement(sid))
                                    .map(|s| s.pos)
                            });

                        let fallback_sid = state.entity(building_id)
                            .and_then(|e| e.building.as_ref())
                            .and_then(|b| b.settlement_id)
                            .or(home_sid)
                            .unwrap_or(0);

                        let npc = state.entities[i].npc.as_mut().unwrap();
                        if let Some(target_pos) = storage_pos {
                            npc.work_state = WorkState::CarryingToStorage {
                                commodity: commodity as u8,
                                amount,
                                target_pos,
                            };
                        } else {
                            deposits.push((fallback_sid, commodity, amount));
                            npc.work_state = WorkState::Idle;
                        }
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
            WorkState::CarryingToStorage { commodity, amount, target_pos } => {
                let dx = target_pos.0 - entity_pos.0;
                let dy = target_pos.1 - entity_pos.1;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < ARRIVAL_DIST {
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
const HUNGER_THRESHOLD: f32 = 30.0;

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

    // Pre-compute nearest food building (Inn/Market with food) per settlement.
    // Map: settlement_id -> Option<(entity_id, pos)>.
    let food_building_info: Vec<(u32, Option<(u32, (f32, f32))>)> = state.settlements.iter()
        .map(|s| {
            let food_bld = state.entities.iter()
                .filter(|e| {
                    e.alive && e.kind == EntityKind::Building
                        && e.building.as_ref().map_or(false, |b| {
                            b.settlement_id == Some(s.id)
                                && b.construction_progress >= 1.0
                                && matches!(b.building_type, BuildingType::Inn | BuildingType::Market)
                        })
                })
                .map(|e| (e.id, e.pos))
                .next();
            (s.id, food_bld)
        })
        .collect();

    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &mut entity.npc { Some(n) => n, None => continue };

        // Only eat when hungry enough and not in combat.
        if npc.needs.hunger >= HUNGER_THRESHOLD { continue; }
        // Don't interrupt combat, but DO interrupt work when critically hungry.
        if matches!(npc.economic_intent, EconomicIntent::Adventuring { .. }) { continue; }
        // If working but critically hungry (<15), interrupt to eat.
        if !matches!(npc.work_state, WorkState::Idle) && npc.needs.hunger > 15.0 { continue; }

        // Find food source — settlement with food.
        let sid = match npc.home_settlement_id { Some(id) => id, None => continue };
        let (_, food_available, settlement_pos) = match settlement_food.iter()
            .find(|(id, _, _)| *id == sid) {
            Some(s) => *s,
            None => continue,
        };

        // Walk toward the nearest food building, falling back to settlement center.
        let food_info = food_building_info.iter()
            .find(|(id, _)| *id == sid)
            .and_then(|(_, info)| *info);
        let food_target = food_info.map(|(_, pos)| pos).unwrap_or(settlement_pos);
        let _food_bid = food_info.map(|(bid, _)| bid);

        let dx = food_target.0 - entity.pos.0;
        let dy = food_target.1 - entity.pos.1;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist > 5.0 {
            // Move toward food. Use a moderate speed.
            let speed = entity.move_speed * DT_SEC * 0.5;
            entity.pos.0 += dx / dist * speed;
            entity.pos.1 += dy / dist * speed;
            continue;
        }

        // Arrived at food source — try to eat.
        if food_available >= FOOD_PER_MEAL {
            // Pay for the meal: NPC gold → settlement treasury.
            let can_pay = npc.gold >= MEAL_GOLD_COST;
            let cost = if can_pay { MEAL_GOLD_COST } else { npc.gold.max(0.0) }; // pay what you can
            npc.gold -= cost;

            // Consume food from settlement stockpile.
            let si = sid as usize;
            if si < state.settlement_index.len() {
                let idx = state.settlement_index[si] as usize;
                if idx < state.settlements.len() {
                    state.settlements[idx].stockpile[commodity::FOOD] -= FOOD_PER_MEAL;
                    state.settlements[idx].treasury += cost;
                }
            }
            npc.needs.hunger = (npc.needs.hunger + MEAL_HUNGER_RESTORE).min(100.0);
        }
        // If no food: hunger stays low, anxiety builds (handled in agent_inner drift).
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
