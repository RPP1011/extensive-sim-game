//! NPC-driven building system.
//!
//! CityGrid-based growth has been replaced by the VoxelWorld/tile system.
//! This module now handles:
//!   - `compute_buildings()` — delta-based treasury upgrades
//!   - `grow_cities()` — stub (specialization updates only)
//!   - `process_npc_builds()` — NPC-driven building placement via VoxelWorld
//!   - `assign_npcs_to_buildings()` — assign NPCs to buildings
//!   - `update_building_specializations()` — worker class emergence
//!   - `advance_construction()` — construction progress per tick

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{WorldState, Entity, EntityKind, EconomicIntent, ChronicleEntry, ChronicleCategory, tags, BuildingType, BuildingData, ActionTags, WorkState, MemoryEvent, MemEventType, GoalKind, entity_hash};
use crate::world_sim::NUM_COMMODITIES;

/// Building tick interval for delta-based compute (treasury upgrades).
const BUILDING_TICK_INTERVAL: u64 = 3;

/// CA growth tick interval.
const GROWTH_TICK_INTERVAL: u64 = 10;

/// Minimum treasury to trigger an auto-upgrade.
const UPGRADE_TREASURY_THRESHOLD: f32 = 200.0;

/// Cost of an upgrade (deducted from treasury).
const UPGRADE_COST: f32 = 100.0;

/// Minimum accumulated worker-class ticks before specialization is considered.
const SPECIALIZATION_THRESHOLD: u64 = 500;

/// Local zone type classification for building–NPC matching.
/// Mirrors the old CityGrid ZoneType but lives entirely in this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZoneType {
    None,
    Residential,
    Commercial,
    Industrial,
    Religious,
    Arcane,
    Noble,
    Military,
}

/// Map building type to zone type for work-assignment matching.
fn building_type_to_zone(bt: BuildingType) -> ZoneType {
    match bt {
        BuildingType::House | BuildingType::Longhouse | BuildingType::Manor => ZoneType::Residential,
        BuildingType::Market | BuildingType::Warehouse | BuildingType::TradePost | BuildingType::Inn => ZoneType::Commercial,
        BuildingType::Farm | BuildingType::Mine | BuildingType::Sawmill | BuildingType::Forge
        | BuildingType::Workshop | BuildingType::Apothecary => ZoneType::Industrial,
        BuildingType::Barracks | BuildingType::Watchtower | BuildingType::Wall | BuildingType::Gate => ZoneType::Military,
        BuildingType::Temple | BuildingType::Shrine => ZoneType::Religious,
        BuildingType::GuildHall | BuildingType::CourtHouse | BuildingType::Treasury | BuildingType::Library => ZoneType::Noble,
        _ => ZoneType::Residential,
    }
}

// ---------------------------------------------------------------------------
// Delta-based compute (unchanged from original)
// ---------------------------------------------------------------------------

pub fn compute_buildings(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % BUILDING_TICK_INTERVAL != 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_buildings_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_buildings_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    _entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % BUILDING_TICK_INTERVAL != 0 {
        return;
    }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    if settlement.treasury >= UPGRADE_TREASURY_THRESHOLD && settlement.treasury > 0.0 {
        out.push(WorldDelta::UpdateTreasury {
            settlement_id: settlement_id,
            delta: -UPGRADE_COST,
        });
    }
}

// ---------------------------------------------------------------------------
// CA growth loop — stub (CityGrid removed)
// ---------------------------------------------------------------------------

/// Entry point called from the runtime post-apply phase.
/// CityGrid-based growth has been removed. This stub drives construction
/// advancement, NPC assignments, and specialization updates.
pub fn grow_cities(state: &mut WorldState) {
    if state.tick % GROWTH_TICK_INTERVAL != 0 {
        return;
    }

    // Cap chronicle length.
    const MAX_CHRONICLE: usize = 2000;
    if state.chronicle.len() > MAX_CHRONICLE {
        let drain = state.chronicle.len() - MAX_CHRONICLE;
        state.chronicle.drain(..drain);
    }

    // Advance construction on incomplete buildings (builders do physical work).
    advance_construction(state);

    // Assign unhoused/unassigned NPCs to buildings.
    assign_npcs_to_buildings(state);

    // NPC-driven building: process NPCs with Build goals that have resources.
    process_npc_builds(state);
}

/// NPC-driven building placement. Called each grow_cities tick.
/// NPCs with GoalKind::Build and sufficient resources in their inventory
/// place a building shell at a world-space position near the settlement.
pub fn process_npc_builds(state: &mut WorldState) {
    use crate::world_sim::commodity;
    use super::super::interior_gen::footprint_size;

    // Collect NPCs wanting to build: (entity_idx, settlement_id, building_type)
    let mut build_requests: Vec<(usize, u32, BuildingType)> = Vec::new();

    for (i, entity) in state.entities.iter().enumerate() {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };
        let sid = match npc.home_settlement_id { Some(s) => s, None => continue };

        // Check if NPC has an active Build goal
        let build_type = npc.goal_stack.goals.iter().find_map(|g| {
            match g.kind {
                GoalKind::Build { .. } => Some(BuildingType::House), // default to House for shelter goal
                _ => None,
            }
        });
        let building_type = match build_type { Some(bt) => bt, None => continue };

        // Check if NPC has resources in their inventory
        let inv = match &entity.inventory { Some(inv) => inv, None => continue };
        let (wood_cost, iron_cost) = building_type.build_cost();
        if inv.commodities[commodity::WOOD] < wood_cost { continue; }
        if inv.commodities[commodity::IRON] < iron_cost { continue; }

        build_requests.push((i, sid, building_type));
    }

    if build_requests.is_empty() { return; }

    let tick = state.tick;
    let mut new_entities: Vec<Entity> = Vec::new();

    for (entity_idx, settlement_id, building_type) in build_requests {
        // Get settlement position.
        let settlement_pos = match state.settlement(settlement_id) {
            Some(s) => s.pos,
            None => continue,
        };

        // Pick a world-space position for the building.
        // Use the NPC's current position as the build site (they go to a spot and build).
        let npc_pos = state.entities[entity_idx].pos;
        let world_pos = npc_pos;

        // Determine grid_col/grid_row as integer offsets from settlement center (2 units/cell).
        let cell_size = 2.0_f32;
        let col = ((world_pos.0 - settlement_pos.0) / cell_size).round() as i32 + 32;
        let row = ((world_pos.1 - settlement_pos.1) / cell_size).round() as i32 + 32;
        let col = col.max(0) as usize;
        let row = row.max(0) as usize;

        let (fp_w, fp_h) = footprint_size(building_type, 0);

        // Deduct resources from NPC inventory
        if let Some(inv) = &mut state.entities[entity_idx].inventory {
            let (wood_cost, iron_cost) = building_type.build_cost();
            inv.commodities[commodity::WOOD] -= wood_cost;
            inv.commodities[commodity::IRON] -= iron_cost;
        }

        // Spawn building shell
        state.sync_next_id();
        let new_id = state.next_entity_id();

        let mut entity = Entity::new_building(new_id, world_pos);
        entity.building = Some(BuildingData {
            building_type,
            settlement_id: Some(settlement_id),
            grid_col: col as u16,
            grid_row: row as u16,
            footprint_w: fp_w as u8,
            footprint_h: fp_h as u8,
            tier: 0,
            room_seed: entity_hash(new_id, tick, 0x800E) as u64,
            rooms: building_type.default_rooms(),
            residential_capacity: building_type.residential_capacity(),
            work_capacity: building_type.work_capacity(),
            resident_ids: Vec::new(),
            worker_ids: Vec::new(),
            construction_progress: 0.0,
            built_tick: tick,
            builder_id: Some(state.entities[entity_idx].id),
            temporary: false,
            ttl_ticks: None,
            name: generate_building_name(building_type, new_id),
            storage: [0.0; NUM_COMMODITIES],
            storage_capacity: building_type.storage_capacity(),
            owner_id: Some(state.entities[entity_idx].id),
            builder_modifiers: Vec::new(),
            owner_modifiers: Vec::new(),
            worker_class_ticks: Vec::new(),
            specialization_tag: None,
            specialization_strength: 0.0,
            specialization_name: String::new(),
            structural: None,
        });

        new_entities.push(entity);

        // Pop the Build goal from the NPC's stack
        if let Some(npc) = &mut state.entities[entity_idx].npc {
            npc.goal_stack.goals.retain(|g| !matches!(g.kind, GoalKind::Build { .. }));
        }

        // Chronicle
        let npc_name = state.entities[entity_idx].npc.as_ref()
            .map(|n| n.name.clone()).unwrap_or_default();
        state.chronicle.push(ChronicleEntry {
            tick,
            category: ChronicleCategory::Economy,
            text: format!("{} began building a {:?}", npc_name, building_type),
            entity_ids: vec![state.entities[entity_idx].id, new_id],
        });
    }

    if !new_entities.is_empty() {
        for e in new_entities {
            state.entities.push(e);
        }
        state.rebuild_entity_cache();
    }
}

// ---------------------------------------------------------------------------
// Construction as work — builders advance incomplete buildings
// ---------------------------------------------------------------------------

/// Advance construction on buildings with `construction_progress < 1.0`.
///
/// For each settlement, finds the first incomplete building and assigns the
/// best available builder (idle NPC with CONSTRUCTION or LABOR tags) to work
/// on it. Only one building per settlement is advanced per tick so builders
/// focus on finishing one project before starting the next.
fn advance_construction(state: &mut WorldState) {
    // Collect settlement IDs.
    let settlement_ids: Vec<u32> = state.settlements.iter()
        .map(|s| s.id)
        .collect();

    let tick = state.tick;
    let mut new_chronicles: Vec<ChronicleEntry> = Vec::new();

    for settlement_id in settlement_ids {
        // Find ALL incomplete buildings at this settlement (up to 5 per tick).
        let incomplete_buildings: Vec<(usize, u32, String)> = state.entities.iter()
            .enumerate()
            .filter_map(|(idx, e)| {
                if !e.alive || e.kind != EntityKind::Building { return None; }
                let bd = e.building.as_ref()?;
                if bd.settlement_id != Some(settlement_id) { return None; }
                if bd.construction_progress >= 1.0 { return None; }
                Some((idx, e.id, bd.name.clone()))
            })
            .take(5)
            .collect();

        if incomplete_buildings.is_empty() { continue; }

        for (building_idx, building_id, building_name) in incomplete_buildings {

        // Find the best idle builder at this settlement.
        // A builder is an NPC with CONSTRUCTION or LABOR behavior tags who
        // is idle (WorkState::Idle) and not adventuring.
        let mut best_builder: Option<(usize, u32, String, f32)> = None; // (idx, id, name, score)
        for (idx, entity) in state.entities.iter().enumerate() {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let npc = match &entity.npc { Some(n) => n, None => continue };
            if npc.home_settlement_id != Some(settlement_id) { continue; }
            if !matches!(npc.work_state, WorkState::Idle) { continue; }
            if matches!(npc.economic_intent, EconomicIntent::Adventuring { .. }) { continue; }

            let construction_val = npc.behavior_value(tags::CONSTRUCTION);
            let labor_val = npc.behavior_value(tags::LABOR);
            let score = construction_val + labor_val;
            // Any idle NPC can build (score 0 is fine), but prefer skilled builders.
            match &best_builder {
                Some((_, _, _, best_score)) if *best_score >= score => {}
                _ => best_builder = Some((idx, entity.id, npc.name.clone(), score)),
            }
        }

        let (builder_idx, _builder_id, builder_name, _builder_score) = match best_builder {
            Some(b) => b,
            None => continue, // no available builder
        };

        // Read the builder's CONSTRUCTION tag value for the progress formula.
        let construction_tag_val = state.entities[builder_idx]
            .npc.as_ref()
            .map(|n| n.behavior_value(tags::CONSTRUCTION))
            .unwrap_or(0.0);

        // Progress increment: 0.05 * (1.0 + construction_tags * 0.01) per 10-tick interval.
        let increment = 0.05 * (1.0 + construction_tag_val * 0.01);

        // Apply progress to the building.
        let builder_entity_id = state.entities[builder_idx].id;
        let completed = {
            let bd = state.entities[building_idx].building.as_mut().unwrap();
            bd.construction_progress += increment;
            if bd.construction_progress >= 1.0 {
                bd.construction_progress = 1.0;
                bd.built_tick = tick;
                bd.builder_id = Some(builder_entity_id);
                true
            } else {
                false
            }
        };

        // Emit CONSTRUCTION + LABOR + MASONRY behavior tags on the builder while working.
        let builder_pos = state.entities[builder_idx].pos;
        if let Some(npc) = &mut state.entities[builder_idx].npc {
            // Record BuiltSomething memory event on completion.
            if completed {
                npc.memory.record_event(MemoryEvent {
                    tick,
                    event_type: MemEventType::BuiltSomething,
                    location: builder_pos,
                    entity_ids: vec![building_id],
                    emotional_impact: 0.3,
                });
            }

            // Accumulate construction-related behavior tags on the builder.
            let mut action = ActionTags::empty();
            action.add(tags::CONSTRUCTION, 1.0);
            action.add(tags::LABOR, 0.5);
            action.add(tags::MASONRY, 0.5);
            npc.accumulate_tags(&action);
        }

        if completed {
            // Chronicle entry for building completion.
            let settlement_name = state.settlements.iter()
                .find(|s| s.id == settlement_id)
                .map(|s| s.name.as_str())
                .unwrap_or("unknown");

            new_chronicles.push(ChronicleEntry {
                tick,
                category: ChronicleCategory::Economy,
                text: format!(
                    "{} completed {} at {}",
                    builder_name, building_name, settlement_name
                ),
                entity_ids: vec![builder_entity_id, building_id],
            });
        }
        } // end for incomplete_buildings
    }

    for entry in new_chronicles {
        state.chronicle.push(entry);
    }
}
/// Generate a simple procedural name for a building.
fn generate_building_name(bt: BuildingType, id: u32) -> String {
    match bt {
        BuildingType::House => format!("House #{}", id),
        BuildingType::Longhouse => format!("Longhouse #{}", id),
        BuildingType::Manor => format!("Manor #{}", id),
        BuildingType::Farm => format!("Farm #{}", id),
        BuildingType::Mine => format!("Mine #{}", id),
        BuildingType::Sawmill => format!("Sawmill #{}", id),
        BuildingType::Forge => format!("The Iron Forge #{}", id),
        BuildingType::Workshop => format!("Workshop #{}", id),
        BuildingType::Apothecary => format!("Apothecary #{}", id),
        BuildingType::Market => format!("Market Square #{}", id),
        BuildingType::Warehouse => format!("Warehouse #{}", id),
        BuildingType::Inn => format!("The Traveler's Rest #{}", id),
        BuildingType::TradePost => format!("Trade Post #{}", id),
        BuildingType::GuildHall => format!("Guild Hall #{}", id),
        BuildingType::Temple => format!("Temple #{}", id),
        BuildingType::Barracks => format!("Barracks #{}", id),
        BuildingType::Watchtower => format!("Watchtower #{}", id),
        BuildingType::Library => format!("Library #{}", id),
        BuildingType::CourtHouse => format!("Court House #{}", id),
        BuildingType::Wall => format!("Wall #{}", id),
        BuildingType::Gate => format!("Gate #{}", id),
        BuildingType::Well => format!("Well #{}", id),
        BuildingType::Tent => format!("Tent #{}", id),
        BuildingType::Camp => format!("Camp #{}", id),
        BuildingType::Shrine => format!("Shrine #{}", id),
        BuildingType::Treasury => format!("Treasury #{}", id),
    }
}

// ---------------------------------------------------------------------------
// NPC-building assignment
// ---------------------------------------------------------------------------

/// Assign unhoused NPCs to residential buildings and unassigned workers to
/// production buildings. Uses real building entities with typed capacities.
/// Called at the end of `grow_cities`.
pub fn assign_npcs_to_buildings(state: &mut WorldState) {
    let settlement_ids: Vec<u32> = state.settlements.iter()
        .map(|s| s.id)
        .collect();

    for settlement_id in settlement_ids {
        // -----------------------------------------------------------------
        // Collect building entities for this settlement
        // -----------------------------------------------------------------
        let building_indices: Vec<usize> = state.entities.iter().enumerate()
            .filter(|(_, e)| {
                e.alive && e.kind == EntityKind::Building
                    && e.building.as_ref().map_or(false, |b| b.settlement_id == Some(settlement_id))
            })
            .map(|(idx, _)| idx)
            .collect();

        // -----------------------------------------------------------------
        // (a) Residential assignment: find homes with capacity
        // -----------------------------------------------------------------

        // Collect buildings with residential capacity and room.
        // Skip buildings still under construction (progress < 1.0).
        // Each entry: (entity_index, building_entity_id, available_slots)
        let mut available_homes: Vec<(usize, u32)> = Vec::new();
        for &bidx in &building_indices {
            let bd = match &state.entities[bidx].building {
                Some(b) => b,
                None => continue,
            };
            if bd.construction_progress < 1.0 { continue; }
            if bd.residential_capacity == 0 { continue; }
            let used = bd.resident_ids.len() as u8;
            let avail = bd.residential_capacity.saturating_sub(used);
            for _ in 0..avail {
                available_homes.push((bidx, state.entities[bidx].id));
            }
        }

        // Collect unhoused NPC entity indices at this settlement.
        let mut unhoused: Vec<usize> = Vec::new();
        for (idx, entity) in state.entities.iter().enumerate() {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            if let Some(npc) = &entity.npc {
                if npc.home_settlement_id == Some(settlement_id) && npc.home_building_id.is_none() {
                    unhoused.push(idx);
                }
            }
        }

        // Home assignment is now NPC-driven: NPCs build or rent homes.
        // Auto-assignment disabled — NPCs must earn their shelter.

        // -----------------------------------------------------------------
        // (b) Work assignment: match workers to work buildings by tag affinity
        // -----------------------------------------------------------------

        // Collect work buildings with available capacity.
        // Skip buildings still under construction (progress < 1.0).
        // Each entry: (entity_index, building_entity_id, zone)
        let mut available_work: Vec<(usize, u32, ZoneType)> = Vec::new();
        for &bidx in &building_indices {
            let bd = match &state.entities[bidx].building {
                Some(b) => b,
                None => continue,
            };
            if bd.construction_progress < 1.0 { continue; }
            if bd.work_capacity == 0 { continue; }
            let used = bd.worker_ids.len() as u8;
            let avail = bd.work_capacity.saturating_sub(used);
            let zone = building_type_to_zone(bd.building_type);
            for _ in 0..avail {
                available_work.push((bidx, state.entities[bidx].id, zone));
            }
        }

        // Collect unassigned worker NPC indices at this settlement.
        let mut unassigned: Vec<usize> = Vec::new();
        for (idx, entity) in state.entities.iter().enumerate() {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            if let Some(npc) = &entity.npc {
                if npc.home_settlement_id != Some(settlement_id) { continue; }
                if npc.work_building_id.is_some() { continue; }
                if !matches!(npc.economic_intent, EconomicIntent::Produce) { continue; }
                unassigned.push(idx);
            }
        }

        // Match workers to buildings by tag affinity.
        let mut work_slot_taken = vec![false; available_work.len()];
        for &eidx in &unassigned {
            let npc = match &state.entities[eidx].npc {
                Some(n) => n,
                None => continue,
            };

            let mining_val = npc.behavior_value(tags::MINING);
            let farming_val = npc.behavior_value(tags::FARMING);
            let trade_val = npc.behavior_value(tags::TRADE);
            let faith_val = npc.behavior_value(tags::FAITH);
            let research_val = npc.behavior_value(tags::RESEARCH);
            let lore_val = npc.behavior_value(tags::LORE);

            let zone_score = |zone: ZoneType| -> f32 {
                match zone {
                    ZoneType::Industrial => mining_val + farming_val,
                    ZoneType::Commercial => trade_val,
                    ZoneType::Religious => faith_val,
                    ZoneType::Arcane => research_val + lore_val,
                    _ => 0.0,
                }
            };

            let mut best_slot: Option<usize> = None;
            let mut best_score = -1.0_f32;

            for (slot_idx, (_, _, zone)) in available_work.iter().enumerate() {
                if work_slot_taken[slot_idx] { continue; }
                let score = zone_score(*zone);
                if score > best_score {
                    best_score = score;
                    best_slot = Some(slot_idx);
                }
            }

            // Assign if we found any slot (even score 0 -- NPCs need somewhere to work).
            if let Some(slot_idx) = best_slot {
                let (bidx, bid, _) = available_work[slot_idx];
                work_slot_taken[slot_idx] = true;
                let npc_id = state.entities[eidx].id;
                if let Some(npc) = &mut state.entities[eidx].npc {
                    npc.work_building_id = Some(bid);
                }
                // Push NPC ID into building's worker list.
                if let Some(bd) = &mut state.entities[bidx].building {
                    bd.worker_ids.push(npc_id);
                }
            }
        }
    }
}
pub fn update_building_specializations(state: &mut WorldState) {
    if state.tick % 50 != 0 || state.tick == 0 { return; }

    // Collect worker class data: (building_entity_idx, vec of (class_hash, display_name)).
    // We gather this first to avoid aliasing issues with mutable borrow later.
    let entity_count = state.entities.len();
    let mut building_worker_classes: Vec<(usize, Vec<(u32, String)>)> = Vec::new();

    for i in 0..entity_count {
        let entity = &state.entities[i];
        if !entity.alive || entity.kind != EntityKind::Building { continue; }
        let bld = match &entity.building { Some(b) => b, None => continue };
        if bld.worker_ids.is_empty() { continue; }
        if bld.construction_progress < 1.0 { continue; }

        // Collect primary class from each worker.
        let mut classes: Vec<(u32, String)> = Vec::new();
        for &worker_id in &bld.worker_ids {
            if let Some(worker) = state.entity(worker_id) {
                if let Some(npc) = &worker.npc {
                    // Use the worker's highest-level class as their primary.
                    if let Some(primary) = npc.classes.iter()
                        .max_by_key(|c| c.level)
                    {
                        classes.push((primary.class_name_hash, primary.display_name.clone()));
                    }
                }
            }
        }
        if !classes.is_empty() {
            building_worker_classes.push((i, classes));
        }
    }

    // Now apply updates to buildings.
    for (bld_idx, worker_classes) in building_worker_classes {
        let bld = match &mut state.entities[bld_idx].building {
            Some(b) => b,
            None => continue,
        };

        // Accumulate 50 ticks (one interval) per worker class present.
        for (class_hash, _name) in &worker_classes {
            if let Some(entry) = bld.worker_class_ticks.iter_mut()
                .find(|(h, _)| *h == *class_hash)
            {
                entry.1 += 50;
            } else {
                bld.worker_class_ticks.push((*class_hash, 50));
            }
        }

        // Find the dominant class (highest accumulated ticks).
        let total_ticks: u64 = bld.worker_class_ticks.iter().map(|(_, t)| *t).sum();
        if total_ticks < SPECIALIZATION_THRESHOLD { continue; }

        let dominant = bld.worker_class_ticks.iter()
            .max_by_key(|(_, t)| *t);
        let (dom_hash, dom_ticks) = match dominant {
            Some(&(h, t)) => (h, t),
            None => continue,
        };

        // Dominance ratio: what fraction of all accumulated ticks belong to the top class.
        let ratio = dom_ticks as f32 / total_ticks as f32;
        // Require at least 40% dominance to specialize.
        if ratio < 0.4 { continue; }

        // Strength scales from 0.0 at 40% dominance to 1.0 at 80%+.
        let strength = ((ratio - 0.4) / 0.4).min(1.0);

        bld.specialization_tag = Some(dom_hash);
        bld.specialization_strength = strength;

        // Generate a specialization name if not already set or if the tag changed.
        if bld.specialization_name.is_empty() {
            // Find the display name from the worker classes that contributed.
            let class_display = worker_classes.iter()
                .find(|(h, _)| *h == dom_hash)
                .map(|(_, n)| n.as_str())
                .unwrap_or("Specialist");

            let btype_name = match bld.building_type {
                BuildingType::Farm => "Farm",
                BuildingType::Mine => "Mine",
                BuildingType::Sawmill => "Sawmill",
                BuildingType::Forge => "Forge",
                BuildingType::Apothecary => "Apothecary",
                BuildingType::Market => "Market",
                BuildingType::Warehouse => "Warehouse",
                BuildingType::Barracks => "Barracks",
                BuildingType::Temple => "Temple",
                BuildingType::Inn => "Inn",
                _ => "Workshop",
            };

            bld.specialization_name = format!("{} {}", class_display, btype_name);
        }
    }
}
