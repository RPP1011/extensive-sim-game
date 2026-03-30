//! CA-driven city growth system.
//!
//! Runs every 10 ticks. For each settlement with a `city_grid_idx`, scores
//! frontier cells for suitability, picks the top candidates, assigns zone
//! types, transitions cells to Building state, and updates the influence map.
//!
//! Authority bias: the highest-level NPC at each settlement with leadership-
//! related tags biases zone suitability scores.
//!
//! Road extension runs every 50 ticks: A* from the nearest road cell to the
//! highest-scoring unconnected frontier cluster.
//!
//! This module provides two entry points:
//!   - `compute_buildings()` — delta-based stub (unchanged, still emits treasury deltas)
//!   - `grow_cities()` — called post-apply from the runtime, directly mutates city grids

use crate::world_sim::city_grid::{CellState, CityGrid, InfluenceMap, ZoneType, CellTerrain, suitability_score};
use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{WorldState, Entity, EntityKind, EconomicIntent, ChronicleEntry, ChronicleCategory, tags, BuildingType, BuildingData, SettlementSpecialty, ActionTags, WorkState, MemoryEvent, MemEventType, entity_hash};
use crate::world_sim::NUM_COMMODITIES;

/// Building tick interval for delta-based compute (treasury upgrades).
const BUILDING_TICK_INTERVAL: u64 = 3;

/// CA growth tick interval.
const GROWTH_TICK_INTERVAL: u64 = 10;

/// Road extension interval (ticks).
const ROAD_EXTENSION_INTERVAL: u64 = 50;

/// NPC cascade influence interval (ticks).
const NPC_INFLUENCE_INTERVAL: u64 = 100;

/// City events interval (decay, ruin — ticks).
const CITY_EVENTS_INTERVAL: u64 = 50;

/// Road decay interval (ticks).
const ROAD_DECAY_INTERVAL: u64 = 100;

/// Minimum treasury to trigger an auto-upgrade.
const UPGRADE_TREASURY_THRESHOLD: f32 = 200.0;

/// Cost of an upgrade (deducted from treasury).
const UPGRADE_COST: f32 = 100.0;

/// Minimum NPC level to qualify as "legendary" for cascade influence.
const LEGENDARY_NPC_LEVEL: u32 = 40;

/// Minimum class count to qualify as "legendary" for cascade influence.
const LEGENDARY_NPC_MIN_CLASSES: usize = 5;

/// Score threshold for NPC cascade building placement.
const CASCADE_SCORE_THRESHOLD: f32 = 0.15;

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
            location_id: settlement_id,
            delta: -UPGRADE_COST,
        });
    }
}

// ---------------------------------------------------------------------------
// CA growth loop — called post-apply from runtime
// ---------------------------------------------------------------------------

/// Grow all settlement city grids. Called from `WorldSim::grow_cities()` after
/// the apply phase, so it can mutate grids directly.
pub fn grow_cities(state: &mut WorldState) {
    if state.tick % GROWTH_TICK_INTERVAL != 0 {
        return;
    }

    let do_roads = state.tick % ROAD_EXTENSION_INTERVAL == 0;

    // Collect settlement info we need before mutating grids.
    // (settlement_id, grid_idx, population, settlement_name, noise_seed, authority_bias, specialty, pos)
    let settlement_info: Vec<(u32, usize, u32, String, u64, AuthorityBias, SettlementSpecialty, (f32, f32))> = state.settlements.iter()
        .filter_map(|s| {
            let grid_idx = s.city_grid_idx?;
            let noise_seed = entity_hash(s.id, state.tick, 0xB14D) as u64;
            let authority = find_authority_bias(state, s.id);
            Some((s.id, grid_idx, s.population, s.name.clone(), noise_seed, authority, s.specialty, s.pos))
        })
        .collect();

    let mut new_chronicles: Vec<ChronicleEntry> = Vec::new();
    let mut all_new_entities: Vec<Entity> = Vec::new();
    let base_max_id = state.entities.iter().map(|e| e.id).max().unwrap_or(0);

    for (settlement_id, grid_idx, population, settlement_name, noise_seed, authority, specialty, settlement_pos) in &settlement_info {
        let grid_idx = *grid_idx;
        if grid_idx >= state.city_grids.len() { continue; }

        // Rebuild frontier if needed (deserialization skips frontier).
        if state.city_grids[grid_idx].frontier.is_empty() {
            state.city_grids[grid_idx].rebuild_frontier();
        }

        // Calculate demand.
        let building_counts = state.city_grids[grid_idx].building_counts();
        let total_buildings: u32 = building_counts.iter().sum();
        let demand = calculate_demand(*population, &building_counts, total_buildings);

        // ~1 building per 3 people (houses hold families, workshops employ multiple).
        let max_buildings = (*population as f32 / 3.0).max(10.0) as u32;
        if total_buildings >= max_buildings { continue; } // at capacity

        let base_budget = if *population > 200 { 2 } else { 1 };
        let budget = base_budget.min(state.city_grids[grid_idx].frontier.len());
        if budget == 0 { continue; }

        // Score all frontier cells for each zone type, pick best (zone, score) per cell.
        let frontier_snapshot: Vec<(usize, usize)> = state.city_grids[grid_idx].frontier.iter().copied().collect();

        let mut scored: Vec<(usize, usize, ZoneType, f32)> = Vec::with_capacity(frontier_snapshot.len());

        for &(col, row) in &frontier_snapshot {
            // Assign zone directly from demand weights — suitability scoring is disabled
            // until the affinity matrix tuning is fixed (prevents Library domination).
            let zones = [
                (ZoneType::Residential, demand.residential),
                (ZoneType::Commercial,  demand.commercial),
                (ZoneType::Industrial,  demand.industrial),
                (ZoneType::Military,    demand.military),
                (ZoneType::Religious,   demand.religious),
                (ZoneType::Noble,       demand.noble),
                (ZoneType::Arcane,      demand.arcane),
            ];
            let best = zones.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
            let (best_zone, best_score) = (best.0, best.1);
            if best_zone != ZoneType::None {
                scored.push((col, row, best_zone, best_score));
            }
        }

        // Sort descending by score.
        scored.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        // Pick top N cells and place buildings.
        let placed = scored.len().min(budget);
        for i in 0..placed {
            let (col, row, zone, _score) = scored[i];

            // Assign zone and transition cell.
            {
                let cell = state.city_grids[grid_idx].cell_mut(col, row);
                cell.state = CellState::Building;
                cell.zone = zone;
                cell.density = 1;
                cell.age = 0;
            }

            // Spawn a real building entity.
            let new_id = base_max_id + 1 + all_new_entities.len() as u32;
            let world_pos = state.city_grids[grid_idx].grid_to_world(col, row, *settlement_pos);
            let building_type = zone_to_building_type(zone, specialty);

            let mut entity = Entity::new_building(new_id, world_pos);
            entity.building = Some(BuildingData {
                building_type,
                settlement_id: Some(*settlement_id),
                grid_col: col as u16,
                grid_row: row as u16,
                tier: 0,
                room_seed: entity_hash(new_id, state.tick, 0x800E) as u64,
                rooms: building_type.default_rooms(),
                residential_capacity: building_type.residential_capacity(),
                work_capacity: building_type.work_capacity(),
                resident_ids: Vec::new(),
                worker_ids: Vec::new(),
                // First 30 buildings per settlement start complete (bootstrap).
                // Later buildings require construction by builders.
                construction_progress: if total_buildings < 30 { 1.0 } else { 0.0 },
                built_tick: state.tick,
                builder_id: None,
                temporary: false,
                ttl_ticks: None,
                name: generate_building_name(building_type, new_id),
                storage: [0.0; NUM_COMMODITIES],
                storage_capacity: building_type.storage_capacity(),
                owner_id: None,
                builder_modifiers: Vec::new(),
                owner_modifiers: Vec::new(),
            });

            // Store entity ID in grid cell.
            state.city_grids[grid_idx].cell_mut(col, row).building_id = Some(new_id);
            all_new_entities.push(entity);

            // Update frontier incrementally.
            state.city_grids[grid_idx].update_frontier_around(col, row);

            // Update influence map.
            let strength = 1.0;
            // We need to propagate influence but can't borrow both mutably.
            // Copy the grid reference data we need for propagation.
            let grid_ref = state.city_grids[grid_idx].clone();
            state.influence_maps[grid_idx].propagate_building(col, row, zone, strength, &grid_ref);
        }

        // Chronicle entry for notable construction milestones.
        let new_total: u32 = state.city_grids[grid_idx].building_counts().iter().sum();
        if new_total > 0 && new_total % 100 == 0 && total_buildings < new_total {
            new_chronicles.push(ChronicleEntry {
                tick: state.tick,
                category: ChronicleCategory::Economy,
                text: format!("{} has grown to {} buildings", settlement_name, new_total),
                entity_ids: Vec::new(),
            });
        }

        // Road extension (every 50 ticks).
        if do_roads {
            extend_roads(&mut state.city_grids[grid_idx], *noise_seed);
        }

        // Density upgrades: cells with age > 200 and >= 6 developed neighbors.
        age_and_upgrade_density(&mut state.city_grids[grid_idx]);
    }

    // Append chronicles.
    for entry in new_chronicles {
        state.chronicle.push(entry);
    }

    // Push all newly spawned building entities into the world state.
    if !all_new_entities.is_empty() {
        for e in all_new_entities {
            state.entities.push(e);
        }
        state.rebuild_entity_cache();
    }

    // Legendary NPC cascade influence (every 100 ticks).
    if state.tick % NPC_INFLUENCE_INTERVAL == 0 {
        apply_npc_influence(state);
    }

    // City events: decay, ruin, road decay, density tier caps (every 50 ticks).
    if state.tick % CITY_EVENTS_INTERVAL == 0 {
        apply_city_events(state);
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
        .filter(|s| s.city_grid_idx.is_some())
        .map(|s| s.id)
        .collect();

    let tick = state.tick;
    let mut new_chronicles: Vec<ChronicleEntry> = Vec::new();

    for settlement_id in settlement_ids {
        // Find the first incomplete building at this settlement.
        let incomplete_building: Option<(usize, u32, String)> = state.entities.iter()
            .enumerate()
            .filter_map(|(idx, e)| {
                if !e.alive || e.kind != EntityKind::Building { return None; }
                let bd = e.building.as_ref()?;
                if bd.settlement_id != Some(settlement_id) { return None; }
                if bd.construction_progress >= 1.0 { return None; }
                Some((idx, e.id, bd.name.clone()))
            })
            .next();

        let (building_idx, building_id, building_name) = match incomplete_building {
            Some(b) => b,
            None => continue,
        };

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
    }

    for entry in new_chronicles {
        state.chronicle.push(entry);
    }
}

// ---------------------------------------------------------------------------
// Demand calculation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct ZoneDemand {
    residential: f32,
    commercial: f32,
    industrial: f32,
    religious: f32,
    arcane: f32,
    noble: f32,
    military: f32,
}

fn calculate_demand(population: u32, counts: &[u32; 8], total: u32) -> ZoneDemand {
    let pop = population.max(1) as f32;
    let total_f = total.max(1) as f32;

    // Target ratios for a balanced settlement.
    let res_ratio = counts[ZoneType::Residential as usize] as f32 / total_f;
    let com_ratio = counts[ZoneType::Commercial as usize] as f32 / total_f;
    let ind_ratio = counts[ZoneType::Industrial as usize] as f32 / total_f;
    let rel_ratio = counts[ZoneType::Religious as usize] as f32 / total_f;
    let arc_ratio = counts[ZoneType::Arcane as usize] as f32 / total_f;
    let nob_ratio = counts[ZoneType::Noble as usize] as f32 / total_f;
    let mil_ratio = counts[ZoneType::Military as usize] as f32 / total_f;

    // Demand = (target_ratio - current_ratio) * population_scale.
    // Higher population creates more demand overall.
    let pop_scale = (pop / 50.0).sqrt();

    ZoneDemand {
        residential: (0.40 - res_ratio).max(0.0) * pop_scale + 0.1,
        commercial:  (0.20 - com_ratio).max(0.0) * pop_scale + 0.05,
        industrial:  (0.15 - ind_ratio).max(0.0) * pop_scale + 0.05,
        religious:   (0.05 - rel_ratio).max(0.0) * pop_scale + 0.02,
        arcane:      (0.05 - arc_ratio).max(0.0) * pop_scale + 0.02,
        noble:       (0.05 - nob_ratio).max(0.0) * pop_scale + 0.02,
        military:    (0.10 - mil_ratio).max(0.0) * pop_scale + 0.03,
    }
}

// ---------------------------------------------------------------------------
// Authority bias
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct AuthorityBias {
    combat: f32,
    trade: f32,
    research: f32,
    faith: f32,
    leadership: f32,
}

fn find_authority_bias(state: &WorldState, settlement_id: u32) -> AuthorityBias {
    let mut best_level: u32 = 0;
    let mut best_bias = AuthorityBias::default();

    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };
        if npc.home_settlement_id != Some(settlement_id) { continue; }

        // Check if this NPC has leadership-related tags.
        let leadership_val = npc.behavior_value(tags::LEADERSHIP);
        let diplomacy_val = npc.behavior_value(tags::DIPLOMACY);
        let trade_val = npc.behavior_value(tags::TRADE);
        let combat_val = npc.behavior_value(tags::COMBAT);
        let tactics_val = npc.behavior_value(tags::TACTICS);
        let research_val = npc.behavior_value(tags::RESEARCH);
        let lore_val = npc.behavior_value(tags::LORE);
        let faith_val = npc.behavior_value(tags::FAITH);

        let authority_score = leadership_val + diplomacy_val + trade_val + combat_val;
        if authority_score < 1.0 { continue; }

        if entity.level > best_level {
            best_level = entity.level;
            // Scale authority bias to be comparable to demand weights (~0.01-0.1 range).
            best_bias = AuthorityBias {
                combat: ((combat_val + tactics_val) * 0.0001).min(0.05),
                trade: ((trade_val + npc.behavior_value(tags::NEGOTIATION)) * 0.0001).min(0.05),
                research: ((research_val + lore_val) * 0.0001).min(0.05),
                faith: (faith_val * 0.0001).min(0.05),
                leadership: (leadership_val * 0.0001).min(0.05),
            };
        }
    }

    best_bias
}

// ---------------------------------------------------------------------------
// Zone scoring
// ---------------------------------------------------------------------------

/// Score a frontier cell for all zone types. Returns the best (zone, score).
fn score_cell_all_zones(
    grid: &CityGrid,
    influence: &InfluenceMap,
    col: usize,
    row: usize,
    noise_seed: u64,
    demand: &ZoneDemand,
    authority: AuthorityBias,
) -> (ZoneType, f32) {
    let zones = [
        (ZoneType::Residential, demand.residential, 0.0),
        (ZoneType::Commercial,  demand.commercial,  authority.trade),
        (ZoneType::Industrial,  demand.industrial,  0.0),
        (ZoneType::Religious,   demand.religious,    authority.faith),
        (ZoneType::Arcane,      demand.arcane,       authority.research),
        (ZoneType::Noble,       demand.noble,        authority.leadership),
        (ZoneType::Military,    demand.military,     authority.combat),
    ];

    let mut best_zone = ZoneType::None;
    let mut best_score = f32::NEG_INFINITY;

    for &(zone, demand_weight, auth_bias) in &zones {
        let base = suitability_score(grid, influence, col, row, zone, noise_seed);
        // Demand dominates: multiply base by demand squared so high-demand zones win.
        let score = base * demand_weight * demand_weight * 10.0 + auth_bias;
        if score > best_score {
            best_score = score;
            best_zone = zone;
        }
    }

    (best_zone, best_score)
}

// ---------------------------------------------------------------------------
// Road extension
// ---------------------------------------------------------------------------

/// Extend roads toward the highest-scoring unconnected frontier cluster.
/// Finds the frontier cell farthest from any road, then A* from nearest road.
fn extend_roads(grid: &mut CityGrid, _noise_seed: u64) {
    // Find frontier cell with highest road distance (most disconnected).
    let mut best_target: Option<(usize, usize, u16)> = None;
    for &(col, row) in &grid.frontier {
        let idx = grid.idx(col, row);
        let dist = grid.road_distance[idx];
        if dist > 3 { // only extend if significantly disconnected
            match &best_target {
                Some((_, _, d)) if *d >= dist => {}
                _ => best_target = Some((col, row, dist)),
            }
        }
    }

    let (target_col, target_row, _) = match best_target {
        Some(t) => t,
        None => return,
    };

    // Find nearest road cell to the target using BFS.
    let start = find_nearest_road(grid, target_col, target_row);
    let (start_col, start_row) = match start {
        Some(s) => s,
        None => return,
    };

    // A* from start to target, placing road cells along the path.
    let path = astar_road(grid, start_col, start_row, target_col, target_row);
    for (col, row) in path {
        let cell = grid.cell_mut(col, row);
        if cell.state == CellState::Empty
            && cell.terrain != CellTerrain::Water
            && cell.terrain != CellTerrain::Cliff
        {
            cell.state = CellState::Road;
            cell.road_tier = 1; // alley
            grid.update_frontier_around(col, row);
        }
    }

    // Rebuild road distances after new roads.
    grid.update_road_distances();
}

/// Find the nearest road cell to a target position via BFS.
fn find_nearest_road(grid: &CityGrid, target_col: usize, target_row: usize) -> Option<(usize, usize)> {
    use std::collections::VecDeque;
    let mut visited = vec![false; grid.cols * grid.rows];
    let mut queue = VecDeque::new();

    let idx = grid.idx(target_col, target_row);
    visited[idx] = true;
    queue.push_back((target_col, target_row));

    while let Some((col, row)) = queue.pop_front() {
        if grid.cell(col, row).state == CellState::Road {
            return Some((col, row));
        }

        for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let nx = col as i32 + dx;
            let ny = row as i32 + dy;
            if nx < 0 || ny < 0 { continue; }
            let nx = nx as usize;
            let ny = ny as usize;
            if !grid.in_bounds(nx, ny) { continue; }
            let nidx = grid.idx(nx, ny);
            if visited[nidx] { continue; }
            visited[nidx] = true;
            queue.push_back((nx, ny));
        }
    }
    None
}

/// Simple A* pathfinding from (sx,sy) to (tx,ty), avoiding water/cliff.
fn astar_road(grid: &CityGrid, sx: usize, sy: usize, tx: usize, ty: usize) -> Vec<(usize, usize)> {
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;

    let n = grid.cols * grid.rows;
    let mut g_score = vec![u32::MAX; n];
    let mut came_from = vec![(usize::MAX, usize::MAX); n];
    let mut open = BinaryHeap::new();

    let start_idx = grid.idx(sx, sy);
    g_score[start_idx] = 0;
    let h = heuristic(sx, sy, tx, ty);
    open.push(Reverse((h, sx, sy)));

    while let Some(Reverse((_f, col, row))) = open.pop() {
        if col == tx && row == ty {
            // Reconstruct path.
            return reconstruct_path(&came_from, grid, tx, ty, sx, sy);
        }

        let idx = grid.idx(col, row);
        let current_g = g_score[idx];

        for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let nx = col as i32 + dx;
            let ny = row as i32 + dy;
            if nx < 0 || ny < 0 { continue; }
            let nx = nx as usize;
            let ny = ny as usize;
            if !grid.in_bounds(nx, ny) { continue; }

            let cell = grid.cell(nx, ny);
            if cell.terrain == CellTerrain::Water || cell.terrain == CellTerrain::Cliff { continue; }

            // Cost: 1 for empty/road, 3 for buildings (prefer routing around).
            let step_cost = match cell.state {
                CellState::Road => 1,
                CellState::Empty => 1,
                _ => 3,
            };

            let nidx = grid.idx(nx, ny);
            let tentative_g = current_g + step_cost;
            if tentative_g < g_score[nidx] {
                g_score[nidx] = tentative_g;
                came_from[nidx] = (col, row);
                let h = heuristic(nx, ny, tx, ty);
                open.push(Reverse((tentative_g + h, nx, ny)));
            }
        }
    }

    Vec::new() // no path found
}

fn heuristic(ax: usize, ay: usize, bx: usize, by: usize) -> u32 {
    let dx = (ax as i32 - bx as i32).unsigned_abs();
    let dy = (ay as i32 - by as i32).unsigned_abs();
    dx + dy
}

fn reconstruct_path(
    came_from: &[(usize, usize)],
    grid: &CityGrid,
    tx: usize, ty: usize,
    sx: usize, sy: usize,
) -> Vec<(usize, usize)> {
    let mut path = Vec::new();
    let (mut cx, mut cy) = (tx, ty);
    let max_steps = grid.cols * grid.rows; // safety bound
    let mut steps = 0;
    while (cx, cy) != (sx, sy) && steps < max_steps {
        path.push((cx, cy));
        let idx = grid.idx(cx, cy);
        let (px, py) = came_from[idx];
        if px == usize::MAX { break; }
        cx = px;
        cy = py;
        steps += 1;
    }
    path.reverse();
    path
}

// ---------------------------------------------------------------------------
// Density upgrades
// ---------------------------------------------------------------------------

/// Age all building cells and upgrade density for mature cells with many neighbors.
/// Density caps are enforced by adjacent road tier:
///   - Adjacent to road_tier >= 3 (avenue): density can reach 3
///   - Adjacent to road_tier 2 (street): density capped at 2
///   - Adjacent only to road_tier 1 (alley) or no roads: density capped at 1
fn age_and_upgrade_density(grid: &mut CityGrid) {
    let cols = grid.cols;
    let rows = grid.rows;

    // First pass: increment age for all building cells.
    for row in 0..rows {
        for col in 0..cols {
            let cell = grid.cell_mut(col, row);
            if cell.state == CellState::Building {
                cell.age = cell.age.saturating_add(1);
            }
        }
    }

    // Second pass: upgrade density for eligible cells.
    // Collect upgrades first to avoid aliasing issues with count_developed_neighbors.
    let mut upgrades: Vec<(usize, usize)> = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let cell = grid.cell(col, row);
            if cell.state != CellState::Building { continue; }
            if cell.density >= 3 { continue; } // max density
            if cell.age < 200 { continue; } // must be mature

            // Determine max density allowed by adjacent road tier.
            let max_road_tier = best_adjacent_road_tier(grid, col, row);
            let density_cap = road_tier_density_cap(max_road_tier);
            if cell.density >= density_cap { continue; }

            if grid.count_developed_neighbors(col, row) >= 6 {
                upgrades.push((col, row));
            }
        }
    }

    for (col, row) in upgrades {
        grid.cell_mut(col, row).density += 1;
    }
}

/// Find the highest road tier among 4-connected (cardinal) neighbors.
fn best_adjacent_road_tier(grid: &CityGrid, col: usize, row: usize) -> u8 {
    let mut best = 0u8;
    for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
        let nx = col as i32 + dx;
        let ny = row as i32 + dy;
        if nx < 0 || ny < 0 { continue; }
        let nx = nx as usize;
        let ny = ny as usize;
        if !grid.in_bounds(nx, ny) { continue; }
        let cell = grid.cell(nx, ny);
        if cell.state == CellState::Road && cell.road_tier > best {
            best = cell.road_tier;
        }
    }
    best
}

/// Maximum building density allowed by the best adjacent road tier.
fn road_tier_density_cap(road_tier: u8) -> u8 {
    match road_tier {
        0 | 1 => 1,  // alley or no road: max density 1
        2 => 2,      // street: max density 2
        _ => 3,      // avenue (3) or highway (4): max density 3
    }
}

// ---------------------------------------------------------------------------
// Building type helpers
// ---------------------------------------------------------------------------

/// Pick a building type from zone + settlement specialty.
fn zone_to_building_type(zone: ZoneType, specialty: &SettlementSpecialty) -> BuildingType {
    match zone {
        ZoneType::Residential => BuildingType::House,
        ZoneType::Commercial => {
            // Vary commercial buildings by specialty.
            match specialty {
                SettlementSpecialty::TradeHub | SettlementSpecialty::PortTown => BuildingType::Market,
                _ => BuildingType::Inn,
            }
        }
        ZoneType::Industrial => {
            match specialty {
                SettlementSpecialty::FarmingVillage => BuildingType::Farm,
                SettlementSpecialty::MiningTown => BuildingType::Mine,
                SettlementSpecialty::CraftingGuild => BuildingType::Forge,
                _ => BuildingType::Workshop,
            }
        }
        ZoneType::Noble => BuildingType::GuildHall,
        ZoneType::Religious => BuildingType::Temple,
        ZoneType::Arcane => BuildingType::Library,
        ZoneType::Military => BuildingType::Barracks,
        ZoneType::None => BuildingType::Well,
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
        .filter(|s| s.city_grid_idx.is_some())
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

        // Assign homes: workers try workplace first (if residential capacity),
        // then fall back to nearest House/Longhouse.
        let home_count = unhoused.len().min(available_homes.len());
        for i in 0..home_count {
            let eidx = unhoused[i];
            let (bidx, bid) = available_homes[i];
            let npc_id = state.entities[eidx].id;
            if let Some(npc) = &mut state.entities[eidx].npc {
                npc.home_building_id = Some(bid);
            }
            // Push NPC ID into building's resident list.
            if let Some(bd) = &mut state.entities[bidx].building {
                bd.resident_ids.push(npc_id);
            }
        }

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
            let zone = bd.building_type.zone();
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

// ---------------------------------------------------------------------------
// Milestone 4: High-Impact NPC Cascade Influence
// ---------------------------------------------------------------------------

/// Legendary NPC info extracted from entity scan (avoids holding borrow on state).
struct LegendaryNpc {
    entity_id: u32,
    name: String,
    level: u32,
    settlement_id: u32,
    dominant_zone: ZoneType,
}

/// Determine the dominant zone type for a legendary NPC based on their behavior tags.
fn dominant_zone_for_npc(npc: &crate::world_sim::state::NpcData) -> ZoneType {
    let trade = npc.behavior_value(tags::TRADE) + npc.behavior_value(tags::NEGOTIATION);
    let combat = npc.behavior_value(tags::COMBAT) + npc.behavior_value(tags::TACTICS);
    let research = npc.behavior_value(tags::RESEARCH) + npc.behavior_value(tags::LORE);
    let faith = npc.behavior_value(tags::FAITH);
    let leadership = npc.behavior_value(tags::LEADERSHIP) + npc.behavior_value(tags::DIPLOMACY);

    let scores = [
        (trade, ZoneType::Commercial),
        (combat, ZoneType::Military),
        (research, ZoneType::Arcane),
        (faith, ZoneType::Religious),
        (leadership, ZoneType::Noble),
    ];

    scores.iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|&(_, zone)| zone)
        .unwrap_or(ZoneType::Commercial)
}

/// Zone type display name for chronicle entries.
fn zone_display_name(zone: ZoneType) -> &'static str {
    match zone {
        ZoneType::None => "undeveloped",
        ZoneType::Residential => "residential",
        ZoneType::Commercial => "commercial",
        ZoneType::Industrial => "industrial",
        ZoneType::Religious => "religious",
        ZoneType::Arcane => "arcane",
        ZoneType::Noble => "noble",
        ZoneType::Military => "military",
    }
}

/// Apply cascade growth from legendary NPCs (level >= 40, 5+ classes).
///
/// Each legendary NPC propagates influence into the influence map with strength
/// proportional to their level, in their dominant zone type. After propagation,
/// nearby frontier cells are re-scored and 1-2 additional buildings are placed
/// if they score above threshold — the "cascade" that pulls an entire quarter
/// into existence around a legendary figure.
fn apply_npc_influence(state: &mut WorldState) {
    // NOTE: Cascade building disabled until zone distribution is balanced.
    // The cascade was creating hundreds of Libraries from research-heavy NPCs.
    return;
    // Build a mapping: settlement_id -> grid_idx + name + specialty + pos.
    let settlement_grid_map: Vec<(u32, usize, String, SettlementSpecialty, (f32, f32))> = state.settlements.iter()
        .filter_map(|s| {
            let grid_idx = s.city_grid_idx?;
            if grid_idx >= state.city_grids.len() { return None; }
            Some((s.id, grid_idx, s.name.clone(), s.specialty, s.pos))
        })
        .collect();

    // Scan for legendary NPCs. Extract info to avoid holding a borrow on state.
    let legendary_npcs: Vec<LegendaryNpc> = state.entities.iter()
        .filter_map(|entity| {
            if !entity.alive || entity.kind != EntityKind::Npc { return None; }
            if entity.level < LEGENDARY_NPC_LEVEL { return None; }
            let npc = entity.npc.as_ref()?;
            if npc.classes.len() < LEGENDARY_NPC_MIN_CLASSES { return None; }
            let settlement_id = npc.home_settlement_id?;
            Some(LegendaryNpc {
                entity_id: entity.id,
                name: npc.name.clone(),
                level: entity.level,
                settlement_id,
                dominant_zone: dominant_zone_for_npc(npc),
            })
        })
        .collect();

    if legendary_npcs.is_empty() { return; }

    let tick = state.tick;
    let mut new_chronicles: Vec<ChronicleEntry> = Vec::new();
    let mut cascade_entities: Vec<Entity> = Vec::new();
    let cascade_base_max_id = state.entities.iter().map(|e| e.id).max().unwrap_or(0);

    for legend in &legendary_npcs {
        // Find the grid index for this NPC's settlement.
        let (grid_idx, settlement_name, settlement_specialty, settlement_pos) = match settlement_grid_map.iter()
            .find(|(sid, _, _, _, _)| *sid == legend.settlement_id)
        {
            Some((_, gidx, name, spec, pos)) => (*gidx, name.clone(), *spec, *pos),
            None => continue,
        };

        // Propagate influence with strength = level * 0.1 (vs normal 1.0).
        let strength = legend.level as f32 * 0.1;
        let zone = legend.dominant_zone;

        // Pick an influence origin: the grid center (legendary figures claim the heart).
        let origin_col = state.city_grids[grid_idx].center.0;
        let origin_row = state.city_grids[grid_idx].center.1;

        let grid_ref = state.city_grids[grid_idx].clone();
        state.influence_maps[grid_idx].propagate_building(
            origin_col, origin_row, zone, strength, &grid_ref,
        );

        // Re-score nearby frontier cells and place 1-2 cascade buildings.
        // Collect frontier within a 20-cell manhattan radius of center.
        let frontier_snapshot: Vec<(usize, usize)> = state.city_grids[grid_idx].frontier.iter()
            .filter(|&&(col, row)| {
                let dx = (col as i32 - origin_col as i32).abs();
                let dy = (row as i32 - origin_row as i32).abs();
                dx + dy <= 20
            })
            .copied()
            .collect();

        if frontier_snapshot.is_empty() { continue; }

        let noise_seed = entity_hash(legend.entity_id, tick, 0xB14D) as u64;

        // Score frontier cells for the NPC's dominant zone type.
        let mut scored: Vec<(usize, usize, f32)> = frontier_snapshot.iter()
            .map(|&(col, row)| {
                let score = suitability_score(
                    &state.city_grids[grid_idx],
                    &state.influence_maps[grid_idx],
                    col, row, zone, noise_seed,
                );
                (col, row, score)
            })
            .filter(|&(_, _, score)| score > CASCADE_SCORE_THRESHOLD)
            .collect();

        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Place 1-2 cascade buildings.
        let cascade_budget = scored.len().min(2);
        let mut placed_any = false;

        for i in 0..cascade_budget {
            let (col, row, _score) = scored[i];

            {
                let cell = state.city_grids[grid_idx].cell_mut(col, row);
                cell.state = CellState::Building;
                cell.zone = zone;
                cell.density = 1;
                cell.age = 0;
            }

            // Spawn a real building entity for cascade placement.
            let new_id = cascade_base_max_id + 1 + cascade_entities.len() as u32;
            let world_pos = state.city_grids[grid_idx].grid_to_world(col, row, settlement_pos);
            let building_type = zone_to_building_type(zone, &settlement_specialty);

            let mut entity = Entity::new_building(new_id, world_pos);
            entity.building = Some(BuildingData {
                building_type,
                settlement_id: Some(legend.settlement_id),
                grid_col: col as u16,
                grid_row: row as u16,
                tier: 0,
                room_seed: entity_hash(new_id, tick, 0x800E) as u64,
                rooms: building_type.default_rooms(),
                residential_capacity: building_type.residential_capacity(),
                work_capacity: building_type.work_capacity(),
                resident_ids: Vec::new(),
                worker_ids: Vec::new(),
                construction_progress: 0.0, // must be constructed by NPC builders
                built_tick: tick,
                builder_id: None,
                temporary: false,
                ttl_ticks: None,
                name: generate_building_name(building_type, new_id),
                storage: [0.0; NUM_COMMODITIES],
                storage_capacity: building_type.storage_capacity(),
                owner_id: None,
                builder_modifiers: Vec::new(),
                owner_modifiers: Vec::new(),
            });

            state.city_grids[grid_idx].cell_mut(col, row).building_id = Some(new_id);
            cascade_entities.push(entity);

            state.city_grids[grid_idx].update_frontier_around(col, row);

            let grid_ref = state.city_grids[grid_idx].clone();
            state.influence_maps[grid_idx].propagate_building(
                col, row, zone, 1.0, &grid_ref,
            );
            placed_any = true;
        }

        // Chronicle entry for cascade.
        if placed_any {
            let npc_name = if legend.name.is_empty() {
                format!("Entity #{}", legend.entity_id)
            } else {
                legend.name.clone()
            };
            new_chronicles.push(ChronicleEntry {
                tick,
                category: ChronicleCategory::Economy,
                text: format!(
                    "{}'s presence transforms the {} quarter of {}",
                    npc_name,
                    zone_display_name(zone),
                    settlement_name,
                ),
                entity_ids: vec![legend.entity_id],
            });
        }
    }

    for entry in new_chronicles {
        state.chronicle.push(entry);
    }

    // Push cascade-spawned building entities into world state.
    if !cascade_entities.is_empty() {
        for e in cascade_entities {
            state.entities.push(e);
        }
        state.rebuild_entity_cache();
    }
}

// ---------------------------------------------------------------------------
// Milestone 6: City Events (Decay, Ruin, Road Decay, Density Caps)
// ---------------------------------------------------------------------------

/// Count adjacent building cells (4-connected cardinal neighbors).
fn count_adjacent_buildings(grid: &CityGrid, col: usize, row: usize) -> u8 {
    let mut count = 0u8;
    for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
        let nx = col as i32 + dx;
        let ny = row as i32 + dy;
        if nx < 0 || ny < 0 { continue; }
        let nx = nx as usize;
        let ny = ny as usize;
        if !grid.in_bounds(nx, ny) { continue; }
        if grid.cell(nx, ny).state == CellState::Building {
            count += 1;
        }
    }
    count
}

/// Build a set of building_ids that are referenced by any NPC's home or work assignment.
fn occupied_building_ids(state: &WorldState) -> std::collections::HashSet<u32> {
    let mut occupied = std::collections::HashSet::new();
    for entity in &state.entities {
        if !entity.alive { continue; }
        if let Some(npc) = &entity.npc {
            if let Some(bid) = npc.home_building_id {
                occupied.insert(bid);
            }
            if let Some(bid) = npc.work_building_id {
                occupied.insert(bid);
            }
        }
    }
    occupied
}

/// Check if any cardinal neighbor of (col, row) is a building with an occupied building_id.
fn has_adjacent_occupied_building(
    grid: &CityGrid,
    col: usize,
    row: usize,
    occupied: &std::collections::HashSet<u32>,
) -> bool {
    for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
        let nx = col as i32 + dx;
        let ny = row as i32 + dy;
        if nx < 0 || ny < 0 { continue; }
        let nx = nx as usize;
        let ny = ny as usize;
        if !grid.in_bounds(nx, ny) { continue; }
        let cell = grid.cell(nx, ny);
        if cell.state == CellState::Building {
            if let Some(bid) = cell.building_id {
                if occupied.contains(&bid) {
                    return true;
                }
            }
        }
    }
    false
}

/// Propagate danger influence from a ruin cell outward via BFS.
fn propagate_danger(
    influence: &mut InfluenceMap,
    col: usize,
    row: usize,
    strength: f32,
    grid: &CityGrid,
) {
    use std::collections::VecDeque;
    let max_dist = 10u16;
    let mut visited = vec![false; influence.cols * influence.rows];
    let mut queue = VecDeque::new();

    let start = row * influence.cols + col;
    visited[start] = true;
    queue.push_back((col, row, 0u16));

    while let Some((cx, cy, dist)) = queue.pop_front() {
        if dist > max_dist { continue; }

        let falloff = strength / (1.0 + dist as f32);
        let idx = cy * influence.cols + cx;
        influence.danger[idx] += falloff;

        for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            if nx < 0 || ny < 0 { continue; }
            let nx = nx as usize;
            let ny = ny as usize;
            if nx >= influence.cols || ny >= influence.rows { continue; }
            let nidx = ny * influence.cols + nx;
            if visited[nidx] { continue; }
            visited[nidx] = true;

            let step = if grid.cell(nx, ny).state == CellState::Road { 1 } else { 2 };
            queue.push_back((nx, ny, dist + step));
        }
    }
}

/// Apply city events: building decay/ruin, road decay, and road-tier density caps.
///
/// Called every 50 ticks from `grow_cities`.
///
/// - **Decay and ruin**: Unoccupied building cells with no adjacent occupied buildings
///   age faster (age += 5). If age > 500 they transition to Ruin. Ruin cells propagate
///   danger influence.
///
/// - **Road decay** (every 100 ticks): High-tier roads with few adjacent buildings
///   downgrade. Isolated tier-1 roads are removed.
///
/// - **Density tier caps**: Buildings next to avenues (tier >= 3) can reach density 3;
///   buildings only next to alleys (tier 1) are capped at density 1. Over-dense
///   buildings are downgraded.
fn apply_city_events(state: &mut WorldState) {
    let occupied = occupied_building_ids(state);
    let do_road_decay = state.tick % ROAD_DECAY_INTERVAL == 0;
    let tick = state.tick;

    // Collect settlement info.
    let settlement_info: Vec<(usize, String)> = state.settlements.iter()
        .filter_map(|s| {
            let grid_idx = s.city_grid_idx?;
            if grid_idx >= state.city_grids.len() { return None; }
            Some((grid_idx, s.name.clone()))
        })
        .collect();

    let mut new_chronicles: Vec<ChronicleEntry> = Vec::new();

    for (grid_idx, settlement_name) in &settlement_info {
        let grid_idx = *grid_idx;
        let cols = state.city_grids[grid_idx].cols;
        let rows = state.city_grids[grid_idx].rows;

        // --- Decay and ruin ---

        // First pass: identify building cells that are unoccupied AND have no
        // adjacent occupied buildings.
        let mut decay_candidates: Vec<(usize, usize)> = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let cell = state.city_grids[grid_idx].cell(col, row);
                if cell.state != CellState::Building { continue; }

                // Check if this cell's building_id is occupied by any NPC.
                let cell_occupied = cell.building_id
                    .map(|bid| occupied.contains(&bid))
                    .unwrap_or(false);
                if cell_occupied { continue; }

                // Check if any adjacent building is occupied.
                let has_occupied_neighbor = has_adjacent_occupied_building(
                    &state.city_grids[grid_idx], col, row, &occupied,
                );
                if has_occupied_neighbor { continue; }

                decay_candidates.push((col, row));
            }
        }

        // Apply accelerated aging and ruin transitions.
        let mut new_ruins = 0u32;
        let mut ruin_cells: Vec<(usize, usize)> = Vec::new();
        for &(col, row) in &decay_candidates {
            let cell = state.city_grids[grid_idx].cell_mut(col, row);
            // Accelerated aging: +5 instead of normal +1.
            cell.age = cell.age.saturating_add(5);

            if cell.age > 500 {
                cell.state = CellState::Ruin;
                cell.density = 0;
                new_ruins += 1;
                ruin_cells.push((col, row));
            }
        }

        // Propagate danger influence from new ruin cells.
        for &(col, row) in &ruin_cells {
            let grid_ref = state.city_grids[grid_idx].clone();
            propagate_danger(&mut state.influence_maps[grid_idx], col, row, 1.0, &grid_ref);

            // Update frontier: ruin cells may change neighbor frontier status.
            state.city_grids[grid_idx].update_frontier_around(col, row);
        }

        // Chronicle for ruin (once per 10 ruins to avoid spam).
        if new_ruins > 0 && new_ruins % 10 < 2 {
            new_chronicles.push(ChronicleEntry {
                tick,
                category: ChronicleCategory::Crisis,
                text: format!("{} district falls into ruin", settlement_name),
                entity_ids: Vec::new(),
            });
        }

        // --- Road decay (every 100 ticks) ---

        if do_road_decay {
            let mut road_changes: Vec<(usize, usize, u8)> = Vec::new();
            let mut road_removals: Vec<(usize, usize)> = Vec::new();

            for row in 0..rows {
                for col in 0..cols {
                    let cell = state.city_grids[grid_idx].cell(col, row);
                    if cell.state != CellState::Road { continue; }

                    let adj_buildings = count_adjacent_buildings(
                        &state.city_grids[grid_idx], col, row,
                    );

                    if cell.road_tier >= 2 && adj_buildings < 2 {
                        // Downgrade high-tier roads with few adjacent buildings.
                        road_changes.push((col, row, cell.road_tier - 1));
                    } else if cell.road_tier == 1 && adj_buildings == 0 {
                        // Remove isolated alleys.
                        road_removals.push((col, row));
                    }
                }
            }

            for (col, row, new_tier) in road_changes {
                state.city_grids[grid_idx].cell_mut(col, row).road_tier = new_tier;
            }

            for &(col, row) in &road_removals {
                let cell = state.city_grids[grid_idx].cell_mut(col, row);
                cell.state = CellState::Empty;
                cell.road_tier = 0;
                state.city_grids[grid_idx].update_frontier_around(col, row);
            }

            // Rebuild road distances if any roads were removed.
            if !road_removals.is_empty() {
                state.city_grids[grid_idx].update_road_distances();
            }
        }

        // --- Density cap enforcement by road tier ---
        // Downgrade buildings that exceed the density allowed by their adjacent roads.
        let mut density_downgrades: Vec<(usize, usize, u8)> = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let cell = state.city_grids[grid_idx].cell(col, row);
                if cell.state != CellState::Building { continue; }
                let max_tier = best_adjacent_road_tier(&state.city_grids[grid_idx], col, row);
                let cap = road_tier_density_cap(max_tier);
                if cell.density > cap {
                    density_downgrades.push((col, row, cap));
                }
            }
        }

        for (col, row, cap) in density_downgrades {
            state.city_grids[grid_idx].cell_mut(col, row).density = cap;
        }
    }

    for entry in new_chronicles {
        state.chronicle.push(entry);
    }
}
