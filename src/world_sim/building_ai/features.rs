//! Pre-computed spatial features — the observation space for building decisions.
//!
//! The model never pathfinds, flood-fills, or does LOS from scratch.
//! These functions compute the answers and pack them into fixed structs.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::types::BuildMaterial;
use crate::world_sim::city_grid::{CellState, CellTerrain, ZoneType};
use crate::world_sim::state::{
    BuildingType, WorldState,
    tags,
};

// ---------------------------------------------------------------------------
// Top-level spatial features
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpatialFeatures {
    pub connectivity: ConnectivityFeatures,
    pub defensive: DefensiveFeatures,
    pub environmental: EnvironmentalFeatures,
    pub economic: EconomicFeatures,
    pub population: PopulationFeatures,
    pub garrison: GarrisonFeatures,
}

// ---------------------------------------------------------------------------
// Connectivity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConnectivityFeatures {
    /// (building_id_a, building_id_b, path_exists, distance_cells).
    pub key_building_paths: Vec<BuildingPathEntry>,
    /// Number of disconnected sub-graphs in the settlement layout.
    pub connected_components: u8,
    /// Grid cells that, if removed, disconnect the graph.
    pub chokepoints: Vec<(u16, u16)>,
    /// Fraction of residential cells that can reach any gate within 20 steps.
    pub evacuation_reachability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingPathEntry {
    pub id_a: u32,
    pub id_b: u32,
    pub path_exists: bool,
    pub distance: f32,
}

// ---------------------------------------------------------------------------
// Defensive
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DefensiveFeatures {
    /// Fraction of perimeter enclosed by walls (0.0–1.0).
    pub wall_coverage: f32,
    /// Per-segment detail.
    pub wall_segments: Vec<WallSegmentInfo>,
    /// Recent breach records from memory.
    pub breach_history: Vec<BreachRecord>,
    /// Fraction of perimeter visible from watchtower positions.
    pub watchtower_coverage: f32,
    /// Gate positions relative to settlement center.
    pub gate_positions: Vec<GateInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WallSegmentInfo {
    pub start: (u16, u16),
    pub end: (u16, u16),
    pub height: u8,
    pub thickness: u8,
    pub material: BuildMaterial,
    /// 0.0–1.0, 1.0 = pristine.
    pub condition: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachRecord {
    pub location: (u16, u16),
    pub tick: u64,
    pub breach_method: BreachMethod,
    pub attacker_type_tag: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BreachMethod {
    Ram = 0,
    Climb = 1,
    Jump = 2,
    Tunnel = 3,
    Fly = 4,
    Catapult = 5,
    Collapse = 6,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateInfo {
    pub position: (u16, u16),
    pub facing: super::types::Direction,
    pub reinforced: bool,
}

// ---------------------------------------------------------------------------
// Garrison
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GarrisonFeatures {
    /// Per-perimeter-cell effective defense value (structure + garrison + synergy).
    pub coverage_map: Vec<f32>,
    /// Sum of all garrison combat effectiveness.
    pub total_garrison_strength: f32,
    /// Per-perimeter-cell: ticks for first defender to arrive.
    pub response_time_map: Vec<f32>,
    /// Locations where unit + structure synergy creates force multiplication.
    pub synergy_hotspots: Vec<SynergyEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynergyEntry {
    pub position: (u16, u16),
    pub unit_id: u32,
    pub structure_id: u32,
    /// Multiplier on effective defense (e.g. 1.5 = archer on tower).
    pub multiplier: f32,
}

// ---------------------------------------------------------------------------
// Environmental
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvironmentalFeatures {
    /// Per-cell elevation (compact: only non-flat cells).
    pub elevation_map: Vec<ElevationEntry>,
    /// Cells at risk of flooding (below water table or river-adjacent).
    pub flood_risk_cells: Vec<(u16, u16)>,
    /// Clusters of adjacent wood buildings (fire propagation risk).
    pub fire_risk_clusters: Vec<FireCluster>,
    /// Prevailing wind direction (affects fire spread, siege arcs).
    pub wind_direction: (f32, f32),
    /// Current season index (0=spring, 1=summer, 2=autumn, 3=winter).
    pub season: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElevationEntry {
    pub cell: (u16, u16),
    pub elevation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireCluster {
    pub cells: Vec<(u16, u16)>,
    pub building_count: u8,
    pub total_wood_fraction: f32,
}

// ---------------------------------------------------------------------------
// Economic
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EconomicFeatures {
    /// Per-commodity stockpile levels.
    pub stockpiles: [f32; 8],
    /// Workers available by broad skill: (construction, masonry, labor, total).
    pub worker_counts: WorkerCounts,
    /// Occupancy / capacity per building type.
    pub utilization: Vec<UtilizationEntry>,
    /// Storage used / storage max across all warehouses+markets.
    pub storage_utilization: f32,
    /// Mean distance from resource sources to settlement stockpile.
    pub resource_access_distances: [f32; 8],
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct WorkerCounts {
    pub construction: u16,
    pub masonry: u16,
    pub labor: u16,
    pub total: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilizationEntry {
    pub building_id: u32,
    pub building_type: crate::world_sim::state::BuildingType,
    pub occupancy: u8,
    pub capacity: u8,
}

// ---------------------------------------------------------------------------
// Population
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PopulationFeatures {
    /// population / residential_capacity.
    pub housing_pressure: f32,
    /// Grid cells with NPC density above threshold.
    pub crowding_hotspots: Vec<(u16, u16)>,
    pub unhoused_count: u16,
    /// Population delta over last 200 ticks.
    pub growth_trend: f32,
    pub total_population: u16,
    pub total_residential_capacity: u16,
}

// ---------------------------------------------------------------------------
// Compute entry points (implemented by utility-functions workstream)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// BFS distance from a set of source cells. Returns per-cell distance (u16::MAX = unreachable).
/// `walkable` decides which cells can be traversed (4-connected).
fn bfs_distances(
    cols: usize,
    rows: usize,
    sources: &[(usize, usize)],
    walkable: impl Fn(usize, usize) -> bool,
    max_dist: u16,
) -> Vec<u16> {
    let n = cols * rows;
    let mut dist = vec![u16::MAX; n];
    let mut queue = VecDeque::new();
    for &(c, r) in sources {
        let idx = r * cols + c;
        if idx < n {
            dist[idx] = 0;
            queue.push_back((c, r));
        }
    }
    while let Some((cx, cy)) = queue.pop_front() {
        let d = dist[cy * cols + cx];
        if d >= max_dist {
            continue;
        }
        for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            if nx < 0 || ny < 0 || nx >= cols as i32 || ny >= rows as i32 {
                continue;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            if !walkable(nx, ny) {
                continue;
            }
            let ni = ny * cols + nx;
            if d + 1 < dist[ni] {
                dist[ni] = d + 1;
                queue.push_back((nx, ny));
            }
        }
    }
    dist
}

/// Compute connected components on walkable cells (4-connected). Returns count.
fn count_connected_components(
    cols: usize,
    rows: usize,
    walkable: impl Fn(usize, usize) -> bool,
) -> u8 {
    let n = cols * rows;
    let mut visited = vec![false; n];
    let mut components = 0u8;
    let mut queue = VecDeque::new();
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            if visited[idx] || !walkable(c, r) {
                continue;
            }
            // BFS flood fill
            components = components.saturating_add(1);
            visited[idx] = true;
            queue.push_back((c, r));
            while let Some((cx, cy)) = queue.pop_front() {
                for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= cols as i32 || ny >= rows as i32 {
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    let ni = ny * cols + nx;
                    if !visited[ni] && walkable(nx, ny) {
                        visited[ni] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }
        }
    }
    components
}

/// Naïve articulation-point detection via repeated BFS: remove candidate cell,
/// check if component count changes. Only checks cells on walkable/road tiles
/// with >=2 walkable neighbors. Capped to avoid blowup on large grids.
fn find_chokepoints(
    cols: usize,
    rows: usize,
    walkable: &dyn Fn(usize, usize) -> bool,
    max_candidates: usize,
) -> Vec<(u16, u16)> {
    // Gather candidate cells: walkable cells with >=3 walkable 4-neighbors.
    let mut candidates = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if !walkable(c, r) {
                continue;
            }
            let mut nw = 0u8;
            for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                let nx = c as i32 + dx;
                let ny = r as i32 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < cols && (ny as usize) < rows {
                    if walkable(nx as usize, ny as usize) {
                        nw += 1;
                    }
                }
            }
            if nw >= 3 {
                candidates.push((c, r));
            }
        }
    }
    // Limit to the most central candidates (heuristic: sort by distance to center).
    let cx = cols as f32 / 2.0;
    let cy = rows as f32 / 2.0;
    candidates.sort_by(|a, b| {
        let da = (a.0 as f32 - cx).powi(2) + (a.1 as f32 - cy).powi(2);
        let db = (b.0 as f32 - cx).powi(2) + (b.1 as f32 - cy).powi(2);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(max_candidates);

    // Baseline component count.
    let base = count_connected_components(cols, rows, walkable);

    let mut chokepoints = Vec::new();
    for &(ec, er) in &candidates {
        let new_count = count_connected_components(cols, rows, |c, r| {
            if c == ec && r == er {
                return false;
            }
            walkable(c, r)
        });
        if new_count > base {
            chokepoints.push((ec as u16, er as u16));
        }
    }
    chokepoints
}

/// Perimeter cell detection: cells on the outermost ring of developed area.
/// A developed cell is "perimeter" if it has at least one non-developed 4-neighbor
/// or is on the grid edge.
fn perimeter_cells(
    grid: &crate::world_sim::city_grid::CityGrid,
) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    for r in 0..grid.rows {
        for c in 0..grid.cols {
            let cell = grid.cell(c, r);
            if !matches!(cell.state, CellState::Building | CellState::Wall | CellState::Road | CellState::Plaza) {
                continue;
            }
            // On edge of grid = perimeter.
            if c == 0 || r == 0 || c + 1 >= grid.cols || r + 1 >= grid.rows {
                result.push((c, r));
                continue;
            }
            // Check 4-neighbors for undeveloped.
            let mut on_perimeter = false;
            for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                let nx = (c as i32 + dx) as usize;
                let ny = (r as i32 + dy) as usize;
                if grid.in_bounds(nx, ny) {
                    let ns = grid.cell(nx, ny).state;
                    if matches!(ns, CellState::Empty | CellState::Water) {
                        on_perimeter = true;
                        break;
                    }
                }
            }
            if on_perimeter {
                result.push((c, r));
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// compute_spatial_features
// ---------------------------------------------------------------------------

/// Compute all spatial features for a settlement. Called once per decision cadence.
pub fn compute_spatial_features(state: &WorldState, settlement_id: u32) -> SpatialFeatures {
    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return SpatialFeatures::default(),
    };
    let grid_idx = match settlement.city_grid_idx {
        Some(idx) if idx < state.city_grids.len() => idx,
        _ => return SpatialFeatures::default(),
    };
    let grid = &state.city_grids[grid_idx];
    let cols = grid.cols;
    let rows = grid.rows;

    // -----------------------------------------------------------------------
    // 1. Connectivity
    // -----------------------------------------------------------------------

    // Collect key buildings: one per type that matters.
    let building_range = state.group_index.settlement_buildings(settlement_id);
    let mut key_buildings: Vec<(u32, usize, usize)> = Vec::new(); // (entity_id, col, row)
    for idx in building_range.clone() {
        if idx >= state.entities.len() { break; }
        let e = &state.entities[idx];
        if !e.alive { continue; }
        if let Some(bd) = &e.building {
            match bd.building_type {
                BuildingType::Barracks | BuildingType::Market | BuildingType::Warehouse
                | BuildingType::Gate | BuildingType::Temple | BuildingType::GuildHall
                | BuildingType::Treasury | BuildingType::Watchtower => {
                    key_buildings.push((e.id, bd.grid_col as usize, bd.grid_row as usize));
                }
                _ => {}
            }
        }
    }
    // Cap to 12 key buildings to limit O(n^2) pathfinding.
    key_buildings.truncate(12);

    let mut key_building_paths = Vec::new();
    for i in 0..key_buildings.len() {
        for j in (i + 1)..key_buildings.len() {
            let (id_a, ca, ra) = key_buildings[i];
            let (id_b, cb, rb) = key_buildings[j];
            let path = grid.find_path((ca, ra), (cb, rb));
            let (path_exists, distance) = match &path {
                Some(p) => (true, p.len() as f32),
                None => (false, f32::MAX),
            };
            key_building_paths.push(BuildingPathEntry {
                id_a,
                id_b,
                path_exists,
                distance,
            });
        }
    }

    // Connected components among walkable cells.
    let connected_components = count_connected_components(cols, rows, |c, r| grid.is_walkable(c, r));

    // Chokepoints (articulation points, capped to 60 candidates).
    let chokepoints = find_chokepoints(cols, rows, &|c, r| grid.is_walkable(c, r), 60);

    // Evacuation reachability: fraction of residential cells within 20 steps of a gate.
    let gate_cells: Vec<(usize, usize)> = {
        let mut v = Vec::new();
        for idx in building_range.clone() {
            if idx >= state.entities.len() { break; }
            let e = &state.entities[idx];
            if !e.alive { continue; }
            if let Some(bd) = &e.building {
                if bd.building_type == BuildingType::Gate {
                    v.push((bd.grid_col as usize, bd.grid_row as usize));
                }
            }
        }
        v
    };
    let gate_dist = bfs_distances(cols, rows, &gate_cells, |c, r| grid.is_walkable(c, r), 20);
    let mut residential_count = 0u32;
    let mut residential_reachable = 0u32;
    for r in 0..rows {
        for c in 0..cols {
            let cell = grid.cell(c, r);
            if cell.state == CellState::Building && cell.zone == ZoneType::Residential {
                residential_count += 1;
                if gate_dist[r * cols + c] <= 20 {
                    residential_reachable += 1;
                }
            }
        }
    }
    let evacuation_reachability = if residential_count > 0 {
        residential_reachable as f32 / residential_count as f32
    } else {
        1.0
    };

    let connectivity = ConnectivityFeatures {
        key_building_paths,
        connected_components,
        chokepoints,
        evacuation_reachability,
    };

    // -----------------------------------------------------------------------
    // 2. Defensive
    // -----------------------------------------------------------------------

    let perimeter = perimeter_cells(grid);
    let perimeter_total = perimeter.len() as f32;

    // Wall coverage: fraction of perimeter cells that are Wall or Gate.
    let mut wall_perimeter_count = 0u32;
    let mut wall_segments: Vec<WallSegmentInfo> = Vec::new();
    let mut gate_positions: Vec<GateInfo> = Vec::new();
    let mut watchtower_positions: Vec<(usize, usize)> = Vec::new();

    for idx in building_range.clone() {
        if idx >= state.entities.len() { break; }
        let e = &state.entities[idx];
        if !e.alive { continue; }
        if let Some(bd) = &e.building {
            match bd.building_type {
                BuildingType::Wall => {
                    let start = (bd.grid_col, bd.grid_row);
                    let end_col = bd.grid_col.saturating_add(bd.footprint_w as u16).saturating_sub(1);
                    let end_row = bd.grid_row.saturating_add(bd.footprint_h as u16).saturating_sub(1);
                    wall_segments.push(WallSegmentInfo {
                        start,
                        end: (end_col, end_row),
                        height: (bd.tier + 1).min(3),
                        thickness: 1,
                        material: if bd.tier >= 2 { BuildMaterial::Stone } else { BuildMaterial::Wood },
                        condition: bd.construction_progress,
                    });
                }
                BuildingType::Gate => {
                    // Determine facing heuristically from position relative to center.
                    let dc = bd.grid_col as f32 - grid.center.0 as f32;
                    let dr = bd.grid_row as f32 - grid.center.1 as f32;
                    let facing = if dc.abs() > dr.abs() {
                        if dc > 0.0 { super::types::Direction::East } else { super::types::Direction::West }
                    } else if dr > 0.0 {
                        super::types::Direction::South
                    } else {
                        super::types::Direction::North
                    };
                    gate_positions.push(GateInfo {
                        position: (bd.grid_col, bd.grid_row),
                        facing,
                        reinforced: bd.tier >= 2,
                    });
                }
                BuildingType::Watchtower => {
                    watchtower_positions.push((bd.grid_col as usize, bd.grid_row as usize));
                }
                _ => {}
            }
        }
    }
    // Count perimeter cells occupied by Wall state.
    for &(pc, pr) in &perimeter {
        if grid.cell(pc, pr).state == CellState::Wall {
            wall_perimeter_count += 1;
        }
    }
    let wall_coverage = if perimeter_total > 0.0 {
        wall_perimeter_count as f32 / perimeter_total
    } else {
        0.0
    };

    // Watchtower coverage: fraction of perimeter cells within LOS range (10 cells, simple distance).
    let tower_range: f32 = 10.0;
    let mut tower_visible = 0u32;
    for &(pc, pr) in &perimeter {
        let visible = watchtower_positions.iter().any(|&(tc, tr)| {
            let dx = (pc as f32 - tc as f32).abs();
            let dy = (pr as f32 - tr as f32).abs();
            (dx * dx + dy * dy).sqrt() <= tower_range
        });
        if visible {
            tower_visible += 1;
        }
    }
    let watchtower_coverage = if perimeter_total > 0.0 {
        tower_visible as f32 / perimeter_total
    } else {
        0.0
    };

    let defensive = DefensiveFeatures {
        wall_coverage,
        wall_segments,
        breach_history: Vec::new(), // populated from ConstructionMemory by caller
        watchtower_coverage,
        gate_positions,
    };

    // -----------------------------------------------------------------------
    // 3. Environmental
    // -----------------------------------------------------------------------

    // Elevation from cell terrain (CellTerrain → synthetic elevation).
    let mut elevation_map = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let cell = grid.cell(c, r);
            let elev = match cell.terrain {
                CellTerrain::Flat => continue,
                CellTerrain::Slope => 1.0,
                CellTerrain::Steep => 2.0,
                CellTerrain::Water => -1.0,
                CellTerrain::Cliff => 3.0,
            };
            elevation_map.push(ElevationEntry {
                cell: (c as u16, r as u16),
                elevation: elev,
            });
        }
    }

    // Flood risk: water-adjacent cells at low elevation (Flat terrain next to Water).
    let mut flood_risk_cells = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let cell = grid.cell(c, r);
            if cell.terrain == CellTerrain::Water {
                continue;
            }
            if cell.terrain != CellTerrain::Flat {
                continue;
            }
            // Check if any 4-neighbor is water.
            let mut near_water = false;
            for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                let nx = c as i32 + dx;
                let ny = r as i32 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < cols && (ny as usize) < rows {
                    if grid.cell(nx as usize, ny as usize).terrain == CellTerrain::Water {
                        near_water = true;
                        break;
                    }
                }
            }
            if near_water {
                flood_risk_cells.push((c as u16, r as u16));
            }
        }
    }

    // Fire risk: clusters of adjacent wood buildings (Wood = tier < 2 buildings).
    // Simple flood-fill on building cells that are likely wood.
    let mut fire_visited = vec![false; cols * rows];
    let mut fire_risk_clusters = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let fi = r * cols + c;
            if fire_visited[fi] {
                continue;
            }
            let cell = grid.cell(c, r);
            if cell.state != CellState::Building {
                continue;
            }
            // Heuristic: low density or residential zone ≈ wood buildings.
            if cell.density >= 2 {
                continue; // stone/landmark
            }
            fire_visited[fi] = true;
            let mut cluster_cells = vec![(c as u16, r as u16)];
            let mut queue = VecDeque::new();
            queue.push_back((c, r));
            while let Some((cx, cy)) = queue.pop_front() {
                for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= cols as i32 || ny >= rows as i32 {
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    let ni = ny * cols + nx;
                    if fire_visited[ni] {
                        continue;
                    }
                    let nc = grid.cell(nx, ny);
                    if nc.state == CellState::Building && nc.density < 2 {
                        fire_visited[ni] = true;
                        cluster_cells.push((nx as u16, ny as u16));
                        queue.push_back((nx, ny));
                    }
                }
            }
            if cluster_cells.len() >= 3 {
                let bcount = cluster_cells.len() as u8;
                fire_risk_clusters.push(FireCluster {
                    cells: cluster_cells,
                    building_count: bcount,
                    total_wood_fraction: 1.0, // all low-density ≈ wood
                });
            }
        }
    }

    // Wind direction and season: derive from tick.
    let season = ((state.tick / 2000) % 4) as u8;
    // Simple deterministic wind from settlement position hash.
    let wind_angle = ((settlement_id.wrapping_mul(2654435761)) as f32 / u32::MAX as f32) * std::f32::consts::TAU;
    let wind_direction = (wind_angle.cos(), wind_angle.sin());

    let environmental = EnvironmentalFeatures {
        elevation_map,
        flood_risk_cells,
        fire_risk_clusters,
        wind_direction,
        season,
    };

    // -----------------------------------------------------------------------
    // 4. Economic
    // -----------------------------------------------------------------------

    let stockpiles = {
        let mut s = [0.0f32; 8];
        for (i, val) in settlement.stockpile.iter().enumerate() {
            if i < 8 {
                s[i] = *val;
            }
        }
        s
    };

    // Worker counts by skill tag.
    let npc_range = state.group_index.settlement_npcs(settlement_id);
    let mut wc = WorkerCounts::default();
    for idx in npc_range.clone() {
        if idx >= state.entities.len() { break; }
        let e = &state.entities[idx];
        if !e.alive { continue; }
        wc.total += 1;
        if let Some(npc) = &e.npc {
            for &(tag_hash, weight) in &npc.behavior_profile {
                if weight < 0.1 { continue; }
                if tag_hash == tags::CONSTRUCTION || tag_hash == tags::ARCHITECTURE {
                    wc.construction += 1;
                    break;
                }
                if tag_hash == tags::MASONRY {
                    wc.masonry += 1;
                    break;
                }
                if tag_hash == tags::LABOR {
                    wc.labor += 1;
                    break;
                }
            }
        }
    }

    // Building utilization.
    let mut utilization = Vec::new();
    let mut total_storage_used = 0.0f32;
    let mut total_storage_cap = 0.0f32;
    for idx in building_range.clone() {
        if idx >= state.entities.len() { break; }
        let e = &state.entities[idx];
        if !e.alive { continue; }
        if let Some(bd) = &e.building {
            let cap = bd.work_capacity.max(bd.residential_capacity);
            if cap > 0 {
                let occupancy = (bd.worker_ids.len() + bd.resident_ids.len()) as u8;
                utilization.push(UtilizationEntry {
                    building_id: e.id,
                    building_type: bd.building_type,
                    occupancy,
                    capacity: cap,
                });
            }
            if bd.storage_capacity > 0.0 {
                total_storage_used += bd.storage_used();
                total_storage_cap += bd.storage_capacity;
            }
        }
    }
    let storage_utilization = if total_storage_cap > 0.0 {
        total_storage_used / total_storage_cap
    } else {
        0.0
    };

    let economic = EconomicFeatures {
        stockpiles,
        worker_counts: wc,
        utilization,
        storage_utilization,
        resource_access_distances: [0.0; 8], // TODO: compute from resource node BFS
    };

    // -----------------------------------------------------------------------
    // 5. Population
    // -----------------------------------------------------------------------

    let total_population = npc_range.len() as u16;
    let mut total_residential_capacity = 0u16;
    let mut housed_set = std::collections::HashSet::new();
    for idx in building_range.clone() {
        if idx >= state.entities.len() { break; }
        let e = &state.entities[idx];
        if !e.alive { continue; }
        if let Some(bd) = &e.building {
            total_residential_capacity += bd.residential_capacity as u16;
            for &rid in &bd.resident_ids {
                housed_set.insert(rid);
            }
        }
    }
    let unhoused_count = total_population.saturating_sub(housed_set.len() as u16);
    let housing_pressure = if total_residential_capacity > 0 {
        total_population as f32 / total_residential_capacity as f32
    } else if total_population > 0 {
        f32::MAX
    } else {
        0.0
    };

    // Crowding hotspots: grid cells with >3 NPCs in the same cell.
    let mut cell_npc_counts: std::collections::HashMap<(u16, u16), u16> = std::collections::HashMap::new();
    for idx in npc_range {
        if idx >= state.entities.len() { break; }
        let e = &state.entities[idx];
        if !e.alive { continue; }
        let gc = grid.world_to_grid(e.pos, settlement.pos);
        let key = (gc.0 as u16, gc.1 as u16);
        *cell_npc_counts.entry(key).or_insert(0) += 1;
    }
    let crowding_hotspots: Vec<(u16, u16)> = cell_npc_counts
        .into_iter()
        .filter(|(_, count)| *count > 3)
        .map(|(pos, _)| pos)
        .collect();

    // Growth trend: approximated from population vs capacity. Positive if pop < capacity.
    let growth_trend = if total_residential_capacity > 0 {
        (total_residential_capacity as f32 - total_population as f32)
            / total_residential_capacity as f32
    } else {
        0.0
    };

    let population = PopulationFeatures {
        housing_pressure,
        crowding_hotspots,
        unhoused_count,
        growth_trend,
        total_population,
        total_residential_capacity,
    };

    // -----------------------------------------------------------------------
    // 6. Garrison (embedded in spatial features)
    // -----------------------------------------------------------------------

    let garrison = compute_garrison_features(state, settlement_id);

    SpatialFeatures {
        connectivity,
        defensive,
        environmental,
        economic,
        population,
        garrison,
    }
}

// ---------------------------------------------------------------------------
// compute_garrison_features
// ---------------------------------------------------------------------------

/// Compute garrison-specific features (subset, can be called more frequently).
pub fn compute_garrison_features(state: &WorldState, settlement_id: u32) -> GarrisonFeatures {
    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return GarrisonFeatures::default(),
    };
    let grid_idx = match settlement.city_grid_idx {
        Some(idx) if idx < state.city_grids.len() => idx,
        _ => return GarrisonFeatures::default(),
    };
    let grid = &state.city_grids[grid_idx];
    let cols = grid.cols;
    let rows = grid.rows;

    // Identify garrison NPCs: combat-tagged NPCs assigned to defensive buildings
    // (Barracks, Watchtower, Wall, Gate).
    let npc_range = state.group_index.settlement_npcs(settlement_id);
    let building_range = state.group_index.settlement_buildings(settlement_id);

    // Collect defensive building positions and IDs.
    struct DefBuilding {
        id: u32,
        col: usize,
        row: usize,
        btype: BuildingType,
    }
    let mut def_buildings: Vec<DefBuilding> = Vec::new();
    for idx in building_range {
        if idx >= state.entities.len() { break; }
        let e = &state.entities[idx];
        if !e.alive { continue; }
        if let Some(bd) = &e.building {
            match bd.building_type {
                BuildingType::Barracks | BuildingType::Watchtower
                | BuildingType::Wall | BuildingType::Gate => {
                    def_buildings.push(DefBuilding {
                        id: e.id,
                        col: bd.grid_col as usize,
                        row: bd.grid_row as usize,
                        btype: bd.building_type,
                    });
                }
                _ => {}
            }
        }
    }

    // Collect garrison NPCs: those with combat tags in behavior_profile whose
    // work_building_id points to a defensive building, or who are inside one.
    struct GarrisonNpc {
        id: u32,
        col: usize,
        row: usize,
        combat_value: f32,
        is_ranged: bool,
    }
    let def_building_ids: std::collections::HashSet<u32> =
        def_buildings.iter().map(|b| b.id).collect();
    let mut garrison_npcs: Vec<GarrisonNpc> = Vec::new();

    for idx in npc_range {
        if idx >= state.entities.len() { break; }
        let e = &state.entities[idx];
        if !e.alive { continue; }
        let npc = match &e.npc {
            Some(n) => n,
            None => continue,
        };
        // Check if assigned to or inside a defensive building.
        let in_garrison = npc.work_building_id.map_or(false, |wid| def_building_ids.contains(&wid))
            || npc.inside_building_id.map_or(false, |bid| def_building_ids.contains(&bid));
        if !in_garrison {
            continue;
        }
        // Check for combat tags.
        let has_combat = npc.behavior_profile.iter().any(|&(t, w)| {
            w > 0.1 && (t == tags::COMBAT || t == tags::MELEE || t == tags::RANGED || t == tags::DEFENSE)
        });
        if !has_combat {
            continue;
        }
        let is_ranged = npc.behavior_profile.iter().any(|&(t, w)| w > 0.1 && t == tags::RANGED);
        let gc = grid.world_to_grid(e.pos, settlement.pos);
        let combat_value = e.attack_damage + e.hp * 0.1 + e.armor * 0.5;
        garrison_npcs.push(GarrisonNpc {
            id: e.id,
            col: gc.0,
            row: gc.1,
            combat_value,
            is_ranged,
        });
    }

    let total_garrison_strength: f32 = garrison_npcs.iter().map(|g| g.combat_value).sum();

    // Perimeter cells.
    let perim = perimeter_cells(grid);
    if perim.is_empty() {
        return GarrisonFeatures {
            coverage_map: Vec::new(),
            total_garrison_strength,
            response_time_map: Vec::new(),
            synergy_hotspots: Vec::new(),
        };
    }

    // BFS from all garrison NPC positions to get response times.
    let garrison_sources: Vec<(usize, usize)> = garrison_npcs.iter().map(|g| (g.col, g.row)).collect();
    let garrison_dist = bfs_distances(cols, rows, &garrison_sources, |c, r| grid.is_walkable(c, r), 30);

    let mut coverage_map = Vec::with_capacity(perim.len());
    let mut response_time_map = Vec::with_capacity(perim.len());
    let mut synergy_hotspots: Vec<SynergyEntry> = Vec::new();

    for &(pc, pr) in &perim {
        let pi = pr * cols + pc;

        // Structural defense value from cell state.
        let structural = match grid.cell(pc, pr).state {
            CellState::Wall => 3.0,
            CellState::Building => 1.0,
            _ => 0.5,
        };

        // Garrison contribution: sum of garrison NPCs weighted by proximity.
        let mut garrison_value = 0.0f32;
        for g in &garrison_npcs {
            let dx = (pc as f32 - g.col as f32).abs();
            let dy = (pr as f32 - g.row as f32).abs();
            let dist = (dx * dx + dy * dy).sqrt().max(1.0);
            garrison_value += g.combat_value / dist;
        }

        // Synergy: check for force-multiplying combinations.
        for g in &garrison_npcs {
            let dx = (pc as f32 - g.col as f32).abs();
            let dy = (pr as f32 - g.row as f32).abs();
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > 5.0 { continue; }

            // Check synergy with nearby defensive buildings.
            for db in &def_buildings {
                let bx = (g.col as f32 - db.col as f32).abs();
                let by = (g.row as f32 - db.row as f32).abs();
                let bdist = (bx * bx + by * by).sqrt();
                if bdist > 3.0 { continue; }

                let multiplier = match (g.is_ranged, db.btype) {
                    (true, BuildingType::Watchtower) => 1.5,  // archer on tower
                    (false, BuildingType::Gate) => 1.3,        // fighter at gate
                    (false, BuildingType::Wall) => 1.2,        // fighter on wall
                    _ => 1.0,
                };
                if multiplier > 1.0 {
                    synergy_hotspots.push(SynergyEntry {
                        position: (pc as u16, pr as u16),
                        unit_id: g.id,
                        structure_id: db.id,
                        multiplier,
                    });
                    garrison_value *= multiplier;
                }
            }
        }

        let effective_defense = structural + garrison_value;
        coverage_map.push(effective_defense);

        // Response time for this perimeter cell.
        let resp = garrison_dist[pi];
        response_time_map.push(if resp == u16::MAX { f32::MAX } else { resp as f32 });
    }

    // Deduplicate synergy hotspots (keep highest multiplier per position).
    synergy_hotspots.sort_by(|a, b| {
        a.position.cmp(&b.position)
            .then(b.multiplier.partial_cmp(&a.multiplier).unwrap_or(std::cmp::Ordering::Equal))
    });
    synergy_hotspots.dedup_by(|a, b| a.position == b.position && a.unit_id == b.unit_id);

    GarrisonFeatures {
        coverage_map,
        total_garrison_strength,
        response_time_map,
        synergy_hotspots,
    }
}
