//! Pre-computed spatial features — the observation space for building decisions.
//!
//! The model never pathfinds, flood-fills, or does LOS from scratch.
//! These functions compute the answers and pack them into fixed structs.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::types::BuildMaterial;
use crate::world_sim::state::{
    BuildingType, WorldState,
    tags,
};
use crate::world_sim::voxel::{world_to_voxel, VoxelMaterial};

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

/// Virtual grid size for spatial feature computations (VoxelWorld is unbounded).
const VIRT: usize = 64;

/// Convert a world-space position to a virtual grid cell coordinate.
/// Settlement center maps to (VIRT/2, VIRT/2); each grid cell = 1 voxel unit.
fn world_to_virtual(wx: f32, wy: f32, settlement_x: f32, settlement_y: f32) -> (usize, usize) {
    let half = (VIRT / 2) as f32;
    let c = (wx - settlement_x + half).round() as i32;
    let r = (wy - settlement_y + half).round() as i32;
    let c = c.clamp(0, VIRT as i32 - 1) as usize;
    let r = r.clamp(0, VIRT as i32 - 1) as usize;
    (c, r)
}

/// Build a set of developed cells (cells occupied by a building entity) in the virtual grid.
fn build_developed_cells(state: &WorldState, settlement_id: u32) -> Vec<bool> {
    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return vec![false; VIRT * VIRT],
    };
    let mut developed = vec![false; VIRT * VIRT];
    let building_range = state.group_index.settlement_buildings(settlement_id);
    for idx in building_range {
        if idx >= state.entities.len() { break; }
        let e = &state.entities[idx];
        if !e.alive { continue; }
        let (c, r) = world_to_virtual(e.pos.0, e.pos.1, settlement.pos.0, settlement.pos.1);
        developed[r * VIRT + c] = true;
    }
    developed
}

/// Perimeter cells: virtual grid cells that contain a building and have at least one
/// non-developed 4-neighbor (or lie on the grid edge).
fn perimeter_cells_virtual(developed: &[bool]) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    for r in 0..VIRT {
        for c in 0..VIRT {
            if !developed[r * VIRT + c] {
                continue;
            }
            if c == 0 || r == 0 || c + 1 >= VIRT || r + 1 >= VIRT {
                result.push((c, r));
                continue;
            }
            let on_perimeter = [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)].iter().any(|&(dx, dy)| {
                let nx = (c as i32 + dx) as usize;
                let ny = (r as i32 + dy) as usize;
                !developed[ny * VIRT + nx]
            });
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
    let cols = VIRT;
    let rows = VIRT;
    let sx = settlement.pos.0;
    let sy = settlement.pos.1;

    let developed = build_developed_cells(state, settlement_id);
    let is_developed = |c: usize, r: usize| developed[r * VIRT + c];

    // -----------------------------------------------------------------------
    // 1. Connectivity
    // -----------------------------------------------------------------------

    let building_range = state.group_index.settlement_buildings(settlement_id);

    // Collect key buildings: one per type that matters.
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
                    let (c, r) = world_to_virtual(e.pos.0, e.pos.1, sx, sy);
                    key_buildings.push((e.id, c, r));
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
            // Simple Euclidean distance (BFS over entities would be expensive without a nav grid).
            let dx = ca as f32 - cb as f32;
            let dy = ra as f32 - rb as f32;
            let distance = (dx * dx + dy * dy).sqrt();
            key_building_paths.push(BuildingPathEntry {
                id_a,
                id_b,
                path_exists: true,
                distance,
            });
        }
    }

    // Connected components among developed cells.
    let connected_components = count_connected_components(cols, rows, is_developed);

    // Chokepoints (articulation points among developed cells, capped to 60).
    let chokepoints = find_chokepoints(cols, rows, &is_developed, 60);

    // Evacuation reachability: fraction of residential buildings within 20 cells of a gate.
    let gate_positions_vc: Vec<(usize, usize)> = {
        let mut v = Vec::new();
        for idx in building_range.clone() {
            if idx >= state.entities.len() { break; }
            let e = &state.entities[idx];
            if !e.alive { continue; }
            if let Some(bd) = &e.building {
                if bd.building_type == BuildingType::Gate {
                    let (c, r) = world_to_virtual(e.pos.0, e.pos.1, sx, sy);
                    v.push((c, r));
                }
            }
        }
        v
    };
    let gate_dist = bfs_distances(cols, rows, &gate_positions_vc, is_developed, 20);
    let mut residential_count = 0u32;
    let mut residential_reachable = 0u32;
    for idx in building_range.clone() {
        if idx >= state.entities.len() { break; }
        let e = &state.entities[idx];
        if !e.alive { continue; }
        if let Some(bd) = &e.building {
            if bd.building_type == BuildingType::House || bd.building_type == BuildingType::Longhouse {
                residential_count += 1;
                let (c, r) = world_to_virtual(e.pos.0, e.pos.1, sx, sy);
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

    let perimeter = perimeter_cells_virtual(&developed);
    let perimeter_total = perimeter.len() as f32;

    let mut wall_perimeter_count = 0u32;
    let mut wall_segments: Vec<WallSegmentInfo> = Vec::new();
    let mut gate_infos: Vec<GateInfo> = Vec::new();
    let mut watchtower_positions: Vec<(usize, usize)> = Vec::new();

    // Build set of wall entity virtual positions for perimeter counting.
    let mut wall_cells: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

    let half = (VIRT / 2) as f32;
    for idx in building_range.clone() {
        if idx >= state.entities.len() { break; }
        let e = &state.entities[idx];
        if !e.alive { continue; }
        if let Some(bd) = &e.building {
            let (vc, vr) = world_to_virtual(e.pos.0, e.pos.1, sx, sy);
            match bd.building_type {
                BuildingType::Wall => {
                    let start = (vc as u16, vr as u16);
                    let end_col = (vc as u16).saturating_add(bd.footprint_w as u16).saturating_sub(1);
                    let end_row = (vr as u16).saturating_add(bd.footprint_h as u16).saturating_sub(1);
                    wall_segments.push(WallSegmentInfo {
                        start,
                        end: (end_col, end_row),
                        height: (bd.tier + 1).min(3),
                        thickness: 1,
                        material: if bd.tier >= 2 { BuildMaterial::Stone } else { BuildMaterial::Wood },
                        condition: bd.construction_progress,
                    });
                    wall_cells.insert((vc, vr));
                }
                BuildingType::Gate => {
                    let dc = e.pos.0 - sx;
                    let dr = e.pos.1 - sy;
                    let facing = if dc.abs() > dr.abs() {
                        if dc > 0.0 { super::types::Direction::East } else { super::types::Direction::West }
                    } else if dr > 0.0 {
                        super::types::Direction::South
                    } else {
                        super::types::Direction::North
                    };
                    gate_infos.push(GateInfo {
                        position: (vc as u16, vr as u16),
                        facing,
                        reinforced: bd.tier >= 2,
                    });
                    wall_cells.insert((vc, vr));
                }
                BuildingType::Watchtower => {
                    watchtower_positions.push((vc, vr));
                }
                _ => {}
            }
        }
    }
    // Count perimeter cells occupied by wall entities.
    for &(pc, pr) in &perimeter {
        if wall_cells.contains(&(pc, pr)) {
            wall_perimeter_count += 1;
        }
    }
    let wall_coverage = if perimeter_total > 0.0 {
        wall_perimeter_count as f32 / perimeter_total
    } else {
        0.0
    };

    // Watchtower coverage: fraction of perimeter cells within tower range.
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
        gate_positions: gate_infos,
    };

    // -----------------------------------------------------------------------
    // 3. Environmental
    // -----------------------------------------------------------------------

    // Elevation from VoxelWorld surface height (relative to sea level).
    let mut elevation_map = Vec::new();
    // Sample every 4th cell to avoid O(VIRT^2) per-cell voxel lookups being too slow.
    for r in (0..VIRT).step_by(4) {
        for c in (0..VIRT).step_by(4) {
            let wx = sx + (c as f32 - half);
            let wy = sy + (r as f32 - half);
            let (vx, vy, _) = world_to_voxel(wx, wy, 0.0);
            let sz = state.voxel_world.surface_height(vx, vy);
            let elev = sz as f32 - state.voxel_world.sea_level as f32;
            if elev.abs() > 0.5 {
                elevation_map.push(ElevationEntry {
                    cell: (c as u16, r as u16),
                    elevation: elev,
                });
            }
        }
    }

    // Flood risk: cells at or below sea level adjacent to water voxels.
    let mut flood_risk_cells = Vec::new();
    for r in (0..VIRT).step_by(4) {
        for c in (0..VIRT).step_by(4) {
            let wx = sx + (c as f32 - half);
            let wy = sy + (r as f32 - half);
            let (vx, vy, _) = world_to_voxel(wx, wy, 0.0);
            let sz = state.voxel_world.surface_height(vx, vy);
            if sz > state.voxel_world.sea_level + 2 {
                continue; // high ground
            }
            // Check if any 4-neighbor has water surface.
            let near_water = [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)].iter().any(|&(dx, dy)| {
                let nsz = state.voxel_world.surface_height(vx + dx, vy + dy);
                let nv = state.voxel_world.get_voxel(vx + dx, vy + dy, nsz - 1);
                matches!(nv.material, VoxelMaterial::Water)
            });
            if near_water {
                flood_risk_cells.push((c as u16, r as u16));
            }
        }
    }

    // Fire risk: clusters of adjacent low-tier building entities.
    let mut fire_visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut fire_risk_clusters = Vec::new();
    for idx in building_range.clone() {
        if idx >= state.entities.len() { break; }
        let e = &state.entities[idx];
        if !e.alive { continue; }
        let bd = match &e.building {
            Some(bd) => bd,
            None => continue,
        };
        if fire_visited.contains(&e.id) { continue; }
        if bd.tier >= 2 { continue; } // stone/landmark
        fire_visited.insert(e.id);
        let (c0, r0) = world_to_virtual(e.pos.0, e.pos.1, sx, sy);
        let mut cluster_cells = vec![(c0 as u16, r0 as u16)];
        // BFS over nearby low-tier building entities.
        let mut queue = VecDeque::new();
        queue.push_back(e.id);
        // Collect all low-tier buildings for proximity check.
        let low_tier: Vec<(u32, usize, usize)> = building_range.clone()
            .filter_map(|i| {
                let ent = state.entities.get(i)?;
                if !ent.alive { return None; }
                let bdata = ent.building.as_ref()?;
                if bdata.tier >= 2 { return None; }
                let (c, r) = world_to_virtual(ent.pos.0, ent.pos.1, sx, sy);
                Some((ent.id, c, r))
            })
            .collect();
        while let Some(bid) = queue.pop_front() {
            let (_, bc, br) = match low_tier.iter().find(|&&(id, _, _)| id == bid) {
                Some(x) => *x,
                None => continue,
            };
            for &(other_id, oc, or_) in &low_tier {
                if fire_visited.contains(&other_id) { continue; }
                let dx = (bc as i32 - oc as i32).abs();
                let dy = (br as i32 - or_ as i32).abs();
                if dx <= 2 && dy <= 2 {
                    fire_visited.insert(other_id);
                    cluster_cells.push((oc as u16, or_ as u16));
                    queue.push_back(other_id);
                }
            }
        }
        if cluster_cells.len() >= 3 {
            let bcount = cluster_cells.len() as u8;
            fire_risk_clusters.push(FireCluster {
                cells: cluster_cells,
                building_count: bcount,
                total_wood_fraction: 1.0,
            });
        }
    }

    // Wind direction and season.
    let season = ((state.tick / 2000) % 4) as u8;
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
            if i < 8 { s[i] = *val; }
        }
        s
    };

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
        resource_access_distances: [0.0; 8],
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

    // Crowding hotspots: virtual grid cells with >3 NPCs.
    let mut cell_npc_counts: std::collections::HashMap<(u16, u16), u16> = std::collections::HashMap::new();
    for idx in npc_range {
        if idx >= state.entities.len() { break; }
        let e = &state.entities[idx];
        if !e.alive { continue; }
        let (c, r) = world_to_virtual(e.pos.0, e.pos.1, sx, sy);
        let key = (c as u16, r as u16);
        *cell_npc_counts.entry(key).or_insert(0) += 1;
    }
    let crowding_hotspots: Vec<(u16, u16)> = cell_npc_counts
        .into_iter()
        .filter(|(_, count)| *count > 3)
        .map(|(pos, _)| pos)
        .collect();

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
    // 6. Garrison
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
    let sx = settlement.pos.0;
    let sy = settlement.pos.1;
    let cols = VIRT;
    let rows = VIRT;

    let npc_range = state.group_index.settlement_npcs(settlement_id);
    let building_range = state.group_index.settlement_buildings(settlement_id);

    // Collect defensive building virtual positions.
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
                    let (col, row) = world_to_virtual(e.pos.0, e.pos.1, sx, sy);
                    def_buildings.push(DefBuilding {
                        id: e.id,
                        col,
                        row,
                        btype: bd.building_type,
                    });
                }
                _ => {}
            }
        }
    }

    // Collect garrison NPCs.
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
        let in_garrison = npc.work_building_id.map_or(false, |wid| def_building_ids.contains(&wid))
            || npc.inside_building_id.map_or(false, |bid| def_building_ids.contains(&bid));
        if !in_garrison { continue; }
        let has_combat = npc.behavior_profile.iter().any(|&(t, w)| {
            w > 0.1 && (t == tags::COMBAT || t == tags::MELEE || t == tags::RANGED || t == tags::DEFENSE)
        });
        if !has_combat { continue; }
        let is_ranged = npc.behavior_profile.iter().any(|&(t, w)| w > 0.1 && t == tags::RANGED);
        let (col, row) = world_to_virtual(e.pos.0, e.pos.1, sx, sy);
        let combat_value = e.attack_damage + e.hp * 0.1 + e.armor * 0.5;
        garrison_npcs.push(GarrisonNpc { id: e.id, col, row, combat_value, is_ranged });
    }

    let total_garrison_strength: f32 = garrison_npcs.iter().map(|g| g.combat_value).sum();

    let developed = build_developed_cells(state, settlement_id);
    let perim = perimeter_cells_virtual(&developed);
    if perim.is_empty() {
        return GarrisonFeatures {
            coverage_map: Vec::new(),
            total_garrison_strength,
            response_time_map: Vec::new(),
            synergy_hotspots: Vec::new(),
        };
    }

    // BFS from garrison NPC positions.
    let garrison_sources: Vec<(usize, usize)> = garrison_npcs.iter().map(|g| (g.col, g.row)).collect();
    let is_developed = |c: usize, r: usize| developed[r * VIRT + c];
    let garrison_dist = bfs_distances(cols, rows, &garrison_sources, is_developed, 30);

    let mut coverage_map = Vec::with_capacity(perim.len());
    let mut response_time_map = Vec::with_capacity(perim.len());
    let mut synergy_hotspots: Vec<SynergyEntry> = Vec::new();

    for &(pc, pr) in &perim {
        let pi = pr * cols + pc;

        // Structural defense: wall entity at this cell = 3.0, any building = 1.0.
        let has_wall = def_buildings.iter().any(|db| {
            db.col == pc && db.row == pr
                && matches!(db.btype, BuildingType::Wall | BuildingType::Gate)
        });
        let structural = if has_wall { 3.0 } else if developed[pi] { 1.0 } else { 0.5 };

        let mut garrison_value = 0.0f32;
        for g in &garrison_npcs {
            let dx = (pc as f32 - g.col as f32).abs();
            let dy = (pr as f32 - g.row as f32).abs();
            let dist = (dx * dx + dy * dy).sqrt().max(1.0);
            garrison_value += g.combat_value / dist;
        }

        for g in &garrison_npcs {
            let dx = (pc as f32 - g.col as f32).abs();
            let dy = (pr as f32 - g.row as f32).abs();
            if (dx * dx + dy * dy).sqrt() > 5.0 { continue; }

            for db in &def_buildings {
                let bx = (g.col as f32 - db.col as f32).abs();
                let by = (g.row as f32 - db.row as f32).abs();
                if (bx * bx + by * by).sqrt() > 3.0 { continue; }
                let multiplier = match (g.is_ranged, db.btype) {
                    (true, BuildingType::Watchtower) => 1.5,
                    (false, BuildingType::Gate) => 1.3,
                    (false, BuildingType::Wall) => 1.2,
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

        coverage_map.push(structural + garrison_value);
        let resp = garrison_dist[pi];
        response_time_map.push(if resp == u16::MAX { f32::MAX } else { resp as f32 });
    }

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
