//! City grid — tile-based settlement layout with CA-driven growth.
//!
//! Each settlement gets a 128×128 cell grid. Buildings are organic tile shapes
//! placed by agents evaluating suitability scores. Influence propagates along
//! road networks. Zone types cluster via attraction/repulsion matrix.
//!
//! Based on the Chimera City Generation plan:
//! - Layer 1: CA suitability field (per-cell scores)
//! - Layer 2: Infrastructure skeleton (roads, walls, plazas)
//! - Layer 3: Agent decisions (NPCs place buildings, radiate influence)

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Cell types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum CellState {
    Empty = 0,
    Road = 1,
    Building = 2,
    Plaza = 3,
    Wall = 4,
    Water = 5,
    Ruin = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ZoneType {
    None = 0,
    Residential = 1,
    Commercial = 2,
    Industrial = 3,
    Religious = 4,
    Arcane = 5,
    Noble = 6,
    Military = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum CellTerrain {
    Flat = 0,
    Slope = 1,
    Steep = 2,
    Water = 3,
    Cliff = 4,
}

impl Default for CellTerrain {
    fn default() -> Self { CellTerrain::Flat }
}

/// A single cell in the city grid.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Cell {
    pub state: CellState,
    pub zone: ZoneType,
    pub density: u8,              // 0-3 (hovel → stone → multi-story → landmark)
    pub terrain: CellTerrain,
    pub building_id: Option<u32>, // Links to Entity with BuildingData
    pub road_tier: u8,            // 0 (none), 1 (alley), 2 (street), 3 (avenue), 4 (highway)
    pub age: u16,                 // Ticks since construction
}

impl Default for Cell {
    fn default() -> Self {
        Cell {
            state: CellState::Empty,
            zone: ZoneType::None,
            density: 0,
            terrain: CellTerrain::Flat,
            building_id: None,
            road_tier: 0,
            age: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// City grid
// ---------------------------------------------------------------------------

/// A 2D tile grid representing a settlement's spatial layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CityGrid {
    pub cols: usize,
    pub rows: usize,
    pub cells: Vec<Cell>,
    pub settlement_id: u32,
    pub center: (usize, usize),
    /// Incrementally maintained set of empty cells with ≥1 developed neighbor.
    #[serde(skip)]
    pub frontier: HashSet<(usize, usize)>,
    /// Cached road distance from nearest road cell (BFS, updated incrementally).
    pub road_distance: Vec<u16>,
    /// Cached flow field to the grid center. Precomputed once, used by all NPCs.
    pub center_flow: Vec<u32>,
}

impl CityGrid {
    /// Create a new empty grid with terrain stamped from the region type.
    pub fn new(cols: usize, rows: usize, settlement_id: u32, terrain: &str, seed: u64) -> Self {
        let center = (cols / 2, rows / 2);
        let mut cells = vec![Cell::default(); cols * rows];

        // Stamp terrain based on region type.
        stamp_terrain(&mut cells, cols, rows, terrain, seed);

        let road_distance = vec![u16::MAX; cols * rows];

        let center_flow = vec![u32::MAX; cols * rows]; // initialized after roads

        let mut grid = CityGrid {
            cols, rows, cells, settlement_id, center,
            frontier: HashSet::new(),
            road_distance,
            center_flow,
        };

        // Place founding road intersection at center.
        grid.place_road_cross(center.0, center.1);

        // Rebuild frontier and precompute flow field.
        grid.rebuild_frontier();
        grid.center_flow = grid.compute_flow_field(center);

        grid
    }

    #[inline]
    pub fn idx(&self, col: usize, row: usize) -> usize {
        row * self.cols + col
    }

    #[inline]
    pub fn in_bounds(&self, col: usize, row: usize) -> bool {
        col < self.cols && row < self.rows
    }

    pub fn cell(&self, col: usize, row: usize) -> &Cell {
        &self.cells[self.idx(col, row)]
    }

    pub fn cell_mut(&mut self, col: usize, row: usize) -> &mut Cell {
        let idx = self.idx(col, row);
        &mut self.cells[idx]
    }

    /// Place a road cross (+) at the given position.
    pub fn place_road_cross(&mut self, cx: usize, cy: usize) {
        let extent = 8; // road arms extend 8 cells from center
        for d in 0..=extent {
            for &(dx, dy) in &[(d, 0), (0, d)] {
                for &(sx, sy) in &[(1i32, 1i32), (1, -1), (-1, 1), (-1, -1)] {
                    let x = (cx as i32 + dx as i32 * sx) as usize;
                    let y = (cy as i32 + dy as i32 * sy) as usize;
                    if self.in_bounds(x, y) {
                        let cell = self.cell_mut(x, y);
                        if cell.terrain != CellTerrain::Water && cell.terrain != CellTerrain::Cliff {
                            cell.state = CellState::Road;
                            cell.road_tier = if d == 0 { 3 } else { 2 }; // center = avenue, arms = street
                        }
                    }
                }
            }
        }
        self.update_road_distances();
    }

    /// Check if a cell is "developed" (not empty, not water).
    pub fn is_developed(&self, col: usize, row: usize) -> bool {
        if !self.in_bounds(col, row) { return false; }
        let cell = self.cell(col, row);
        !matches!(cell.state, CellState::Empty | CellState::Water)
    }

    /// Rebuild the frontier set from scratch.
    pub fn rebuild_frontier(&mut self) {
        self.frontier.clear();
        for row in 0..self.rows {
            for col in 0..self.cols {
                if self.cell(col, row).state == CellState::Empty
                    && self.cell(col, row).terrain != CellTerrain::Water
                    && self.cell(col, row).terrain != CellTerrain::Cliff
                    && self.has_developed_neighbor(col, row)
                {
                    self.frontier.insert((col, row));
                }
            }
        }
    }

    /// Check if any Moore neighbor (8-connected) is developed.
    fn has_developed_neighbor(&self, col: usize, row: usize) -> bool {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 { continue; }
                let nx = col as i32 + dx;
                let ny = row as i32 + dy;
                if nx >= 0 && ny >= 0 {
                    let nx = nx as usize;
                    let ny = ny as usize;
                    if self.is_developed(nx, ny) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Count developed Moore neighbors.
    pub fn count_developed_neighbors(&self, col: usize, row: usize) -> u8 {
        let mut count = 0u8;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 { continue; }
                let nx = col as i32 + dx;
                let ny = row as i32 + dy;
                if nx >= 0 && ny >= 0 && self.is_developed(nx as usize, ny as usize) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Update road distance cache via BFS from all road cells.
    pub fn update_road_distances(&mut self) {
        use std::collections::VecDeque;
        self.road_distance.fill(u16::MAX);
        let mut queue = VecDeque::new();

        // Seed with all road cells.
        for row in 0..self.rows {
            for col in 0..self.cols {
                if self.cell(col, row).state == CellState::Road {
                    let idx = self.idx(col, row);
                    self.road_distance[idx] = 0;
                    queue.push_back((col, row));
                }
            }
        }

        // BFS.
        while let Some((col, row)) = queue.pop_front() {
            let current_dist = self.road_distance[self.idx(col, row)];
            if current_dist >= 20 { continue; } // cap at 20

            for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                let nx = col as i32 + dx;
                let ny = row as i32 + dy;
                if nx < 0 || ny < 0 { continue; }
                let nx = nx as usize;
                let ny = ny as usize;
                if !self.in_bounds(nx, ny) { continue; }
                let nidx = self.idx(nx, ny);
                let new_dist = current_dist + 1;
                if new_dist < self.road_distance[nidx] {
                    self.road_distance[nidx] = new_dist;
                    queue.push_back((nx, ny));
                }
            }
        }
    }

    /// Update frontier incrementally after a cell at (col, row) changed state.
    pub fn update_frontier_around(&mut self, col: usize, row: usize) {
        // The changed cell is no longer frontier.
        self.frontier.remove(&(col, row));

        // Check all neighbors — they might become frontier.
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 { continue; }
                let nx = col as i32 + dx;
                let ny = row as i32 + dy;
                if nx < 0 || ny < 0 { continue; }
                let nx = nx as usize;
                let ny = ny as usize;
                if !self.in_bounds(nx, ny) { continue; }

                let cell = self.cell(nx, ny);
                if cell.state == CellState::Empty
                    && cell.terrain != CellTerrain::Water
                    && cell.terrain != CellTerrain::Cliff
                    && self.has_developed_neighbor(nx, ny)
                {
                    self.frontier.insert((nx, ny));
                } else {
                    self.frontier.remove(&(nx, ny));
                }
            }
        }
    }

    /// Convert grid coordinates to world-space position relative to settlement center.
    pub fn grid_to_world(&self, col: usize, row: usize, settlement_pos: (f32, f32)) -> (f32, f32) {
        let cell_size = 2.0; // each grid cell = 2 world units
        let offset_x = (col as f32 - self.center.0 as f32) * cell_size;
        let offset_y = (row as f32 - self.center.1 as f32) * cell_size;
        (settlement_pos.0 + offset_x, settlement_pos.1 + offset_y)
    }

    /// Convert world-space position to grid coordinates.
    pub fn world_to_grid(&self, world_pos: (f32, f32), settlement_pos: (f32, f32)) -> (usize, usize) {
        let cell_size = 2.0;
        let col = ((world_pos.0 - settlement_pos.0) / cell_size + self.center.0 as f32).round() as isize;
        let row = ((world_pos.1 - settlement_pos.1) / cell_size + self.center.1 as f32).round() as isize;
        (col.clamp(0, self.cols as isize - 1) as usize,
         row.clamp(0, self.rows as isize - 1) as usize)
    }

    /// Check if a cell is walkable (NPCs can path through it).
    pub fn is_walkable(&self, col: usize, row: usize) -> bool {
        if !self.in_bounds(col, row) { return false; }
        let cell = self.cell(col, row);
        matches!(cell.state, CellState::Empty | CellState::Road | CellState::Plaza)
            && !matches!(cell.terrain, CellTerrain::Water | CellTerrain::Cliff)
    }

    /// Movement cost for a walkable cell. Roads are cheap, empty is moderate, slopes are expensive.
    pub fn move_cost(&self, col: usize, row: usize) -> f32 {
        if !self.in_bounds(col, row) { return f32::MAX; }
        let cell = self.cell(col, row);
        let base = match cell.state {
            CellState::Road => 1.0,
            CellState::Plaza => 1.0,
            CellState::Empty => 2.0,
            _ => f32::MAX, // impassable
        };
        let terrain_mult = match cell.terrain {
            CellTerrain::Flat => 1.0,
            CellTerrain::Slope => 1.5,
            CellTerrain::Steep => 3.0,
            CellTerrain::Water | CellTerrain::Cliff => f32::MAX,
        };
        base * terrain_mult
    }

    /// A* pathfinding on the city grid. Returns a path of (col, row) from start to goal,
    /// or None if no path exists. Max 500 iterations to prevent runaway.
    pub fn find_path(
        &self,
        start: (usize, usize),
        goal: (usize, usize),
    ) -> Option<Vec<(usize, usize)>> {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;

        if !self.in_bounds(start.0, start.1) || !self.in_bounds(goal.0, goal.1) {
            return None;
        }
        if start == goal { return Some(vec![goal]); }

        let n = self.cols * self.rows;
        let mut g_score = vec![f32::MAX; n];
        let mut came_from = vec![u32::MAX; n];
        // (Reverse(f_score_fixed), index) — use fixed-point for Ord.
        let mut open: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();

        let start_idx = start.1 * self.cols + start.0;
        let goal_idx = goal.1 * self.cols + goal.0;
        g_score[start_idx] = 0.0;

        let heuristic = |idx: usize| -> f32 {
            let c = idx % self.cols;
            let r = idx / self.cols;
            let dx = (c as f32 - goal.0 as f32).abs();
            let dy = (r as f32 - goal.1 as f32).abs();
            dx + dy // Manhattan distance
        };

        let f0 = (heuristic(start_idx) * 100.0) as u32;
        open.push(Reverse((f0, start_idx as u32)));

        let mut iterations = 0u32;
        const MAX_ITER: u32 = 500;

        while let Some(Reverse((_, current_u32))) = open.pop() {
            let current = current_u32 as usize;
            if current == goal_idx {
                // Reconstruct path.
                let mut path = Vec::new();
                let mut c = current;
                while c != start_idx {
                    path.push((c % self.cols, c / self.cols));
                    c = came_from[c] as usize;
                    if c == u32::MAX as usize { return None; } // shouldn't happen
                }
                path.reverse();
                return Some(path);
            }

            iterations += 1;
            if iterations >= MAX_ITER { return None; }

            let cx = current % self.cols;
            let cy = current / self.cols;
            let current_g = g_score[current];

            // 8-connected neighbors.
            for &(dx, dy) in &[
                (-1i32, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ] {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx < 0 || ny < 0 || nx >= self.cols as i32 || ny >= self.rows as i32 {
                    continue;
                }
                let nx = nx as usize;
                let ny = ny as usize;
                let ni = ny * self.cols + nx;

                // Walkability check (goal cell is always allowed — may be a building entrance).
                if ni != goal_idx && !self.is_walkable(nx, ny) { continue; }

                // Diagonal corner-cutting prevention.
                if dx != 0 && dy != 0 {
                    if !self.is_walkable(cx, ny) || !self.is_walkable(nx, cy) {
                        continue;
                    }
                }

                let step_cost = if dx != 0 && dy != 0 {
                    self.move_cost(nx, ny) * 1.414 // diagonal
                } else {
                    self.move_cost(nx, ny)
                };
                if step_cost >= f32::MAX { continue; }

                let tentative_g = current_g + step_cost;
                if tentative_g < g_score[ni] {
                    g_score[ni] = tentative_g;
                    came_from[ni] = current as u32;
                    let f = (tentative_g + heuristic(ni)) * 100.0;
                    open.push(Reverse((f as u32, ni as u32)));
                }
            }
        }

        None // no path found
    }

    /// Precompute a BFS flow field from a target cell. Returns a direction map
    /// where each cell stores the next cell to move toward to reach the target.
    /// `flow[idx] = next_idx` (or u32::MAX if unreachable).
    pub fn compute_flow_field(&self, target: (usize, usize)) -> Vec<u32> {
        let n = self.cols * self.rows;
        let mut flow = vec![u32::MAX; n];
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();

        let target_idx = target.1 * self.cols + target.0;
        flow[target_idx] = target_idx as u32; // target points to itself
        visited[target_idx] = true;
        queue.push_back(target_idx);

        while let Some(current) = queue.pop_front() {
            let cx = current % self.cols;
            let cy = current / self.cols;

            // 4-connected neighbors (simpler than 8 for flow fields).
            for &(dx, dy) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx < 0 || ny < 0 || nx >= self.cols as i32 || ny >= self.rows as i32 { continue; }
                let ni = ny as usize * self.cols + nx as usize;
                if visited[ni] { continue; }
                if !self.is_walkable(nx as usize, ny as usize) { continue; }

                visited[ni] = true;
                flow[ni] = current as u32; // "to reach target, go to `current`"
                queue.push_back(ni);
            }
        }
        flow
    }

    /// Follow a flow field one step from a position. Returns the next cell to move to.
    pub fn flow_field_next(&self, flow: &[u32], pos: (usize, usize)) -> Option<(usize, usize)> {
        let idx = pos.1 * self.cols + pos.0;
        if idx >= flow.len() { return None; }
        let next = flow[idx] as usize;
        if next == u32::MAX as usize || next == idx { return None; }
        Some((next % self.cols, next / self.cols))
    }

    /// Count buildings by zone type.
    pub fn building_counts(&self) -> [u32; 8] {
        let mut counts = [0u32; 8];
        for cell in &self.cells {
            if cell.state == CellState::Building {
                counts[cell.zone as usize] += 1;
            }
        }
        counts
    }

    /// Total developed cells (buildings + roads + plazas + walls).
    pub fn developed_count(&self) -> usize {
        self.cells.iter().filter(|c| self.is_state_developed(c.state)).count()
    }

    fn is_state_developed(&self, state: CellState) -> bool {
        !matches!(state, CellState::Empty | CellState::Water)
    }

    /// Count adjacent building cells (4-connected) for a road cell.
    pub fn road_frontage(&self, col: usize, row: usize) -> u8 {
        let mut count = 0u8;
        for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let nx = col as i32 + dx;
            let ny = row as i32 + dy;
            if nx >= 0 && ny >= 0 {
                let nx = nx as usize;
                let ny = ny as usize;
                if self.in_bounds(nx, ny) && self.cell(nx, ny).state == CellState::Building {
                    count += 1;
                }
            }
        }
        count
    }

    /// Upgrade road tiers based on adjacent building frontage.
    /// Alley (1) → Street (2) at 3+ buildings, Street → Avenue (3) at 6+ buildings.
    pub fn upgrade_roads(&mut self) {
        let mut snapshot: Vec<(usize, usize, u8)> = Vec::new();
        for row in 0..self.rows {
            for col in 0..self.cols {
                let cell = &self.cells[row * self.cols + col];
                if cell.state == CellState::Road && cell.road_tier < 4 {
                    snapshot.push((col, row, cell.road_tier));
                }
            }
        }

        for (col, row, tier) in snapshot {
            let frontage = self.road_frontage(col, row);
            let new_tier = if frontage >= 6 { 3 }       // avenue
                else if frontage >= 3 { tier.max(2) }    // street
                else { tier };
            if new_tier > tier {
                self.cell_mut(col, row).road_tier = new_tier;
            }
        }
    }

    /// Place walls around the settlement perimeter (outermost developed ring).
    /// Walls go on empty cells adjacent to the outermost buildings.
    pub fn place_walls(&mut self, max_walls: usize) {
        let mut wall_candidates: Vec<(usize, usize, f32)> = Vec::new();

        for row in 0..self.rows {
            for col in 0..self.cols {
                let cell = self.cell(col, row);
                if cell.state != CellState::Empty { continue; }
                if cell.terrain == CellTerrain::Water || cell.terrain == CellTerrain::Cliff { continue; }

                // Must be adjacent to a developed cell.
                if !self.has_developed_neighbor(col, row) { continue; }

                // Prefer cells far from center (perimeter walls).
                let dx = col as f32 - self.center.0 as f32;
                let dy = row as f32 - self.center.1 as f32;
                let dist = (dx * dx + dy * dy).sqrt();

                wall_candidates.push((col, row, dist));
            }
        }

        // Sort by distance from center (descending) — walls go on the outside.
        wall_candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        for (col, row, _) in wall_candidates.into_iter().take(max_walls) {
            let cell = self.cell_mut(col, row);
            cell.state = CellState::Wall;
            cell.zone = ZoneType::Military;
        }
    }

    /// Summary stats for display.
    pub fn summary(&self) -> CityGridSummary {
        let mut buildings = 0u32;
        let mut roads = 0u32;
        let mut walls = 0u32;
        let mut ruins = 0u32;
        let mut zone_counts = [0u32; 8];

        for cell in &self.cells {
            match cell.state {
                CellState::Building => {
                    buildings += 1;
                    zone_counts[cell.zone as usize] += 1;
                }
                CellState::Road => roads += 1,
                CellState::Wall => walls += 1,
                CellState::Ruin => ruins += 1,
                _ => {}
            }
        }

        CityGridSummary { buildings, roads, walls, ruins, zone_counts }
    }
}

/// Summary statistics for a city grid.
#[derive(Debug, Clone)]
pub struct CityGridSummary {
    pub buildings: u32,
    pub roads: u32,
    pub walls: u32,
    pub ruins: u32,
    pub zone_counts: [u32; 8],
}

// ---------------------------------------------------------------------------
// Influence map
// ---------------------------------------------------------------------------

/// Per-cell influence scores, stored as parallel flat arrays for cache efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfluenceMap {
    pub cols: usize,
    pub rows: usize,
    pub commercial: Vec<f32>,
    pub residential: Vec<f32>,
    pub religious: Vec<f32>,
    pub prestige: Vec<f32>,
    pub danger: Vec<f32>,
    pub pollution: Vec<f32>,
    pub military: Vec<f32>,
}

impl InfluenceMap {
    pub fn new(cols: usize, rows: usize) -> Self {
        let n = cols * rows;
        Self {
            cols, rows,
            commercial: vec![0.0; n],
            residential: vec![0.0; n],
            religious: vec![0.0; n],
            prestige: vec![0.0; n],
            danger: vec![0.0; n],
            pollution: vec![0.0; n],
            military: vec![0.0; n],
        }
    }

    #[inline]
    fn idx(&self, col: usize, row: usize) -> usize {
        row * self.cols + col
    }

    /// Update influence around a newly placed building using road-network BFS.
    pub fn propagate_building(&mut self, col: usize, row: usize, zone: ZoneType,
                              strength: f32, grid: &CityGrid) {
        use std::collections::VecDeque;
        let max_dist = 15u16;
        let mut visited = vec![false; self.cols * self.rows];
        let mut queue = VecDeque::new();

        let start = self.idx(col, row);
        visited[start] = true;
        queue.push_back((col, row, 0u16));

        while let Some((cx, cy, dist)) = queue.pop_front() {
            if dist > max_dist { continue; }

            let falloff = strength / (1.0 + dist as f32);
            let idx = self.idx(cx, cy);

            // Add influence based on zone type.
            match zone {
                ZoneType::Commercial => self.commercial[idx] += falloff,
                ZoneType::Residential => self.residential[idx] += falloff,
                ZoneType::Religious => self.religious[idx] += falloff,
                ZoneType::Noble => self.prestige[idx] += falloff,
                ZoneType::Industrial => self.pollution[idx] += falloff,
                ZoneType::Military => self.military[idx] += falloff,
                ZoneType::Arcane => self.prestige[idx] += falloff * 0.5,
                ZoneType::None => {}
            }

            // Propagate along roads (faster) and through cells (slower).
            for &(dx, dy) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx < 0 || ny < 0 { continue; }
                let nx = nx as usize;
                let ny = ny as usize;
                if nx >= self.cols || ny >= self.rows { continue; }
                let nidx = self.idx(nx, ny);
                if visited[nidx] { continue; }
                visited[nidx] = true;

                // Road cells propagate at distance +1, non-road at +2.
                let step = if grid.cell(nx, ny).state == CellState::Road { 1 } else { 2 };
                queue.push_back((nx, ny, dist + step));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Suitability scoring
// ---------------------------------------------------------------------------

/// Zone attraction/repulsion matrix.
/// `ZONE_AFFINITY[existing_zone][candidate_zone]` = attraction (+) or repulsion (-).
pub const ZONE_AFFINITY: [[f32; 8]; 8] = [
    // None  Res   Com   Ind   Rel   Arc   Nob   Mil
    // Self-affinity kept low (0.1-0.2) to prevent single-zone domination.
    // Cross-zone attraction drives mixed districts.
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], // None
    [0.0,  0.1,  0.4, -0.6,  0.2, -0.1,  0.3, -0.2], // Residential
    [0.0,  0.4,  0.2,  0.1,  0.2,  0.1,  0.4,  0.0], // Commercial
    [0.0, -0.6,  0.1,  0.2, -0.3,  0.0, -0.7,  0.2], // Industrial
    [0.0,  0.2,  0.2, -0.3,  0.2, -0.3,  0.2,  0.0], // Religious
    [0.0, -0.1,  0.1,  0.0, -0.3,  0.2,  0.1, -0.2], // Arcane
    [0.0,  0.3,  0.4, -0.7,  0.2,  0.1,  0.2,  0.1], // Noble
    [0.0, -0.2,  0.0,  0.2,  0.0, -0.2,  0.1,  0.2], // Military
];

/// Compute suitability score for a candidate zone at a frontier cell.
pub fn suitability_score(
    grid: &CityGrid,
    influence: &InfluenceMap,
    col: usize, row: usize,
    candidate_zone: ZoneType,
    noise_seed: u64,
) -> f32 {
    let idx = grid.idx(col, row);

    // 1. Road proximity (0.30 weight)
    let road_dist = grid.road_distance[idx] as f32;
    let road_score = 1.0 / (1.0 + road_dist);

    // 2. Neighbor density (0.25 weight)
    let neighbor_score = grid.count_developed_neighbors(col, row) as f32 / 8.0;

    // 3. Center distance (0.15 weight) — bid-rent centrality
    let dx = col as f32 - grid.center.0 as f32;
    let dy = row as f32 - grid.center.1 as f32;
    let center_dist = (dx * dx + dy * dy).sqrt();
    let center_score = 1.0 / (1.0 + center_dist * 0.1);

    // 4. Terrain (0.10 weight)
    let terrain_score = match grid.cell(col, row).terrain {
        CellTerrain::Flat => 1.0,
        CellTerrain::Slope => 0.6,
        CellTerrain::Steep => 0.2,
        CellTerrain::Water | CellTerrain::Cliff => 0.0,
    };

    // 5. Zone affinity from neighbors (0.10 weight)
    let mut affinity_score = 0.0f32;
    for dy in -3i32..=3 {
        for dx in -3i32..=3 {
            if dx == 0 && dy == 0 { continue; }
            let nx = col as i32 + dx;
            let ny = row as i32 + dy;
            if nx < 0 || ny < 0 { continue; }
            let nx = nx as usize;
            let ny = ny as usize;
            if !grid.in_bounds(nx, ny) { continue; }
            let neighbor = grid.cell(nx, ny);
            if neighbor.state != CellState::Building { continue; }
            let dist = ((dx * dx + dy * dy) as f32).sqrt();
            let weight = 1.0 / dist;
            affinity_score += ZONE_AFFINITY[neighbor.zone as usize][candidate_zone as usize] * weight;
        }
    }
    affinity_score = affinity_score.clamp(-1.0, 1.0);

    // 6. Influence map (0.05 weight)
    let influence_score = match candidate_zone {
        ZoneType::Commercial => influence.commercial[idx],
        ZoneType::Residential => influence.residential[idx] - influence.pollution[idx] * 0.5,
        ZoneType::Industrial => influence.commercial[idx] * 0.3 - influence.residential[idx] * 0.2,
        ZoneType::Religious => influence.religious[idx],
        ZoneType::Noble => influence.prestige[idx] - influence.pollution[idx],
        ZoneType::Military => influence.military[idx],
        ZoneType::Arcane => influence.prestige[idx] * 0.5,
        ZoneType::None => 0.0,
    }.clamp(-1.0, 1.0);

    // 7. Noise (0.05 weight) — heavy-tailed for occasional long-range jumps
    let h = noise_seed.wrapping_mul(6364136223846793005)
        .wrapping_add(col as u64 * 31 + row as u64 * 997);
    let noise = ((h >> 33) as f32 / (1u64 << 31) as f32).sqrt() * 0.1;

    // Weighted sum.
    0.30 * road_score
    + 0.25 * neighbor_score
    + 0.15 * center_score
    + 0.10 * terrain_score
    + 0.10 * affinity_score
    + 0.05 * influence_score
    + 0.05 * noise
}

// ---------------------------------------------------------------------------
// Terrain stamping
// ---------------------------------------------------------------------------

fn stamp_terrain(cells: &mut [Cell], cols: usize, rows: usize, terrain: &str, seed: u64) {
    // Base terrain from region type.
    let base = match terrain {
        "Mountains" => CellTerrain::Slope,
        "Swamp" | "Coast" => CellTerrain::Flat, // with water features
        _ => CellTerrain::Flat,
    };

    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            cells[idx].terrain = base;

            // Add terrain features based on region type.
            let h = seed.wrapping_mul(6364136223846793005)
                .wrapping_add((col as u64 + 1).wrapping_mul(2862933555777941757))
                .wrapping_add((row as u64).wrapping_mul(1103515245));
            let noise = ((h >> 33) as u32 % 1000) as f32 / 1000.0;

            match terrain {
                "Mountains" => {
                    if noise > 0.85 { cells[idx].terrain = CellTerrain::Steep; }
                    if noise > 0.95 { cells[idx].terrain = CellTerrain::Cliff; }
                }
                "Swamp" => {
                    if noise > 0.7 { cells[idx].terrain = CellTerrain::Water; }
                }
                "Coast" => {
                    // Water on one edge.
                    if col < cols / 6 { cells[idx].terrain = CellTerrain::Water; }
                }
                "Forest" => {
                    if noise > 0.9 { cells[idx].terrain = CellTerrain::Slope; }
                }
                _ => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_grid_has_roads_and_frontier() {
        let grid = CityGrid::new(64, 64, 0, "Plains", 42);
        // Center should have roads.
        assert_eq!(grid.cell(32, 32).state, CellState::Road);
        // Frontier should not be empty.
        assert!(!grid.frontier.is_empty(), "frontier should have cells near roads");
        // Road distance at center should be 0.
        assert_eq!(grid.road_distance[grid.idx(32, 32)], 0);
    }

    #[test]
    fn frontier_updates_incrementally() {
        let mut grid = CityGrid::new(32, 32, 0, "Plains", 42);
        let initial_frontier = grid.frontier.len();
        assert!(initial_frontier > 0);

        // Place a building on a frontier cell.
        let (col, row) = *grid.frontier.iter().next().unwrap();
        grid.cell_mut(col, row).state = CellState::Building;
        grid.cell_mut(col, row).zone = ZoneType::Residential;
        grid.update_frontier_around(col, row);

        // That cell should no longer be frontier.
        assert!(!grid.frontier.contains(&(col, row)));
    }

    #[test]
    fn suitability_prefers_near_roads() {
        let grid = CityGrid::new(32, 32, 0, "Plains", 42);
        let influence = InfluenceMap::new(32, 32);

        // Cell near road vs far from road.
        let near_road = (17, 16); // adjacent to center road
        let far = (5, 5);

        let score_near = suitability_score(&grid, &influence, near_road.0, near_road.1, ZoneType::Residential, 42);
        let score_far = suitability_score(&grid, &influence, far.0, far.1, ZoneType::Residential, 42);

        assert!(score_near > score_far, "near-road ({:.3}) should score higher than far ({:.3})", score_near, score_far);
    }

    #[test]
    fn mountain_terrain_has_steep_cells() {
        let grid = CityGrid::new(64, 64, 0, "Mountains", 42);
        let steep_count = grid.cells.iter().filter(|c| c.terrain == CellTerrain::Steep).count();
        assert!(steep_count > 0, "mountain terrain should have steep cells");
    }
}
