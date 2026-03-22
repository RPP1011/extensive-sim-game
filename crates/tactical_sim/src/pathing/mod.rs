mod navigation;

pub use navigation::*;

use std::collections::{HashMap, HashSet};

use crate::sim::{sim_vec2, SimVec2};

#[derive(Debug, Clone)]
pub struct GridNav {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub cell_size: f32,
    pub blocked: HashSet<(i32, i32)>,
    pub elevation_by_cell: HashMap<(i32, i32), f32>,
    pub slope_cost_by_cell: HashMap<(i32, i32), f32>,
    /// Precomputed per-cell: minimum wall distance in 4 cardinal directions.
    /// Computed once after terrain is finalized via `precompute_wall_proximity()`.
    pub wall_proximity_by_cell: HashMap<(i32, i32), f32>,
    /// Precomputed per-cell: number of blocked cardinal neighbors (0-4).
    pub chokepoint_score_by_cell: HashMap<(i32, i32), u8>,
    /// True when the interior has zero blocked cells (e.g. Open room type).
    /// Enables fast paths that skip per-cell iteration in pathfinding, LOS,
    /// and position token extraction.
    pub fully_open: bool,
}

impl GridNav {
    pub fn new(min_x: f32, max_x: f32, min_y: f32, max_y: f32, cell_size: f32) -> Self {
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            cell_size,
            blocked: HashSet::new(),
            elevation_by_cell: HashMap::new(),
            slope_cost_by_cell: HashMap::new(),
            wall_proximity_by_cell: HashMap::new(),
            chokepoint_score_by_cell: HashMap::new(),
            fully_open: false,
        }
    }

    pub fn add_block_rect(&mut self, min_x: f32, max_x: f32, min_y: f32, max_y: f32) {
        let (cx0, cy0) = self.cell_of(sim_vec2(min_x, min_y));
        let (cx1, cy1) = self.cell_of(sim_vec2(max_x, max_y));
        let x0 = cx0.min(cx1);
        let x1 = cx0.max(cx1);
        let y0 = cy0.min(cy1);
        let y1 = cy0.max(cy1);
        for x in x0..=x1 {
            for y in y0..=y1 {
                self.blocked.insert((x, y));
            }
        }
    }

    pub fn carve_rect(&mut self, min_x: f32, max_x: f32, min_y: f32, max_y: f32) {
        let (cx0, cy0) = self.cell_of(sim_vec2(min_x, min_y));
        let (cx1, cy1) = self.cell_of(sim_vec2(max_x, max_y));
        let x0 = cx0.min(cx1);
        let x1 = cx0.max(cx1);
        let y0 = cy0.min(cy1);
        let y1 = cy0.max(cy1);
        for x in x0..=x1 {
            for y in y0..=y1 {
                self.blocked.remove(&(x, y));
            }
        }
    }

    pub fn set_elevation_rect(
        &mut self,
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
        elevation: f32,
    ) {
        let (cx0, cy0) = self.cell_of(sim_vec2(min_x, min_y));
        let (cx1, cy1) = self.cell_of(sim_vec2(max_x, max_y));
        let x0 = cx0.min(cx1);
        let x1 = cx0.max(cx1);
        let y0 = cy0.min(cy1);
        let y1 = cy0.max(cy1);
        for x in x0..=x1 {
            for y in y0..=y1 {
                self.elevation_by_cell.insert((x, y), elevation);
            }
        }
    }

    pub fn elevation_at_cell(&self, cell: (i32, i32)) -> f32 {
        self.elevation_by_cell.get(&cell).copied().unwrap_or(0.0)
    }

    pub fn elevation_at_pos(&self, pos: SimVec2) -> f32 {
        self.elevation_at_cell(self.cell_of(pos))
    }

    pub fn set_slope_cost_rect(
        &mut self,
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
        slope_cost_multiplier: f32,
    ) {
        let (cx0, cy0) = self.cell_of(sim_vec2(min_x, min_y));
        let (cx1, cy1) = self.cell_of(sim_vec2(max_x, max_y));
        let x0 = cx0.min(cx1);
        let x1 = cx0.max(cx1);
        let y0 = cy0.min(cy1);
        let y1 = cy0.max(cy1);
        for x in x0..=x1 {
            for y in y0..=y1 {
                self.slope_cost_by_cell
                    .insert((x, y), slope_cost_multiplier.max(0.1));
            }
        }
    }

    pub fn slope_cost_at_cell(&self, cell: (i32, i32)) -> f32 {
        self.slope_cost_by_cell.get(&cell).copied().unwrap_or(1.0)
    }

    pub fn slope_cost_at_pos(&self, pos: SimVec2) -> f32 {
        self.slope_cost_at_cell(self.cell_of(pos))
    }

    pub fn cell_of(&self, pos: SimVec2) -> (i32, i32) {
        let cx = ((pos.x - self.min_x) / self.cell_size).floor() as i32;
        let cy = ((pos.y - self.min_y) / self.cell_size).floor() as i32;
        (cx, cy)
    }

    pub fn center_of(&self, cell: (i32, i32)) -> SimVec2 {
        let x = self.min_x + (cell.0 as f32 + 0.5) * self.cell_size;
        let y = self.min_y + (cell.1 as f32 + 0.5) * self.cell_size;
        sim_vec2(x, y)
    }

    pub(crate) fn in_bounds(&self, x: i32, y: i32) -> bool {
        let p = self.center_of((x, y));
        p.x >= self.min_x && p.x <= self.max_x && p.y >= self.min_y && p.y <= self.max_y
    }

    pub(crate) fn walkable(&self, x: i32, y: i32) -> bool {
        self.in_bounds(x, y) && !self.blocked.contains(&(x, y))
    }

    pub fn is_walkable_pos(&self, pos: SimVec2) -> bool {
        if self.fully_open {
            // Only check perimeter bounds — no interior obstacles exist.
            return pos.x >= self.min_x && pos.x <= self.max_x
                && pos.y >= self.min_y && pos.y <= self.max_y;
        }
        let c = self.cell_of(pos);
        self.walkable(c.0, c.1)
    }

    /// Precompute wall proximity and chokepoint scores for all walkable cells.
    /// Call once after terrain is finalized (all blocks/carves done).
    pub fn precompute_wall_proximity(&mut self) {
        // Detect fully-open rooms: no interior blocked cells at all.
        self.fully_open = self.blocked.is_empty();

        // For fully_open rooms, skip per-cell iteration entirely.
        // wall_proximity defaults to 5.0 and chokepoint defaults to 0 via
        // the accessor methods, which is correct for open rooms.
        if self.fully_open {
            self.wall_proximity_by_cell.clear();
            self.chokepoint_score_by_cell.clear();
            return;
        }

        let cx0 = ((self.min_x) / self.cell_size).floor() as i32;
        let cx1 = ((self.max_x) / self.cell_size).ceil() as i32;
        let cy0 = ((self.min_y) / self.cell_size).floor() as i32;
        let cy1 = ((self.max_y) / self.cell_size).ceil() as i32;
        let max_steps = (5.0 / self.cell_size).ceil() as i32; // match raycast max_dist=5.0

        self.wall_proximity_by_cell.clear();
        self.chokepoint_score_by_cell.clear();

        for x in cx0..=cx1 {
            for y in cy0..=cy1 {
                if self.blocked.contains(&(x, y)) {
                    continue;
                }

                // Chokepoint: count blocked cardinal neighbors
                let blocked_n = [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .filter(|(ox, oy)| self.blocked.contains(&(x + ox, y + oy)))
                    .count() as u8;
                self.chokepoint_score_by_cell.insert((x, y), blocked_n);

                // Wall proximity: min distance to nearest wall in 4 cardinal directions
                let dirs: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
                let mut min_dist = 5.0f32; // max_dist cap
                for (dx, dy) in &dirs {
                    for s in 1..=max_steps {
                        let nx = x + dx * s;
                        let ny = y + dy * s;
                        if self.blocked.contains(&(nx, ny)) || !self.in_bounds(nx, ny) {
                            let d = s as f32 * self.cell_size;
                            if d < min_dist {
                                min_dist = d;
                            }
                            break;
                        }
                    }
                }
                self.wall_proximity_by_cell.insert((x, y), min_dist);
            }
        }
    }

    /// Get precomputed wall proximity for a position (falls back to 5.0 if not precomputed).
    pub fn wall_proximity_at_pos(&self, pos: SimVec2) -> f32 {
        self.wall_proximity_by_cell.get(&self.cell_of(pos)).copied().unwrap_or(5.0)
    }

    /// Get precomputed chokepoint score for a position (falls back to 0 if not precomputed).
    pub fn chokepoint_at_pos(&self, pos: SimVec2) -> u8 {
        self.chokepoint_score_by_cell.get(&self.cell_of(pos)).copied().unwrap_or(0)
    }

    /// Compute which walkable cells are visible from a given position.
    /// Returns a set of (col, row) cells that have line of sight to `from`.
    /// Used to build per-enemy vision cones for stealth pathfinding.
    pub fn visible_cells_from(&self, from: SimVec2, max_range: f32) -> HashSet<(i32, i32)> {
        let mut visible = HashSet::new();
        if self.fully_open {
            // Everything is visible in open rooms
            let cx0 = (self.min_x / self.cell_size).floor() as i32;
            let cx1 = (self.max_x / self.cell_size).ceil() as i32;
            let cy0 = (self.min_y / self.cell_size).floor() as i32;
            let cy1 = (self.max_y / self.cell_size).ceil() as i32;
            for x in cx0..=cx1 {
                for y in cy0..=cy1 {
                    let pos = self.cell_center(x, y);
                    let dx = pos.x - from.x;
                    let dy = pos.y - from.y;
                    if (dx * dx + dy * dy).sqrt() <= max_range {
                        visible.insert((x, y));
                    }
                }
            }
            return visible;
        }

        let range_cells = (max_range / self.cell_size).ceil() as i32;
        let (fx, fy) = self.cell_of(from);

        for dx in -range_cells..=range_cells {
            for dy in -range_cells..=range_cells {
                let cx = fx + dx;
                let cy = fy + dy;
                if self.blocked.contains(&(cx, cy)) || !self.in_bounds(cx, cy) {
                    continue;
                }
                let target = self.cell_center(cx, cy);
                let dist_sq = (target.x - from.x).powi(2) + (target.y - from.y).powi(2);
                if dist_sq > max_range * max_range {
                    continue;
                }
                if has_line_of_sight(self, from, target) {
                    visible.insert((cx, cy));
                }
            }
        }
        visible
    }

    /// Get the world-space center of a grid cell.
    pub fn cell_center(&self, cx: i32, cy: i32) -> SimVec2 {
        sim_vec2(
            (cx as f32 + 0.5) * self.cell_size,
            (cy as f32 + 0.5) * self.cell_size,
        )
    }

    /// Concealment score for a cell: 0.0 = fully exposed, 1.0 = maximum concealment.
    /// Based on wall proximity — cells near walls/in corners are more concealed.
    pub fn concealment_at(&self, cell: (i32, i32)) -> f32 {
        let prox = self.wall_proximity_by_cell.get(&cell).copied().unwrap_or(5.0);
        if prox < 1.0 {
            0.9 // right against a wall
        } else if prox < 1.5 {
            0.7 // very close to wall
        } else if prox < 2.5 {
            0.4 // near wall
        } else {
            0.0 // open field
        }
    }

    /// Find the best adjacent cell to move to that maximizes concealment
    /// while making progress toward a goal. Used by Skulk intent.
    pub fn skulk_next_cell(
        &self,
        current: (i32, i32),
        goal: (i32, i32),
        enemy_visible_cells: &HashSet<(i32, i32)>,
    ) -> (i32, i32) {
        let neighbors = [
            (current.0 + 1, current.1),
            (current.0 - 1, current.1),
            (current.0, current.1 + 1),
            (current.0, current.1 - 1),
            (current.0 + 1, current.1 + 1),
            (current.0 - 1, current.1 - 1),
            (current.0 + 1, current.1 - 1),
            (current.0 - 1, current.1 + 1),
        ];

        let goal_dist = |c: (i32, i32)| -> f32 {
            ((c.0 - goal.0) as f32).abs() + ((c.1 - goal.1) as f32).abs()
        };
        let current_goal_dist = goal_dist(current);

        let mut best = current;
        let mut best_score = f32::MIN;

        for &n in &neighbors {
            if self.blocked.contains(&n) || !self.in_bounds(n.0, n.1) {
                continue;
            }

            // Score: progress toward goal + concealment - exposure penalty
            let progress = current_goal_dist - goal_dist(n); // positive = closer to goal
            let concealment = self.concealment_at(n);
            let exposed = if enemy_visible_cells.contains(&n) { -3.0 } else { 0.0 };

            let score = progress * 2.0 + concealment * 1.5 + exposed;

            if score > best_score {
                best_score = score;
                best = n;
            }
        }

        best
    }

    /// Returns true if a straight line between two positions passes only
    /// through walkable cells.  When `fully_open` is true this is an O(1)
    /// bounds check instead of per-cell Bresenham iteration.
    pub fn is_convex_open(&self, pos_a: SimVec2, pos_b: SimVec2) -> bool {
        if self.fully_open {
            // No interior obstacles — just confirm both endpoints are in bounds.
            return pos_a.x >= self.min_x && pos_a.x <= self.max_x
                && pos_a.y >= self.min_y && pos_a.y <= self.max_y
                && pos_b.x >= self.min_x && pos_b.x <= self.max_x
                && pos_b.y >= self.min_y && pos_b.y <= self.max_y;
        }
        // Fall back to Bresenham line check through the blocked set.
        let from = self.cell_of(pos_a);
        let to = self.cell_of(pos_b);
        let dx = (to.0 - from.0).abs();
        let dy = (to.1 - from.1).abs();
        let sx = if from.0 < to.0 { 1 } else { -1 };
        let sy = if from.1 < to.1 { 1 } else { -1 };
        let mut err = dx - dy;
        let mut x = from.0;
        let mut y = from.1;
        loop {
            if self.blocked.contains(&(x, y)) {
                return false;
            }
            if x == to.0 && y == to.1 {
                break;
            }
            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
        true
    }

    /// Returns all grid cells covered by a rectangle centered at `center`
    /// with the given `width` (x-axis) and `height` (y-axis).
    pub fn cells_in_rect(&self, center: SimVec2, width: f32, height: f32) -> Vec<(i32, i32)> {
        let half_w = width / 2.0;
        let half_h = height / 2.0;
        let min = sim_vec2(center.x - half_w, center.y - half_h);
        let max = sim_vec2(center.x + half_w, center.y + half_h);
        let (cx0, cy0) = self.cell_of(min);
        let (cx1, cy1) = self.cell_of(max);
        let x0 = cx0.min(cx1);
        let x1 = cx0.max(cx1);
        let y0 = cy0.min(cy1);
        let y1 = cy0.max(cy1);
        let mut cells = Vec::new();
        for x in x0..=x1 {
            for y in y0..=y1 {
                cells.push((x, y));
            }
        }
        cells
    }
}
