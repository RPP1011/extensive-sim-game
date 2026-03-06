mod navigation;

pub use navigation::*;

use std::collections::{HashMap, HashSet};

use crate::ai::core::{sim_vec2, SimVec2};

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
        let c = self.cell_of(pos);
        self.walkable(c.0, c.1)
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
