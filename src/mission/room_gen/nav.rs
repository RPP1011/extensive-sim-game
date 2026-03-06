use crate::ai::core::SimVec2;
use crate::ai::pathing::GridNav;

// ---------------------------------------------------------------------------
// Core public data types
// ---------------------------------------------------------------------------

/// Flat, row-major navigation grid for the mission room.
///
/// Index formula: `row * cols + col`
/// SimVec2 coordinate system: x -> world X, y -> world Z.
#[derive(Debug, Clone)]
pub struct NavGrid {
    pub cell_size: f32,
    pub cols: usize,
    pub rows: usize,
    /// `true` = an agent may walk through this cell.
    pub walkable: Vec<bool>,
    /// Height at the centre of each cell (metres).
    pub elevation: Vec<f32>,
}

impl NavGrid {
    pub fn new(cols: usize, rows: usize, cell_size: f32) -> Self {
        let n = cols * rows;
        Self {
            cell_size,
            cols,
            rows,
            walkable: vec![true; n],
            elevation: vec![0.0; n],
        }
    }

    #[inline]
    pub fn idx(&self, col: usize, row: usize) -> usize {
        row * self.cols + col
    }

    pub fn set_walkable_rect(
        &mut self,
        col0: usize,
        row0: usize,
        col1: usize,
        row1: usize,
        value: bool,
    ) {
        let c0 = col0.min(col1);
        let c1 = col0.max(col1).min(self.cols.saturating_sub(1));
        let r0 = row0.min(row1);
        let r1 = row0.max(row1).min(self.rows.saturating_sub(1));
        for r in r0..=r1 {
            for c in c0..=c1 {
                let i = self.idx(c, r);
                self.walkable[i] = value;
            }
        }
    }

    pub fn set_elevation_rect(
        &mut self,
        col0: usize,
        row0: usize,
        col1: usize,
        row1: usize,
        elev: f32,
    ) {
        let c0 = col0.min(col1);
        let c1 = col0.max(col1).min(self.cols.saturating_sub(1));
        let r0 = row0.min(row1);
        let r1 = row0.max(row1).min(self.rows.saturating_sub(1));
        for r in r0..=r1 {
            for c in c0..=c1 {
                let i = self.idx(c, r);
                self.elevation[i] = elev;
            }
        }
    }

    /// World-space centre of a cell (SimVec2: x -> world X, y -> world Z).
    pub fn cell_centre(&self, col: usize, row: usize) -> SimVec2 {
        SimVec2 {
            x: (col as f32 + 0.5) * self.cell_size,
            y: (row as f32 + 0.5) * self.cell_size,
        }
    }

    /// Convert to a [`GridNav`] for AI pathfinding, computing slope costs
    /// from elevation transitions.
    pub fn to_gridnav(&self) -> GridNav {
        let cs = self.cell_size;
        let mut grid = GridNav::new(0.0, self.cols as f32 * cs, 0.0, self.rows as f32 * cs, cs);
        for r in 0..self.rows {
            for c in 0..self.cols {
                let idx = r * self.cols + c;
                if !self.walkable[idx] {
                    grid.blocked.insert((c as i32, r as i32));
                }
                let elev = self.elevation[idx];
                if elev != 0.0 {
                    grid.elevation_by_cell.insert((c as i32, r as i32), elev);
                }
            }
        }
        // Compute slope costs from elevation deltas between adjacent cells.
        for r in 0..self.rows {
            for c in 0..self.cols {
                let idx = r * self.cols + c;
                if !self.walkable[idx] {
                    continue;
                }
                let elev = self.elevation[idx];
                let mut max_delta: f32 = 0.0;
                for &(dc, dr) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nc = c as i32 + dc;
                    let nr = r as i32 + dr;
                    if nc < 0 || nr < 0 || nc >= self.cols as i32 || nr >= self.rows as i32 {
                        continue;
                    }
                    let ni = nr as usize * self.cols + nc as usize;
                    let neighbor_elev = self.elevation[ni];
                    let delta = (neighbor_elev - elev).abs();
                    max_delta = max_delta.max(delta);
                }
                if max_delta > 0.1 {
                    let slope_cost = 1.0 + max_delta * 0.5;
                    grid.slope_cost_by_cell.insert((c as i32, r as i32), slope_cost);
                }
            }
        }
        grid
    }

    /// Column and row index for a world-space SimVec2.
    pub fn cell_of(&self, pos: SimVec2) -> (usize, usize) {
        let col = (pos.x / self.cell_size).floor() as usize;
        let row = (pos.y / self.cell_size).floor() as usize;
        (
            col.min(self.cols.saturating_sub(1)),
            row.min(self.rows.saturating_sub(1)),
        )
    }
}

/// A collection of world-space spawn points.
#[derive(Debug, Clone)]
pub struct SpawnZone {
    /// World-space positions (SimVec2: x -> world X, y -> world Z).
    pub positions: Vec<SimVec2>,
}
