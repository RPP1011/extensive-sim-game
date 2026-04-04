//! NavGrid — baked 2D walkable surface derived from VoxelWorld.
//!
//! Each (x, y) position stores whether it's walkable, the surface z level,
//! and movement cost. Pathfinding (A* and flow fields) operates on NavGrid,
//! not on VoxelWorld directly.
//!
//! Rebaked when VoxelWorld undergoes structural changes.

use serde::{Deserialize, Serialize};

use super::voxel::VoxelWorld;

/// A baked 2D walkable surface from a region of VoxelWorld.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavGrid {
    /// Origin in voxel-space (min corner).
    pub origin_vx: i32,
    pub origin_vy: i32,
    pub width: u32,
    pub height: u32,
    pub nodes: Vec<NavNode>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NavNode {
    pub walkable: bool,
    /// Z of the walkable surface (top of highest solid voxel).
    pub surface_z: i32,
    /// Movement cost (material-based, 0.0 for non-walkable).
    pub move_cost: f32,
}

impl Default for NavNode {
    fn default() -> Self {
        Self { walkable: false, surface_z: 0, move_cost: 0.0 }
    }
}

impl NavGrid {
    /// Bake a NavGrid from a rectangular region of VoxelWorld.
    /// Scans each (x, y) column from top (max_z) down to find the surface.
    pub fn bake(world: &VoxelWorld, origin_vx: i32, origin_vy: i32, width: u32, height: u32, max_z: i32) -> Self {
        let mut nodes = vec![NavNode::default(); (width * height) as usize];

        for dy in 0..height {
            for dx in 0..width {
                let vx = origin_vx + dx as i32;
                let vy = origin_vy + dy as i32;
                let idx = (dy * width + dx) as usize;

                // Find surface: highest solid voxel with integrity > 0 in column
                let mut surface_z = -1i32;
                for vz in (0..=max_z).rev() {
                    let v = world.get_voxel(vx, vy, vz);
                    if v.material.is_solid() && v.integrity > 0.0 {
                        surface_z = vz;
                        break;
                    }
                }

                if surface_z < 0 {
                    nodes[idx] = NavNode { walkable: false, surface_z: 0, move_cost: 0.0 };
                    continue;
                }

                // Check that the cell above the surface is air (walkable space)
                let above = world.get_voxel(vx, vy, surface_z + 1);
                let walkable = !above.material.is_solid();

                let surface_mat = world.get_voxel(vx, vy, surface_z);
                let props = surface_mat.material.properties();
                let move_cost = if walkable {
                    if surface_mat.integrity == 0.0 {
                        1.0 + props.rubble_move_cost // Rubble penalty
                    } else {
                        1.0 // normal surface
                    }
                } else {
                    0.0
                };

                nodes[idx] = NavNode { walkable, surface_z, move_cost };
            }
        }

        Self { origin_vx, origin_vy, width, height, nodes }
    }

    #[inline]
    fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    #[inline]
    pub fn in_bounds(&self, x: u32, y: u32) -> bool {
        x < self.width && y < self.height
    }

    pub fn is_walkable(&self, x: u32, y: u32) -> bool {
        if !self.in_bounds(x, y) { return false; }
        self.nodes[self.idx(x, y)].walkable
    }

    pub fn surface_z_at(&self, x: u32, y: u32) -> i32 {
        if !self.in_bounds(x, y) { return 0; }
        self.nodes[self.idx(x, y)].surface_z
    }

    pub fn move_cost(&self, x: u32, y: u32) -> f32 {
        if !self.in_bounds(x, y) { return f32::MAX; }
        let node = &self.nodes[self.idx(x, y)];
        if !node.walkable { return f32::MAX; }
        node.move_cost
    }

    /// A* pathfinding. Returns path of (x, y) positions from start to goal,
    /// or None if no path exists. 8-connected. Max 1000 iterations.
    pub fn find_path(&self, start: (u32, u32), goal: (u32, u32)) -> Option<Vec<(u32, u32)>> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        if !self.in_bounds(start.0, start.1) || !self.in_bounds(goal.0, goal.1) {
            return None;
        }
        if start == goal { return Some(vec![goal]); }

        let n = (self.width * self.height) as usize;
        let mut g_score = vec![f32::MAX; n];
        let mut came_from = vec![u32::MAX; n];
        let mut open: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();

        let start_idx = self.idx(start.0, start.1);
        let goal_idx = self.idx(goal.0, goal.1);
        g_score[start_idx] = 0.0;

        let heuristic = |idx: usize| -> f32 {
            let x = (idx % self.width as usize) as f32;
            let y = (idx / self.width as usize) as f32;
            (x - goal.0 as f32).abs() + (y - goal.1 as f32).abs()
        };

        let f0 = (heuristic(start_idx) * 100.0) as u32;
        open.push(Reverse((f0, start_idx as u32)));

        let mut iterations = 0u32;
        const MAX_ITER: u32 = 1000;

        while let Some(Reverse((_, current_u32))) = open.pop() {
            let current = current_u32 as usize;
            if current == goal_idx {
                let mut path = Vec::new();
                let mut c = current;
                while c != start_idx {
                    path.push(((c % self.width as usize) as u32, (c / self.width as usize) as u32));
                    c = came_from[c] as usize;
                    if c == u32::MAX as usize { return None; }
                }
                path.reverse();
                return Some(path);
            }

            iterations += 1;
            if iterations >= MAX_ITER { return None; }

            let cx = (current % self.width as usize) as i32;
            let cy = (current / self.width as usize) as i32;
            let current_g = g_score[current];

            for &(dx, dy) in &[
                (-1i32, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ] {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 {
                    continue;
                }
                let nx_u = nx as u32;
                let ny_u = ny as u32;
                let ni = self.idx(nx_u, ny_u);

                if ni != goal_idx && !self.is_walkable(nx_u, ny_u) { continue; }

                // Diagonal corner-cutting prevention
                if dx != 0 && dy != 0 {
                    if !self.is_walkable(cx as u32, ny_u) || !self.is_walkable(nx_u, cy as u32) {
                        continue;
                    }
                }

                let step_cost = if dx != 0 && dy != 0 {
                    self.move_cost(nx_u, ny_u) * 1.414
                } else {
                    self.move_cost(nx_u, ny_u)
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
        None
    }

    /// Check if a voxel-space position falls within this NavGrid's coverage.
    pub fn contains_voxel(&self, vx: i32, vy: i32) -> bool {
        vx >= self.origin_vx
            && vx < self.origin_vx + self.width as i32
            && vy >= self.origin_vy
            && vy < self.origin_vy + self.height as i32
    }

    /// Rebake a rectangular sub-region of columns. Use after voxel modifications
    /// to avoid rebaking the entire grid.
    pub fn rebake_columns(&mut self, world: &VoxelWorld, min_vx: i32, min_vy: i32, max_vx: i32, max_vy: i32, max_z: i32) {
        for vy in min_vy..=max_vy {
            for vx in min_vx..=max_vx {
                if !self.contains_voxel(vx, vy) { continue; }

                let dx = (vx - self.origin_vx) as u32;
                let dy = (vy - self.origin_vy) as u32;
                let idx = self.idx(dx, dy);

                let mut surface_z = -1i32;
                for vz in (0..=max_z).rev() {
                    let v = world.get_voxel(vx, vy, vz);
                    if v.material.is_solid() && v.integrity > 0.0 {
                        surface_z = vz;
                        break;
                    }
                }

                if surface_z < 0 {
                    self.nodes[idx] = NavNode { walkable: false, surface_z: 0, move_cost: 0.0 };
                    continue;
                }

                let above = world.get_voxel(vx, vy, surface_z + 1);
                let walkable = !above.material.is_solid();

                let surface_mat = world.get_voxel(vx, vy, surface_z);
                let props = surface_mat.material.properties();
                let move_cost = if walkable {
                    if surface_mat.integrity == 0.0 {
                        1.0 + props.rubble_move_cost
                    } else {
                        1.0
                    }
                } else {
                    0.0
                };

                self.nodes[idx] = NavNode { walkable, surface_z, move_cost };
            }
        }
    }

    /// BFS flow field toward a target. Returns direction map where
    /// `flow[idx] = next_idx` (u32::MAX if unreachable). 4-connected.
    pub fn compute_flow_field(&self, target: (u32, u32)) -> Vec<u32> {
        let n = (self.width * self.height) as usize;
        let mut flow = vec![u32::MAX; n];
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();

        let target_idx = self.idx(target.0, target.1);
        flow[target_idx] = target_idx as u32;
        visited[target_idx] = true;
        queue.push_back(target_idx);

        while let Some(current) = queue.pop_front() {
            let cx = (current % self.width as usize) as i32;
            let cy = (current / self.width as usize) as i32;

            for &(dx, dy) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 { continue; }
                let ni = self.idx(nx as u32, ny as u32);
                if visited[ni] { continue; }
                if !self.is_walkable(nx as u32, ny as u32) { continue; }

                visited[ni] = true;
                flow[ni] = current as u32;
                queue.push_back(ni);
            }
        }
        flow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::voxel::{Voxel, VoxelMaterial};

    fn make_flat_world() -> VoxelWorld {
        let mut world = VoxelWorld::default();
        for y in 0..10 {
            for x in 0..10 {
                for z in 0..=5 {
                    world.set_voxel(x, y, z, Voxel::new(VoxelMaterial::Stone));
                }
            }
        }
        world
    }

    #[test]
    fn bake_flat_surface() {
        let world = make_flat_world();
        let nav = NavGrid::bake(&world, 0, 0, 10, 10, 20);

        assert_eq!(nav.width, 10);
        assert_eq!(nav.height, 10);

        for y in 0..10 {
            for x in 0..10 {
                assert!(nav.is_walkable(x, y), "({}, {}) should be walkable", x, y);
                assert_eq!(nav.surface_z_at(x, y), 5);
                assert!(nav.move_cost(x, y) > 0.0);
            }
        }
    }

    #[test]
    fn find_path_simple() {
        let world = make_flat_world();
        let nav = NavGrid::bake(&world, 0, 0, 10, 10, 20);

        let path = nav.find_path((0, 0), (5, 5));
        assert!(path.is_some());
        let path = path.unwrap();
        assert!(!path.is_empty());
        assert_eq!(*path.last().unwrap(), (5, 5));
    }

    #[test]
    fn find_path_blocked() {
        let mut world = make_flat_world();
        // Place a wall across y=5
        for x in 0..10 {
            world.set_voxel(x, 5, 6, Voxel::new(VoxelMaterial::StoneBlock));
        }
        let nav = NavGrid::bake(&world, 0, 0, 10, 10, 20);

        // Path should still exist (surface shifts up over the wall)
        let path = nav.find_path((0, 0), (0, 9));
        assert!(path.is_some());
    }

    #[test]
    fn flow_field_basic() {
        let world = make_flat_world();
        let nav = NavGrid::bake(&world, 0, 0, 10, 10, 20);

        let flow = nav.compute_flow_field((5, 5));
        let target_idx = nav.idx(5, 5);

        assert_eq!(flow[target_idx], target_idx as u32);

        for y in 0..10u32 {
            for x in 0..10u32 {
                assert_ne!(flow[nav.idx(x, y)], u32::MAX, "({}, {}) should be reachable", x, y);
            }
        }
    }
}
