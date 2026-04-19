//! Pure voxel-grid helpers. No Vulkan, no sim state.
//! Engine convention: Y-up. 1 voxel = 1 m. Grid is GRID_SIDE³.

use glam::Vec3;
use voxel_engine::voxel::grid::VoxelGrid;

use crate::palette::{PAL_AIR, PAL_GROUND};

pub const GRID_SIDE: u32 = 128;
pub const GROUND_Y:  u32 = 2;

pub fn paint_ground_plane(grid: &mut VoxelGrid) {
    let (w, _h, d) = grid.dimensions();
    for z in 0..d {
        for x in 0..w {
            grid.set(x, GROUND_Y, z, PAL_GROUND);
        }
    }
}

pub fn clear_above_ground(grid: &mut VoxelGrid) {
    let (w, h, d) = grid.dimensions();
    for y in (GROUND_Y + 1)..h {
        for z in 0..d {
            for x in 0..w {
                grid.set(x, y, z, PAL_AIR);
            }
        }
    }
}

pub fn grid_index_of(pos: Vec3, grid: &VoxelGrid) -> Option<(u32, u32, u32)> {
    let (w, h, d) = grid.dimensions();
    let x = pos.x.floor() as i32;
    let y = pos.y.floor() as i32;
    let z = pos.z.floor() as i32;
    if x < 0 || y < 0 || z < 0 { return None; }
    if x as u32 >= w || y as u32 >= h || z as u32 >= d { return None; }
    Some((x as u32, y as u32, z as u32))
}

pub fn paint_agent(grid: &mut VoxelGrid, pos: Vec3, palette_idx: u8) {
    // Lift to sit on top of the ground plane.
    let lifted = Vec3::new(pos.x, pos.y.max((GROUND_Y + 1) as f32), pos.z);
    if let Some((x, y, z)) = grid_index_of(lifted, grid) {
        grid.set(x, y, z, palette_idx);
    }
}

/// 3D Bresenham-ish parametric line. Out-of-bounds voxels skipped.
pub fn paint_line(grid: &mut VoxelGrid, from: Vec3, to: Vec3, palette_idx: u8) {
    let dx = (to.x - from.x).abs();
    let dy = (to.y - from.y).abs();
    let dz = (to.z - from.z).abs();
    let steps = dx.max(dy).max(dz).ceil() as i32;
    if steps <= 0 {
        if let Some((x, y, z)) = grid_index_of(from, grid) {
            grid.set(x, y, z, palette_idx);
        }
        return;
    }
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let p = from.lerp(to, t);
        if let Some((x, y, z)) = grid_index_of(p, grid) {
            grid.set(x, y, z, palette_idx);
        }
    }
}

/// Axis-aligned ring (constant y) via polar sampling.
pub fn paint_ring(grid: &mut VoxelGrid, center: Vec3, radius: f32, palette_idx: u8) {
    let r = radius.max(0.0);
    let steps = ((2.0 * std::f32::consts::PI * r).ceil() as i32).max(8);
    for i in 0..steps {
        let theta = (i as f32) / (steps as f32) * std::f32::consts::TAU;
        let x = center.x + r * theta.cos();
        let z = center.z + r * theta.sin();
        if let Some((gx, gy, gz)) = grid_index_of(Vec3::new(x, center.y, z), grid) {
            grid.set(gx, gy, gz, palette_idx);
        }
    }
}
