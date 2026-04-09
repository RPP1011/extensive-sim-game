//! River carving pass — carves a valley and water channel for each `RiverPath`.
//!
//! For each voxel in the chunk, the closest distance to the river polyline is
//! tested.  Within the river width the column is carved:
//!   - Center channel (dist < width*0.4): lower half Water, upper Air
//!   - Bank zone     (dist < width*0.7): gradual slope carved to Air

use crate::world_sim::voxel::{Chunk, ChunkPos, Voxel, VoxelMaterial, CHUNK_SIZE, local_index};
use super::region_plan::RiverPath;

// ---------------------------------------------------------------------------
// Closest-point helpers
// ---------------------------------------------------------------------------

/// Project point P onto segment AB, return clamped parameter t in [0,1].
fn segment_closest_t(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-6 {
        return 0.0;
    }
    ((px - ax) * dx + (py - ay) * dy) / len2
}

/// Walk the river polyline and return `(distance, interpolated_width)` for
/// the closest point on any segment to `(px, py)`.
pub fn closest_river_distance(px: f32, py: f32, river: &RiverPath) -> (f32, f32) {
    let mut best_dist = f32::MAX;
    let mut best_width = 1.0f32;

    let pts = &river.points;
    let wids = &river.widths;

    for i in 0..(pts.len().saturating_sub(1)) {
        let (ax, ay) = pts[i];
        let (bx, by) = pts[i + 1];
        let t = segment_closest_t(px, py, ax, ay, bx, by).clamp(0.0, 1.0);
        let cx = ax + (bx - ax) * t;
        let cy = ay + (by - ay) * t;
        let dist = ((px - cx) * (px - cx) + (py - cy) * (py - cy)).sqrt();
        if dist < best_dist {
            best_dist = dist;
            // Interpolate width between the two endpoints
            let w0 = wids[i];
            let w1 = if i + 1 < wids.len() { wids[i + 1] } else { wids[i] };
            best_width = w0 + (w1 - w0) * t;
        }
    }

    (best_dist, best_width)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Carve a river valley and water channel into a chunk.
///
/// `surface_z_fn` gives the terrain surface height (voxel Z) at any (x, y).
pub fn carve_river_in_chunk(
    chunk: &mut Chunk,
    cp: ChunkPos,
    river: &RiverPath,
    surface_z_fn: &dyn Fn(f32, f32) -> i32,
) {
    if river.points.len() < 2 {
        return;
    }

    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;
    let base_z = cp.z * CHUNK_SIZE as i32;

    for ly in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let vx = base_x + lx as i32;
            let vy = base_y + ly as i32;

            let (dist, width) = closest_river_distance(vx as f32, vy as f32, river);

            if dist >= width {
                continue; // outside river influence
            }

            let surface_z = surface_z_fn(vx as f32, vy as f32);
            let bed_depth = (width * 0.3).max(2.0) as i32;
            let river_surface_z = surface_z; // river runs at terrain surface
            let river_bed_z = river_surface_z - bed_depth;

            let dist_ratio = dist / width;

            for lz in 0..CHUNK_SIZE {
                let vz = base_z + lz as i32;
                let idx = local_index(lx, ly, lz);

                // Never touch non-solid (e.g. air above terrain, water already placed)
                // but do carve into solid rock/soil
                let mat = chunk.voxels[idx].material;
                if mat == VoxelMaterial::Granite {
                    continue; // never carve bedrock
                }

                if dist_ratio < 0.4 {
                    // Center channel — carve valley, fill with water in lower half
                    if vz <= river_surface_z && vz >= river_bed_z {
                        // Lower half of channel range → Water; upper half → Air
                        let midpoint = (river_bed_z + river_surface_z) / 2;
                        let fill = if vz <= midpoint {
                            VoxelMaterial::Water
                        } else {
                            VoxelMaterial::Air
                        };
                        chunk.voxels[idx] = Voxel::new(fill);
                    } else if vz > river_surface_z && vz <= river_surface_z + 1 {
                        // One voxel above surface: make air (open top of channel)
                        if mat.is_solid() {
                            chunk.voxels[idx] = Voxel::new(VoxelMaterial::Air);
                        }
                    }
                } else if dist_ratio < 0.7 {
                    // Bank zone — gradual slope: carve top few voxels to Air
                    let bank_t = (dist_ratio - 0.4) / 0.3; // 0 at inner edge, 1 at outer
                    let carve_depth = ((1.0 - bank_t) * bed_depth as f32 * 0.5) as i32;
                    let bank_top = river_surface_z;
                    let bank_bot = river_surface_z - carve_depth;
                    if vz <= bank_top && vz >= bank_bot && mat.is_solid() {
                        chunk.voxels[idx] = Voxel::new(VoxelMaterial::Air);
                    }
                }
            }
        }
    }

    chunk.dirty = true;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::voxel::{ChunkPos, CHUNK_SIZE};
    use crate::world_sim::terrain::region_plan::RiverPath;

    fn fill_chunk(cp: ChunkPos, mat: VoxelMaterial) -> Chunk {
        let mut chunk = Chunk::new_air(cp);
        for v in chunk.voxels.iter_mut() {
            *v = Voxel::new(mat);
        }
        chunk
    }

    /// A perfectly straight river running through the middle of chunk (0,0,z)
    /// in the X direction.
    fn straight_river_x(y_world: f32, width: f32) -> RiverPath {
        RiverPath {
            points: vec![(-100.0, y_world), (300.0, y_world)],
            widths: vec![width, width],
        }
    }

    #[test]
    fn river_carves_water_channel() {
        // Surface chunk: cp.z=5 → base_z=80, so voxels 80..96
        let cp = ChunkPos::new(0, 0, 5);
        let mut chunk = fill_chunk(cp, VoxelMaterial::Dirt);

        // River centered at y=8 (middle of chunk column), width=20
        // Surface z function returns 88 (roughly middle of chunk)
        let river = straight_river_x(8.0, 20.0);
        let surface_fn = |_vx: f32, _vy: f32| -> i32 { 88 };

        carve_river_in_chunk(&mut chunk, cp, &river, &surface_fn);

        let water_count = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Water).count();
        assert!(water_count > 0, "river carved no water voxels");
    }

    #[test]
    fn river_carving_is_deterministic() {
        let cp = ChunkPos::new(1, 0, 5);
        let river = straight_river_x(8.0, 16.0);
        let surface_fn = |_: f32, _: f32| -> i32 { 85 };

        let mut a = fill_chunk(cp, VoxelMaterial::Dirt);
        let mut b = fill_chunk(cp, VoxelMaterial::Dirt);
        carve_river_in_chunk(&mut a, cp, &river, &surface_fn);
        carve_river_in_chunk(&mut b, cp, &river, &surface_fn);

        for i in 0..a.voxels.len() {
            assert_eq!(a.voxels[i].material, b.voxels[i].material, "voxel {i} differs");
        }
    }

    #[test]
    fn granite_not_carved_by_river() {
        let cp = ChunkPos::new(0, 0, -10);
        let mut chunk = fill_chunk(cp, VoxelMaterial::Granite);
        let river = straight_river_x(8.0, 30.0);
        let surface_fn = |_: f32, _: f32| -> i32 { -150 };
        carve_river_in_chunk(&mut chunk, cp, &river, &surface_fn);
        let granite_count = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Granite).count();
        assert_eq!(granite_count, CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE, "granite was modified");
    }
}
