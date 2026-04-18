//! Structural integrity tick — detects and collapses unsupported voxels.
//!
//! When NPCs harvest a tree trunk or mine away supporting stone, the canopy
//! or overhang above may become disconnected from the ground. This system
//! detects such "floating" fragments via BFS from ground-anchored voxels and
//! removes them (simplified collapse for Phase 1 — no rigid-body physics).

use std::collections::{HashSet, VecDeque};

/// HashSet<(i32,i32,i32)> with ahash. The default SipHash was measured at
/// ~27% of total program time doing voxel-position lookups in this system.
type VoxelSet = HashSet<(i32, i32, i32), ahash::RandomState>;

use crate::world_sim::state::{CollapseCase, StructuralEvent, WorldState};
use crate::world_sim::voxel::{ChunkPos, VoxelMaterial, Voxel, CHUNK_SIZE};

/// Maximum number of dirty chunks to process per structural tick.
const MAX_CHUNKS_PER_TICK: usize = 4;

/// Run structural integrity checks on dirty chunks. Unsupported voxels are
/// set to Air and a `StructuralEvent::FragmentCollapse` is emitted for each
/// disconnected fragment.
pub fn structural_tick(state: &mut WorldState) {
    // Collect dirty chunk positions (up to limit).
    let dirty_chunks: Vec<ChunkPos> = state
        .voxel_world
        .chunks
        .iter()
        .filter(|(_, c)| c.dirty)
        .map(|(cp, _)| *cp)
        .take(MAX_CHUNKS_PER_TICK)
        .collect();

    for cp in dirty_chunks {
        let unsupported = find_unsupported_voxels(state, cp);
        if unsupported.is_empty() {
            continue;
        }

        // Remove unsupported voxels (simplified collapse).
        for &(vx, vy, vz) in &unsupported {
            state.voxel_world.set_voxel(vx, vy, vz, Voxel::default());
        }

        state.structural_events.push(StructuralEvent::FragmentCollapse {
            chunk_x: cp.x,
            chunk_y: cp.y,
            chunk_z: cp.z,
            fragment_voxel_count: unsupported.len() as u32,
            cause: CollapseCase::Natural,
        });
    }
}

/// Find solid voxels in `cp` that are not connected to the ground via
/// face-adjacent solid neighbors. Uses BFS from ground-anchored voxels;
/// any solid voxel not reached is unsupported.
fn find_unsupported_voxels(state: &WorldState, cp: ChunkPos) -> Vec<(i32, i32, i32)> {
    let chunk = match state.voxel_world.chunks.get(&cp) {
        Some(c) => c,
        None => return Vec::new(),
    };

    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;
    let base_z = cp.z * CHUNK_SIZE as i32;

    // Collect all solid voxel world-positions in this chunk. Pre-size to
    // avoid the rehash cascade — flamegraph showed `reserve_rehash` at
    // 19% of total program time when the sets grew from empty. A chunk
    // holds at most CHUNK_SIZE³ voxels; allocating that upfront once is
    // cheaper than the log₂(N) rehashes during incremental growth.
    let chunk_cap = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;
    let mut solid_set: VoxelSet = VoxelSet::with_capacity_and_hasher(
        chunk_cap, ahash::RandomState::default());
    for lz in 0..CHUNK_SIZE {
        for ly in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let v = chunk.get(lx, ly, lz);
                if v.material.is_solid() {
                    solid_set.insert((base_x + lx as i32, base_y + ly as i32, base_z + lz as i32));
                }
            }
        }
    }

    if solid_set.is_empty() {
        return Vec::new();
    }

    // Seed BFS with anchored voxels — those at z<=0 or whose below-neighbor is solid.
    let mut visited: VoxelSet = VoxelSet::with_capacity_and_hasher(
        solid_set.len(), ahash::RandomState::default());
    let mut queue: VecDeque<(i32, i32, i32)> = VecDeque::new();

    for &(vx, vy, vz) in &solid_set {
        let anchored = if vz <= 0 {
            // Bottom of the world — always anchored.
            true
        } else if state.voxel_world.get_voxel(vx, vy, vz).material == VoxelMaterial::Granite {
            // Bedrock is always anchored.
            true
        } else {
            // Anchored if the voxel directly below is solid (cross-chunk safe).
            state.voxel_world.get_voxel(vx, vy, vz - 1).material.is_solid()
        };

        if anchored {
            visited.insert((vx, vy, vz));
            queue.push_back((vx, vy, vz));
        }
    }

    // BFS through 6-connected solid neighbors (cross-chunk via get_voxel).
    let offsets: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ];

    while let Some((vx, vy, vz)) = queue.pop_front() {
        for (dx, dy, dz) in &offsets {
            let nx = vx + dx;
            let ny = vy + dy;
            let nz = vz + dz;
            let pos = (nx, ny, nz);

            if visited.contains(&pos) {
                continue;
            }

            // Only propagate through solid voxels. We check the world (cross-chunk)
            // but only track voxels that belong to the chunk we're analyzing.
            if solid_set.contains(&pos) {
                visited.insert(pos);
                queue.push_back(pos);
            } else if state.voxel_world.get_voxel(nx, ny, nz).material.is_solid() {
                // Solid neighbor in an adjacent chunk — follow BFS so it can
                // reach back into our chunk from the other side.
                visited.insert(pos);
                queue.push_back(pos);
            }
        }
    }

    // Unsupported = solid voxels in this chunk not reached by BFS.
    solid_set
        .into_iter()
        .filter(|pos| !visited.contains(pos))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::voxel::{Voxel, VoxelMaterial};

    #[test]
    fn unsupported_voxels_collapse() {
        let mut state = WorldState::new(42);
        // Place a floating voxel at (5, 5, 40) with nothing below.
        state.voxel_world.set_voxel(5, 5, 40, Voxel::new(VoxelMaterial::Dirt));
        // Run structural tick.
        structural_tick(&mut state);
        // Floating voxel should be gone (collapsed).
        assert_eq!(
            state.voxel_world.get_voxel(5, 5, 40).material,
            VoxelMaterial::Air
        );
        assert!(!state.structural_events.is_empty());
    }

    #[test]
    fn supported_voxels_survive() {
        let mut state = WorldState::new(42);
        // Generate terrain chunks so ground exists.
        state
            .voxel_world
            .generate_chunk(ChunkPos::new(0, 0, 0), 42, None);
        state
            .voxel_world
            .generate_chunk(ChunkPos::new(0, 0, 1), 42, None);
        // Voxels on terrain are supported and should NOT collapse.
        let surface = state.voxel_world.surface_height(8, 8);
        let mat = state.voxel_world.get_voxel(8, 8, surface - 1).material;
        assert!(mat.is_solid(), "expected solid voxel at surface-1");
        structural_tick(&mut state);
        // Surface voxel should still be there.
        assert_eq!(
            state.voxel_world.get_voxel(8, 8, surface - 1).material,
            mat
        );
        assert!(state.structural_events.is_empty());
    }

    #[test]
    fn column_of_voxels_stays_supported() {
        let mut state = WorldState::new(42);
        // Build a column from z=0 up to z=5 — all should survive.
        for z in 0..=5 {
            state.voxel_world.set_voxel(3, 3, z, Voxel::new(VoxelMaterial::Stone));
        }
        structural_tick(&mut state);
        for z in 0..=5 {
            assert_eq!(
                state.voxel_world.get_voxel(3, 3, z).material,
                VoxelMaterial::Stone,
                "voxel at z={z} should survive"
            );
        }
        assert!(state.structural_events.is_empty());
    }

    #[test]
    fn floating_horizontal_slab_collapses() {
        let mut state = WorldState::new(42);
        // Place a 2x2 horizontal slab floating at z=30, no solid below any of them.
        for x in 10..12 {
            for y in 10..12 {
                state.voxel_world.set_voxel(x, y, 30, Voxel::new(VoxelMaterial::Stone));
            }
        }
        structural_tick(&mut state);
        // All slab voxels should collapse — none have solid below.
        for x in 10..12 {
            for y in 10..12 {
                assert_eq!(
                    state.voxel_world.get_voxel(x, y, 30).material,
                    VoxelMaterial::Air,
                    "floating slab voxel at ({x},{y},30) should collapse"
                );
            }
        }
        assert!(!state.structural_events.is_empty());
    }

    #[test]
    fn stacked_pair_with_gap_survives() {
        let mut state = WorldState::new(42);
        // z=0,1 solid, z=2 air, z=3,4 solid.
        // z=4 has z=3 solid below it, so z=4 is anchored.
        // BFS from z=4 reaches z=3. Both survive (per spec: below-neighbor = anchor).
        for z in 0..=1 {
            state.voxel_world.set_voxel(3, 3, z, Voxel::new(VoxelMaterial::Stone));
        }
        for z in 3..=4 {
            state.voxel_world.set_voxel(3, 3, z, Voxel::new(VoxelMaterial::Stone));
        }
        structural_tick(&mut state);
        for z in [0, 1, 3, 4] {
            assert_eq!(
                state.voxel_world.get_voxel(3, 3, z).material,
                VoxelMaterial::Stone,
                "voxel at z={z} should survive (anchored chain)"
            );
        }
        assert!(state.structural_events.is_empty());
    }
}
