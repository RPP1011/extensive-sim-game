//! Structural integrity tick — detects and collapses unsupported voxels.
//!
//! When NPCs harvest a tree trunk or mine away supporting stone, the canopy
//! or overhang above may become disconnected from the ground. This system
//! detects such "floating" fragments via BFS from ground-anchored voxels and
//! removes them (simplified collapse for Phase 1 — no rigid-body physics).

use std::collections::{HashSet, VecDeque};

/// HashSet<(i32,i32,i32)> with ahash — used only for the rare cross-chunk
/// BFS case. In-chunk lookups go through flat bool arrays (zero hashing).
type VoxelSet = HashSet<(i32, i32, i32), ahash::RandomState>;

use crate::world_sim::state::{CollapseCase, StructuralEvent, WorldState};
use crate::world_sim::voxel::{local_index, ChunkPos, VoxelMaterial, Voxel, CHUNK_SIZE};

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

    if dirty_chunks.is_empty() {
        return;
    }

    // Allocate the flat-bool BFS buffers ONCE per invocation and reuse
    // across the up-to-4 dirty chunks. Each is CHUNK_SIZE³ bools = ~256KB,
    // so avoiding the per-chunk realloc + memset is measurable.
    const N: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
    let mut buf = FindBuffers {
        solid_local: vec![false; N],
        visited_local: vec![false; N],
    };

    for cp in dirty_chunks {
        let unsupported = find_unsupported_voxels(state, cp, &mut buf);

        // Clear dirty after structural processing. Without this the chunk
        // stays dirty forever (only the Vulkan renderer + NDJSON bridge
        // otherwise clear it — neither runs in headless bench paths), so
        // structural_tick rechecks the same chunks every cadence for no
        // gain. Clearing here turns structural_tick into genuine idle
        // work once chunks settle.
        if let Some(chunk) = state.voxel_world.chunks.get_mut(&cp) {
            chunk.dirty = false;
        }

        if unsupported.is_empty() {
            continue;
        }

        // Remove unsupported voxels (simplified collapse). set_voxel
        // re-marks the containing chunk dirty, so the next structural
        // tick will revisit it — correct: new fragments may emerge.
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

/// Reusable scratch buffers for `find_unsupported_voxels`. Allocated once
/// per `structural_tick` and reset between dirty chunks. Each Vec is
/// CHUNK_SIZE³ bools (~256KB); one alloc per tick instead of one per
/// chunk meaningfully reduces memset + allocator overhead.
struct FindBuffers {
    solid_local: Vec<bool>,
    visited_local: Vec<bool>,
}

impl FindBuffers {
    fn reset(&mut self) {
        // fill is vectorizable; much cheaper than allocating a fresh Vec.
        self.solid_local.fill(false);
        self.visited_local.fill(false);
    }
}

/// Find solid voxels in `cp` that are not connected to the ground via
/// face-adjacent solid neighbors. Uses BFS from ground-anchored voxels;
/// any solid voxel not reached is unsupported.
fn find_unsupported_voxels(
    state: &WorldState,
    cp: ChunkPos,
    buf: &mut FindBuffers,
) -> Vec<(i32, i32, i32)> {
    let chunk = match state.voxel_world.chunks.get(&cp) {
        Some(c) => c,
        None => return Vec::new(),
    };

    let cs = CHUNK_SIZE as i32;
    let base_x = cp.x * cs;
    let base_y = cp.y * cs;
    let base_z = cp.z * cs;

    // Reuse the caller's BFS buffers. Flat bool arrays for in-chunk
    // membership — CHUNK_SIZE³ ≈ 262K bools each — with zero hashing.
    buf.reset();
    let solid_local = &mut buf.solid_local[..];
    let visited_local = &mut buf.visited_local[..];

    // Cross-chunk BFS tracking. Preserves the original unlimited-depth
    // cross-chunk traversal so arches spanning multiple chunks still anchor
    // correctly. Typically very small (<64 entries) so ahash overhead is
    // negligible here.
    let mut visited_ext: VoxelSet =
        VoxelSet::with_capacity_and_hasher(64, ahash::RandomState::default());

    // Pass 1: scan chunk.voxels into solid_local and count solids.
    let mut any_solid = false;
    for lz in 0..CHUNK_SIZE {
        for ly in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                if chunk.get(lx, ly, lz).material.is_solid() {
                    solid_local[local_index(lx, ly, lz)] = true;
                    any_solid = true;
                }
            }
        }
    }
    if !any_solid {
        return Vec::new();
    }

    // Helper: world coord → Some(local_index) if in this chunk, else None.
    let to_local_idx = |vx: i32, vy: i32, vz: i32| -> Option<usize> {
        let lx = vx - base_x;
        let ly = vy - base_y;
        let lz = vz - base_z;
        if lx >= 0 && lx < cs && ly >= 0 && ly < cs && lz >= 0 && lz < cs {
            Some(local_index(lx as usize, ly as usize, lz as usize))
        } else {
            None
        }
    };

    // Seed BFS with anchored voxels. Queue stores world coords so the
    // BFS step can equally traverse in-chunk (fast path via
    // solid_local/visited_local) and cross-chunk (fallback via
    // visited_ext + world get_voxel).
    let mut queue: VecDeque<(i32, i32, i32)> = VecDeque::new();
    for lz in 0..CHUNK_SIZE {
        for ly in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let idx = local_index(lx, ly, lz);
                if !solid_local[idx] {
                    continue;
                }
                let vx = base_x + lx as i32;
                let vy = base_y + ly as i32;
                let vz = base_z + lz as i32;
                let anchored = if vz <= 0 {
                    true
                } else if chunk.get(lx, ly, lz).material == VoxelMaterial::Granite {
                    true
                } else if lz > 0 {
                    // Below is in-chunk: flat-array lookup.
                    solid_local[local_index(lx, ly, lz - 1)]
                } else {
                    // Below is cross-chunk: world get_voxel.
                    state
                        .voxel_world
                        .get_voxel(vx, vy, vz - 1)
                        .material
                        .is_solid()
                };

                if anchored {
                    visited_local[idx] = true;
                    queue.push_back((vx, vy, vz));
                }
            }
        }
    }

    // BFS through 6-connected solid neighbors. In-chunk neighbors use
    // flat-array lookups (hot path); cross-chunk neighbors use the
    // visited_ext HashSet + world get_voxel (rare).
    let offsets: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ];

    while let Some((vx, vy, vz)) = queue.pop_front() {
        for &(dx, dy, dz) in &offsets {
            let nx = vx + dx;
            let ny = vy + dy;
            let nz = vz + dz;

            match to_local_idx(nx, ny, nz) {
                Some(ni) => {
                    if visited_local[ni] || !solid_local[ni] {
                        continue;
                    }
                    visited_local[ni] = true;
                    queue.push_back((nx, ny, nz));
                }
                None => {
                    let pos = (nx, ny, nz);
                    if visited_ext.contains(&pos) {
                        continue;
                    }
                    if state.voxel_world.get_voxel(nx, ny, nz).material.is_solid() {
                        visited_ext.insert(pos);
                        queue.push_back(pos);
                    }
                }
            }
        }
    }

    // Unsupported = in-chunk solid voxels not reached by BFS.
    let mut result = Vec::new();
    for lz in 0..CHUNK_SIZE {
        for ly in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let idx = local_index(lx, ly, lz);
                if solid_local[idx] && !visited_local[idx] {
                    result.push((
                        base_x + lx as i32,
                        base_y + ly as i32,
                        base_z + lz as i32,
                    ));
                }
            }
        }
    }
    result
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
