//! Map exploration tracking system — every 3 ticks.
//!
//! Tracks NPC movement across the world to reveal regions and discover
//! landmarks. Explored areas near settlements boost treasury (representing
//! cartography bonuses). Milestone rewards are expressed as treasury updates
//! and fidelity changes.
//!
//! **Gold conservation:** Frontier exploration rewards are paid from the
//! nearest settlement treasury. If no settlement can afford it, no gold
//! is paid.
//!
//! Ported from `crates/headless_campaign/src/systems/exploration.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::fidelity::Fidelity;
use crate::world_sim::state::{ActionTags, EntityKind, VoxelResourceKnowledge, WorldState, WorldTeam, tags};
use crate::world_sim::voxel::{VoxelMaterial, world_to_voxel, VOXEL_SCALE};

use std::sync::OnceLock;

/// Cadence: runs every 3 ticks.
const EXPLORATION_INTERVAL: u64 = 3;

/// Tile cell size in world units.
const TILE_SIZE: f32 = 10.0;

/// Base sight range in tiles for a normal NPC.
const BASE_SIGHT_RANGE: i32 = 2;

/// Landmark discovery radius in world units.
const LANDMARK_DISCOVERY_RADIUS_SQ: f32 = 144.0; // 12^2

/// World bounds for exploration grid.
const WORLD_MIN: f32 = -50.0;
const WORLD_MAX: f32 = 50.0;

/// Treasury bonus when an NPC explores near a settlement.
const EXPLORATION_TREASURY_BONUS: f32 = 0.1;

pub fn compute_exploration(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % EXPLORATION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_exploration_for_settlement(state, settlement.id, &state.entities[range], out);
    }

    // Frontier exploration: unaffiliated NPCs far from settlements.
    let unaffiliated = state.group_index.unaffiliated_entities();
    for entity in &state.entities[unaffiliated] {
        if entity.kind != EntityKind::Npc || !entity.alive || entity.team != WorldTeam::Friendly {
            continue;
        }
        let nearest = state
            .settlements
            .iter()
            .min_by(|a, b| {
                let da = dist_sq(entity.pos, a.pos);
                let db = dist_sq(entity.pos, b.pos);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

        let min_settlement_dist = nearest.map(|s| dist_sq(entity.pos, s.pos)).unwrap_or(f32::MAX);

        if min_settlement_dist > 900.0 {
            // Pay frontier exploration stipend from nearest settlement treasury
            if let Some(s) = nearest {
                if s.treasury > 0.5 {
                    out.push(WorldDelta::TransferGold {
                        from_entity: s.id,
                        to_entity: entity.id,
                        amount: 0.5,
                    });
                }
            }

            // Behavior tags: frontier exploration.
            let mut action = ActionTags::empty();
            action.add(tags::EXPLORATION, 1.0);
            action.add(tags::NAVIGATION, 0.5);
            let action = crate::world_sim::action_context::with_context(&action, entity, state);
            out.push(WorldDelta::AddBehaviorTags { entity_id: entity.id, tags: action.tags, count: action.count });
        }
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_exploration_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % EXPLORATION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // NPCs at this settlement: boost treasury via exploration intel.
    let npc_count = entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc && e.team == WorldTeam::Friendly)
        .count();
    if npc_count > 0 {
        out.push(WorldDelta::UpdateTreasury {
            settlement_id: settlement_id,
            delta: EXPLORATION_TREASURY_BONUS * npc_count as f32,
        });
    }

    // Scouting visibility boost: nearby NPCs escalate grid fidelity.
    let grid_id = match settlement.grid_id {
        Some(gid) => gid,
        None => return,
    };

    let nearby_npc_count = state
        .entities
        .iter()
        .filter(|e| {
            e.kind == EntityKind::Npc
                && e.alive
                && e.team == WorldTeam::Friendly
                && dist_sq(e.pos, settlement.pos) <= 400.0
        })
        .count();

    if nearby_npc_count >= 3 {
        let current_fidelity = state
            .fidelity_zone(grid_id)
            .map(|g| g.fidelity)
            .unwrap_or(Fidelity::Low);

        if matches!(current_fidelity, Fidelity::Low | Fidelity::Background) {
            out.push(WorldDelta::EscalateFidelity {
                grid_id,
                new_fidelity: Fidelity::Medium,
            });
        }
    }
}

fn dist_sq(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

// ---------------------------------------------------------------------------
// Voxel resource scanning
// ---------------------------------------------------------------------------

/// Cadence for voxel resource scanning (every 20 ticks).
const RESOURCE_SCAN_INTERVAL: u64 = 20;

/// Default sight range in voxels for resource scanning.
/// 80 voxels * 0.025 m/voxel = 2m. That's quite small for a simulation with
/// tile sizes of 10 world-units; use a larger value appropriate for the sim.
const SIGHT_RANGE_VOXELS: i32 = 80;

/// Coarse grid cell size in voxels for grouping discovered resources.
const RESOURCE_CELL_SIZE: i32 = 128;

/// Materials that NPCs look for when scanning.
const TARGET_MATERIALS: &[VoxelMaterial] = &[
    VoxelMaterial::WoodLog,
    VoxelMaterial::IronOre,
    VoxelMaterial::CopperOre,
    VoxelMaterial::GoldOre,
    VoxelMaterial::Coal,
    VoxelMaterial::Crystal,
];

const NUM_TARGET_MATERIALS: usize = 6;

/// Map a VoxelMaterial to its index in TARGET_MATERIALS, or None if not tracked.
#[inline]
fn target_material_idx(m: VoxelMaterial) -> Option<usize> {
    match m {
        VoxelMaterial::WoodLog => Some(0),
        VoxelMaterial::IronOre => Some(1),
        VoxelMaterial::CopperOre => Some(2),
        VoxelMaterial::GoldOre => Some(3),
        VoxelMaterial::Coal => Some(4),
        VoxelMaterial::Crystal => Some(5),
        _ => None,
    }
}

/// Per-resource-cell count of target materials. Key: `(cell_x, cell_y)` where
/// a cell spans `RESOURCE_CELL_SIZE` voxels. Values: count of each target
/// material anywhere in the cell's surface band (surface-5..surface+15).
///
/// Persistent on WorldState; populated lazily when any NPC can see a cell.
/// The old per-NPC scan did 420K voxel reads PER NPC to produce ~40 cell
/// counts; the per-cell approach does ~344K voxel reads PER CELL ONCE and
/// reuses across all NPCs (and across ticks — the world is mostly static).
pub type CellCensus = std::collections::HashMap<(i32, i32), [u32; NUM_TARGET_MATERIALS], ahash::RandomState>;

/// Scan nearby voxels around an NPC and record harvestable deposits in its
/// `known_voxel_resources`. This is called as a post-apply step in runtime.rs,
/// not via the delta system, because it mutates NpcData directly.
/// Pack (vx, vy) i32 pair into u64. Used as the surface_cache key —
/// ahash has a fast path for single u64 hashing (one multiply + fold),
/// measurably cheaper than hashing a (i32,i32) tuple which goes through
/// write_i32 twice. Identity-hash was attempted and caused catastrophic
/// bucket collisions on small-origin worlds; do not inline a naive
/// u64-as-hash scheme here.
#[inline]
fn pack_xy(vx: i32, vy: i32) -> u64 {
    ((vx as u32 as u64) << 32) | (vy as u32 as u64)
}

/// Cache of surface heights keyed by packed (vx, vy). Persistent on
/// WorldState; survives across ticks.
pub type SurfaceCache = std::collections::HashMap<u64, i32, ahash::RandomState>;

/// Precomputed (dx, dy) offsets inside the SIGHT_RANGE_VOXELS disk.
/// Computed once, shared across all scans. Iterating a Vec<(i32, i32)> is
/// cheaper than two nested i32 ranges with an inner circle test
/// (flamegraph showed 32% `next<i32>` + 22% div_euclid in the hot path).
fn disk_offsets() -> &'static [(i32, i32)] {
    static DISK: OnceLock<Vec<(i32, i32)>> = OnceLock::new();
    DISK.get_or_init(|| {
        let r = SIGHT_RANGE_VOXELS;
        let rsq = (r as i64) * (r as i64);
        let mut v = Vec::with_capacity((r as usize * 2 + 1).pow(2));
        for dx in -r..=r {
            for dy in -r..=r {
                let d2 = (dx as i64) * (dx as i64) + (dy as i64) * (dy as i64);
                if d2 <= rsq {
                    v.push((dx, dy));
                }
            }
        }
        v
    })
}

pub fn scan_voxel_resources(state: &mut WorldState, entity_idx: usize, sight_range_voxels: i32) {
    let mut cache = SurfaceCache::default();
    scan_voxel_resources_cached(state, entity_idx, sight_range_voxels, &mut cache);
}

fn scan_voxel_resources_cached(
    state: &mut WorldState,
    entity_idx: usize,
    sight_range_voxels: i32,
    surface_cache: &mut SurfaceCache,
) {
    let entity = &state.entities[entity_idx];
    if !entity.alive || entity.kind != EntityKind::Npc || entity.npc.is_none() {
        return;
    }

    let (cvx, cvy, _) = world_to_voxel(entity.pos.0, entity.pos.1, 0.0);

    // Accumulate counts per (cell_x, cell_y, material). ahash here too —
    // keyed by a small composite, default SipHash was measurable.
    let mut counts: std::collections::HashMap<(i32, i32, VoxelMaterial), u32, ahash::RandomState> =
        std::collections::HashMap::default();

    debug_assert_eq!(sight_range_voxels, SIGHT_RANGE_VOXELS);

    // Chunk-major iteration. The old per-position loop called
    // chunks.get(&cp) once per (vx, vy) ≈ 20K times per NPC. Iterating by
    // overlapping chunk XY bbox (~9-16 chunks for a 80-radius disk at
    // CHUNK_SIZE=64) then iterating positions within each chunk turns that
    // into a handful of chunk lookups. Eliminates the 9% chunks.get cost
    // and several div_euclid calls per position.
    let cs = crate::world_sim::voxel::CHUNK_SIZE as i32;
    let sight = SIGHT_RANGE_VOXELS;
    let sight_sq = (sight as i64) * (sight as i64);
    let cx_min = (cvx - sight).div_euclid(cs);
    let cx_max = (cvx + sight).div_euclid(cs);
    let cy_min = (cvy - sight).div_euclid(cs);
    let cy_max = (cvy + sight).div_euclid(cs);

    // Using raw pointers to dodge the borrow-checker: we hold the
    // immutable borrow of state.voxel_world.chunks for the whole inner
    // loop, and pointer dereferences stay inside that scope.
    let chunks = &state.voxel_world.chunks;

    // RESOURCE_CELL_SIZE=128 is exactly 2×CHUNK_SIZE=64, so cell_x
    // is constant within a chunk column (all (vx) in [cx*64, cx*64+64)
    // map to the same cell = cx/2). Precompute per-column to hoist two
    // div_euclid calls out of the hot inner loop.
    const CELL_DIV: i32 = RESOURCE_CELL_SIZE / (crate::world_sim::voxel::CHUNK_SIZE as i32);

    for cx in cx_min..=cx_max {
        let cell_x = cx.div_euclid(CELL_DIV);
        for cy in cy_min..=cy_max {
            let cell_y = cy.div_euclid(CELL_DIV);

            // Two-slot scalar cache for Z chunks in this column. The vast
            // majority of the time z_min..z_max fits in one chunk; second
            // slot covers the occasional spill across a chunk boundary.
            // Scalar tracking (cz_a_valid, etc.) is much cheaper than a
            // Vec with .first()/.last() freshness checks (was 25% of the
            // scan cost across next/find/is_empty).
            type ChunkSlot = Option<*const crate::world_sim::voxel::Chunk>;
            let mut cached_cz_min: i32 = i32::MIN;
            let mut cached_cz_max: i32 = i32::MIN;
            let mut slot_a_cz: i32 = 0;
            let mut slot_a_ptr: ChunkSlot = None;
            let mut slot_b_cz: i32 = 0;
            let mut slot_b_ptr: ChunkSlot = None;
            let mut slot_b_valid: bool = false;

            let chunk_base_x = cx * cs;
            let chunk_base_y = cy * cs;
            // Clip the inner local-coord range to positions that could be
            // in the disk (conservative AABB vs circle test done inside).
            let lx_start = (cvx - sight - chunk_base_x).max(0).min(cs) as usize;
            let lx_end = (cvx + sight - chunk_base_x + 1).max(0).min(cs) as usize;
            let ly_start = (cvy - sight - chunk_base_y).max(0).min(cs) as usize;
            let ly_end = (cvy + sight - chunk_base_y + 1).max(0).min(cs) as usize;
            if lx_start >= lx_end || ly_start >= ly_end { continue; }

            for ly in ly_start..ly_end {
                let vy = chunk_base_y + ly as i32;
                let dy = vy - cvy;
                let dy2 = (dy as i64) * (dy as i64);

                for lx in lx_start..lx_end {
                    let vx = chunk_base_x + lx as i32;
                    let dx = vx - cvx;
                    let d2 = dy2 + (dx as i64) * (dx as i64);
                    if d2 > sight_sq { continue; }

                    // Surface lookup (persistent cache).
                    let key = pack_xy(vx, vy);
                    let surface = match surface_cache.get(&key) {
                        Some(&h) => h,
                        None => {
                            let h = state.voxel_world.surface_height(vx, vy);
                            surface_cache.insert(key, h);
                            h
                        }
                    };

                    let z_min = surface - 5;
                    let z_max = surface + 15;
                    // Rust's i32 `>>` is arithmetic shift (rounds toward
                    // negative infinity), which matches div_euclid semantics
                    // for power-of-2 divisors. CHUNK_SIZE is compile-time 64.
                    // LLVM doesn't consistently reduce div_euclid to this
                    // even at opt-level=3; the explicit shift measurably
                    // cuts the 32% div_euclid cost in the hot loop.
                    const CHUNK_SHIFT: i32 =
                        (crate::world_sim::voxel::CHUNK_SIZE as i32).trailing_zeros() as i32;
                    let cz_min = z_min >> CHUNK_SHIFT;
                    let cz_max = z_max >> CHUNK_SHIFT;

                    // Refresh Z-chunk cache if (cz_min, cz_max) changed.
                    // Scalar compare is cheaper than Vec.first()/last().
                    if cz_min != cached_cz_min || cz_max != cached_cz_max {
                        slot_a_cz = cz_min;
                        let cp = crate::world_sim::voxel::ChunkPos::new(cx, cy, cz_min);
                        slot_a_ptr = chunks.get(&cp).map(|c| c as *const _);
                        if cz_max > cz_min {
                            slot_b_cz = cz_max;
                            let cp = crate::world_sim::voxel::ChunkPos::new(cx, cy, cz_max);
                            slot_b_ptr = chunks.get(&cp).map(|c| c as *const _);
                            slot_b_valid = true;
                        } else {
                            slot_b_valid = false;
                        }
                        cached_cz_min = cz_min;
                        cached_cz_max = cz_max;
                    }

                    // Scan slot A (always present).
                    if let Some(ptr) = slot_a_ptr {
                        // SAFETY: `chunks` is borrowed immutably for the
                        // entire outer scope; no mutation happens between
                        // lookup and deref.
                        let chunk = unsafe { &*ptr };
                        let chunk_base_z = slot_a_cz * cs;
                        let lz_start = (z_min - chunk_base_z).max(0) as usize;
                        let lz_end = (z_max - chunk_base_z).min(cs - 1) as usize;
                        for lz in lz_start..=lz_end {
                            let voxel = chunk.voxels[
                                crate::world_sim::voxel::local_index(lx, ly, lz)
                            ];
                            if TARGET_MATERIALS.contains(&voxel.material) {
                                *counts.entry((cell_x, cell_y, voxel.material)).or_insert(0) += 1;
                            }
                        }
                    }

                    // Scan slot B (rare: scan spans a chunk boundary).
                    if slot_b_valid {
                        if let Some(ptr) = slot_b_ptr {
                            let chunk = unsafe { &*ptr };
                            let chunk_base_z = slot_b_cz * cs;
                            let lz_start = (z_min - chunk_base_z).max(0) as usize;
                            let lz_end = (z_max - chunk_base_z).min(cs - 1) as usize;
                            for lz in lz_start..=lz_end {
                                let voxel = chunk.voxels[
                                    crate::world_sim::voxel::local_index(lx, ly, lz)
                                ];
                                if TARGET_MATERIALS.contains(&voxel.material) {
                                    *counts.entry((cell_x, cell_y, voxel.material)).or_insert(0) += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let tick = state.tick;
    let half_cell_world = RESOURCE_CELL_SIZE as f32 * VOXEL_SCALE * 0.5;

    let npc = state.entities[entity_idx].npc.as_mut().unwrap();

    for ((cell_x, cell_y, material), count) in counts {
        let center_x = cell_x as f32 * RESOURCE_CELL_SIZE as f32 * VOXEL_SCALE + half_cell_world;
        let center_y = cell_y as f32 * RESOURCE_CELL_SIZE as f32 * VOXEL_SCALE + half_cell_world;

        // Update existing entry or insert new one.
        if let Some(existing) = npc
            .known_voxel_resources
            .iter_mut()
            .find(|k| k.material == material && k.center == (center_x, center_y))
        {
            existing.estimated_count = count;
            existing.tick_observed = tick;
        } else {
            npc.known_voxel_resources.push(VoxelResourceKnowledge {
                center: (center_x, center_y),
                material,
                estimated_count: count,
                tick_observed: tick,
            });
        }
    }
}

/// Eagerly populate `state.surface_cache` for voxel positions around each
/// settlement, so the first scan pass doesn't trigger 16K fbm calls per
/// new cell inside the tick loop. Moves fbm cost from "lazy in-tick" to
/// "eager at world load".
pub fn warm_surface_cache(state: &mut WorldState) {
    if state.voxel_world.region_plan.is_none() {
        return; // analytical surface_height only works with a region plan
    }

    let margin = SIGHT_RANGE_VOXELS + RESOURCE_CELL_SIZE;
    let mut surface_cache = std::mem::take(&mut state.surface_cache);
    if surface_cache.capacity() < 524288 {
        surface_cache.reserve(1 << 20);
    }

    for settlement in &state.settlements {
        let (cvx, cvy, _) = world_to_voxel(settlement.pos.0, settlement.pos.1, 0.0);
        for dy in -margin..=margin {
            for dx in -margin..=margin {
                let vx = cvx + dx;
                let vy = cvy + dy;
                let key = pack_xy(vx, vy);
                if surface_cache.contains_key(&key) { continue; }
                let h = state.voxel_world.surface_height(vx, vy);
                surface_cache.insert(key, h);
            }
        }
    }
    state.surface_cache = surface_cache;
}

/// Compute a per-cell census: count of each target material in the cell's
/// surface band (surface-5 .. surface+15). Called once per cell lazily —
/// the result persists on WorldState.cell_census across ticks.
fn compute_cell_census(
    voxel_world: &crate::world_sim::voxel::VoxelWorld,
    cell: (i32, i32),
    surface_cache: &mut SurfaceCache,
) -> [u32; NUM_TARGET_MATERIALS] {
    use crate::world_sim::voxel::{local_index, ChunkPos, CHUNK_SIZE};
    let mut census = [0u32; NUM_TARGET_MATERIALS];
    let cs = CHUNK_SIZE as i32;
    let base_vx = cell.0 * RESOURCE_CELL_SIZE;
    let base_vy = cell.1 * RESOURCE_CELL_SIZE;

    // Chunk-major iteration: the cell is 128×128 voxels, CHUNK_SIZE=64,
    // so the XY extent spans exactly 2 chunks × 2 chunks = 4 chunk columns.
    // Z range varies by (vx, vy) surface but is 1-2 chunks per column.
    // Looking up the chunk once per (cx, cy, cz) slice instead of per voxel
    // eliminates ~99% of chunks.get calls.
    let cx_min = base_vx.div_euclid(cs);
    let cx_max = (base_vx + RESOURCE_CELL_SIZE - 1).div_euclid(cs);
    let cy_min = base_vy.div_euclid(cs);
    let cy_max = (base_vy + RESOURCE_CELL_SIZE - 1).div_euclid(cs);
    const CHUNK_SHIFT: i32 = (crate::world_sim::voxel::CHUNK_SIZE as i32).trailing_zeros() as i32;

    for cx in cx_min..=cx_max {
        for cy in cy_min..=cy_max {
            let chunk_base_x = cx * cs;
            let chunk_base_y = cy * cs;
            // Overlap of chunk with this cell (clipped).
            let lx_start = (base_vx - chunk_base_x).max(0) as usize;
            let lx_end = ((base_vx + RESOURCE_CELL_SIZE - 1 - chunk_base_x).min(cs - 1)) as usize;
            let ly_start = (base_vy - chunk_base_y).max(0) as usize;
            let ly_end = ((base_vy + RESOURCE_CELL_SIZE - 1 - chunk_base_y).min(cs - 1)) as usize;

            // Cache z-chunks for this column (2 slots cover the common case).
            let mut cached_cz_min: i32 = i32::MIN;
            let mut cached_cz_max: i32 = i32::MIN;
            let mut slot_a_cz: i32 = 0;
            let mut slot_a: Option<&crate::world_sim::voxel::Chunk> = None;
            let mut slot_b_cz: i32 = 0;
            let mut slot_b: Option<&crate::world_sim::voxel::Chunk> = None;
            let mut slot_b_valid = false;

            for ly in ly_start..=ly_end {
                let vy = chunk_base_y + ly as i32;
                for lx in lx_start..=lx_end {
                    let vx = chunk_base_x + lx as i32;
                    let key = pack_xy(vx, vy);
                    let surface = match surface_cache.get(&key) {
                        Some(&h) => h,
                        None => {
                            let h = voxel_world.surface_height(vx, vy);
                            surface_cache.insert(key, h);
                            h
                        }
                    };
                    let z_min = surface - 5;
                    let z_max = surface + 15;
                    let cz_min = z_min >> CHUNK_SHIFT;
                    let cz_max = z_max >> CHUNK_SHIFT;
                    if cz_min != cached_cz_min || cz_max != cached_cz_max {
                        slot_a_cz = cz_min;
                        slot_a = voxel_world.chunks.get(&ChunkPos::new(cx, cy, cz_min));
                        if cz_max > cz_min {
                            slot_b_cz = cz_max;
                            slot_b = voxel_world.chunks.get(&ChunkPos::new(cx, cy, cz_max));
                            slot_b_valid = true;
                        } else {
                            slot_b_valid = false;
                        }
                        cached_cz_min = cz_min;
                        cached_cz_max = cz_max;
                    }
                    if let Some(chunk) = slot_a {
                        let chunk_base_z = slot_a_cz * cs;
                        let lz_start = (z_min - chunk_base_z).max(0) as usize;
                        let lz_end = (z_max - chunk_base_z).min(cs - 1) as usize;
                        for lz in lz_start..=lz_end {
                            let voxel = chunk.voxels[local_index(lx, ly, lz)];
                            if let Some(idx) = target_material_idx(voxel.material) {
                                census[idx] += 1;
                            }
                        }
                    }
                    if slot_b_valid {
                        if let Some(chunk) = slot_b {
                            let chunk_base_z = slot_b_cz * cs;
                            let lz_start = (z_min - chunk_base_z).max(0) as usize;
                            let lz_end = (z_max - chunk_base_z).min(cs - 1) as usize;
                            for lz in lz_start..=lz_end {
                                let voxel = chunk.voxels[local_index(lx, ly, lz)];
                                if let Some(idx) = target_material_idx(voxel.material) {
                                    census[idx] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    census
}

/// Run voxel resource scanning for all alive NPCs. Called from runtime.rs
/// every `RESOURCE_SCAN_INTERVAL` ticks.
///
/// Per-cell census architecture: instead of each NPC re-scanning 420K
/// voxels in its sight disk to produce ~40 cell counts, we compute the
/// census once per cell (across all NPCs, across ticks) and then each
/// NPC just reads cached counts for cells in its disk. Semantic change:
/// NPCs now discover resources at cell-level granularity rather than
/// partial-visibility — a cell is either fully known or unknown.
pub fn scan_all_npc_resources(state: &mut WorldState) {
    if state.tick % RESOURCE_SCAN_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.voxel_world.chunks.is_empty() {
        return;
    }

    // Pool npc_indices/npc_pos_voxel/visible_cells on state.sim_scratch.
    let mut npc_indices = std::mem::take(&mut state.sim_scratch.npc_indices);
    npc_indices.clear();
    for (i, e) in state.entities.iter().enumerate() {
        if e.alive && e.kind == EntityKind::Npc && e.npc.is_some() {
            npc_indices.push(i);
        }
    }
    if npc_indices.is_empty() {
        state.sim_scratch.npc_indices = npc_indices;
        return;
    }

    let sight = SIGHT_RANGE_VOXELS;
    let sight_sq = (sight as f32) * (sight as f32);

    // Step 1: collect the union of (cell_x, cell_y) cells visible across
    // all NPCs. Each NPC sees a 3×3 grid of cells at most.
    let mut visible_cells = std::mem::take(&mut state.sim_scratch.visible_cells);
    visible_cells.clear();
    let mut npc_pos_voxel = std::mem::take(&mut state.sim_scratch.npc_pos_voxel);
    npc_pos_voxel.clear();
    if npc_pos_voxel.capacity() < npc_indices.len() {
        npc_pos_voxel.reserve(npc_indices.len() - npc_pos_voxel.capacity());
    }

    for &idx in &npc_indices {
        let entity = &state.entities[idx];
        let (cvx, cvy, _) = world_to_voxel(entity.pos.0, entity.pos.1, 0.0);
        npc_pos_voxel.push((idx, cvx, cvy));

        let cell_vx_min = (cvx - sight).div_euclid(RESOURCE_CELL_SIZE);
        let cell_vx_max = (cvx + sight).div_euclid(RESOURCE_CELL_SIZE);
        let cell_vy_min = (cvy - sight).div_euclid(RESOURCE_CELL_SIZE);
        let cell_vy_max = (cvy + sight).div_euclid(RESOURCE_CELL_SIZE);

        for cy in cell_vy_min..=cell_vy_max {
            for cx in cell_vx_min..=cell_vx_max {
                visible_cells.insert((cx, cy));
            }
        }
    }

    // Step 2: populate census for any newly-visible cells.
    let mut surface_cache = std::mem::take(&mut state.surface_cache);
    let mut cell_census = std::mem::take(&mut state.cell_census);
    // Pre-size surface_cache to avoid the log(N) cascade of rehashes as
    // it grows across the run. Default-world NPC drift + census expansion
    // can exceed 1M positions; over-allocate to stay head of the load-
    // factor threshold. Costs ~30MB RAM up front — cheap.
    if surface_cache.capacity() < 524288 {
        surface_cache.reserve(1 << 20); // 1M entries
    }
    for &cell in &visible_cells {
        if cell_census.contains_key(&cell) {
            continue;
        }
        let census = compute_cell_census(&state.voxel_world, cell, &mut surface_cache);
        cell_census.insert(cell, census);
    }
    state.surface_cache = surface_cache;
    state.cell_census = cell_census;

    // Step 3: per-NPC, emit findings from the census for each visible cell.
    let tick = state.tick;
    let half_cell_world = RESOURCE_CELL_SIZE as f32 * VOXEL_SCALE * 0.5;

    for &(idx, cvx, cvy) in &npc_pos_voxel {
        let cell_vx_min = (cvx - sight).div_euclid(RESOURCE_CELL_SIZE);
        let cell_vx_max = (cvx + sight).div_euclid(RESOURCE_CELL_SIZE);
        let cell_vy_min = (cvy - sight).div_euclid(RESOURCE_CELL_SIZE);
        let cell_vy_max = (cvy + sight).div_euclid(RESOURCE_CELL_SIZE);

        let npc = match state.entities[idx].npc.as_mut() {
            Some(n) => n,
            None => continue,
        };

        for cy in cell_vy_min..=cell_vy_max {
            for cx in cell_vx_min..=cell_vx_max {
                // Skip cells whose nearest corner to the NPC is outside sight.
                let cell_vx_base = cx * RESOURCE_CELL_SIZE;
                let cell_vy_base = cy * RESOURCE_CELL_SIZE;
                let nearest_vx = cvx.clamp(cell_vx_base, cell_vx_base + RESOURCE_CELL_SIZE - 1);
                let nearest_vy = cvy.clamp(cell_vy_base, cell_vy_base + RESOURCE_CELL_SIZE - 1);
                let dx = (nearest_vx - cvx) as f32;
                let dy = (nearest_vy - cvy) as f32;
                if dx * dx + dy * dy > sight_sq {
                    continue;
                }

                let census = match state.cell_census.get(&(cx, cy)) {
                    Some(c) => c,
                    None => continue, // Should have been populated in step 2.
                };

                let center_x =
                    cx as f32 * RESOURCE_CELL_SIZE as f32 * VOXEL_SCALE + half_cell_world;
                let center_y =
                    cy as f32 * RESOURCE_CELL_SIZE as f32 * VOXEL_SCALE + half_cell_world;

                for (mi, &count) in census.iter().enumerate() {
                    if count == 0 {
                        continue;
                    }
                    let material = TARGET_MATERIALS[mi];
                    if let Some(existing) = npc
                        .known_voxel_resources
                        .iter_mut()
                        .find(|k| k.material == material && k.center == (center_x, center_y))
                    {
                        existing.estimated_count = count;
                        existing.tick_observed = tick;
                    } else {
                        npc.known_voxel_resources.push(VoxelResourceKnowledge {
                            center: (center_x, center_y),
                            material,
                            estimated_count: count,
                            tick_observed: tick,
                        });
                    }
                }
            }
        }
    }

    // Restore pooled buffers.
    state.sim_scratch.npc_indices = npc_indices;
    state.sim_scratch.npc_pos_voxel = npc_pos_voxel;
    state.sim_scratch.visible_cells = visible_cells;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::{Entity, WorldState};
    use crate::world_sim::voxel::{Voxel, VoxelMaterial};

    #[test]
    fn npc_discovers_wood_logs() {
        let mut state = WorldState::new(42);
        state.tick = 20;

        // Place some WoodLog voxels near origin.
        let log_positions = [(5, 5, 30), (5, 6, 31), (6, 5, 30), (7, 7, 32)];
        for (vx, vy, vz) in &log_positions {
            state
                .voxel_world
                .set_voxel(*vx, *vy, *vz, Voxel::new(VoxelMaterial::WoodLog));
        }

        // Place an NPC at world origin — voxel (0,0).
        // All log voxels are within ~10 voxels, well within sight range of 80.
        state.entities.push(Entity::new_npc(1, (0.0, 0.0)));

        scan_voxel_resources(&mut state, 0, SIGHT_RANGE_VOXELS);

        let known = &state.entities[0].npc.as_ref().unwrap().known_voxel_resources;
        assert!(!known.is_empty(), "NPC should have discovered WoodLog resources");

        let wood_entries: Vec<_> = known
            .iter()
            .filter(|k| k.material == VoxelMaterial::WoodLog)
            .collect();
        assert!(!wood_entries.is_empty(), "should have WoodLog entries");

        let total_count: u32 = wood_entries.iter().map(|e| e.estimated_count).sum();
        assert_eq!(
            total_count,
            log_positions.len() as u32,
            "should count all placed WoodLog voxels"
        );
        assert_eq!(wood_entries[0].tick_observed, 20);
    }

    #[test]
    fn scan_updates_existing_knowledge() {
        let mut state = WorldState::new(42);
        state.tick = 20;

        // Place some iron ore.
        state
            .voxel_world
            .set_voxel(3, 3, 10, Voxel::new(VoxelMaterial::IronOre));

        state.entities.push(Entity::new_npc(1, (0.0, 0.0)));

        scan_voxel_resources(&mut state, 0, SIGHT_RANGE_VOXELS);

        let known = &state.entities[0].npc.as_ref().unwrap().known_voxel_resources;
        let iron_count = known
            .iter()
            .filter(|k| k.material == VoxelMaterial::IronOre)
            .map(|k| k.estimated_count)
            .sum::<u32>();
        assert_eq!(iron_count, 1);

        // Add another ore voxel and rescan at a later tick.
        state
            .voxel_world
            .set_voxel(3, 4, 10, Voxel::new(VoxelMaterial::IronOre));
        state.tick = 40;

        scan_voxel_resources(&mut state, 0, SIGHT_RANGE_VOXELS);

        let known = &state.entities[0].npc.as_ref().unwrap().known_voxel_resources;
        let iron: Vec<_> = known
            .iter()
            .filter(|k| k.material == VoxelMaterial::IronOre)
            .collect();
        let total: u32 = iron.iter().map(|k| k.estimated_count).sum();
        assert_eq!(total, 2, "count should be updated to 2");
        // tick_observed should be updated
        assert!(iron.iter().all(|k| k.tick_observed == 40));
    }

    #[test]
    fn no_scan_for_dead_npc() {
        let mut state = WorldState::new(42);
        state.tick = 20;
        state
            .voxel_world
            .set_voxel(3, 3, 10, Voxel::new(VoxelMaterial::Coal));

        let mut npc = Entity::new_npc(1, (0.0, 0.0));
        npc.alive = false;
        state.entities.push(npc);

        scan_voxel_resources(&mut state, 0, SIGHT_RANGE_VOXELS);

        let known = &state.entities[0].npc.as_ref().unwrap().known_voxel_resources;
        assert!(known.is_empty(), "dead NPC should not discover resources");
    }
}
