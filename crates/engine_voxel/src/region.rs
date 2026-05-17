//! Voxel-region runtime — per spec
//! `docs/superpowers/specs/2026-04-25-voxel-region-indices-design.md`
//! §6.1. Game-owned named volumes of voxel space with stable
//! generational ids and a per-runtime registry that the rest of
//! the engine consults via `covering_regions(world_pos)`.
//!
//! **Phase 3 scope** (this slice): runtime types + registry with
//! register / destroy / get / covering walks. Subset of bounds
//! variants (Aabb + ChunkSet). No event-driven lifecycle, no
//! static pool sizing — both come when Phase 4 wires the build
//! kernel and a real fixture consumes the registry.
//!
//! ## Design decisions
//!
//! - **Generational ids**. `VoxelRegionId(u64)` packs `gen<<32 |
//!   slot`. Mirrors the engine's `Pool<T>` precedent (spec §3) —
//!   reusing a freed slot bumps the generation so stale refs
//!   surface at lookup as `None` rather than aliasing to the new
//!   region. This is the only way to keep cross-tick refs (e.g.
//!   a guard remembering "the region the assassin entered last
//!   tick") sound under churn.
//!
//! - **Bounds subset**. `Aabb` + `ChunkSet` only. `Mask { aabb,
//!   mask: Arc<BitGrid> }` and `Sphere { center, radius }` are
//!   spec'd but skipped — no fixture in the project today needs
//!   them; adding them is purely additive when one does.
//!
//! - **No static pool sizing**. The spec defines the registry's
//!   max-active per kind as a compile-time bound enforced by the
//!   DSL compiler (sum of `region_kind.max_active ×
//!   index.per_region_storage`). This Phase 3 ships a runtime Vec
//!   with checked-grow + per-kind counters; Phase 4 will surface
//!   the build.rs-emitted bounds + enforce at register-time.
//!
//! - **No event-driven lifecycle**. The spec routes
//!   `VoxelRegionRegistered` / `VoxelRegionDestroyed` through the
//!   chronicle. Phase 3 ships direct `Registry::register` /
//!   `Registry::destroy` methods — the event-driven wrapper is a
//!   Phase 4 concern that lands when a fixture's physics rule
//!   actually emits these events.

use std::collections::HashMap;

/// Generational handle for a [`VoxelRegion`]. Packs `generation`
/// in the high 32 bits and `slot` in the low 32. Stale refs (after
/// a slot is freed + reused) compare unequal and lookup returns
/// `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelRegionId(u64);

impl VoxelRegionId {
    pub fn slot(self) -> u32 {
        (self.0 & 0xFFFF_FFFF) as u32
    }
    pub fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }
    fn pack(gen: u32, slot: u32) -> Self {
        Self(((gen as u64) << 32) | (slot as u64))
    }

    /// Test-only constructor. Real ids should always come from
    /// [`VoxelRegionRegistry::register`] so the generation +
    /// slot are sound; use this for synthetic test fixtures that
    /// construct a [`VoxelRegion`] directly without a registry.
    #[doc(hidden)]
    pub fn from_raw_for_test() -> Self {
        Self(0)
    }
}

/// Opaque region-kind tag. The DSL compiler assigns one of these
/// per declared `region_kind`; the engine treats it as a u32 key
/// for looking up "which indices does this kind get?" The
/// per-fixture mapping is provided to the registry at construction
/// time (Phase 4 will wire the build.rs side to emit it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelRegionKind(pub u32);

/// 3D axis-aligned bounding box in world space. Inclusive min,
/// exclusive max — same convention as `voxel_engine::Aabb` would
/// if it existed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub fn contains(&self, p: [f32; 3]) -> bool {
        p[0] >= self.min[0]
            && p[0] < self.max[0]
            && p[1] >= self.min[1]
            && p[1] < self.max[1]
            && p[2] >= self.min[2]
            && p[2] < self.max[2]
    }
}

/// Bounds describing the voxel coverage of a region.
#[derive(Debug, Clone, PartialEq)]
pub enum VoxelRegionBounds {
    /// AABB in world space. The registry's `covering_regions`
    /// walk is O(N) over registered regions; with bounds-based
    /// pruning it's worst-case N per query — acceptable for the
    /// hundreds-of-active-regions target.
    Aabb(Aabb),
    /// Explicit set of chunk ids. Used when a region's coverage
    /// doesn't form a clean AABB (e.g. an L-shaped settlement
    /// spanning 7 chunks at the corner of two rivers).
    ChunkSet(Vec<u32>),
}

impl VoxelRegionBounds {
    pub fn contains_world_pos(&self, p: [f32; 3]) -> bool {
        match self {
            VoxelRegionBounds::Aabb(a) => a.contains(p),
            VoxelRegionBounds::ChunkSet(_) => {
                // Chunk-set containment needs a chunk-coord
                // lookup. Phase 4 wires the
                // world-to-chunk transform; until then return
                // false so the covering walk skips chunk-set
                // regions for world-pos queries (callers should
                // walk explicit chunk ids instead).
                false
            }
        }
    }
}

/// Per spec §6.1 — a logical named volume of voxel space.
#[derive(Debug, Clone)]
pub struct VoxelRegion {
    pub id: VoxelRegionId,
    pub bounds: VoxelRegionBounds,
    pub kind: VoxelRegionKind,
    pub created_at_tick: u64,
}

/// Slot in the registry's storage. `Some` for live regions;
/// `None` for freed slots (re-allocatable via the freelist with
/// a bumped generation).
#[derive(Debug, Clone)]
struct Slot {
    region: Option<VoxelRegion>,
    generation: u32,
}

/// Per-runtime registry of [`VoxelRegion`] instances. Mirrors the
/// `Pool<T>` shape (spec §3): slot indices reused via freelist,
/// generations invalidate stale refs.
#[derive(Debug, Clone, Default)]
pub struct VoxelRegionRegistry {
    slots: Vec<Slot>,
    freelist: Vec<u32>,
    /// Per-kind active count — for diagnostics + the future
    /// max-active enforcement Phase 4 ships.
    kind_counts: HashMap<VoxelRegionKind, u32>,
}

impl VoxelRegionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new region. Returns its fresh generational id.
    /// Per spec §6.2, regions are game-owned — the registry doesn't
    /// infer regions, only registers what the caller declares.
    ///
    /// Phase 3 has no per-kind max-active enforcement — Phase 4
    /// adds it when the build.rs-emitted bounds are threaded in.
    pub fn register(
        &mut self,
        kind: VoxelRegionKind,
        bounds: VoxelRegionBounds,
        created_at_tick: u64,
    ) -> VoxelRegionId {
        let (slot_idx, generation) = if let Some(reused) = self.freelist.pop() {
            // Re-use a freed slot. Generation was bumped on free.
            let gen = self.slots[reused as usize].generation;
            (reused, gen)
        } else {
            // Append a fresh slot. Generation starts at 0.
            let idx = self.slots.len() as u32;
            self.slots.push(Slot {
                region: None,
                generation: 0,
            });
            (idx, 0)
        };
        let id = VoxelRegionId::pack(generation, slot_idx);
        let region = VoxelRegion {
            id,
            bounds,
            kind,
            created_at_tick,
        };
        self.slots[slot_idx as usize].region = Some(region);
        *self.kind_counts.entry(kind).or_insert(0) += 1;
        id
    }

    /// Destroy a region. Returns `true` if a live region was
    /// destroyed, `false` if `id` was stale (already freed) or
    /// out-of-bounds.
    pub fn destroy(&mut self, id: VoxelRegionId) -> bool {
        let slot_idx = id.slot();
        let slot = match self.slots.get_mut(slot_idx as usize) {
            Some(s) => s,
            None => return false,
        };
        if slot.generation != id.generation() {
            return false; // stale ref
        }
        let region = match slot.region.take() {
            Some(r) => r,
            None => return false, // already freed
        };
        // Bump generation so future refs to this id surface as None.
        // Wrapping is fine — a 32-bit gen + same slot collision
        // requires 2³² registrations on the same slot, far beyond
        // any realistic sim run.
        slot.generation = slot.generation.wrapping_add(1);
        self.freelist.push(slot_idx);
        if let Some(count) = self.kind_counts.get_mut(&region.kind) {
            *count = count.saturating_sub(1);
        }
        true
    }

    /// Get a region by id. Returns `None` for stale or freed ids.
    pub fn get(&self, id: VoxelRegionId) -> Option<&VoxelRegion> {
        let slot = self.slots.get(id.slot() as usize)?;
        if slot.generation != id.generation() {
            return None;
        }
        slot.region.as_ref()
    }

    /// Iterate every live region. Order is slot-index; freed slots
    /// are skipped. Generational id is on each yielded region.
    pub fn iter(&self) -> impl Iterator<Item = &VoxelRegion> {
        self.slots.iter().filter_map(|s| s.region.as_ref())
    }

    /// Per spec §6.1 — list every region that covers `world_pos`,
    /// ranked (today: insertion order; spec leaves "ranked" open
    /// for a future cut where inner-most-first ordering matters).
    pub fn covering_regions(&self, world_pos: [f32; 3]) -> Vec<VoxelRegionId> {
        self.slots
            .iter()
            .filter_map(|s| s.region.as_ref())
            .filter(|r| r.bounds.contains_world_pos(world_pos))
            .map(|r| r.id)
            .collect()
    }

    /// Active count of regions of the given kind.
    pub fn count_of_kind(&self, kind: VoxelRegionKind) -> u32 {
        self.kind_counts.get(&kind).copied().unwrap_or(0)
    }

    /// Total live region count.
    pub fn len(&self) -> usize {
        self.slots.len() - self.freelist.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aabb(min: [f32; 3], max: [f32; 3]) -> VoxelRegionBounds {
        VoxelRegionBounds::Aabb(Aabb { min, max })
    }

    #[test]
    fn register_and_get_round_trip() {
        let mut reg = VoxelRegionRegistry::new();
        let id = reg.register(
            VoxelRegionKind(0),
            aabb([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
            0,
        );
        assert_eq!(reg.len(), 1);
        let region = reg.get(id).expect("live");
        assert_eq!(region.id, id);
        assert_eq!(region.kind, VoxelRegionKind(0));
    }

    #[test]
    fn destroy_invalidates_id() {
        let mut reg = VoxelRegionRegistry::new();
        let id = reg.register(VoxelRegionKind(1), aabb([0.0; 3], [1.0; 3]), 0);
        assert!(reg.destroy(id));
        assert!(reg.get(id).is_none());
        // Double-destroy is a no-op (returns false).
        assert!(!reg.destroy(id));
    }

    #[test]
    fn stale_id_after_slot_reuse_returns_none() {
        // The core soundness guarantee of generational ids.
        let mut reg = VoxelRegionRegistry::new();
        let id_a = reg.register(VoxelRegionKind(0), aabb([0.0; 3], [1.0; 3]), 0);
        assert!(reg.destroy(id_a));
        // Re-register reuses the same slot but bumps generation.
        let id_b = reg.register(VoxelRegionKind(0), aabb([0.0; 3], [1.0; 3]), 1);
        assert_eq!(id_a.slot(), id_b.slot());
        assert_ne!(id_a.generation(), id_b.generation());
        // Stale id_a returns None even though slot is now live.
        assert!(reg.get(id_a).is_none());
        // Fresh id_b resolves.
        assert!(reg.get(id_b).is_some());
    }

    #[test]
    fn covering_regions_returns_all_aabb_coverers() {
        let mut reg = VoxelRegionRegistry::new();
        let big = reg.register(
            VoxelRegionKind(0),
            aabb([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
            0,
        );
        let small = reg.register(
            VoxelRegionKind(0),
            aabb([10.0, 10.0, 10.0], [20.0, 20.0, 20.0]),
            0,
        );
        let elsewhere = reg.register(
            VoxelRegionKind(0),
            aabb([200.0, 200.0, 200.0], [210.0, 210.0, 210.0]),
            0,
        );
        let covering = reg.covering_regions([15.0, 15.0, 15.0]);
        assert!(covering.contains(&big));
        assert!(covering.contains(&small));
        assert!(!covering.contains(&elsewhere));
    }

    #[test]
    fn count_of_kind_tracks_register_destroy() {
        let mut reg = VoxelRegionRegistry::new();
        let settlement = VoxelRegionKind(1);
        let building = VoxelRegionKind(2);
        let s1 = reg.register(settlement, aabb([0.0; 3], [1.0; 3]), 0);
        let _s2 = reg.register(settlement, aabb([0.0; 3], [1.0; 3]), 0);
        let _b1 = reg.register(building, aabb([0.0; 3], [1.0; 3]), 0);
        assert_eq!(reg.count_of_kind(settlement), 2);
        assert_eq!(reg.count_of_kind(building), 1);
        reg.destroy(s1);
        assert_eq!(reg.count_of_kind(settlement), 1);
    }

    #[test]
    fn freelist_recycles_slots_in_lifo_order() {
        let mut reg = VoxelRegionRegistry::new();
        let a = reg.register(VoxelRegionKind(0), aabb([0.0; 3], [1.0; 3]), 0);
        let b = reg.register(VoxelRegionKind(0), aabb([0.0; 3], [1.0; 3]), 0);
        let c = reg.register(VoxelRegionKind(0), aabb([0.0; 3], [1.0; 3]), 0);
        // Destroy middle, then end. Next register reuses end's slot.
        reg.destroy(b);
        reg.destroy(c);
        let d = reg.register(VoxelRegionKind(0), aabb([0.0; 3], [1.0; 3]), 0);
        assert_eq!(d.slot(), c.slot()); // LIFO freelist
        let e = reg.register(VoxelRegionKind(0), aabb([0.0; 3], [1.0; 3]), 0);
        assert_eq!(e.slot(), b.slot());
        // Original `a` still live.
        assert!(reg.get(a).is_some());
    }
}
