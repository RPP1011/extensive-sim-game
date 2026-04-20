use crate::ids::AgentId;
use crate::state::{MovementMode, SimState};
use glam::Vec3;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

pub const CELL_SIZE: f32 = 16.0;

/// Per-cell agent bucket. Inline `[AgentId; 4]` covers the common case where a
/// 16×16 m cell holds at most a handful of walkers; the spill onto the heap
/// only kicks in for crowded cells. AgentIds inside a bucket are stored in
/// insertion order, with `swap_remove` on takedown — within-cell order can
/// jumble, but query results are always sorted by `AgentId::raw()` before
/// returning so callers see the same `Vec<AgentId>` regardless of bucket
/// state.
type CellBucket = SmallVec<[AgentId; 4]>;

/// Incremental 2D-column spatial hash for ground-moving agents, with a
/// movement-mode sidecar for non-walk agents (flyers, swimmers, climbers,
/// fallers). Replaces the prior `BTreeMap<(i32,i32), Vec<(f32, AgentId)>>` +
/// per-cell z-sort.
///
/// **Per-agent cell tracking** lets `update` early-out when the agent is still
/// in the same cell *and* hasn't crossed the walk↔non-walk boundary —
/// sub-cell movement (the overwhelmingly common case under typical agent
/// speeds vs. the 16 m cell size) costs an array write and an `Eq` compare.
/// Cell-crossing moves cost one `swap_remove` from the old bucket and one
/// `push` to the new bucket. Every mutation is `O(1)` amortised — the
/// `O(N)` rebuild in the old `build_from_slices` path is gone.
///
/// **Determinism contract.** `FxHashMap` (rustc-hash) is fixed-seeded by
/// construction — same `(cx, cy)` always lands in the same bucket across
/// runs. Within a bucket, `swap_remove` jumbles insertion order, but
/// `within_radius` / `within_planar` sort their result by `AgentId::raw()`
/// before returning, so callers observe a bit-identical `Vec<AgentId>`
/// regardless of bucket order. (We deliberately do NOT use AHasher here —
/// even with `BuildHasherDefault<AHasher>`, ahash's `runtime-rng` feature
/// would seed the hasher from the OS RNG, breaking cross-run determinism.)
pub struct SpatialHash {
    /// (cx, cy) → walk-mode agents currently in that cell.
    cells: FxHashMap<(i32, i32), CellBucket>,
    /// `slot → Some(cell)` for walkers, `slot → None` for sidecar agents and
    /// dead/never-spawned slots. Indexed by `(AgentId.raw() - 1) as usize`.
    agent_cell: Vec<Option<(i32, i32)>>,
    /// Non-walk agents (any `MovementMode != Walk`). Scanned linearly; the
    /// per-tick population of flyers/swimmers/climbers/fallers is small in
    /// practice, and the linear scan keeps the data-structure trivial.
    sidecar: Vec<AgentId>,
}

#[inline]
fn cell(x: f32, y: f32) -> (i32, i32) {
    ((x / CELL_SIZE) as i32, (y / CELL_SIZE) as i32)
}

impl SpatialHash {
    /// Construct an empty index sized for `cap` agent slots. Mutator calls
    /// (`insert`, `remove`, `update`) are O(1) amortised — there is no
    /// `build_from_slices` style bulk rebuild.
    pub fn new(cap: u32) -> Self {
        Self {
            cells: FxHashMap::default(),
            agent_cell: vec![None; cap as usize],
            sidecar: Vec::new(),
        }
    }

    /// Insert an agent. Caller is `SimState::spawn_agent` after the SoA fields
    /// for `id` are written. Idempotency is the caller's contract: spawning
    /// the same id twice without an intervening `remove` is a logic bug.
    pub fn insert(&mut self, id: AgentId, pos: Vec3, mode: MovementMode) {
        let slot = (id.raw() - 1) as usize;
        if slot >= self.agent_cell.len() {
            // Slot beyond cap. Mirrors the SoA's bounds-tolerant accessors.
            return;
        }
        if mode == MovementMode::Walk {
            let key = cell(pos.x, pos.y);
            self.cells.entry(key).or_default().push(id);
            self.agent_cell[slot] = Some(key);
        } else {
            self.sidecar.push(id);
            self.agent_cell[slot] = None;
        }
    }

    /// Remove an agent from whichever bucket / sidecar it currently lives in.
    /// No-op when the slot wasn't tracked. Caller is `SimState::kill_agent`.
    pub fn remove(&mut self, id: AgentId) {
        let slot = (id.raw() - 1) as usize;
        if slot >= self.agent_cell.len() {
            return;
        }
        match self.agent_cell[slot] {
            Some(key) => {
                if let Some(bucket) = self.cells.get_mut(&key) {
                    if let Some(pos_in_bucket) = bucket.iter().position(|&x| x == id) {
                        bucket.swap_remove(pos_in_bucket);
                        if bucket.is_empty() {
                            self.cells.remove(&key);
                        }
                    }
                }
                self.agent_cell[slot] = None;
            }
            None => {
                if let Some(pos_in_side) = self.sidecar.iter().position(|&x| x == id) {
                    self.sidecar.swap_remove(pos_in_side);
                }
            }
        }
    }

    /// Update an agent's recorded position / movement mode. Common case
    /// (walk-mode agent moves within the same cell) is a no-op aside from
    /// the cell-key compare. Cell-crossing moves are one `swap_remove` +
    /// one `push`. Mode transitions across the walk↔non-walk boundary
    /// move the agent from columns to sidecar (or vice versa).
    pub fn update(&mut self, id: AgentId, new_pos: Vec3, new_mode: MovementMode) {
        let slot = (id.raw() - 1) as usize;
        if slot >= self.agent_cell.len() {
            return;
        }
        let was_walk = self.agent_cell[slot].is_some();
        let is_walk = new_mode == MovementMode::Walk;

        match (was_walk, is_walk) {
            (true, true) => {
                let new_key = cell(new_pos.x, new_pos.y);
                let old_key = self.agent_cell[slot].expect("was_walk implies Some");
                if old_key == new_key {
                    // Sub-cell move — fast path. Nothing changes.
                    return;
                }
                // Cell crossing. Move from old bucket to new bucket.
                if let Some(bucket) = self.cells.get_mut(&old_key) {
                    if let Some(pos_in_bucket) = bucket.iter().position(|&x| x == id) {
                        bucket.swap_remove(pos_in_bucket);
                        if bucket.is_empty() {
                            self.cells.remove(&old_key);
                        }
                    }
                }
                self.cells.entry(new_key).or_default().push(id);
                self.agent_cell[slot] = Some(new_key);
            }
            (true, false) => {
                // Walk → non-walk. Pull from columns, push to sidecar.
                let old_key = self.agent_cell[slot].expect("was_walk implies Some");
                if let Some(bucket) = self.cells.get_mut(&old_key) {
                    if let Some(pos_in_bucket) = bucket.iter().position(|&x| x == id) {
                        bucket.swap_remove(pos_in_bucket);
                        if bucket.is_empty() {
                            self.cells.remove(&old_key);
                        }
                    }
                }
                self.agent_cell[slot] = None;
                self.sidecar.push(id);
            }
            (false, true) => {
                // Non-walk → walk. Pull from sidecar, push to columns.
                if let Some(pos_in_side) = self.sidecar.iter().position(|&x| x == id) {
                    self.sidecar.swap_remove(pos_in_side);
                }
                let new_key = cell(new_pos.x, new_pos.y);
                self.cells.entry(new_key).or_default().push(id);
                self.agent_cell[slot] = Some(new_key);
            }
            (false, false) => {
                // Stay in sidecar — no positional bookkeeping needed.
            }
        }
    }

    /// 3-D Euclidean radius query. Walk agents are gathered from cells in the
    /// `cell_reach_for_radius`-sized neighbourhood around `center`; non-walk
    /// agents from the linear sidecar scan. Distance is checked against the
    /// agent's *current* `state.agent_pos` (the index records cells, not raw
    /// positions). Returned `Vec<AgentId>` is sorted by `AgentId::raw()`
    /// ascending so callers observe a bit-identical result regardless of
    /// bucket-internal ordering.
    pub fn within_radius(&self, state: &SimState, center: Vec3, radius: f32) -> Vec<AgentId> {
        let r2 = radius * radius;
        let (cx, cy) = cell(center.x, center.y);
        let cell_reach = cell_reach_for_radius(radius);

        let mut hits: Vec<AgentId> = Vec::new();

        // Cell-rect scan. Nested loops emit no per-cell-key Vec.
        for dx in -cell_reach..=cell_reach {
            for dy in -cell_reach..=cell_reach {
                let key = (cx + dx, cy + dy);
                if let Some(bucket) = self.cells.get(&key) {
                    for &id in bucket.iter() {
                        if let Some(p) = state.agent_pos(id) {
                            if (p - center).length_squared() <= r2 {
                                hits.push(id);
                            }
                        }
                    }
                }
            }
        }

        for &id in &self.sidecar {
            if let Some(p) = state.agent_pos(id) {
                if (p - center).length_squared() <= r2 {
                    hits.push(id);
                }
            }
        }

        hits.sort_unstable_by_key(|id| id.raw());
        hits
    }

    /// XY-only radius query. Z is ignored for both walkers and sidecar agents
    /// — useful for area-of-effect logic that should apply across elevations
    /// within an XY footprint.
    pub fn within_planar(&self, state: &SimState, center: Vec3, radius: f32) -> Vec<AgentId> {
        let r2 = radius * radius;
        let (cx, cy) = cell(center.x, center.y);
        let cell_reach = cell_reach_for_radius(radius);

        let mut hits: Vec<AgentId> = Vec::new();

        for dx in -cell_reach..=cell_reach {
            for dy in -cell_reach..=cell_reach {
                let key = (cx + dx, cy + dy);
                if let Some(bucket) = self.cells.get(&key) {
                    for &id in bucket.iter() {
                        if let Some(p) = state.agent_pos(id) {
                            let ex = p.x - center.x;
                            let ey = p.y - center.y;
                            if ex * ex + ey * ey <= r2 {
                                hits.push(id);
                            }
                        }
                    }
                }
            }
        }

        for &id in &self.sidecar {
            if let Some(p) = state.agent_pos(id) {
                let ex = p.x - center.x;
                let ey = p.y - center.y;
                if ex * ex + ey * ey <= r2 {
                    hits.push(id);
                }
            }
        }

        hits.sort_unstable_by_key(|id| id.raw());
        hits
    }

    /// Test-only accessor — the cell key currently assigned to an agent
    /// slot, or `None` for sidecar / dead / never-spawned slots.
    #[doc(hidden)]
    pub fn cell_of_agent(&self, id: AgentId) -> Option<(i32, i32)> {
        let slot = (id.raw() - 1) as usize;
        self.agent_cell.get(slot).copied().flatten()
    }

    /// Test-only accessor — the count of distinct populated cells.
    #[doc(hidden)]
    pub fn populated_cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Test-only accessor — the current sidecar membership.
    #[doc(hidden)]
    pub fn sidecar_ids(&self) -> &[AgentId] {
        &self.sidecar
    }
}

/// Nearest-hostile spatial query — used by the DSL `engagement_on_move`
/// physics rule (task 163). Returns the `AgentId` of the closest hostile
/// within `radius` of `mover`, or `None` if nothing matches. Hostility is
/// species-level (`CreatureType::is_hostile_to`); a dead / unspawned /
/// out-of-range `mover` returns `None`.
///
/// Ties on distance are broken on raw `AgentId` ascending — matches the
/// legacy hand-written `engagement::recompute_engagement_for` discipline
/// (Task 163 moves that function to a DSL rule; the wolves+humans
/// baseline pins the exact tie-break outcome so this helper must preserve
/// it bit-for-bit). The iteration order over `SpatialHash::within_radius`
/// is the same — that helper already sorts by raw id before returning, so
/// "first candidate with equal distance wins" collapses to "lowest raw
/// id wins".
pub fn nearest_hostile_to(state: &SimState, mover: AgentId, radius: f32) -> Option<AgentId> {
    if !state.agent_alive(mover) {
        return None;
    }
    let pos = state.agent_pos(mover)?;
    let ct = state.agent_creature_type(mover)?;
    let spatial = state.spatial();
    let mut best: Option<(AgentId, f32)> = None;
    for other in spatial.within_radius(state, pos, radius) {
        if other == mover {
            continue;
        }
        let op = match state.agent_pos(other) {
            Some(p) => p,
            None => continue,
        };
        let oc = match state.agent_creature_type(other) {
            Some(c) => c,
            None => continue,
        };
        if !ct.is_hostile_to(oc) {
            continue;
        }
        let d = pos.distance(op);
        match best {
            None => best = Some((other, d)),
            Some((_, bd)) if d < bd => best = Some((other, d)),
            Some((b, bd)) if (d - bd).abs() < f32::EPSILON && other.raw() < b.raw() => {
                best = Some((other, d));
            }
            _ => {}
        }
    }
    best.map(|(a, _)| a)
}

/// Number of cell steps in each cardinal direction the query must scan
/// to guarantee no in-range walker is missed. For `radius <= CELL_SIZE`
/// this is 1 (the original 3×3 neighbourhood); grows linearly with
/// `radius`. Capped at 256 cells to prevent runaway scans on pathological
/// radii (256 × 16 m = 4096 m spans 65536 cells).
fn cell_reach_for_radius(radius: f32) -> i32 {
    if radius <= 0.0 { return 0; }
    let cells = (radius / CELL_SIZE).ceil() as i32;
    cells.clamp(1, 256)
}
