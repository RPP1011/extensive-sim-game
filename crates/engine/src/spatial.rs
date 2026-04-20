use crate::ids::AgentId;
use crate::state::{MovementMode, SimState};
use glam::Vec3;
use std::collections::BTreeMap;

pub const CELL_SIZE: f32 = 16.0;

/// 2D-column spatial index for ground-moving agents, with a movement-mode sidecar
/// for agents that don't walk (flyers, swimmers, etc.).
///
/// Columns are keyed by `(cx, cy)` grid cell and contain `(z, AgentId)` pairs sorted
/// by z ascending. Non-walk agents bypass the grid and are scanned linearly via the
/// sidecar, keeping the hot path fast while still covering all agents.
///
/// Uses `BTreeMap` (not `AHashMap`) for deterministic iteration order across runs,
/// which is required by Task 12's cross-run determinism contract.
pub struct SpatialIndex {
    /// (cx, cy) → entries sorted by z ascending as `(z, AgentId)`.
    pub columns: BTreeMap<(i32, i32), Vec<(f32, AgentId)>>,
    /// Agents whose movement mode is not `Walk`; scanned linearly.
    pub sidecar: Vec<AgentId>,
}

#[inline]
fn cell(x: f32, y: f32) -> (i32, i32) {
    ((x / CELL_SIZE) as i32, (y / CELL_SIZE) as i32)
}

impl SpatialIndex {
    /// Construct an empty index with no columns and no sidecar entries.
    /// Used as the `SimState` initial value; subsequent mutations (spawn,
    /// kill, set_pos, set_movement_mode) rebuild the index eagerly.
    pub fn empty() -> Self {
        Self { columns: BTreeMap::new(), sidecar: Vec::new() }
    }

    /// Build the index from the current alive population in `state`.
    pub fn build(state: &SimState) -> Self {
        let mut columns: BTreeMap<(i32, i32), Vec<(f32, AgentId)>> = BTreeMap::new();
        let mut sidecar: Vec<AgentId> = Vec::new();

        for id in state.agents_alive() {
            let pos = match state.agent_pos(id) {
                Some(p) => p,
                None => continue,
            };
            let mode = match state.agent_movement_mode(id) {
                Some(m) => m,
                None => continue,
            };
            if mode == MovementMode::Walk {
                let key = cell(pos.x, pos.y);
                columns.entry(key).or_default().push((pos.z, id));
            } else {
                sidecar.push(id);
            }
        }

        for col in columns.values_mut() {
            col.sort_by(|a, b| a.0.total_cmp(&b.0));
        }

        Self { columns, sidecar }
    }

    /// Build the index from the raw SoA slices. Slot `i` corresponds to
    /// `AgentId::new(i + 1)`. Used by `SimState` to avoid borrowing the
    /// whole `&SimState` from within `&mut self` mutators.
    pub fn build_from_slices(
        hot_alive:         &[bool],
        hot_pos:           &[Vec3],
        hot_movement_mode: &[MovementMode],
    ) -> Self {
        let mut columns: BTreeMap<(i32, i32), Vec<(f32, AgentId)>> = BTreeMap::new();
        let mut sidecar: Vec<AgentId> = Vec::new();

        let n = hot_alive.len().min(hot_pos.len()).min(hot_movement_mode.len());
        for slot in 0..n {
            if !hot_alive[slot] { continue; }
            let id = match AgentId::new((slot + 1) as u32) {
                Some(a) => a,
                None    => continue,
            };
            let pos = hot_pos[slot];
            let mode = hot_movement_mode[slot];
            if mode == MovementMode::Walk {
                let key = cell(pos.x, pos.y);
                columns.entry(key).or_default().push((pos.z, id));
            } else {
                sidecar.push(id);
            }
        }

        for col in columns.values_mut() {
            col.sort_by(|a, b| a.0.total_cmp(&b.0));
        }

        Self { columns, sidecar }
    }

    /// Return all agents within `radius` of `center` in 3-D Euclidean distance.
    ///
    /// Walk agents are queried via the column grid — the cell neighbourhood
    /// is sized by `radius` so large-radius queries (up to
    /// `MAX_ANNOUNCE_RADIUS = 80m`) visit every cell that could contain a
    /// hit. Non-walk agents are scanned from the sidecar using the same 3D
    /// distance.
    pub fn query_within_radius<'a>(
        &'a self,
        state: &'a SimState,
        center: Vec3,
        radius: f32,
    ) -> impl Iterator<Item = AgentId> + 'a {
        let r2 = radius * radius;
        let (cx, cy) = cell(center.x, center.y);
        let cell_reach = cell_reach_for_radius(radius);

        let cells: Vec<(i32, i32)> = (-cell_reach..=cell_reach)
            .flat_map(move |dx| (-cell_reach..=cell_reach).map(move |dy| (cx + dx, cy + dy)))
            .collect();

        let column_hits = cells
            .into_iter()
            .flat_map(move |key| {
                self.columns
                    .get(&key)
                    .into_iter()
                    .flat_map(|col| col.iter().copied())
            })
            .filter_map(move |(_z, id)| {
                state
                    .agent_pos(id)
                    .filter(|p| (*p - center).length_squared() <= r2)
                    .map(|_| id)
            });

        let sidecar_hits = self.sidecar.iter().copied().filter_map(move |id| {
            state
                .agent_pos(id)
                .filter(|p| (*p - center).length_squared() <= r2)
                .map(|_| id)
        });

        column_hits.chain(sidecar_hits)
    }

    /// Return all agents within `radius` of `center` using only XY (planar) distance.
    ///
    /// Z is ignored for both column walkers and sidecar agents. Useful for
    /// area-of-effect logic that should affect all elevation layers within an XY footprint.
    pub fn query_within_planar<'a>(
        &'a self,
        state: &'a SimState,
        center: Vec3,
        radius: f32,
    ) -> impl Iterator<Item = AgentId> + 'a {
        let r2 = radius * radius;
        let (cx, cy) = cell(center.x, center.y);
        let cell_reach = cell_reach_for_radius(radius);

        let cells: Vec<(i32, i32)> = (-cell_reach..=cell_reach)
            .flat_map(move |dx| (-cell_reach..=cell_reach).map(move |dy| (cx + dx, cy + dy)))
            .collect();

        let column_hits = cells
            .into_iter()
            .flat_map(move |key| {
                self.columns
                    .get(&key)
                    .into_iter()
                    .flat_map(|col| col.iter().copied())
            })
            .filter_map(move |(_z, id)| {
                state
                    .agent_pos(id)
                    .filter(|p| {
                        let dx = p.x - center.x;
                        let dy = p.y - center.y;
                        dx * dx + dy * dy <= r2
                    })
                    .map(|_| id)
            });

        let sidecar_hits = self.sidecar.iter().copied().filter_map(move |id| {
            state
                .agent_pos(id)
                .filter(|p| {
                    let dx = p.x - center.x;
                    let dy = p.y - center.y;
                    dx * dx + dy * dy <= r2
                })
                .map(|_| id)
        });

        column_hits.chain(sidecar_hits)
    }
}

/// Number of cell steps in each cardinal direction the query must scan
/// to guarantee no in-range walker is missed. For `radius <= CELL_SIZE`
/// this is 1 (the original 3×3 neighbourhood); grows linearly with
/// `radius`. Capped at 256 cells to prevent runaway scans on pathological
/// radii (256 × 16m = 4096m spans 65536 cells).
fn cell_reach_for_radius(radius: f32) -> i32 {
    if radius <= 0.0 { return 0; }
    let cells = (radius / CELL_SIZE).ceil() as i32;
    cells.clamp(1, 256)
}
