use std::time::Instant;

use rayon::prelude::*;

use super::delta::{WorldDelta, merge_deltas};
use super::fidelity::Fidelity;
use super::state::WorldState;

/// Per-tick profiling data.
#[derive(Debug, Clone, Default)]
pub struct TickProfile {
    /// Wall time for the compute phase (entity + grid delta generation).
    pub compute_us: u64,
    /// Wall time for the merge phase (commutative fold of all deltas).
    pub merge_us: u64,
    /// Wall time for the apply phase (produce next state from merged deltas).
    pub apply_us: u64,
    /// Total wall time for the entire tick.
    pub total_us: u64,
    /// Number of deltas produced.
    pub delta_count: usize,
    /// Number of alive entities processed.
    pub entities_processed: usize,
    /// Breakdown by fidelity level.
    pub high_count: usize,
    pub medium_count: usize,
    pub low_count: usize,
    pub background_count: usize,

    // --- Per-fidelity compute timing ---
    pub compute_high_us: u64,
    pub compute_medium_us: u64,
    pub compute_low_us: u64,
    pub compute_grid_us: u64,

    // --- Apply sub-phase timing ---
    pub apply_clone_us: u64,
    pub apply_hp_us: u64,
    pub apply_movement_us: u64,
    pub apply_status_us: u64,
    pub apply_economy_us: u64,
    pub apply_transfers_us: u64,
    pub apply_deaths_us: u64,
    pub apply_grid_us: u64,
    pub apply_fidelity_us: u64,
    pub apply_price_reports_us: u64,
    /// Post-apply systems (agent_inner, goals, work, pathfinding, families, etc.)
    pub postapply_us: u64,
}

impl std::fmt::Display for TickProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tick: {}µs (compute {}µs [H:{}µs M:{}µs L:{}µs G:{}µs], merge {}µs, apply {}µs [clone:{}µs hp:{}µs mv:{}µs econ:{}µs xfer:{}µs]) | {} deltas, {} entities (H:{} M:{} L:{} B:{})",
            self.total_us, self.compute_us,
            self.compute_high_us, self.compute_medium_us, self.compute_low_us, self.compute_grid_us,
            self.merge_us, self.apply_us,
            self.apply_clone_us, self.apply_hp_us, self.apply_movement_us,
            self.apply_economy_us, self.apply_transfers_us,
            self.delta_count, self.entities_processed,
            self.high_count, self.medium_count, self.low_count, self.background_count,
        )
    }
}

/// Accumulated profiling stats over many ticks.
#[derive(Debug, Clone, Default)]
pub struct ProfileAccumulator {
    pub tick_count: u64,
    pub total_compute_us: u64,
    pub total_merge_us: u64,
    pub total_apply_us: u64,
    pub total_us: u64,
    pub total_deltas: u64,
    pub max_tick_us: u64,
    pub min_tick_us: u64,
}

impl ProfileAccumulator {
    pub fn record(&mut self, p: &TickProfile) {
        self.tick_count += 1;
        self.total_compute_us += p.compute_us;
        self.total_merge_us += p.merge_us;
        self.total_apply_us += p.apply_us;
        self.total_us += p.total_us;
        self.total_deltas += p.delta_count as u64;
        if p.total_us > self.max_tick_us { self.max_tick_us = p.total_us; }
        if self.tick_count == 1 || p.total_us < self.min_tick_us {
            self.min_tick_us = p.total_us;
        }
    }

    pub fn avg_tick_us(&self) -> u64 {
        if self.tick_count == 0 { 0 } else { self.total_us / self.tick_count }
    }

    pub fn avg_compute_us(&self) -> u64 {
        if self.tick_count == 0 { 0 } else { self.total_compute_us / self.tick_count }
    }

    pub fn avg_merge_us(&self) -> u64 {
        if self.tick_count == 0 { 0 } else { self.total_merge_us / self.tick_count }
    }

    pub fn avg_apply_us(&self) -> u64 {
        if self.tick_count == 0 { 0 } else { self.total_apply_us / self.tick_count }
    }
}

impl std::fmt::Display for ProfileAccumulator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ticks | avg {}µs (compute {}µs, merge {}µs, apply {}µs) | min {}µs, max {}µs | {:.1} deltas/tick",
            self.tick_count,
            self.avg_tick_us(), self.avg_compute_us(), self.avg_merge_us(), self.avg_apply_us(),
            self.min_tick_us, self.max_tick_us,
            if self.tick_count == 0 { 0.0 } else { self.total_deltas as f64 / self.tick_count as f64 },
        )
    }
}

/// Advance the world by one tick (sequential).
pub fn tick(state: &WorldState) -> WorldState {
    tick_profiled(state, false).0
}

/// Advance the world by one tick (parallel via rayon).
pub fn tick_par(state: &WorldState) -> WorldState {
    tick_profiled(state, true).0
}

/// Advance the world by one tick, returning the new state and profiling data.
pub fn tick_profiled(state: &WorldState, parallel: bool) -> (WorldState, TickProfile) {
    let tick_start = Instant::now();
    let mut profile = TickProfile::default();

    // COMPUTE
    let compute_start = Instant::now();
    let (all_deltas, fidelity_counts, fidelity_timings) = if parallel {
        compute_all_deltas_par_counted(state)
    } else {
        compute_all_deltas_seq_counted(state)
    };
    profile.compute_us = compute_start.elapsed().as_micros() as u64;
    profile.delta_count = all_deltas.len();
    profile.high_count = fidelity_counts.0;
    profile.medium_count = fidelity_counts.1;
    profile.low_count = fidelity_counts.2;
    profile.background_count = fidelity_counts.3;
    profile.entities_processed = fidelity_counts.0 + fidelity_counts.1 + fidelity_counts.2 + fidelity_counts.3;
    profile.compute_high_us = fidelity_timings.0;
    profile.compute_medium_us = fidelity_timings.1;
    profile.compute_low_us = fidelity_timings.2;
    profile.compute_grid_us = fidelity_timings.3;

    // MERGE
    let merge_start = Instant::now();
    let merged = merge_deltas(all_deltas);
    profile.merge_us = merge_start.elapsed().as_micros() as u64;

    // APPLY (sub-phase timing)
    let apply_start = Instant::now();
    let (next, apply_profile) = super::apply::apply_deltas_profiled(state, &merged);
    profile.apply_us = apply_start.elapsed().as_micros() as u64;
    profile.apply_clone_us = apply_profile.clone_us;
    profile.apply_hp_us = apply_profile.hp_us;
    profile.apply_movement_us = apply_profile.movement_us;
    profile.apply_status_us = apply_profile.status_us;
    profile.apply_economy_us = apply_profile.economy_us;
    profile.apply_transfers_us = apply_profile.transfers_us;
    profile.apply_deaths_us = apply_profile.deaths_us;
    profile.apply_grid_us = apply_profile.grid_us;
    profile.apply_fidelity_us = apply_profile.fidelity_us;
    profile.apply_price_reports_us = apply_profile.price_reports_us;

    profile.total_us = tick_start.elapsed().as_micros() as u64;
    (next, profile)
}

/// (high, medium, low, background) entity counts + per-fidelity timings in µs.
type FidelityCounts = (usize, usize, usize, usize);
type FidelityTimings = (u64, u64, u64, u64); // high_us, medium_us, low_us, grid_us

/// Sequential delta collection with fidelity counting and timing.
fn compute_all_deltas_seq_counted(state: &WorldState) -> (Vec<WorldDelta>, FidelityCounts, FidelityTimings) {
    let spatial = super::spatial::SpatialIndex::build(&state.entities);

    let mut all_deltas: Vec<WorldDelta> = Vec::new();
    let mut counts: FidelityCounts = (0, 0, 0, 0);
    let mut timings: FidelityTimings = (0, 0, 0, 0);

    for entity in &state.entities {
        if !entity.alive { continue; }
        let fid = entity_fidelity(entity, state);
        let t = Instant::now();
        match fid {
            Fidelity::High => counts.0 += 1,
            Fidelity::Medium => counts.1 += 1,
            Fidelity::Low => counts.2 += 1,
            Fidelity::Background => counts.3 += 1,
        }
        all_deltas.extend(compute_entity_at(entity, state, fid));
        let elapsed = t.elapsed().as_micros() as u64;
        match fid {
            Fidelity::High => timings.0 += elapsed,
            Fidelity::Medium => timings.1 += elapsed,
            Fidelity::Low => timings.2 += elapsed,
            Fidelity::Background => {}
        }
    }

    let grid_start = Instant::now();
    for grid in &state.fidelity_zones {
        all_deltas.extend(compute_grid_deltas(grid, &spatial));
    }
    timings.3 = grid_start.elapsed().as_micros() as u64;

    (all_deltas, counts, timings)
}

/// Parallel delta collection with fidelity counting.
fn compute_all_deltas_par_counted(state: &WorldState) -> (Vec<WorldDelta>, FidelityCounts, FidelityTimings) {
    let spatial = super::spatial::SpatialIndex::build(&state.entities);

    let compute_start = Instant::now();
    let (entity_deltas, counts) = state.entities
        .par_iter()
        .filter(|e| e.alive)
        .fold(
            || (Vec::new(), (0usize, 0usize, 0usize, 0usize)),
            |(mut deltas, mut c), entity| {
                let fid = entity_fidelity(entity, state);
                match fid {
                    Fidelity::High => c.0 += 1,
                    Fidelity::Medium => c.1 += 1,
                    Fidelity::Low => c.2 += 1,
                    Fidelity::Background => c.3 += 1,
                }
                deltas.extend(compute_entity_at(entity, state, fid));
                (deltas, c)
            },
        )
        .reduce(
            || (Vec::new(), (0, 0, 0, 0)),
            |(mut a_d, a_c), (b_d, b_c)| {
                a_d.extend(b_d);
                (a_d, (a_c.0 + b_c.0, a_c.1 + b_c.1, a_c.2 + b_c.2, a_c.3 + b_c.3))
            },
        );
    let entity_compute_us = compute_start.elapsed().as_micros() as u64;

    let grid_start = Instant::now();
    for _grid in &state.fidelity_zones {
        // Grid deltas are cheap with spatial index — no need to parallelize.
    }
    let grid_deltas: Vec<WorldDelta> = state.fidelity_zones
        .iter()
        .flat_map(|grid| compute_grid_deltas(grid, &spatial))
        .collect();
    let grid_us = grid_start.elapsed().as_micros() as u64;

    let mut all = entity_deltas;
    all.extend(grid_deltas);
    (all, counts, (0, entity_compute_us, 0, grid_us))
}

/// Compute deltas for a single entity at a known fidelity level.
fn compute_entity_at(entity: &super::state::Entity, state: &WorldState, fid: Fidelity) -> Vec<WorldDelta> {
    match fid {
        Fidelity::High => super::compute_high::compute_entity_deltas(entity, state),
        Fidelity::Medium => super::compute_medium::compute_entity_deltas(entity, state),
        Fidelity::Low => super::compute_low::compute_entity_deltas(entity, state),
        Fidelity::Background => Vec::new(),
    }
}

/// Determine an entity's simulation fidelity from its location.
fn entity_fidelity(entity: &super::state::Entity, state: &WorldState) -> Fidelity {
    if let Some(grid_id) = entity.grid_id {
        state.fidelity_zone(grid_id)
            .map(|g| g.fidelity)
            .unwrap_or(Fidelity::Low)
    } else {
        // On the overworld — default to Low.
        Fidelity::Low
    }
}

/// Compute fidelity change deltas for a grid using the spatial index.
fn compute_grid_deltas(
    grid: &super::state::FidelityZone,
    spatial: &super::spatial::SpatialIndex,
) -> Vec<WorldDelta> {
    let has_hostiles = spatial.has_hostiles_in_radius(grid.center, grid.radius);
    let has_friendlies = spatial.has_friendlies_in_radius(grid.center, grid.radius);

    let desired = if has_hostiles && has_friendlies {
        Fidelity::High
    } else if has_friendlies {
        Fidelity::Medium
    } else if has_hostiles {
        Fidelity::Low
    } else {
        Fidelity::Background
    };

    if desired != grid.fidelity {
        vec![WorldDelta::EscalateFidelity {
            grid_id: grid.id,
            new_fidelity: desired,
        }]
    } else {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::*;

    #[test]
    fn tick_advances_state() {
        let state = WorldState::new(42);
        let next = tick(&state);
        assert_eq!(next.tick, 1);
        let next2 = tick(&next);
        assert_eq!(next2.tick, 2);
    }

    #[test]
    fn empty_world_ticks() {
        let state = WorldState::new(0);
        // 100 ticks of an empty world should just increment tick counter.
        let mut s = state;
        for _ in 0..100 {
            s = tick(&s);
        }
        assert_eq!(s.tick, 100);
    }

    #[test]
    fn grid_fidelity_escalates_on_contact() {
        let mut state = WorldState::new(42);
        state.entities.push(Entity::new_npc(1, (0.0, 0.0)));
        state.entities.push(Entity::new_monster(2, (1.0, 0.0), 1));
        state.entities[0].grid_id = Some(100);
        state.entities[1].grid_id = Some(100);
        state.fidelity_zones.push(FidelityZone {
            id: 100,
            fidelity: Fidelity::Medium,
            center: (0.0, 0.0),
            radius: 10.0,
            entity_ids: vec![1, 2],
        });

        let next = tick(&state);
        // Should escalate to High because both hostile and friendly present.
        assert_eq!(next.fidelity_zone(100).unwrap().fidelity, Fidelity::High);
    }

    #[test]
    fn grid_fidelity_deescalates_when_no_hostiles() {
        let mut state = WorldState::new(42);
        state.entities.push(Entity::new_npc(1, (0.0, 0.0)));
        state.entities[0].grid_id = Some(100);
        state.fidelity_zones.push(FidelityZone {
            id: 100,
            fidelity: Fidelity::High,
            center: (0.0, 0.0),
            radius: 10.0,
            entity_ids: vec![1],
        });

        let next = tick(&state);
        // Only friendlies → Medium.
        assert_eq!(next.fidelity_zone(100).unwrap().fidelity, Fidelity::Medium);
    }

    /// Build a world with many entities for determinism testing.
    fn big_world() -> WorldState {
        let mut state = WorldState::new(42);
        // Create a settlement with grid.
        state.settlements.push(super::super::state::SettlementState::new(
            10, "Town".into(), (50.0, 50.0),
        ));
        state.settlements[0].stockpile[0] = 1000.0; // food
        state.fidelity_zones.push(FidelityZone {
            id: 10,
            fidelity: Fidelity::Medium,
            center: (50.0, 50.0),
            radius: 30.0,
            entity_ids: Vec::new(),
        });

        // 50 NPCs in the settlement.
        for i in 0..50 {
            let mut npc = Entity::new_npc(i, (50.0 + (i as f32) * 0.1, 50.0));
            npc.grid_id = Some(10);
            let npc_data = npc.npc.as_mut().unwrap();
            npc_data.home_settlement_id = Some(10);
            npc_data.behavior_production = vec![(0, 0.1)]; // produce food
            state.entities.push(npc);
            state.fidelity_zones[0].entity_ids.push(i);
        }

        // 20 monsters approaching.
        for i in 50..70 {
            state.entities.push(Entity::new_monster(
                i,
                (100.0 + (i as f32) * 2.0, 50.0),
                (i % 5) as u32 + 1,
            ));
        }

        state
    }

    #[test]
    fn sequential_and_parallel_produce_identical_results() {
        let state = big_world();
        // Run 10 ticks sequentially.
        let mut seq = state.clone();
        for _ in 0..10 {
            seq = tick(&seq);
        }
        // Run 10 ticks in parallel.
        let mut par = state;
        for _ in 0..10 {
            par = tick_par(&par);
        }
        // Compare tick counters.
        assert_eq!(seq.tick, par.tick);
        // Compare entity states.
        assert_eq!(seq.entities.len(), par.entities.len());
        for (s, p) in seq.entities.iter().zip(par.entities.iter()) {
            assert_eq!(s.id, p.id);
            assert!((s.hp - p.hp).abs() < 1e-4, "HP mismatch for entity {}: {} vs {}", s.id, s.hp, p.hp);
            assert!((s.pos.0 - p.pos.0).abs() < 1e-4, "pos.x mismatch for entity {}", s.id);
            assert!((s.pos.1 - p.pos.1).abs() < 1e-4, "pos.y mismatch for entity {}", s.id);
            assert_eq!(s.alive, p.alive, "alive mismatch for entity {}", s.id);
        }
        // Compare settlement stockpiles.
        for (s, p) in seq.settlements.iter().zip(par.settlements.iter()) {
            for i in 0..8 {
                assert!((s.stockpile[i] - p.stockpile[i]).abs() < 1e-4,
                    "stockpile[{}] mismatch for settlement {}", i, s.id);
            }
        }
    }
}
