//! Foraging-colony coverage demo. Drives the [`foraging_runtime`]
//! integrator through the same generic harness shape as the other
//! fixture binaries. The fixture is the next coverage probe past
//! `bartering_app` (which exercised `entity Coin : Item` field reads
//! via `items.weight(0)`); here, the .sim file declares both
//! `entity Food : Item` AND `entity Colony : Group`, but neither is
//! exercised by the active rule body — the per-Item and per-Group
//! views (`pheromone_trail (ant: Agent, food: Food)` with
//! `storage = pair_map` and `colony_intake (colony: Group)`) stay
//! commented out behind `// GAP:` markers in the .sim file.
//!
//! What this app DOES exercise:
//!
//! - Single-Agent SoA (`Ant : Agent`).
//! - Per-tick `emit Drop { ant: self, carried: 0, pos: new_pos }`
//!   from `physics WanderAndDrop` — one event per alive ant per
//!   tick. `carried` is hard-coded to slot 0 so the pair_map
//!   `pheromone_trail` view's per-(ant, 0) column accumulates
//!   while every other (ant, food) cell stays at 0 — confirms
//!   `agent_count × FOOD_COUNT` storage sizing (NOT `agent_count²`).
//! - Single `pheromone_deposits(ant: Agent) -> f32` view with
//!   `@decay(rate=0.88)`. Per-event the matching slot increments by
//!   1.0 BEFORE the next per-tick anchor multiply lands, so the
//!   per-slot steady-state is `1 / (1 - 0.88) ≈ 8.33`.
//!
//! After T=100 ticks the geometric series sum
//! `sum_{k=0..T-1} 0.88^k ≈ 1 / (1 - 0.88) ≈ 8.333` already pins
//! the per-slot value to within < 1e-3 of the steady state (0.88^100
//! ≈ 2.65e-6, well below the ±5% acceptance band).

use engine::CompiledSim;
use foraging_runtime::ForagingState;
use glam::Vec3;

const SEED: u64 = 0xF02A_6_47C0_10_E5;
const AGENT_COUNT: u32 = 64;
const TICKS: u64 = 100;
const LOG_INTERVAL_TICKS: u64 = 25;

fn main() {
    let mut sim = ForagingState::new(SEED, AGENT_COUNT);
    println!(
        "foraging_app: starting run — seed=0x{:016X} ants={} ticks={}",
        SEED, AGENT_COUNT, TICKS,
    );
    log_sample(&mut sim);
    for _ in 0..TICKS {
        sim.step();
        if sim.tick() % LOG_INTERVAL_TICKS == 0 {
            log_sample(&mut sim);
        }
    }
    println!(
        "foraging_app: finished — final tick={} ants={}",
        sim.tick(),
        sim.agent_count(),
    );

    // ---- View readback + steady-state assertion ----
    let pd = sim.pheromone_deposits().to_vec();
    let mut total = 0.0_f32;
    let mut min = f32::INFINITY;
    let mut max = 0.0_f32;
    for &v in &pd {
        total += v;
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let mean = total / AGENT_COUNT as f32;
    // `pheromone_deposits` per-slot steady state.
    // Per-event += 1.0 (only one Drop per tick lands on each ant
    // slot — `ant: self`); `@decay(rate=0.88)` multiplies BEFORE the
    // fold so steady = 1 / (1 - 0.88) ≈ 8.333.
    let steady = 1.0_f32 / (1.0 - 0.88);
    let total_expected = AGENT_COUNT as f32 * steady;
    println!(
        "foraging_app: pheromone_deposits  (rate=0.88, steady ~{:.3}/slot) \
         — total={:.2} (expected {:.2}); per-slot min={:.3} mean={:.3} max={:.3}",
        steady, total, total_expected, min, mean, max,
    );

    // Acceptance band per the task spec: per-slot value within ±5%
    // of analytical steady state (0.88^100 ≈ 2.65e-6 — geometric tail
    // pins us well under the band).
    let tol = steady * 0.05;
    assert!(
        (mean - steady).abs() <= tol,
        "pheromone_deposits per-slot mean {:.4} not within ±5% of analytical {:.4} \
         (band ±{:.4})",
        mean, steady, tol,
    );
    assert!(
        (min - steady).abs() <= tol && (max - steady).abs() <= tol,
        "pheromone_deposits per-slot min/max ({:.4}/{:.4}) not within ±5% of \
         analytical {:.4}; uniform-population fixture should have identical \
         per-slot values across all 64 ants",
        min, max, steady,
    );
    println!(
        "foraging_app: OK — pheromone_deposits per-slot mean ≈ {:.3} (target {:.3}, \
         tol ±{:.3})",
        mean, steady, tol,
    );

    // SLICE 3 PROBE: Group-keyed view readback. Deposited is never
    // emitted by any active rule, so the buffer should stay at the
    // all-zeros initial state. The fold ran every tick with
    // event_count=0 (no-op) — this confirms the Group-keyed view's
    // decay + fold dispatch path is healthy end-to-end without a
    // producer emit.
    let ci = sim.colony_intakes().to_vec();
    let ci_total: f32 = ci.iter().sum();
    let ci_max: f32 = ci.iter().copied().fold(0.0, f32::max);
    println!(
        "foraging_app: colony_intake       (rate=0.95, no Deposited emitter) \
         — total={:.4} max={:.4} (both expected 0.0)",
        ci_total, ci_max,
    );
    assert!(
        ci_total.abs() < 1e-3 && ci_max.abs() < 1e-3,
        "colony_intake should stay at zero (no Deposited emitter wired); \
         got total={ci_total}, max={ci_max}"
    );
    println!(
        "foraging_app: OK — colony_intake stayed zero across {} ticks (Group-keyed \
         view's compile + decay + fold-dispatch path is healthy)",
        TICKS,
    );

    // Item-population-aware pair_map sizing observable
    // (closes the slice-2 wrap-up gap from `cd93cb04`).
    // `pheromone_trail(ant: Agent, food: Food)` storage now sizes
    // as `agent_count × FOOD_COUNT` — NOT `agent_count²` —
    // because the runtime supplies `second_key_pop = FOOD_COUNT`
    // independently of `agent_cap`. With WanderAndDrop emitting
    // `Drop { ant: self, carried: 0 }`, every Drop hashes to food
    // slot 0 of its ant's row; the per-`(ant, 0)` cell converges
    // to `1 / (1 - 0.88) ≈ 8.33`, every other `(ant, k)` stays at
    // exactly 0, and the total is `agent_count × 8.33`
    // (= 533.33 at agent_count=64) rather than the prior
    // `agent_count² × 8.33` over-allocation.
    let pt = sim.pheromone_trail().to_vec();
    let agent_count = AGENT_COUNT as usize;
    let food_count = foraging_runtime::FOOD_COUNT as usize;
    assert_eq!(
        pt.len(),
        agent_count * food_count,
        "pheromone_trail storage should be agent_count × FOOD_COUNT \
         (Item-population-aware sizing; was agent_count² before the gap fix)",
    );
    let pt_steady = 1.0_f32 / (1.0 - 0.88);

    let mut col0_min = f32::INFINITY;
    let mut col0_max = 0.0_f32;
    let mut col0_sum = 0.0_f32;
    let mut other_max = 0.0_f32;
    let mut other_sum = 0.0_f32;
    let mut other_n = 0u32;
    for ant in 0..agent_count {
        for k in 0..food_count {
            let v = pt[ant * food_count + k];
            if k == 0 {
                col0_min = col0_min.min(v);
                col0_max = col0_max.max(v);
                col0_sum += v;
            } else {
                other_max = other_max.max(v);
                other_sum += v;
                other_n += 1;
            }
        }
    }
    let col0_mean = col0_sum / agent_count as f32;
    let other_mean = if other_n > 0 {
        other_sum / other_n as f32
    } else {
        0.0
    };
    let total: f32 = pt.iter().sum();
    let total_expected = agent_count as f32 * pt_steady;
    println!(
        "foraging_app: pheromone_trail (pair_map, rate=0.88, per-(ant,0) steady ~{:.3}) \
         — slots={} (= {} × {}); per-(ant,0): min={:.3} mean={:.3} max={:.3}; \
         per-(ant,k!=0): max={:.4} mean={:.4} (expected 0.0); \
         total={:.2} (expected {:.2})",
        pt_steady,
        pt.len(),
        agent_count,
        food_count,
        col0_min,
        col0_mean,
        col0_max,
        other_max,
        other_mean,
        total,
        total_expected,
    );
    let pt_tol = pt_steady * 0.05;
    assert!(
        (col0_mean - pt_steady).abs() <= pt_tol,
        "pheromone_trail per-(ant,0) mean {:.4} not within ±5% of analytical {:.4}",
        col0_mean,
        pt_steady,
    );
    assert!(
        (col0_min - pt_steady).abs() <= pt_tol && (col0_max - pt_steady).abs() <= pt_tol,
        "pheromone_trail per-(ant,0) min/max ({:.4}/{:.4}) not within ±5% of \
         analytical {:.4}; uniform-population fixture should have identical \
         per-(ant,0) values across all 64 ants",
        col0_min,
        col0_max,
        pt_steady,
    );
    assert!(
        other_max.abs() < 1e-3,
        "pheromone_trail per-(ant,k!=0) max {} should be ~0 \
         (Drop only writes to carried=0)",
        other_max,
    );
    let total_tol = total_expected * 0.05;
    assert!(
        (total - total_expected).abs() <= total_tol,
        "pheromone_trail total {:.2} not within ±5% of analytical {:.2} \
         (= agent_count × steady; the prior agent_count² over-allocation \
         would have totalled the same value but spread it across a much \
         larger buffer)",
        total,
        total_expected,
    );
    println!(
        "foraging_app: OK — pheromone_trail per-(ant,0) mean ≈ {:.3} (target {:.3}, \
         tol ±{:.3}); per-(ant,k!=0) stays at 0; total = {:.2} (target {:.2}, NOT \
         agent_count² × steady — Item-population sizing closure)",
        col0_mean, pt_steady, pt_tol, total, total_expected,
    );
}

/// One per-tick log line: centroid + max axis-aligned span.
fn log_sample(sim: &mut ForagingState) {
    let tick = sim.tick();
    let positions = sim.positions();
    if positions.is_empty() {
        println!("  tick {:>4}: (no ants)", tick);
        return;
    }
    let mut min = positions[0];
    let mut max = positions[0];
    let mut sum = Vec3::ZERO;
    for p in positions {
        min = min.min(*p);
        max = max.max(*p);
        sum += *p;
    }
    let centroid = sum / positions.len() as f32;
    let span = max - min;
    let max_diameter = span.x.max(span.y).max(span.z);
    println!(
        "  tick {:>4}: centroid=({:>+7.3}, {:>+7.3}, {:>+7.3}) max_diameter={:>+6.2}",
        tick, centroid.x, centroid.y, centroid.z, max_diameter,
    );
}
