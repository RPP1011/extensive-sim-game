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
//! - Per-tick `emit Drop { ant: self, carried: self, pos: new_pos }`
//!   from `physics WanderAndDrop` — one event per alive ant per
//!   tick.
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
