//! Generic harness — drives any per-fixture runtime that implements
//! [`CompiledSim`]. The active fixture is selected by `sim_app/Cargo.toml`'s
//! `package = "..."` rename of the `sim_runtime` dep; this file imports only
//! through the trait so it stays sim-agnostic.
//!
//! Today the loop is the absolute minimum: construct the sim, tick it
//! N times, log the per-tick header + a per-tick *cluster spread*
//! summary (max axis-aligned diameter and centroid) so flocking
//! convergence is visible without a renderer. The boids fixture's
//! steering pulls the swarm together; the spread should shrink over
//! time once cohesion / alignment dominate.
//!
//! Configuration is hard-coded for v1 (seed, agent_count, ticks, log
//! interval). Promote to CLI args / config file when the harness needs
//! more than one shape of run.

use engine::CompiledSim;
use glam::Vec3;

const SEED: u64 = 0xB01D_5_C0FF_EE_42;
const AGENT_COUNT: u32 = 64;
const TICKS: u64 = 200;
const LOG_INTERVAL_TICKS: u64 = 25;

fn main() {
    let mut sim = sim_runtime::make_sim(SEED, AGENT_COUNT);
    println!(
        "sim_app: starting run — seed=0x{:016X} agents={} ticks={}",
        SEED,
        sim.agent_count(),
        TICKS,
    );
    log_sample(sim.as_mut());

    for _ in 0..TICKS {
        sim.step();
        if sim.tick() % LOG_INTERVAL_TICKS == 0 {
            log_sample(sim.as_mut());
        }
    }

    println!(
        "sim_app: finished — final tick={} agents={}",
        sim.tick(),
        sim.agent_count(),
    );
}

/// Print one summary line per call: tick, centroid, axis-aligned
/// bounding-box dimensions (max - min on each axis), and the maximum
/// axis-aligned diameter (max of the three). The diameter is the
/// flocking-convergence signal — boids that are flocking will see it
/// shrink as cohesion + alignment pull the swarm together; constant-
/// velocity drift would keep it roughly constant.
fn log_sample(sim: &mut dyn CompiledSim) {
    let tick = sim.tick();
    let positions = sim.positions();
    if positions.is_empty() {
        println!("  tick {:>4}: (no agents)", tick);
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
    let centroid = sum / (positions.len() as f32);
    let span = max - min;
    let diameter = span.x.max(span.y).max(span.z);
    println!(
        "  tick {:>4}: centroid=({:>+8.3}, {:>+8.3}, {:>+8.3}) \
         span=({:>+6.2}, {:>+6.2}, {:>+6.2}) max_diameter={:>+6.2}",
        tick, centroid.x, centroid.y, centroid.z, span.x, span.y, span.z, diameter,
    );
}
