//! Generic harness — drives any per-fixture runtime that implements
//! [`CompiledSim`]. The active fixture is selected by `sim_app/Cargo.toml`'s
//! `package = "..."` rename of the `sim_runtime` dep; this file imports only
//! through the trait so it stays sim-agnostic.
//!
//! Today the loop is the absolute minimum: construct the sim, tick it N
//! times, log the per-tick header + a position sample at fixed intervals.
//! Once the per-fixture runtime's `step()` body lands a real per-tick
//! update (today it's a no-op that only advances the tick counter), the
//! sampled positions will start moving without any change to this file.
//!
//! Configuration is hard-coded for v1 (seed, agent_count, ticks, log
//! interval). Promote to CLI args / config file when the harness needs
//! more than one shape of run.

use engine::CompiledSim;

const SEED: u64 = 0xB01D_5_C0FF_EE_42;
const AGENT_COUNT: u32 = 16;
const TICKS: u64 = 100;
const LOG_INTERVAL_TICKS: u64 = 25;
const SAMPLE_AGENTS: usize = 4;

fn main() {
    let mut sim = sim_runtime::make_sim(SEED, AGENT_COUNT);
    println!(
        "sim_app: starting run — seed=0x{:016X} agents={} ticks={}",
        SEED,
        sim.agent_count(),
        TICKS,
    );
    log_sample(sim.as_ref());

    for _ in 0..TICKS {
        sim.step();
        if sim.tick() % LOG_INTERVAL_TICKS == 0 {
            log_sample(sim.as_ref());
        }
    }

    println!(
        "sim_app: finished — final tick={} agents={}",
        sim.tick(),
        sim.agent_count(),
    );
}

/// Print the tick header + position of up to [`SAMPLE_AGENTS`] agents.
/// Boids stay frozen until the runtime's `step()` lands real physics;
/// that's expected — the loop's job is to dispatch step + observe state,
/// not to compute it.
fn log_sample(sim: &dyn CompiledSim) {
    let positions = sim.positions();
    let n = SAMPLE_AGENTS.min(positions.len());
    println!("  tick {:>4}:", sim.tick());
    for (i, p) in positions.iter().take(n).enumerate() {
        println!(
            "    agent {:>3} pos=({:>+8.3}, {:>+8.3}, {:>+8.3})",
            i, p.x, p.y, p.z
        );
    }
}
