//! Throughput probe for the boids GPU pipeline. Runs `--ticks` step()
//! calls at each `--agents` count, measuring step throughput (no
//! readback in the hot loop — readback is on demand) and one
//! end-of-run positions() readback. Defaults exercise a sweep from
//! 1k → 1M agents.
//!
//! Usage:
//!   cargo run --release --example bench
//!   cargo run --release --example bench -- --ticks 200 --max 4000000

use boids_runtime::BoidsState;
use engine::CompiledSim;
use std::time::Instant;

fn main() {
    let mut ticks: u64 = 100;
    let mut max_agents: u32 = 1_000_000;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ticks" => ticks = args.next().unwrap().parse().unwrap(),
            "--max" => max_agents = args.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg: {other}"),
        }
    }

    println!(
        "boids_runtime bench — {} ticks per cap, GPU dispatch only (no readback in loop)",
        ticks
    );
    println!(
        "{:>10}  {:>12}  {:>14}  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}  {:>10}  {:>10}",
        "agents",
        "step_total_ms",
        "agent-ticks/s",
        "readback_ms",
        "clear_us",
        "build_us",
        "moveb_us",
        "total_us",
        "max/cell",
        "dropped",
        "nonempty",
    );

    let caps: Vec<u32> = [1_000u32, 10_000, 100_000, 1_000_000, 4_000_000]
        .into_iter()
        .filter(|n| *n <= max_agents)
        .collect();

    for n in caps {
        let mut sim = BoidsState::new(0xB01D_5_C0FF_EE_42, n);
        // Warm up — first dispatch lazy-inits the pipeline; we don't
        // want that one-shot init time in the loop measurement.
        sim.step();
        let _ = sim.positions();
        let _ = sim.metrics();

        let t0 = Instant::now();
        for _ in 0..ticks {
            sim.step();
        }
        let step_total = t0.elapsed();
        let agent_ticks_per_s = (n as f64 * ticks as f64) / step_total.as_secs_f64();

        let t1 = Instant::now();
        let _ = sim.positions();
        let readback = t1.elapsed();

        // Last tick's per-kernel GPU times — surfaces zeros if the
        // adapter doesn't expose `Features::TIMESTAMP_QUERY`. Convert
        // to microseconds for readability.
        let m = sim.metrics();
        let to_us = |ns: u64| (ns as f64) / 1_000.0;

        println!(
            "{:>10}  {:>12.2}  {:>14.2e}  {:>12.2}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}  {:>8}  {:>10}  {:>10}",
            n,
            step_total.as_secs_f64() * 1_000.0,
            agent_ticks_per_s,
            readback.as_secs_f64() * 1_000.0,
            to_us(m.clear_ns),
            to_us(m.build_hash_ns),
            to_us(m.move_boid_ns),
            to_us(m.total_ns),
            m.max_per_cell_seen,
            m.dropped_agents,
            m.nonempty_cells,
        );
    }
}
