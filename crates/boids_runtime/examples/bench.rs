//! Throughput probe for the boids GPU pipeline. Runs `--ticks` step()
//! calls at each `--agents` count, measuring step throughput (no
//! readback in the hot loop — readback is on demand) and one
//! end-of-run positions() readback. Defaults exercise a sweep from
//! 1k → 1M agents.
//!
//! ## Spread mode (density vs legacy)
//!
//! By default the harness uses [`BoidsState::new`]'s density-aware
//! auto-spread — the initial-position cube grows with `cbrt(N)` so
//! average spatial-grid density stays near one agent per cell. Pass
//! `--legacy-spread` to force the old fixed `±8` cube every cap, which
//! crushes large-N runs into a tiny corner of the world (most agents
//! pile into a handful of cells, BuildHash drops them past
//! `MAX_PER_CELL`, the spatial walks miss almost every neighbor). The
//! `dropped` and `dropped%` columns make the difference obvious — at
//! 1M agents, legacy mode reports ~93% drops while density-aware mode
//! reports a small fraction. The two modes share everything else
//! (kernels, buffers, seed, tick count) so a side-by-side run isolates
//! the spread variable cleanly.
//!
//! Usage:
//!   cargo run --release --example bench
//!   cargo run --release --example bench -- --ticks 200 --max 4000000
//!   cargo run --release --example bench -- --legacy-spread

use boids_runtime::BoidsState;
use engine::CompiledSim;
use std::time::Instant;

fn main() {
    let mut ticks: u64 = 100;
    let mut max_agents: u32 = 1_000_000;
    let mut legacy_spread = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ticks" => ticks = args.next().unwrap().parse().unwrap(),
            "--max" => max_agents = args.next().unwrap().parse().unwrap(),
            "--legacy-spread" => legacy_spread = true,
            other => panic!("unknown arg: {other}"),
        }
    }

    let mode = if legacy_spread {
        "legacy fixed ±8 cube"
    } else {
        "density-aware auto-spread"
    };
    println!(
        "boids_runtime bench — {} ticks per cap, GPU dispatch only (no readback in loop) — spread mode: {}",
        ticks, mode,
    );
    // Two-row layout per cap so the t0 and tN diag can sit side-by-
    // side. The t0 row captures the raw initial-distribution health
    // (what the spread parameter directly controls); the tN row
    // captures the end-of-run health (which can drift as the flock
    // clusters under cohesion). The dropped column on the t0 row is
    // the headline number for this fix — the legacy ±8 cube produces
    // catastrophic drops there at large N, while density-aware
    // auto-spread keeps them low.
    println!(
        "{:>10}  {:>8}  {:>5}  {:>12}  {:>14}  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}  {:>10}  {:>9}  {:>10}",
        "agents",
        "spread",
        "phase",
        "step_total_ms",
        "agent-ticks/s",
        "readback_ms",
        "clear_us",
        "build_us",
        "moveb_us",
        "total_us",
        "max/cell",
        "dropped",
        "dropped%",
        "nonempty",
    );

    let caps: Vec<u32> = [1_000u32, 10_000, 100_000, 1_000_000, 4_000_000]
        .into_iter()
        .filter(|n| *n <= max_agents)
        .collect();

    for n in caps {
        // Per-cap spread choice — explicit so the harness output is
        // self-documenting + the legacy mode reproduces the pre-fix
        // behavior bit-for-bit (same `±8` constant the old hand-coded
        // path used).
        let spread = if legacy_spread {
            8.0_f32
        } else {
            BoidsState::auto_spread(n)
        };
        let mut sim =
            BoidsState::new_with_spread(0xB01D_5_C0FF_EE_42, n, Some(spread));
        // Warm up — first dispatch lazy-inits the pipeline; we don't
        // want that one-shot init time in the loop measurement.
        sim.step();
        let _ = sim.positions();
        // t0 metrics — the spatial-grid health on tick 1, i.e. what
        // BuildHash sees from the *initial* position distribution.
        // This is the row to read for "did the spread parameter give
        // us a sensible density?".
        let m_init = sim.metrics();

        let t0 = Instant::now();
        for _ in 0..ticks {
            sim.step();
        }
        let step_total = t0.elapsed();
        let agent_ticks_per_s = (n as f64 * ticks as f64) / step_total.as_secs_f64();

        let t1 = Instant::now();
        let _ = sim.positions();
        let readback = t1.elapsed();

        // tN metrics — health after `ticks + 1` steps. Cohesion pulls
        // the swarm together over time, so dropped/max-per-cell here
        // is "what density does the kernel actually see during the
        // perf measurement window?".
        let m_end = sim.metrics();
        let to_us = |ns: u64| (ns as f64) / 1_000.0;
        let pct = |drops: u32| {
            if n > 0 {
                (drops as f64) * 100.0 / (n as f64)
            } else {
                0.0
            }
        };

        // Row 1 — initial distribution health (what `spread` directly
        // controls). Per-kernel timings + step-loop totals are
        // recycled from the perf measurement; their values are
        // identical on both rows since we only have one timing
        // source per cap.
        println!(
            "{:>10}  {:>8.2}  {:>5}  {:>12.2}  {:>14.2e}  {:>12.2}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}  {:>8}  {:>10}  {:>8.2}%  {:>10}",
            n,
            spread,
            "t0",
            step_total.as_secs_f64() * 1_000.0,
            agent_ticks_per_s,
            readback.as_secs_f64() * 1_000.0,
            to_us(m_init.clear_ns),
            to_us(m_init.build_hash_ns),
            to_us(m_init.move_boid_ns),
            to_us(m_init.total_ns),
            m_init.max_per_cell_seen,
            m_init.dropped_agents,
            pct(m_init.dropped_agents),
            m_init.nonempty_cells,
        );
        // Row 2 — end-of-run density, after cohesion-driven clustering.
        println!(
            "{:>10}  {:>8}  {:>5}  {:>12}  {:>14}  {:>12}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}  {:>8}  {:>10}  {:>8.2}%  {:>10}",
            "",
            "",
            "tN",
            "",
            "",
            "",
            to_us(m_end.clear_ns),
            to_us(m_end.build_hash_ns),
            to_us(m_end.move_boid_ns),
            to_us(m_end.total_ns),
            m_end.max_per_cell_seen,
            m_end.dropped_agents,
            pct(m_end.dropped_agents),
            m_end.nonempty_cells,
        );
    }
}
