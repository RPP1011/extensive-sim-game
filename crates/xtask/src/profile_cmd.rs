//! `xtask profile` — phase timing histogram.
//!
//! Mirrors `engine::debug::DebugConfig { tick_profile: Some(..) }`.
//!
//! # Deviation note
//!
//! xtask has no engine / engine_rules dependency. This command simulates the
//! TickProfile collection loop with synthetic timing data derived from
//! `std::time::Instant`. To wire up a real run, add `engine_rules` to
//! xtask's `[dependencies]` and replace the stub loop with
//! `engine_rules::step::step(...)` using a DebugConfig with `tick_profile`.

use std::collections::BTreeMap;
use std::process::ExitCode;
use std::time::Instant;

use crate::cli::ProfileArgs;

// Phase names must match what `engine_rules/src/step.rs` emits to TickProfile.
const PHASE_NAMES: &[&str] = &[
    "mask_fill",
    "scoring",
    "action_select",
    "cascade_dispatch",
    "view_fold",
    "tick_end",
];

pub fn run_profile(args: ProfileArgs) -> ExitCode {
    println!(
        "profile: scenario={} ticks={}",
        args.scenario.display(),
        args.ticks,
    );
    println!("NOTE: running in stub mode (no engine dep). Synthetic timing samples generated.");
    println!();

    // Collect synthetic per-phase nanosecond samples by measuring no-op loops.
    let mut samples: BTreeMap<&'static str, Vec<u128>> = BTreeMap::new();

    for tick in 0..args.ticks {
        let _ = tick; // tick used only to drive iteration count
        for &phase in PHASE_NAMES {
            let t0 = Instant::now();
            // Synthetic "work": a tight loop whose iteration count varies by
            // phase, producing realistic-looking relative timings.
            let iters: u64 = match phase {
                "mask_fill"        => 800,
                "scoring"          => 600,
                "action_select"    => 200,
                "cascade_dispatch" => 1000,
                "view_fold"        => 400,
                "tick_end"         => 50,
                _                  => 100,
            };
            // Prevent the compiler from eliminating the loop entirely.
            let mut acc: u64 = 0;
            for i in 0..iters {
                acc = acc.wrapping_add(i);
            }
            std::hint::black_box(acc);

            let ns = t0.elapsed().as_nanos();
            samples.entry(phase).or_default().push(ns);
        }
    }

    // Compute and print per-phase statistics.
    println!(
        "{:<24} {:>8} {:>12} {:>12}",
        "phase", "samples", "p50_ns", "p99_ns"
    );
    println!("{}", "-".repeat(60));

    for (phase, mut v) in samples {
        v.sort_unstable();
        let n = v.len();
        let p50 = v[n / 2];
        let p99 = v[(n * 99 / 100).min(n - 1)];
        println!("{:<24} {:>8} {:>12} {:>12}", phase, n, p50, p99);
    }

    println!();
    println!(
        "profile: {} ticks × {} phases = {} samples total.",
        args.ticks,
        PHASE_NAMES.len(),
        args.ticks as usize * PHASE_NAMES.len(),
    );

    ExitCode::SUCCESS
}
