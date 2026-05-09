//! Driver binary for the cast-density stress sweep (task #238).
//!
//! Usage:
//!
//! ```bash
//! ./target/release/stress_cast_density_app <agent_cap> [tick_budget]
//! ```
//!
//! Emits one NDJSON line per tick of the form:
//!
//! ```json
//! {"agent_cap": 1000, "tick": 0, "wall_clock_us": 1234,
//!  "ring_high_water": 65536, "ring_overflowed": true,
//!  "spread_sort_us": 1100}
//! ```
//!
//! Plus a final summary line:
//!
//! ```json
//! {"summary": true, "agent_cap": 1000, "ticks_completed": 100,
//!  "median_us": 1234, "p99_us": 1456, "max_ring_high_water": 65536,
//!  "first_overflow_at_tick": 0, "panic": null}
//! ```
//!
//! Wraps the per-cap stress run in `std::panic::catch_unwind` (P10) so
//! a single failing cap doesn't abort the parent sweep harness.

use std::env;
use std::panic;
use stress_cast_density_runtime::{run_stress, RING_SLOT_CAP};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!(
            "usage: {} <agent_cap> [tick_budget=100]",
            args.first().map(String::as_str).unwrap_or("stress_cast_density_app"),
        );
        std::process::exit(2);
    }
    let agent_cap: u32 = args[1]
        .parse()
        .expect("agent_cap must parse as u32");
    let tick_budget: u32 = if args.len() == 3 {
        args[2].parse().expect("tick_budget must parse as u32")
    } else {
        100
    };

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        run_stress(agent_cap, tick_budget)
    }));

    match result {
        Ok(Some(metrics)) => {
            for tick in &metrics.per_tick {
                println!(
                    "{{\"agent_cap\":{},\"tick\":{},\"wall_clock_us\":{},\
                      \"ring_high_water\":{},\"ring_overflowed\":{},\
                      \"spread_sort_us\":{}}}",
                    metrics.agent_cap,
                    tick.tick,
                    tick.wall_clock_us,
                    tick.ring_high_water,
                    tick.ring_overflowed,
                    tick.spread_sort_us,
                );
                // Compiler debug mode (D3) per-kernel breakdown — one
                // NDJSON line per kernel, every 10 ticks. Empty when
                // the adapter doesn't expose TIMESTAMP_QUERY (the
                // kernel_timings vec is empty in that case).
                if tick.tick % 10 == 0 {
                    for (kernel, wall_ns) in &tick.kernel_timings {
                        println!(
                            "{{\"tick\":{},\"kernel\":\"{}\",\"wall_ns\":{},\
                              \"host_to_gpu_bytes\":0,\"gpu_to_host_bytes\":0}}",
                            tick.tick,
                            kernel.replace('\\', "\\\\").replace('"', "\\\""),
                            wall_ns,
                        );
                    }
                }
            }
            let median_us = metrics.median_us();
            let p99_us = metrics.p99_us();
            let median_sort = metrics.median_spread_sort_us();
            let first_overflow = match metrics.first_overflow_at_tick {
                Some(t) => format!("{t}"),
                None => "null".to_string(),
            };
            println!(
                "{{\"summary\":true,\"agent_cap\":{},\"ticks_completed\":{},\
                  \"median_us\":{},\"p99_us\":{},\"max_ring_high_water\":{},\
                  \"ring_slot_cap\":{},\"first_overflow_at_tick\":{},\
                  \"median_spread_sort_us\":{},\"panic\":null}}",
                metrics.agent_cap,
                metrics.ticks_completed,
                median_us,
                p99_us,
                metrics.max_ring_high_water,
                RING_SLOT_CAP,
                first_overflow,
                median_sort,
            );
        }
        Ok(None) => {
            // No GPU adapter — emit a skip summary.
            println!(
                "{{\"summary\":true,\"agent_cap\":{},\"ticks_completed\":0,\
                  \"median_us\":0,\"p99_us\":0,\"max_ring_high_water\":0,\
                  \"ring_slot_cap\":{},\"first_overflow_at_tick\":null,\
                  \"median_spread_sort_us\":0,\
                  \"panic\":\"no_wgpu_adapter\"}}",
                agent_cap, RING_SLOT_CAP,
            );
            std::process::exit(0);
        }
        Err(panic_box) => {
            // Recover the panic message if it's a String-ish payload.
            let panic_msg: String = if let Some(s) = panic_box.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = panic_box.downcast_ref::<String>() {
                s.clone()
            } else {
                "non-string panic payload".to_string()
            };
            // JSON-escape the panic msg minimally: backslash + quote +
            // newline. The harness cares about the agent_cap + the
            // truthiness of `panic`; the exact bytes are diagnostic.
            let escaped = panic_msg
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n");
            println!(
                "{{\"summary\":true,\"agent_cap\":{},\"ticks_completed\":0,\
                  \"median_us\":0,\"p99_us\":0,\"max_ring_high_water\":0,\
                  \"ring_slot_cap\":{},\"first_overflow_at_tick\":null,\
                  \"median_spread_sort_us\":0,\"panic\":\"{}\"}}",
                agent_cap, RING_SLOT_CAP, escaped,
            );
            std::process::exit(1);
        }
    }
}
