//! `stress_agent_count_app` — driver binary for Fixture A from
//! `docs/superpowers/plans/2026-05-09-stress-test-runtime-ceilings.md`.
//!
//! Reads `agent_cap` from `argv[1]` (and an optional `tick_budget` from
//! `argv[2]`, defaulting to 100). Wraps every step in
//! `std::panic::catch_unwind` so panics surface as the breakpoint
//! instead of being lost in the process abort. Emits NDJSON to stdout:
//! one record per tick + one final summary record.
//!
//! ## Output format
//!
//! Per-tick records:
//!   `{"agent_cap": <u32>, "tick": <u64>, "wall_clock_us": <u128>,
//!     "peak_mem_mb": <u64>, "dispatcher_kernel_bytes": <usize>}`
//!
//! Final summary record:
//!   `{"summary": true, "agent_cap": <u32>, "ticks_completed": <u64>,
//!     "median_us": <u128>, "p99_us": <u128>,
//!     "panic": null | "<msg>"}`
//!
//! All NDJSON records are flushed line-by-line so a panicking child
//! still leaves a valid trail in the captured stdout.

use std::io::Write;
use std::panic::AssertUnwindSafe;
use std::time::Instant;

use engine::sim_trait::CompiledSim;
use stress_agent_count_runtime::{
    StressAgentCountState, DISPATCHER_KERNEL_BYTES, STRESS_SEED,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let agent_cap: u32 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            eprintln!(
                "usage: {} <agent_cap> [tick_budget]",
                args.first().map(|s| s.as_str()).unwrap_or("stress_agent_count_app"),
            );
            std::process::exit(2);
        });
    let tick_budget: u32 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    eprintln!(
        "[stress_agent_count_app] starting agent_cap={} tick_budget={} seed=0x{:016X}",
        agent_cap, tick_budget, STRESS_SEED,
    );

    let mut stdout = std::io::stdout().lock();
    let mut wall_us_samples: Vec<u128> = Vec::with_capacity(tick_budget as usize);
    let mut ticks_completed: u64 = 0;
    let mut panic_msg: Option<String> = None;

    // -- Construction. Wrap in catch_unwind so OOM / GPU-init panic at
    //    construction time gets recorded too.
    let state_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        StressAgentCountState::new(STRESS_SEED, agent_cap)
    }));
    let mut state = match state_result {
        Ok(s) => s,
        Err(payload) => {
            let msg = panic_payload_msg(payload.as_ref());
            eprintln!("[stress_agent_count_app] PANIC at construction: {msg}");
            emit_summary(&mut stdout, agent_cap, 0, &[], Some(msg));
            return;
        }
    };

    // -- Tick loop. Panic-catch each individual step so a failing
    //    tick still flushes the per-tick records that succeeded plus a
    //    summary with the panic message.
    for tick_idx in 0..tick_budget {
        let start = Instant::now();
        let step_result = std::panic::catch_unwind(AssertUnwindSafe(|| state.step()));
        let elapsed_us = start.elapsed().as_micros();

        match step_result {
            Ok(()) => {
                ticks_completed += 1;
                wall_us_samples.push(elapsed_us);
                let peak_mem_mb = peak_rss_mb();
                emit_per_tick(
                    &mut stdout,
                    agent_cap,
                    tick_idx as u64,
                    elapsed_us,
                    peak_mem_mb,
                    DISPATCHER_KERNEL_BYTES,
                );
                // Compiler debug mode (D3) per-kernel breakdown — one
                // NDJSON line per kernel, every 10 ticks. Empty when
                // the adapter doesn't expose TIMESTAMP_QUERY (the
                // sim still runs; we just don't emit per-kernel
                // attribution lines on that host).
                if tick_idx % 10 == 0 {
                    for k in state.last_kernel_timings() {
                        emit_per_kernel(
                            &mut stdout,
                            tick_idx as u64,
                            &k.kernel,
                            k.wall_ns,
                            // Per-kernel host↔GPU bytes accounting
                            // requires runtime-side wiring against
                            // MemTrafficTable; leave at 0 for now —
                            // the per-tick wall times are the
                            // dominant attribution surface. Phase 1
                            // ships the slot; later subagents wire
                            // the byte counts.
                            0,
                            0,
                        );
                    }
                }
            }
            Err(payload) => {
                let msg = panic_payload_msg(payload.as_ref());
                eprintln!(
                    "[stress_agent_count_app] PANIC at tick {}: {msg}",
                    tick_idx,
                );
                panic_msg = Some(msg);
                break;
            }
        }
    }

    emit_summary(
        &mut stdout,
        agent_cap,
        ticks_completed,
        &wall_us_samples,
        panic_msg,
    );
}

fn emit_per_tick(
    w: &mut impl Write,
    agent_cap: u32,
    tick: u64,
    wall_us: u128,
    peak_mem_mb: u64,
    dispatcher_kernel_bytes: usize,
) {
    let line = format!(
        "{{\"agent_cap\": {}, \"tick\": {}, \"wall_clock_us\": {}, \"peak_mem_mb\": {}, \"dispatcher_kernel_bytes\": {}}}\n",
        agent_cap, tick, wall_us, peak_mem_mb, dispatcher_kernel_bytes,
    );
    let _ = w.write_all(line.as_bytes());
    let _ = w.flush();
}

/// Per-kernel breakdown line emitted from the compiler debug-mode
/// (D3) instrumentation surface. Format matches the plan doc:
///
/// ```json
/// {"tick": 10, "kernel": "mask_pulse", "wall_ns": 12345,
///  "host_to_gpu_bytes": 4096, "gpu_to_host_bytes": 0}
/// ```
fn emit_per_kernel(
    w: &mut impl Write,
    tick: u64,
    kernel: &str,
    wall_ns: u64,
    host_to_gpu_bytes: u64,
    gpu_to_host_bytes: u64,
) {
    let line = format!(
        "{{\"tick\": {}, \"kernel\": \"{}\", \"wall_ns\": {}, \"host_to_gpu_bytes\": {}, \"gpu_to_host_bytes\": {}}}\n",
        tick,
        json_escape(kernel),
        wall_ns,
        host_to_gpu_bytes,
        gpu_to_host_bytes,
    );
    let _ = w.write_all(line.as_bytes());
    let _ = w.flush();
}

fn emit_summary(
    w: &mut impl Write,
    agent_cap: u32,
    ticks_completed: u64,
    wall_us_samples: &[u128],
    panic_msg: Option<String>,
) {
    let median_us = percentile(wall_us_samples, 0.50);
    let p99_us = percentile(wall_us_samples, 0.99);
    let panic_field = match &panic_msg {
        Some(m) => format!("\"{}\"", json_escape(m)),
        None => "null".to_string(),
    };
    let line = format!(
        "{{\"summary\": true, \"agent_cap\": {}, \"ticks_completed\": {}, \"median_us\": {}, \"p99_us\": {}, \"panic\": {}}}\n",
        agent_cap, ticks_completed, median_us, p99_us, panic_field,
    );
    let _ = w.write_all(line.as_bytes());
    let _ = w.flush();
}

fn percentile(samples: &[u128], q: f64) -> u128 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort();
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn panic_payload_msg(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<panic with non-string payload>".to_string()
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Best-effort peak resident set size in MB. Reads /proc/self/status's
/// `VmHWM` line on Linux; returns 0 on other platforms (the harness
/// only runs on Linux per env).
fn peak_rss_mb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("VmHWM:") {
                    let kb: u64 = rest
                        .trim()
                        .split_whitespace()
                        .next()
                        .and_then(|w| w.parse().ok())
                        .unwrap_or(0);
                    return kb / 1024;
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}
