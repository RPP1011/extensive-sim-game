//! Stdlib math probe harness — drives `stdlib_math_probe_runtime` for
//! N ticks across two seeded runs, asserts per-slot count = TICKS
//! and byte-identical determinism (P5).
//!
//! Outcome (a) FULL FIRE for every RETAINED surface: Tier 1 math
//! stdlib (`floor` / `ceil` / `round` / `log2` / `log10` / `abs`),
//! Tier 2 spatial stdlib (`planar_distance` / `z_separation`), Tier 3
//! `rng.coin()`, and Tier 4 `rng.action() % 4u` bucket emit. Three
//! of the five originally-surfaced gaps are closed (Gaps #A, #B, #D,
//! 2026-05-04); two remain (Gap #C `rng.uniform_int`, Gap #E
//! `rng.uniform`/`rng.gauss`) and stay as commented-out `let`s in
//! the .sim with citations to the responsible compiler arms. The
//! catch_unwind wrappers below surface any regression that
//! re-introduces a still-open gap surface as OUTCOME (b) WGSL
//! VALIDATION FAILED.
//!
//! Discovery doc: `docs/superpowers/notes/2026-05-04-stdlib_math_probe.md`.

use engine::CompiledSim;
use std::panic::{catch_unwind, AssertUnwindSafe};
use stdlib_math_probe_runtime::StdlibMathProbeState;

const SEED: u64 = 0x57D11B_5A77F005_u64;
const AGENT_COUNT: u32 = 32;
const TICKS: u64 = 100;
/// Expected per-slot count under unconditional emit. The view fold
/// adds `log_amount = 1.0` once per Sampled event.
const EXPECTED_PER_SLOT: f32 = TICKS as f32;

fn main() {
    println!(
        "stdlib_math_probe_app: starting — seed=0x{:016X} agents={} ticks={}",
        SEED, AGENT_COUNT, TICKS,
    );

    // Run #1 inside catch_unwind — if the probe re-introduces a
    // naga-rejecting surface (e.g. a regression on the omitted-gap
    // surfaces), `create_shader_module` panics deep inside the first
    // dispatch_* call.
    let run1 = catch_unwind(AssertUnwindSafe(run_once));
    match run1 {
        Err(_) => {
            print_outcome_b_gap(
                "create_shader_module / first dispatch panicked — naga rejected the emitted WGSL.",
            );
            return;
        }
        Ok(counts1) => {
            // Run #2 — same seed, fresh state. Must produce byte-
            // identical sampled_count to verify P5 determinism.
            let run2 = catch_unwind(AssertUnwindSafe(run_once));
            match run2 {
                Err(_) => {
                    print_outcome_b_gap(
                        "second run panicked after first run succeeded — non-deterministic GPU \
                         init? (unexpected)",
                    );
                    return;
                }
                Ok(counts2) => {
                    report_outcome(&counts1, &counts2);
                }
            }
        }
    }
}

fn run_once() -> Vec<f32> {
    let mut sim = StdlibMathProbeState::new(SEED, AGENT_COUNT);
    for _ in 0..TICKS {
        sim.step();
    }
    sim.sampled_count().to_vec()
}

fn report_outcome(counts1: &[f32], counts2: &[f32]) {
    println!(
        "stdlib_math_probe_app: finished both runs — counts1.len()={} counts2.len()={}",
        counts1.len(),
        counts2.len(),
    );

    // ----- Determinism check (P5) -----
    let det_ok = counts1 == counts2;
    if det_ok {
        println!(
            "stdlib_math_probe_app: DETERMINISM OK — both runs produced byte-identical \
             sampled_count (P5: per_agent_u32(seed, agent, tick, purpose) is a pure fn).",
        );
    } else {
        let mut first_mismatch = None;
        for (i, (a, b)) in counts1.iter().zip(counts2.iter()).enumerate() {
            if a != b {
                first_mismatch = Some((i, *a, *b));
                break;
            }
        }
        println!(
            "stdlib_math_probe_app: DETERMINISM FAIL — runs diverged. First mismatch: {:?}. \
             P5 violation: same seed must reproduce.",
            first_mismatch,
        );
    }

    // ----- Per-slot count check -----
    let (min, mean, max, sum, zero_slots) = stats(counts1);
    let nonzero_slots = counts1.len() - zero_slots;
    println!(
        "stdlib_math_probe_app: sampled_count readback (run #1) — min={:.3} mean={:.3} \
         max={:.3} sum={:.3}",
        min, mean, max, sum,
    );
    println!(
        "stdlib_math_probe_app: nonzero slots: {}/{} (fraction = {:.1}%)",
        nonzero_slots,
        counts1.len(),
        (nonzero_slots as f32) / (counts1.len().max(1) as f32) * 100.0,
    );
    println!(
        "stdlib_math_probe_app: expected per-slot count = TICKS = {} (unconditional emit)",
        EXPECTED_PER_SLOT as u32,
    );

    let mut exact_matches = 0usize;
    let mut max_dev: f64 = 0.0;
    for &got in counts1.iter() {
        let dev = (got as f64 - EXPECTED_PER_SLOT as f64).abs();
        if dev < 0.5 {
            exact_matches += 1;
        }
        if dev > max_dev {
            max_dev = dev;
        }
    }
    println!(
        "stdlib_math_probe_app: per-slot exact matches: {}/{} (max_dev = {:.3})",
        exact_matches,
        counts1.len(),
        max_dev,
    );

    let preview: Vec<f32> = counts1.iter().take(8).copied().collect();
    println!(
        "stdlib_math_probe_app: preview sampled_count[0..8] = {:?}",
        preview,
    );

    // OUTCOME classification.
    if det_ok && exact_matches == counts1.len() {
        println!(
            "stdlib_math_probe_app: OUTCOME = (a) FULL FIRE — every retained surface \
             wired end-to-end:\n  \
             1. Tier 1 math stdlib (floor/ceil/round/log2/log10/abs) lowers + emits + \
                FULL-validator-clean (log10 via Gap #A inline rewrite to \
                log2(x)/log2(10.0))\n  \
             2. Tier 2 spatial stdlib (planar_distance / z_separation) lowers + emits + \
                FULL-validator-clean (Gap #B prelude-shim inject in cg/emit/program.rs)\n  \
             3. Tier 3 rng.coin() emits ((per_agent_u32(...) & 1u) == 0u) (Gap #D bool-from-u32)\n  \
             4. Tier 4 bucket emit via rng.action() % 4 fires every tick (proven \
                pure-by-stochastic-probe Action purpose)\n  \
             5. P5 determinism holds (byte-identical sampled_count across two runs)\n  \
             6. Per-slot count = TICKS = {} on every slot\n  \
             — Gaps #C (rng.uniform_int — needs i32 source) and #E (rng.uniform / \
             rng.gauss — needs per-purpose abstract-type-clean conversion) remain open",
            EXPECTED_PER_SLOT as u32,
        );
    } else if !det_ok {
        println!(
            "stdlib_math_probe_app: OUTCOME = (b) DETERMINISM FAIL — runs diverged. The \
             RNG primitive isn't pure-by-(seed, agent, tick, purpose) at the WGSL level. \
             See discovery doc.",
        );
    } else if max == 0.0 {
        println!(
            "stdlib_math_probe_app: OUTCOME = (b) NO FIRE — every slot stayed at 0.0. \
             The physics body's emit never lands; either the kernel didn't dispatch or \
             the fold's tag-filter dropped all events.",
        );
    } else {
        println!(
            "stdlib_math_probe_app: OUTCOME = (b) PARTIAL FIRE — per-slot count doesn't \
             match TICKS. {}/{} slots match exactly; max_dev = {:.3}.",
            exact_matches,
            counts1.len(),
            max_dev,
        );
    }
}

fn print_outcome_b_gap(headline: &str) {
    println!("stdlib_math_probe_app: OUTCOME = (b) WGSL VALIDATION FAILED");
    println!("stdlib_math_probe_app:   headline: {}", headline);
    println!("stdlib_math_probe_app: ");
    println!(
        "stdlib_math_probe_app: see docs/superpowers/notes/2026-05-04-stdlib_math_probe.md \
         for the gap punch list (Gaps #A–#E, four hard rejects + one silent semantic gap)."
    );
}

fn stats(v: &[f32]) -> (f32, f32, f32, f32, usize) {
    let mut min = f32::INFINITY;
    let mut max = 0.0_f32;
    let mut sum = 0.0_f32;
    let mut zero = 0usize;
    for &x in v {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += x;
        if x == 0.0 {
            zero += 1;
        }
    }
    let mean = if v.is_empty() { 0.0 } else { sum / v.len() as f32 };
    (min, mean, max, sum, zero)
}
