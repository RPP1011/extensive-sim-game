//! Stdlib math probe harness — drives `stdlib_math_probe_runtime` for
//! N ticks across two seeded runs, asserts per-slot count = TICKS
//! and byte-identical determinism (P5).
//!
//! Outcome (a) FULL FIRE for every advertised stdlib math + RNG
//! surface — ALL FIVE original gaps are now closed:
//!
//!   - Tier 1 math stdlib (`floor` / `ceil` / `round` / `log2` /
//!     `log10` / `abs`) — Gap #A close (8d7c2673) inlined `log10` to
//!     `log2(x) / log2(10.0)`.
//!   - Tier 2 spatial stdlib (`planar_distance` / `z_separation`) —
//!     Gap #B close (8d7c2673) injected the kernel-prelude shim.
//!   - Tier 3 `rng.coin()` — Gap #D close (8d7c2673) wraps the u32
//!     draw in `((per_agent_u32(...) & 1u) == 0u)`.
//!   - Tier 4 `rng.uniform` / `rng.gauss` — Gap #E close (this
//!     followup) emits per-purpose `f32(per_agent_u32(...)) /
//!     f32(4294967295u)` for Uniform, plus a Box-Muller pair-draw
//!     for Gauss using purposes 6 + 9.
//!   - Tier 5 `rng.uniform_int(lo, hi)` — Gap #C close (this
//!     followup) flipped the surface signature from `(i32, i32) ->
//!     i32` (unreachable from any `.sim` — DSL has no i32 source)
//!     to `(u32, u32) -> u32`, so bare-positive-literal pairs
//!     typecheck straight through.
//!   - Tier 6 `rng.action() % 4u` bucket emit — closed by
//!     stochastic_probe Gaps #1/#2/#3 (2026-05-04).
//!
//! The catch_unwind wrappers below surface any regression as
//! OUTCOME (b) WGSL VALIDATION FAILED.
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
            "stdlib_math_probe_app: OUTCOME = (a) FULL FIRE — every advertised stdlib \
             math + RNG surface wired end-to-end (ALL FIVE original gaps closed):\n  \
             1. Tier 1 math stdlib (floor/ceil/round/log2/log10/abs) lowers + emits + \
                FULL-validator-clean (log10 via Gap #A inline rewrite to \
                log2(x)/log2(10.0))\n  \
             2. Tier 2 spatial stdlib (planar_distance / z_separation) lowers + emits + \
                FULL-validator-clean (Gap #B prelude-shim inject in cg/emit/program.rs)\n  \
             3. Tier 3 rng.coin() emits ((per_agent_u32(...) & 1u) == 0u) \
                (Gap #D bool-from-u32)\n  \
             4. Tier 4 rng.uniform/rng.gauss emit per-purpose f32 conversion at the \
                CgExpr::Rng site so the surrounding f32 arithmetic is concretely-typed \
                (Gap #E close, this commit; Box-Muller pair-draw uses purposes 6 + 9)\n  \
             5. Tier 5 rng.uniform_int(lo, hi) lowers cleanly with the new (u32, u32) \
                -> u32 signature — bare positive literals typecheck straight through \
                (Gap #C close, this commit)\n  \
             6. Tier 6 bucket emit via rng.action() % 4 fires every tick (proven \
                pure-by-stochastic-probe Action purpose)\n  \
             7. P5 determinism holds (byte-identical sampled_count across two runs)\n  \
             8. Per-slot count = TICKS = {} on every slot",
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
