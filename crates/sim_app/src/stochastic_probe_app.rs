//! Stochastic probe harness — drives `stochastic_probe_runtime` for N
//! ticks and reports the observed per-agent `activations` view AND
//! verifies determinism (P5: same seed → byte-identical observable).
//!
//! ## Predicted observable shapes
//!
//! ### (a) FULL FIRE — `rng.action()` lowering wires through cleanly
//!
//! With AGENT_COUNT=32, TICKS=1000, threshold=30%, log_amount=1.0:
//!
//!   - `activations[N]` ≈ TICKS × 0.30 = 300, ±5% per slot under
//!      uniform PCG draws (i.e. each slot in [285, 315])
//!   - sum ≈ 32 × 300 = 9600
//!   - **Determinism**: two independent `StochasticProbeState`
//!     instances with the same seed must produce byte-identical
//!     activations buffers across all 32 slots. P5 requires the per-
//!     agent stream to be a pure function of (seed, agent_id, tick,
//!     purpose).
//!
//! ### (b) WGSL VALIDATION FAILED — most likely outcome today
//!
//! The `physics_MaybeFire.wgsl` body contains a `per_agent_u32(seed,
//! agent_id, tick, "action")` call but neither `seed` (kernel
//! preamble local) nor `per_agent_u32` (WGSL prelude function) nor
//! the string-literal purpose tag is bound. Naga rejects the module
//! at `create_shader_module` time. The harness catches the panic and
//! reports the gap punch list.
//!
//! Discovery doc: `docs/superpowers/notes/2026-05-04-stochastic_probe.md`.

use engine::CompiledSim;
use std::panic::{catch_unwind, AssertUnwindSafe};
use stochastic_probe_runtime::StochasticProbeState;

const SEED: u64 = 0x57_0CA571_C_DEC0DE5;
const AGENT_COUNT: u32 = 32;
const TICKS: u64 = 1000;
/// 30% activation gate from the .sim — held in sync with
/// `assets/sim/stochastic_probe.sim::config.stochastic.
/// activation_threshold_q100u`.
const PROBABILITY: f64 = 0.30;
/// Per-slot tolerance for the binomial distribution. With p=0.30,
/// T=1000, std-dev ≈ √(1000 × 0.3 × 0.7) ≈ 14.5; ±5% of mean (= ±15)
/// is roughly 1σ, so most slots should fit, but the harness reports
/// all slots so any miss is visible.
const PER_SLOT_TOLERANCE_FRACTION: f64 = 0.05;

fn main() {
    println!(
        "stochastic_probe_app: starting — seed=0x{:016X} agents={} ticks={} prob={:.2}",
        SEED, AGENT_COUNT, TICKS, PROBABILITY,
    );

    // Try run #1 inside catch_unwind — naga validation rejection
    // surfaces as a panic deep inside `create_shader_module`. Without
    // unwinding the harness can't print OUTCOME (b).
    let run1 = catch_unwind(AssertUnwindSafe(|| run_once()));
    match run1 {
        Err(_) => {
            print_outcome_b_gap(
                "create_shader_module / first dispatch panicked — naga rejected the emitted WGSL.",
            );
            return;
        }
        Ok(activations1) => {
            // Run #2 — same seed, fresh state. Must produce byte-
            // identical activations to verify P5 determinism.
            let run2 = catch_unwind(AssertUnwindSafe(|| run_once()));
            match run2 {
                Err(_) => {
                    print_outcome_b_gap(
                        "second run panicked after first run succeeded — non-deterministic GPU \
                         init? (unexpected)",
                    );
                    return;
                }
                Ok(activations2) => {
                    report_outcome_a(&activations1, &activations2);
                }
            }
        }
    }
}

fn run_once() -> Vec<f32> {
    let mut sim = StochasticProbeState::new(SEED, AGENT_COUNT);
    for _ in 0..TICKS {
        sim.step();
    }
    sim.activations().to_vec()
}

fn report_outcome_a(activations1: &[f32], activations2: &[f32]) {
    println!(
        "stochastic_probe_app: finished both runs — activations1.len()={} activations2.len()={}",
        activations1.len(),
        activations2.len(),
    );

    // ----- Determinism check (P5) -----
    let det_ok = activations1 == activations2;
    if det_ok {
        println!(
            "stochastic_probe_app: DETERMINISM OK — both runs produced byte-identical activations \
             (P5: per_agent_u32(seed, agent, tick, purpose) is a pure fn).",
        );
    } else {
        let mut first_mismatch = None;
        for (i, (a, b)) in activations1.iter().zip(activations2.iter()).enumerate() {
            if a != b {
                first_mismatch = Some((i, *a, *b));
                break;
            }
        }
        println!(
            "stochastic_probe_app: DETERMINISM FAIL — runs diverged. First mismatch: {:?}. \
             P5 violation: same seed must reproduce.",
            first_mismatch,
        );
    }

    // ----- Distribution check -----
    let expected_mean = TICKS as f64 * PROBABILITY;
    let tolerance = expected_mean * PER_SLOT_TOLERANCE_FRACTION;

    let (min, mean, max, sum, zero_slots) = stats(activations1);
    let nonzero_slots = activations1.len() - zero_slots;
    println!(
        "stochastic_probe_app: activations readback (run #1) — min={:.3} mean={:.3} max={:.3} sum={:.3}",
        min, mean, max, sum,
    );
    println!(
        "stochastic_probe_app: nonzero slots: {}/{} (fraction = {:.1}%)",
        nonzero_slots,
        activations1.len(),
        (nonzero_slots as f32) / (activations1.len().max(1) as f32) * 100.0,
    );
    println!(
        "stochastic_probe_app: expected per-slot ≈ {:.0} (TICKS × p), tolerance ±{:.0} \
         ({:.0}%)",
        expected_mean,
        tolerance,
        PER_SLOT_TOLERANCE_FRACTION * 100.0,
    );

    let mut in_tol = 0usize;
    let mut max_dev: f64 = 0.0;
    for (n, &got) in activations1.iter().enumerate() {
        let dev = (got as f64 - expected_mean).abs();
        if dev <= tolerance {
            in_tol += 1;
        }
        if dev > max_dev {
            max_dev = dev;
            let _ = n;
        }
    }
    println!(
        "stochastic_probe_app: per-slot in-tolerance: {}/{} (max deviation = {:.3})",
        in_tol,
        activations1.len(),
        max_dev,
    );

    let preview: Vec<f32> = activations1.iter().take(8).copied().collect();
    println!(
        "stochastic_probe_app: preview activations[0..8] = {:?}",
        preview,
    );

    // OUTCOME classification.
    let dist_ok = max == 0.0_f32 || (in_tol as f64) >= 0.95 * (activations1.len() as f64);
    if det_ok && in_tol == activations1.len() {
        println!(
            "stochastic_probe_app: OUTCOME = (a) FULL FIRE — both surfaces verified end-to-end:\n  \
             1. rng.action() lowers + emits per_agent_u32(seed, agent_id, tick, \"action\")\n  \
             2. P5 determinism holds (byte-identical observables across two runs)\n  \
             3. Per-slot count matches T × p = {:.0} within ±{:.0}% on every slot",
            expected_mean,
            PER_SLOT_TOLERANCE_FRACTION * 100.0,
        );
    } else if !det_ok {
        println!(
            "stochastic_probe_app: OUTCOME = (b) DETERMINISM FAIL — distribution may be fine but \
             two runs diverge. The RNG primitive isn't pure-by-(seed, agent, tick, purpose) at \
             the WGSL level. See discovery doc for root-cause.",
        );
    } else if max == 0.0_f32 {
        println!(
            "stochastic_probe_app: OUTCOME = (b) NO FIRE — every slot stayed at 0.0. The physics \
             body's emit never lands; either the rng-derived predicate always returns false, or \
             the IfStmt's gate path is broken with rng.action() on the LHS.",
        );
    } else if !dist_ok {
        println!(
            "stochastic_probe_app: OUTCOME = (b) PARTIAL FIRE — distribution doesn't converge to \
             T × p. {}/{} slots in tolerance; max deviation = {:.3}. Either the RNG isn't \
             uniformly distributed at the WGSL emit, or the modulo/comparison wired wrong.",
            in_tol,
            activations1.len(),
            max_dev,
        );
    }
}

fn print_outcome_b_gap(headline: &str) {
    println!("stochastic_probe_app: OUTCOME = (b) WGSL VALIDATION FAILED");
    println!("stochastic_probe_app:   headline: {}", headline);
    println!("stochastic_probe_app: ");
    println!("stochastic_probe_app: gap punch list (severity HIGH for each — `rng.*` is");
    println!("stochastic_probe_app:   non-functional in compiler-emitted GPU bodies until");
    println!("stochastic_probe_app:   ALL THREE are closed):");
    println!("stochastic_probe_app:");
    println!(
        "stochastic_probe_app:   #1  WGSL emit body references bare `seed` identifier."
    );
    println!(
        "stochastic_probe_app:       file: crates/dsl_compiler/src/cg/emit/wgsl_body.rs:944"
    );
    println!(
        "stochastic_probe_app:       Generated body line: `per_agent_u32(seed, agent_id, tick, \"action\")`"
    );
    println!(
        "stochastic_probe_app:       Gap: `seed` is not bound by `thread_indexing_preamble`,"
    );
    println!(
        "stochastic_probe_app:       not a field of any per-rule cfg uniform, not a global const."
    );
    println!("stochastic_probe_app:");
    println!(
        "stochastic_probe_app:   #2  WGSL emit calls `per_agent_u32(...)` but no WGSL impl"
    );
    println!(
        "stochastic_probe_app:       is emitted into any kernel module. The host Rust"
    );
    println!(
        "stochastic_probe_app:       function lives at crates/engine/src/rng.rs:50 (uses ahash)"
    );
    println!(
        "stochastic_probe_app:       — that algorithm doesn't translate to WGSL. A WGSL-side"
    );
    println!(
        "stochastic_probe_app:       hash (e.g. PCG-XSH-RR over the 4-tuple) needs to be"
    );
    println!(
        "stochastic_probe_app:       emitted as a kernel prelude alongside any rng-touching kernel."
    );
    println!("stochastic_probe_app:");
    println!(
        "stochastic_probe_app:   #3  WGSL emit passes `\"action\"` (string literal) as the"
    );
    println!(
        "stochastic_probe_app:       purpose tag. WGSL has no string type. The purpose must"
    );
    println!(
        "stochastic_probe_app:       be encoded as a u32 constant (hash of the bytes, or a"
    );
    println!(
        "stochastic_probe_app:       small enum: action=1, sample=2, shuffle=3, conception=4)."
    );
    println!("stochastic_probe_app:");
    println!(
        "stochastic_probe_app:   #4  Spec drift (LOW) — docs/spec/dsl.md:922 advertises"
    );
    println!(
        "stochastic_probe_app:       `rng.uniform(lo, hi) -> f32`, `rng.coin() -> bool`,"
    );
    println!(
        "stochastic_probe_app:       `rng.gauss(mu, sigma) -> f32`, `rng.uniform_int(lo, hi)`."
    );
    println!(
        "stochastic_probe_app:       The CG lowering pass at"
    );
    println!(
        "stochastic_probe_app:       crates/dsl_compiler/src/cg/lower/expr.rs:2532-2544"
    );
    println!(
        "stochastic_probe_app:       only routes `action` / `sample` / `shuffle` / `conception`"
    );
    println!(
        "stochastic_probe_app:       (all nullary, all u32-typed). Spec-advertised f32/bool"
    );
    println!(
        "stochastic_probe_app:       surfaces lower to UnsupportedNamespaceCall (test pin"
    );
    println!(
        "stochastic_probe_app:       at expr.rs:4029)."
    );
    println!("stochastic_probe_app:");
    println!(
        "stochastic_probe_app: See docs/superpowers/notes/2026-05-04-stochastic_probe.md."
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
