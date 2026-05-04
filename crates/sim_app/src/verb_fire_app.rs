//! Verb-fire probe harness — drives `verb_probe_runtime` for N ticks
//! and reports observed per-agent faith counts.
//!
//! ## Expected (cascade fires end-to-end, gap-free)
//!
//! With `faith_step = 1.0`, `agent_count = 32`, `ticks = 100`,
//! every alive agent's `verb_Pray` row wins argmax every tick;
//! every tick produces:
//!   - 32 `ActionSelected` events (scoring kernel)
//!   - 32 `PrayCompleted` events (chronicle physics rule)
//! After 100 ticks: faith[i] = 100.0 for every i.
//!
//! ## Observed (with Gaps #1–#4 closed)
//!
//! The verb cascade now fires end-to-end:
//!   - Gap #1 (compiler `fresh_local_after` LocalRef collision) →
//!     closed by stateful binder allocation (`bb176401`).
//!   - Gap #2 (schedule fusion eating chronicle→fold boundary) →
//!     closed by no-fuse rule for producer + same-event consumer.
//!   - Gap #3 (scoring kernel ignored mask bitmap) → closed by
//!     compiler emitting the bitmap gate.
//!   - Gap #4 (runtime did not dispatch the chronicle physics rule
//!     between scoring and fold) → closed by `verb_probe_runtime`
//!     adding the chronicle dispatch + sentinel pre-stamp of the
//!     ring buffer (so workgroup-rounding threads beyond the actual
//!     scoring tail don't spuriously emit).
//!
//! Result: every agent slot accumulates `faith_step = 1.0` per tick,
//! so after 100 ticks `faith[i] = 100.0` for all i — matching the
//! analytical observable.
//!
//! Discovery write-up: `docs/superpowers/notes/2026-05-04-verb-fire-probe.md`.

use engine::CompiledSim;
use verb_probe_runtime::VerbProbeState;

const SEED: u64 = 0xDEAD_BEEF_FACE_CAFE;
const AGENT_COUNT: u32 = 32;
const TICKS: u64 = 100;
const FAITH_STEP: f32 = 1.0;

fn main() {
    let mut sim = VerbProbeState::new(SEED, AGENT_COUNT);
    println!(
        "verb_fire_app: starting — seed=0x{:016X} agents={} ticks={}",
        SEED, AGENT_COUNT, TICKS,
    );

    for _ in 0..TICKS {
        sim.step();
    }

    let faith = sim.faith().to_vec();
    println!(
        "verb_fire_app: finished — final tick={} agents={} faith.len()={}",
        sim.tick(),
        sim.agent_count(),
        faith.len(),
    );

    // Expected analytical observable: faith[i] = TICKS * FAITH_STEP
    let expected_per_slot = (TICKS as f32) * FAITH_STEP;

    let mut min = f32::INFINITY;
    let mut max = 0.0_f32;
    let mut sum = 0.0_f32;
    let mut zero_slots = 0usize;
    for &v in &faith {
        min = min.min(v);
        max = max.max(v);
        sum += v;
        if v == 0.0 {
            zero_slots += 1;
        }
    }
    let mean = sum / faith.len() as f32;
    let nonzero_slots = faith.len() - zero_slots;
    let observed_fraction = (nonzero_slots as f32) / (faith.len() as f32);

    println!(
        "verb_fire_app: faith readback — min={:.3} mean={:.3} max={:.3}",
        min, mean, max,
    );
    println!(
        "verb_fire_app: nonzero slots: {}/{} (fraction = {:.3}%)",
        nonzero_slots,
        faith.len(),
        observed_fraction * 100.0,
    );
    println!(
        "verb_fire_app: expected per-slot value (full cascade): {:.3} \
         (= TICKS={} × faith_step={:.3})",
        expected_per_slot, TICKS, FAITH_STEP,
    );

    // Diagnostic classification — surface the OUTCOME explicitly so
    // the run is human-greppable for either success mode (a) or
    // partial / no-fire mode (b).
    if min >= expected_per_slot * 0.99 {
        println!(
            "verb_fire_app: OUTCOME = (a) FULL FIRE — every slot ≈ expected value",
        );
    } else if max == 0.0 {
        println!(
            "verb_fire_app: OUTCOME = (b) NO FIRE — every slot stayed at 0.0; \
             see docs/superpowers/notes/2026-05-04-verb-fire-probe.md for the gap chain",
        );
    } else {
        println!(
            "verb_fire_app: OUTCOME = (b) PARTIAL FIRE — {:.1}% of slots fired \
             (mean = {:.3} vs expected {:.3})",
            observed_fraction * 100.0,
            mean,
            expected_per_slot,
        );
    }
}
