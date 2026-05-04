//! Cooldown probe harness — drives `cooldown_probe_runtime` for N
//! ticks and reports the observed per-caster `activations` view.
//!
//! ## Predicted observable shapes
//!
//! ### (a) FULL FIRE — both surfaces wired end-to-end
//!
//! With AGENT_COUNT=32, TICKS=100, `ready_at[N] = N` (staggered),
//! `log_amount = 1.0`:
//!
//!   - `activations[N]` = max(0, TICKS - N) = 100 - N
//!   - `activations[0]`  = 100   (always ready)
//!   - `activations[1]`  = 99
//!   - `activations[31]` = 69
//!   - sum = 32*100 - sum(0..32) = 3200 - 496 = 2704
//!
//! ### (b) NO FIRE — gap surfaced
//!
//! Most likely root cause: the per-agent SoA field read for
//! `cooldown_next_ready_tick` doesn't lower (no kernel ever reads
//! the runtime-allocated buffer), so the gate `tick >= ready_at`
//! sees a zero default and either always fires (giving `activations
//! [N] = TICKS` per slot) or never fires.
//!
//! Discovery doc: `docs/superpowers/notes/2026-05-04-cooldown_probe.md`.

use cooldown_probe_runtime::CooldownProbeState;
use engine::CompiledSim;

const SEED: u64 = 0xC001_DA_1771_5005;
const AGENT_COUNT: u32 = 32;
const TICKS: u64 = 100;

fn main() {
    let mut sim = CooldownProbeState::new(SEED, AGENT_COUNT);
    println!(
        "cooldown_probe_app: starting — seed=0x{:016X} agents={} ticks={}",
        SEED, AGENT_COUNT, TICKS,
    );

    for _ in 0..TICKS {
        sim.step();
    }

    let activations = sim.activations().to_vec();
    println!(
        "cooldown_probe_app: finished — final tick={} agents={} activations.len()={}",
        sim.tick(),
        sim.agent_count(),
        activations.len(),
    );

    // Compute analytical expected pattern: activations[N] = max(0,
    // TICKS - N).
    let expected: Vec<f32> = (0..AGENT_COUNT)
        .map(|n| (TICKS as i64 - n as i64).max(0) as f32)
        .collect();
    let expected_sum: f32 = expected.iter().sum();

    let (min, mean, max, sum, zero_slots) = stats(&activations);
    let nonzero_slots = activations.len() - zero_slots;
    let observed_fraction = (nonzero_slots as f32) / (activations.len().max(1) as f32);

    println!(
        "cooldown_probe_app: activations readback — min={:.3} mean={:.3} max={:.3} sum={:.3}",
        min, mean, max, sum,
    );
    println!(
        "cooldown_probe_app: nonzero slots: {}/{} (fraction = {:.1}%)",
        nonzero_slots,
        activations.len(),
        observed_fraction * 100.0,
    );
    println!(
        "cooldown_probe_app: expected pattern (staggered): activations[N] = max(0, {} - N)  → expected sum = {:.0}",
        TICKS, expected_sum,
    );

    // Per-slot match: count slots that exactly match the analytical
    // pattern (within 1 unit slack to absorb edge atomic-CAS races
    // that aren't actually expected here, but cheap insurance).
    let mut exact_matches = 0;
    let mut max_diff: f32 = 0.0;
    for (n, (&got, &want)) in activations.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        if diff < 0.5 {
            exact_matches += 1;
        }
        if diff > max_diff {
            max_diff = diff;
            // Note: we don't bail; we want to see the full picture.
            let _ = n;
        }
    }

    println!(
        "cooldown_probe_app: per-slot matches (|got-want| < 0.5): {}/{} (max_diff = {:.3})",
        exact_matches,
        activations.len(),
        max_diff,
    );

    // Print the first few slots so the failure mode is visible.
    let preview = activations.iter().take(8).copied().collect::<Vec<_>>();
    let preview_expected = expected.iter().take(8).copied().collect::<Vec<_>>();
    println!(
        "cooldown_probe_app: preview activations[0..8] = {:?}  (expected = {:?})",
        preview, preview_expected,
    );

    // OUTCOME classification.
    if exact_matches == activations.len() as usize {
        println!(
            "cooldown_probe_app: OUTCOME = (a) FULL FIRE — every slot matched the staggered \
             analytical pattern. Two surfaces verified end-to-end:\n  \
             1. agents.cooldown_next_ready_tick(self) lowers + reads correctly\n  \
             2. world.tick >= ready_at gating + emit fires per-tick",
        );
    } else if max == 0.0 {
        println!(
            "cooldown_probe_app: OUTCOME = (b) NO FIRE — every slot stayed at 0.0. \
             Likely gap: the physics body's emit never lands, OR the agent_cooldown_next_ \
             ready_tick SoA read returns garbage that defeats the `tick >= ready_at` gate. \
             See docs/superpowers/notes/2026-05-04-cooldown_probe.md.",
        );
    } else if min as f32 == TICKS as f32 && max as f32 == TICKS as f32 {
        println!(
            "cooldown_probe_app: OUTCOME = (b) PARTIAL — every slot fired EVERY tick \
             (per-slot count = TICKS). The cooldown SoA read is returning 0 (default) \
             for every slot, so `tick >= 0` is always true and the staggered init was \
             not applied. The gap is in the cooldown_next_ready_tick SoA wiring.",
        );
    } else {
        println!(
            "cooldown_probe_app: OUTCOME = (b) PARTIAL FIRE — pattern doesn't match \
             analytical prediction. {}/{} slots match exactly; max diff = {:.3}.",
            exact_matches,
            activations.len(),
            max_diff,
        );
    }
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
