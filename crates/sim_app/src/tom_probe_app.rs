//! ToM probe harness — drives `tom_probe_runtime` for N ticks and
//! reports observed per-Knower fact_witnesses counts.
//!
//! ## Expected (belief-read cascade fires end-to-end, gap-free)
//!
//! With `agent_count = 32`, `ticks = 100`, every Knower's
//! `beliefs(self).about(self).confidence > 0.5` predicate evaluates
//! true (initial confidence = 1.0 by construction); every tick
//! produces 32 `LearnedFact` events; after 100 ticks: fact_witnesses
//! [i] = 100.0 for every i.
//!
//! ## Observed today (BeliefsAccessor + believes_knows lowering gaps)
//!
//! Both `CheckBelief` (BeliefsAccessor surface) and `CheckBeliefBit`
//! (theory_of_mind.believes_knows surface) physics rules drop out at
//! CG-lower time. No producer kernel is emitted; no LearnedFact
//! events are ever written; fact_witnesses[i] = 0.0 for every slot.
//!
//! Discovery write-up: `docs/superpowers/notes/2026-05-04-tom-probe.md`.

use engine::CompiledSim;
use tom_probe_runtime::TomProbeState;

const SEED: u64 = 0xCAFE_BABE_DEAD_BEEF;
const AGENT_COUNT: u32 = 32;
const TICKS: u64 = 100;
// Each LearnedFact emit would bump fact_witnesses[observer] by 1.0
// (per the view's `self += 1.0` body), so the analytical observable
// after N ticks is N per slot. With 2 producer rules dropping (both
// emit LearnedFact for the same observer/tick), the gap-free run
// would produce 2 × N per slot — but the simpler 1-rule case is the
// success threshold for the probe's OUTCOME (a) classification.
const EMITS_PER_TICK: f32 = 1.0;

fn main() {
    let mut sim = TomProbeState::new(SEED, AGENT_COUNT);
    println!(
        "tom_probe_app: starting — seed=0x{:016X} agents={} ticks={}",
        SEED, AGENT_COUNT, TICKS,
    );

    for _ in 0..TICKS {
        sim.step();
    }

    let fact_witnesses = sim.fact_witnesses().to_vec();
    println!(
        "tom_probe_app: finished — final tick={} agents={} fact_witnesses.len()={}",
        sim.tick(),
        sim.agent_count(),
        fact_witnesses.len(),
    );

    // Expected analytical observable: fact_witnesses[i] = TICKS *
    // EMITS_PER_TICK (one rule firing).
    let expected_per_slot = (TICKS as f32) * EMITS_PER_TICK;

    let mut min = f32::INFINITY;
    let mut max = 0.0_f32;
    let mut sum = 0.0_f32;
    let mut zero_slots = 0usize;
    for &v in &fact_witnesses {
        min = min.min(v);
        max = max.max(v);
        sum += v;
        if v == 0.0 {
            zero_slots += 1;
        }
    }
    let mean = sum / fact_witnesses.len() as f32;
    let nonzero_slots = fact_witnesses.len() - zero_slots;
    let observed_fraction = (nonzero_slots as f32) / (fact_witnesses.len() as f32);

    println!(
        "tom_probe_app: fact_witnesses readback — min={:.3} mean={:.3} max={:.3}",
        min, mean, max,
    );
    println!(
        "tom_probe_app: nonzero slots: {}/{} (fraction = {:.3}%)",
        nonzero_slots,
        fact_witnesses.len(),
        observed_fraction * 100.0,
    );
    println!(
        "tom_probe_app: expected per-slot value (single rule firing): {:.3} \
         (= TICKS={} × emits_per_tick={:.3})",
        expected_per_slot, TICKS, EMITS_PER_TICK,
    );

    if min >= expected_per_slot * 0.99 {
        println!(
            "tom_probe_app: OUTCOME = (a) FULL FIRE — every slot ≈ expected value; \
             the BeliefsAccessor + believes_knows lowering gaps closed!",
        );
    } else if max == 0.0 {
        println!(
            "tom_probe_app: OUTCOME = (b) NO FIRE — every slot stayed at 0.0; \
             see docs/superpowers/notes/2026-05-04-tom-probe.md for the gap chain",
        );
    } else {
        println!(
            "tom_probe_app: OUTCOME = (b) PARTIAL FIRE — {:.1}% of slots fired \
             (mean = {:.3} vs expected {:.3})",
            observed_fraction * 100.0,
            mean,
            expected_per_slot,
        );
    }
}
