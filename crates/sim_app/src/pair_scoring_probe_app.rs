//! Pair-scoring probe harness — drives `pair_scoring_probe_runtime`
//! for N ticks and reports the observed per-target `received` view.
//!
//! ## Predicted observable shapes
//!
//! ### (a) FULL FIRE — pair-field scoring wired end-to-end
//!
//! With AGENT_COUNT=8, TICKS=100, init `cooldown_next_ready_tick[N] = N*10`,
//! `heal.amount = 5.0`. Every Medic argmaxes over candidate Medics with
//! score `(1000 - target.cooldown_next_ready_tick)` → slot 0 wins for
//! every healer (lowest cooldown → highest inverted score):
//!
//!   - `received[0]` = TICKS × (AGENT_COUNT - 1) × heal.amount
//!                   = 100 × 7 × 5.0 = 3500.0
//!   - `received[N]` = 0.0 for N ∈ [1, 8)
//!   - sum = 3500.0
//!
//! ### (b) NO FIRE — pair-field scoring blocked at compile time
//!
//! ACTUAL outcome today. The verb's synthesised mask has head
//! `Positional([("target", _, AgentId)])` and `lower_mask` rejects
//! positional heads without a `from` clause (Task 2.6 still open).
//! The mask + scoring kernels are dropped from the partial CG program.
//! The chronicle physics rule + view-fold DO emit, but with no
//! ActionSelected events landing in the ring, the chronicle never
//! fires → no Healed events → `received[N]` stays at 0.0 every slot.
//!
//! Discovery doc: `docs/superpowers/notes/2026-05-04-pair_scoring_probe.md`.

use engine::CompiledSim;
use pair_scoring_probe_runtime::PairScoringProbeState;

const SEED: u64 = 0xA11_C0_1471_5005_u64;
const AGENT_COUNT: u32 = 8;
const TICKS: u64 = 100;

fn main() {
    let mut sim = PairScoringProbeState::new(SEED, AGENT_COUNT);
    println!(
        "pair_scoring_probe_app: starting — seed=0x{:016X} agents={} ticks={}",
        SEED, AGENT_COUNT, TICKS,
    );

    for _ in 0..TICKS {
        sim.step();
    }

    let received = sim.received().to_vec();
    println!(
        "pair_scoring_probe_app: finished — final tick={} agents={} received.len()={}",
        sim.tick(),
        sim.agent_count(),
        received.len(),
    );

    // FULL FIRE expected pattern: every healer picks slot 0; everyone
    // else gets nothing.
    let expected_slot_0: f32 =
        (TICKS as f32) * ((AGENT_COUNT - 1) as f32) * 5.0;
    let mut expected = vec![0.0_f32; AGENT_COUNT as usize];
    expected[0] = expected_slot_0;
    let expected_sum: f32 = expected.iter().sum();

    let (min, mean, max, sum, zero_slots) = stats(&received);
    let nonzero_slots = received.len() - zero_slots;
    let observed_fraction = (nonzero_slots as f32) / (received.len().max(1) as f32);

    println!(
        "pair_scoring_probe_app: received readback — min={:.3} mean={:.3} max={:.3} sum={:.3}",
        min, mean, max, sum,
    );
    println!(
        "pair_scoring_probe_app: nonzero slots: {}/{} (fraction = {:.1}%)",
        nonzero_slots,
        received.len(),
        observed_fraction * 100.0,
    );
    println!(
        "pair_scoring_probe_app: expected (FULL FIRE): received[0]={:.0}, received[1..]=0  → expected sum = {:.0}",
        expected_slot_0, expected_sum,
    );

    let preview: Vec<f32> = received.iter().take(8).copied().collect();
    println!(
        "pair_scoring_probe_app: preview received[0..8] = {:?}",
        preview,
    );

    // OUTCOME classification.
    if (received[0] - expected_slot_0).abs() < 0.5
        && received.iter().skip(1).all(|&x| x.abs() < 0.5)
    {
        println!(
            "pair_scoring_probe_app: OUTCOME = (a) FULL FIRE — pair-field \
             scoring picked the lowest-cooldown target every tick. The four \
             gap layers (parser / resolver / verb-injection scoring head / \
             N×N dispatch) all closed.",
        );
    } else if max == 0.0 {
        println!(
            "pair_scoring_probe_app: OUTCOME = (b) NO FIRE — every slot stayed \
             at 0.0 — confirms the gap chain in \
             docs/superpowers/notes/2026-05-04-pair_scoring_probe.md:\n  \
             Gap #1 (BLOCKING): mask `Positional` head from verb's `target` param \
             requires `from` clause routing (Task 2.6).\n  \
             Gap #2: score expression `1000.0 - <u32>` rejects with f32/u32 mismatch.\n  \
             Gap #3: verb-injected scoring entry hardcodes IrActionHeadShape::None — \
             needs Positional shape so target_local flips during row body lowering.\n  \
             Gap #4: ScoringArgmax dispatch is per-actor (1D) — pair-field scoring \
             needs per-(actor, candidate) pair (2D / per-actor inner loop).",
        );
    } else {
        println!(
            "pair_scoring_probe_app: OUTCOME = (b) PARTIAL — observed pattern \
             doesn't match either FULL FIRE or NO FIRE; some kernel partially \
             fired. max={max:.3} sum={sum:.3} nonzero={nonzero_slots}.",
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
