//! Bartering coverage demo. Drives the [`bartering_runtime`]
//! integrator through the same generic harness shape as the other
//! fixture binaries. The fixture is the entity-root coverage probe
//! (declares `entity Coin : Item` + `entity Caravan : Group` plus
//! a `Trade` event with an `item: ItemId` payload field), and now
//! the Item AND Group READ probe — the IdleDrift physics rule reads
//! BOTH `items.weight(0)` AND `groups.size(0)` so the centroid drift
//! observable below pins both SoA buffer reads end-to-end.
//!
//! Engaged-with topology: every slot points at slot 0, so slot 0
//! receives every Trade and all other slots receive none. After
//! T ticks with N alive agents the analytical observables are:
//!
//! - **trade_count[0]** = N * T          (every tick, N events
//!   land on slot 0)
//! - **trade_count[i > 0]** = 0           (no trades route here)
//! - **sum(trade_count)** = N * T
//!
//! With AGENT_COUNT=64 + TICKS=100 the expected slot-0 + total
//! values are both 6400. (Subject to the swarm_storm-side
//! `compose_view_fold_record_method dispatch_workgroups(1, 1, 1)`
//! gap — see ses_app for context — but at 1 emit/agent/tick the
//! event-count dispatch sizing handles it cleanly.)

use bartering_runtime::{BarteringState, CARAVAN_SIZE_VALUE, COIN_WEIGHT_VALUE};
use engine::CompiledSim;
use glam::Vec3;

const SEED: u64 = 0xBA47_E101_4A4D_2026;
const AGENT_COUNT: u32 = 64;
const TICKS: u64 = 100;
const LOG_INTERVAL_TICKS: u64 = 25;

fn main() {
    let mut sim = BarteringState::new(SEED, AGENT_COUNT);
    let initial_centroid = centroid(sim.positions());
    println!(
        "bartering_app: starting run — seed=0x{:016X} agents={} ticks={} coin_weight={:.2} caravan_size={:.2}",
        SEED, AGENT_COUNT, TICKS, COIN_WEIGHT_VALUE, CARAVAN_SIZE_VALUE,
    );
    log_sample(&mut sim);
    for _ in 0..TICKS {
        sim.step();
        if sim.tick() % LOG_INTERVAL_TICKS == 0 {
            log_sample(&mut sim);
        }
    }
    println!(
        "bartering_app: finished — final tick={} agents={}",
        sim.tick(),
        sim.agent_count(),
    );

    let final_centroid = centroid(sim.positions());
    let tc = sim.trade_counts().to_vec();
    let total: f32 = tc.iter().sum();
    let slot0 = tc.first().copied().unwrap_or(0.0);
    let max_other = tc
        .iter()
        .skip(1)
        .copied()
        .fold(0.0_f32, f32::max);
    let expected_total = (AGENT_COUNT as f32) * (TICKS as f32);
    println!(
        "bartering_app: trade_count       — total={:.2} (expected {:.0}); slot[0]={:.2} max(slot[i>0])={:.2}",
        total, expected_total, slot0, max_other,
    );

    // The "every slot routes to slot 0" engaged_with topology means
    // every Trade event lands on slot 0. If folding silently drops
    // events (B1-style), or if the ItemId field shifts the payload
    // packing wrong and the receiver field reads garbage, slot 0
    // would underrun and other slots would have spurious values —
    // both surface here.
    assert!(
        (slot0 - expected_total).abs() < 1.0,
        "expected slot[0] ≈ {expected_total:.0}, got {slot0:.2}; full counts: {tc:?}"
    );
    assert!(
        max_other < 0.5,
        "expected slot[i>0] = 0 (every Trade routes to slot 0), got max_other={max_other:.2}; full counts: {tc:?}"
    );
    assert!(
        (total - expected_total).abs() < 1.0,
        "expected total ≈ {expected_total:.0}, got {total:.2}"
    );

    // Item + Group field-read assertion: the IdleDrift physics rule
    // body multiplies its per-tick velocity drift by BOTH
    // `items.weight(0)` AND `groups.size(0)`, which the per-fixture
    // runtime backs with single-record `coin_weight: array<f32>` and
    // `caravan_size: array<f32>` SoA buffers (values
    // `COIN_WEIGHT_VALUE` and `CARAVAN_SIZE_VALUE`). With both
    // multipliers in place every agent's centroid drift is
    // `weight × size × baseline_drift`; with either silently dropped
    // the drift would collapse by the corresponding factor. We compare
    // the centroid delta against the `baseline × COIN_WEIGHT_VALUE ×
    // CARAVAN_SIZE_VALUE` band — values in that band confirm BOTH the
    // Item-field read AND the Group-field read landed on the right
    // buffers and propagated the right values through the WGSL body.
    //
    // Baseline drift per axis (collected with the runtime pinned to
    // both multipliers = 1.0 for one calibration run):
    //   centroid drift over 100 ticks ≈ (-0.022, -0.013, +0.049)
    //   magnitude ≈ 0.054
    // With weight=2.0 × size=1.5 we expect:
    //   magnitude ≈ 0.054 × 2.0 × 1.5 ≈ 0.162.
    // We assert the magnitude of the drift sits inside a tight band
    // around the combined expected value. Silently dropping either
    // multiplier collapses the drift to one of:
    //   - drop both: ~0.054 (3× off)
    //   - drop weight, keep size: ~0.081 (2× off)
    //   - drop size, keep weight: ~0.108 (1.5× off)
    // The band rejects all three regression scenarios.
    let drift = final_centroid - initial_centroid;
    let drift_mag = drift.length();
    // Baseline (weight=size=1) drift magnitude over the full run,
    // observed empirically with both multipliers pinned to 1.0.
    let baseline_drift_mag = 0.054_f32;
    let expected_drift_mag = baseline_drift_mag * COIN_WEIGHT_VALUE * CARAVAN_SIZE_VALUE;
    // Tight band around the expected magnitude. The narrowest
    // regression to catch is "drop the Group-field read" which
    // collapses drift to `baseline × COIN_WEIGHT_VALUE = 0.108`, a
    // distance of `baseline × COIN_WEIGHT_VALUE × (CARAVAN_SIZE_VALUE
    // - 1.0) = 0.054` from the expected value. A band of
    // `baseline × 0.3 = 0.0162` is well inside that gap so a
    // silent-drop regression on either field fails the assertion.
    let band = baseline_drift_mag * 0.3;
    let lower_bound = expected_drift_mag - band;
    let upper_bound = expected_drift_mag + band;
    println!(
        "bartering_app: centroid drift     — magnitude={:.4} (expected ≈ {:.4} from coin_weight × caravan_size × baseline)",
        drift_mag, expected_drift_mag,
    );
    assert!(
        drift_mag > lower_bound && drift_mag < upper_bound,
        "expected centroid drift magnitude in ({lower_bound:.4}, {upper_bound:.4}) \
         (coin_weight={COIN_WEIGHT_VALUE}, caravan_size={CARAVAN_SIZE_VALUE}); got {drift_mag:.4}. \
         A drift near {baseline_drift_mag:.4} means BOTH reads dropped; near \
         {:.4} means `groups.size(0)` dropped (Group-field read regression); near \
         {:.4} means `items.weight(0)` dropped (Item-field read regression).",
        baseline_drift_mag * COIN_WEIGHT_VALUE,
        baseline_drift_mag * CARAVAN_SIZE_VALUE,
    );

    println!("bartering_app: OK — all assertions passed");
}

fn centroid(positions: &[Vec3]) -> Vec3 {
    if positions.is_empty() {
        return Vec3::ZERO;
    }
    let mut sum = Vec3::ZERO;
    for p in positions {
        sum += *p;
    }
    sum / positions.len() as f32
}

fn log_sample(sim: &mut BarteringState) {
    let tick = sim.tick();
    let positions = sim.positions();
    if positions.is_empty() {
        println!("  tick {:>4}: (no agents)", tick);
        return;
    }
    let mut min = positions[0];
    let mut max = positions[0];
    let mut sum = Vec3::ZERO;
    for p in positions {
        min = min.min(*p);
        max = max.max(*p);
        sum += *p;
    }
    let centroid = sum / positions.len() as f32;
    let span = max - min;
    let max_diameter = span.x.max(span.y).max(span.z);
    println!(
        "  tick {:>4}: centroid=({:>+7.3}, {:>+7.3}, {:>+7.3}) max_diameter={:>+6.2}",
        tick, centroid.x, centroid.y, centroid.z, max_diameter,
    );
}
