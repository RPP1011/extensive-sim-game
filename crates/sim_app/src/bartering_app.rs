//! Bartering coverage demo. Drives the [`bartering_runtime`]
//! integrator through the same generic harness shape as the other
//! fixture binaries. The fixture is the entity-root coverage probe
//! (declares `entity Coin : Item` + `entity Caravan : Group` plus
//! a `Trade` event with an `item: ItemId` payload field), so the
//! interesting part is whether the run reaches steady-state at all
//! — the per-tick observable is the per-receiver Trade counter.
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

use bartering_runtime::BarteringState;
use engine::CompiledSim;
use glam::Vec3;

const SEED: u64 = 0xBA47_E101_4A4D_2026;
const AGENT_COUNT: u32 = 64;
const TICKS: u64 = 100;
const LOG_INTERVAL_TICKS: u64 = 25;

fn main() {
    let mut sim = BarteringState::new(SEED, AGENT_COUNT);
    println!(
        "bartering_app: starting run — seed=0x{:016X} agents={} ticks={}",
        SEED, AGENT_COUNT, TICKS,
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
    println!("bartering_app: OK — all assertions passed");
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
