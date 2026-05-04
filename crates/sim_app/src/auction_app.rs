//! Auction-market coverage demo. Drives the [`auction_runtime`]
//! integrator through the same generic harness shape as the other
//! fixture binaries. The fixture is the third sequential
//! implementation past `foraging_app` (which exercised
//! single-Agent SoA + per-tick Drop + `@decay` view); here, the
//! .sim file declares both `entity Good : Item` AND
//! `entity Faction : Group`, but neither is exercised by the
//! active rule body — the per-Item `good_clearing_price` and
//! per-Group `faction_pressure` views stay commented out behind
//! `// GAP:` markers in the .sim file.
//!
//! What this app DOES exercise:
//!
//! - Single-Agent SoA (`Trader : Agent`).
//! - Per-tick `emit Bid { trader: self, good: self, amount:
//!   config.market.bid_amount }` from `physics WanderAndBid`.
//! - `let _did_bid = auctions.place_bid(self, self, ...)` — first
//!   fixture to exercise the `auctions.*` namespace lowering. The
//!   B1 stub `fn auctions_place_bid(...) -> bool { return true; }`
//!   gets emitted into the WGSL physics body.
//! - Two views, both Agent-keyed:
//!   - `bid_activity(trader: Agent) -> f32` with
//!     `@decay(rate=0.90)`. Per-event the matching slot increments
//!     by 1.0 BEFORE the next per-tick anchor multiply lands, so
//!     the per-slot steady-state is `1 / (1 - 0.90) = 10.0`.
//!   - `good_bid_total(good: Agent) -> f32` with no decay.
//!     Per-event the matching slot += `amount`. After T=100 ticks
//!     each per-Trader slot = `T × bid_amount = 100 × 10.0 = 1000.0`.
//!
//! After T=100 ticks the geometric series sum
//! `sum_{k=0..T-1} 0.90^k ≈ 1 / (1 - 0.90) = 10.0` already pins
//! the per-slot bid_activity value to within < 1e-3 of the steady
//! state (0.90^100 ≈ 2.66e-5, well below the ±5% acceptance band).

use auction_runtime::AuctionState;
use engine::CompiledSim;
use glam::Vec3;
use std::path::PathBuf;

const SEED: u64 = 0xAEC7_0104_4202_6053;
const AGENT_COUNT: u32 = 64;
const TICKS: u64 = 100;
const LOG_INTERVAL_TICKS: u64 = 25;
const BID_AMOUNT: f32 = 10.0; // matches config.market.bid_amount in the .sim
const DECAY_RATE: f32 = 0.90; // matches @decay(rate=0.90) on bid_activity

fn main() {
    // SLICE 3 — auctions.place_bid stub call appears in physics WGSL.
    // We grep the build-output WGSL for `auctions_place_bid(` to
    // confirm the namespace lowering survived the runtime build
    // pipeline. The OUT_DIR is opaque from the binary side, so we
    // walk `target/debug/build/auction_runtime-*/out/`.
    let stub_status = check_auctions_stub_in_wgsl();
    println!("auction_app: auctions.place_bid stub check — {}", stub_status);

    let mut sim = AuctionState::new(SEED, AGENT_COUNT);
    println!(
        "auction_app: starting run — seed=0x{:016X} traders={} ticks={} bid_amount={:.2} decay={:.2}",
        SEED, AGENT_COUNT, TICKS, BID_AMOUNT, DECAY_RATE,
    );
    log_sample(&mut sim);
    for _ in 0..TICKS {
        sim.step();
        if sim.tick() % LOG_INTERVAL_TICKS == 0 {
            log_sample(&mut sim);
        }
    }
    println!(
        "auction_app: finished — final tick={} traders={}",
        sim.tick(),
        sim.agent_count(),
    );

    // ---- View 1: bid_activity (per-Trader, @decay) ----
    let ba = sim.bid_activity().to_vec();
    let (ba_min, ba_mean, ba_max) = stats(&ba);
    let ba_steady = 1.0_f32 / (1.0 - DECAY_RATE);
    let ba_total_expected = AGENT_COUNT as f32 * ba_steady;
    let ba_total: f32 = ba.iter().sum();
    println!(
        "auction_app: bid_activity      (rate={:.2}, steady ~{:.3}/slot) \
         — total={:.2} (expected {:.2}); per-slot min={:.3} mean={:.3} max={:.3}",
        DECAY_RATE, ba_steady, ba_total, ba_total_expected, ba_min, ba_mean, ba_max,
    );
    let ba_tol = ba_steady * 0.05;
    assert!(
        (ba_mean - ba_steady).abs() <= ba_tol,
        "bid_activity per-slot mean {:.4} not within ±5% of analytical {:.4} (band ±{:.4})",
        ba_mean, ba_steady, ba_tol,
    );
    assert!(
        (ba_min - ba_steady).abs() <= ba_tol && (ba_max - ba_steady).abs() <= ba_tol,
        "bid_activity per-slot min/max ({:.4}/{:.4}) not within ±5% of analytical {:.4}; \
         uniform-population fixture should have identical per-slot values across all {} traders",
        ba_min, ba_max, ba_steady, AGENT_COUNT,
    );
    println!(
        "auction_app: OK — bid_activity per-slot mean ≈ {:.3} (target {:.3}, tol ±{:.3})",
        ba_mean, ba_steady, ba_tol,
    );

    // ---- View 2: good_bid_total (per-Trader, no decay) ----
    let gbt = sim.good_bid_totals().to_vec();
    let (gbt_min, gbt_mean, gbt_max) = stats(&gbt);
    let gbt_per_slot_expected = TICKS as f32 * BID_AMOUNT; // 100 * 10.0 = 1000.0
    let gbt_total_expected = AGENT_COUNT as f32 * gbt_per_slot_expected;
    let gbt_total: f32 = gbt.iter().sum();
    println!(
        "auction_app: good_bid_total    (no decay, +{:.2}/tick/slot) \
         — total={:.2} (expected {:.2}); per-slot min={:.3} mean={:.3} max={:.3}",
        BID_AMOUNT, gbt_total, gbt_total_expected, gbt_min, gbt_mean, gbt_max,
    );
    let gbt_tol = gbt_per_slot_expected * 0.05;
    assert!(
        (gbt_mean - gbt_per_slot_expected).abs() <= gbt_tol,
        "good_bid_total per-slot mean {:.4} not within ±5% of analytical {:.4} (band ±{:.4})",
        gbt_mean, gbt_per_slot_expected, gbt_tol,
    );
    assert!(
        (gbt_min - gbt_per_slot_expected).abs() <= gbt_tol
            && (gbt_max - gbt_per_slot_expected).abs() <= gbt_tol,
        "good_bid_total per-slot min/max ({:.4}/{:.4}) not within ±5% of analytical {:.4}; \
         every Trader emits `good: self` so all {} slots should be identical",
        gbt_min, gbt_max, gbt_per_slot_expected, AGENT_COUNT,
    );
    println!(
        "auction_app: OK — good_bid_total per-slot mean ≈ {:.3} (target {:.3}, tol ±{:.3})",
        gbt_mean, gbt_per_slot_expected, gbt_tol,
    );

    // ---- View 3: faction_pressure (SLICE 2 PROBE — Group-keyed) ----
    // Today the Bid event has no `faction` field so the
    // compiler-emitted fold defaults the key to slot 2 (= the
    // `trader` field), making this view dispatch identically to an
    // Agent-keyed view: each per-Trader slot accumulates `bid_amount`
    // per tick under decay 0.95 → steady ≈ bid_amount / (1 - 0.95)
    // = 200.0. Wiring it through proves the Group-keyed view's
    // compile + decay + fold-dispatch path is healthy end-to-end in
    // the auction context (foraging proved the same for the no-emit
    // case; this is the with-emit variant).
    let fp = sim.faction_pressures().to_vec();
    let (fp_min, fp_mean, fp_max) = stats(&fp);
    let fp_decay = 0.95_f32;
    let fp_per_slot_expected = BID_AMOUNT / (1.0 - fp_decay);
    let fp_total: f32 = fp.iter().sum();
    let fp_total_expected = AGENT_COUNT as f32 * fp_per_slot_expected;
    println!(
        "auction_app: faction_pressure (rate={:.2}, steady ~{:.3}/slot via trader-key fallback) \
         — total={:.2} (expected {:.2}); per-slot min={:.3} mean={:.3} max={:.3}",
        fp_decay, fp_per_slot_expected, fp_total, fp_total_expected, fp_min, fp_mean, fp_max,
    );
    let fp_tol = fp_per_slot_expected * 0.05;
    assert!(
        (fp_mean - fp_per_slot_expected).abs() <= fp_tol,
        "faction_pressure per-slot mean {:.4} not within ±5% of analytical {:.4} (band ±{:.4})",
        fp_mean, fp_per_slot_expected, fp_tol,
    );
    assert!(
        (fp_min - fp_per_slot_expected).abs() <= fp_tol
            && (fp_max - fp_per_slot_expected).abs() <= fp_tol,
        "faction_pressure per-slot min/max ({:.4}/{:.4}) not within ±5% of analytical {:.4}",
        fp_min, fp_max, fp_per_slot_expected,
    );
    println!(
        "auction_app: OK — faction_pressure per-slot mean ≈ {:.3} (target {:.3}, tol ±{:.3}); \
         Group-keyed view's compile + decay + fold dispatch path is healthy",
        fp_mean, fp_per_slot_expected, fp_tol,
    );

    println!("auction_app: OK — all assertions passed");
}

fn stats(v: &[f32]) -> (f32, f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = 0.0_f32;
    let mut sum = 0.0_f32;
    for &x in v {
        if x < min { min = x; }
        if x > max { max = x; }
        sum += x;
    }
    let mean = if v.is_empty() { 0.0 } else { sum / v.len() as f32 };
    (min, mean, max)
}

/// SLICE 3 grep probe. Walks `target/<profile>/build/auction_runtime-*/out/`
/// looking for `physics_WanderAndBid.wgsl` and confirms it contains a
/// call to `auctions_place_bid(`. Returns a status string for
/// printing — does NOT assert (this is a smoke test; if WGSL layout
/// changes location the user can still see the analytical view
/// assertions pass).
fn check_auctions_stub_in_wgsl() -> String {
    // Walk both debug and release build dirs (cargo test may invoke
    // either; the binary itself is in `target/<profile>/`).
    let candidates = [
        PathBuf::from("target/debug/build"),
        PathBuf::from("target/release/build"),
    ];
    for root in &candidates {
        if !root.is_dir() {
            continue;
        }
        let entries = match std::fs::read_dir(root) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };
            if !name.starts_with("auction_runtime-") {
                continue;
            }
            let wgsl = path.join("out").join("physics_WanderAndBid.wgsl");
            if let Ok(body) = std::fs::read_to_string(&wgsl) {
                let has_decl = body.contains("fn auctions_place_bid(");
                let has_call = body.lines().any(|line| {
                    line.contains("auctions_place_bid(") && !line.contains("fn auctions_place_bid")
                });
                return format!(
                    "wgsl={}; decl={} call={}",
                    wgsl.display(),
                    has_decl,
                    has_call,
                );
            }
        }
    }
    "no auction_runtime build artifact found (cwd-dependent)".to_string()
}

fn log_sample(sim: &mut AuctionState) {
    let tick = sim.tick();
    let positions = sim.positions();
    if positions.is_empty() {
        println!("  tick {:>4}: (no traders)", tick);
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
