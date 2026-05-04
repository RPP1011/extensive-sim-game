//! Trade-market probe harness — drives `trade_market_runtime` for
//! N ticks and asserts the per-Trader trade volume + per-hub volume +
//! per-(observer, hub) belief bitset all converge to the analytical
//! observable shape under the placeholder routing (every emit binds
//! its key fields to `self`).
//!
//! ## Expected (FULL FIRE)
//!
//! With agent_count = 32, ticks = 100, trade_amount = 1.0,
//! observation_bit = 1u, decay = 0.95:
//!
//!   - trader_volume[i] (f32 with @decay) per slot ≈
//!     `trade_amount / (1 - 0.95)` = 20.0. The geometric series
//!     `Σ_{k=0..T-1} 0.95^k` converges to 20.0 with `0.95^100 ≈
//!     5.9e-3`, so per-slot is within 0.6% of the analytical limit.
//!   - hub_volume[i] (f32 no decay) per slot = `T * trade_amount`
//!     = 100.0 (monotonic accumulator, placeholder seller=self).
//!   - price_belief[i*N + i] (u32 pair_map) == 1u (diagonal —
//!     observation_bit OR'd into (self, self) every tick; idempotent).
//!   - price_belief[i*N + j] == 0u for every i != j (off-diagonal
//!     stays at 0u under placeholder routing observer=hub=self).
//!
//! ## OUTCOME classification
//!
//! - **(a) FULL FIRE** — all three views converge to expected shape.
//! - **(b) NO FIRE** — every slot stayed at 0; producer or fold
//!   kernel dropped at compile time, OR multi-event-kind ring
//!   partition broken (filter on tag mismatched producer's emit).
//! - **(b) PARTIAL FIRE** — some views fire, others don't; or
//!   off-diagonal price_belief slots have garbage (would indicate
//!   second_key_pop / event-field-offset mismatch between compiler
//!   emit and runtime cfg).
//!
//! Discovery write-up:
//! `docs/superpowers/notes/2026-05-04-trade_market_probe.md`.

use engine::CompiledSim;
use trade_market_runtime::TradeMarketState;

const SEED: u64 = 0xBEEF_FEED_CAFE_F00D;
const AGENT_COUNT: u32 = 32;
const TICKS: u64 = 100;
const TRADE_AMOUNT: f32 = 1.0; // matches config.market.trade_amount
const DECAY_RATE: f32 = 0.95;  // matches @decay(rate=0.95) on trader_volume
const OBSERVATION_BIT: u32 = 1; // matches config.market.observation_bit

fn main() {
    let mut sim = TradeMarketState::new(SEED, AGENT_COUNT);
    println!(
        "trade_market_app: starting — seed=0x{:016X} agents={} ticks={}",
        SEED, AGENT_COUNT, TICKS,
    );

    for _ in 0..TICKS {
        sim.step();
    }

    let n = AGENT_COUNT as usize;
    println!(
        "trade_market_app: finished — final tick={} agents={}",
        sim.tick(),
        sim.agent_count(),
    );

    // ---- View 1: trader_volume (f32, @decay) ----
    let tv = sim.trader_volumes().to_vec();
    let (tv_min, tv_mean, tv_max) = stats(&tv);
    let tv_steady = TRADE_AMOUNT / (1.0 - DECAY_RATE);
    let tv_total: f32 = tv.iter().sum();
    let tv_total_expected = AGENT_COUNT as f32 * tv_steady;
    println!(
        "trade_market_app: trader_volume   (decay={:.2}, steady ~{:.3}/slot) — \
         total={:.2} (expected {:.2}); per-slot min={:.3} mean={:.3} max={:.3}",
        DECAY_RATE, tv_steady, tv_total, tv_total_expected, tv_min, tv_mean, tv_max,
    );

    // ---- View 2: hub_volume (f32, no decay) ----
    let hv = sim.hub_volumes().to_vec();
    let (hv_min, hv_mean, hv_max) = stats(&hv);
    let hv_per_slot_expected = TICKS as f32 * TRADE_AMOUNT;
    let hv_total: f32 = hv.iter().sum();
    let hv_total_expected = AGENT_COUNT as f32 * hv_per_slot_expected;
    println!(
        "trade_market_app: hub_volume      (no decay, +{:.2}/tick/slot) — \
         total={:.2} (expected {:.2}); per-slot min={:.3} mean={:.3} max={:.3}",
        TRADE_AMOUNT, hv_total, hv_total_expected, hv_min, hv_mean, hv_max,
    );

    // ---- View 3: price_belief (u32, pair_map, atomicOr) ----
    let pb = sim.price_belief().to_vec();
    assert_eq!(
        pb.len(),
        n * n,
        "price_belief must be sized agent_count^2 = {}",
        n * n,
    );
    let mut diag_ok = 0usize;
    let mut diag_bad: Vec<(usize, u32)> = Vec::new();
    for i in 0..n {
        let v = pb[i * n + i];
        if v == OBSERVATION_BIT {
            diag_ok += 1;
        } else {
            diag_bad.push((i, v));
        }
    }
    let mut offdiag_ok = 0usize;
    let mut offdiag_bad: Vec<(usize, usize, u32)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let v = pb[i * n + j];
            if v == 0 {
                offdiag_ok += 1;
            } else {
                offdiag_bad.push((i, j, v));
            }
        }
    }
    println!(
        "trade_market_app: price_belief    diagonal {}/{} == {}u, off-diagonal {}/{} == 0u",
        diag_ok,
        n,
        OBSERVATION_BIT,
        offdiag_ok,
        n * (n - 1),
    );
    if !diag_bad.is_empty() {
        println!(
            "trade_market_app: price_belief diagonal mismatches (first 8): {:?}",
            &diag_bad[..diag_bad.len().min(8)],
        );
    }
    if !offdiag_bad.is_empty() {
        println!(
            "trade_market_app: price_belief off-diagonal mismatches (first 8): {:?}",
            &offdiag_bad[..offdiag_bad.len().min(8)],
        );
    }

    // ---- Outcome classification ----
    let trader_band = tv_steady * 0.05; // ±5%
    let trader_ok = (tv_mean - tv_steady).abs() <= trader_band
        && (tv_min - tv_steady).abs() <= trader_band
        && (tv_max - tv_steady).abs() <= trader_band;
    let hub_band = hv_per_slot_expected * 0.05; // ±5%
    let hub_ok = (hv_mean - hv_per_slot_expected).abs() <= hub_band
        && (hv_min - hv_per_slot_expected).abs() <= hub_band
        && (hv_max - hv_per_slot_expected).abs() <= hub_band;
    let belief_ok = diag_ok == n && offdiag_ok == n * (n - 1);

    println!(
        "trade_market_app: per-view checks — trader_volume={} hub_volume={} price_belief={}",
        if trader_ok { "OK" } else { "FAIL" },
        if hub_ok { "OK" } else { "FAIL" },
        if belief_ok { "OK" } else { "FAIL" },
    );

    if trader_ok && hub_ok && belief_ok {
        println!(
            "trade_market_app: OUTCOME = (a) FULL FIRE — multi-event-kind ring \
             (TradeExecuted + PriceObserved) partitions cleanly via per-handler \
             tag filter; mixed view-fold storage (u32 pair_map atomicOr + f32 \
             with @decay + f32 no-decay) coexists in one program."
        );
    } else {
        let mut what = Vec::new();
        if !trader_ok {
            what.push(format!(
                "trader_volume mean={:.3} (expected {:.3})",
                tv_mean, tv_steady
            ));
        }
        if !hub_ok {
            what.push(format!(
                "hub_volume mean={:.3} (expected {:.3})",
                hv_mean, hv_per_slot_expected
            ));
        }
        if !belief_ok {
            what.push(format!(
                "price_belief diag_ok={}/{} offdiag_ok={}/{}",
                diag_ok,
                n,
                offdiag_ok,
                n * (n - 1)
            ));
        }
        println!(
            "trade_market_app: OUTCOME = (b) PARTIAL FIRE / NO FIRE — {}; \
             see docs/superpowers/notes/2026-05-04-trade_market_probe.md \
             for the gap chain.",
            what.join("; "),
        );
        std::process::exit(1);
    }

    // The hard asserts come last so the human-readable OUTCOME line
    // prints first regardless of which view fails. These mirror the
    // auction_app pattern (analytical assertions on per-slot mean +
    // bounds within a 5% tolerance band).
    assert!(
        trader_ok,
        "trader_volume per-slot mean {:.4} not within ±5% of analytical {:.4}",
        tv_mean, tv_steady,
    );
    assert!(
        hub_ok,
        "hub_volume per-slot mean {:.4} not within ±5% of analytical {:.4}",
        hv_mean, hv_per_slot_expected,
    );
    assert!(
        belief_ok,
        "price_belief diagonal/off-diagonal didn't fully converge \
         (diag_ok={}/{}, offdiag_ok={}/{})",
        diag_ok,
        n,
        offdiag_ok,
        n * (n - 1),
    );
    println!("trade_market_app: OK — all assertions passed");
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
