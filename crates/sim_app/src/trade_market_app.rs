//! Trade-market probe harness — drives `trade_market_runtime` for
//! N ticks and asserts the per-Trader trade volume + per-hub volume +
//! per-(observer, hub) belief bitset all converge to the analytical
//! observable shape under the v2 dynamics:
//!
//! - **trader_volume** + **hub_volume** are fed by `TradeExecuted`
//!   events emitted by the verb chronicle (`physics_verb_chronicle_
//!   ExecuteTrade`). The verb fires once per agent per tick (single-
//!   row scoring picks `ExecuteTrade` action_id=0; mask gates on
//!   `self.alive`). Same per-slot dynamics as v1 (the verb body has
//!   no spatial access — buyer=self, seller=self placeholders).
//!
//! - **price_belief** is fed by BOTH `PriceObserved` (kind=2u) AND
//!   `PriceGossip` (kind=3u) events emitted by the multi-emit
//!   physics rule `physics_WanderAndTrade`. That rule walks the
//!   27-cell spatial neighbourhood (slice 2b body-form) and per
//!   candidate emits BOTH event kinds — so EVERY (observer, hub)
//!   cell where the two agents are spatial neighbours flips its
//!   belief bit to 1u. With AGENT_COUNT=32 auto-spread (cube-root
//!   spread = ~3.17) inside a 6.0-cell-edge spatial grid, all
//!   agents land within 1 cell of each other — so the belief
//!   matrix's off-diagonal IS NO LONGER zero. Every off-diagonal
//!   cell flips to 1u.
//!
//! ## Expected (FULL FIRE) — v2 dynamics
//!
//! With agent_count = 32, ticks = 100, trade_amount = 1.0,
//! observation_bit = 1u, decay = 0.95:
//!
//!   - trader_volume[i] (f32 with @decay) per slot ≈ 20.000 (geometric
//!     series limit; per-tick `+= 1.0` then `*= 0.95`).
//!   - hub_volume[i]    (f32 no-decay) per slot = `T * 1.0` = 100.0.
//!   - price_belief[i*N + j]            == 1u for ALL i, j (diagonal +
//!     off-diagonal). Auto-spread layout puts every agent within
//!     every other agent's 27-cell neighbourhood, so the spatial
//!     walk's per-pair emit hits every cell.
//!
//! ## Verb cascade integration confirmation
//!
//! If `trader_volume` and `hub_volume` are non-zero, the verb cascade
//! end-to-end fires: mask -> scoring -> chronicle -> fold. Without
//! the chronicle path running (and emitting `TradeExecuted`), both
//! views would stay at their initial 0.0.
//!
//! ## Spatial body-form integration confirmation
//!
//! If `price_belief` has any off-diagonal cell == 1u, the slice-2b
//! body-form spatial walk inside `WanderAndTrade` is firing. Without
//! the spatial loop, the only emit would be a placeholder per agent
//! (one PriceObserved per (self, self) cell) and the off-diagonal
//! would stay 0u.
//!
//! ## OUTCOME classification
//!
//! - **(a) FULL FIRE** — all three views converge to expected v2 shape.
//! - **(b) PARTIAL** — at least one view didn't converge.
//!
//! Discovery write-up:
//! `docs/superpowers/notes/2026-05-04-trade_market_probe.md`.

use engine::CompiledSim;
use trade_market_runtime::TradeMarketState;

const SEED: u64 = 0xBEEF_FEED_CAFE_F00D;
const AGENT_COUNT: u32 = 32;
const TICKS: u64 = 100;
const TRADE_AMOUNT: f32 = 1.0;
const DECAY_RATE: f32 = 0.95;
const OBSERVATION_BIT: u32 = 1;

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
        "trade_market_app: trader_volume   (verb chronicle, decay={:.2}, steady ~{:.3}/slot) — \
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
        "trade_market_app: hub_volume      (verb chronicle, no decay, +{:.2}/tick/slot) — \
         total={:.2} (expected {:.2}); per-slot min={:.3} mean={:.3} max={:.3}",
        TRADE_AMOUNT, hv_total, hv_total_expected, hv_min, hv_mean, hv_max,
    );

    // ---- View 3: price_belief (u32, pair_map, atomicOr) ----
    //
    // v2 prediction: at AGENT_COUNT=32 the cube-root spread (~3.17)
    // fits well inside one 6.0-edge spatial cell, so every agent is
    // a spatial neighbour of every other agent. The body-form spatial
    // walk emits one PriceObserved + one PriceGossip per (self,
    // candidate) candidate slot in the 27-cell neighbourhood; every
    // candidate sets a bit in the observer's row of `price_belief`
    // via the `|=` accumulator. Result: ALL N×N cells flip to 1u
    // (diagonal AND off-diagonal). The diagonal flips because each
    // agent's spatial walk encounters its own slot in its own cell
    // (no per-pair "exclude self" filter at the body-iter site
    // today; same as particle_collision_min's emit pattern).
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
    let mut offdiag_zero = 0usize;
    let mut offdiag_bad: Vec<(usize, usize, u32)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let v = pb[i * n + j];
            if v == OBSERVATION_BIT {
                offdiag_ok += 1;
            } else if v == 0 {
                offdiag_zero += 1;
            } else {
                offdiag_bad.push((i, j, v));
            }
        }
    }
    println!(
        "trade_market_app: price_belief    diagonal {}/{} == {}u, \
         off-diagonal {}/{} == {}u (zero={}, other={})",
        diag_ok,
        n,
        OBSERVATION_BIT,
        offdiag_ok,
        n * (n - 1),
        OBSERVATION_BIT,
        offdiag_zero,
        offdiag_bad.len(),
    );
    if !diag_bad.is_empty() {
        println!(
            "trade_market_app: price_belief diagonal mismatches (first 8): {:?}",
            &diag_bad[..diag_bad.len().min(8)],
        );
    }
    if !offdiag_bad.is_empty() {
        println!(
            "trade_market_app: price_belief off-diagonal unexpected values (first 8): {:?}",
            &offdiag_bad[..offdiag_bad.len().min(8)],
        );
    }

    // ---- Outcome classification ----
    let trader_band = tv_steady * 0.05;
    let trader_ok = (tv_mean - tv_steady).abs() <= trader_band
        && (tv_min - tv_steady).abs() <= trader_band
        && (tv_max - tv_steady).abs() <= trader_band;
    let hub_band = hv_per_slot_expected * 0.05;
    let hub_ok = (hv_mean - hv_per_slot_expected).abs() <= hub_band
        && (hv_min - hv_per_slot_expected).abs() <= hub_band
        && (hv_max - hv_per_slot_expected).abs() <= hub_band;
    // v2: every diagonal AND off-diagonal cell flips to 1u under the
    // auto-spread + 6.0-cell-edge spatial grid (all 32 agents fit
    // inside one cell ⇒ every pair is a spatial neighbour). If a
    // future seed/agent_count combination spreads agents across
    // multiple cells, the predicate would weaken to "≥ N off-
    // diagonal cells flipped" + "diagonal fully set"; for the
    // pinned probe seed we keep the strict full-fire predicate.
    let belief_ok = diag_ok == n && offdiag_ok == n * (n - 1);

    println!(
        "trade_market_app: per-view checks — trader_volume={} hub_volume={} price_belief={}",
        if trader_ok { "OK" } else { "FAIL" },
        if hub_ok { "OK" } else { "FAIL" },
        if belief_ok { "OK" } else { "FAIL" },
    );

    if trader_ok && hub_ok && belief_ok {
        println!(
            "trade_market_app: OUTCOME = (a) FULL FIRE — verb cascade end-to-end \
             (mask -> scoring -> chronicle -> TradeExecuted -> trader/hub_volume folds) \
             integrated with multi-emit spatial body-form physics (WanderAndTrade \
             emits PriceObserved + PriceGossip per spatial-neighbour candidate; \
             both fold into the same price_belief pair_map u32 view via per-handler \
             tag filter)."
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
    // prints first regardless of which view fails.
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
