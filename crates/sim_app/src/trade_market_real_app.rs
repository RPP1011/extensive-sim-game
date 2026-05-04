//! Trade market real harness — drives `trade_market_real_runtime` for
//! 200+ ticks and reports the per-N-tick combat log + final wealth /
//! price distribution.
//!
//! ## Predicted observable shapes
//!
//! ### (a) FULL FIRE — full cascade plays out
//!
//! 50 Traders start at hp=wealth=100, mana=trader_marker=1.
//! 10 Goods start at hp=price=5+g (5..14), mana=550 (= role_marker +
//! quantity_50). Per tick:
//!   - Each trader's mask bit is set (manually, sidestepping the mask
//!     k=1 limit — see runtime comment).
//!   - Scoring inner loop over 60 candidates picks the lowest hp slot
//!     for each trader: typically good 0 (price=5) or whatever good
//!     currently has the lowest price.
//!   - Chronicle emits a Trade event per trader per tick.
//!   - ApplyTrade decrements buyer wealth by price, decrements seller
//!     mana by 1.0, and increments seller hp by price_step (= 0.5).
//!
//! After T ticks: total trade volume ≈ T * NUM_TRADERS = 50T (capped
//! by the event ring's per-tick slot budget). The cheapest good's
//! price drifts up by price_step per buy until a sibling good
//! becomes cheaper; the market should oscillate as different goods
//! take turns being cheapest.
//!
//! ### (b) NO FIRE — gap surfaces
//!
//! - trader_volume[*] all zero  → Trade events never emitted; the
//!   chronicle gate (`action_id == 0u`) never fires. Likely cause:
//!   scoring's mask gate skips actors (mask bit not set despite our
//!   override) OR scoring's inner loop never picks a target (best_
//!   utility stays at -inf).
//! - trader_volume > 0 but agent_hp[buyer] unchanged → the ApplyTrade
//!   kernel binding shape mismatch swallowed the writes.

use trade_market_real_runtime::{
    TradeMarketRealState, AGENT_COUNT, INITIAL_WEALTH, NUM_GOODS, NUM_TRADERS,
};
use engine::CompiledSim;

const SEED: u64 = 0xC0FF_EE_5_DEC1_DEC1;
const MAX_TICKS: u64 = 200;
const LOG_EVERY_N: u64 = 25;

fn main() {
    let mut sim = TradeMarketRealState::new(SEED, AGENT_COUNT);
    println!("================================================================");
    println!(" TRADE MARKET REAL — {} traders + {} goods", NUM_TRADERS, NUM_GOODS);
    println!("   seed=0x{:016X} agents={} max_ticks={}", SEED, AGENT_COUNT, MAX_TICKS);
    println!("   initial wealth/trader={:.1}, goods priced 5..14, qty 50/each",
        INITIAL_WEALTH);
    println!("================================================================");

    log_state(&mut sim, "init");

    // First-tick probe: how many actors picked a non-sentinel target?
    // Helps catch a (b) gap chain (mask kernel never sets bits) before
    // the per-N-tick aggregate masks the symptom.
    sim.step();
    {
        let scoring = sim.read_scoring_output();
        let active: usize = (0..(NUM_TRADERS + NUM_GOODS) as usize)
            .filter(|i| scoring[i * 4 + 1] != 0xFFFFFFFFu32)
            .count();
        let first_active = (0..(NUM_TRADERS + NUM_GOODS) as usize)
            .find(|i| scoring[i * 4 + 1] != 0xFFFFFFFFu32);
        println!(
            "  [t=  1] scoring active actors={} first=(actor={:?}, target={:?})",
            active, first_active,
            first_active.map(|a| scoring[a * 4 + 1]),
        );
    }
    for tick in 2..=MAX_TICKS {
        sim.step();
        if tick % LOG_EVERY_N == 0 {
            log_state(&mut sim, &format!("t={tick:>3}"));
        }
    }

    println!();
    println!("================================================================");
    println!(" FINAL RESULTS");
    println!("================================================================");

    let final_hp = sim.read_hp();
    let final_mana = sim.read_mana();
    let final_alive = sim.read_alive();
    let trader_volume = sim.trader_volume().to_vec();
    let good_revenue = sim.good_revenue().to_vec();

    let total_trades: f32 = trader_volume.iter().take(NUM_TRADERS as usize).sum();
    let total_revenue: f32 = good_revenue.iter().skip(NUM_TRADERS as usize).sum();

    println!("  Total trades:    {:.0}", total_trades);
    println!("  Total revenue:   {:.2} gold", total_revenue);
    println!("  Avg trades/trader: {:.2}", total_trades / NUM_TRADERS as f32);
    println!();

    // Top-5 traders by trade count
    let mut trader_ranking: Vec<(usize, f32, f32, u32)> = (0..NUM_TRADERS as usize)
        .map(|i| (i, trader_volume[i], final_hp[i], final_alive[i]))
        .collect();
    trader_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("  Top-5 traders by purchase count:");
    for (slot, count, hp, alive) in trader_ranking.iter().take(5) {
        let spent = INITIAL_WEALTH - hp;
        let bankrupt = if *alive == 0 { " [BANKRUPT]" } else { "" };
        println!(
            "    Trader #{:<3} buys={:>4.0} wealth={:>7.2} (spent {:>6.2}){}",
            slot, count, hp, spent, bankrupt,
        );
    }
    println!();

    let alive_traders = trader_ranking.iter().filter(|(_, _, _, a)| *a == 1).count();
    let bankrupt_traders = NUM_TRADERS as usize - alive_traders;
    println!("  Traders alive:   {} / {}", alive_traders, NUM_TRADERS);
    println!("  Traders bankrupt: {}", bankrupt_traders);
    println!();

    // Goods state
    println!("  Final goods state (slot, price, quantity_remaining, revenue):");
    let role_marker = 500.0_f32; // ROLE_MARKER from .sim
    for g in 0..NUM_GOODS as usize {
        let slot = NUM_TRADERS as usize + g;
        let price = final_hp[slot];
        let qty = (final_mana[slot] - role_marker).max(0.0);
        let revenue = good_revenue[slot];
        let initial_price = 5.0 + g as f32;
        let drift = price - initial_price;
        println!(
            "    Good #{:<2} (init={:>5.2}) price={:>6.2} (Δ{:+.2}) qty={:>5.1} revenue={:>7.2}",
            g, initial_price, price, drift, qty, revenue,
        );
    }

    println!();
    println!("================================================================");
    println!(" OUTCOME");
    println!("================================================================");

    let any_trade_fired = total_trades > 0.0;
    let any_wealth_dropped = final_hp.iter().take(NUM_TRADERS as usize)
        .any(|h| (INITIAL_WEALTH - h) > 0.01);
    let any_price_drifted = (0..NUM_GOODS as usize)
        .any(|g| {
            let slot = NUM_TRADERS as usize + g;
            let initial = 5.0 + g as f32;
            (final_hp[slot] - initial) > 0.01
        });

    if any_trade_fired && any_wealth_dropped && any_price_drifted {
        println!(
            "  (a) FULL FIRE — economic gameplay played out. {:.0} total \
             trades, {} traders' wealth dropped, prices drifted on at least \
             one good. Pair-field scoring picked CHEAPEST good across {} \
             traders × 60 candidate slots; ApplyTrade chronicle wrote both \
             buyer (wealth) and seller (price + quantity) sides.",
            total_trades, NUM_TRADERS, NUM_TRADERS,
        );
    } else if any_trade_fired {
        println!(
            "  (a-partial) TRADES FIRED — {:.0} Trade events emitted but not \
             all observables landed (wealth_dropped={}, price_drifted={}). \
             ApplyTrade may have skipped writes.",
            total_trades, any_wealth_dropped, any_price_drifted,
        );
    } else {
        println!(
            "  (b) NO TRADES — trader_volume empty. The verb cascade did \
             not reach the chronicle. Likely cause: scoring's mask gate or \
             chronicle's action_id check failed."
        );
    }

    assert!(
        any_trade_fired,
        "trade_market_real_app: ASSERTION FAILED — no trades fired \
         (total_trades=0). The verb cascade did not reach ApplyTrade."
    );
    assert!(
        any_wealth_dropped,
        "trade_market_real_app: ASSERTION FAILED — no trader wealth dropped. \
         ApplyTrade did not write to agent_hp."
    );
    assert!(
        any_price_drifted,
        "trade_market_real_app: ASSERTION FAILED — no good price drifted. \
         ApplyTrade's seller-side hp write did not land."
    );
}

fn log_state(sim: &mut TradeMarketRealState, label: &str) {
    let hp = sim.read_hp();
    let mana = sim.read_mana();
    let role_marker = 500.0_f32;

    let total_wealth: f32 = (0..NUM_TRADERS as usize).map(|i| hp[i]).sum();
    let avg_wealth = total_wealth / NUM_TRADERS as f32;
    let min_wealth = (0..NUM_TRADERS as usize).map(|i| hp[i]).fold(f32::INFINITY, f32::min);
    let max_wealth = (0..NUM_TRADERS as usize).map(|i| hp[i]).fold(f32::NEG_INFINITY, f32::max);

    let cheapest_good = (0..NUM_GOODS as usize)
        .map(|g| (g, hp[NUM_TRADERS as usize + g]))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let total_qty_remaining: f32 = (0..NUM_GOODS as usize)
        .map(|g| (mana[NUM_TRADERS as usize + g] - role_marker).max(0.0))
        .sum();
    let total_qty_initial = NUM_GOODS as f32 * 50.0;

    println!(
        "  [{:>5}] wealth: avg={:>6.2} min={:>6.2} max={:>6.2}  \
         cheapest_good=#{} @ {:.2}  qty_remaining={:.0}/{:.0}",
        label, avg_wealth, min_wealth, max_wealth,
        cheapest_good.0, cheapest_good.1,
        total_qty_remaining, total_qty_initial,
    );
}
