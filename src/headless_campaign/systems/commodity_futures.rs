//! Commodity futures market — ticks every 500 ticks.
//!
//! Guilds can hold futures contracts on trade goods. On settlement, profit/loss
//! is the difference between the strike price and the current market price.
//! New contracts are generated based on current trade goods prices with a
//! random premium or discount.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;
use crate::headless_campaign::systems::trade_goods::player_trade_price;

/// How often the futures market ticks (settle + generate).
const FUTURES_TICK_INTERVAL: u64 = 17;

/// Maximum number of active contracts the guild can hold.
const MAX_ACTIVE_CONTRACTS: usize = 5;

/// Number of new contracts offered each cycle.
const CONTRACTS_OFFERED_PER_CYCLE: usize = 3;

/// Maximum premium/discount as a fraction of market price (±30%).
const MAX_PREMIUM_FRACTION: f32 = 0.3;

/// How many ticks until a newly offered contract settles.
const SETTLEMENT_HORIZON: u64 = 17;

pub fn tick_commodity_futures(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % FUTURES_TICK_INTERVAL != 0 {
        return;
    }

    // --- Phase 1: Settle mature contracts ---
    settle_contracts(state, events);

    // --- Phase 2: Generate new contract offers ---
    generate_contracts(state, events);
}

/// Settle all contracts whose settlement tick has arrived.
/// Profit/loss = (market_price - strike_price) * quantity for buy contracts,
/// inverted for sell contracts.
fn settle_contracts(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let current_tick = state.tick;

    // Collect contracts that are due for settlement.
    let (settling, remaining): (Vec<_>, Vec<_>) = state
        .futures_contracts
        .drain(..)
        .partition(|c| c.settlement_tick <= current_tick);

    state.futures_contracts = remaining;

    for contract in settling {
        // Find current market price for this good type (average across regions).
        let market_price = avg_market_price(&state.trade_goods, contract.good_type);

        let profit = if contract.is_buy {
            // Buy contract: guild profits when market price > strike price.
            (market_price - contract.strike_price) * contract.quantity
        } else {
            // Sell contract: guild profits when market price < strike price.
            (contract.strike_price - market_price) * contract.quantity
        };

        state.guild.gold += profit;
        state.futures_profit_total += profit;

        events.push(WorldEvent::FuturesContractSettled {
            contract_id: contract.contract_id,
            profit,
        });
    }
}

/// Generate new contract offers based on current trade good prices.
fn generate_contracts(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Don't generate if guild already holds max contracts.
    if state.futures_contracts.len() >= MAX_ACTIVE_CONTRACTS {
        return;
    }

    if state.trade_goods.is_empty() {
        return;
    }

    let settlement_tick = state.tick + SETTLEMENT_HORIZON;
    let num_goods = state.trade_goods.len();

    let slots_available = MAX_ACTIVE_CONTRACTS - state.futures_contracts.len();
    let to_generate = CONTRACTS_OFFERED_PER_CYCLE.min(slots_available);

    for _ in 0..to_generate {
        // Pick a random trade good as the basis.
        let idx = lcg_next(&mut state.rng) as usize % num_goods;
        let good = &state.trade_goods[idx];
        let good_type = good.good_type;
        let market_price = player_trade_price(good);

        // Random premium/discount: -30% to +30%.
        let r = lcg_f32(&mut state.rng); // 0.0 .. 1.0
        let premium = (r * 2.0 - 1.0) * MAX_PREMIUM_FRACTION; // -0.3 .. +0.3
        let strike_price = (market_price * (1.0 + premium)).max(0.01);

        // Random quantity: 1.0 to 5.0.
        let quantity = 1.0 + lcg_f32(&mut state.rng) * 4.0;

        // Random direction: buy or sell.
        let is_buy = lcg_next(&mut state.rng) % 2 == 0;

        let contract_id = state.next_futures_contract_id;
        state.next_futures_contract_id += 1;

        let contract = FuturesContract {
            contract_id,
            good_type,
            quantity,
            strike_price,
            settlement_tick,
            is_buy,
        };

        state.futures_contracts.push(contract);

        events.push(WorldEvent::FuturesContractOffered {
            contract_id,
            good_type: good_type.as_str().to_string(),
            strike_price,
        });
    }
}

/// Average market price for a good type across all regions.
fn avg_market_price(trade_goods: &[TradeGood], good_type: TradeGoodType) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0u32;
    for good in trade_goods {
        if good.good_type == good_type {
            total += player_trade_price(good);
            count += 1;
        }
    }
    if count == 0 {
        1.0 // Fallback if no goods of this type exist.
    } else {
        total / count as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::actions::StepDeltas;

    fn make_test_state() -> CampaignState {
        let mut state = CampaignState::default_test_campaign(42);
        // Add some trade goods so contracts can reference them.
        state.trade_goods.push(TradeGood {
            good_type: TradeGoodType::Grain,
            region_id: 0,
            supply: 5.0,
            demand: 5.0,
            base_price: 10.0,
        });
        state.trade_goods.push(TradeGood {
            good_type: TradeGoodType::Ore,
            region_id: 1,
            supply: 3.0,
            demand: 6.0,
            base_price: 8.0,
        });
        state
    }

    #[test]
    fn no_tick_before_interval() {
        let mut state = make_test_state();
        state.tick = 1;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_commodity_futures(&mut state, &mut deltas, &mut events);
        assert!(events.is_empty());
        assert!(state.futures_contracts.is_empty());
    }

    #[test]
    fn generates_contracts_at_interval() {
        let mut state = make_test_state();
        state.tick = 500;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_commodity_futures(&mut state, &mut deltas, &mut events);
        assert!(!state.futures_contracts.is_empty());
        assert!(state.futures_contracts.len() <= MAX_ACTIVE_CONTRACTS);
        // All generated contracts should have settlement at tick 1000.
        for c in &state.futures_contracts {
            assert_eq!(c.settlement_tick, 1000);
        }
    }

    #[test]
    fn settles_mature_contracts() {
        let mut state = make_test_state();
        let gold_before = state.guild.gold;
        // Manually add a contract that settles at tick 500.
        state.futures_contracts.push(FuturesContract {
            contract_id: 1,
            good_type: TradeGoodType::Grain,
            quantity: 2.0,
            strike_price: 5.0, // Below market price (10.0), so buy = profit.
            settlement_tick: 500,
            is_buy: true,
        });
        state.tick = 500;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_commodity_futures(&mut state, &mut deltas, &mut events);
        // Should have settled with profit.
        let settled_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, WorldEvent::FuturesContractSettled { .. }))
            .collect();
        assert_eq!(settled_events.len(), 1);
        // Profit = (10.0 - 5.0) * 2.0 = 10.0.
        assert!(state.guild.gold > gold_before);
    }

    #[test]
    fn max_contracts_respected() {
        let mut state = make_test_state();
        // Fill up to max contracts.
        for i in 0..MAX_ACTIVE_CONTRACTS {
            state.futures_contracts.push(FuturesContract {
                contract_id: i as u32 + 100,
                good_type: TradeGoodType::Grain,
                quantity: 1.0,
                strike_price: 10.0,
                settlement_tick: 9999,
                is_buy: true,
            });
        }
        state.tick = 500;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_commodity_futures(&mut state, &mut deltas, &mut events);
        // Should not generate more contracts.
        assert_eq!(state.futures_contracts.len(), MAX_ACTIVE_CONTRACTS);
        let offered: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, WorldEvent::FuturesContractOffered { .. }))
            .collect();
        assert!(offered.is_empty());
    }
}
