//! Price controls system — council sets ceiling/floor prices on commodities.
//!
//! Every 500 ticks, evaluates whether the council should enact new price controls
//! and applies the effects of existing ones. Binding controls create shortages,
//! black market premiums, and treasury drain. Controls expire after 2000 ticks.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;
use crate::systems::trade_goods::player_trade_price;

/// How often the price controls system ticks (in ticks).
const PRICE_CONTROL_INTERVAL: u64 = 17;

/// How long a price control lasts before automatic expiry (in ticks).
const CONTROL_EXPIRY_TICKS: u64 = 67;

/// Supply reduction per tick when a price ceiling is binding.
const CEILING_SUPPLY_REDUCTION: f32 = 0.10;

/// Black market premium when a ceiling is binding (50% above ceiling).
const BLACK_MARKET_PREMIUM_PCT: f32 = 50.0;

/// Price gap threshold (30%) above which black market heat increases.
const BLACK_MARKET_GAP_THRESHOLD: f32 = 0.30;

/// Heat increase per tick when price gap exceeds threshold.
const HEAT_PER_BINDING_TICK: f32 = 0.1;

/// Treasury drain per tick when a price floor requires subsidies.
const FLOOR_SUBSIDY_DRAIN: f32 = 2.0;

/// Price spike threshold — council enacts controls when price > 2x baseline.
const PRICE_SPIKE_THRESHOLD: f32 = 2.0;

/// Unrest threshold — council enacts controls when avg region unrest > 50.
const UNREST_THRESHOLD: f32 = 50.0;

/// Tick the price controls system every `PRICE_CONTROL_INTERVAL` ticks.
pub fn tick_price_controls(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % PRICE_CONTROL_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Don't activate before tick 1000 (early game grace period, same as black market)
    if state.tick < 1000 {
        return;
    }

    // --- Expire old controls ---
    expire_controls(state, events);

    // --- Enforce existing controls ---
    enforce_controls(state, events);

    // --- Council auto-enacts new controls when conditions are met ---
    auto_enact_controls(state, events);
}

/// Remove controls that have exceeded their expiry duration.
fn expire_controls(state: &mut CampaignState, _events: &mut Vec<WorldEvent>) {
    let tick = state.tick;
    state.price_controls.retain(|pc| {
        let age = tick.saturating_sub(pc.enacted_tick as u64);
        age < CONTROL_EXPIRY_TICKS
    });
}

/// Enforce existing price controls — apply shortage/surplus effects.
fn enforce_controls(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Snapshot trade goods prices to avoid borrow issues.
    let price_snapshot: Vec<(String, f32)> = state
        .trade_goods
        .iter()
        .map(|g| (g.good_type.as_str().to_string(), player_trade_price(g)))
        .collect();

    // Compute average price per good type.
    let mut price_sums: std::collections::HashMap<String, (f32, usize)> =
        std::collections::HashMap::new();
    for (gt, price) in &price_snapshot {
        let entry = price_sums.entry(gt.clone()).or_insert((0.0, 0));
        entry.0 += price;
        entry.1 += 1;
    }
    let avg_prices: std::collections::HashMap<String, f32> = price_sums
        .into_iter()
        .map(|(k, (sum, count))| (k, sum / count.max(1) as f32))
        .collect();

    for control in &state.price_controls {
        let avg_price = match avg_prices.get(&control.good_type) {
            Some(&p) => p,
            None => continue,
        };

        // --- Price ceiling enforcement ---
        if let Some(ceiling) = control.ceiling {
            if avg_price > ceiling {
                // Ceiling is binding — apply shortage effects
                let gap_pct = (avg_price - ceiling) / ceiling.max(0.01);

                // Reduce supply for all goods of this type
                for good in &mut state.trade_goods {
                    if good.good_type.as_str() == control.good_type {
                        good.supply *= 1.0 - CEILING_SUPPLY_REDUCTION;
                        good.supply = good.supply.max(0.1);
                    }
                }

                events.push(WorldEvent::PriceControlViolation {
                    good_type: control.good_type.clone(),
                    actual_price: avg_price,
                    controlled_price: ceiling,
                });

                // Black market heat increase when gap > 30%
                if gap_pct > BLACK_MARKET_GAP_THRESHOLD {
                    state.black_market.heat += HEAT_PER_BINDING_TICK;

                    events.push(WorldEvent::BlackMarketSurge {
                        good_type: control.good_type.clone(),
                        premium_pct: BLACK_MARKET_PREMIUM_PCT,
                    });
                }
            }
        }

        // --- Price floor enforcement ---
        if let Some(floor) = control.floor {
            if avg_price < floor {
                // Floor is binding — treasury subsidies required
                let subsidy = FLOOR_SUBSIDY_DRAIN;
                state.guild.gold = (state.guild.gold - subsidy).max(0.0);

                events.push(WorldEvent::PriceControlViolation {
                    good_type: control.good_type.clone(),
                    actual_price: avg_price,
                    controlled_price: floor,
                });

                // Floor binding also creates black market gap (goods sold below floor)
                let gap_pct = (floor - avg_price) / floor.max(0.01);
                if gap_pct > BLACK_MARKET_GAP_THRESHOLD {
                    state.black_market.heat += HEAT_PER_BINDING_TICK;

                    events.push(WorldEvent::BlackMarketSurge {
                        good_type: control.good_type.clone(),
                        premium_pct: gap_pct * 100.0,
                    });
                }
            }
        }
    }
}

/// Council auto-enacts price controls when conditions are met:
/// - Price of food/supplies spikes > 2x baseline
/// - Average region unrest > 50
fn auto_enact_controls(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;

    // Check if any good type already has an active control.
    let controlled_types: Vec<String> = state
        .price_controls
        .iter()
        .map(|pc| pc.good_type.clone())
        .collect();

    // Compute average unrest across all regions.
    let avg_unrest = if state.overworld.regions.is_empty() {
        0.0
    } else {
        state.overworld.regions.iter().map(|r| r.unrest).sum::<f32>()
            / state.overworld.regions.len() as f32
    };

    // Check each trade good for price spikes.
    // Group by good type to find average price vs base price.
    let mut type_prices: std::collections::HashMap<String, (f32, f32, usize)> =
        std::collections::HashMap::new();
    for good in &state.trade_goods {
        let gt = good.good_type.as_str().to_string();
        let price = player_trade_price(good);
        let entry = type_prices.entry(gt).or_insert((0.0, good.base_price, 0));
        entry.0 += price;
        entry.2 += 1;
    }

    // Essential goods that the council cares about (food and basic supplies).
    let essential_goods = ["grain", "livestock", "timber"];

    for (good_type, (price_sum, base_price, count)) in &type_prices {
        if controlled_types.contains(good_type) {
            continue; // Already controlled
        }

        let avg_price = price_sum / (*count).max(1) as f32;
        let price_ratio = avg_price / base_price.max(0.01);

        // Enact ceiling if price spikes > 2x for essential goods
        let is_essential = essential_goods.contains(&good_type.as_str());
        if is_essential && price_ratio > PRICE_SPIKE_THRESHOLD {
            let ceiling = base_price * 1.5; // Set ceiling at 1.5x base
            state.price_controls.push(PriceControl {
                good_type: good_type.clone(),
                ceiling: Some(ceiling),
                floor: None,
                enacted_tick: tick as u32,
            });

            events.push(WorldEvent::PriceControlEnacted {
                good_type: good_type.clone(),
                ceiling: Some(ceiling),
                floor: None,
            });
        }

        // Enact floor if unrest is high and price has crashed (< 0.5x base)
        if avg_unrest > UNREST_THRESHOLD && price_ratio < 0.5 {
            let floor = base_price * 0.8; // Set floor at 0.8x base
            state.price_controls.push(PriceControl {
                good_type: good_type.clone(),
                ceiling: None,
                floor: Some(floor),
                enacted_tick: tick as u32,
            });

            events.push(WorldEvent::PriceControlEnacted {
                good_type: good_type.clone(),
                ceiling: None,
                floor: Some(floor),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_state() -> CampaignState {
        let mut state = CampaignState::default_test_campaign(42);
        state.tick = 1500; // Past early-game grace period

        // Add some trade goods
        state.trade_goods.push(TradeGood {
            good_type: TradeGoodType::Grain,
            region_id: 0,
            supply: 2.0,
            demand: 8.0,
            base_price: 5.0,
        });
        state.trade_goods.push(TradeGood {
            good_type: TradeGoodType::Ore,
            region_id: 1,
            supply: 5.0,
            demand: 5.0,
            base_price: 8.0,
        });

        // Add a region for unrest checks
        state.overworld.regions.push(Region {
            id: 0,
            name: "Testland".into(),
            owner_faction_id: 0,
            neighbors: vec![],
            unrest: 0.0,
            control: 80.0,
            threat_level: 10.0,
            visibility: 1.0,
            population: 500,
            civilian_morale: 50.0,
            tax_rate: 0.1,
            growth_rate: 0.0,
        });

        state
    }

    #[test]
    fn ceiling_control_reduces_supply() {
        let mut state = make_test_state();
        // Grain price = base(5) * demand(8)/supply(2) = 20, capped at 5*5=25
        // Set ceiling at 10 — should be binding
        state.price_controls.push(PriceControl {
            good_type: "grain".into(),
            ceiling: Some(10.0),
            floor: None,
            enacted_tick: 1000,
        });

        let original_supply = state.trade_goods[0].supply;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_price_controls(&mut state, &mut deltas, &mut events);

        // Supply should have decreased
        assert!(state.trade_goods[0].supply < original_supply);
        // Should have violation event
        assert!(events.iter().any(|e| matches!(e, WorldEvent::PriceControlViolation { .. })));
    }

    #[test]
    fn floor_control_drains_treasury() {
        let mut state = make_test_state();
        state.guild.gold = 100.0;

        // Ore price = base(8) * demand(5)/supply(5) = 8
        // Set floor at 20 — should be binding
        state.price_controls.push(PriceControl {
            good_type: "ore".into(),
            ceiling: None,
            floor: Some(20.0),
            enacted_tick: 1000,
        });

        let original_gold = state.guild.gold;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_price_controls(&mut state, &mut deltas, &mut events);

        // Gold should have decreased from subsidies
        assert!(state.guild.gold < original_gold);
        assert!(events.iter().any(|e| matches!(e, WorldEvent::PriceControlViolation { .. })));
    }

    #[test]
    fn controls_expire_after_2000_ticks() {
        let mut state = make_test_state();
        state.tick = 4000; // Well past expiry

        state.price_controls.push(PriceControl {
            good_type: "grain".into(),
            ceiling: Some(10.0),
            floor: None,
            enacted_tick: 1000, // enacted at 1000, now at 4000 => 3000 ticks old > 2000
        });

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_price_controls(&mut state, &mut deltas, &mut events);

        assert!(state.price_controls.is_empty());
    }

    #[test]
    fn price_spike_auto_enacts_ceiling() {
        let mut state = make_test_state();
        // Grain has very high demand/supply ratio → price spike > 2x
        // Price = 5 * (8/2) = 20, ratio = 20/5 = 4.0 > 2.0 threshold

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_price_controls(&mut state, &mut deltas, &mut events);

        // Should have auto-enacted a ceiling on grain
        assert!(state.price_controls.iter().any(|pc| pc.good_type == "grain" && pc.ceiling.is_some()));
        assert!(events.iter().any(|e| matches!(e, WorldEvent::PriceControlEnacted { .. })));
    }

    #[test]
    fn black_market_heat_increases_on_large_gap() {
        let mut state = make_test_state();
        let initial_heat = state.black_market.heat;

        // Grain price = 20 (capped at 25), ceiling at 5 → gap = (20-5)/5 = 3.0 > 0.3
        state.price_controls.push(PriceControl {
            good_type: "grain".into(),
            ceiling: Some(5.0),
            floor: None,
            enacted_tick: 1000,
        });

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_price_controls(&mut state, &mut deltas, &mut events);

        assert!(state.black_market.heat > initial_heat);
        assert!(events.iter().any(|e| matches!(e, WorldEvent::BlackMarketSurge { .. })));
    }

    #[test]
    fn non_essential_goods_no_auto_ceiling() {
        let mut state = make_test_state();
        // Make ore have a huge price spike (ore is non-essential)
        state.trade_goods[1].demand = 20.0;
        state.trade_goods[1].supply = 1.0;

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_price_controls(&mut state, &mut deltas, &mut events);

        // Ore should NOT get auto-enacted controls (not essential)
        assert!(!state.price_controls.iter().any(|pc| pc.good_type == "ore"));
    }

    #[test]
    fn interval_gating() {
        let mut state = make_test_state();
        state.tick = 1501; // Not a multiple of 500

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_price_controls(&mut state, &mut deltas, &mut events);

        // Nothing should happen on non-interval ticks
        assert!(events.is_empty());
    }
}
