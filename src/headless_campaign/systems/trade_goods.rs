//! Regional trade goods system — ticks every 200 ticks.
//!
//! Each region produces unique goods. Supply regenerates based on region type,
//! demand fluctuates based on scarcity, and prices follow supply/demand curves.
//! Seasons and wars affect supply. Automated caravans move goods when profitable.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often the trade goods system ticks.
const TRADE_TICK_INTERVAL: u64 = 7;

/// Maximum price multiplier (price = base_price * demand/supply, capped here).
const MAX_PRICE_MULTIPLIER: f32 = 5.0;

/// Minimum supply floor to prevent division by zero.
const MIN_SUPPLY: f32 = 0.1;

/// Supply regeneration rate per tick.
const SUPPLY_REGEN_RATE: f32 = 0.05;

/// Demand drift rate per tick.
const DEMAND_DRIFT_RATE: f32 = 0.03;

/// Caravan profit threshold — only move goods if profit margin exceeds this.
const CARAVAN_PROFIT_THRESHOLD: f32 = 1.5;

/// Amount moved per caravan run.
const CARAVAN_AMOUNT: f32 = 2.0;

/// How much war reduces supply (multiplicative).
const WAR_SUPPLY_PENALTY: f32 = 0.5;

pub fn tick_trade_goods(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % TRADE_TICK_INTERVAL != 0 {
        return;
    }

    let season = state.overworld.season;
    let num_regions = state.overworld.regions.len();

    // Collect war status per region (region is at war if its owner faction is at war).
    let region_at_war: Vec<bool> = state
        .overworld
        .regions
        .iter()
        .map(|r| {
            state
                .factions
                .iter()
                .find(|f| f.id == r.owner_faction_id)
                .map(|f| !f.at_war_with.is_empty())
                .unwrap_or(false)
        })
        .collect();

    // --- Supply regeneration and seasonal/war modifiers ---
    for good in &mut state.trade_goods {
        if good.region_id >= num_regions {
            continue;
        }

        // Base regeneration
        good.supply += SUPPLY_REGEN_RATE;

        // Season modifier on supply
        let season_mult = season_supply_modifier(good.good_type, season);
        good.supply *= season_mult;

        // War disruption
        if good.region_id < region_at_war.len() && region_at_war[good.region_id] {
            good.supply *= WAR_SUPPLY_PENALTY;

            // Emit shortage event if supply is critically low
            if good.supply < 1.0 {
                events.push(WorldEvent::SupplyShortage {
                    region_id: good.region_id,
                    good_type: good.good_type.as_str().to_string(),
                });
            }
        }

        // Clamp supply
        good.supply = good.supply.max(MIN_SUPPLY).min(50.0);
    }

    // --- Demand fluctuation ---
    // Regions lacking a good type have increased demand for it.
    // Regions producing it have decreased demand.
    let trade_goods_snapshot: Vec<(usize, TradeGoodType, f32)> = state
        .trade_goods
        .iter()
        .map(|g| (g.region_id, g.good_type, g.supply))
        .collect();

    for good in &mut state.trade_goods {
        // Check if other regions have this good type abundantly
        let other_supply: f32 = trade_goods_snapshot
            .iter()
            .filter(|(rid, gt, _)| *rid != good.region_id && *gt == good.good_type)
            .map(|(_, _, s)| s)
            .sum();

        if other_supply < good.supply {
            // This region has more supply than others — demand decreases locally
            good.demand = (good.demand - DEMAND_DRIFT_RATE).max(0.5);
        } else {
            // Scarce locally — demand increases
            good.demand = (good.demand + DEMAND_DRIFT_RATE).min(10.0);
        }
    }

    // --- Automated caravans ---
    // For each good type, find the cheapest source and most expensive destination.
    // If the profit margin exceeds the threshold, move goods automatically.
    let good_types = [
        TradeGoodType::Grain,
        TradeGoodType::Ore,
        TradeGoodType::Timber,
        TradeGoodType::Silk,
        TradeGoodType::Spices,
        TradeGoodType::Wine,
        TradeGoodType::Gems,
        TradeGoodType::Livestock,
    ];

    for &gt in &good_types {
        // Find cheapest source (lowest price, sufficient supply)
        let source = state
            .trade_goods
            .iter()
            .enumerate()
            .filter(|(_, g)| g.good_type == gt && g.supply > CARAVAN_AMOUNT)
            .min_by(|(_, a), (_, b)| {
                let pa = trade_price(a);
                let pb = trade_price(b);
                pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, g)| (i, trade_price(g), g.region_id));

        // Find most expensive destination (highest price)
        let dest = state
            .trade_goods
            .iter()
            .enumerate()
            .filter(|(_, g)| g.good_type == gt)
            .max_by(|(_, a), (_, b)| {
                let pa = trade_price(a);
                let pb = trade_price(b);
                pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, g)| (i, trade_price(g), g.region_id));

        if let (Some((src_idx, src_price, src_rid)), Some((dst_idx, dst_price, dst_rid))) =
            (source, dest)
        {
            if src_idx != dst_idx
                && src_rid != dst_rid
                && dst_price / src_price.max(0.01) > CARAVAN_PROFIT_THRESHOLD
            {
                // Only move goods between neighboring regions
                let are_neighbors = state
                    .overworld
                    .regions
                    .get(src_rid)
                    .map(|r| r.neighbors.contains(&dst_rid))
                    .unwrap_or(false);

                if are_neighbors {
                    // Move goods: reduce source supply, increase dest supply
                    state.trade_goods[src_idx].supply -= CARAVAN_AMOUNT;
                    state.trade_goods[dst_idx].supply += CARAVAN_AMOUNT * 0.8; // 20% loss in transit

                    let profit = (dst_price - src_price) * CARAVAN_AMOUNT;
                    if profit > 0.0 {
                        // Guild gets a cut of caravan profits from guild-controlled regions
                        let guild_faction = state.diplomacy.guild_faction_id;
                        let src_is_guild = state
                            .overworld
                            .regions
                            .get(src_rid)
                            .map(|r| r.owner_faction_id == guild_faction)
                            .unwrap_or(false);
                        let dst_is_guild = state
                            .overworld
                            .regions
                            .get(dst_rid)
                            .map(|r| r.owner_faction_id == guild_faction)
                            .unwrap_or(false);

                        if src_is_guild || dst_is_guild {
                            let guild_cut = profit * 0.1; // 10% tax
                            state.guild.gold += guild_cut;
                            events.push(WorldEvent::TradeProfitMade {
                                source_region: src_rid,
                                dest_region: dst_rid,
                                good_type: gt.as_str().to_string(),
                                profit: guild_cut,
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Compute the current price of a trade good.
fn trade_price(good: &TradeGood) -> f32 {
    let ratio = good.demand / good.supply.max(MIN_SUPPLY);
    (good.base_price * ratio).min(good.base_price * MAX_PRICE_MULTIPLIER)
}

/// Compute the price the player pays/receives for a trade good.
pub fn player_trade_price(good: &TradeGood) -> f32 {
    trade_price(good)
}

/// Seasonal supply modifier per good type.
fn season_supply_modifier(good_type: TradeGoodType, season: Season) -> f32 {
    match (good_type, season) {
        // Grain: high in autumn (harvest), low in winter
        (TradeGoodType::Grain, Season::Autumn) => 1.3,
        (TradeGoodType::Grain, Season::Winter) => 0.6,
        (TradeGoodType::Grain, Season::Spring) => 0.9,
        (TradeGoodType::Grain, Season::Summer) => 1.1,

        // Livestock: low in winter (animals sheltered)
        (TradeGoodType::Livestock, Season::Winter) => 0.7,
        (TradeGoodType::Livestock, Season::Spring) => 1.2, // breeding season

        // Timber: hard to harvest in winter
        (TradeGoodType::Timber, Season::Winter) => 0.7,
        (TradeGoodType::Timber, Season::Summer) => 1.1,

        // Wine: harvest in autumn
        (TradeGoodType::Wine, Season::Autumn) => 1.4,
        (TradeGoodType::Wine, Season::Winter) => 0.8,

        // Ore/Gems: relatively stable, slightly harder in winter
        (TradeGoodType::Ore, Season::Winter) => 0.9,
        (TradeGoodType::Gems, Season::Winter) => 0.9,

        // Silk/Spices: affected by trade route weather
        (TradeGoodType::Silk, Season::Winter) => 0.7,
        (TradeGoodType::Spices, Season::Winter) => 0.7,

        // Default: no modifier
        _ => 1.0,
    }
}

/// Initialize trade goods for regions based on region names/characteristics.
/// Called during world template application.
pub fn init_trade_goods_for_regions(regions: &[Region], rng: &mut u64) -> Vec<TradeGood> {
    let mut goods = Vec::new();

    for region in regions {
        // Assign 2-3 goods per region based on region name heuristics
        let region_goods = goods_for_region(region, rng);
        for (good_type, base_supply, base_price) in region_goods {
            goods.push(TradeGood {
                good_type,
                region_id: region.id,
                supply: base_supply,
                demand: 1.0,
                base_price,
            });
        }
    }

    goods
}

/// Determine which goods a region produces based on its name.
fn goods_for_region(
    region: &Region,
    rng: &mut u64,
) -> Vec<(TradeGoodType, f32, f32)> {
    let name_lower = region.name.to_lowercase();

    // Primary goods based on region name heuristics
    let mut result = Vec::new();

    if name_lower.contains("green")
        || name_lower.contains("hollow")
        || name_lower.contains("meadow")
        || name_lower.contains("vale")
        || name_lower.contains("farm")
    {
        // Fertile region
        result.push((TradeGoodType::Grain, 8.0, 5.0));
        result.push((TradeGoodType::Livestock, 5.0, 12.0));
    } else if name_lower.contains("iron")
        || name_lower.contains("ridge")
        || name_lower.contains("mountain")
        || name_lower.contains("peak")
        || name_lower.contains("crag")
    {
        // Mountainous region
        result.push((TradeGoodType::Ore, 7.0, 8.0));
        result.push((TradeGoodType::Gems, 3.0, 25.0));
    } else if name_lower.contains("mist")
        || name_lower.contains("fen")
        || name_lower.contains("marsh")
        || name_lower.contains("wood")
        || name_lower.contains("forest")
    {
        // Forest/wetland region
        result.push((TradeGoodType::Timber, 8.0, 6.0));
        result.push((TradeGoodType::Spices, 3.0, 20.0));
    } else {
        // Default: generic trade hub
        result.push((TradeGoodType::Silk, 4.0, 15.0));
        result.push((TradeGoodType::Wine, 5.0, 10.0));
    }

    // Add a random third good using deterministic RNG
    let extra_roll = lcg_next(rng) % 8;
    let extra = match extra_roll {
        0 => (TradeGoodType::Grain, 3.0, 5.0),
        1 => (TradeGoodType::Ore, 3.0, 8.0),
        2 => (TradeGoodType::Timber, 3.0, 6.0),
        3 => (TradeGoodType::Silk, 2.0, 15.0),
        4 => (TradeGoodType::Spices, 2.0, 20.0),
        5 => (TradeGoodType::Wine, 3.0, 10.0),
        6 => (TradeGoodType::Gems, 1.0, 25.0),
        _ => (TradeGoodType::Livestock, 3.0, 12.0),
    };

    // Don't duplicate existing good types
    if !result.iter().any(|(gt, _, _)| *gt == extra.0) {
        result.push(extra);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trade_price_calculation() {
        let good = TradeGood {
            good_type: TradeGoodType::Grain,
            region_id: 0,
            supply: 5.0,
            demand: 5.0,
            base_price: 10.0,
        };
        // demand/supply = 1.0, so price = base_price
        assert!((trade_price(&good) - 10.0).abs() < 0.01);
    }

    #[test]
    fn trade_price_capped() {
        let good = TradeGood {
            good_type: TradeGoodType::Gems,
            region_id: 0,
            supply: MIN_SUPPLY,
            demand: 100.0,
            base_price: 10.0,
        };
        // Should be capped at 5x base
        assert!((trade_price(&good) - 50.0).abs() < 0.01);
    }

    #[test]
    fn season_modifiers_reasonable() {
        for gt in [
            TradeGoodType::Grain,
            TradeGoodType::Ore,
            TradeGoodType::Timber,
        ] {
            for season in [Season::Spring, Season::Summer, Season::Autumn, Season::Winter] {
                let m = season_supply_modifier(gt, season);
                assert!(m > 0.0 && m <= 2.0, "modifier {m} out of range for {gt:?} {season:?}");
            }
        }
    }

    #[test]
    fn init_produces_goods() {
        let regions = vec![
            Region {
                id: 0,
                name: "Greenhollow".into(),
                owner_faction_id: 0,
                neighbors: vec![1],
                unrest: 0.0,
                control: 80.0,
                threat_level: 10.0,
                visibility: 0.8,
            },
            Region {
                id: 1,
                name: "Ironridge".into(),
                owner_faction_id: 1,
                neighbors: vec![0],
                unrest: 0.0,
                control: 60.0,
                threat_level: 30.0,
                visibility: 0.3,
            },
        ];
        let mut rng = 42u64;
        let goods = init_trade_goods_for_regions(&regions, &mut rng);
        assert!(goods.len() >= 4, "Expected at least 4 goods, got {}", goods.len());
        // Greenhollow should have Grain
        assert!(goods.iter().any(|g| g.good_type == TradeGoodType::Grain && g.region_id == 0));
        // Ironridge should have Ore
        assert!(goods.iter().any(|g| g.good_type == TradeGoodType::Ore && g.region_id == 1));
    }
}
