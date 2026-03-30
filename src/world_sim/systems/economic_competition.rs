//! Faction economic competition system — ticks every 17 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/economic_competition.rs`.
//! Factions compete for trade dominance, resource control, and market share,
//! creating economic warfare that affects settlement prices and treasuries.
//!
//!   (id, name, military_strength, territory_size, diplomatic_stance,
//!    relationship_to_guild, at_war_with)
//!   (faction_id, market_share, trade_routes_controlled, resource_monopolies)
//!   (faction_a, faction_b, trade_war, embargo, price_war, started_tick)

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::NUM_COMMODITIES;

/// Tick cadence for economic competition.
const TICK_CADENCE: u64 = 17;

/// Trade war income penalty — applied as price inflation.
const TRADE_WAR_PRICE_INFLATION: f32 = 0.05;

/// Embargo effect — prices spike at affected settlements.
const EMBARGO_PRICE_SPIKE: f32 = 0.10;

/// Price war effect — prices drop at competing settlements.
const PRICE_WAR_DEFLATION: f32 = 0.03;

/// Compute economic competition deltas.
///
/// Without explicit faction and rivalry state, this system models
/// inter-settlement economic competition: settlements with high
/// treasury compete by adjusting prices, and settlements with low
/// treasury suffer trade penalties.
///
/// When two settlements have large treasury disparity, the wealthier
/// one imposes competitive pressure (price reduction) while the poorer
/// one sees price inflation.
pub fn compute_economic_competition(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % TICK_CADENCE != 0 {
        return;
    }

    if state.settlements.len() < 2 {
        return;
    }

    // Compute average treasury across settlements.
    let avg_treasury: f32 = state
        .settlements
        .iter()
        .map(|s| s.treasury)
        .sum::<f32>()
        / state.settlements.len() as f32;

    if avg_treasury <= 0.0 {
        return;
    }

    for settlement in &state.settlements {
        let wealth_ratio = settlement.treasury / avg_treasury.max(1.0);

        if wealth_ratio > 1.5 {
            // Wealthy settlement: competitive advantage — prices drop,
            // drawing trade away from poorer settlements.
            let mut deflated_prices = settlement.prices;
            for c in 0..NUM_COMMODITIES {
                deflated_prices[c] *= 1.0 - PRICE_WAR_DEFLATION;
                deflated_prices[c] = deflated_prices[c].max(0.01);
            }
            out.push(WorldDelta::UpdatePrices {
                settlement_id: settlement.id,
                prices: deflated_prices,
            });
        } else if wealth_ratio < 0.5 && settlement.treasury > 0.0 {
            // Poor settlement: economic pressure — prices inflate,
            // treasury drains from trade disruption.
            let mut inflated_prices = settlement.prices;
            for c in 0..NUM_COMMODITIES {
                inflated_prices[c] *= 1.0 + TRADE_WAR_PRICE_INFLATION;
            }
            out.push(WorldDelta::UpdatePrices {
                settlement_id: settlement.id,
                prices: inflated_prices,
            });

            // Treasury drain from competitive disadvantage (only above floor).
            if settlement.treasury > -100.0 {
                let drain = settlement.treasury.max(0.0) * 0.01;
                out.push(WorldDelta::UpdateTreasury {
                    settlement_id: settlement.id,
                    delta: -drain,
                });
            }
        }
    }

    // --- Region-level competition: regions with high threat drain nearby
    //     settlement treasuries (economic disruption from conflict). ---
    for region in &state.regions {
        if region.threat_level <= 30.0 {
            continue;
        }

        // Find settlements near this region.
        for settlement in &state.settlements {
            // Proximity heuristic.
            let dx = settlement.pos.0 - 0.0; // Regions don't have pos yet; use origin.
            let dy = settlement.pos.1 - 0.0;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < 100.0 && settlement.treasury > -100.0 {
                let disruption = region.threat_level * 0.001;
                out.push(WorldDelta::UpdateTreasury {
                    settlement_id: settlement.id,
                    delta: -disruption,
                });
            }
        }
    }
}
