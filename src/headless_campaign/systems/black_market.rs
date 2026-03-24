//! Black market system — fires every 500 ticks (~50s game time).
//!
//! Illegal but lucrative trade with high gold returns but reputation risk.
//! Deals refresh every 1000 ticks, and accumulating heat risks discovery
//! which hurts reputation and faction relations with lawful factions.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often the black market ticks (in ticks).
const BLACK_MARKET_INTERVAL: u64 = 500;

/// How often deals refresh (in ticks).
const DEAL_REFRESH_INTERVAL: u64 = 1000;

/// Heat decay per tick interval when not dealing.
const HEAT_DECAY_PER_TICK: f32 = 2.0;

/// Heat threshold above which discovery checks begin.
const DISCOVERY_THRESHOLD: f32 = 50.0;

/// Reputation loss on discovery.
const DISCOVERY_REPUTATION_LOSS: f32 = 15.0;

/// Faction relation penalty for lawful factions on discovery.
const LAWFUL_FACTION_PENALTY: f32 = 12.0;

/// Tick the black market system every `BLACK_MARKET_INTERVAL` ticks.
pub fn tick_black_market(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % BLACK_MARKET_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Don't activate before tick 1000 (early game grace period)
    if state.tick < 1000 {
        return;
    }

    // --- Heat decay ---
    if state.black_market.heat > 0.0 {
        state.black_market.heat = (state.black_market.heat - HEAT_DECAY_PER_TICK).max(0.0);
    }

    // --- Discovery check ---
    if state.black_market.heat > DISCOVERY_THRESHOLD {
        let discovery_chance = state.black_market.heat / 200.0;
        let roll = lcg_f32(&mut state.rng);
        if roll < discovery_chance {
            // Discovered! Reputation and faction relation penalties.
            let rep_loss = DISCOVERY_REPUTATION_LOSS;
            state.guild.reputation = (state.guild.reputation - rep_loss).max(0.0);

            // Penalize relations with lawful/friendly factions
            for faction in &mut state.factions {
                if matches!(
                    faction.diplomatic_stance,
                    DiplomaticStance::Friendly | DiplomaticStance::Coalition | DiplomaticStance::Neutral
                ) {
                    faction.relationship_to_guild =
                        (faction.relationship_to_guild - LAWFUL_FACTION_PENALTY).max(-100.0);
                }
            }

            events.push(WorldEvent::BlackMarketDiscovered {
                reputation_lost: rep_loss,
            });

            // Reset heat after discovery
            state.black_market.heat = 0.0;
        }
    }

    // --- Refresh deals ---
    let ticks_since_refresh = state.tick.saturating_sub(state.black_market.last_refresh_tick);
    if ticks_since_refresh >= DEAL_REFRESH_INTERVAL || state.black_market.available_deals.is_empty()
    {
        refresh_deals(state);
    }
}

/// Generate a fresh set of 3-5 black market deals.
fn refresh_deals(state: &mut CampaignState) {
    state.black_market.last_refresh_tick = state.tick;
    state.black_market.available_deals.clear();

    // Determine deal count: 3-5
    let count = 3 + (lcg_next(&mut state.rng) % 3) as usize; // 3, 4, or 5

    // Scale rewards with guild tier (reputation) and threat level
    let tier_mult = 1.0 + (state.guild.reputation / 100.0) * 0.5; // 1.0-1.5
    let threat_mult = 1.0 + (state.overworld.global_threat_level / 100.0) * 0.8; // 1.0-1.8

    let deal_types = [
        DealType::StolenGoods,
        DealType::SmuggledSupplies,
        DealType::ForgedDocuments,
        DealType::PoisonContracts,
        DealType::InformationBrokering,
        DealType::RelicFencing,
    ];

    for i in 0..count {
        let dt_idx = (lcg_next(&mut state.rng) as usize) % deal_types.len();
        let deal_type = deal_types[dt_idx].clone();

        let base_id = state.tick as u32 * 10 + i as u32;

        let (description, base_gold, rep_risk, supply_cost, heat_cost) = match deal_type {
            DealType::StolenGoods => (
                "Fence stolen merchandise from a recent caravan raid".into(),
                40.0,
                5.0,
                0.0,
                15.0,
            ),
            DealType::SmuggledSupplies => (
                "Smuggle contraband supplies through guild channels".into(),
                60.0,
                8.0,
                20.0, // costs supplies, returns 3x gold
                20.0,
            ),
            DealType::ForgedDocuments => (
                "Forge official faction documents for underground buyers".into(),
                50.0,
                10.0,
                0.0,
                25.0,
            ),
            DealType::PoisonContracts => (
                "Fulfill an assassination contract with rare poisons".into(),
                80.0,
                15.0,
                5.0,
                30.0,
            ),
            DealType::InformationBrokering => (
                "Sell faction intelligence to interested parties".into(),
                35.0,
                6.0,
                0.0,
                18.0,
            ),
            DealType::RelicFencing => (
                "Fence a recovered dungeon relic on the black market".into(),
                70.0,
                12.0,
                0.0,
                22.0,
            ),
        };

        let gold_reward = base_gold * tier_mult * threat_mult;

        state.black_market.available_deals.push(BlackMarketDeal {
            id: base_id,
            description,
            deal_type,
            gold_reward,
            reputation_risk: rep_risk,
            supply_cost,
            heat_cost,
        });
    }
}

/// Execute a black market deal. Called from `apply_action` in step.rs.
pub fn execute_deal(
    state: &mut CampaignState,
    deal_id: u32,
    events: &mut Vec<WorldEvent>,
) -> Result<String, String> {
    // Validate heat threshold
    if state.black_market.heat >= 90.0 {
        return Err("Too much heat — lay low before dealing again".into());
    }

    // Find the deal
    let deal_idx = state
        .black_market
        .available_deals
        .iter()
        .position(|d| d.id == deal_id);
    let deal_idx = match deal_idx {
        Some(i) => i,
        None => return Err(format!("Deal {} not found on the black market", deal_id)),
    };

    let deal = state.black_market.available_deals[deal_idx].clone();

    // Check supply cost
    if deal.supply_cost > 0.0 && state.guild.supplies < deal.supply_cost {
        return Err(format!(
            "Not enough supplies ({:.0} needed, {:.0} available)",
            deal.supply_cost, state.guild.supplies
        ));
    }

    // Execute the deal
    if deal.supply_cost > 0.0 {
        state.guild.supplies -= deal.supply_cost;
    }
    state.guild.gold += deal.gold_reward;
    state.black_market.heat += deal.heat_cost;
    state.black_market.total_profit += deal.gold_reward;

    // Small reputation risk per deal (based on deal's rep_risk)
    let rep_roll = lcg_f32(&mut state.rng);
    if rep_roll < deal.reputation_risk / 100.0 {
        state.guild.reputation = (state.guild.reputation - 2.0).max(0.0);
    }

    // Information brokering reveals faction intel
    if deal.deal_type == DealType::InformationBrokering && !state.factions.is_empty() {
        let faction_idx = (lcg_next(&mut state.rng) as usize) % state.factions.len();
        // Increase visibility of a random region
        if !state.overworld.regions.is_empty() {
            let region_idx = (lcg_next(&mut state.rng) as usize) % state.overworld.regions.len();
            state.overworld.regions[region_idx].visibility =
                (state.overworld.regions[region_idx].visibility + 0.3).min(1.0);
            events.push(WorldEvent::ScoutReport {
                location_id: faction_idx,
                threat_level: state.overworld.regions[region_idx].unrest,
            });
        }
    }

    let profit = deal.gold_reward;
    let desc = deal.description.clone();

    // Remove the deal
    state.black_market.available_deals.remove(deal_idx);

    events.push(WorldEvent::BlackMarketDeal {
        description: desc.clone(),
        profit,
    });

    Ok(format!(
        "Black market deal completed: {} (+{:.0} gold, heat now {:.0})",
        desc, profit, state.black_market.heat
    ))
}
