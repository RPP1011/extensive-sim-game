//! Economy tick — every tick.
//!
//! Applies rewards from completed quests, passive income, trade income,
//! market price inflation/decay, investment returns, and supply chain penalties.

use crate::headless_campaign::actions::{SpendPriority, StepDeltas, WorldEvent};
use crate::headless_campaign::state::{AdventurerStatus, CampaignState, CAMPAIGN_TICK_MS};
use super::class_system::effective_noncombat_stats;

pub fn tick_economy(
    state: &mut CampaignState,
    deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let dt_sec = CAMPAIGN_TICK_MS as f32 / 1000.0;
    deltas.gold_before = state.guild.gold;
    deltas.supplies_before = state.guild.supplies;
    deltas.reputation_before = state.guild.reputation;

    // Passive income
    state.guild.gold += state.config.economy.passive_gold_per_sec * dt_sec;

    // --- Trade income from guild-controlled regions with settlements ---
    tick_trade_income(state, dt_sec);

    // --- Threat reward bonus (risk = reward) ---
    let threat_bonus = state.overworld.global_threat_level
        * state.config.economy.threat_reward_bonus
        * dt_sec;
    state.guild.gold += threat_bonus;

    // --- Market price decay (rolling history decays toward 0) ---
    tick_market_decay(state, dt_sec);

    // --- Investment drain and returns ---
    tick_investment(state, dt_sec);

    // --- Supply chain penalties for distant parties ---
    tick_supply_chain(state, dt_sec);

    // Apply completed quest rewards (process newly completed quests)
    // Quest rewards are applied by quest_lifecycle when quests complete,
    // but the actual gold/rep/supply changes happen here.
    let newly_completed: Vec<_> = state
        .completed_quests
        .iter()
        .filter(|q| q.completed_at_ms == state.elapsed_ms)
        .cloned()
        .collect();

    for quest in &newly_completed {
        let reward = &quest.reward_applied;
        if reward.gold > 0.0 {
            state.guild.gold += reward.gold;
            events.push(WorldEvent::GoldChanged {
                amount: reward.gold,
                reason: format!("Quest {} reward", quest.id),
            });
        }
        if reward.reputation > 0.0 {
            state.guild.reputation =
                (state.guild.reputation + reward.reputation).min(100.0);
        }
        if reward.supply_reward > 0.0 {
            state.guild.supplies += reward.supply_reward;
            events.push(WorldEvent::SupplyChanged {
                amount: reward.supply_reward,
                reason: format!("Quest {} reward", quest.id),
            });
        }
        // Apply faction relation change
        if let Some(fid) = reward.relation_faction_id {
            if let Some(faction) = state.factions.iter_mut().find(|f| f.id == fid) {
                let old = faction.relationship_to_guild;
                faction.relationship_to_guild =
                    (faction.relationship_to_guild + reward.relation_change).clamp(-100.0, 100.0);
                events.push(WorldEvent::FactionRelationChanged {
                    faction_id: fid,
                    old,
                    new: faction.relationship_to_guild,
                });
            }
        }
    }

    deltas.gold_after = state.guild.gold;
    deltas.supplies_after = state.guild.supplies;
    deltas.reputation_after = state.guild.reputation;
}

/// Compute the effective cost of an action after market inflation.
/// Call this before deducting gold to account for dynamic pricing.
pub fn effective_cost(base_cost: f32, multiplier: f32) -> f32 {
    base_cost * multiplier
}

// ---------------------------------------------------------------------------
// Trade income
// ---------------------------------------------------------------------------

/// Guild-controlled regions with settlements generate gold proportional to
/// control * (1 - unrest/100). War in the region disrupts trade.
fn tick_trade_income(state: &mut CampaignState, dt_sec: f32) {
    let guild_faction_id = state.diplomacy.guild_faction_id;
    let rate = state.config.economy.trade_income_per_control;

    // Check which factions are at war with the guild
    let at_war_factions: Vec<usize> = state
        .factions
        .iter()
        .filter(|f| {
            f.at_war_with.contains(&guild_faction_id)
                || matches!(
                    f.diplomatic_stance,
                    crate::headless_campaign::state::DiplomaticStance::AtWar
                )
        })
        .map(|f| f.id)
        .collect();

    let mut trade_income = 0.0f32;
    for region in &state.overworld.regions {
        if region.owner_faction_id != guild_faction_id {
            continue;
        }

        // Check if any settlement exists in this region (approximate by location)
        let has_settlement = state.overworld.locations.iter().any(|loc| {
            loc.location_type == crate::headless_campaign::state::LocationType::Settlement
                && loc.faction_owner == Some(guild_faction_id)
        });
        if !has_settlement {
            continue;
        }

        // War disrupts trade — reduce income if neighbors are at war
        let war_disruption = if region
            .neighbors
            .iter()
            .any(|n| at_war_factions.contains(n))
        {
            0.5 // 50% trade disruption from nearby war
        } else {
            1.0
        };

        let stability = 1.0 - (region.unrest / 100.0).min(1.0);
        trade_income += region.control * rate * stability * war_disruption * dt_sec;
    }

    // Commerce bonus from adventurer class stats: +0.5 gold per trade tick per commerce point
    let commerce_bonus: f32 = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead && a.faction_id.is_none())
        .map(|a| effective_noncombat_stats(a).1) // commerce component
        .sum();
    trade_income += commerce_bonus * 0.5 * dt_sec;

    state.guild.gold += trade_income;
    state.guild.total_trade_income += trade_income;
}

// ---------------------------------------------------------------------------
// Market price decay
// ---------------------------------------------------------------------------

/// Purchase history decays toward 0, letting prices normalize over time.
fn tick_market_decay(state: &mut CampaignState, dt_sec: f32) {
    let decay = 1.0 - state.config.economy.market_decay_rate * dt_sec;
    let decay = decay.max(0.0);
    let max_mult = state.config.economy.market_max_multiplier;

    state.guild.purchase_history.supply_purchases *= decay;
    state.guild.purchase_history.recruitment_purchases *= decay;
    state.guild.purchase_history.training_purchases *= decay;
    state.guild.purchase_history.mercenary_purchases *= decay;

    // Recompute multipliers: 1.0 + history * inflation_rate, capped
    let inf = state.config.economy.market_inflation_rate;
    state.guild.market_prices.supply_multiplier =
        (1.0 + state.guild.purchase_history.supply_purchases * inf).min(max_mult);
    state.guild.market_prices.recruitment_multiplier =
        (1.0 + state.guild.purchase_history.recruitment_purchases * inf).min(max_mult);
    state.guild.market_prices.training_multiplier =
        (1.0 + state.guild.purchase_history.training_purchases * inf).min(max_mult);
    state.guild.market_prices.mercenary_multiplier =
        (1.0 + state.guild.purchase_history.mercenary_purchases * inf).min(max_mult);
}

// ---------------------------------------------------------------------------
// Investment
// ---------------------------------------------------------------------------

/// Drains gold into the active spend priority category, increasing its
/// investment level with diminishing returns: gain = rate * (1 - level/max).
fn tick_investment(state: &mut CampaignState, dt_sec: f32) {
    let gold_cost = state.config.economy.investment_gold_per_sec * dt_sec;
    if state.guild.gold < gold_cost {
        return; // Can't afford investment this tick
    }

    let rate = state.config.economy.investment_return_rate;
    let max_level = state.config.economy.investment_max_level;

    // Determine which category to invest in
    let (current_level, level_mut) = match state.guild.spend_priority {
        SpendPriority::MilitaryFocus => {
            let lvl = state.guild.investment.defense_level;
            (lvl, &mut state.guild.investment.defense_level)
        }
        SpendPriority::InvestInGrowth => {
            let lvl = state.guild.investment.infrastructure_level;
            (lvl, &mut state.guild.investment.infrastructure_level)
        }
        SpendPriority::SaveForEmergencies => {
            // Save mode: don't invest, just keep gold
            return;
        }
        SpendPriority::Balanced => {
            // Balanced: split across all 4 categories equally
            let quarter_cost = gold_cost / 4.0;
            if state.guild.gold < gold_cost {
                return;
            }
            state.guild.gold = (state.guild.gold - gold_cost).max(0.0);

            for level in [
                &mut state.guild.investment.defense_level,
                &mut state.guild.investment.recruitment_level,
                &mut state.guild.investment.infrastructure_level,
                &mut state.guild.investment.intelligence_level,
            ] {
                let diminishing = (1.0 - *level / max_level).max(0.0);
                *level = (*level + rate * diminishing * quarter_cost).min(max_level);
            }
            return;
        }
    };

    let diminishing = (1.0 - current_level / max_level).max(0.0);
    state.guild.gold = (state.guild.gold - gold_cost).max(0.0);
    *level_mut = (*level_mut + rate * diminishing * gold_cost).min(max_level);
}

// ---------------------------------------------------------------------------
// Supply chain
// ---------------------------------------------------------------------------

/// Parties far from the guild base consume supplies faster.
/// Out-of-supply parties get fatigue and morale penalties.
fn tick_supply_chain(state: &mut CampaignState, dt_sec: f32) {
    let base_pos = state.guild.base.position;
    let dist_penalty = state.config.economy.supply_distance_penalty;
    let oos_fatigue = state.config.economy.out_of_supply_fatigue;
    let oos_morale = state.config.economy.out_of_supply_morale;

    // Collect party data to avoid borrow issues
    let party_info: Vec<(u32, (f32, f32), f32, Vec<u32>)> = state
        .parties
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                crate::headless_campaign::state::PartyStatus::Traveling
                    | crate::headless_campaign::state::PartyStatus::OnMission
                    | crate::headless_campaign::state::PartyStatus::Fighting
            )
        })
        .map(|p| (p.id, p.position, p.supply_level, p.member_ids.clone()))
        .collect();

    for (party_id, pos, supply_level, member_ids) in &party_info {
        let dx = pos.0 - base_pos.0;
        let dy = pos.1 - base_pos.1;
        let distance = (dx * dx + dy * dy).sqrt();

        // Extra supply drain proportional to distance
        let extra_drain = distance * dist_penalty * dt_sec;
        if let Some(party) = state.parties.iter_mut().find(|p| p.id == *party_id) {
            party.supply_level = (party.supply_level - extra_drain).max(0.0);
        }

        // Out-of-supply penalties
        if *supply_level <= 0.0 {
            for mid in member_ids {
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *mid) {
                    if adv.status == crate::headless_campaign::state::AdventurerStatus::Dead {
                        continue;
                    }
                    adv.fatigue = (adv.fatigue + oos_fatigue * dt_sec).min(100.0);
                    adv.morale = (adv.morale - oos_morale * dt_sec).max(0.0);
                }
            }
            // Party morale drain
            if let Some(party) = state.parties.iter_mut().find(|p| p.id == *party_id) {
                party.morale = (party.morale - oos_morale * dt_sec).max(0.0);
            }
        }
    }
}

/// Record a purchase in the rolling history (call when an action costs gold).
/// This drives market inflation for the relevant category.
pub fn record_purchase(state: &mut CampaignState, category: PurchaseCategory) {
    match category {
        PurchaseCategory::Supply => state.guild.purchase_history.supply_purchases += 1.0,
        PurchaseCategory::Recruitment => state.guild.purchase_history.recruitment_purchases += 1.0,
        PurchaseCategory::Training => state.guild.purchase_history.training_purchases += 1.0,
        PurchaseCategory::Mercenary => state.guild.purchase_history.mercenary_purchases += 1.0,
    }
}

/// Categories of purchases that affect market prices.
pub enum PurchaseCategory {
    Supply,
    Recruitment,
    Training,
    Mercenary,
}
