//! Faction economic competition system.
//!
//! Factions compete for trade dominance, resource control, and market share,
//! creating economic warfare. Ticks every 500 ticks.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Tick cadence for economic competition (every 500 ticks).
const TICK_CADENCE: u64 = 500;

/// Market share threshold above which a faction is considered dominant.
const DOMINANCE_THRESHOLD: f32 = 0.40;

/// Market share threshold below which a faction loses a trade war.
const DEFEAT_THRESHOLD: f32 = 0.20;

/// Maximum duration of a rivalry (trade war / embargo / price war) in ticks.
const MAX_RIVALRY_DURATION: u64 = 3000;

/// Trade war income penalty for both sides.
const TRADE_WAR_INCOME_PENALTY: f32 = 0.30;

/// Trade war supply cost increase.
const TRADE_WAR_SUPPLY_COST_INCREASE: f32 = 0.20;

/// Embargo trade income penalty (total loss).
const EMBARGO_TRADE_PENALTY: f32 = 1.0;

/// Embargo relation penalty per tick cycle.
const EMBARGO_RELATION_PENALTY: f32 = 20.0;

/// Price war profit penalty.
const PRICE_WAR_PROFIT_PENALTY: f32 = 0.50;

/// Price war market share gain per tick cycle.
const PRICE_WAR_SHARE_GAIN: f32 = 0.02;

pub fn tick_economic_competition(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % TICK_CADENCE != 0 {
        return;
    }

    // Need at least 2 factions for competition.
    if state.factions.len() < 2 {
        return;
    }

    // 1. Compute market share per faction.
    compute_market_dominance(state);

    // 2. Check for new rivalries from dominant factions.
    check_new_rivalries(state, events);

    // 3. Apply active rivalry effects.
    apply_rivalry_effects(state, events);

    // 4. Guild involvement in economic conflicts.
    tick_guild_involvement(state, events);

    // 5. Resolve finished rivalries.
    resolve_rivalries(state, events);

    // 6. Update system trackers.
    update_trackers(state);
}

/// Compute market share for each faction based on trade routes, resources, and trade goods.
fn compute_market_dominance(state: &mut CampaignState) {
    let faction_count = state.factions.len();
    if faction_count == 0 {
        return;
    }

    // Compute raw economic score per faction.
    let mut scores: Vec<f32> = vec![0.0; faction_count];

    for (fi, faction) in state.factions.iter().enumerate() {
        // Trade routes controlled by this faction (regions with settlements).
        let controlled_regions = state
            .overworld
            .regions
            .iter()
            .filter(|r| r.owner_faction_id == faction.id)
            .count() as f32;
        scores[fi] += controlled_regions * 10.0;

        // Resource nodes in controlled regions.
        let resource_score: f32 = state
            .resource_nodes
            .iter()
            .filter(|rn| {
                state
                    .overworld
                    .regions
                    .get(rn.region_id)
                    .map(|r| r.owner_faction_id == faction.id)
                    .unwrap_or(false)
            })
            .map(|rn| rn.amount.min(rn.max_amount))
            .sum();
        scores[fi] += resource_score * 0.5;

        // Military strength as economic backing.
        scores[fi] += faction.military_strength * 0.3;

        // Active trade routes touching this faction's territory.
        let active_routes = state
            .trade_routes
            .iter()
            .filter(|r| r.active)
            .count() as f32;
        // Distribute trade route score proportionally to territory control.
        let total_regions = state.overworld.regions.len().max(1) as f32;
        scores[fi] += active_routes * (controlled_regions / total_regions) * 5.0;
    }

    // Normalize to market share (0.0 - 1.0).
    let total_score: f32 = scores.iter().sum::<f32>().max(1.0);

    // Build or update market dominance entries.
    state.market_dominance.clear();
    for (fi, faction) in state.factions.iter().enumerate() {
        let share = scores[fi] / total_score;

        // Count resource monopolies (faction controls >60% of a resource type).
        let monopolies = compute_monopolies(state, faction.id);

        // Count trade routes in faction territory.
        let controlled_regions: Vec<usize> = state
            .overworld
            .regions
            .iter()
            .filter(|r| r.owner_faction_id == faction.id)
            .map(|r| r.id)
            .collect();

        let routes_controlled = state
            .trade_routes
            .iter()
            .filter(|r| r.active)
            .filter(|_r| !controlled_regions.is_empty())
            .count() as u32;

        state.market_dominance.push(MarketDominance {
            faction_id: faction.id,
            market_share: share,
            trade_routes_controlled: routes_controlled,
            resource_monopolies: monopolies,
        });
    }
}

/// Determine which resource types a faction has a monopoly on (>60% of total).
fn compute_monopolies(state: &CampaignState, faction_id: usize) -> Vec<String> {
    use std::collections::HashMap;

    // Total and faction-owned amounts per resource type.
    let mut totals: HashMap<ResourceType, f32> = HashMap::new();
    let mut faction_totals: HashMap<ResourceType, f32> = HashMap::new();

    for rn in &state.resource_nodes {
        let owned_by = state
            .overworld
            .regions
            .get(rn.region_id)
            .map(|r| r.owner_faction_id)
            .unwrap_or(usize::MAX);

        *totals.entry(rn.resource_type).or_insert(0.0) += rn.amount;
        if owned_by == faction_id {
            *faction_totals.entry(rn.resource_type).or_insert(0.0) += rn.amount;
        }
    }

    let mut monopolies = Vec::new();
    for (rtype, total) in &totals {
        if *total <= 0.0 {
            continue;
        }
        let faction_amount = faction_totals.get(rtype).copied().unwrap_or(0.0);
        if faction_amount / total > 0.6 {
            monopolies.push(format!("{:?}", rtype));
        }
    }
    monopolies
}

/// Check if any dominant faction should trigger a new trade war, embargo, or price war.
fn check_new_rivalries(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let dominance_snapshot: Vec<(usize, f32)> = state
        .market_dominance
        .iter()
        .map(|md| (md.faction_id, md.market_share))
        .collect();

    for &(faction_id, share) in &dominance_snapshot {
        if share <= DOMINANCE_THRESHOLD {
            continue;
        }

        // Find the second-largest faction as the rivalry target.
        let target = dominance_snapshot
            .iter()
            .filter(|(fid, _)| *fid != faction_id)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let target_id = match target {
            Some((tid, _)) => *tid,
            None => continue,
        };

        // Skip if rivalry already exists between these two.
        let already_exists = state.economic_rivalries.iter().any(|r| {
            (r.faction_a == faction_id && r.faction_b == target_id)
                || (r.faction_a == target_id && r.faction_b == faction_id)
        });
        if already_exists {
            continue;
        }

        // Roll for rivalry type.
        let mut rng = state.rng;
        let roll = lcg_next(&mut rng) % 100;
        state.rng = rng;

        let (trade_war, embargo, price_war) = if roll < 40 {
            // Trade war (40% chance).
            (true, false, false)
        } else if roll < 60 {
            // Embargo (20% chance).
            (false, true, false)
        } else if roll < 85 {
            // Price war (25% chance).
            (false, false, true)
        } else {
            // No rivalry this cycle (15% chance).
            continue;
        };

        state.economic_rivalries.push(EconomicRivalry {
            faction_a: faction_id,
            faction_b: target_id,
            trade_war,
            embargo,
            price_war,
            started_tick: state.tick,
        });

        let faction_a_name = state
            .factions
            .iter()
            .find(|f| f.id == faction_id)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| format!("Faction {}", faction_id));
        let faction_b_name = state
            .factions
            .iter()
            .find(|f| f.id == target_id)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| format!("Faction {}", target_id));

        if trade_war {
            events.push(WorldEvent::TradeWarDeclared {
                aggressor_faction_id: faction_id,
                target_faction_id: target_id,
                aggressor_share: share,
                description: format!(
                    "{} has declared a trade war against {}, disrupting commerce across the region.",
                    faction_a_name, faction_b_name,
                ),
            });
        } else if embargo {
            events.push(WorldEvent::EmbargoImposed {
                imposer_faction_id: faction_id,
                target_faction_id: target_id,
                description: format!(
                    "{} has imposed a trade embargo on {}, cutting off all commerce.",
                    faction_a_name, faction_b_name,
                ),
            });
        } else if price_war {
            events.push(WorldEvent::PriceWarStarted {
                faction_a: faction_id,
                faction_b: target_id,
                description: format!(
                    "{} and {} have entered a price war, undercutting each other's trade goods.",
                    faction_a_name, faction_b_name,
                ),
            });
        }
    }
}

/// Apply the effects of active economic rivalries.
fn apply_rivalry_effects(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Snapshot rivalry data to avoid borrow issues.
    let rivalries: Vec<EconomicRivalry> = state.economic_rivalries.clone();
    let guild_faction_id = state.diplomacy.guild_faction_id;

    for rivalry in &rivalries {
        if rivalry.trade_war {
            // Trade war: -30% trade income for both sides.
            // Affects guild if guild faction is involved.
            if rivalry.faction_a == guild_faction_id
                || rivalry.faction_b == guild_faction_id
            {
                let penalty = state.guild.total_trade_income * TRADE_WAR_INCOME_PENALTY * 0.01;
                state.guild.gold = (state.guild.gold - penalty).max(0.0);

                // Supply costs +20%.
                state.guild.market_prices.supply_multiplier *= 1.0 + TRADE_WAR_SUPPLY_COST_INCREASE;
            }

            // Weaken both factions' military (economic strain).
            for fid in [rivalry.faction_a, rivalry.faction_b] {
                if let Some(faction) = state.factions.iter_mut().find(|f| f.id == fid) {
                    faction.military_strength *= 0.99;
                }
            }
        }

        if rivalry.embargo {
            // Embargo: target faction loses all trade, relation drops.
            if rivalry.faction_b == guild_faction_id {
                // Guild is the embargo target — lose trade income.
                let penalty = state.guild.total_trade_income * EMBARGO_TRADE_PENALTY * 0.01;
                state.guild.gold = (state.guild.gold - penalty).max(0.0);
            }

            // Relation between the two factions drops.
            if let Some(target_faction) = state
                .factions
                .iter_mut()
                .find(|f| f.id == rivalry.faction_b)
            {
                if rivalry.faction_a == guild_faction_id {
                    target_faction.relationship_to_guild =
                        (target_faction.relationship_to_guild - EMBARGO_RELATION_PENALTY * 0.1)
                            .max(-100.0);
                }
            }
            if let Some(imposer_faction) = state
                .factions
                .iter_mut()
                .find(|f| f.id == rivalry.faction_a)
            {
                if rivalry.faction_b == guild_faction_id {
                    imposer_faction.relationship_to_guild =
                        (imposer_faction.relationship_to_guild - EMBARGO_RELATION_PENALTY * 0.1)
                            .max(-100.0);
                }
            }
        }

        if rivalry.price_war {
            // Price war: both factions lose profit but gain market share.
            if rivalry.faction_a == guild_faction_id
                || rivalry.faction_b == guild_faction_id
            {
                // Guild profits reduced.
                let profit_loss = state.guild.gold * PRICE_WAR_PROFIT_PENALTY * 0.01;
                state.guild.gold = (state.guild.gold - profit_loss).max(0.0);
            }

            // Shift market shares slightly toward the price-war factions.
            for md in &mut state.market_dominance {
                if md.faction_id == rivalry.faction_a || md.faction_id == rivalry.faction_b {
                    md.market_share = (md.market_share + PRICE_WAR_SHARE_GAIN).min(1.0);
                } else {
                    md.market_share = (md.market_share - PRICE_WAR_SHARE_GAIN * 0.5).max(0.0);
                }
            }

            // Renormalize shares.
            let total: f32 = state.market_dominance.iter().map(|md| md.market_share).sum();
            if total > 0.0 {
                for md in &mut state.market_dominance {
                    md.market_share /= total;
                }
            }
        }
    }

    // Emit market dominance shift events when a faction crosses thresholds.
    for md in &state.market_dominance {
        if md.market_share > DOMINANCE_THRESHOLD {
            let name = state
                .factions
                .iter()
                .find(|f| f.id == md.faction_id)
                .map(|f| f.name.clone())
                .unwrap_or_else(|| format!("Faction {}", md.faction_id));

            // Only emit periodically to avoid spam.
            if state.tick % 2000 == 0 {
                events.push(WorldEvent::MarketDominanceShift {
                    faction_id: md.faction_id,
                    market_share: md.market_share,
                    description: format!(
                        "{} dominates {:.0}% of the regional market.",
                        name,
                        md.market_share * 100.0,
                    ),
                });
            }
        }
    }
}

/// Guild involvement: the guild can exploit economic conflicts.
fn tick_guild_involvement(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let guild_faction_id = state.diplomacy.guild_faction_id;

    // Snapshot to avoid borrow issues.
    let rivalries: Vec<EconomicRivalry> = state.economic_rivalries.clone();

    for rivalry in &rivalries {
        // Skip if guild is directly involved (effects already applied above).
        if rivalry.faction_a == guild_faction_id || rivalry.faction_b == guild_faction_id {
            continue;
        }

        let mut rng = state.rng;
        let roll = lcg_f32(&mut rng);
        state.rng = rng;

        // 15% chance per cycle the guild exploits a conflict.
        if roll >= 0.15 {
            continue;
        }

        let action_roll = lcg_next(&mut state.rng) % 3;

        match action_roll {
            0 if rivalry.trade_war => {
                // Guild joins a side of the trade war for profit.
                let ally_id = rivalry.faction_a;
                let profit = 5.0 + lcg_f32(&mut state.rng) * 15.0;
                state.guild.gold += profit;

                // Improve relation with ally, worsen with enemy.
                if let Some(f) = state.factions.iter_mut().find(|f| f.id == ally_id) {
                    f.relationship_to_guild =
                        (f.relationship_to_guild + 3.0).min(100.0);
                }
                if let Some(f) = state.factions.iter_mut().find(|f| f.id == rivalry.faction_b) {
                    f.relationship_to_guild =
                        (f.relationship_to_guild - 2.0).max(-100.0);
                }

                let ally_name = state
                    .factions
                    .iter()
                    .find(|f| f.id == ally_id)
                    .map(|f| f.name.clone())
                    .unwrap_or_default();

                events.push(WorldEvent::CampaignMilestone {
                    description: format!(
                        "Your guild joined {}'s side in a trade war, earning {:.0} gold.",
                        ally_name, profit,
                    ),
                });
            }
            1 if rivalry.embargo => {
                // Guild smuggles goods via black market, breaking the embargo.
                let smuggle_profit = 8.0 + lcg_f32(&mut state.rng) * 20.0;
                state.guild.gold += smuggle_profit;
                state.black_market.heat += 5.0;

                events.push(WorldEvent::CampaignMilestone {
                    description: format!(
                        "Your guild smuggled goods past an embargo, earning {:.0} gold (black market heat increased).",
                        smuggle_profit,
                    ),
                });
            }
            _ if rivalry.price_war => {
                // Guild buys cheap goods from warring factions.
                let savings = 3.0 + lcg_f32(&mut state.rng) * 10.0;
                state.guild.supplies += savings;

                events.push(WorldEvent::CampaignMilestone {
                    description: format!(
                        "Your guild exploited a price war to stockpile {:.0} supplies cheaply.",
                        savings,
                    ),
                });
            }
            _ => {}
        }
    }
}

/// Resolve rivalries that have exceeded their duration or where one side lost.
fn resolve_rivalries(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let dominance_snapshot: Vec<(usize, f32)> = state
        .market_dominance
        .iter()
        .map(|md| (md.faction_id, md.market_share))
        .collect();

    let current_tick = state.tick;

    state.economic_rivalries.retain(|rivalry| {
        let elapsed = current_tick.saturating_sub(rivalry.started_tick);

        // Check if either faction dropped below defeat threshold.
        let a_share = dominance_snapshot
            .iter()
            .find(|(fid, _)| *fid == rivalry.faction_a)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let b_share = dominance_snapshot
            .iter()
            .find(|(fid, _)| *fid == rivalry.faction_b)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);

        let should_resolve = elapsed >= MAX_RIVALRY_DURATION
            || a_share < DEFEAT_THRESHOLD
            || b_share < DEFEAT_THRESHOLD;

        if should_resolve {
            let rivalry_type = if rivalry.trade_war {
                "trade war"
            } else if rivalry.embargo {
                "embargo"
            } else {
                "price war"
            };

            let loser = if a_share < b_share {
                rivalry.faction_a
            } else {
                rivalry.faction_b
            };
            let winner = if loser == rivalry.faction_a {
                rivalry.faction_b
            } else {
                rivalry.faction_a
            };

            let winner_name = state
                .factions
                .iter()
                .find(|f| f.id == winner)
                .map(|f| f.name.clone())
                .unwrap_or_else(|| format!("Faction {}", winner));
            let loser_name = state
                .factions
                .iter()
                .find(|f| f.id == loser)
                .map(|f| f.name.clone())
                .unwrap_or_else(|| format!("Faction {}", loser));

            let reason = if elapsed >= MAX_RIVALRY_DURATION {
                format!("The {} between {} and {} has ended after prolonged economic attrition.",
                    rivalry_type, winner_name, loser_name)
            } else {
                format!("{} lost the {} against {} as their market share collapsed.",
                    loser_name, rivalry_type, winner_name)
            };

            events.push(WorldEvent::TradeWarResolved {
                winner_faction_id: winner,
                loser_faction_id: loser,
                rivalry_type: rivalry_type.to_string(),
                description: reason,
            });

            false // Remove from list.
        } else {
            true // Keep active.
        }
    });
}

/// Update system trackers with economic competition state.
fn update_trackers(state: &mut CampaignState) {
    state.system_trackers.active_economic_rivalries = state.economic_rivalries.len() as u32;
    state.system_trackers.max_market_share = state
        .market_dominance
        .iter()
        .map(|md| md.market_share)
        .fold(0.0f32, f32::max);
    state.system_trackers.total_monopolies = state
        .market_dominance
        .iter()
        .map(|md| md.resource_monopolies.len() as u32)
        .sum();
}
