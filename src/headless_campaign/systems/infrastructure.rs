//! Infrastructure system — every 200 ticks (~20s).
//!
//! Manages roads, bridges, waypoints, trade posts, and watchtowers between
//! regions. Infrastructure provides passive bonuses:
//! - **Road**: +30% travel speed per level/100
//! - **Bridge**: opens routes between non-adjacent regions
//! - **Waypoint**: -20% supply drain for parties on this route
//! - **TradePost**: +20% trade income for caravans on this route
//! - **Watchtower**: +0.2 visibility for both connected regions
//!
//! Infrastructure degrades without maintenance gold and can be damaged by
//! enemy factions or natural disasters.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{
    lcg_f32, CampaignState, InfraType, Infrastructure,
};

/// Cadence: runs every 200 ticks.
const INFRA_TICK_INTERVAL: u64 = 200;

/// Maintenance cost per level per tick (gold).
const MAINTENANCE_PER_LEVEL: f32 = 0.02;

/// Degradation per tick when maintenance is unpaid.
const UNPAID_DEGRADATION: f32 = 2.0;

/// War damage per hostile faction attack event.
const WAR_DAMAGE: f32 = 10.0;

/// Natural disaster minimum damage.
const DISASTER_MIN_DAMAGE: f32 = 20.0;

/// Natural disaster maximum damage.
const DISASTER_MAX_DAMAGE: f32 = 50.0;

/// Probability of a natural disaster damaging a piece of infrastructure per tick.
const DISASTER_CHANCE: f32 = 0.02;

/// Growth per tick while building (infrastructure starts at level 10, grows to 100).
const GROWTH_PER_TICK: f32 = 5.0;

/// Base cost to build infrastructure.
pub const INFRA_BUILD_COST_BASE: f32 = 50.0;

/// Extra cost for Bridge type (more expensive).
pub const BRIDGE_EXTRA_COST: f32 = 50.0;

/// Returns the build cost for a given infrastructure type.
pub fn build_cost(infra_type: InfraType) -> f32 {
    match infra_type {
        InfraType::Bridge => INFRA_BUILD_COST_BASE + BRIDGE_EXTRA_COST,
        _ => INFRA_BUILD_COST_BASE,
    }
}

/// Parse an infra type string to InfraType enum.
pub fn parse_infra_type(s: &str) -> Option<InfraType> {
    match s {
        "Road" => Some(InfraType::Road),
        "Bridge" => Some(InfraType::Bridge),
        "Waypoint" => Some(InfraType::Waypoint),
        "TradePost" => Some(InfraType::TradePost),
        "Watchtower" => Some(InfraType::Watchtower),
        _ => None,
    }
}

/// Returns the travel speed bonus for a route between two regions.
/// Accounts for all Road infrastructure on that route.
pub fn travel_speed_bonus(infra: &[Infrastructure], region_a: usize, region_b: usize) -> f32 {
    infra
        .iter()
        .filter(|i| {
            i.infra_type == InfraType::Road
                && ((i.region_a == region_a && i.region_b == region_b)
                    || (i.region_a == region_b && i.region_b == region_a))
        })
        .map(|i| 0.30 * (i.level / 100.0))
        .fold(0.0_f32, f32::max)
}

/// Returns the supply drain multiplier for parties on a route.
/// Waypoints reduce supply drain by up to 20%.
pub fn supply_drain_multiplier(infra: &[Infrastructure], region_a: usize, region_b: usize) -> f32 {
    let reduction = infra
        .iter()
        .filter(|i| {
            i.infra_type == InfraType::Waypoint
                && ((i.region_a == region_a && i.region_b == region_b)
                    || (i.region_a == region_b && i.region_b == region_a))
        })
        .map(|i| 0.20 * (i.level / 100.0))
        .fold(0.0_f32, f32::max);
    1.0 - reduction
}

/// Returns the trade income bonus for caravans on a route.
pub fn trade_income_bonus(infra: &[Infrastructure], region_a: usize, region_b: usize) -> f32 {
    infra
        .iter()
        .filter(|i| {
            i.infra_type == InfraType::TradePost
                && ((i.region_a == region_a && i.region_b == region_b)
                    || (i.region_a == region_b && i.region_b == region_a))
        })
        .map(|i| 0.20 * (i.level / 100.0))
        .fold(0.0_f32, f32::max)
}

/// Returns true if a bridge connects two regions (enabling non-neighbor travel).
pub fn has_bridge(infra: &[Infrastructure], region_a: usize, region_b: usize) -> bool {
    infra.iter().any(|i| {
        i.infra_type == InfraType::Bridge
            && i.level > 0.0
            && ((i.region_a == region_a && i.region_b == region_b)
                || (i.region_a == region_b && i.region_b == region_a))
    })
}

pub fn tick_infrastructure(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % INFRA_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Maintenance and growth ---
    let guild_gold = state.guild.gold;
    let mut total_maintenance = 0.0_f32;

    for infra in &state.infrastructure {
        total_maintenance += infra.maintenance_cost;
    }

    let can_pay = guild_gold >= total_maintenance;

    if can_pay {
        state.guild.gold -= total_maintenance;
    }

    for infra in &mut state.infrastructure {
        if can_pay {
            // Infrastructure grows toward 100 if maintained
            if infra.level < 100.0 {
                let old_level = infra.level;
                infra.level = (infra.level + GROWTH_PER_TICK).min(100.0);
                if old_level < 100.0 && infra.level >= 100.0 {
                    // Mark for completion event (handled below)
                }
            }
        } else {
            // Degrades if maintenance unpaid
            infra.level = (infra.level - UNPAID_DEGRADATION).max(0.0);
            if infra.level > 0.0 {
                events.push(WorldEvent::InfrastructureDamaged {
                    infra_id: infra.id,
                    infra_type: format!("{:?}", infra.infra_type),
                    amount: UNPAID_DEGRADATION,
                    cause: "Unpaid maintenance".into(),
                });
            }
        }
    }

    // --- Watchtower visibility bonus ---
    for infra in &state.infrastructure {
        if infra.infra_type == InfraType::Watchtower && infra.level > 0.0 {
            let bonus = 0.2 * (infra.level / 100.0);
            for &rid in &[infra.region_a, infra.region_b] {
                if let Some(region) = state.overworld.regions.get_mut(rid) {
                    region.visibility = (region.visibility + bonus).min(1.0);
                }
            }
        }
    }

    // --- War damage: enemy factions can damage infrastructure in contested regions ---
    {
        let mut damage_events = Vec::new();
        for infra in &mut state.infrastructure {
            // Check if either region is controlled by a hostile faction
            let region_a_hostile = state
                .overworld
                .regions
                .get(infra.region_a)
                .map(|r| {
                    state
                        .factions
                        .iter()
                        .any(|f| f.id == r.owner_faction_id && f.relationship_to_guild < -30.0)
                })
                .unwrap_or(false);
            let region_b_hostile = state
                .overworld
                .regions
                .get(infra.region_b)
                .map(|r| {
                    state
                        .factions
                        .iter()
                        .any(|f| f.id == r.owner_faction_id && f.relationship_to_guild < -30.0)
                })
                .unwrap_or(false);

            if region_a_hostile || region_b_hostile {
                let roll = lcg_f32(&mut state.rng);
                if roll < 0.1 {
                    infra.level = (infra.level - WAR_DAMAGE).max(0.0);
                    damage_events.push(WorldEvent::InfrastructureDamaged {
                        infra_id: infra.id,
                        infra_type: format!("{:?}", infra.infra_type),
                        amount: WAR_DAMAGE,
                        cause: "Enemy faction attack".into(),
                    });
                }
            }
        }
        events.extend(damage_events);
    }

    // --- Natural disasters ---
    {
        let mut disaster_events = Vec::new();
        for infra in &mut state.infrastructure {
            let roll = lcg_f32(&mut state.rng);
            if roll < DISASTER_CHANCE {
                let severity_roll = lcg_f32(&mut state.rng);
                let damage =
                    DISASTER_MIN_DAMAGE + severity_roll * (DISASTER_MAX_DAMAGE - DISASTER_MIN_DAMAGE);
                infra.level = (infra.level - damage).max(0.0);
                disaster_events.push(WorldEvent::InfrastructureDamaged {
                    infra_id: infra.id,
                    infra_type: format!("{:?}", infra.infra_type),
                    amount: damage,
                    cause: "Natural disaster".into(),
                });
            }
        }
        events.extend(disaster_events);
    }

    // --- Emit completion events for newly completed infrastructure ---
    for infra in &state.infrastructure {
        // "Just completed" = level is 100 and was below 100 last tick
        if infra.level >= 100.0 && (infra.level - GROWTH_PER_TICK) < 100.0 {
            events.push(WorldEvent::InfrastructureCompleted {
                infra_id: infra.id,
                infra_type: format!("{:?}", infra.infra_type),
                region_a: infra.region_a,
                region_b: infra.region_b,
            });
        }
    }

    // --- Remove infrastructure that has decayed to zero ---
    state.infrastructure.retain(|i| i.level > 0.0);
}
