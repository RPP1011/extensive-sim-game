//! Smuggling routes system — fires every 300 ticks (~30s game time).
//!
//! Secret trade routes through dungeons and hostile territory for high profit
//! but high risk. Routes can be busted, causing gold penalties, reputation loss,
//! and faction relation damage. Rogue adventurers and faction blessings reduce
//! bust chance.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often smuggling routes tick (in ticks).
const SMUGGLING_INTERVAL: u64 = 10;

/// Gold penalty when a route is busted (multiplier on profit_per_trip).
const BUST_GOLD_PENALTY_MULT: f32 = 3.0;

/// Reputation loss when busted.
const BUST_REPUTATION_LOSS: f32 = 5.0;

/// Faction relation loss when busted.
const BUST_RELATION_LOSS: f32 = 10.0;

/// Ticks a route is suspended after being busted.
const SUSPENSION_TICKS: u64 = 33;

/// Maximum number of active smuggling routes.
const MAX_ACTIVE_ROUTES: usize = 3;

/// Cost to establish a smuggling route.
pub const SMUGGLING_ROUTE_COST: f32 = 40.0;

/// Black market heat threshold for route discovery.
const HEAT_DISCOVERY_THRESHOLD: f32 = 30.0;

/// Espionage intel threshold for route discovery.
const INTEL_DISCOVERY_THRESHOLD: f32 = 50.0;

/// Tick the smuggling system every `SMUGGLING_INTERVAL` ticks.
pub fn tick_smuggling(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % SMUGGLING_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Don't activate before tick 1500 (early game grace period)
    if state.tick < 1500 {
        return;
    }

    // --- Check for rogue adventurers (reduce bust chance by 30%) ---
    let has_rogue = state
        .adventurers
        .iter()
        .any(|a| a.status != AdventurerStatus::Dead && a.archetype == "rogue");

    // --- Check for Shadow Covenant faction blessing (reduce bust chance by 20%) ---
    let has_shadow_blessing = state
        .factions
        .iter()
        .any(|f| f.name.contains("Shadow") && f.relationship_to_guild > 50.0);

    // --- Process active routes ---
    let route_count = state.smuggling_routes.len();
    for i in 0..route_count {
        if !state.smuggling_routes[i].active {
            // Check if suspended route can be reactivated
            if state.smuggling_routes[i].suspended_until > 0
                && state.tick >= state.smuggling_routes[i].suspended_until
            {
                state.smuggling_routes[i].active = true;
                state.smuggling_routes[i].suspended_until = 0;
            }
            continue;
        }

        let profit = state.smuggling_routes[i].profit_per_trip;
        let risk = state.smuggling_routes[i].risk;

        // --- Generate gold ---
        state.guild.gold += profit;
        state.smuggling_routes[i].trips_completed += 1;

        events.push(WorldEvent::SmugglingProfit { gold: profit });

        // --- Risk check ---
        let mut effective_risk = risk;
        if has_rogue {
            effective_risk *= 0.7; // 30% reduction
        }
        if has_shadow_blessing {
            effective_risk *= 0.8; // 20% reduction
        }

        let roll = lcg_f32(&mut state.rng);
        if roll < effective_risk {
            // Busted!
            let penalty = profit * BUST_GOLD_PENALTY_MULT;
            state.guild.gold = (state.guild.gold - penalty).max(0.0);
            state.guild.reputation =
                (state.guild.reputation - BUST_REPUTATION_LOSS).max(0.0);

            // Damage relations with factions in affected regions
            let start_region = state.smuggling_routes[i].start_region;
            let end_region = state.smuggling_routes[i].end_region;
            let num_factions = state.factions.len().max(1);
            for faction in &mut state.factions {
                // Penalize factions that control either endpoint region
                if faction.territory_size > 0
                    && (faction.id == start_region % num_factions
                        || faction.id == end_region % num_factions)
                {
                    faction.relationship_to_guild =
                        (faction.relationship_to_guild - BUST_RELATION_LOSS).max(-100.0);
                }
            }

            state.smuggling_routes[i].busted_count += 1;
            state.smuggling_routes[i].active = false;
            state.smuggling_routes[i].suspended_until = state.tick + SUSPENSION_TICKS;

            let route_id = state.smuggling_routes[i].id;
            events.push(WorldEvent::SmugglingBusted {
                route_id,
                penalty,
            });
            events.push(WorldEvent::SmugglingRouteSuspended { route_id });
        }
    }

    // --- Route discovery: check if guild can discover new routes ---
    let active_count = state
        .smuggling_routes
        .iter()
        .filter(|r| r.active)
        .count();
    if active_count < MAX_ACTIVE_ROUTES {
        let heat = state.system_trackers.black_market_heat;
        let intel = state.system_trackers.total_intel_gathered;

        if heat > HEAT_DISCOVERY_THRESHOLD || intel > INTEL_DISCOVERY_THRESHOLD {
            // Chance to discover a new route
            let discovery_chance = if heat > HEAT_DISCOVERY_THRESHOLD && intel > INTEL_DISCOVERY_THRESHOLD {
                0.15
            } else {
                0.08
            };

            let roll = lcg_f32(&mut state.rng);
            if roll < discovery_chance && state.overworld.regions.len() >= 2 {
                let num_regions = state.overworld.regions.len();
                let start = (lcg_next(&mut state.rng) as usize) % num_regions;
                let mut end = (lcg_next(&mut state.rng) as usize) % num_regions;
                if end == start {
                    end = (start + 1) % num_regions;
                }

                // Determine route type
                let route_type_roll = lcg_next(&mut state.rng) % 4;
                let route_type = match route_type_roll {
                    0 => SmugglingRouteType::DungeonPassage,
                    1 => SmugglingRouteType::WildernessTrail,
                    2 => SmugglingRouteType::FactionBorderCrossing,
                    _ => SmugglingRouteType::SeaRoute,
                };

                // Check if a route between these regions already exists
                let already_exists = state.smuggling_routes.iter().any(|r| {
                    (r.start_region == start && r.end_region == end)
                        || (r.start_region == end && r.end_region == start)
                });

                if !already_exists {
                    // Calculate profit based on distance and danger
                    let distance = ((start as f32 - end as f32).abs() + 1.0).min(5.0);
                    let danger_mult = match route_type {
                        SmugglingRouteType::FactionBorderCrossing => 3.0,
                        SmugglingRouteType::DungeonPassage => 2.0,
                        SmugglingRouteType::SeaRoute => 2.5,
                        SmugglingRouteType::WildernessTrail => 1.5,
                    };
                    let base_profit = 10.0 * distance * danger_mult;
                    let base_risk = match route_type {
                        SmugglingRouteType::FactionBorderCrossing => 0.25,
                        SmugglingRouteType::DungeonPassage => 0.15,
                        SmugglingRouteType::SeaRoute => 0.20,
                        SmugglingRouteType::WildernessTrail => 0.10,
                    };

                    let route_id = state.next_smuggling_route_id;
                    state.next_smuggling_route_id += 1;

                    state.smuggling_routes.push(SmugglingRoute {
                        id: route_id,
                        start_region: start,
                        end_region: end,
                        route_type,
                        profit_per_trip: base_profit,
                        risk: base_risk,
                        active: false, // Discovered but not yet established
                        trips_completed: 0,
                        busted_count: 0,
                        suspended_until: 0,
                    });

                    events.push(WorldEvent::SmugglingRouteDiscovered {
                        route_id,
                        start_region: start,
                        end_region: end,
                    });
                }
            }
        }
    }

    // --- Update system trackers ---
    state.system_trackers.smuggling_active_routes = state
        .smuggling_routes
        .iter()
        .filter(|r| r.active)
        .count() as u32;
    state.system_trackers.smuggling_total_profit = state
        .smuggling_routes
        .iter()
        .map(|r| r.profit_per_trip * r.trips_completed as f32)
        .sum();
    state.system_trackers.smuggling_total_busts = state
        .smuggling_routes
        .iter()
        .map(|r| r.busted_count)
        .sum();
}

/// Establish a smuggling route. Called from `apply_action` in step.rs.
pub fn establish_smuggling_route(
    state: &mut CampaignState,
    start_region: usize,
    end_region: usize,
    events: &mut Vec<WorldEvent>,
) -> Result<String, String> {
    // Validate cost
    if state.guild.gold < SMUGGLING_ROUTE_COST {
        return Err(format!(
            "Not enough gold ({:.0} needed, {:.0} available)",
            SMUGGLING_ROUTE_COST, state.guild.gold
        ));
    }

    // Validate regions exist
    if start_region >= state.overworld.regions.len() {
        return Err(format!("Start region {} does not exist", start_region));
    }
    if end_region >= state.overworld.regions.len() {
        return Err(format!("End region {} does not exist", end_region));
    }
    if start_region == end_region {
        return Err("Start and end regions must be different".into());
    }

    // Check for rogue adventurer
    let has_rogue = state
        .adventurers
        .iter()
        .any(|a| a.status != AdventurerStatus::Dead && a.archetype == "rogue");
    if !has_rogue {
        return Err("Need a rogue adventurer to establish smuggling routes".into());
    }

    // Check max active routes
    let active_count = state.smuggling_routes.iter().filter(|r| r.active).count();
    if active_count >= MAX_ACTIVE_ROUTES {
        return Err(format!(
            "Maximum active smuggling routes reached ({})",
            MAX_ACTIVE_ROUTES
        ));
    }

    // Check if this route already exists and is active
    let existing = state.smuggling_routes.iter_mut().find(|r| {
        (r.start_region == start_region && r.end_region == end_region)
            || (r.start_region == end_region && r.end_region == start_region)
    });

    if let Some(route) = existing {
        if route.active {
            return Err("A smuggling route between these regions is already active".into());
        }
        // Reactivate a discovered but inactive route
        route.active = true;
        route.suspended_until = 0;
        state.guild.gold -= SMUGGLING_ROUTE_COST;

        let route_id = route.id;
        events.push(WorldEvent::SmugglingRouteEstablished {
            route_id,
            start_region,
            end_region,
        });

        return Ok(format!(
            "Reactivated smuggling route {} (region {} -> {})",
            route_id, start_region, end_region
        ));
    }

    // Create new route
    state.guild.gold -= SMUGGLING_ROUTE_COST;

    // Determine route type based on world state
    let route_type_roll = lcg_next(&mut state.rng) % 4;
    let route_type = match route_type_roll {
        0 => SmugglingRouteType::DungeonPassage,
        1 => SmugglingRouteType::WildernessTrail,
        2 => SmugglingRouteType::FactionBorderCrossing,
        _ => SmugglingRouteType::SeaRoute,
    };

    let distance = ((start_region as f32 - end_region as f32).abs() + 1.0).min(5.0);
    let danger_mult = match route_type {
        SmugglingRouteType::FactionBorderCrossing => 3.0,
        SmugglingRouteType::DungeonPassage => 2.0,
        SmugglingRouteType::SeaRoute => 2.5,
        SmugglingRouteType::WildernessTrail => 1.5,
    };
    let base_profit = 10.0 * distance * danger_mult;
    let base_risk = match route_type {
        SmugglingRouteType::FactionBorderCrossing => 0.25,
        SmugglingRouteType::DungeonPassage => 0.15,
        SmugglingRouteType::SeaRoute => 0.20,
        SmugglingRouteType::WildernessTrail => 0.10,
    };

    let route_id = state.next_smuggling_route_id;
    state.next_smuggling_route_id += 1;

    state.smuggling_routes.push(SmugglingRoute {
        id: route_id,
        start_region,
        end_region,
        route_type,
        profit_per_trip: base_profit,
        risk: base_risk,
        active: true,
        trips_completed: 0,
        busted_count: 0,
        suspended_until: 0,
    });

    events.push(WorldEvent::SmugglingRouteEstablished {
        route_id,
        start_region,
        end_region,
    });

    Ok(format!(
        "Established {:?} smuggling route {} (region {} -> {}, profit: {:.0}, risk: {:.0}%)",
        route_type, route_id, start_region, end_region, base_profit, base_risk * 100.0
    ))
}
