//! Caravan and trade route system — every 50 ticks (~5s).
//!
//! Caravans travel between guild-controlled settlements along established
//! trade routes, generating passive gold on delivery. Hostile faction parties
//! within range can raid caravans, stealing gold and destroying them.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Cadence: tick every 50 ticks (5 seconds game time).
const CARAVAN_TICK_INTERVAL: u64 = 50;

/// Distance (tiles) at which hostile parties can intercept caravans.
const RAID_RANGE: f32 = 3.0;

/// Base caravan speed (progress per tick, at cadence).
const CARAVAN_SPEED: f32 = 0.05;

/// Base guard strength for a new caravan (no assigned guards).
const BASE_GUARD_STRENGTH: f32 = 10.0;

/// Gold carried by a caravan (fraction of route gold_per_trip held in transit).
const CARAVAN_GOLD_RATIO: f32 = 1.0;

pub fn tick_caravans(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % CARAVAN_TICK_INTERVAL != 0 {
        return;
    }

    // Spawn caravans for active routes that don't already have one
    spawn_caravans(state);

    // Move caravans and handle arrivals
    move_caravans(state, events);

    // Check for raids by hostile faction parties
    check_raids(state, events);
}

/// Spawn a caravan for each active route that doesn't already have one in transit.
fn spawn_caravans(state: &mut CampaignState) {
    let active_route_ids: Vec<u32> = state
        .trade_routes
        .iter()
        .filter(|r| r.active)
        .map(|r| r.id)
        .collect();

    for route_id in active_route_ids {
        let has_caravan = state.caravans.iter().any(|c| c.route_id == route_id);
        if has_caravan {
            continue;
        }

        let route = match state.trade_routes.iter().find(|r| r.id == route_id) {
            Some(r) => r,
            None => continue,
        };

        let start_pos = match state
            .overworld
            .locations
            .iter()
            .find(|loc| loc.id == route.start_location_id)
        {
            Some(loc) => loc.position,
            None => continue,
        };

        let caravan_id = state.next_caravan_id;
        state.next_caravan_id += 1;

        state.caravans.push(Caravan {
            id: caravan_id,
            route_id,
            position: start_pos,
            progress: 0.0,
            gold_carried: route.gold_per_trip * CARAVAN_GOLD_RATIO,
            guard_strength: BASE_GUARD_STRENGTH,
            forward: true,
        });
    }
}

/// Advance caravan progress and handle arrivals.
fn move_caravans(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect route data to avoid borrow conflicts
    struct RouteInfo {
        id: u32,
        start_id: usize,
        end_id: usize,
        gold_per_trip: f32,
    }
    let route_infos: Vec<RouteInfo> = state
        .trade_routes
        .iter()
        .map(|r| RouteInfo {
            id: r.id,
            start_id: r.start_location_id,
            end_id: r.end_location_id,
            gold_per_trip: r.gold_per_trip,
        })
        .collect();

    // Collect location positions
    let locations: Vec<(usize, (f32, f32))> = state
        .overworld
        .locations
        .iter()
        .map(|loc| (loc.id, loc.position))
        .collect();

    let find_pos = |loc_id: usize| -> Option<(f32, f32)> {
        locations.iter().find(|(id, _)| *id == loc_id).map(|(_, p)| *p)
    };

    let mut delivered: Vec<(u32, f32)> = Vec::new(); // (route_id, gold)

    for caravan in &mut state.caravans {
        let route = match route_infos.iter().find(|r| r.id == caravan.route_id) {
            Some(r) => r,
            None => continue,
        };

        let start_pos = match find_pos(route.start_id) {
            Some(p) => p,
            None => continue,
        };
        let end_pos = match find_pos(route.end_id) {
            Some(p) => p,
            None => continue,
        };

        // Advance progress
        caravan.progress += CARAVAN_SPEED;

        // Interpolate position
        let (from, to) = if caravan.forward {
            (start_pos, end_pos)
        } else {
            (end_pos, start_pos)
        };
        let t = caravan.progress.clamp(0.0, 1.0);
        caravan.position = (
            from.0 + (to.0 - from.0) * t,
            from.1 + (to.1 - from.1) * t,
        );

        // Check arrival
        if caravan.progress >= 1.0 {
            delivered.push((route.id, caravan.gold_carried));
            // Reset: reverse direction
            caravan.progress = 0.0;
            caravan.forward = !caravan.forward;
            caravan.gold_carried = route.gold_per_trip * CARAVAN_GOLD_RATIO;
        }
    }

    // Apply gold deliveries
    for (route_id, gold) in delivered {
        state.guild.gold += gold;
        events.push(WorldEvent::CaravanCompleted {
            route_id,
            gold_delivered: gold,
        });
        events.push(WorldEvent::GoldChanged {
            amount: gold,
            reason: format!("Caravan delivery on route {}", route_id),
        });
    }
}

/// Check if hostile faction parties are close enough to raid caravans.
fn check_raids(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let guild_faction_id = state.diplomacy.guild_faction_id;

    // Collect hostile faction IDs
    let hostile_faction_ids: Vec<usize> = state
        .factions
        .iter()
        .filter(|f| {
            f.id != guild_faction_id
                && (f.at_war_with.contains(&guild_faction_id)
                    || matches!(
                        f.diplomatic_stance,
                        DiplomaticStance::AtWar | DiplomaticStance::Hostile
                    ))
        })
        .map(|f| f.id)
        .collect();

    if hostile_faction_ids.is_empty() {
        return;
    }

    // Collect hostile parties with positions and faction info
    struct HostileParty {
        _id: u32,
        position: (f32, f32),
        faction_id: usize,
        strength: f32,
    }

    let hostile_parties: Vec<HostileParty> = state
        .parties
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                PartyStatus::Traveling | PartyStatus::OnMission
            )
        })
        .filter_map(|p| {
            // Find the faction of this party's members
            for mid in &p.member_ids {
                if let Some(adv) = state.adventurers.iter().find(|a| a.id == *mid) {
                    if let Some(fid) = adv.faction_id {
                        if hostile_faction_ids.contains(&fid) {
                            let strength = p.member_ids.len() as f32 * 15.0;
                            return Some(HostileParty {
                                _id: p.id,
                                position: p.position,
                                faction_id: fid,
                                strength,
                            });
                        }
                    }
                }
            }
            None
        })
        .collect();

    if hostile_parties.is_empty() {
        return;
    }

    // Check each caravan against hostile parties
    let mut raided_caravan_ids: Vec<(u32, f32, usize)> = Vec::new(); // (caravan_id, gold_stolen, faction_id)

    for caravan in &state.caravans {
        for raider in &hostile_parties {
            let dx = caravan.position.0 - raider.position.0;
            let dy = caravan.position.1 - raider.position.1;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < RAID_RANGE {
                // Raid check: compare raider strength vs guard strength
                // Use deterministic RNG
                let roll = lcg_f32(&mut state.rng);
                let raid_chance = raider.strength / (raider.strength + caravan.guard_strength);

                if roll < raid_chance {
                    // Raid succeeds
                    raided_caravan_ids.push((caravan.id, caravan.gold_carried, raider.faction_id));
                }
                // Failed raid: caravan continues (raider takes minor losses but
                // we don't model raider HP here, just let the caravan survive)
                break; // Only one raid attempt per caravan per tick
            }
        }
    }

    // Remove raided caravans and emit events
    for (caravan_id, gold_stolen, faction_id) in raided_caravan_ids {
        let route_id = state
            .caravans
            .iter()
            .find(|c| c.id == caravan_id)
            .map(|c| c.route_id)
            .unwrap_or(0);

        state.caravans.retain(|c| c.id != caravan_id);

        let faction_name = state
            .factions
            .iter()
            .find(|f| f.id == faction_id)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| format!("Faction {}", faction_id));

        events.push(WorldEvent::CaravanRaided {
            route_id,
            gold_stolen,
            raider_faction: faction_name,
        });
    }
}
