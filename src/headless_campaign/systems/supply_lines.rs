//! Supply line interdiction system — every 200 ticks.
//!
//! Manages supply lines between the guild base and active parties in the field.
//! Enemy factions with military presence between source and destination can
//! interdict supply lines, doubling supply drain and halving trade income.
//! The guild can patrol routes to prevent interdiction, or interdict enemy
//! supply lines to weaken hostile faction operations.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Cadence: tick every 200 ticks (20 seconds game time).
const SUPPLY_LINE_TICK_INTERVAL: u64 = 200;

/// Base chance per tick that an unprotected supply line gets interdicted
/// by a hostile faction with military presence along the route.
const INTERDICTION_CHANCE: f32 = 0.20;

/// Minimum faction military strength to attempt interdiction.
const MIN_INTERDICTION_STRENGTH: f32 = 15.0;

/// Throughput penalty when a supply line passes through enemy territory.
const ENEMY_TERRITORY_PENALTY: f32 = 20.0;

/// Throughput bonus for road infrastructure (friendly territory).
const FRIENDLY_TERRITORY_BONUS: f32 = 10.0;

/// Weather throughput penalty (applied during winter).
const WINTER_PENALTY: f32 = 15.0;

pub fn tick_supply_lines(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % SUPPLY_LINE_TICK_INTERVAL != 0 {
        return;
    }

    let guild_faction_id = state.diplomacy.guild_faction_id;

    // Phase 1: Auto-generate supply lines from guild base to each active party
    regenerate_guild_supply_lines(state, guild_faction_id);

    // Phase 2: Update throughput based on distance, infrastructure, weather, territory
    update_throughput(state, guild_faction_id);

    // Phase 3: Enemy interdiction attempts
    attempt_enemy_interdiction(state, guild_faction_id, events);

    // Phase 4: Counter-interdiction from patrolling guild parties
    check_counter_interdiction(state, events);

    // Phase 5: Apply interdiction effects
    apply_interdiction_effects(state);

    // Phase 6: Guild interdiction of enemy supply lines
    tick_guild_interdiction(state, guild_faction_id, events);

    // Update system trackers
    update_trackers(state);
}

/// Create or refresh supply lines from guild base to each traveling/on-mission party.
fn regenerate_guild_supply_lines(state: &mut CampaignState, guild_faction_id: usize) {
    // Find the guild base region
    let base_pos = state.guild.base.position;
    let base_region_id = find_nearest_region(state, base_pos);

    // Collect active party info
    let party_info: Vec<(u32, usize)> = state
        .parties
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Fighting
            )
        })
        .map(|p| {
            let dest_region = find_nearest_region(state, p.position);
            (p.id, dest_region)
        })
        .collect();

    // Remove supply lines for parties that no longer exist or are idle/returned
    let active_party_ids: Vec<u32> = party_info.iter().map(|(id, _)| *id).collect();
    state.supply_lines.retain(|sl| {
        // Keep enemy supply lines (no party_id association)
        // Only prune guild supply lines whose party is gone
        sl.interdictor_faction_id.is_some()
            || active_party_ids.iter().any(|&pid| {
                // Match by destination region — each party has one supply line
                sl.source_region_id == base_region_id
                    && sl.destination_region_id
                        == party_info
                            .iter()
                            .find(|(id, _)| *id == pid)
                            .map(|(_, r)| *r)
                            .unwrap_or(usize::MAX)
            })
    });

    // Create supply lines for parties that don't have one yet
    for (party_id, dest_region) in &party_info {
        let exists = state.supply_lines.iter().any(|sl| {
            sl.source_region_id == base_region_id && sl.destination_region_id == *dest_region
        });
        if !exists && *dest_region != base_region_id {
            let id = state.next_supply_line_id;
            state.next_supply_line_id += 1;
            state.supply_lines.push(SupplyLine {
                id,
                source_region_id: base_region_id,
                destination_region_id: *dest_region,
                throughput: 100.0,
                interdicted: false,
                interdictor_faction_id: None,
                patrol_party_id: None,
                is_enemy_line: false,
                owner_faction_id: Some(guild_faction_id),
                associated_party_id: Some(*party_id),
            });
        }
    }
}

/// Update throughput based on distance, infrastructure, weather, and territory.
fn update_throughput(state: &mut CampaignState, guild_faction_id: usize) {
    let is_winter = state.overworld.season == Season::Winter;
    let region_count = state.overworld.regions.len();

    for sl in &mut state.supply_lines {
        if sl.is_enemy_line {
            continue; // Enemy lines managed separately
        }

        let mut throughput = 100.0_f32;

        // Distance penalty: each region of separation costs throughput
        let distance = region_distance(
            sl.source_region_id,
            sl.destination_region_id,
            &state.overworld.regions,
        );
        throughput -= (distance as f32) * 5.0;

        // Territory check: regions along the path
        let path = simple_path(
            sl.source_region_id,
            sl.destination_region_id,
            &state.overworld.regions,
        );
        for &region_id in &path {
            if region_id < region_count {
                let owner = state.overworld.regions[region_id].owner_faction_id;
                if owner == guild_faction_id {
                    throughput += FRIENDLY_TERRITORY_BONUS;
                } else {
                    // Check if hostile
                    let hostile = state
                        .factions
                        .iter()
                        .any(|f| f.id == owner && matches!(f.diplomatic_stance, DiplomaticStance::Hostile | DiplomaticStance::AtWar));
                    if hostile {
                        throughput -= ENEMY_TERRITORY_PENALTY;
                    }
                }
            }
        }

        // Weather penalty
        if is_winter {
            throughput -= WINTER_PENALTY;
        }

        sl.throughput = throughput.clamp(0.0, 100.0);
    }
}

/// Enemy factions attempt to interdict unprotected guild supply lines.
fn attempt_enemy_interdiction(
    state: &mut CampaignState,
    _guild_faction_id: usize,
    events: &mut Vec<WorldEvent>,
) {
    // Collect hostile faction info
    let hostile_factions: Vec<(usize, f32, Vec<usize>)> = state
        .factions
        .iter()
        .filter(|f| {
            matches!(
                f.diplomatic_stance,
                DiplomaticStance::Hostile | DiplomaticStance::AtWar
            ) && f.military_strength >= MIN_INTERDICTION_STRENGTH
        })
        .map(|f| {
            let territories: Vec<usize> = state
                .overworld
                .regions
                .iter()
                .filter(|r| r.owner_faction_id == f.id)
                .map(|r| r.id)
                .collect();
            (f.id, f.military_strength, territories)
        })
        .collect();

    let num_supply_lines = state.supply_lines.len();
    for i in 0..num_supply_lines {
        let sl = &state.supply_lines[i];
        if sl.interdicted || sl.is_enemy_line {
            continue;
        }

        // Check if a patrol party is protecting this route
        if sl.patrol_party_id.is_some() {
            continue;
        }

        // Check if any hostile faction has territory along this route
        let path = simple_path(
            sl.source_region_id,
            sl.destination_region_id,
            &state.overworld.regions,
        );

        for &(faction_id, strength, ref territories) in &hostile_factions {
            let has_presence = path.iter().any(|r| territories.contains(r));
            if !has_presence {
                continue;
            }

            // Chance of interdiction scales with faction strength
            let chance = INTERDICTION_CHANCE * (strength / 100.0).min(1.0);
            let roll = lcg_f32(&mut state.rng);
            if roll < chance {
                let sl = &mut state.supply_lines[i];
                sl.interdicted = true;
                sl.interdictor_faction_id = Some(faction_id);

                let src = sl.source_region_id;
                let dst = sl.destination_region_id;
                events.push(WorldEvent::SupplyLineInterdicted {
                    supply_line_id: state.supply_lines[i].id,
                    source_region_id: src,
                    destination_region_id: dst,
                    faction_id,
                });
                break; // Only one faction can interdict per tick
            }
        }
    }
}

/// Guild parties patrolling supply lines can restore interdicted lines.
fn check_counter_interdiction(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let num_supply_lines = state.supply_lines.len();
    for i in 0..num_supply_lines {
        let sl = &state.supply_lines[i];
        if !sl.interdicted || sl.is_enemy_line {
            continue;
        }

        // Check if a patrol party is assigned
        if let Some(patrol_id) = sl.patrol_party_id {
            // Verify patrol party still exists and is active
            let patrol_active = state
                .parties
                .iter()
                .any(|p| p.id == patrol_id && p.status != PartyStatus::Idle);

            if patrol_active {
                // Patrol clears the interdiction
                let sl = &mut state.supply_lines[i];
                sl.interdicted = false;
                sl.interdictor_faction_id = None;

                events.push(WorldEvent::SupplyLineRestored {
                    supply_line_id: sl.id,
                    source_region_id: sl.source_region_id,
                    destination_region_id: sl.destination_region_id,
                });
            } else {
                // Patrol party disbanded; clear assignment
                state.supply_lines[i].patrol_party_id = None;
            }
        }
    }
}

/// Apply effects of interdiction: double supply drain, halve trade income.
fn apply_interdiction_effects(state: &mut CampaignState) {
    // Collect interdicted party IDs
    let interdicted_parties: Vec<u32> = state
        .supply_lines
        .iter()
        .filter(|sl| sl.interdicted && !sl.is_enemy_line)
        .filter_map(|sl| sl.associated_party_id)
        .collect();

    // Double supply drain for interdicted parties (applied as extra drain)
    let dt_sec = CAMPAIGN_TICK_MS as f32 / 1000.0;
    let drain_rate = state.config.supply.drain_per_member_per_sec;

    for party in &mut state.parties {
        if interdicted_parties.contains(&party.id) {
            // Extra drain (doubles effective rate — base drain already applied by supply system)
            let member_count = party.member_ids.len() as f32;
            let extra_drain = drain_rate * member_count * dt_sec;
            party.supply_level = (party.supply_level - extra_drain).max(0.0);
        }
    }

    // Halve trade income from interdicted routes
    let interdicted_route_count = state
        .supply_lines
        .iter()
        .filter(|sl| sl.interdicted && !sl.is_enemy_line)
        .count();

    if interdicted_route_count > 0 {
        // Reduce guild trade income proportionally
        let total_lines = state
            .supply_lines
            .iter()
            .filter(|sl| !sl.is_enemy_line)
            .count()
            .max(1);
        let interdiction_ratio = interdicted_route_count as f32 / total_lines as f32;
        let income_loss = state.guild.total_trade_income * interdiction_ratio * 0.5;
        state.guild.gold -= income_loss.min(state.guild.gold);
    }
}

/// Guild parties interdicting enemy supply lines weaken hostile factions.
fn tick_guild_interdiction(
    state: &mut CampaignState,
    guild_faction_id: usize,
    events: &mut Vec<WorldEvent>,
) {
    // Auto-generate enemy supply lines for hostile factions with multiple territories
    let hostile_factions: Vec<(usize, Vec<usize>)> = state
        .factions
        .iter()
        .filter(|f| {
            matches!(
                f.diplomatic_stance,
                DiplomaticStance::Hostile | DiplomaticStance::AtWar
            )
        })
        .map(|f| {
            let territories: Vec<usize> = state
                .overworld
                .regions
                .iter()
                .filter(|r| r.owner_faction_id == f.id)
                .map(|r| r.id)
                .collect();
            (f.id, territories)
        })
        .filter(|(_, t)| t.len() >= 2)
        .collect();

    for (faction_id, territories) in &hostile_factions {
        // Create enemy supply line between first and last territory if not exists
        let src = territories[0];
        let dst = *territories.last().unwrap();
        let exists = state.supply_lines.iter().any(|sl| {
            sl.is_enemy_line
                && sl.owner_faction_id == Some(*faction_id)
                && sl.source_region_id == src
                && sl.destination_region_id == dst
        });
        if !exists {
            let id = state.next_supply_line_id;
            state.next_supply_line_id += 1;
            state.supply_lines.push(SupplyLine {
                id,
                source_region_id: src,
                destination_region_id: dst,
                throughput: 100.0,
                interdicted: false,
                interdictor_faction_id: None,
                patrol_party_id: None,
                is_enemy_line: true,
                owner_faction_id: Some(*faction_id),
                associated_party_id: None,
            });
        }
    }

    // Check guild parties assigned to interdict enemy supply lines
    let num_supply_lines = state.supply_lines.len();
    for i in 0..num_supply_lines {
        let sl = &state.supply_lines[i];
        if !sl.is_enemy_line || sl.interdicted {
            continue;
        }

        // Check if any guild party is interdicting this enemy line
        if let Some(patrol_id) = sl.patrol_party_id {
            let patrol_active = state
                .parties
                .iter()
                .any(|p| p.id == patrol_id && p.status != PartyStatus::Idle);

            if patrol_active {
                let roll = lcg_f32(&mut state.rng);
                if roll < 0.30 {
                    // Successfully interdict the enemy supply line
                    let sl = &mut state.supply_lines[i];
                    sl.interdicted = true;
                    sl.interdictor_faction_id = Some(guild_faction_id);

                    let faction_id = sl.owner_faction_id.unwrap_or(0);
                    events.push(WorldEvent::EnemySupplyDisrupted {
                        supply_line_id: sl.id,
                        faction_id,
                    });

                    // Weaken the enemy faction's military strength
                    if let Some(f) = state.factions.iter_mut().find(|f| f.id == faction_id) {
                        f.military_strength = (f.military_strength - 5.0).max(0.0);
                    }
                }
            } else {
                state.supply_lines[i].patrol_party_id = None;
            }
        }
    }
}

/// Update system trackers for supply lines.
fn update_trackers(state: &mut CampaignState) {
    let guild_lines = state
        .supply_lines
        .iter()
        .filter(|sl| !sl.is_enemy_line)
        .count() as u32;
    let interdicted = state
        .supply_lines
        .iter()
        .filter(|sl| sl.interdicted && !sl.is_enemy_line)
        .count() as u32;
    let enemy_disrupted = state
        .supply_lines
        .iter()
        .filter(|sl| sl.is_enemy_line && sl.interdicted)
        .count() as u32;

    state.system_trackers.active_supply_lines = guild_lines;
    state.system_trackers.interdicted_supply_lines = interdicted;
    state.system_trackers.enemy_supply_lines_disrupted = enemy_disrupted;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the region nearest to a world position.
fn find_nearest_region(state: &CampaignState, pos: (f32, f32)) -> usize {
    // Use the region whose center is closest. Since regions don't store position,
    // find by matching locations in the region, or fall back to region index 0.
    if state.overworld.regions.is_empty() {
        return 0;
    }

    // Simple heuristic: check locations owned by each region's faction
    let mut best_region = 0;
    let mut best_dist = f32::MAX;

    for region in &state.overworld.regions {
        // Use locations in the region as proxies for region center
        for loc in &state.overworld.locations {
            if loc.faction_owner == Some(region.owner_faction_id) {
                let dx = loc.position.0 - pos.0;
                let dy = loc.position.1 - pos.1;
                let dist = dx * dx + dy * dy;
                if dist < best_dist {
                    best_dist = dist;
                    best_region = region.id;
                }
            }
        }
    }

    best_region
}

/// Simple BFS distance between two regions via neighbor graph.
fn region_distance(from: usize, to: usize, regions: &[Region]) -> usize {
    if from == to {
        return 0;
    }
    if regions.is_empty() {
        return 1;
    }

    let mut visited = vec![false; regions.len()];
    let mut queue = std::collections::VecDeque::new();
    if from < visited.len() {
        visited[from] = true;
    }
    queue.push_back((from, 0usize));

    while let Some((current, dist)) = queue.pop_front() {
        if current == to {
            return dist;
        }
        if let Some(region) = regions.iter().find(|r| r.id == current) {
            for &neighbor in &region.neighbors {
                if neighbor < visited.len() && !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }
    }

    // Unreachable: return large distance
    regions.len()
}

/// Return the regions along a simple BFS path (excluding source and destination).
fn simple_path(from: usize, to: usize, regions: &[Region]) -> Vec<usize> {
    if from == to || regions.is_empty() {
        return Vec::new();
    }

    let n = regions.len();
    let mut visited = vec![false; n];
    let mut parent = vec![usize::MAX; n];
    let mut queue = std::collections::VecDeque::new();

    if from < n {
        visited[from] = true;
    }
    queue.push_back(from);

    while let Some(current) = queue.pop_front() {
        if current == to {
            break;
        }
        if let Some(region) = regions.iter().find(|r| r.id == current) {
            for &neighbor in &region.neighbors {
                if neighbor < n && !visited[neighbor] {
                    visited[neighbor] = true;
                    parent[neighbor] = current;
                    queue.push_back(neighbor);
                }
            }
        }
    }

    // Reconstruct path (intermediate nodes only)
    let mut path = Vec::new();
    let mut cur = to;
    while cur != from && cur < n && parent[cur] != usize::MAX {
        if cur != to && cur != from {
            path.push(cur);
        }
        cur = parent[cur];
    }
    path.reverse();
    path
}
