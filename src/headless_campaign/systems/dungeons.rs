//! Underground dungeon network system — every 300 ticks.
//!
//! Manages a network of connected underground dungeons beneath regions.
//! Dungeons provide shortcuts between regions but are dangerous.
//! Monster strength and loot regenerate slowly over time.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Cadence: runs every 300 ticks.
const DUNGEON_INTERVAL: u64 = 10;

/// Monster strength regeneration per tick (capped at base strength).
const MONSTER_REGEN_RATE: f32 = 1.0;

/// Loot regeneration per tick.
const LOOT_REGEN_RATE: f32 = 0.1;

/// Deep dungeon depth threshold for spawning crisis threats.
const DEEP_DUNGEON_DEPTH: u32 = 4;

/// Threat level multiplier for deep dungeon crisis events.
const DEEP_DUNGEON_THREAT_MULTIPLIER: f32 = 1.5;

/// Visibility threshold for dungeon rumor discovery.
const RUMOR_VISIBILITY_THRESHOLD: f32 = 0.5;

pub fn tick_dungeons(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % DUNGEON_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.dungeons.is_empty() {
        return;
    }

    let num_dungeons = state.dungeons.len();

    for i in 0..num_dungeons {
        // Monster strength regeneration: +1 per tick if below base (depth * 20)
        let base_strength = state.dungeons[i].depth as f32 * 20.0;
        if state.dungeons[i].monster_strength < base_strength {
            state.dungeons[i].monster_strength =
                (state.dungeons[i].monster_strength + MONSTER_REGEN_RATE).min(base_strength);
        }

        // Loot regeneration: +0.1 per tick, cap at 100
        if state.dungeons[i].loot_remaining < 100.0 {
            state.dungeons[i].loot_remaining =
                (state.dungeons[i].loot_remaining + LOOT_REGEN_RATE).min(100.0);
        }

        // Unexplored dungeons emit rumors when scouting visibility > 0.5 in their region
        let region_id = state.dungeons[i].region_id;
        let explored = state.dungeons[i].explored;
        let dungeon_name = state.dungeons[i].name.clone();
        let dungeon_depth = state.dungeons[i].depth;

        if explored < 1.0 {
            if let Some(region) = state.overworld.regions.get(region_id) {
                if region.visibility > RUMOR_VISIBILITY_THRESHOLD {
                    events.push(WorldEvent::DungeonRumorHeard {
                        dungeon_id: state.dungeons[i].id,
                        dungeon_name: dungeon_name.clone(),
                        region_name: region.name.clone(),
                    });
                }
            }
        }

        // Deep dungeons (depth 4-5) can spawn rare crisis-level threats if ignored
        if dungeon_depth >= DEEP_DUNGEON_DEPTH
            && state.dungeons[i].monster_strength >= base_strength
            && explored < 20.0
        {
            // Roll for threat emergence using deterministic RNG
            let roll = lcg_f32(&mut state.rng);
            if roll < 0.05 {
                // 5% chance per 300-tick interval
                let threat = state.dungeons[i].monster_strength * DEEP_DUNGEON_THREAT_MULTIPLIER;
                // Increase region threat level
                if let Some(region) = state.overworld.regions.get_mut(region_id) {
                    region.threat_level = (region.threat_level + threat * 0.2).min(100.0);
                }
                events.push(WorldEvent::DungeonThreatEmerged {
                    dungeon_id: state.dungeons[i].id,
                    dungeon_name,
                    threat_level: threat,
                });
            }
        }
    }
}

/// Process an ExploreDungeon action. Returns (success, message).
pub fn apply_explore_dungeon(
    state: &mut CampaignState,
    dungeon_id: u32,
    party_id: u32,
    events: &mut Vec<WorldEvent>,
) -> (bool, String) {
    // Find dungeon
    let dungeon_idx = match state.dungeons.iter().position(|d| d.id == dungeon_id) {
        Some(idx) => idx,
        None => return (false, format!("Dungeon {} not found", dungeon_id)),
    };

    // Find party
    let party_idx = match state.parties.iter().position(|p| p.id == party_id) {
        Some(idx) => idx,
        None => return (false, format!("Party {} not found", party_id)),
    };

    // Party must be idle
    if state.parties[party_idx].status != PartyStatus::Idle {
        return (false, "Party is not idle".into());
    }

    // Calculate party strength from members
    let party_strength: f32 = state.parties[party_idx]
        .member_ids
        .iter()
        .filter_map(|id| state.adventurers.iter().find(|a| a.id == *id))
        .map(|a| a.stats.attack + a.stats.defense + a.stats.ability_power)
        .sum();

    let monster_strength = state.dungeons[dungeon_idx].monster_strength;
    let dungeon_name = state.dungeons[dungeon_idx].name.clone();

    // Combat check: party strength vs monster_strength with RNG variance
    let roll = lcg_f32(&mut state.rng);
    let effective_party = party_strength * (0.7 + roll * 0.6); // 70-130% variance
    let success = effective_party > monster_strength;

    if success {
        // Exploration gain (10-25% per successful exploration)
        let explore_gain = 10.0 + lcg_f32(&mut state.rng) * 15.0;
        state.dungeons[dungeon_idx].explored =
            (state.dungeons[dungeon_idx].explored + explore_gain).min(100.0);

        // Reduce monster strength from clearing
        state.dungeons[dungeon_idx].monster_strength *= 0.6;

        // Loot rewards
        let loot_found = state.dungeons[dungeon_idx].loot_remaining.min(
            10.0 + lcg_f32(&mut state.rng) * 20.0,
        );
        state.dungeons[dungeon_idx].loot_remaining -= loot_found;
        let gold_reward = loot_found * 3.0; // Each loot point = 3 gold
        state.guild.gold += gold_reward;

        events.push(WorldEvent::DungeonExplored {
            dungeon_id,
            dungeon_name: dungeon_name.clone(),
            explored_pct: state.dungeons[dungeon_idx].explored,
        });

        if gold_reward > 0.0 {
            events.push(WorldEvent::DungeonLootFound {
                dungeon_id,
                dungeon_name: dungeon_name.clone(),
                gold_found: gold_reward,
            });
        }

        // Discover connections to other dungeons (if exploration > 50%)
        if state.dungeons[dungeon_idx].explored > 50.0 {
            // Check for undiscovered connections
            let region_id = state.dungeons[dungeon_idx].region_id;
            let current_connections = state.dungeons[dungeon_idx].connected_to.clone();

            // Find adjacent dungeons (in neighboring regions) not yet connected
            let neighbor_regions: Vec<usize> = state
                .overworld
                .regions
                .get(region_id)
                .map(|r| r.neighbors.clone())
                .unwrap_or_default();

            let mut new_connections = Vec::new();
            for other_dungeon in &state.dungeons {
                if other_dungeon.id != dungeon_id
                    && neighbor_regions.contains(&other_dungeon.region_id)
                    && !current_connections.contains(&other_dungeon.id)
                {
                    new_connections.push((other_dungeon.id, other_dungeon.name.clone()));
                }
            }

            // Discover one connection at a time (deterministic via RNG)
            if !new_connections.is_empty() {
                let pick = (lcg_next(&mut state.rng) as usize) % new_connections.len();
                let (connected_id, connected_name) = new_connections[pick].clone();

                // Add bidirectional connection
                state.dungeons[dungeon_idx].connected_to.push(connected_id);
                if let Some(other) = state.dungeons.iter_mut().find(|d| d.id == connected_id) {
                    if !other.connected_to.contains(&dungeon_id) {
                        other.connected_to.push(dungeon_id);
                    }
                }

                events.push(WorldEvent::DungeonConnectionDiscovered {
                    from_dungeon_id: dungeon_id,
                    from_name: dungeon_name.clone(),
                    to_dungeon_id: connected_id,
                    to_name: connected_name,
                });
            }
        }

        // Give party members some stress/fatigue from dungeon exploration
        for member_id in &state.parties[party_idx].member_ids {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *member_id) {
                adv.stress = (adv.stress + 8.0).min(100.0);
                adv.fatigue = (adv.fatigue + 12.0).min(100.0);
            }
        }

        (true, format!("Explored {}. Found {:.0} gold.", dungeon_name, gold_reward))
    } else {
        // Failure: party takes casualties, retreats
        for member_id in &state.parties[party_idx].member_ids {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *member_id) {
                let injury_amount = 15.0 + lcg_f32(&mut state.rng) * 20.0;
                adv.injury = (adv.injury + injury_amount).min(100.0);
                adv.stress = (adv.stress + 20.0).min(100.0);
                adv.fatigue = (adv.fatigue + 25.0).min(100.0);
                adv.morale = (adv.morale - 15.0).max(0.0);
                if adv.injury >= 90.0 {
                    adv.status = AdventurerStatus::Injured;
                }
            }
        }

        (false, format!("Failed to explore {}. Party retreated with injuries.", dungeon_name))
    }
}

/// Process a UseDungeonShortcut action. Returns (success, message).
pub fn apply_dungeon_shortcut(
    state: &mut CampaignState,
    from_dungeon_id: u32,
    to_dungeon_id: u32,
    party_id: u32,
    _events: &mut Vec<WorldEvent>,
) -> (bool, String) {
    // Validate both dungeons exist
    let from_idx = match state.dungeons.iter().position(|d| d.id == from_dungeon_id) {
        Some(idx) => idx,
        None => return (false, format!("Dungeon {} not found", from_dungeon_id)),
    };

    let to_idx = match state.dungeons.iter().position(|d| d.id == to_dungeon_id) {
        Some(idx) => idx,
        None => return (false, format!("Dungeon {} not found", to_dungeon_id)),
    };

    // Validate they are connected
    if !state.dungeons[from_idx].connected_to.contains(&to_dungeon_id) {
        return (false, "Dungeons are not connected".into());
    }

    // Both must be explored (>= 50%)
    if state.dungeons[from_idx].explored < 50.0 {
        return (false, "Origin dungeon not sufficiently explored".into());
    }
    if state.dungeons[to_idx].explored < 50.0 {
        return (false, "Destination dungeon not sufficiently explored".into());
    }

    // Find and validate party
    let party_idx = match state.parties.iter().position(|p| p.id == party_id) {
        Some(idx) => idx,
        None => return (false, format!("Party {} not found", party_id)),
    };

    if state.parties[party_idx].status != PartyStatus::Idle {
        return (false, "Party is not idle".into());
    }

    // Teleport party to destination region's approximate position
    let dest_region_id = state.dungeons[to_idx].region_id;
    let dest_name = state.dungeons[to_idx].name.clone();

    // Use first location in destination region as reference, or region center (0,0)
    let dest_pos = state
        .overworld
        .locations
        .iter()
        .find(|loc| {
            // Match location to region by faction ownership or proximity
            loc.faction_owner.map_or(false, |f| {
                state.overworld.regions.get(dest_region_id)
                    .map_or(false, |r| r.owner_faction_id == f)
            })
        })
        .map(|loc| loc.position)
        .unwrap_or((0.0, 0.0));

    state.parties[party_idx].position = dest_pos;

    // Minor stress from dungeon traversal
    for member_id in &state.parties[party_idx].member_ids {
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *member_id) {
            adv.stress = (adv.stress + 5.0).min(100.0);
            adv.fatigue = (adv.fatigue + 5.0).min(100.0);
        }
    }

    (true, format!("Party traveled through dungeon shortcut to {}", dest_name))
}
