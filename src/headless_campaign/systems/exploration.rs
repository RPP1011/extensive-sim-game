//! Map exploration tracking system — every 100 ticks (~10s).
//!
//! Tracks which tiles of the overworld have been revealed by party movement,
//! discovers landmarks when parties explore nearby, and fires milestone events
//! at 25%/50%/75%/100% exploration coverage. Integrates with the fog-of-war
//! scouting system by boosting region visibility for explored areas.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Cadence: runs every 100 ticks.
const EXPLORATION_INTERVAL: u64 = 3;

/// Tile cell size in world units. The overworld is divided into a grid of
/// cells this wide/tall for exploration tracking.
const TILE_SIZE: f32 = 10.0;

/// Sight range in tiles for a normal party.
const BASE_SIGHT_RANGE: i32 = 2;

/// Extra sight range for parties with a Cartography adventurer (50% wider).
const CARTOGRAPHY_SIGHT_RANGE: i32 = 3;

/// Discovery radius: how close (in world units) a party must be to a landmark
/// to discover it.
const LANDMARK_DISCOVERY_RADIUS: f32 = 12.0;

/// World bounds for the exploration grid. Positions in the overworld range
/// roughly from -50 to +50 on each axis, giving a 10x10 grid of 10-unit cells.
const WORLD_MIN: f32 = -50.0;
const WORLD_MAX: f32 = 50.0;

/// Number of landmark templates to scatter across the map.
const LANDMARK_TEMPLATES: &[(&str, &str)] = &[
    ("Ancient Watchtower", "ancient_ruins"),
    ("Forgotten Library", "ancient_ruins"),
    ("Collapsed Mine", "ancient_ruins"),
    ("Mysterious Shrine", "mysterious_shrine"),
    ("Spirit Well", "mysterious_shrine"),
    ("Moonstone Altar", "mysterious_shrine"),
    ("Abandoned Camp", "abandoned_camp"),
    ("Deserted Caravan", "abandoned_camp"),
    ("Old Supply Cache", "abandoned_camp"),
    ("Windswept Summit", "scenic_vista"),
    ("Crystal Overlook", "scenic_vista"),
    ("Dragon's Perch", "scenic_vista"),
    ("Sunken Garden", "ancient_ruins"),
    ("Hermit's Cave", "mysterious_shrine"),
    ("Ranger Outpost", "abandoned_camp"),
];

/// Initialize exploration state for a campaign. Called once when the campaign
/// starts (total_tiles == 0 indicates uninitialized). Scatters landmarks
/// deterministically using the campaign RNG.
fn init_exploration(state: &mut CampaignState) {
    let grid_w = ((WORLD_MAX - WORLD_MIN) / TILE_SIZE).ceil() as u32;
    let grid_h = grid_w;
    state.exploration.total_tiles = grid_w * grid_h;

    // Scatter 10-15 landmarks across the map using deterministic RNG.
    let landmark_count = 10 + (lcg_next(&mut state.rng) % 6) as usize;
    let count = landmark_count.min(LANDMARK_TEMPLATES.len());

    // Shuffle template indices deterministically.
    let mut indices: Vec<usize> = (0..LANDMARK_TEMPLATES.len()).collect();
    for i in (1..indices.len()).rev() {
        let j = (lcg_next(&mut state.rng) as usize) % (i + 1);
        indices.swap(i, j);
    }

    for &idx in indices.iter().take(count) {
        let (name, reward_type) = LANDMARK_TEMPLATES[idx];
        // Random position within world bounds, biased away from center.
        let x = WORLD_MIN + lcg_f32(&mut state.rng) * (WORLD_MAX - WORLD_MIN);
        let y = WORLD_MIN + lcg_f32(&mut state.rng) * (WORLD_MAX - WORLD_MIN);
        state.exploration.landmarks.push(Landmark {
            name: name.to_string(),
            position: (x, y),
            reward_type: reward_type.to_string(),
            discovered: false,
        });
    }

    // Mark guild base tiles as explored (the guild knows its own surroundings).
    let base_pos = state.guild.base.position;
    let base_tile = world_to_tile(base_pos);
    for dx in -BASE_SIGHT_RANGE..=BASE_SIGHT_RANGE {
        for dy in -BASE_SIGHT_RANGE..=BASE_SIGHT_RANGE {
            let tx = base_tile.0 + dx;
            let ty = base_tile.1 + dy;
            if is_valid_tile(tx, ty) {
                state.exploration.explored_tiles.insert((tx, ty));
            }
        }
    }
    update_percentage(&mut state.exploration);
}

/// Main exploration tick. Runs every `EXPLORATION_INTERVAL` ticks.
pub fn tick_exploration(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % EXPLORATION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Lazy initialization: first tick that runs, set up the grid and landmarks.
    if state.exploration.total_tiles == 0 {
        init_exploration(state);
    }

    let prev_percentage = state.exploration.exploration_percentage;

    // 1. Track party positions — mark tiles within sight range as explored.
    let party_data: Vec<((f32, f32), bool)> = state
        .parties
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Returning
            )
        })
        .map(|p| {
            // Check if any member has the Cartography trait.
            let has_cartography = p.member_ids.iter().any(|&mid| {
                state
                    .adventurers
                    .iter()
                    .find(|a| a.id == mid)
                    .map(|a| {
                        a.traits
                            .iter()
                            .any(|t| t.eq_ignore_ascii_case("cartography"))
                    })
                    .unwrap_or(false)
            });
            (p.position, has_cartography)
        })
        .collect();

    for (pos, has_cartography) in &party_data {
        let sight = if *has_cartography {
            CARTOGRAPHY_SIGHT_RANGE
        } else {
            BASE_SIGHT_RANGE
        };
        let center = world_to_tile(*pos);
        for dx in -sight..=sight {
            for dy in -sight..=sight {
                let tx = center.0 + dx;
                let ty = center.1 + dy;
                if is_valid_tile(tx, ty) {
                    state.exploration.explored_tiles.insert((tx, ty));
                }
            }
        }
    }

    update_percentage(&mut state.exploration);
    let new_percentage = state.exploration.exploration_percentage;

    // 2. Emit TileExplored event if percentage changed meaningfully (>= 1%).
    if (new_percentage - prev_percentage).abs() >= 1.0 {
        events.push(WorldEvent::TileExplored {
            percentage: new_percentage,
        });
    }

    // 3. Check exploration milestones.
    for &milestone in &[25u32, 50, 75, 100] {
        if new_percentage >= milestone as f32
            && !state.exploration.milestones_fired.contains(&milestone)
        {
            state.exploration.milestones_fired.push(milestone);
            events.push(WorldEvent::ExplorationMilestone {
                percentage: milestone,
            });
            apply_milestone_reward(state, milestone, events);
        }
    }

    // 4. Landmark discovery — check if any party is near an undiscovered landmark.
    //    We need to collect discoveries first to avoid borrow issues.
    let mut discoveries: Vec<(usize, u32)> = Vec::new();
    for (i, landmark) in state.exploration.landmarks.iter().enumerate() {
        if landmark.discovered {
            continue;
        }
        for party in &state.parties {
            if !matches!(
                party.status,
                PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Returning
            ) {
                continue;
            }
            let dx = party.position.0 - landmark.position.0;
            let dy = party.position.1 - landmark.position.1;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq <= LANDMARK_DISCOVERY_RADIUS * LANDMARK_DISCOVERY_RADIUS {
                // Use first member as discoverer.
                let discoverer = party.member_ids.first().copied().unwrap_or(0);
                discoveries.push((i, discoverer));
                break; // Only one party needs to find it.
            }
        }
    }

    let tick = state.tick;
    for (idx, discoverer) in discoveries {
        let landmark = &mut state.exploration.landmarks[idx];
        landmark.discovered = true;

        let discovery = LandmarkDiscovery {
            name: landmark.name.clone(),
            position: landmark.position,
            discovery_tick: tick,
            reward_type: landmark.reward_type.clone(),
            discovered_by: discoverer,
        };

        events.push(WorldEvent::LandmarkDiscovered {
            name: discovery.name.clone(),
            reward: discovery.reward_type.clone(),
        });

        // Apply landmark reward.
        apply_landmark_reward(state, &discovery);
        state.exploration.landmarks_discovered.push(discovery);
    }

    // 5. Feed exploration into scouting visibility: explored tiles near region
    //    locations boost that region's visibility slightly.
    boost_scouting_from_exploration(state);
}

/// Convert a world position to a tile coordinate.
fn world_to_tile(pos: (f32, f32)) -> (i32, i32) {
    let tx = ((pos.0 - WORLD_MIN) / TILE_SIZE).floor() as i32;
    let ty = ((pos.1 - WORLD_MIN) / TILE_SIZE).floor() as i32;
    (tx, ty)
}

/// Check if a tile coordinate is within the valid grid.
fn is_valid_tile(tx: i32, ty: i32) -> bool {
    let grid_size = ((WORLD_MAX - WORLD_MIN) / TILE_SIZE).ceil() as i32;
    tx >= 0 && ty >= 0 && tx < grid_size && ty < grid_size
}

/// Recompute the cached exploration percentage.
fn update_percentage(exploration: &mut ExplorationState) {
    if exploration.total_tiles == 0 {
        exploration.exploration_percentage = 0.0;
    } else {
        exploration.exploration_percentage =
            (exploration.explored_tiles.len() as f32 / exploration.total_tiles as f32) * 100.0;
        // Clamp to 100 in case of float imprecision.
        if exploration.exploration_percentage > 100.0 {
            exploration.exploration_percentage = 100.0;
        }
    }
}

/// Apply rewards for reaching an exploration milestone.
fn apply_milestone_reward(
    state: &mut CampaignState,
    milestone: u32,
    events: &mut Vec<WorldEvent>,
) {
    match milestone {
        25 => {
            // Reveal one hidden dungeon: mark the first unscouted dungeon as scouted.
            if let Some(loc) = state
                .overworld
                .locations
                .iter_mut()
                .find(|l| l.location_type == LocationType::Dungeon && !l.scouted)
            {
                loc.scouted = true;
                events.push(WorldEvent::ScoutReport {
                    location_id: loc.id,
                    threat_level: loc.threat_level,
                });
            }
        }
        50 => {
            // +10 reputation, unlock cartography bonuses.
            state.guild.reputation = (state.guild.reputation + 10.0).min(100.0);
            events.push(WorldEvent::CampaignMilestone {
                description: "50% explored — Cartography bonus unlocked, +10 reputation"
                    .to_string(),
            });
        }
        75 => {
            // Reveal all faction positions (boost all region visibility).
            for region in &mut state.overworld.regions {
                region.visibility = region.visibility.max(0.8);
            }
            // Discover an ancient ruin: mark first unscouted ruin as scouted.
            if let Some(loc) = state
                .overworld
                .locations
                .iter_mut()
                .find(|l| l.location_type == LocationType::Ruin && !l.scouted)
            {
                loc.scouted = true;
                events.push(WorldEvent::ScoutReport {
                    location_id: loc.id,
                    threat_level: loc.threat_level,
                });
            }
        }
        100 => {
            // Massive reputation bonus + achievement.
            state.guild.reputation = (state.guild.reputation + 25.0).min(100.0);
            events.push(WorldEvent::CampaignMilestone {
                description: "100% explored — Master Cartographer achievement, +25 reputation"
                    .to_string(),
            });
        }
        _ => {}
    }
}

/// Apply reward effects from discovering a landmark.
fn apply_landmark_reward(state: &mut CampaignState, discovery: &LandmarkDiscovery) {
    match discovery.reward_type.as_str() {
        "ancient_ruins" => {
            // Loot: bonus gold.
            state.guild.gold += 50.0;
        }
        "mysterious_shrine" => {
            // Blessing: boost morale for the discoverer.
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == discovery.discovered_by)
            {
                adv.morale = (adv.morale + 15.0).min(100.0);
                adv.stress = (adv.stress - 10.0).max(0.0);
            }
        }
        "abandoned_camp" => {
            // Supplies: bonus supplies.
            state.guild.supplies += 30.0;
        }
        "scenic_vista" => {
            // Morale boost to all adventurers in the discoverer's party.
            let party_id = state
                .adventurers
                .iter()
                .find(|a| a.id == discovery.discovered_by)
                .and_then(|a| a.party_id);
            if let Some(pid) = party_id {
                let member_ids: Vec<u32> = state
                    .parties
                    .iter()
                    .find(|p| p.id == pid)
                    .map(|p| p.member_ids.clone())
                    .unwrap_or_default();
                for mid in member_ids {
                    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                        adv.morale = (adv.morale + 10.0).min(100.0);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Boost scouting visibility for regions whose locations are near explored tiles.
fn boost_scouting_from_exploration(state: &mut CampaignState) {
    // For each location, check how many nearby tiles are explored.
    // If substantial exploration exists near a location, boost its region's visibility.
    let location_data: Vec<(Option<usize>, (f32, f32))> = state
        .overworld
        .locations
        .iter()
        .map(|l| (l.faction_owner, l.position))
        .collect();

    for (faction_owner, pos) in &location_data {
        let center = world_to_tile(*pos);
        let mut explored_nearby = 0u32;
        let check_range = 2;
        let total_nearby = ((check_range * 2 + 1) * (check_range * 2 + 1)) as u32;

        for dx in -check_range..=check_range {
            for dy in -check_range..=check_range {
                if state
                    .exploration
                    .explored_tiles
                    .contains(&(center.0 + dx, center.1 + dy))
                {
                    explored_nearby += 1;
                }
            }
        }

        // If more than half the nearby tiles are explored, give a small visibility bump.
        if explored_nearby > total_nearby / 2 {
            if let Some(fid) = faction_owner {
                for region in &mut state.overworld.regions {
                    if region.owner_faction_id == *fid {
                        region.visibility = (region.visibility + 0.02).min(1.0);
                    }
                }
            }
        }
    }
}
