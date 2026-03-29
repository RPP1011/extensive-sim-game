#![allow(unused)]
//! Map exploration tracking system — every 3 ticks.
//!
//! Tracks NPC movement across the world to reveal regions and discover
//! landmarks. Explored areas near settlements boost treasury (representing
//! cartography bonuses). Milestone rewards are expressed as treasury updates
//! and fidelity changes.
//!
//! Ported from `crates/headless_campaign/src/systems/exploration.rs`.
//!
//! NEEDS STATE: `exploration: ExplorationState` on WorldState
//! NEEDS STATE: `ExplorationState { explored_tiles: HashSet<(i32,i32)>, total_tiles: u32,
//!              exploration_percentage: f32, milestones_fired: Vec<u32>,
//!              landmarks: Vec<Landmark>, landmarks_discovered: Vec<LandmarkDiscovery> }`
//! NEEDS STATE: `Landmark { name, position, reward_type, discovered }`
//! NEEDS STATE: `LandmarkDiscovery { name, position, discovery_tick, reward_type, discovered_by }`
//! NEEDS DELTA: RevealTile { x: i32, y: i32 }
//! NEEDS DELTA: DiscoverLandmark { landmark_index: usize, discovered_by: u32 }
//! NEEDS DELTA: UpdateReputation { entity_id, delta }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::fidelity::Fidelity;
use crate::world_sim::state::{EntityKind, WorldState, WorldTeam};

/// Cadence: runs every 3 ticks.
const EXPLORATION_INTERVAL: u64 = 3;

/// Tile cell size in world units.
const TILE_SIZE: f32 = 10.0;

/// Base sight range in tiles for a normal NPC.
const BASE_SIGHT_RANGE: i32 = 2;

/// Landmark discovery radius in world units.
const LANDMARK_DISCOVERY_RADIUS_SQ: f32 = 144.0; // 12^2

/// World bounds for exploration grid.
const WORLD_MIN: f32 = -50.0;
const WORLD_MAX: f32 = 50.0;

/// Treasury bonus when an NPC explores near a settlement.
const EXPLORATION_TREASURY_BONUS: f32 = 0.1;

pub fn compute_exploration(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % EXPLORATION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Track NPC movement: NPCs that are alive and moving reveal areas ---
    // In the full system, this would update explored_tiles on ExplorationState.
    // Without that field on WorldState, we express exploration as economic benefits:
    // NPCs near settlements boost that settlement's treasury (cartography intel).

    // NPCs at settlements: boost treasury via exploration intel.
    let sight_world = (BASE_SIGHT_RANGE as f32) * TILE_SIZE;
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        let npc_count = state.entities[range].iter()
            .filter(|e| e.alive && e.kind == EntityKind::Npc && e.team == WorldTeam::Friendly)
            .count();
        if npc_count > 0 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: EXPLORATION_TREASURY_BONUS * npc_count as f32,
            });
        }
    }

    // Frontier exploration: unaffiliated NPCs far from settlements.
    let unaffiliated = state.group_index.unaffiliated_entities();
    for entity in &state.entities[unaffiliated] {
        if entity.kind != EntityKind::Npc || !entity.alive || entity.team != WorldTeam::Friendly {
            continue;
        }
        let min_settlement_dist = state
            .settlements
            .iter()
            .map(|s| dist_sq(entity.pos, s.pos))
            .fold(f32::MAX, f32::min);

        // If NPC is far from settlements (>30 units), they're in frontier territory
        // Grant a small gold bonus representing exploration discoveries
        if min_settlement_dist > 900.0 {
            // 30^2
            out.push(WorldDelta::TransferGold {
                from_id: 0, // guild funds the expedition
                to_id: entity.id,
                amount: 0.5, // small per-tick frontier bonus
            });
        }
    }

    // --- Scouting visibility boost: explored areas near regions ---
    // Settlements with many nearby NPCs get fidelity upgrades
    // (representing better map knowledge of the area).
    for settlement in &state.settlements {
        let grid_id = match settlement.grid_id {
            Some(gid) => gid,
            None => continue,
        };

        let nearby_npc_count = state
            .entities
            .iter()
            .filter(|e| {
                e.kind == EntityKind::Npc
                    && e.alive
                    && e.team == WorldTeam::Friendly
                    && dist_sq(e.pos, settlement.pos) <= 400.0 // within 20 units
            })
            .count();

        // If 3+ NPCs are scouting near a settlement, escalate its grid fidelity
        if nearby_npc_count >= 3 {
            let current_fidelity = state
                .grid(grid_id)
                .map(|g| g.fidelity)
                .unwrap_or(Fidelity::Low);

            if matches!(current_fidelity, Fidelity::Low | Fidelity::Background) {
                out.push(WorldDelta::EscalateFidelity {
                    grid_id,
                    new_fidelity: Fidelity::Medium,
                });
            }
        }
    }
}

fn dist_sq(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}
