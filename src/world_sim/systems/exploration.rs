#![allow(unused)]
//! Map exploration tracking system — every 3 ticks.
//!
//! Tracks NPC movement across the world to reveal regions and discover
//! landmarks. Explored areas near settlements boost treasury (representing
//! cartography bonuses). Milestone rewards are expressed as treasury updates
//! and fidelity changes.
//!
//! **Gold conservation:** Frontier exploration rewards are paid from the
//! nearest settlement treasury. If no settlement can afford it, no gold
//! is paid.
//!
//! Ported from `crates/headless_campaign/src/systems/exploration.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::fidelity::Fidelity;
use crate::world_sim::state::{ActionTags, EntityKind, WorldState, WorldTeam, tags};

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

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_exploration_for_settlement(state, settlement.id, &state.entities[range], out);
    }

    // Frontier exploration: unaffiliated NPCs far from settlements.
    let unaffiliated = state.group_index.unaffiliated_entities();
    for entity in &state.entities[unaffiliated] {
        if entity.kind != EntityKind::Npc || !entity.alive || entity.team != WorldTeam::Friendly {
            continue;
        }
        let nearest = state
            .settlements
            .iter()
            .min_by(|a, b| {
                let da = dist_sq(entity.pos, a.pos);
                let db = dist_sq(entity.pos, b.pos);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

        let min_settlement_dist = nearest.map(|s| dist_sq(entity.pos, s.pos)).unwrap_or(f32::MAX);

        if min_settlement_dist > 900.0 {
            // Pay frontier exploration stipend from nearest settlement treasury
            if let Some(s) = nearest {
                if s.treasury > 0.5 {
                    out.push(WorldDelta::TransferGold {
                        from_id: s.id,
                        to_id: entity.id,
                        amount: 0.5,
                    });
                }
            }

            // Behavior tags: frontier exploration.
            let mut action = ActionTags::empty();
            action.add(tags::EXPLORATION, 1.0);
            action.add(tags::NAVIGATION, 0.5);
            let action = crate::world_sim::action_context::with_context(&action, entity, state);
            out.push(WorldDelta::AddBehaviorTags { entity_id: entity.id, tags: action.tags, count: action.count });
        }
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_exploration_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % EXPLORATION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // NPCs at this settlement: boost treasury via exploration intel.
    let npc_count = entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc && e.team == WorldTeam::Friendly)
        .count();
    if npc_count > 0 {
        out.push(WorldDelta::UpdateTreasury {
            location_id: settlement_id,
            delta: EXPLORATION_TREASURY_BONUS * npc_count as f32,
        });
    }

    // Scouting visibility boost: nearby NPCs escalate grid fidelity.
    let grid_id = match settlement.grid_id {
        Some(gid) => gid,
        None => return,
    };

    let nearby_npc_count = state
        .entities
        .iter()
        .filter(|e| {
            e.kind == EntityKind::Npc
                && e.alive
                && e.team == WorldTeam::Friendly
                && dist_sq(e.pos, settlement.pos) <= 400.0
        })
        .count();

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

fn dist_sq(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}
