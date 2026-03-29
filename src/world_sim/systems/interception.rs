#![allow(unused)]
//! Party interception — fires every tick.
//!
//! Hostile entities within interception range of friendly NPCs trigger
//! combat (escalate grid fidelity). Maps to EscalateFidelity and Damage.
//!
//! Original: `crates/headless_campaign/src/systems/interception.rs`

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::fidelity::Fidelity;
use crate::world_sim::state::{EntityKind, WorldState, WorldTeam};

/// Distance threshold for interception (world units).
const INTERCEPTION_RANGE: f32 = 5.0;

/// Interception check cadence (every tick for responsiveness).
const INTERCEPTION_INTERVAL: u64 = 1;

pub fn compute_interception(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % INTERCEPTION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Check proximity between friendly and hostile entities.
    // When they're within range, escalate the grid fidelity to High
    // (triggering detailed combat simulation).

    let friendlies: Vec<(u32, (f32, f32), Option<u32>)> = state
        .entities
        .iter()
        .filter(|e| e.alive && e.team == WorldTeam::Friendly)
        .map(|e| (e.id, e.pos, e.grid_id))
        .collect();

    let hostiles: Vec<(u32, (f32, f32), Option<u32>)> = state
        .entities
        .iter()
        .filter(|e| e.alive && e.team == WorldTeam::Hostile)
        .map(|e| (e.id, e.pos, e.grid_id))
        .collect();

    let mut escalated_grids = std::collections::HashSet::new();

    for &(f_id, f_pos, f_grid) in &friendlies {
        for &(h_id, h_pos, _h_grid) in &hostiles {
            let dx = f_pos.0 - h_pos.0;
            let dy = f_pos.1 - h_pos.1;
            let dist_sq = dx * dx + dy * dy;

            if dist_sq < INTERCEPTION_RANGE * INTERCEPTION_RANGE {
                // Escalate the grid containing the friendly entity
                if let Some(grid_id) = f_grid {
                    if escalated_grids.insert(grid_id) {
                        out.push(WorldDelta::EscalateFidelity {
                            grid_id,
                            new_fidelity: Fidelity::High,
                        });
                    }
                }
            }
        }
    }
}
