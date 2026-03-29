#![allow(unused)]
//! Escalation protocol — every 13 ticks.
//!
//! When hostile entities are killed near settlements, the local threat
//! level rises, spawning stronger monsters via the fidelity system.
//! If the player stops engaging, threat de-escalates over time.
//!
//! Original: `crates/headless_campaign/src/systems/escalation_protocol.rs`
//!
//! NEEDS STATE: `escalation_level: u32` on RegionState
//! NEEDS STATE: `patrol_losses: u32` on RegionState or faction data
//! NEEDS STATE: `last_patrol_loss_tick: u64` on RegionState or faction data
//! NEEDS DELTA: ModifyEscalation { region_id, delta }
//! NEEDS DELTA: SpawnEliteSquad { region_id, power }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::fidelity::Fidelity;
use crate::world_sim::state::{EntityKind, WorldState, WorldTeam};

/// How often the escalation system evaluates (in ticks).
const ESCALATION_INTERVAL: u64 = 13;

/// Threat threshold above which escalation is considered active.
const ESCALATION_THREAT_THRESHOLD: f32 = 60.0;

/// Threat increase per escalation tick when hostiles are dying.
const ESCALATION_RATE: f32 = 2.0;

/// De-escalation: threat decreases per tick when no recent kills.
/// Only applies when threat is above the threshold.
const DE_ESCALATION_RATE: f32 = 0.5;

/// Deterministic hash for pseudo-random decisions from immutable state.
fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_escalation_protocol(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % ESCALATION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_escalation_protocol_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
///
/// Evaluates escalation for this settlement based on nearby region threat levels.
pub fn compute_escalation_protocol_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    _entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % ESCALATION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.regions.is_empty() {
        return;
    }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // Count dead/alive hostile entities globally (proxy for recent kills).
    let dead_hostiles = state
        .entities
        .iter()
        .filter(|e| !e.alive && e.team == WorldTeam::Hostile && e.kind == EntityKind::Monster)
        .count();

    let alive_hostiles_nearby = state
        .entities
        .iter()
        .filter(|e| e.alive && e.team == WorldTeam::Hostile && e.kind == EntityKind::Monster)
        .count();

    for (ri, region) in state.regions.iter().enumerate() {
        if region.threat_level <= ESCALATION_THREAT_THRESHOLD {
            continue;
        }

        let dx = settlement.pos.0 - (ri as f32 * 20.0);
        let dy = settlement.pos.1 - (ri as f32 * 15.0);
        if dx * dx + dy * dy >= 400.0 {
            continue;
        }

        if dead_hostiles > alive_hostiles_nearby && alive_hostiles_nearby > 0 {
            // Player is winning — escalate grid fidelity.
            if let Some(grid_id) = settlement.grid_id {
                if let Some(grid) = state.grid(grid_id) {
                    if grid.fidelity != Fidelity::High {
                        out.push(WorldDelta::EscalateFidelity {
                            grid_id,
                            new_fidelity: Fidelity::High,
                        });
                    }
                }
            }

            // War exhaustion treasury drain.
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement_id,
                delta: -ESCALATION_RATE,
            });
        } else if dead_hostiles == 0 {
            // De-escalate.
            if let Some(grid_id) = settlement.grid_id {
                if let Some(grid) = state.grid(grid_id) {
                    if grid.fidelity == Fidelity::High {
                        out.push(WorldDelta::EscalateFidelity {
                            grid_id,
                            new_fidelity: Fidelity::Medium,
                        });
                    }
                }
            }
        }
    }
}
