#![allow(unused)]
//! Skill challenge system — every 7 ticks.
//!
//! Non-combat skill checks occur during active quests. Resolution is based on
//! NPC level, archetype class tags, and a deterministic roll. Outcomes produce
//! gold rewards (success) or damage (failure) via deltas.
//!
//! **Gold conservation:** Rewards are paid from the NPC's home settlement
//! treasury. If the settlement cannot afford the reward, no gold is paid.
//!
//! Ported from `crates/headless_campaign/src/systems/skill_challenges.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState, WorldTeam};
use crate::world_sim::state::entity_hash;

/// How often to tick skill challenges (in ticks).
const CHALLENGE_INTERVAL: u64 = 7;

/// Gold reward for a normal success.
const SUCCESS_GOLD: f32 = 5.0;

/// Gold reward for a critical success.
const CRITICAL_SUCCESS_GOLD: f32 = 10.0;

/// Gold penalty for a critical failure.
const CRITICAL_FAILURE_GOLD_PENALTY: f32 = 5.0;

/// Damage dealt to entity on failure.
const FAILURE_DAMAGE: f32 = 5.0;

/// Damage dealt to entity on critical failure.
const CRITICAL_FAILURE_DAMAGE: f32 = 15.0;

pub fn compute_skill_challenges(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CHALLENGE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_skill_challenges_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_skill_challenges_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % CHALLENGE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    for entity in entities {
        if !entity.alive || entity.team != WorldTeam::Friendly {
            continue;
        }

        // Only NPCs on grids (engaged in encounters) face challenges
        let grid_id = match entity.grid_id {
            Some(gid) => gid,
            None => continue,
        };

        // Check if the grid has hostiles (challenge context)
        let grid_has_hostiles = state
            .grid(grid_id)
            .map(|g| g.fidelity == crate::world_sim::fidelity::Fidelity::High)
            .unwrap_or(false);

        if !grid_has_hostiles {
            continue;
        }

        // Deterministic skill check using entity id and tick
        let base_skill = 30.0 + entity.level as f32 * 5.0;

        // Difficulty scales with number of hostiles on grid
        let hostile_count = state
            .grid(grid_id)
            .map(|g| {
                g.entity_ids
                    .iter()
                    .filter(|&&eid| {
                        state
                            .entity(eid)
                            .map(|e| e.kind == EntityKind::Monster && e.alive)
                            .unwrap_or(false)
                    })
                    .count()
            })
            .unwrap_or(0);

        let difficulty = 30.0 + hostile_count as f32 * 10.0;

        // Deterministic roll based on entity id and tick (no RNG on immutable state)
        let roll = (entity_hash(entity.id, state.tick, 0x5411) % 2000) as f32 / 100.0; // [0, 20)

        let total = base_skill + roll;
        let succeeded = total > difficulty;

        let is_critical_success = base_skill > difficulty + 30.0;
        let is_critical_failure = base_skill < difficulty - 30.0 && !succeeded;

        if is_critical_success {
            // Paid from settlement treasury
            if settlement.treasury > CRITICAL_SUCCESS_GOLD {
                out.push(WorldDelta::TransferGold {
                    from_id: settlement_id,
                    to_id: entity.id,
                    amount: CRITICAL_SUCCESS_GOLD,
                });
            }
            out.push(WorldDelta::Heal {
                target_id: entity.id,
                amount: 10.0,
                source_id: entity.id,
            });
        } else if is_critical_failure {
            out.push(WorldDelta::Damage {
                target_id: entity.id,
                amount: CRITICAL_FAILURE_DAMAGE,
                source_id: entity.id,
            });
            // Gold penalty goes to settlement treasury
            out.push(WorldDelta::TransferGold {
                from_id: entity.id,
                to_id: settlement_id,
                amount: CRITICAL_FAILURE_GOLD_PENALTY,
            });
        } else if succeeded {
            // Paid from settlement treasury
            if settlement.treasury > SUCCESS_GOLD {
                out.push(WorldDelta::TransferGold {
                    from_id: settlement_id,
                    to_id: entity.id,
                    amount: SUCCESS_GOLD,
                });
            }
        } else {
            out.push(WorldDelta::Damage {
                target_id: entity.id,
                amount: FAILURE_DAMAGE,
                source_id: entity.id,
            });
        }
    }
}
