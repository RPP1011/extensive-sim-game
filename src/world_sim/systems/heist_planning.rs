//! Heist planning system — every 10 ticks, after tick 1500.
//!
//! Multi-phase heist preparation with crew skill factors. Each heist
//! progresses through phases (Planning, Scouting, Infiltration, Execution,
//! Escape). Success yields gold; failure results in damage to crew and
//! gold penalties.
//!
//! **Gold conservation:** Heist gold is stolen from the target settlement
//! treasury (UpdateTreasury drain + TransferGold). No gold is minted.
//!
//! Ported from `crates/headless_campaign/src/systems/heist_planning.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{WorldState, WorldTeam};
use crate::world_sim::state::entity_hash;

/// How often the heist system ticks (in ticks).
const HEIST_INTERVAL: u64 = 10;

/// Minimum tick before heists can occur.
const MIN_HEIST_TICK: u64 = 1500;

/// Ticks each phase lasts before advancing.
const PHASE_DURATION: u64 = 17;

/// Reward scaling: base gold for successful heist.
const BASE_HEIST_REWARD: f32 = 80.0;

/// Gold penalty on failed heist.
const FAILURE_GOLD_PENALTY: f32 = 30.0;

/// Damage to crew members on heist failure.
const FAILURE_CREW_DAMAGE: f32 = 25.0;

pub fn compute_heist_planning(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % HEIST_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.tick < MIN_HEIST_TICK {
        return;
    }

    // --- Heist approximation via NPC covert operations ---
    // Without explicit heist state, we model heists as "rogue NPC operations":
    // friendly NPCs on grids WITHOUT hostiles but near hostile-faction settlements
    // are performing covert ops (scouting, infiltrating, executing heists).

    // Identify NPCs near settlements they don't belong to (infiltrators)
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        for entity in &state.entities[range] {
            if !entity.alive || entity.team != WorldTeam::Friendly {
                continue;
            }

            // Skip NPCs on combat grids (they're fighting, not heisting)
            if let Some(gid) = entity.grid_id {
                if state
                    .grid(gid)
                    .map(|g| g.fidelity == crate::world_sim::fidelity::Fidelity::High)
                    .unwrap_or(false)
                {
                    continue;
                }
            }

            let npc = match &entity.npc {
                Some(n) => n,
                None => continue,
            };

            // Check if NPC is near a settlement that isn't their home (infiltration)
            let home_id = npc.home_settlement_id;
            for target_settlement in &state.settlements {
                if Some(target_settlement.id) == home_id {
                    continue; // Skip home settlement
                }

                let dist = dist_sq(entity.pos, target_settlement.pos);
                if dist > 400.0 {
                    // 20 units
                    continue; // Too far
                }

                // NPC is infiltrating a foreign settlement
                // Success chance based on level (higher level = better heist skills)
                let skill_factor = (entity.level as f32 * 0.05 + 0.5).min(1.0);

                // Deterministic outcome based on entity id + tick
                let roll = (entity_hash(entity.id, state.tick, 0x4E15) % 100) as f32 / 100.0;

                if roll < skill_factor * 0.1 && target_settlement.treasury > -100.0 {
                    // Heist success: steal gold from settlement
                    let reward = BASE_HEIST_REWARD * skill_factor;
                    out.push(WorldDelta::UpdateTreasury {
                        settlement_id: target_settlement.id,
                        delta: -reward,
                    });
                    out.push(WorldDelta::TransferGold {
                        from_entity: target_settlement.id,
                        to_entity: entity.id,
                        amount: reward,
                    });
                } else if roll > 0.95 {
                    // Heist failure: NPC takes damage (caught by guards)
                    out.push(WorldDelta::Damage {
                        target_id: entity.id,
                        amount: FAILURE_CREW_DAMAGE,
                        source_id: entity.id,
                    });
                    // Gold penalty
                    out.push(WorldDelta::TransferGold {
                        from_entity: entity.id,
                        to_entity: target_settlement.id,
                        amount: FAILURE_GOLD_PENALTY,
                    });
                }
                // Most of the time (85-95%) the NPC is still in planning/scouting phase
                // and no delta is emitted (preparation continues silently).

                break; // Only one infiltration target per NPC per tick
            }
        }
    }
}

fn dist_sq(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}
