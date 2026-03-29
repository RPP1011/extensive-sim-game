#![allow(unused)]
//! Skill challenge system — every 7 ticks.
//!
//! Non-combat skill checks occur during active quests. Resolution is based on
//! NPC level, archetype class tags, and a deterministic roll. Outcomes produce
//! gold rewards (success) or damage (failure) via deltas.
//!
//! Ported from `crates/headless_campaign/src/systems/skill_challenges.rs`.
//!
//! NEEDS STATE: `active_quests: Vec<ActiveQuest>` on WorldState
//! NEEDS STATE: `ActiveQuest { id, quest_type, dispatched_party_id, threat_level, status }`
//! NEEDS STATE: `parties: Vec<Party>` on WorldState
//! NEEDS STATE: `Party { id, member_ids, status }`
//! NEEDS STATE: `skill_challenges: Vec<SkillChallenge>` on WorldState
//! NEEDS STATE: `SkillChallenge { id, skill_type, difficulty, quest_id, adventurer_id,
//!              succeeded, resolved }`
//! NEEDS STATE: `SkillType` enum on WorldState
//! NEEDS STATE: `archetype: String` on NpcData (for base skill computation)
//! NEEDS STATE: `morale: f32` on NpcData (mood bonus)
//! NEEDS DELTA: GrantXp { entity_id, amount }
//! NEEDS DELTA: UpdateMorale { entity_id, delta }
//! NEEDS DELTA: UpdateStress { entity_id, delta }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState, WorldTeam};

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

/// Guild entity ID sentinel.
const GUILD_ENTITY_ID: u32 = 0;

pub fn compute_skill_challenges(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CHALLENGE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without full quest/party state on WorldState, we approximate skill
    // challenges as NPC-level checks. Each alive friendly NPC on a grid
    // with hostiles may face a "skill challenge" — success heals, failure
    // damages.
    //
    // We use a deterministic approach based on entity level and tick:
    // higher-level NPCs have better skill scores.

    for entity in &state.entities {
        if entity.kind != EntityKind::Npc || !entity.alive || entity.team != WorldTeam::Friendly {
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
        // Base skill from level: 30 + level * 5
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
        let roll_seed = (entity.id as u64).wrapping_mul(2654435761) ^ state.tick;
        let roll = ((roll_seed % 2000) as f32) / 100.0; // [0, 20)

        let total = base_skill + roll;
        let succeeded = total > difficulty;

        let is_critical_success = base_skill > difficulty + 30.0;
        let is_critical_failure = base_skill < difficulty - 30.0 && !succeeded;

        if is_critical_success {
            // Critical success: gold bonus + heal
            out.push(WorldDelta::TransferGold {
                from_id: GUILD_ENTITY_ID,
                to_id: entity.id,
                amount: CRITICAL_SUCCESS_GOLD,
            });
            out.push(WorldDelta::Heal {
                target_id: entity.id,
                amount: 10.0,
                source_id: entity.id,
            });
        } else if is_critical_failure {
            // Critical failure: damage + gold loss
            out.push(WorldDelta::Damage {
                target_id: entity.id,
                amount: CRITICAL_FAILURE_DAMAGE,
                source_id: entity.id,
            });
            out.push(WorldDelta::TransferGold {
                from_id: entity.id,
                to_id: GUILD_ENTITY_ID,
                amount: CRITICAL_FAILURE_GOLD_PENALTY,
            });
        } else if succeeded {
            // Normal success: small gold bonus
            out.push(WorldDelta::TransferGold {
                from_id: GUILD_ENTITY_ID,
                to_id: entity.id,
                amount: SUCCESS_GOLD,
            });
        } else {
            // Normal failure: minor damage
            out.push(WorldDelta::Damage {
                target_id: entity.id,
                amount: FAILURE_DAMAGE,
                source_id: entity.id,
            });
        }
    }
}
