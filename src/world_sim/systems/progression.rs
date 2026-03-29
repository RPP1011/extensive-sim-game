#![allow(unused)]
//! Progression system — ported from headless_campaign.
//!
//! Grants XP and triggers level-ups for NPC entities based on activity.
//! In the original system, progression fired on quest-completion milestones.
//! In the world sim, we trigger on entity-level activity: combat participation,
//! mission completion (approximated by surviving grid encounters), and
//! mentorship XP gains.
//!
//! Original: `crates/headless_campaign/src/systems/progression.rs`

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, WorldState};

// NEEDS STATE: xp: u32 on NpcData (or Entity)
// NEEDS STATE: completed_quests: Vec<QuestRecord> on WorldState (or per-entity quest history)
// NEEDS STATE: progression_history: Vec<ProgressionEvent> on WorldState
// NEEDS STATE: unlocks: Vec<UnlockInstance> on WorldState
// NEEDS DELTA: GrantXp { entity_id: u32, amount: u32, source: String }
// NEEDS DELTA: LevelUp { entity_id: u32, new_level: u32 }
// NEEDS DELTA: GrantUnlock { unlock_id: u32, category: String, name: String }

/// How often progression checks run (in ticks).
const PROGRESSION_INTERVAL: u64 = 50;

/// XP granted per progression tick for NPCs on hostile grids (combat XP).
const COMBAT_XP_PER_TICK: f32 = 2.0;

/// XP granted per progression tick for NPCs on non-hostile grids (mission XP).
const MISSION_XP_PER_TICK: f32 = 0.5;

/// XP threshold multiplier: threshold = level * level * XP_MULT.
const XP_LEVEL_MULT: u32 = 100;

/// HP gain per level-up.
const LEVEL_HP_GAIN: f32 = 10.0;
/// Attack gain per level-up.
const LEVEL_ATTACK_GAIN: f32 = 1.0;
/// Armor gain per level-up.
const LEVEL_ARMOR_GAIN: f32 = 0.5;

/// Compute progression deltas for all NPC entities.
///
/// Awards XP based on activity and emits level-up deltas when thresholds
/// are crossed. Level-ups are expressed as stat increases via Heal (HP)
/// and ApplyStatus (permanent stat buffs).
pub fn compute_progression(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % PROGRESSION_INTERVAL != 0 {
        return;
    }

    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Determine XP gain based on current activity.
        let on_hostile_grid = entity
            .grid_id
            .and_then(|gid| state.grid(gid))
            .map(|g| g.has_hostiles(state))
            .unwrap_or(false);

        let on_grid = entity.grid_id.is_some();

        let xp_gain = if on_hostile_grid {
            COMBAT_XP_PER_TICK * (1.0 + entity.level as f32 * 0.1)
        } else if on_grid {
            MISSION_XP_PER_TICK
        } else {
            0.0 // idle NPCs don't gain XP passively
        };

        if xp_gain <= 0.0 {
            continue;
        }

        // Without a mutable XP counter, we can't directly track accumulated XP
        // in the compute phase. Instead, we use a deterministic threshold check
        // based on the entity's current level and tick count.
        //
        // Approximate: if this NPC has been on hostile grids long enough at its
        // current level, emit a level-up. The heuristic uses tick as a proxy
        // for total XP accumulated.
        //
        // Real implementation needs GrantXp delta and xp field on NpcData.
        // For now, we compute an approximate level-up condition.

        // Level-up check: every (level * level * XP_MULT / COMBAT_XP_PER_TICK)
        // ticks of combat, the NPC levels up. We use a simpler cadence:
        // NPCs level up roughly every 500 * level ticks of activity.
        let level_up_cadence = 500 * entity.level.max(1) as u64;
        let effective_tick = state.tick / PROGRESSION_INTERVAL;

        // Use entity ID to stagger level-ups across different NPCs.
        let stagger = entity.id as u64 % level_up_cadence;

        if effective_tick > 0
            && (effective_tick + stagger) % level_up_cadence == 0
            && on_hostile_grid
        {
            // Emit level-up as stat increases.
            let new_level = entity.level + 1;

            // HP increase: heal to represent increased max_hp.
            out.push(WorldDelta::Heal {
                target_id: entity.id,
                amount: LEVEL_HP_GAIN,
                source_id: entity.id,
            });

            // Attack increase: permanent buff.
            out.push(WorldDelta::ApplyStatus {
                target_id: entity.id,
                status: crate::world_sim::state::StatusEffect {
                    kind: crate::world_sim::state::StatusEffectKind::Buff {
                        stat: "attack".into(),
                        factor: 1.0 + LEVEL_ATTACK_GAIN / entity.attack_damage.max(1.0),
                    },
                    source_id: entity.id,
                    remaining_ms: u32::MAX, // permanent
                },
            });

            // Armor increase: permanent buff.
            out.push(WorldDelta::ApplyStatus {
                target_id: entity.id,
                status: crate::world_sim::state::StatusEffect {
                    kind: crate::world_sim::state::StatusEffectKind::Buff {
                        stat: "armor".into(),
                        factor: 1.0 + LEVEL_ARMOR_GAIN / entity.armor.max(1.0),
                    },
                    source_id: entity.id,
                    remaining_ms: u32::MAX, // permanent
                },
            });
        }
    }
}
