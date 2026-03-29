#![allow(unused)]
//! Progression system — level-up from accumulated XP.
//!
//! This system does NOT grant XP. XP is granted by action systems:
//! - battles.rs: combat XP on monster kill
//! - food.rs: labor XP on production
//! - mentorship.rs: teaching/learning XP
//! - trade_goods.rs: merchant XP on trade completion
//! - wound_persistence.rs: resilience XP on recovery
//!
//! This system only checks if XP has crossed the level threshold
//! and applies permanent stat gains.
//!
//! Cadence: every 50 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, EntityField, WorldState};

const PROGRESSION_INTERVAL: u64 = 50;

/// XP threshold: level N requires N*N*100 total XP to reach level N+1.
const XP_LEVEL_MULT: u32 = 100;
const MAX_LEVEL: u32 = 50;

/// Stats gained per level-up.
const HP_PER_LEVEL: f32 = 8.0;
const ATTACK_PER_LEVEL: f32 = 0.8;
const ARMOR_PER_LEVEL: f32 = 0.3;
const SPEED_PER_LEVEL: f32 = 0.05;

pub fn compute_progression(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % PROGRESSION_INTERVAL != 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_progression_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

pub fn compute_progression_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % PROGRESSION_INTERVAL != 0 {
        return;
    }

    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        if entity.level >= MAX_LEVEL { continue; }

        let threshold = entity.level * entity.level * XP_LEVEL_MULT;
        if npc.xp < threshold { continue; }

        // Level-up: permanent stat increases.
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::Hp,
            value: HP_PER_LEVEL,
        });
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::AttackDamage,
            value: ATTACK_PER_LEVEL,
        });
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::Armor,
            value: ARMOR_PER_LEVEL,
        });
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::MoveSpeed,
            value: SPEED_PER_LEVEL,
        });

        // Note: actual level increment requires a delta for entity.level.
        // Currently EntityField doesn't include Level. The XP reset also
        // needs a dedicated mechanism. For now, the stat gains accumulate
        // and the XP threshold check triggers every progression tick
        // until XP is consumed.
    }
}
