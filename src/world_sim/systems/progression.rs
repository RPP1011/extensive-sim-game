#![allow(unused)]
//! Progression system — entity level = total class levels.
//!
//! Entity level is derived from the sum of all class levels.
//! Stat bonuses are granted when class levels increase (detected by
//! comparing entity.level to the current class level sum).
//!
//! This system does NOT grant XP or drive leveling. Class leveling
//! happens in run_class_matching(). This system syncs entity.level
//! and applies stat gains from new class levels.
//!
//! Cadence: every 50 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, EntityField, WorldState, tag};

const PROGRESSION_INTERVAL: u64 = 50;

// Base stat gains per level-up (before class modifiers).
const BASE_MAX_HP: f32 = 5.0;
const BASE_ATTACK: f32 = 0.5;
const BASE_ARMOR: f32 = 0.2;
const BASE_SPEED: f32 = 0.02;

// Class-specific bonus multipliers (added ON TOP of base).
fn class_bonus(class_hash: u32) -> (f32, f32, f32, f32) {
    match class_hash {
        h if h == tag(b"Warrior")   => (5.0, 2.0, 1.5, 0.0),
        h if h == tag(b"Guardian")  => (8.0, 0.5, 3.0, 0.0),
        h if h == tag(b"Ranger")    => (2.0, 1.5, 0.3, 0.1),
        h if h == tag(b"Healer")    => (10.0, 0.0, 0.5, 0.0),
        h if h == tag(b"Merchant")  => (3.0, 0.0, 0.3, 0.05),
        h if h == tag(b"Scholar")   => (2.0, 0.0, 0.2, 0.0),
        h if h == tag(b"Rogue")     => (1.0, 1.0, 0.0, 0.15),
        h if h == tag(b"Artisan")   => (3.0, 0.3, 0.5, 0.0),
        h if h == tag(b"Diplomat")  => (3.0, 0.0, 0.3, 0.03),
        h if h == tag(b"Commander") => (5.0, 1.0, 1.0, 0.0),
        h if h == tag(b"Farmer")    => (4.0, 0.3, 0.3, 0.0),
        h if h == tag(b"Miner")     => (6.0, 0.5, 1.0, 0.0),
        h if h == tag(b"Woodsman")  => (4.0, 0.8, 0.5, 0.05),
        h if h == tag(b"Alchemist") => (2.0, 0.0, 0.2, 0.0),
        h if h == tag(b"Herbalist") => (5.0, 0.0, 0.3, 0.0),
        h if h == tag(b"Explorer")  => (3.0, 0.5, 0.3, 0.1),
        h if h == tag(b"Mentor")    => (4.0, 0.0, 0.5, 0.0),
        h if h == tag(b"Sentinel")  => (6.0, 0.5, 2.0, 0.0),
        h if h == tag(b"Survivor")  => (8.0, 0.3, 0.5, 0.05),
        h if h == tag(b"Warden")    => (5.0, 1.0, 2.0, 0.0),
        h if h == tag(b"Veteran")   => (4.0, 2.0, 1.0, 0.0),
        h if h == tag(b"Stalwart")  => (10.0, 0.0, 1.5, 0.0),
        h if h == tag(b"Bard")      => (3.0, 0.0, 0.2, 0.1),   // mobile, charismatic
        h if h == tag(b"Mariner")   => (5.0, 0.5, 0.5, 0.1),   // hardy sea dog
        h if h == tag(b"Sea Captain") => (5.0, 1.0, 1.0, 0.05), // tough commander
        h if h == tag(b"Delver")    => (4.0, 0.8, 1.0, 0.0),   // dungeon survivor
        h if h == tag(b"Dungeon Master") => (3.0, 1.5, 1.5, 0.0), // tactical underground
        h if h == tag(b"Oathkeeper")    => (8.0, 0.5, 2.0, 0.0),  // stalwart defender
        h if h == tag(b"Betrayer")      => (2.0, 1.5, 0.0, 0.2),  // fast, deadly, fragile
        _ => (2.0, 0.3, 0.2, 0.0),
    }
}

pub fn compute_progression(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % PROGRESSION_INTERVAL != 0 { return; }

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
    if state.tick % PROGRESSION_INTERVAL != 0 { return; }

    for entity in entities {
        if !entity.alive { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        // Entity level = total class levels (hard-derived).
        let class_level_sum: u32 = npc.classes.iter().map(|c| c.level as u32).sum();
        if class_level_sum == 0 { continue; }

        // How many NEW levels since last sync?
        let new_levels = if class_level_sum > entity.level {
            class_level_sum - entity.level
        } else {
            continue; // already synced or went down (consolidation)
        };

        // Compute class-weighted stat bonuses per level.
        let (mut hp_bonus, mut atk_bonus, mut armor_bonus, mut speed_bonus) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

        if npc.classes.is_empty() {
            // No class — base gains only (shouldn't happen since class_level_sum > 0).
        } else {
            // Weight by class level: higher-level classes contribute more to stat gains.
            let total_weight: f32 = npc.classes.iter().map(|c| c.level as f32).sum();
            for class in &npc.classes {
                let (hp, atk, arm, spd) = class_bonus(class.class_name_hash);
                let w = class.level as f32 / total_weight.max(1.0);
                hp_bonus += hp * w;
                atk_bonus += atk * w;
                armor_bonus += arm * w;
                speed_bonus += spd * w;
            }
        }

        let total_hp = (BASE_MAX_HP + hp_bonus) * new_levels as f32;
        let total_atk = (BASE_ATTACK + atk_bonus) * new_levels as f32;
        let total_armor = (BASE_ARMOR + armor_bonus) * new_levels as f32;
        let total_speed = (BASE_SPEED + speed_bonus) * new_levels as f32;

        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::MaxHp,
            value: total_hp,
        });
        out.push(WorldDelta::Heal {
            target_id: entity.id,
            amount: total_hp,
            source_id: entity.id,
        });
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::AttackDamage,
            value: total_atk,
        });
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::Armor,
            value: total_armor,
        });
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::MoveSpeed,
            value: total_speed,
        });

        // Sync entity level to class level sum.
        let level_delta = class_level_sum as f32 - entity.level as f32;
        if level_delta.abs() > 0.5 {
            out.push(WorldDelta::UpdateEntityField {
                entity_id: entity.id,
                field: EntityField::Level,
                value: level_delta,
            });
        }
    }
}
