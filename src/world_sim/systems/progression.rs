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
use crate::world_sim::registry::Registry;
use crate::world_sim::state::{Entity, EntityField, WorldState};

const PROGRESSION_INTERVAL: u64 = 50;

// Base stat gains per level-up (before class modifiers).
const BASE_MAX_HP: f32 = 5.0;
const BASE_ATTACK: f32 = 0.5;
const BASE_ARMOR: f32 = 0.2;
const BASE_SPEED: f32 = 0.02;

/// Look up per-level stat bonuses for a class from the registry.
/// Falls back to default if no registry or class not found.
/// Returns (hp, attack, armor, speed) per level — includes base gains.
fn class_per_level(registry: Option<&Registry>, class_hash: u32) -> (f32, f32, f32, f32) {
    if let Some(reg) = registry {
        if let Some(def) = reg.classes.get(&class_hash) {
            return (
                def.per_level.hp,
                def.per_level.attack,
                def.per_level.armor,
                def.per_level.speed,
            );
        }
    }
    // Fallback: BASE + default class_bonus.
    (BASE_MAX_HP + 2.0, BASE_ATTACK + 0.3, BASE_ARMOR + 0.2, BASE_SPEED + 0.0)
}

pub fn compute_progression(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % PROGRESSION_INTERVAL != 0 { return; }

    let registry = state.registry.as_deref();

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_progression_for_settlement(state, settlement.id, &state.entities[range], registry, out);
    }
}

pub fn compute_progression_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    registry: Option<&Registry>,
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

        // Compute class-weighted stat gains per level.
        let (mut hp_per_level, mut atk_per_level, mut armor_per_level, mut speed_per_level) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

        if npc.classes.is_empty() {
            // No class — base gains only (shouldn't happen since class_level_sum > 0).
            hp_per_level = BASE_MAX_HP;
            atk_per_level = BASE_ATTACK;
            armor_per_level = BASE_ARMOR;
            speed_per_level = BASE_SPEED;
        } else {
            // Weight by class level: higher-level classes contribute more to stat gains.
            let total_weight: f32 = npc.classes.iter().map(|c| c.level as f32).sum();
            for class in &npc.classes {
                let (hp, atk, arm, spd) = class_per_level(registry, class.class_name_hash);
                let w = class.level as f32 / total_weight.max(1.0);
                hp_per_level += hp * w;
                atk_per_level += atk * w;
                armor_per_level += arm * w;
                speed_per_level += spd * w;
            }
        }

        let total_hp = hp_per_level * new_levels as f32;
        let total_atk = atk_per_level * new_levels as f32;
        let total_armor = armor_per_level * new_levels as f32;
        let total_speed = speed_per_level * new_levels as f32;

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
