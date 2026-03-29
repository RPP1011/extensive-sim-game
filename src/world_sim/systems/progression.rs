#![allow(unused)]
//! Progression system — level-up from accumulated XP.
//!
//! This system does NOT grant XP. XP is granted by action systems.
//! It checks if XP has crossed the level threshold, applies stat gains
//! based on both global level AND class specialization.
//!
//! Class-based scaling: each class modifies which stats grow on level-up.
//! A Warrior gains more attack+armor, a Healer gains more HP, etc.
//!
//! Cadence: every 50 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, EntityField, WorldState, tag};

const PROGRESSION_INTERVAL: u64 = 50;
const XP_LEVEL_MULT: u32 = 100;
const MAX_LEVEL: u32 = 50;

// Base stat gains per level-up (before class modifiers).
const BASE_MAX_HP: f32 = 5.0;
const BASE_ATTACK: f32 = 0.5;
const BASE_ARMOR: f32 = 0.2;
const BASE_SPEED: f32 = 0.02;

// Class-specific bonus multipliers (added ON TOP of base).
// Each class hash maps to (hp_bonus, attack_bonus, armor_bonus, speed_bonus).
fn class_bonus(class_hash: u32) -> (f32, f32, f32, f32) {
    match class_hash {
        h if h == tag(b"Warrior")   => (5.0, 2.0, 1.5, 0.0),   // tanky + hard hitting
        h if h == tag(b"Guardian")  => (8.0, 0.5, 3.0, 0.0),   // very tanky
        h if h == tag(b"Ranger")    => (2.0, 1.5, 0.3, 0.1),   // mobile + damage
        h if h == tag(b"Healer")    => (10.0, 0.0, 0.5, 0.0),  // high HP to survive
        h if h == tag(b"Merchant")  => (3.0, 0.0, 0.3, 0.05),  // moderate survivability
        h if h == tag(b"Scholar")   => (2.0, 0.0, 0.2, 0.0),   // squishy
        h if h == tag(b"Rogue")     => (1.0, 1.0, 0.0, 0.15),  // fast + damage
        h if h == tag(b"Artisan")   => (3.0, 0.3, 0.5, 0.0),   // moderate
        h if h == tag(b"Diplomat")  => (3.0, 0.0, 0.3, 0.03),  // moderate
        h if h == tag(b"Commander") => (5.0, 1.0, 1.0, 0.0),   // leadership = tanky
        h if h == tag(b"Farmer")    => (4.0, 0.3, 0.3, 0.0),   // hardy
        h if h == tag(b"Miner")     => (6.0, 0.5, 1.0, 0.0),   // tough
        h if h == tag(b"Woodsman")  => (4.0, 0.8, 0.5, 0.05),  // balanced
        h if h == tag(b"Alchemist") => (2.0, 0.0, 0.2, 0.0),   // squishy
        h if h == tag(b"Herbalist") => (5.0, 0.0, 0.3, 0.0),   // healer-adjacent
        h if h == tag(b"Explorer")  => (3.0, 0.5, 0.3, 0.1),   // mobile
        h if h == tag(b"Mentor")    => (4.0, 0.0, 0.5, 0.0),   // moderate
        _ => (2.0, 0.3, 0.2, 0.0),                              // unknown class
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
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };
        if entity.level >= MAX_LEVEL { continue; }

        // Logarithmic XP curve: 200 × e^(level × 0.06)
        // L1:200, L5:270, L10:364, L20:664, L30:1210, L50:4017
        let threshold = (200.0 * (entity.level as f32 * 0.06).exp()) as u32;
        if npc.xp < threshold { continue; }

        // Compute class-weighted stat bonuses.
        // If NPC has multiple classes, average their bonuses.
        let (mut hp_bonus, mut atk_bonus, mut armor_bonus, mut speed_bonus) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let class_count = npc.classes.len().max(1) as f32;

        if npc.classes.is_empty() {
            // No class — base gains only.
        } else {
            for class in &npc.classes {
                let (hp, atk, arm, spd) = class_bonus(class.class_name_hash);
                hp_bonus += hp;
                atk_bonus += atk;
                armor_bonus += arm;
                speed_bonus += spd;
            }
            hp_bonus /= class_count;
            atk_bonus /= class_count;
            armor_bonus /= class_count;
            speed_bonus /= class_count;
        }

        // Apply: base + class bonus.
        let total_hp = BASE_MAX_HP + hp_bonus;
        let total_atk = BASE_ATTACK + atk_bonus;
        let total_armor = BASE_ARMOR + armor_bonus;
        let total_speed = BASE_SPEED + speed_bonus;

        // Increase MAX HP (not just current HP).
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::MaxHp,
            value: total_hp,
        });
        // Also heal by the same amount so the NPC benefits immediately.
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

        // Increment level.
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::Level,
            value: 1.0,
        });
    }
}
