//! VAE slot extractors — convert parsed game content into flat f32 vectors
//! suitable as decoder targets for the grammar-guided VAE.
//!
//! Each extractor maps a structured game object (AbilityDef, ClassDef, etc.)
//! into a fixed-length slot vector where every dimension has a defined meaning.

use tactical_sim::effects::defs::{AbilityDef, AbilityTargeting, PassiveDef};
use tactical_sim::effects::effect_enum::Effect;
use tactical_sim::effects::types::*;

use super::class_dsl::*;
use super::state::{EquipmentSlot, InventoryItem, QuestRequest, QuestType};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum effects per ability (pad/truncate to this).
pub const MAX_EFFECTS: usize = 4;
/// Dims per effect slot.
pub const EFFECT_SLOT_DIM: usize = 25;
/// Maximum scaling rules per class.
const MAX_SCALING_RULES: usize = 3;
/// Maximum abilities per class.
const MAX_CLASS_ABILITIES: usize = 5;
/// Maximum requirements per class.
const MAX_CLASS_REQUIREMENTS: usize = 4;

// ---------------------------------------------------------------------------
// Ability slots
// ---------------------------------------------------------------------------

/// Total dimension of an ability slot vector.
pub const ABILITY_SLOT_DIM: usize = 3    // output_type (active/passive/class)
    + 8                                    // targeting
    + 4                                    // range, cooldown, cast, cost
    + 5                                    // hint
    + 7                                    // delivery type
    + 6                                    // delivery params
    + 4                                    // has_charges, has_toggle, has_recast, unstoppable
    + 5                                    // charge/toggle/recast params
    + MAX_EFFECTS * EFFECT_SLOT_DIM;       // 4 × 25 = 100
// Total: 3+8+4+5+7+6+4+5+100 = 142

/// Convert an AbilityDef into a flat slot vector.
pub fn ability_to_slots(def: &AbilityDef) -> Vec<f32> {
    let mut v = vec![0.0f32; ABILITY_SLOT_DIM];
    let mut o = 0;

    // output_type: active
    v[o] = 1.0;
    o += 3;

    // targeting (8)
    let ti = targeting_index(&def.targeting);
    if ti < 8 { v[o + ti] = 1.0; }
    o += 8;

    // range, cooldown, cast, cost (4)
    v[o] = def.range / 10.0;
    v[o + 1] = def.cooldown_ms as f32 / 30000.0;
    v[o + 2] = def.cast_time_ms as f32 / 1500.0;
    v[o + 3] = def.resource_cost as f32 / 100.0;
    o += 4;

    // hint (5): damage, crowd_control, defense, utility, heal
    let hi = hint_index(&def.ai_hint);
    if hi < 5 { v[o + hi] = 1.0; }
    o += 5;

    // delivery type (7)
    if let Some(ref d) = def.delivery {
        let di = delivery_index(d);
        if di < 7 { v[o + di] = 1.0; }
    }
    o += 7;

    // delivery params (6): speed, width, duration, bounces, range, falloff
    if let Some(ref d) = def.delivery {
        encode_delivery_params(d, &mut v[o..o + 6]);
    }
    o += 6;

    // has_charges, has_toggle, has_recast, unstoppable (4)
    v[o] = if def.max_charges > 0 { 1.0 } else { 0.0 };
    v[o + 1] = if def.is_toggle { 1.0 } else { 0.0 };
    v[o + 2] = if def.recast_count > 0 { 1.0 } else { 0.0 };
    v[o + 3] = if def.unstoppable { 1.0 } else { 0.0 };
    o += 4;

    // charge/toggle/recast params (5)
    v[o] = def.max_charges as f32 / 5.0;
    v[o + 1] = def.charge_recharge_ms as f32 / 35000.0;
    v[o + 2] = def.toggle_cost_per_sec / 20.0;
    v[o + 3] = def.recast_count as f32 / 4.0;
    v[o + 4] = def.recast_window_ms as f32 / 18000.0;
    o += 5;

    // Effects (MAX_EFFECTS × EFFECT_SLOT_DIM)
    for (i, ce) in def.effects.iter().take(MAX_EFFECTS).enumerate() {
        encode_effect(ce, &mut v[o + i * EFFECT_SLOT_DIM..o + (i + 1) * EFFECT_SLOT_DIM]);
    }

    v
}

/// Convert a PassiveDef into a flat slot vector (same dim as ability).
pub fn passive_to_slots(def: &PassiveDef) -> Vec<f32> {
    let mut v = vec![0.0f32; ABILITY_SLOT_DIM];
    let mut o = 0;

    // output_type: passive
    v[o + 1] = 1.0;
    o += 3;

    // targeting — passives don't have explicit targeting, skip
    o += 8;

    // range, cooldown, cast=0, cost=0
    v[o] = def.range / 10.0;
    v[o + 1] = def.cooldown_ms as f32 / 30000.0;
    o += 4;

    // hint — skip for passives
    o += 5;

    // delivery — none for passives
    o += 7;

    // delivery params — zeros
    o += 6;

    // flags — all zero for passives
    o += 4;

    // charge/toggle/recast params — zeros
    o += 5;

    // Effects
    for (i, ce) in def.effects.iter().take(MAX_EFFECTS).enumerate() {
        encode_effect(ce, &mut v[o + i * EFFECT_SLOT_DIM..o + (i + 1) * EFFECT_SLOT_DIM]);
    }

    v
}

// ---------------------------------------------------------------------------
// Class slots
// ---------------------------------------------------------------------------

/// Total dimension of a class slot vector.
pub const CLASS_SLOT_DIM: usize = 5   // stat_growth
    + 16                                // tags multi-hot
    + 11                                // scaling source one-hot
    + MAX_SCALING_RULES * 8             // 3 × 8 = 24
    + MAX_CLASS_ABILITIES * 2           // 5 × 2 = 10
    + MAX_CLASS_REQUIREMENTS * 2        // 4 × 2 = 8
    + 1;                                // consolidates_at
// Total: 5+16+11+24+10+8+1 = 75

/// Convert a ClassDef into a flat slot vector.
pub fn class_to_slots(def: &ClassDef) -> Vec<f32> {
    let mut v = vec![0.0f32; CLASS_SLOT_DIM];
    let mut o = 0;

    // stat_growth (5)
    v[o] = def.stat_growth.attack / 5.0;
    v[o + 1] = def.stat_growth.defense / 5.0;
    v[o + 2] = def.stat_growth.speed / 5.0;
    v[o + 3] = def.stat_growth.max_hp / 5.0;
    v[o + 4] = def.stat_growth.ability_power / 5.0;
    o += 5;

    // tags multi-hot (16)
    for tag in &def.tags {
        let idx = class_tag_index(tag);
        if idx < 16 { v[o + idx] = 1.0; }
    }
    o += 16;

    // scaling source (11) — use first scaling block's source
    if let Some(sb) = def.scalings.first() {
        let idx = scaling_source_index(&sb.source);
        if idx < 11 { v[o + idx] = 1.0; }
    }
    o += 11;

    // scaling rules (3 × 8)
    if let Some(sb) = def.scalings.first() {
        for (i, rule) in sb.rules.iter().take(MAX_SCALING_RULES).enumerate() {
            encode_scaling_rule(rule, &mut v[o + i * 8..o + (i + 1) * 8]);
        }
    }
    o += MAX_SCALING_RULES * 8;

    // abilities (5 × 2): level normalized, hint index
    for (i, unlock) in def.abilities.iter().take(MAX_CLASS_ABILITIES).enumerate() {
        v[o + i * 2] = unlock.level as f32 / 40.0;
        v[o + i * 2 + 1] = 1.0; // ability present flag
    }
    o += MAX_CLASS_ABILITIES * 2;

    // requirements (4 × 2): type index, threshold
    for (i, req) in def.requirements.iter().take(MAX_CLASS_REQUIREMENTS).enumerate() {
        let (ri, threshold) = requirement_to_slot(req);
        v[o + i * 2] = ri;
        v[o + i * 2 + 1] = threshold;
    }
    o += MAX_CLASS_REQUIREMENTS * 2;

    // consolidates_at
    v[o] = def.consolidates_at.map(|c| c as f32 / 20.0).unwrap_or(0.0);

    v
}

// ---------------------------------------------------------------------------
// Item slots
// ---------------------------------------------------------------------------

pub const ITEM_SLOT_DIM: usize = 10;

/// Convert an InventoryItem into a flat slot vector.
pub fn item_to_slots(item: &InventoryItem) -> Vec<f32> {
    let mut v = vec![0.0f32; ITEM_SLOT_DIM];

    // slot one-hot (5)
    let si = match item.slot {
        EquipmentSlot::Weapon => 0,
        EquipmentSlot::Offhand => 1,
        EquipmentSlot::Chest => 2,
        EquipmentSlot::Boots => 3,
        EquipmentSlot::Accessory => 4,
    };
    v[si] = 1.0;

    // quality (1)
    v[5] = item.quality;

    // stat bonuses (4)
    v[6] = item.stat_bonuses.hp_bonus / 50.0;
    v[7] = item.stat_bonuses.attack_bonus / 25.0;
    v[8] = item.stat_bonuses.defense_bonus / 25.0;
    v[9] = item.stat_bonuses.speed_bonus / 10.0;

    v
}

// ---------------------------------------------------------------------------
// Quest slots
// ---------------------------------------------------------------------------

pub const QUEST_SLOT_DIM: usize = 13;

/// Convert a QuestRequest into a flat slot vector.
pub fn quest_to_slots(req: &QuestRequest) -> Vec<f32> {
    let mut v = vec![0.0f32; QUEST_SLOT_DIM];

    // quest_type one-hot (6)
    let qi = match req.quest_type {
        QuestType::Combat => 0,
        QuestType::Exploration => 1,
        QuestType::Diplomatic => 2,
        QuestType::Escort => 3,
        QuestType::Rescue => 4,
        QuestType::Gather => 5,
    };
    v[qi] = 1.0;

    // threat_level (1)
    v[6] = req.threat_level / 100.0;

    // distance (1)
    v[7] = req.distance / 50.0;

    // reward: gold, rep, supply, relation, loot (5)
    v[8] = req.reward.gold / 200.0;
    v[9] = req.reward.reputation / 10.0;
    v[10] = req.reward.supply_reward / 50.0;
    v[11] = (req.reward.relation_change + 50.0) / 100.0; // center around 0.5
    v[12] = if req.reward.potential_loot { 1.0 } else { 0.0 };

    v
}

// ---------------------------------------------------------------------------
// Effect encoding (shared between ability and passive)
// ---------------------------------------------------------------------------

/// Encode a ConditionalEffect into EFFECT_SLOT_DIM floats.
fn encode_effect(ce: &ConditionalEffect, out: &mut [f32]) {
    debug_assert!(out.len() >= EFFECT_SLOT_DIM);

    // effect_type category (17 categories, collapsed from 52)
    let (cat, param1, param2) = categorize_effect(&ce.effect);
    if cat < 17 { out[cat] = 1.0; }

    // primary param, duration (2)
    out[17] = param1 / 155.0;  // normalize to ~[0,1]
    out[18] = param2 / 10000.0; // duration in ms, normalize

    // area shape (5) — collapsed from 7
    if let Some(ref area) = ce.area {
        let (ai, ap1, ap2) = encode_area(area);
        if ai < 5 { out[19 + ai] = 1.0; }
        // We don't have room for area params in 25 dims, fold into effect params
        out[17] = out[17].max(ap1 / 10.0); // blend area radius into param
        let _ = ap2; // discard second area param for now
    }

    // condition presence (1)
    if let Some(ref cond) = ce.condition {
        if !matches!(cond, Condition::Always) {
            out[24] = 1.0;
        }
    }
}

/// Categorize an Effect into 17 categories + extract primary param + duration.
fn categorize_effect(effect: &Effect) -> (usize, f32, f32) {
    match effect {
        Effect::Damage { amount, duration_ms, .. } => (0, *amount as f32, *duration_ms as f32),
        Effect::Heal { amount, duration_ms, .. } => (1, *amount as f32, *duration_ms as f32),
        Effect::Shield { amount, duration_ms } => (2, *amount as f32, *duration_ms as f32),
        Effect::Stun { duration_ms } => (3, 0.0, *duration_ms as f32),
        Effect::Root { duration_ms } => (4, 0.0, *duration_ms as f32),
        Effect::Silence { duration_ms } => (5, 0.0, *duration_ms as f32),
        Effect::Fear { duration_ms } => (5, 0.0, *duration_ms as f32),
        Effect::Polymorph { duration_ms } => (5, 0.0, *duration_ms as f32),
        Effect::Slow { factor, duration_ms } => (6, *factor, *duration_ms as f32),
        Effect::Knockback { distance } => (7, *distance, 0.0),
        Effect::Pull { distance } => (7, *distance, 0.0),
        Effect::Dash { distance, .. } => (8, *distance, 0.0),
        Effect::Buff { factor, duration_ms, .. } => (9, *factor, *duration_ms as f32),
        Effect::Debuff { factor, duration_ms, .. } => (10, *factor, *duration_ms as f32),
        Effect::Summon { count, .. } => (11, *count as f32, 0.0),
        Effect::Stealth { duration_ms, .. } => (12, 0.0, *duration_ms as f32),
        Effect::Lifesteal { percent, duration_ms } => (13, *percent, *duration_ms as f32),
        Effect::Execute { hp_threshold_percent } => (14, *hp_threshold_percent, 0.0),
        Effect::Resurrect { hp_percent } => (15, *hp_percent, 0.0),
        // Everything else → utility (16)
        _ => (16, 0.0, 0.0),
    }
}

/// Encode an Area into (shape_index, param1, param2).
fn encode_area(area: &Area) -> (usize, f32, f32) {
    match area {
        Area::SingleTarget => (0, 0.0, 0.0),
        Area::Circle { radius } => (1, *radius, 0.0),
        Area::Cone { radius, angle_deg } => (2, *radius, *angle_deg / 360.0),
        Area::Line { length, width } => (3, *length, *width),
        Area::Ring { inner_radius, outer_radius } => (4, *inner_radius, *outer_radius),
        Area::SelfOnly => (0, 0.0, 0.0),
        Area::Spread { radius, .. } => (1, *radius, 0.0),
    }
}

// ---------------------------------------------------------------------------
// Index helpers
// ---------------------------------------------------------------------------

fn targeting_index(t: &AbilityTargeting) -> usize {
    match t {
        AbilityTargeting::TargetEnemy => 0,
        AbilityTargeting::TargetAlly => 1,
        AbilityTargeting::SelfCast => 2,
        AbilityTargeting::SelfAoe => 3,
        AbilityTargeting::GroundTarget => 4,
        AbilityTargeting::Direction => 5,
        AbilityTargeting::Vector => 6,
        AbilityTargeting::Global => 7,
    }
}

fn hint_index(hint: &str) -> usize {
    match hint {
        "damage" => 0,
        "crowd_control" => 1,
        "defense" => 2,
        "utility" => 3,
        "heal" => 4,
        _ => 3, // default to utility
    }
}

fn delivery_index(d: &Delivery) -> usize {
    match d {
        Delivery::Instant => 0,
        Delivery::Projectile { .. } => 1,
        Delivery::Channel { .. } => 2,
        Delivery::Zone { .. } => 3,
        Delivery::Tether { .. } => 4,
        Delivery::Trap { .. } => 5,
        Delivery::Chain { .. } => 6,
    }
}

fn encode_delivery_params(d: &Delivery, out: &mut [f32]) {
    match d {
        Delivery::Projectile { speed, width, .. } => {
            out[0] = *speed / 14.0;
            out[1] = *width / 1.1;
        }
        Delivery::Channel { duration_ms, tick_interval_ms } |
        Delivery::Zone { duration_ms, tick_interval_ms } => {
            out[2] = *duration_ms as f32 / 9000.0;
            out[3] = *tick_interval_ms as f32 / 2500.0;
        }
        Delivery::Chain { bounces, bounce_range, falloff, .. } => {
            out[3] = *bounces as f32 / 6.0;
            out[4] = *bounce_range / 6.0;
            out[5] = *falloff;
        }
        Delivery::Tether { max_range, tick_interval_ms, .. } => {
            out[4] = *max_range / 10.0;
            out[2] = *tick_interval_ms as f32 / 2500.0;
        }
        Delivery::Trap { duration_ms, trigger_radius, .. } => {
            out[2] = *duration_ms as f32 / 9000.0;
            out[4] = *trigger_radius / 5.0;
        }
        Delivery::Instant => {}
    }
}

fn class_tag_index(tag: &str) -> usize {
    match tag {
        "ranged" => 0,
        "nature" => 1,
        "stealth" => 2,
        "tracking" => 3,
        "survival" => 4,
        "melee" => 5,
        "defense" => 6,
        "leadership" => 7,
        "arcane" => 8,
        "elemental" => 9,
        "healing" => 10,
        "divine" => 11,
        "assassination" => 12,
        "agility" => 13,
        "deception" => 14,
        "sabotage" => 15,
        _ => 15, // catch-all
    }
}

fn scaling_source_index(source: &str) -> usize {
    match source {
        "party_alive_count" => 0,
        "party_size" => 1,
        "faction_strength" => 2,
        "coalition_strength" => 3,
        "crisis_severity" => 4,
        "fame" => 5,
        "territory_control" => 6,
        "adventurer_count" => 7,
        "gold" => 8,
        "reputation" => 9,
        "threat_level" => 10,
        _ => 0,
    }
}

fn encode_scaling_rule(rule: &ScalingRule, out: &mut [f32]) {
    // condition type (3): always, when_gt, when_gte, etc.
    match &rule.condition {
        ScalingCondition::Always => out[0] = 1.0,
        ScalingCondition::When { value, .. } => {
            out[1] = 1.0;
            out[2] = *value / 10.0; // threshold normalized
        }
    }

    // bonus type (5): stat_percent, stat_flat, mechanic, aura, conditional
    match &rule.bonus {
        ScalingBonus::StatPercent { percent, .. } => {
            out[3] = 1.0;
            out[7] = *percent / 100.0;
        }
        ScalingBonus::StatFlat { value, .. } => {
            out[4] = 1.0;
            out[7] = *value / 50.0;
        }
        ScalingBonus::Mechanic { value, .. } => {
            out[5] = 1.0;
            out[7] = *value;
        }
        ScalingBonus::Aura { value, .. } => {
            out[6] = 1.0;
            out[7] = *value / 10.0;
        }
        ScalingBonus::ConditionalMechanic { value, .. } => {
            out[5] = 1.0;
            out[7] = *value;
        }
    }
}

fn requirement_to_slot(req: &Requirement) -> (f32, f32) {
    match req {
        Requirement::Level(n) => (0.0 / 7.0, *n as f32 / 20.0),
        Requirement::Fame(f) => (1.0 / 7.0, *f / 2000.0),
        Requirement::Trait(_) => (2.0 / 7.0, 1.0),
        Requirement::QuestsCompleted(n) => (3.0 / 7.0, *n as f32 / 50.0),
        Requirement::ActiveCrisis => (4.0 / 7.0, 1.0),
        Requirement::GoldInvested(g) => (5.0 / 7.0, *g / 1000.0),
        Requirement::GroupSize(n) => (6.0 / 7.0, *n as f32 / 10.0),
        Requirement::AlliedFactions(n) => (1.0, *n as f32 / 5.0),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ability_slot_dim() {
        // Verify the constant matches what we actually produce
        let def = AbilityDef {
            name: "test".into(),
            targeting: AbilityTargeting::TargetEnemy,
            range: 5.0,
            cooldown_ms: 10000,
            cast_time_ms: 300,
            ai_hint: "damage".into(),
            effects: vec![],
            delivery: None,
            resource_cost: 0,
            morph_into: None,
            morph_duration_ms: 0,
            zone_tag: None,
            max_charges: 0,
            charge_recharge_ms: 0,
            is_toggle: false,
            toggle_cost_per_sec: 0.0,
            recast_count: 0,
            recast_window_ms: 0,
            recast_effects: vec![],
            unstoppable: false,
            swap_form: None,
            form: None,
            evolve_into: None,
        };
        let slots = ability_to_slots(&def);
        assert_eq!(slots.len(), ABILITY_SLOT_DIM);
        // No NaN
        for (i, &val) in slots.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at slot {}", i);
        }
    }

    #[test]
    fn test_class_slot_dim() {
        let def = ClassDef {
            name: "Test".into(),
            stat_growth: StatGrowth {
                attack: 2.0, defense: 1.0, speed: 3.0,
                max_hp: 0.0, ability_power: 1.0,
            },
            tags: vec!["ranged".into(), "nature".into()],
            scalings: vec![],
            abilities: vec![],
            requirements: vec![Requirement::Level(5)],
            consolidates_at: None,
        };
        let slots = class_to_slots(&def);
        assert_eq!(slots.len(), CLASS_SLOT_DIM);
    }

    #[test]
    fn test_quest_slot_dim() {
        let req = QuestRequest {
            id: 1,
            source_faction_id: None,
            source_area_id: None,
            quest_type: QuestType::Combat,
            threat_level: 50.0,
            reward: super::super::state::QuestReward::default(),
            distance: 10.0,
            target_position: (0.0, 0.0),
            deadline_ms: 60000,
            description: "test".into(),
            arrived_at_ms: 0,
        };
        let slots = quest_to_slots(&req);
        assert_eq!(slots.len(), QUEST_SLOT_DIM);
    }

    #[test]
    fn test_item_slot_dim() {
        let item = InventoryItem {
            id: 1,
            name: "Test Sword".into(),
            slot: EquipmentSlot::Weapon,
            quality: 0.8,
            stat_bonuses: super::super::state::StatBonuses {
                hp_bonus: 10.0,
                attack_bonus: 5.0,
                defense_bonus: 0.0,
                speed_bonus: 2.0,
            },
        };
        let slots = item_to_slots(&item);
        assert_eq!(slots.len(), ITEM_SLOT_DIM);
    }
}
