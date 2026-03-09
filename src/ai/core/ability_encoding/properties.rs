//! Raw property extraction (80-dim features) and constants.

use crate::ai::effects::{
    AbilityTargeting, Delivery,
};
use crate::ai::core::ability_eval::AbilityCategory;

use super::effects::{collect_all_effects, summarize_effects};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Raw property features per ability.
pub const ABILITY_PROP_DIM: usize = 80;

// ---------------------------------------------------------------------------
// Raw property extraction
// ---------------------------------------------------------------------------

/// Extract raw numeric properties from an AbilityDef into a fixed-size vector.
/// This is the input to the embedding encoder — not used directly by the policy.
pub fn extract_ability_properties(
    def: &crate::ai::effects::AbilityDef,
) -> [f32; ABILITY_PROP_DIM] {
    let mut f = [0.0f32; ABILITY_PROP_DIM];
    let mut i = 0;

    // [0..8] Targeting one-hot
    let tgt_idx = match def.targeting {
        AbilityTargeting::TargetEnemy => 0,
        AbilityTargeting::TargetAlly => 1,
        AbilityTargeting::SelfCast => 2,
        AbilityTargeting::SelfAoe => 3,
        AbilityTargeting::GroundTarget => 4,
        AbilityTargeting::Direction => 5,
        AbilityTargeting::Vector => 6,
        AbilityTargeting::Global => 7,
    };
    f[tgt_idx] = 1.0;
    i += 8;

    // [8..14] Core scalars
    f[i] = def.range / 10.0;
    f[i + 1] = def.cooldown_ms as f32 / 20000.0;
    f[i + 2] = def.cast_time_ms as f32 / 2000.0;
    f[i + 3] = def.resource_cost as f32 / 30.0;
    // slots 4,5 reserved for runtime (filled by encode_ability_slot)
    f[i + 4] = 0.0;
    f[i + 5] = 0.0;
    i += 6;

    // [14..21] Delivery one-hot
    let del_idx = match &def.delivery {
        None => 0,                          // instant/none
        Some(Delivery::Instant) => 0,
        Some(Delivery::Projectile { .. }) => 1,
        Some(Delivery::Channel { .. }) => 2,
        Some(Delivery::Zone { .. }) => 3,
        Some(Delivery::Tether { .. }) => 4,
        Some(Delivery::Trap { .. }) => 5,
        Some(Delivery::Chain { .. }) => 6,
    };
    f[i + del_idx] = 1.0;
    i += 7;

    // [21..27] Delivery properties
    match &def.delivery {
        Some(Delivery::Projectile { speed, pierce, .. }) => {
            f[i] = speed / 20.0;
            f[i + 1] = if *pierce { 1.0 } else { 0.0 };
        }
        Some(Delivery::Channel { duration_ms, tick_interval_ms }) => {
            f[i + 2] = *duration_ms as f32 / 5000.0;
            f[i + 3] = *tick_interval_ms as f32 / 2000.0;
        }
        Some(Delivery::Zone { duration_ms, tick_interval_ms }) => {
            f[i + 2] = *duration_ms as f32 / 5000.0;
            f[i + 3] = *tick_interval_ms as f32 / 2000.0;
        }
        Some(Delivery::Tether { max_range, tick_interval_ms, .. }) => {
            f[i] = max_range / 20.0;
            f[i + 3] = *tick_interval_ms as f32 / 2000.0;
        }
        Some(Delivery::Chain { bounces, bounce_range, .. }) => {
            f[i + 4] = *bounces as f32 / 5.0;
            f[i] = bounce_range / 20.0;
        }
        Some(Delivery::Trap { duration_ms, trigger_radius, .. }) => {
            f[i + 2] = *duration_ms as f32 / 5000.0;
            f[i + 5] = trigger_radius / 5.0;
        }
        _ => {}
    }
    i += 6;

    // [27..32] Mechanic flags
    f[i] = if def.is_toggle { 1.0 } else { 0.0 };
    f[i + 1] = if def.unstoppable { 1.0 } else { 0.0 };
    f[i + 2] = if def.recast_count > 0 { 1.0 } else { 0.0 };
    f[i + 3] = if def.max_charges > 0 { 1.0 } else { 0.0 };
    f[i + 4] = if def.morph_into.is_some() { 1.0 } else { 0.0 };
    i += 5;

    // [32..38] AI hint one-hot
    let hint_idx = match def.ai_hint.as_str() {
        "damage" => 0,
        "heal" => 1,
        "crowd_control" => 2,
        "defense" => 3,
        "utility" => 4,
        _ => 5,
    };
    f[i + hint_idx] = 1.0;
    i += 6;

    // Walk all effects (direct + delivery sub-effects)
    let all_effects = collect_all_effects(&def.effects, &def.delivery);
    let summary = summarize_effects(&all_effects);

    // [38..41] Damage type presence
    f[i] = summary.has_physical;
    f[i + 1] = summary.has_magic;
    f[i + 2] = summary.has_true;
    i += 3;

    // [41..45] Damage
    f[i] = summary.total_instant_damage / 150.0;
    f[i + 1] = summary.dot_dps / 50.0;
    f[i + 2] = summary.has_execute;
    f[i + 3] = summary.damage_modify_factor;
    i += 4;

    // [45..48] Healing
    f[i] = summary.total_instant_heal / 150.0;
    f[i + 1] = summary.hot_hps / 50.0;
    f[i + 2] = summary.total_shield / 100.0;
    i += 3;

    // [48..55] Hard CC durations
    f[i] = summary.stun_ms / 3000.0;
    f[i + 1] = summary.root_ms / 3000.0;
    f[i + 2] = summary.silence_ms / 3000.0;
    f[i + 3] = summary.suppress_ms / 3000.0;
    f[i + 4] = summary.fear_ms / 3000.0;
    f[i + 5] = summary.charm_ms / 3000.0;
    f[i + 6] = summary.polymorph_ms / 3000.0;
    i += 7;

    // [55..59] Soft CC
    f[i] = summary.slow_factor;
    f[i + 1] = summary.slow_ms / 3000.0;
    f[i + 2] = summary.knockback_dist / 5.0;
    f[i + 3] = summary.pull_dist / 5.0;
    i += 4;

    // [59..63] Other CC
    f[i] = summary.taunt_ms / 3000.0;
    f[i + 1] = summary.banish_ms / 3000.0;
    f[i + 2] = summary.confuse_ms / 3000.0;
    f[i + 3] = summary.grounded_ms / 3000.0;
    i += 4;

    // [63..66] Mobility
    f[i] = summary.has_dash;
    f[i + 1] = summary.is_blink;
    f[i + 2] = summary.has_stealth;
    i += 3;

    // [66..70] Buffs/debuffs
    f[i] = summary.buff_factor;
    f[i + 1] = summary.buff_dur_ms / 5000.0;
    f[i + 2] = summary.debuff_factor;
    f[i + 3] = summary.debuff_dur_ms / 5000.0;
    i += 4;

    // [70..75] Area
    f[i] = summary.max_radius / 6.0;
    f[i + 1] = summary.has_cone;
    f[i + 2] = summary.has_line;
    f[i + 3] = summary.has_spread;
    f[i + 4] = summary.is_aoe;
    i += 5;

    // [75..80] Special
    f[i] = summary.has_summon;
    f[i + 1] = summary.has_obstacle;
    f[i + 2] = summary.has_reflect;
    f[i + 3] = summary.has_lifesteal;
    f[i + 4] = summary.num_effects / 8.0;

    debug_assert_eq!(i + 5, ABILITY_PROP_DIM);
    f
}

/// Get the AbilityCategory label for an ability (used as contrastive class).
pub fn ability_category_label(def: &crate::ai::effects::AbilityDef) -> AbilityCategory {
    AbilityCategory::from_ability_full(
        &def.ai_hint,
        &def.targeting,
        &def.effects,
        def.delivery.as_ref(),
    )
}
