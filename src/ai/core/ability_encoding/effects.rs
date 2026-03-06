//! Effect walking & summarization helpers.

use crate::ai::effects::{
    Area, ConditionalEffect, DamageType, Delivery, Effect,
};

// ---------------------------------------------------------------------------
// Effect collection
// ---------------------------------------------------------------------------

/// Collect all ConditionalEffects from direct effects + delivery sub-effects.
pub(super) fn collect_all_effects<'a>(
    effects: &'a [ConditionalEffect],
    delivery: &'a Option<Delivery>,
) -> Vec<&'a ConditionalEffect> {
    let mut all: Vec<&ConditionalEffect> = effects.iter().collect();
    match delivery {
        Some(Delivery::Projectile { on_hit, on_arrival, .. }) => {
            all.extend(on_hit.iter());
            all.extend(on_arrival.iter());
        }
        Some(Delivery::Chain { on_hit, .. }) => {
            all.extend(on_hit.iter());
        }
        Some(Delivery::Tether { on_complete, .. }) => {
            all.extend(on_complete.iter());
        }
        _ => {}
    }
    all
}

// ---------------------------------------------------------------------------
// EffectSummary
// ---------------------------------------------------------------------------

#[derive(Default)]
pub(super) struct EffectSummary {
    // Damage
    pub has_physical: f32,
    pub has_magic: f32,
    pub has_true: f32,
    pub total_instant_damage: f32,
    pub dot_dps: f32,
    pub has_execute: f32,
    pub damage_modify_factor: f32,

    // Healing
    pub total_instant_heal: f32,
    pub hot_hps: f32,
    pub total_shield: f32,

    // Hard CC
    pub stun_ms: f32,
    pub root_ms: f32,
    pub silence_ms: f32,
    pub suppress_ms: f32,
    pub fear_ms: f32,
    pub charm_ms: f32,
    pub polymorph_ms: f32,

    // Soft CC
    pub slow_factor: f32,
    pub slow_ms: f32,
    pub knockback_dist: f32,
    pub pull_dist: f32,

    // Other CC
    pub taunt_ms: f32,
    pub banish_ms: f32,
    pub confuse_ms: f32,
    pub grounded_ms: f32,

    // Mobility
    pub has_dash: f32,
    pub is_blink: f32,
    pub has_stealth: f32,

    // Buffs
    pub buff_factor: f32,
    pub buff_dur_ms: f32,
    pub debuff_factor: f32,
    pub debuff_dur_ms: f32,

    // Area
    pub max_radius: f32,
    pub has_cone: f32,
    pub has_line: f32,
    pub has_spread: f32,
    pub is_aoe: f32,

    // Special
    pub has_summon: f32,
    pub has_obstacle: f32,
    pub has_reflect: f32,
    pub has_lifesteal: f32,
    pub num_effects: f32,
}

// ---------------------------------------------------------------------------
// Summarization
// ---------------------------------------------------------------------------

pub(super) fn summarize_effects(effects: &[&ConditionalEffect]) -> EffectSummary {
    let mut s = EffectSummary::default();
    s.num_effects = effects.len() as f32;

    for ce in effects {
        // Area summary
        if let Some(area) = &ce.area {
            match area {
                Area::Circle { radius } => {
                    s.max_radius = s.max_radius.max(*radius);
                    s.is_aoe = 1.0;
                }
                Area::Cone { radius, .. } => {
                    s.max_radius = s.max_radius.max(*radius);
                    s.has_cone = 1.0;
                    s.is_aoe = 1.0;
                }
                Area::Line { length, .. } => {
                    s.max_radius = s.max_radius.max(*length);
                    s.has_line = 1.0;
                    s.is_aoe = 1.0;
                }
                Area::Ring { outer_radius, .. } => {
                    s.max_radius = s.max_radius.max(*outer_radius);
                    s.is_aoe = 1.0;
                }
                Area::Spread { radius, .. } => {
                    s.max_radius = s.max_radius.max(*radius);
                    s.has_spread = 1.0;
                    s.is_aoe = 1.0;
                }
                Area::SingleTarget | Area::SelfOnly => {}
            }
        }

        // Effect summary
        match &ce.effect {
            Effect::Damage {
                amount,
                amount_per_tick,
                tick_interval_ms,
                damage_type,
                ..
            } => {
                s.total_instant_damage += *amount as f32;
                if *tick_interval_ms > 0 && *amount_per_tick > 0 {
                    s.dot_dps += *amount_per_tick as f32 * 1000.0 / *tick_interval_ms as f32;
                }
                match damage_type {
                    DamageType::Physical => s.has_physical = 1.0,
                    DamageType::Magic => s.has_magic = 1.0,
                    DamageType::True => s.has_true = 1.0,
                }
            }
            Effect::Heal {
                amount,
                amount_per_tick,
                tick_interval_ms,
                ..
            } => {
                s.total_instant_heal += *amount as f32;
                if *tick_interval_ms > 0 && *amount_per_tick > 0 {
                    s.hot_hps += *amount_per_tick as f32 * 1000.0 / *tick_interval_ms as f32;
                }
            }
            Effect::Shield { amount, .. } => {
                s.total_shield += *amount as f32;
            }
            Effect::Stun { duration_ms } => {
                s.stun_ms = s.stun_ms.max(*duration_ms as f32);
            }
            Effect::Slow { factor, duration_ms } => {
                s.slow_factor = s.slow_factor.max(*factor);
                s.slow_ms = s.slow_ms.max(*duration_ms as f32);
            }
            Effect::Knockback { distance } => {
                s.knockback_dist = s.knockback_dist.max(*distance);
            }
            Effect::Dash { is_blink, .. } => {
                s.has_dash = 1.0;
                if *is_blink {
                    s.is_blink = 1.0;
                }
            }
            Effect::Buff { factor, duration_ms, .. } => {
                s.buff_factor = s.buff_factor.max(*factor);
                s.buff_dur_ms = s.buff_dur_ms.max(*duration_ms as f32);
            }
            Effect::Debuff { factor, duration_ms, .. } => {
                s.debuff_factor = s.debuff_factor.max(*factor);
                s.debuff_dur_ms = s.debuff_dur_ms.max(*duration_ms as f32);
            }
            Effect::Root { duration_ms } => {
                s.root_ms = s.root_ms.max(*duration_ms as f32);
            }
            Effect::Silence { duration_ms } => {
                s.silence_ms = s.silence_ms.max(*duration_ms as f32);
            }
            Effect::Fear { duration_ms } => {
                s.fear_ms = s.fear_ms.max(*duration_ms as f32);
            }
            Effect::Taunt { duration_ms } => {
                s.taunt_ms = s.taunt_ms.max(*duration_ms as f32);
            }
            Effect::Pull { distance } => {
                s.pull_dist = s.pull_dist.max(*distance);
            }
            Effect::Charm { duration_ms } => {
                s.charm_ms = s.charm_ms.max(*duration_ms as f32);
            }
            Effect::Polymorph { duration_ms } => {
                s.polymorph_ms = s.polymorph_ms.max(*duration_ms as f32);
            }
            Effect::Banish { duration_ms } => {
                s.banish_ms = s.banish_ms.max(*duration_ms as f32);
            }
            Effect::Confuse { duration_ms } => {
                s.confuse_ms = s.confuse_ms.max(*duration_ms as f32);
            }
            Effect::Suppress { duration_ms } => {
                s.suppress_ms = s.suppress_ms.max(*duration_ms as f32);
            }
            Effect::Grounded { duration_ms } => {
                s.grounded_ms = s.grounded_ms.max(*duration_ms as f32);
            }
            Effect::Stealth { .. } => {
                s.has_stealth = 1.0;
            }
            Effect::Reflect { .. } => {
                s.has_reflect = 1.0;
            }
            Effect::Lifesteal { .. } => {
                s.has_lifesteal = 1.0;
            }
            Effect::DamageModify { factor, .. } => {
                s.damage_modify_factor = s.damage_modify_factor.max(*factor);
            }
            Effect::Execute { .. } => {
                s.has_execute = 1.0;
            }
            Effect::Summon { .. } => {
                s.has_summon = 1.0;
            }
            Effect::Obstacle { .. } => {
                s.has_obstacle = 1.0;
            }
            Effect::PercentHpDamage { percent, damage_type, .. } => {
                // Estimate damage as percent of a nominal 100 HP target
                s.total_instant_damage += percent;
                match damage_type {
                    DamageType::Physical => s.has_physical = 1.0,
                    DamageType::Magic => s.has_magic = 1.0,
                    DamageType::True => s.has_true = 1.0,
                }
            }
            Effect::PercentMissingHpHeal { percent } | Effect::PercentMaxHpHeal { percent } => {
                // Estimate heal as percent of a nominal 100 HP target
                s.total_instant_heal += percent;
            }
            _ => {}
        }
    }

    s
}
