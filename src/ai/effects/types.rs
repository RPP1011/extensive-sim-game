//! Hero Ability Engine — data-driven effect system.
//!
//! Abilities are plain data (`AbilityDef` / `PassiveDef`), not closures.
//! The sim pipeline reads them at hook points and executes via a dispatcher.
//!
//! Five composable dimensions: WHAT (Effect) × WHERE (Area) × HOW (Delivery)
//! × WHEN (Trigger/Condition) + Tags.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use super::effect_enum::Effect;

// ---------------------------------------------------------------------------
// ScalingTerm — composable value expressions for effect amounts
// ---------------------------------------------------------------------------

/// A stat reference that can be resolved at runtime against the sim state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatRef {
    TargetMaxHp,
    TargetCurrentHp,
    TargetMissingHp,
    CasterMaxHp,
    CasterCurrentHp,
    CasterMissingHp,
    CasterAttackDamage,
    /// Number of stacks with the given name on the target.
    TargetStacks { name: String },
    /// Number of stacks with the given name on the caster.
    CasterStacks { name: String },
}

/// One additive scaling term: contributes `percent`% of `stat` to an effect's amount.
/// Multiple terms stack additively: total = base_amount + sum(stat_i * percent_i / 100).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingTerm {
    pub stat: StatRef,
    pub percent: f32,
    /// Hard cap on this term's contribution (0 = uncapped).
    #[serde(default)]
    pub max: i32,
    /// If true, consume (remove) the referenced stacks after reading.
    #[serde(default)]
    pub consume: bool,
}

// ---------------------------------------------------------------------------
// Tags — arbitrary named resistance/power levels
// ---------------------------------------------------------------------------

/// Arbitrary named tags with numeric power levels.
/// Effect tags assert power; unit resistance tags resist if value >= effect's tag.
pub type Tags = HashMap<String, f32>;

// ---------------------------------------------------------------------------
// Stacking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stacking {
    Refresh,
    Extend,
    Strongest,
    Stack,
}

impl Default for Stacking {
    fn default() -> Self {
        Stacking::Refresh
    }
}

// ---------------------------------------------------------------------------
// DamageType — physical/magic/true
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DamageType {
    Physical,
    Magic,
    True,
}

impl Default for DamageType {
    fn default() -> Self {
        DamageType::Physical
    }
}

// ---------------------------------------------------------------------------
// WHERE — Area shapes (7 total)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum Area {
    SingleTarget,
    Circle {
        radius: f32,
    },
    Cone {
        radius: f32,
        angle_deg: f32,
    },
    Line {
        length: f32,
        width: f32,
    },
    Ring {
        inner_radius: f32,
        outer_radius: f32,
    },
    #[serde(rename = "self")]
    SelfOnly,
    Spread {
        radius: f32,
        #[serde(default)]
        max_targets: u32,
    },
}

impl Default for Area {
    fn default() -> Self {
        Area::SingleTarget
    }
}

// ---------------------------------------------------------------------------
// HOW — Delivery methods (7 total)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum Delivery {
    Instant,
    Projectile {
        speed: f32,
        #[serde(default)]
        pierce: bool,
        #[serde(default)]
        width: f32,
        #[serde(default)]
        on_hit: Vec<ConditionalEffect>,
        #[serde(default)]
        on_arrival: Vec<ConditionalEffect>,
    },
    Channel {
        duration_ms: u32,
        tick_interval_ms: u32,
    },
    Zone {
        duration_ms: u32,
        tick_interval_ms: u32,
    },
    Tether {
        max_range: f32,
        #[serde(default)]
        tick_interval_ms: u32,
        #[serde(default)]
        on_complete: Vec<ConditionalEffect>,
    },
    Trap {
        duration_ms: u32,
        trigger_radius: f32,
        #[serde(default)]
        arm_time_ms: u32,
    },
    Chain {
        bounces: u32,
        bounce_range: f32,
        #[serde(default)]
        falloff: f32,
        #[serde(default)]
        on_hit: Vec<ConditionalEffect>,
    },
}

impl Default for Delivery {
    fn default() -> Self {
        Delivery::Instant
    }
}

// ---------------------------------------------------------------------------
// WHEN — Conditions (per-effect) and Triggers (for passives)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Condition {
    Always,
    TargetHpBelow { percent: f32 },
    TargetHpAbove { percent: f32 },
    TargetIsStunned,
    TargetIsSlowed,
    CasterHpBelow { percent: f32 },
    CasterHpAbove { percent: f32 },
    HitCountAbove { count: u32 },
    TargetHasTag { tag: String },
    // --- Phase 9: New Conditions ---
    TargetIsRooted,
    TargetIsSilenced,
    TargetIsFeared,
    TargetIsTaunted,
    TargetIsBanished,
    TargetIsStealthed,
    TargetIsCharmed,
    TargetIsPolymorphed,
    CasterHasStatus { status: String },
    TargetHasStatus { status: String },
    TargetDebuffCount { min_count: u32 },
    CasterBuffCount { min_count: u32 },
    AllyCountBelow { count: u32 },
    EnemyCountBelow { count: u32 },
    TargetStackCount { name: String, min_count: u32 },

    // --- Compound Conditions ---
    /// All sub-conditions must be true.
    And { conditions: Vec<Condition> },
    /// At least one sub-condition must be true.
    Or { conditions: Vec<Condition> },
    /// Inverts the inner condition.
    Not { condition: Box<Condition> },

    // --- Spatial & Resource Conditions ---
    /// True if caster-to-target distance is below `range` units.
    TargetDistanceBelow { range: f32 },
    /// True if caster-to-target distance is above `range` units.
    TargetDistanceAbove { range: f32 },
    /// True if caster's resource is below `percent`% of max.
    CasterResourceBelow { percent: f32 },
    /// True if caster's resource is above `percent`% of max.
    CasterResourceAbove { percent: f32 },
}

impl Default for Condition {
    fn default() -> Self {
        Condition::Always
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Trigger {
    OnDamageDealt,
    OnDamageTaken,
    OnKill,
    OnAllyDamaged {
        #[serde(default = "default_trigger_range")]
        range: f32,
    },
    OnDeath,
    OnAbilityUsed,
    OnHpBelow { percent: f32 },
    OnHpAbove { percent: f32 },
    OnShieldBroken,
    OnStunExpire,
    Periodic { interval_ms: u32 },
    // --- Phase 9: New Triggers ---
    OnHealReceived,
    OnStatusApplied,
    OnStatusExpired,
    OnResurrect,
    OnDodge,
    OnReflect,
    OnAllyKilled {
        #[serde(default = "default_trigger_range")]
        range: f32,
    },
    OnAutoAttack,
    OnStackReached {
        name: String,
        count: u32,
    },
}

fn default_trigger_range() -> f32 {
    5.0
}

// ---------------------------------------------------------------------------
// ConditionalEffect — wraps an Effect with condition, area, tags, stacking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalEffect {
    #[serde(flatten)]
    pub effect: Effect,
    #[serde(default)]
    pub condition: Option<Condition>,
    #[serde(default)]
    pub area: Option<Area>,
    #[serde(default)]
    pub tags: Tags,
    #[serde(default)]
    pub stacking: Stacking,
    /// Probability (0.0–1.0) that this effect fires when its condition passes.
    /// 0.0 = default = always fires (treated as 1.0).
    #[serde(default)]
    pub chance: f32,
    /// Effects to apply instead when the condition evaluates to false.
    #[serde(default)]
    pub else_effects: Vec<ConditionalEffect>,
}
