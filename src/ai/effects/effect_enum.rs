//! The large Effect enum with all variants and serde defaults.

use serde::{Deserialize, Serialize};

use super::types::{ConditionalEffect, DamageType};

// ---------------------------------------------------------------------------
// WHAT — Effect types (52 total)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Effect {
    // --- Existing (some modified) ---
    Damage {
        #[serde(default)]
        amount: i32,
        #[serde(default)]
        amount_per_tick: i32,
        #[serde(default)]
        duration_ms: u32,
        #[serde(default)]
        tick_interval_ms: u32,
        #[serde(default)]
        scaling_stat: Option<String>,
        #[serde(default)]
        scaling_percent: f32,
        #[serde(default)]
        damage_type: DamageType,
    },
    Heal {
        #[serde(default)]
        amount: i32,
        #[serde(default)]
        amount_per_tick: i32,
        #[serde(default)]
        duration_ms: u32,
        #[serde(default)]
        tick_interval_ms: u32,
        #[serde(default)]
        scaling_stat: Option<String>,
        #[serde(default)]
        scaling_percent: f32,
    },
    Shield {
        amount: i32,
        duration_ms: u32,
    },
    Stun {
        duration_ms: u32,
    },
    Slow {
        factor: f32,
        duration_ms: u32,
    },
    Knockback {
        distance: f32,
    },
    Dash {
        #[serde(default)]
        to_target: bool,
        #[serde(default = "default_dash_distance")]
        distance: f32,
        #[serde(default)]
        to_position: bool,
        /// If true, this is a blink (instant teleport, ignores terrain/grounded).
        #[serde(default)]
        is_blink: bool,
    },
    Buff {
        stat: String,
        factor: f32,
        duration_ms: u32,
    },
    Debuff {
        stat: String,
        factor: f32,
        duration_ms: u32,
    },
    Duel {
        duration_ms: u32,
    },
    Summon {
        template: String,
        #[serde(default = "default_summon_count")]
        count: u32,
        #[serde(default = "default_hp_percent")]
        hp_percent: f32,
        /// If true, summon is a clone of the caster (copies stats/abilities).
        #[serde(default)]
        clone: bool,
        /// Damage dealt by clone as % of caster's damage (default 75%).
        #[serde(default = "default_clone_damage_percent")]
        clone_damage_percent: f32,
        /// If true, summon is directed — it doesn't act independently.
        /// Instead it attacks when its owner attacks, from its own position.
        #[serde(default)]
        directed: bool,
    },
    /// Move all owned directed summons toward a target position.
    CommandSummons {
        #[serde(default = "default_command_speed")]
        speed: f32,
    },
    Dispel {
        #[serde(default)]
        target_tags: Vec<String>,
    },

    // --- Phase 2: CC & Positioning ---
    Root {
        duration_ms: u32,
    },
    Silence {
        duration_ms: u32,
    },
    Fear {
        duration_ms: u32,
    },
    Taunt {
        duration_ms: u32,
    },
    Pull {
        distance: f32,
    },
    Swap,

    // --- Phase 3: Damage Modifiers ---
    Reflect {
        percent: f32,
        duration_ms: u32,
    },
    Lifesteal {
        percent: f32,
        duration_ms: u32,
    },
    DamageModify {
        factor: f32,
        duration_ms: u32,
    },
    SelfDamage {
        amount: i32,
    },
    Execute {
        hp_threshold_percent: f32,
    },
    Blind {
        miss_chance: f32,
        duration_ms: u32,
    },
    OnHitBuff {
        duration_ms: u32,
        #[serde(default)]
        on_hit_effects: Vec<ConditionalEffect>,
    },

    // --- Phase 4: Healing & Shield ---
    Resurrect {
        hp_percent: f32,
    },
    OverhealShield {
        duration_ms: u32,
        #[serde(default = "default_conversion_percent")]
        conversion_percent: f32,
    },
    AbsorbToHeal {
        shield_amount: i32,
        duration_ms: u32,
        #[serde(default = "default_heal_percent")]
        heal_percent: f32,
    },
    ShieldSteal {
        amount: i32,
    },
    StatusClone {
        #[serde(default = "default_max_count")]
        max_count: u32,
    },

    // --- Phase 5: Status Interaction ---
    Immunity {
        immune_to: Vec<String>,
        duration_ms: u32,
    },
    Detonate {
        #[serde(default = "default_damage_multiplier")]
        damage_multiplier: f32,
    },
    StatusTransfer {
        #[serde(default)]
        steal_buffs: bool,
    },
    DeathMark {
        duration_ms: u32,
        #[serde(default = "default_damage_percent")]
        damage_percent: f32,
    },

    // --- Phase 6: Control & AI Override ---
    Polymorph {
        duration_ms: u32,
    },
    Banish {
        duration_ms: u32,
    },
    Confuse {
        duration_ms: u32,
    },
    Charm {
        duration_ms: u32,
    },

    // --- Phase 7: Complex Mechanics ---
    Stealth {
        duration_ms: u32,
        #[serde(default)]
        break_on_damage: bool,
        #[serde(default)]
        break_on_ability: bool,
    },
    Leash {
        max_range: f32,
        duration_ms: u32,
    },
    Link {
        duration_ms: u32,
        #[serde(default = "default_share_percent")]
        share_percent: f32,
    },
    Redirect {
        duration_ms: u32,
        #[serde(default = "default_redirect_charges")]
        charges: u32,
    },
    Rewind {
        #[serde(default = "default_lookback_ms")]
        lookback_ms: u32,
    },
    CooldownModify {
        amount_ms: i32,
        #[serde(default)]
        ability_name: Option<String>,
    },
    ApplyStacks {
        name: String,
        #[serde(default = "default_stack_count")]
        count: u32,
        #[serde(default = "default_max_stacks")]
        max_stacks: u32,
        #[serde(default)]
        duration_ms: u32,
    },

    // --- Terrain Modification ---
    Obstacle {
        width: f32,
        height: f32,
    },

    // --- LoL Coverage: New Effects ---
    /// Hard CC: cannot act, cannot be cleansed by normal means.
    Suppress {
        duration_ms: u32,
    },
    /// Prevents dashes, blinks, and movement abilities.
    Grounded {
        duration_ms: u32,
    },
    /// Blocks enemy projectiles in an area for a duration.
    ProjectileBlock {
        duration_ms: u32,
    },
    /// Attach to an ally — become untargetable and move with them.
    Attach {
        #[serde(default)]
        duration_ms: u32,
    },
    /// Evolve an ability — permanently replace it with its `evolve_into` variant.
    EvolveAbility {
        ability_index: usize,
    },

    // --- Expressiveness: Percent-Based HP Effects ---
    /// Deal damage as a percentage of the target's max HP.
    PercentHpDamage {
        percent: f32,
        #[serde(default)]
        damage_type: DamageType,
        /// Optional cap on the damage dealt (0 = uncapped).
        #[serde(default)]
        max_damage: i32,
    },
    /// Heal for a percentage of the target's missing HP.
    PercentMissingHpHeal {
        percent: f32,
    },
    /// Heal for a percentage of the target's max HP.
    PercentMaxHpHeal {
        percent: f32,
    },
}

fn default_summon_count() -> u32 {
    1
}
fn default_hp_percent() -> f32 {
    100.0
}
fn default_conversion_percent() -> f32 {
    100.0
}
fn default_heal_percent() -> f32 {
    50.0
}
fn default_max_count() -> u32 {
    3
}
fn default_damage_multiplier() -> f32 {
    1.0
}
fn default_damage_percent() -> f32 {
    50.0
}
fn default_share_percent() -> f32 {
    50.0
}
fn default_redirect_charges() -> u32 {
    3
}
fn default_lookback_ms() -> u32 {
    3000
}
fn default_dash_distance() -> f32 {
    2.0
}
fn default_command_speed() -> f32 {
    8.0
}
fn default_stack_count() -> u32 {
    1
}
fn default_max_stacks() -> u32 {
    4
}
fn default_clone_damage_percent() -> f32 {
    75.0
}
