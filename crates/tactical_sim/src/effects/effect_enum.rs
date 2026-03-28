//! The large Effect enum with all variants and serde defaults.

use serde::{Deserialize, Serialize};

use super::types::{ConditionalEffect, DamageType, ScalingTerm};

// ---------------------------------------------------------------------------
// WHAT — Effect types (68 total)
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
        /// Legacy single-stat scaling (still supported, prefer `bonus` for new abilities).
        #[serde(default)]
        scaling_stat: Option<String>,
        #[serde(default)]
        scaling_percent: f32,
        #[serde(default)]
        damage_type: DamageType,
        /// Composable scaling terms: total = amount + sum(stat_i * percent_i / 100).
        #[serde(default)]
        bonus: Vec<ScalingTerm>,
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
        /// Legacy single-stat scaling (still supported, prefer `bonus` for new abilities).
        #[serde(default)]
        scaling_stat: Option<String>,
        #[serde(default)]
        scaling_percent: f32,
        /// Composable scaling terms: total = amount + sum(stat_i * percent_i / 100).
        #[serde(default)]
        bonus: Vec<ScalingTerm>,
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

    // =======================================================================
    // Campaign Effects — targeting factions, regions, economy, etc.
    // These no-op in the combat sim; dispatched by campaign_apply_effect().
    // =======================================================================

    // --- Economy ---
    CornerMarket {
        #[serde(default)]
        commodity: String,
        #[serde(default)]
        duration_ticks: u32,
    },
    ForgeTradeRoute {
        #[serde(default)]
        income_per_tick: f32,
        #[serde(default)]
        duration_ticks: u32,
    },
    Appraise,
    GoldenTouch {
        #[serde(default)]
        duration_ticks: u32,
    },
    TradeEmbargo {
        #[serde(default)]
        duration_ticks: u32,
    },
    SilverTongue,

    // --- Diplomacy ---
    DemandAudience,
    CeasefireDeclaration {
        #[serde(default)]
        duration_ticks: u32,
    },
    Destabilize {
        #[serde(default)]
        instability: f32,
        #[serde(default)]
        duration_ticks: u32,
    },
    BrokerAlliance {
        #[serde(default)]
        duration_ticks: u32,
    },
    SubvertLoyalty,
    TreatyBreaker,
    ShatterAlliance,

    // --- Information ---
    Reveal {
        #[serde(default)]
        count: u32,
    },
    PropheticVision {
        #[serde(default)]
        count: u32,
    },
    BeastLore,
    ReadTheRoom,
    AllSeeingEye,
    Decipher,
    TrapSense,
    SapperEye,

    // --- Leadership ---
    Rally {
        #[serde(default)]
        morale_restore: f32,
    },
    Inspire {
        #[serde(default)]
        morale_boost: f32,
        #[serde(default)]
        duration_ticks: u32,
    },
    FieldCommand {
        #[serde(default)]
        duration_ticks: u32,
    },
    CoordinatedStrike,
    WarCry {
        #[serde(default)]
        duration_ticks: u32,
    },
    RallyingCry {
        #[serde(default)]
        morale_restore: f32,
    },

    // --- Stealth / Movement ---
    GhostWalk {
        #[serde(default)]
        duration_ticks: u32,
    },
    ShadowStep {
        #[serde(default)]
        duration_ticks: u32,
    },
    SilentMovement,
    HiddenCamp {
        #[serde(default)]
        duration_ticks: u32,
    },
    Vanish,
    Distraction {
        #[serde(default)]
        duration_ticks: u32,
    },

    // --- Territory ---
    ClaimTerritory,
    Fortify {
        #[serde(default)]
        duration_ticks: u32,
    },
    Sanctuary {
        #[serde(default)]
        duration_ticks: u32,
    },
    PlagueWard {
        #[serde(default)]
        duration_ticks: u32,
    },
    SafeHouse {
        #[serde(default)]
        duration_ticks: u32,
    },

    // --- Supernatural / Body ---
    BloodOath {
        #[serde(default)]
        stat_bonus: f32,
    },
    Unbreakable,
    LifeEternal,
    Purify,
    NameTheNameless,
    ForbiddenKnowledge,

    // --- Passive Skill-State ---
    FieldTriage {
        #[serde(default)]
        heal_rate_multiplier: f32,
    },
    InspiringPresence {
        #[serde(default)]
        morale_boost: f32,
    },
    BattleInstinct,
    QuickStudy,
    Forage {
        #[serde(default)]
        supply_per_tick: f32,
    },
    TrackPrey,
    FieldRepair,
    StabilizeAlly,

    // --- Higher Tier ---
    Disinformation {
        #[serde(default)]
        duration_ticks: u32,
    },
    AcceleratedStudy {
        #[serde(default)]
        duration_ticks: u32,
    },
    TakeTheBlow {
        #[serde(default)]
        duration_ticks: u32,
    },
    HoldTheLine,
    Forgery,
    MasterworkCraft,
    IntelGathering,
    MasterArmorer,
    MarketMaker {
        #[serde(default)]
        duration_ticks: u32,
    },
    ForgeArtifact,
    TradeEmpire {
        #[serde(default)]
        income_per_tick: f32,
    },

    // --- Legendary / Mythic ---
    LivingLegend,
    RewriteHistory,
    TheLastWord,
    WealthOfNations,
    Omniscience,
    ImmortalMoment,
    ClaimByRight,
    RewriteTheRecord,

    // =======================================================================
    // Meta-Effects — abilities that modify other abilities
    // =======================================================================

    /// Reset all ability cooldowns to 0.
    RefreshCooldowns,
    /// Reset a specific ability's cooldown (by index).
    RefreshCooldown {
        #[serde(default)]
        ability_index: u32,
    },
    /// Next N ability casts have multiplied effectiveness.
    Amplify {
        #[serde(default = "default_amplify_multiplier")]
        multiplier: f32,
        #[serde(default = "default_charges")]
        charges: u32,
    },
    /// Next ability cast fires twice.
    Echo {
        #[serde(default = "default_charges")]
        charges: u32,
    },
    /// Extend all active buff/debuff durations by N ms.
    ExtendDurations {
        #[serde(default)]
        amount_ms: u32,
    },
    /// Next ability has 0 cast time.
    InstantCast {
        #[serde(default = "default_charges")]
        charges: u32,
    },
    /// Next ability costs 0 resources.
    FreeCast {
        #[serde(default = "default_charges")]
        charges: u32,
    },
    /// Block the next enemy ability entirely.
    SpellShield {
        #[serde(default = "default_charges")]
        charges: u32,
    },
    /// Increase target's ability costs by multiplier for duration.
    ManaBurn {
        #[serde(default = "default_cost_multiplier")]
        cost_multiplier: f32,
        #[serde(default)]
        duration_ms: u32,
    },
    /// Lock target out of abilities for duration (like silence but uncleansable).
    CooldownLock {
        #[serde(default)]
        duration_ms: u32,
    },

    // =======================================================================
    // Recursive Effects — abilities that reference or contain other abilities
    // =======================================================================

    /// Trigger a named ability when this effect lands.
    OnHitCast {
        #[serde(default)]
        ability_name: String,
    },
    /// Grant a temporary ability to the target for duration (by name reference).
    GrantAbility {
        #[serde(default)]
        ability_name: String,
        #[serde(default)]
        duration_ms: u32,
    },
    /// Copy the last ability used by the target and make it available once.
    CastCopy,
    /// After N casts of this ability, evolve it permanently.
    EvolveAfter {
        #[serde(default)]
        cast_count: u32,
    },

    // =======================================================================
    // Campaign Primitives — composable building blocks
    // =======================================================================

    /// Modify a numeric property on any entity.
    /// `entity`: "adventurer", "party", "guild", "faction", "region"
    /// `property`: "morale", "gold", "supplies", "military_strength", etc.
    /// `op`: "add", "multiply", "set"
    ModifyStat {
        #[serde(default)]
        entity: String,
        #[serde(default)]
        property: String,
        #[serde(default = "default_modify_op")]
        op: String,
        #[serde(default)]
        amount: f32,
        #[serde(default)]
        duration_ticks: u32,
    },
    /// Set a boolean flag on an entity.
    SetFlag {
        #[serde(default)]
        entity: String,
        #[serde(default)]
        flag: String,
        #[serde(default = "default_flag_value")]
        value: bool,
        #[serde(default)]
        duration_ticks: u32,
    },
    /// Reveal hidden information.
    /// `scope`: "faction_stance", "enemy_weaknesses", "upcoming_events", "map", "all"
    RevealInfo {
        #[serde(default)]
        target_type: String,
        #[serde(default)]
        scope: String,
    },
    /// Create a new entity or modifier.
    /// `entity_type`: "location", "trade_route", "agreement", "item", "region_modifier"
    CreateEntity {
        #[serde(default)]
        entity_type: String,
        #[serde(default)]
        subtype: String,
        #[serde(default)]
        duration_ticks: u32,
    },
    /// Remove/destroy something.
    /// `target_type`: "disease", "agreement", "war_state", "buff", "event"
    DestroyEntity {
        #[serde(default)]
        target_type: String,
    },
    /// Transfer a property value between two entities.
    /// E.g. transfer adventurer from faction to guild, or damage from ally to self.
    TransferValue {
        #[serde(default)]
        from_entity: String,
        #[serde(default)]
        to_entity: String,
        #[serde(default)]
        property: String,
        #[serde(default)]
        amount: f32,
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
fn default_amplify_multiplier() -> f32 {
    1.5
}
fn default_charges() -> u32 {
    1
}
fn default_cost_multiplier() -> f32 {
    2.0
}
fn default_modify_op() -> String {
    "add".to_string()
}
fn default_flag_value() -> bool {
    true
}
