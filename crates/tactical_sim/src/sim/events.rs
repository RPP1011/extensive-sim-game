use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimEvent {
    Moved {
        tick: u64,
        unit_id: u32,
        from_x100: i32,
        from_y100: i32,
        to_x100: i32,
        to_y100: i32,
    },
    CastStarted {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    CastFailedOutOfRange {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    DamageApplied {
        tick: u64,
        source_id: u32,
        target_id: u32,
        amount: i32,
        target_hp_before: i32,
        target_hp_after: i32,
    },
    UnitDied {
        tick: u64,
        unit_id: u32,
    },
    AttackBlockedCooldown {
        tick: u64,
        unit_id: u32,
        target_id: u32,
        cooldown_remaining_ms: u32,
    },
    AbilityBlockedCooldown {
        tick: u64,
        unit_id: u32,
        target_id: u32,
        cooldown_remaining_ms: u32,
    },
    AttackBlockedInvalidTarget {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    AbilityBlockedInvalidTarget {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    AttackRepositioned {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    AbilityBlockedOutOfRange {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    AbilityCastStarted {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    HealBlockedCooldown {
        tick: u64,
        unit_id: u32,
        target_id: u32,
        cooldown_remaining_ms: u32,
    },
    HealBlockedInvalidTarget {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    HealBlockedOutOfRange {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    HealCastStarted {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    ControlBlockedCooldown {
        tick: u64,
        unit_id: u32,
        target_id: u32,
        cooldown_remaining_ms: u32,
    },
    ControlBlockedInvalidTarget {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    ControlBlockedOutOfRange {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    ControlCastStarted {
        tick: u64,
        unit_id: u32,
        target_id: u32,
    },
    ControlApplied {
        tick: u64,
        source_id: u32,
        target_id: u32,
        duration_ms: u32,
    },
    UnitControlled {
        tick: u64,
        unit_id: u32,
    },
    HealApplied {
        tick: u64,
        source_id: u32,
        target_id: u32,
        amount: i32,
        target_hp_before: i32,
        target_hp_after: i32,
    },
    // --- Hero ability engine events ---
    AbilityUsed {
        tick: u64,
        unit_id: u32,
        ability_index: usize,
        ability_name: String,
    },
    ShieldApplied {
        tick: u64,
        unit_id: u32,
        amount: i32,
    },
    ShieldAbsorbed {
        tick: u64,
        unit_id: u32,
        absorbed: i32,
        remaining: i32,
    },
    StatusEffectApplied {
        tick: u64,
        unit_id: u32,
        effect_name: String,
    },
    StatusEffectExpired {
        tick: u64,
        unit_id: u32,
        effect_name: String,
    },
    DashPerformed {
        tick: u64,
        unit_id: u32,
        from_x100: i32,
        from_y100: i32,
        to_x100: i32,
        to_y100: i32,
    },
    KnockbackApplied {
        tick: u64,
        source_id: u32,
        target_id: u32,
        distance_x100: i32,
    },
    PassiveTriggered {
        tick: u64,
        unit_id: u32,
        passive_name: String,
    },
    ProjectileSpawned {
        tick: u64,
        source_id: u32,
        target_id: u32,
    },
    ProjectileHit {
        tick: u64,
        target_id: u32,
    },
    ProjectileArrived {
        tick: u64,
    },
    DuelStarted {
        tick: u64,
        unit_a: u32,
        unit_b: u32,
    },
    DuelEnded {
        tick: u64,
        unit_a: u32,
        unit_b: u32,
    },
    EffectResisted {
        tick: u64,
        unit_id: u32,
        resisted_tag: String,
    },
    UnitSummoned {
        tick: u64,
        unit_id: u32,
        template: String,
    },
    UnitResurrected {
        tick: u64,
        unit_id: u32,
    },
    AttackMissed {
        tick: u64,
        source_id: u32,
        target_id: u32,
    },
    ReflectDamage {
        tick: u64,
        source_id: u32,
        target_id: u32,
        amount: i32,
    },
    LifestealHeal {
        tick: u64,
        unit_id: u32,
        amount: i32,
    },
    DispelApplied {
        tick: u64,
        unit_id: u32,
        removed_count: u32,
    },
    SwapPerformed {
        tick: u64,
        unit_a: u32,
        unit_b: u32,
    },
    PullApplied {
        tick: u64,
        source_id: u32,
        target_id: u32,
    },
    ChainBounce {
        tick: u64,
        source_id: u32,
        target_id: u32,
        bounce_num: u32,
    },
    RewindApplied {
        tick: u64,
        unit_id: u32,
    },
    ExecuteTriggered {
        tick: u64,
        source_id: u32,
        target_id: u32,
    },
    DeathMarkDetonated {
        tick: u64,
        unit_id: u32,
        damage: i32,
    },
    // --- Zone events ---
    ZoneCreated {
        tick: u64,
        zone_id: u32,
        source_id: u32,
    },
    ZoneTick {
        tick: u64,
        zone_id: u32,
    },
    ZoneExpired {
        tick: u64,
        zone_id: u32,
    },
    // --- Channel events ---
    ChannelStarted {
        tick: u64,
        unit_id: u32,
        ability_name: String,
    },
    ChannelTick {
        tick: u64,
        unit_id: u32,
    },
    ChannelInterrupted {
        tick: u64,
        unit_id: u32,
    },
    ChannelCompleted {
        tick: u64,
        unit_id: u32,
    },
    // --- Tether events ---
    TetherFormed {
        tick: u64,
        source_id: u32,
        target_id: u32,
    },
    TetherBroken {
        tick: u64,
        source_id: u32,
        target_id: u32,
    },
    TetherCompleted {
        tick: u64,
        source_id: u32,
        target_id: u32,
    },
    // --- Stack events ---
    StacksApplied {
        tick: u64,
        unit_id: u32,
        name: String,
        count: u32,
    },
    /// Emitted when a conditional effect resolves with its condition met.
    ConditionalEffectApplied {
        tick: u64,
        unit_id: u32,
        condition: String,
    },
    /// Emitted when two tagged zones overlap and produce a combo reaction.
    ZoneReaction {
        tick: u64,
        source_id: u32,
        tag_a: String,
        tag_b: String,
        combo_name: String,
    },
}
