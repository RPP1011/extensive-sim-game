use crate::event::{EventLike, EventRing};
use crate::state::SimState;

pub mod __sealed {
    pub trait Sealed {}

    /// Marker trait emitted by `dsl_compiler` next to every compiler-generated
    /// rule type. Combined with the blanket `Sealed` impl below, only types
    /// that go through the DSL compiler satisfy the `Sealed` supertrait of
    /// `CascadeHandler` / `MaterializedView` / `LazyView` / `TopKView`.
    ///
    /// This lives in the same module as `Sealed` so the blanket impl
    /// `impl<T: GeneratedRule> Sealed for T` is coherent (both traits are
    /// local).
    #[doc(hidden)]
    pub trait GeneratedRule {}

    impl<T: GeneratedRule> Sealed for T {}
}

/// Stable ordinal identifying an event variant. Dense so it indexes arrays
/// cheaply. Adding a variant appends; reordering is a schema-hash bump.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum EventKindId {
    AgentMoved           = 0,
    AgentAttacked        = 1,
    AgentDied            = 2,
    AgentFled            = 3,
    AgentAte             = 4,
    AgentDrank           = 5,
    AgentRested          = 6,
    AgentCast            = 7,
    AgentUsedItem        = 8,
    AgentHarvested       = 9,
    AgentPlacedTile      = 10,
    AgentPlacedVoxel     = 11,
    AgentHarvestedVoxel  = 12,
    AgentConversed       = 13,
    AgentSharedStory     = 14,
    AgentCommunicated    = 15,
    InformationRequested = 16,
    AgentRemembered      = 17,
    QuestPosted          = 18,
    QuestAccepted        = 19,
    BidPlaced            = 20,
    AnnounceEmitted      = 21,
    RecordMemory         = 22,
    // Slots 23-24 retired in task 143 along with StunExpired / SlowExpired.
    // Stun / slow expiry is now a synthetic boundary read off
    // `stun_expires_at_tick` / `slow_expires_at_tick`, no event emitted.
    OpportunityAttackTriggered = 25,
    // Combat Foundation Task 9 — effect fan-out + recursion-audit events.
    EffectDamageApplied  = 26,
    EffectHealApplied    = 27,
    EffectShieldApplied  = 28,
    EffectStunApplied    = 29,
    EffectSlowApplied    = 30,
    EffectGoldTransfer   = 31,
    EffectStandingDelta  = 32,
    CastDepthExceeded    = 33,
    // Task 139 — engagement transition events replacing the retired
    // `tick_start` tentative-commit loop.
    EngagementCommitted  = 34,
    EngagementBroken     = 35,
    // Task 167 — fear-spread fan-out from `AgentDied`. One emit per
    // nearby same-species kin; folded by `kin_fear` materialized view.
    FearSpread           = 36,
    // Task 169 — pack-focus fan-out from `EngagementCommitted`. One
    // emit per nearby same-species kin; folded by `pack_focus`
    // materialized view.
    PackAssist           = 37,
    // Task 178 — rally fan-out from `AgentAttacked` on a wounded
    // (alive + hp_pct < 0.5) victim. One emit per nearby same-species
    // kin; folded by `rally_boost` materialized view.
    RallyCall            = 38,
    // Task #138 follow-on (Bleed) — chronicle event for SelfDamage
    // EffectOp (op#17). Written by the apply_ability dispatcher when
    // an `EffectOp::SelfDamage` slot fires; the runtime
    // `ApplyDamageFromSelfDamageChronicle` re-emit physics rule
    // translates these records back into the existing `Damaged`
    // event so the rest of the cascade (shield absorption,
    // lifesteal, damage-modify) keeps working unchanged. Same shape
    // as `EffectDamageApplied` (actor + target + amount + tick) —
    // for self-damage, actor == target by convention but the field
    // is preserved for uniformity with the other Effect* events.
    EffectSelfDamageApplied = 39,
    // Vampirize verb swap (Task #138 follow-on, mirror of Bleed at
    // `486eb08f`) — chronicle event for LifeSteal EffectOp (op#18).
    // Written by the apply_ability dispatcher when an
    // `EffectOp::LifeSteal` slot fires; the runtime
    // `ApplyLifestealFromChronicle` re-emit physics rule translates
    // these records back into the existing `SetLifesteal` event so
    // the rest of the cascade (ApplyLifestealActivation writing the
    // per-agent lifesteal_frac_q8 + lifesteal_expires_at_tick SoA
    // fields, then ApplyDamage's source lookup healing the source
    // for `frac_q8/256 * bleed-through-damage` per Damaged event)
    // keeps working unchanged. Same shape as `EffectSlowApplied`
    // (actor + target + expires_at_tick + fraction_q8 + tick) — for
    // self-cast LifeSteal actor == target == caster by convention.
    EffectLifeStealApplied = 40,
    // Fortify verb swap (Task #138 follow-on, mirror of Vampirize at
    // `60115f64`) — chronicle event for DamageModify EffectOp (op#19).
    // Written by the apply_ability dispatcher when an
    // `EffectOp::DamageModify` slot fires; the runtime
    // `ApplyDamageModFromChronicle` re-emit physics rule translates
    // these records back into the existing `SetDamageMod` event so
    // the rest of the cascade (ApplyDamageModActivation writing the
    // per-agent damage_taken_mult_q8 + damage_taken_mult_expires_at_tick
    // SoA fields, then ApplyDamage scaling incoming damage by
    // `mult_q8/256` while the buff is active) keeps working unchanged.
    // Same shape as `EffectSlowApplied` / `EffectLifeStealApplied`
    // (actor + target + expires_at_tick + multiplier_q8 + tick) — for
    // self-cast DamageModify actor == target == caster by convention.
    EffectDamageModifyApplied = 41,
    // Reap verb swap (Task #138 follow-on, mirror of Fortify at
    // `001ae9a6`) — chronicle event for Execute EffectOp (op#16).
    // Written by the apply_ability dispatcher when an
    // `EffectOp::Execute` slot fires; the runtime
    // `ApplyExecuteFromChronicle` re-emit physics rule translates
    // these records back into the existing `Executed` event (which the
    // duel_abilities runtime drains via ApplyDefeat / inline ApplyDamage
    // hp<=0 path, depending on the host sim) so the rest of the cascade
    // keeps working unchanged. Closes the slice across all 8
    // duel_abilities verbs.
    //
    // SHAPE NOTE: 3-payload-word event (actor + target + hp_threshold).
    // Same shape family as `EffectDamageApplied` (actor + target + amount)
    // — the dispatcher writes caster_slot into actor (slot 2) and target
    // into slot 3, with payload word 4 carrying hp_threshold as a bitcast
    // f32. The when-condition `target.hp < hp_threshold` is NOT evaluated
    // by the GPU dispatcher (apply_program doesn't consult
    // when_per_effect today — registry-driven predicate dispatch is
    // later infrastructure). The duel_abilities Reap verb's outer `when`
    // clause already gates emission on `target.hp <
    // config.combat.reap_threshold`, so the unconditional dispatcher
    // write is gated upstream and the consumer rule can ferry the record
    // directly into Defeated.
    EffectExecuteApplied = 42,
    // Wave 2 piece 1 — control statuses (Root/Silence/Fear/Taunt).
    // Each is a chronicle event for the matching `EffectOp` (op#7..10).
    // Same shape as `EffectStunApplied` (actor + target + expires_at_tick
    // + tick): a target agent and a u32 expiry deadline. The
    // `apply_ability` dispatcher writes these from the per-effect-slot
    // arm chain when the corresponding `EffectOp` slot fires; consumer
    // physics rules can fold them back into per-agent
    // `*_expires_at_tick` SoA fields the same way the Stun consumer
    // does (consumer wiring in duel_abilities.sim is deferred — no sim
    // currently uses these statuses).
    EffectRootApplied    = 43,
    EffectSilenceApplied = 44,
    EffectFearApplied    = 45,
    EffectTauntApplied   = 46,
    // Slots 47-127 reserved for replayable event variants added in later tasks.
    ChronicleEntry       = 128,
}


/// Lane discipline — handlers within a lane run in registration order;
/// lanes run in the order listed here.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum Lane {
    Validation = 0,
    Effect     = 1,
    Reaction   = 2,
    Audit      = 3,
}

impl Lane {
    pub const ALL: &'static [Lane] = &[
        Lane::Validation, Lane::Effect, Lane::Reaction, Lane::Audit,
    ];
}

pub trait CascadeHandler<E: EventLike>: __sealed::Sealed + Send + Sync {
    /// The "views" type this handler expects alongside the mutable state.
    /// Engine-rules handlers set this to `engine_rules::ViewRegistry`.
    /// Test-only handlers that don't use views set this to `()`.
    type Views;

    fn trigger(&self) -> EventKindId;
    fn lane(&self) -> Lane { Lane::Effect }
    fn handle(&self, event: &E, state: &mut SimState, views: &mut Self::Views, events: &mut EventRing<E>);

    /// Downcast hook so registries can look up the concrete handler type
    /// (e.g. `CastHandler`) to expose handler-specific state
    /// (`AbilityRegistry`). Default impl returns `None`; concrete handlers
    /// that want to be discoverable override with `Some(self)`.
    fn as_any(&self) -> Option<&dyn std::any::Any> { None }
}
