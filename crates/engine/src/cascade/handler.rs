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
    // Wave 2 piece 2 — movement EffectOps (Dash/Blink/Knockback/Pull).
    // Each is a chronicle event for the matching `EffectOp` (op#12..15).
    // Dash and Blink are caster-self motion: payload carries the actor +
    // f32 distance (no target). Knockback and Pull are forced motion on
    // a target: payload carries actor + target + f32 distance. The
    // `apply_ability` dispatcher writes these from the per-effect-slot
    // arm chain when the corresponding `EffectOp` slot fires; consumer
    // physics rules can fold the distance into per-agent position
    // updates the same way other effect chronicle records flow into
    // SoA-fold rules (consumer wiring deferred — no sim currently uses
    // these movement effects).
    EffectDashApplied    = 47,
    EffectBlinkApplied   = 48,
    EffectKnockbackApplied = 49,
    EffectPullApplied    = 50,
    // Wave 1.5+ — multi-tick effect EffectOps (DamageOverTime/HealOverTime/
    // TimedShield). Each is a chronicle event for the matching `EffectOp`
    // (op#20..22). DoT/HoT carry an actor + target + per-tick amount + a
    // u32 duration_ticks (the cast records the magnitude + window once;
    // a future consumer rule will re-emit per-tick damage/heal events).
    // TimedShield carries actor + target + total shield amount + duration
    // (the shield is applied once on cast and persists for the window).
    // The `apply_ability` dispatcher writes these from the per-effect-slot
    // arm chain when the corresponding `EffectOp` slot fires; consumer
    // physics rules can fold them into per-agent multi-tick state the
    // same way other effect chronicle records flow into SoA-fold rules
    // (consumer wiring deferred — no sim currently re-emits multi-tick
    // damage/heal/shield events).
    EffectDamageOverTimeApplied = 51,
    EffectHealOverTimeApplied   = 52,
    EffectTimedShieldApplied    = 53,
    // Extended-corpus statuses (Stealth/Charm/Grounded/Suppress).
    // Stealth is caster-self (the actor goes invisible, no target field
    // on the engine event); Charm/Grounded/Suppress mirror Stun's shape
    // (actor + target + duration_ticks) — apply to a target agent for
    // a deadline window. The `apply_ability` dispatcher writes these
    // from the per-effect-slot arm chain when the corresponding
    // `EffectOp` slot fires; consumer physics rules can fold them back
    // into per-agent state the same way other effect chronicle records
    // flow into SoA-fold rules (consumer wiring deferred — no sim
    // currently uses these statuses).
    //
    // SHAPE NOTE: Stealth records carry only `duration_ticks` (no
    // expires_at_tick fold by the dispatcher) — symmetric with the
    // multi-tick effect shapes (DoT/HoT/TimedShield, kinds 51..53)
    // which also store raw `duration_ticks` for a future consumer to
    // re-emit per-tick. Charm/Grounded/Suppress also store raw
    // `duration_ticks` (rather than `expires_at_tick`) — same convention,
    // distinct from the legacy Stun/Slow/Root/Silence/Fear/Taunt arms
    // (kinds 29/30/43..46) which fold the deadline at dispatch time.
    EffectStealthApplied    = 54,
    EffectCharmApplied      = 55,
    EffectGroundedApplied   = 56,
    EffectSuppressApplied   = 57,
    // Slice γ tail — Buff/Harvest/PlaceVoxel/Reflect. Each closes one of
    // the remaining `// TODO slice γ` placeholders in the apply_ability
    // dispatcher's WGSL arm chain. Four distinct shapes:
    //   - Buff (kind 23 → ID 58): target-cast with packed payload.
    //     5-payload-word chronicle record (actor + target + raw payload_a
    //     + raw payload_b). The dispatcher writes payload_a (which packs
    //     `stat_ordinal` in the low byte and `magnitude_q8` in bits 8..)
    //     and payload_b (= duration_ticks) verbatim — consumer rules
    //     decode the packed bits.
    //   - Harvest (kind 25 → ID 59): caster-self resource gather.
    //     4-payload-word chronicle record (actor + kind_hash + amount).
    //     No target field on the engine event; the dispatcher writes
    //     payload_a (= kind_hash u32) and payload_b (= amount u32) at
    //     ring slots 3 and 4.
    //   - PlaceVoxel (kind 26 → ID 60): caster-self voxel placement.
    //     3-payload-word chronicle record (actor + kind_hash). Position
    //     is implicit from the cast's target world position; the engine
    //     event today carries only kind_hash (no position field) per
    //     ApplyEvent::PlaceVoxel's payload shape — apply handlers
    //     reconstruct placement location from the cast's stamped target
    //     position when the runtime spawner wires up.
    //   - Reflect (kind 31 → ID 61): target-cast fraction-of-damage
    //     bounce. 5-payload-word chronicle record (actor + target +
    //     raw payload_a + raw payload_b). The dispatcher writes
    //     payload_a (= duration_ticks) and payload_b (= fraction_q8 i16
    //     packed in low 16 bits) verbatim — consumer rules sign-extend
    //     the fraction. Same convention as Buff: the packed payload
    //     stores raw u32 words for the consumer to decode.
    //
    // SHAPE NOTE: Buff and Reflect carry signed packed payloads
    // (`magnitude_q8 i16` and `fraction_q8 i16`) — the chronicle ring
    // stores the raw u32 from the dispatcher's effect_payload_a /
    // effect_payload_b columns; consumers downcast/sign-extend on read.
    // Harvest and PlaceVoxel store unsigned u32 hashes / counts.
    EffectBuffApplied        = 58,
    EffectHarvestApplied     = 59,
    EffectPlaceVoxelApplied  = 60,
    EffectReflectApplied     = 61,
    // Slice γ closer — Summon (kind 24 → ID 62), the last `// TODO
    // slice γ` placeholder in the apply_ability dispatcher's WGSL arm
    // chain. Caster-self with packed payload — 5-payload-word
    // chronicle record (actor + template_hash + count + lifetime_ticks).
    // The earlier "multi-spawn semantics need a new dispatch shape"
    // deferral was misleading: per `crates/engine/src/ability/apply.rs`
    // the CPU side writes ONE `ApplyEvent::Summon` per cast with
    // packed (count, lifetime); downstream N-entity spawning is a
    // separate consumer concern. The dispatcher writes one record per
    // cast with count + lifetime stored in distinct slots (slot 4 =
    // count widened u8→u32, slot 5 = lifetime_ticks raw u32) — same
    // shape family as DoT/HoT/TimedShield (5 payload words). No
    // target field on the engine event. Position / spawned-entity
    // allocation are downstream consumer concerns, distinct from the
    // dispatcher's record write.
    EffectSummonApplied      = 62,
    // Wave 3 ToM Phase 1 — `plant_belief` bit-flag primitive. Caster
    // CAUSES `target`'s belief map for `subject_idx` to gain
    // `1u << fact_bit` via atomic-OR fold. 5-payload-word chronicle
    // record (actor + target + subject_idx + fact_bit_mask). The
    // dispatcher writes the chronicle record from the per-effect-slot
    // arm chain when the corresponding `EffectOp::PlantBelief` slot
    // fires; downstream view consumers fold the bit mask into a
    // `pair_map`-storage `u32` view via `self |= b` (atomicOr) — same
    // pattern as `tom_probe.sim::beliefs`. The full Wave 3 multi-field
    // BeliefState (creature_type / decay / disguise / slander) is
    // deferred. Slot 63 contiguous with the slice γ closer (Summon=62).
    EffectPlantBeliefApplied = 63,
    // Wave 3 ToM Phase 3 — `observe` self-observe-target verb. Caster
    // refreshes its own belief row about `target`: the chronicle
    // record carries actor + target + tick + target_observer (= u8
    // future-extension hook stored as payload_a). The consumer reads
    // target's current pos / creature_type from the agent SoA at
    // consume tick and writes into the BeliefState SoA's 6 columns at
    // `[actor * agent_cap + target]` indexing. 4-payload-word
    // chronicle record (slots 0..=4 used; slot 4 = target_observer).
    // Slot 64 contiguous with the Wave 3 Phase 1 plant_belief slot
    // (EffectPlantBeliefApplied=63).
    EffectObserveApplied     = 64,
    // Wave 3 ToM Phase 3.5 — `scry` cross-observer access. Caster reads
    // agent C's beliefs about subject B (via C as `target_observer`),
    // writes into caster's beliefs about subject. The chronicle record
    // carries (actor + target=subject + target_observer in payload_a +
    // subject_idx in payload_b). Slot 65 contiguous with the Phase 3
    // observe slot (EffectObserveApplied=64).
    EffectScryApplied        = 65,
    // Wave 3 ToM Phase 3.5 — `reveal` one-to-many propagation. Caster
    // broadcasts its beliefs about `subject` to all observers. The
    // chronicle record carries (actor + target=subject + subject_idx in
    // payload_a). Slot 66 contiguous with the scry slot (65).
    EffectRevealApplied      = 66,
    // Wave 3 ToM Phase 4 — deception verbs (Disguise/Decoy/EraseBelief).
    // Each is the chronicle counterpart of the matching `EffectOp` slot
    // (kinds 36/37/38). The dispatcher writes a single record per cast;
    // downstream BeliefState SoA mutation lives in compiler-emitted
    // `physics @phase(post)` consumer rules in `tom_probe.sim`. Slots
    // 67/68/69 contiguous with the Phase 3.5 reveal slot (66).
    EffectDisguiseApplied    = 67,
    EffectDecoyApplied       = 68,
    EffectEraseBeliefApplied = 69,
    // Lift A — multi-tick procedures + Travel. The dispatcher writes a
    // single chronicle record (kind=70) when an `EffectOp::TravelTo`
    // (op#39) slot fires. The downstream consumer rule sets
    // `busy_until_tick = world.tick + eta_ticks` and populates the
    // `travel_dest_{x,y,z}` SoA cells. A second per-tick PerAgent rule
    // walks alive agents and interpolates `pos` toward the destination
    // by `(dest - pos) / max(1, busy_until_tick - tick)` per tick. On
    // the final tick (`world.tick == busy_until_tick`), the consumer
    // snaps `pos` to the exact destination.
    //
    // SHAPE NOTE: 5-payload-word chronicle record (actor + dest_x_q8 +
    // dest_y_q8 + eta_ticks). The dispatcher writes:
    //   slot 2 = caster_slot (the traveler)
    //   slot 3 = caster_slot (target = caster for self-cast)
    //   slot 4 = packed (dest_y_q8 << 16) | (dest_x_q8 & 0xFFFF) — the
    //            consumer sign-extends each i16 half via bit shifts.
    //   slot 5 = eta_ticks (u32)
    EffectTravelToApplied = 70,
    // Lift B — items / inventory + production / recipes. Two
    // chronicle records per cast pair:
    //   * `EffectRecipeApplied` (kind=71) — fires when an
    //     `EffectOp::Recipe` (op#40) slot fires. Per-fixture consumer
    //     rules read `RecipeRegistry[recipe_id]`, validate the caster's
    //     inventory + tool ownership, and emit ingredient/output deltas.
    //     SHAPE: 4-payload-word record (caster + caster + packed
    //     (target_tool << 16) | recipe_id + 0).
    //   * `EffectWearToolApplied` (kind=72) — fires when an
    //     `EffectOp::WearTool` (op#41) slot fires. Per-fixture consumer
    //     rules look up the caster's tool of `tool_kind` and bump its
    //     wear cell by `amount` (q8 fraction-of-durability). SHAPE:
    //     4-payload-word record (caster + caster + packed (amount << 8)
    //     | tool_kind + 0).
    // See `docs/spec/economy.md §4.1` (recipes) + §4.3 (capital goods).
    EffectRecipeApplied   = 71,
    EffectWearToolApplied = 72,
    // Lift C — bilateral consent + observer fan-out. Two chronicle
    // records per cast pair:
    //   * `EffectProposeApplied` (kind=73) — fires when an
    //     `EffectOp::Propose` (op#42) slot fires. Per-fixture consumer
    //     rules register the proposal in a ContractRegistry keyed on
    //     (caster, target, contract_kind), to be resolved when the
    //     target later fires the companion accept / decline verb
    //     (separate slice). SHAPE: 5-payload-word record (caster +
    //     target + packed contract_kind in low 8 bits + expires_at_tick
    //     u32 in payload_b).
    //   * `EffectAnnounceApplied` (kind=74) — fires when an
    //     `EffectOp::Announce` (op#43) slot fires. Per-fixture consumer
    //     rules walk the spatial hash within `radius_q8` of the caster
    //     and emit per-observer perception events. SHAPE: 4-payload-word
    //     record (caster + caster + packed (radius_q8 << 8) |
    //     announcement_kind + 0).
    // See `docs/spec/economy.md §6` (observer fan-out) + §7 (contracts).
    EffectProposeApplied  = 73,
    EffectAnnounceApplied = 74,
    // Lift D — knowledge / skills + obligation registry. Two chronicle
    // records per cast pair:
    //   * `EffectGainSkillApplied` (kind=75) — fires when an
    //     `EffectOp::GainSkill` (op#44) slot fires. Per-fixture consumer
    //     reads the per-agent per-skill SoA column for `skill_id` and
    //     adds `amount_q8 / 256.0`, clamped to [0.0, 1.0]. SHAPE:
    //     4-payload-word record (caster + caster + packed
    //     (amount_q8 << 8) | skill_id + 0).
    //   * `EffectCreateObligationApplied` (kind=76) — fires when an
    //     `EffectOp::CreateObligation` (op#45) slot fires. Per-fixture
    //     consumer registers the obligation in the AggregatePool and
    //     updates per-agent debtor / creditor indices. SHAPE: 4-payload-
    //     word record (caster + target + packed (kind << 16) |
    //     obligation_id + 0).
    // See `docs/spec/economy.md §7` (obligations) + §8 (skills).
    EffectGainSkillApplied         = 75,
    EffectCreateObligationApplied  = 76,
    /// Plan G (2026-05-09) — generic deferred-cast intent. Written
    /// by the dispatcher when an `apply_ability` call resolves to
    /// `EffectOp::CastBegin`. Consumer sets busy SoA columns + emits
    /// CastBegan as the public lifecycle event for replay/viewer.
    EffectCastBeginApplied         = 77,
    /// Plan G — cast lifecycle event when a multi-tick activity
    /// initiates. Source of truth for telegraph display + replay.
    CastBegan                      = 78,
    /// Plan G — cast lifecycle event when the busy state reaches
    /// `busy_until_tick` without being interrupted. The dispatched
    /// resolution effect writes its own per-effect chronicle
    /// records alongside.
    CastResolved                   = 79,
    /// Plan G — cast lifecycle event when an interrupt source
    /// matches the active cast's interrupt set. Carries the kind
    /// of interrupt that fired so AI / viewer can react.
    CastInterrupted                = 80,
    // Slots 81-127 reserved for replayable event variants added in later tasks.
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
