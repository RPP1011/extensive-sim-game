//! Engine-name → `EventKindId` aliasing table.
//!
//! The downstream engine defines a closed set of replayable event
//! kinds at hardcoded discriminants — see
//! `crates/engine/src/cascade/handler.rs::EventKindId`. The
//! `apply_ability` dispatcher (and other engine-side emitters) write
//! chronicle records tagged with those hardcoded discriminants. When
//! a `.sim` declares an event with one of those reserved engine
//! names (e.g. `event EffectDamageApplied { ... }`), the resulting
//! kernel filter constant must match the engine discriminant — NOT
//! the .sim's source-order index — or downstream consumers never see
//! the records.
//!
//! This table is the single source of truth shared between the
//! resolver (which assigns the kind id to each `EventIR` during the
//! second resolve pass) and the lowering driver (which mirrors the
//! resolver's assignment when registering payload layouts). Adding a
//! new alias is a one-edit-here change.
//!
//! ## Mirrors `EFFECT_KIND_TO_EVENT_KIND_ID`
//!
//! The 7 chronicle-bearing engine events (Damage, Heal, Shield, Stun,
//! Slow, GoldTransfer, StandingDelta) are the exact set the dispatcher
//! kernel writes today via the `EFFECT_KIND_TO_EVENT_KIND_ID` table at
//! `crates/dsl_compiler/src/cg/emit/wgsl_body.rs`. If that table grows
//! a new effect → event mapping, this table must grow the matching
//! entry too — otherwise the consumer/dispatcher kind tags drift again.
//!
//! ## Collision policy
//!
//! User-declared events that don't match a known engine name keep their
//! sequential `EventKindId(i)` (where `i` is the position in
//! `comp.events`). For now there is no `.sim` with enough user events
//! to risk a sequential id colliding with an aliased engine
//! discriminant (the smallest engine alias is 26). If a future sim
//! pushes past ~25 user events and risks collisions, the resolver
//! could be adjusted to skip aliased ids when assigning sequentials.

/// All chronicle-bearing engine event names, paired with their
/// hardcoded `EventKindId` discriminant. Mirrors
/// `EventKindId::Effect*` in `crates/engine/src/cascade/handler.rs` and
/// `EFFECT_KIND_TO_EVENT_KIND_ID` in
/// `crates/dsl_compiler/src/cg/emit/wgsl_body.rs`.
pub const ENGINE_EVENT_KIND_IDS: &[(&str, u32)] = &[
    ("EffectDamageApplied",     26),
    ("EffectHealApplied",       27),
    ("EffectShieldApplied",     28),
    ("EffectStunApplied",       29),
    ("EffectSlowApplied",       30),
    ("EffectGoldTransfer",      31),
    ("EffectStandingDelta",     32),
    // Bleed verb swap (Task #138 follow-on, 2026-05-06): SelfDamage
    // = 17 → EventKindId::EffectSelfDamageApplied = 39 (slot 39, after
    // RallyCall=38 in the engine's `EventKindId` enum).
    ("EffectSelfDamageApplied", 39),
    // Vampirize verb swap (Task #138 follow-on, mirror of Bleed at
    // `486eb08f`): LifeSteal = 18 → EventKindId::EffectLifeStealApplied
    // = 40 (slot 40, after EffectSelfDamageApplied=39).
    ("EffectLifeStealApplied",  40),
    // Fortify verb swap (Task #138 follow-on, mirror of Vampirize at
    // `60115f64`): DamageModify = 19 → EventKindId::EffectDamageModifyApplied
    // = 41 (slot 41, after EffectLifeStealApplied=40).
    ("EffectDamageModifyApplied", 41),
    // Reap verb swap (Task #138 follow-on, mirror of Fortify at
    // `001ae9a6`): Execute = 16 → EventKindId::EffectExecuteApplied = 42
    // (slot 42, after EffectDamageModifyApplied=41). Closes the slice
    // across all 8 duel_abilities verbs.
    ("EffectExecuteApplied", 42),
    // Wave 2 piece 1 — control statuses (Root/Silence/Fear/Taunt). Each
    // mirrors Stun's shape (target agent + u32 expires_at_tick), packed
    // into the 4-payload chronicle record at slots 43..46 contiguous
    // with the Execute=42 wire-up. No consumer rule in any sim today;
    // the dispatcher write end-to-end works without a fold consumer.
    ("EffectRootApplied",    43),
    ("EffectSilenceApplied", 44),
    ("EffectFearApplied",    45),
    ("EffectTauntApplied",   46),
    // Wave 2 piece 2 — movement EffectOps (Dash/Blink/Knockback/Pull).
    // Dash and Blink are caster-self motion: payload is actor + f32
    // distance (no target). Knockback and Pull are forced motion on a
    // target: payload is actor + target + f32 distance. Slots 47..50
    // are contiguous with the Wave 2 piece 1 control statuses (43..46).
    // No consumer rule in any sim today; the dispatcher write
    // end-to-end works without a fold consumer.
    ("EffectDashApplied",      47),
    ("EffectBlinkApplied",     48),
    ("EffectKnockbackApplied", 49),
    ("EffectPullApplied",      50),
    // Wave 1.5+ — multi-tick effects (DamageOverTime/HealOverTime/
    // TimedShield). DoT/HoT carry actor + target + amount-per-tick
    // (f32) + duration_ticks (u32). TimedShield has the same payload
    // shape with `amount` as the one-shot shield magnitude. Slots
    // 51..53 are contiguous with the Wave 2 piece 2 movement EffectOps
    // (47..50). The cast records the magnitude + duration once; a
    // future consumer rule will re-emit per-tick damage/heal events
    // (deferred — no sim re-emits these today).
    ("EffectDamageOverTimeApplied", 51),
    ("EffectHealOverTimeApplied",   52),
    ("EffectTimedShieldApplied",    53),
    // Extended-corpus statuses (Stealth/Charm/Grounded/Suppress).
    // Stealth is caster-self stealth: payload is actor + duration_ticks
    // (no target field on engine event). Charm/Grounded/Suppress are
    // target-cast: payload is actor + target + duration_ticks. All four
    // store raw `duration_ticks` (rather than expires_at_tick) — same
    // convention as the multi-tick effect shapes (51..53). Slots 54..57
    // are contiguous with the Wave 1.5+ multi-tick effects (51..53). No
    // consumer rule in any sim today; the dispatcher write end-to-end
    // works without a fold consumer.
    ("EffectStealthApplied",  54),
    ("EffectCharmApplied",    55),
    ("EffectGroundedApplied", 56),
    ("EffectSuppressApplied", 57),
    // Slice γ tail — Buff/Harvest/PlaceVoxel/Reflect. Buff and Reflect
    // carry signed packed payloads (`magnitude_q8 i16` and `fraction_q8
    // i16`); the chronicle ring stores raw u32 payload words and
    // consumers sign-extend on read. Harvest and PlaceVoxel are caster-
    // self (no target field) — Harvest carries `kind_hash` + `amount`,
    // PlaceVoxel carries `kind_hash` only (placement position is implicit
    // from the cast's target world position). Slots 58..61 are
    // contiguous with the extended-status block (54..57). No consumer
    // rule in any sim today; the dispatcher write end-to-end works
    // without a fold consumer. Summon (kind 24) is the only remaining
    // `// TODO slice γ` arm — its multi-spawn semantics need a new
    // dispatch shape and is deferred.
    ("EffectBuffApplied",       58),
    ("EffectHarvestApplied",    59),
    ("EffectPlaceVoxelApplied", 60),
    ("EffectReflectApplied",    61),
    // Slice γ closer — Summon (kind 24 → ID 62), the last `// TODO
    // slice γ` placeholder. Caster-self with packed payload — actor
    // + template_hash + count (u8 widened to u32) + lifetime_ticks.
    // The CPU side writes ONE `ApplyEvent::Summon` per cast carrying
    // the packed (count, lifetime); downstream N-entity spawning is
    // a separate consumer concern, distinct from the dispatcher's
    // single-record write. Slot 62 contiguous with the slice γ tail
    // block (58..61). No consumer rule in any sim today; the
    // dispatcher write end-to-end works without a fold consumer.
    ("EffectSummonApplied",     62),
    // Wave 3 ToM Phase 1 — `plant_belief` bit-flag primitive. Caster
    // CAUSES target's belief map for `subject_idx` to gain `1u <<
    // fact_bit` via atomic-OR. 5-payload-word chronicle record (actor
    // + target + subject_idx + fact_bit_mask). Slot 63 contiguous with
    // the slice γ closer (Summon=62). The dispatcher writes the
    // chronicle record from the per-effect-slot arm chain when
    // `EffectOp::PlantBelief` (kind=32) fires; downstream view
    // consumers fold the bit mask into a `pair_map` cell via the
    // existing `view ... -> u32 { on EffectPlantBeliefApplied { ... }
    // { self |= b } }` shape (same as `tom_probe.sim::beliefs`). The
    // full Wave 3 multi-field BeliefState (creature_type / decay /
    // disguise / slander) is deferred.
    ("EffectPlantBeliefApplied", 63),
    // Wave 3 ToM Phase 3 — `observe` self-observe-target verb. Caster
    // refreshes its own belief row about `target`. 4-payload-word
    // chronicle record (actor + target + tick + target_observer u8 in
    // payload_a). Slot 64 contiguous with the Wave 3 Phase 1
    // plant_belief slot (EffectPlantBeliefApplied=63). The dispatcher
    // writes the chronicle record from the per-effect-slot arm chain
    // when `EffectOp::Observe` (kind=33) fires; downstream runtime
    // consumers read target's current pos / creature_type from the
    // agent SoA at consume tick and write the BeliefState SoA's 6
    // columns at `[actor * agent_cap + target]` indexing. The DSL
    // view-call lowering shape (`agents.beliefs_<field>(observer,
    // subject)`) for kernel-side reads is deferred to Phase 4
    // alongside the deception verbs (disguise / decoy / erase_belief)
    // and the spy_network sim.
    ("EffectObserveApplied", 64),
    // Wave 3 ToM Phase 3.5 — `scry` cross-observer access. Caster reads
    // agent C's beliefs about subject B (via C as `target_observer`),
    // writes into A's beliefs about B. 5-payload-word chronicle record
    // (actor + subject (= target_slot) + target_observer u8 in payload_a
    // + subject_idx u32 in payload_b). Slot 65 contiguous with the
    // observe slot (EffectObserveApplied=64). The dispatcher writes the
    // chronicle record from the per-effect-slot arm chain when
    // `EffectOp::Scry` (kind=34) fires; downstream runtime consumers copy
    // the 6 BeliefState columns from `[target_observer * N + subject]` to
    // `[caster * N + subject]`.
    ("EffectScryApplied", 65),
    // Wave 3 ToM Phase 3.5 — `reveal` one-to-many propagation. Caster
    // broadcasts its beliefs about `subject` to all observers. 4-payload-
    // word chronicle record (actor + subject (= target_slot) +
    // subject_idx u32 in payload_a). Slot 66 contiguous with the scry
    // slot (65). The dispatcher writes the chronicle record when
    // `EffectOp::Reveal` (kind=35) fires; downstream runtime consumers
    // iterate every observer slot and copy the 6 BeliefState columns from
    // `[caster * N + subject]` to `[observer * N + subject]`.
    ("EffectRevealApplied", 66),
    // Wave 3 ToM Phase 4 — deception verbs (Disguise/Decoy/EraseBelief).
    // Each is the chronicle counterpart of the matching `EffectOp` slot
    // (kinds 36/37/38). The dispatcher writes a single record per cast;
    // downstream BeliefState SoA mutation lives in compiler-emitted
    // `physics @phase(post)` consumer rules in `tom_probe.sim` (mirror
    // of Phase 3.8 observe/scry/reveal authoring).
    //
    // Disguise: caster + (duration_ticks<<8 | fake_type) packed in
    // payload_a. The downstream consumer writes per-agent
    // `disguise_expires_at_tick` and `disguise_fake_type` SoA columns.
    //
    // Decoy: caster + target + subject_idx (= payload_a) + fake_pos
    // (= payload_b, packed quartet). Consumer writes target's row about
    // subject_idx with attacker-controlled values.
    //
    // EraseBelief: caster + target + subject_idx (= payload_a) + fields
    // bitset (= payload_b low byte). Consumer clears specific cells of
    // target's beliefs about subject_idx per the bitset.
    ("EffectDisguiseApplied",    67),
    ("EffectDecoyApplied",       68),
    ("EffectEraseBeliefApplied", 69),
    // Lift A — multi-tick procedure: TravelTo. Dispatcher writes one
    // chronicle record (kind=70) per cast. Payload layout: actor (slot 2)
    // = caster, target (slot 3) = caster (self-cast); slot 4 packs
    // (dest_y_q8 << 16) | (dest_x_q8 & 0xFFFF) — sign-extend each i16
    // half via bit shifts; slot 5 = eta_ticks. The downstream consumer
    // rule sets `busy_until_tick = world.tick + eta_ticks` and
    // populates `travel_dest_{x,y,z}` SoA cells.
    ("EffectTravelToApplied", 70),
    // Lift B — items / inventory + production / recipes. Two chronicle
    // events per cast pair:
    //   * Recipe (kind=71): slot 4 packs (target_tool << 16) | recipe_id;
    //     slot 5 = 0. Self-cast: target = caster (recipes act on the
    //     caster's inventory).
    //   * WearTool (kind=72): slot 4 packs (amount << 8) | tool_kind;
    //     slot 5 = 0. Self-cast: target = caster (wear acts on the
    //     caster's owned tool).
    ("EffectRecipeApplied",   71),
    ("EffectWearToolApplied", 72),
    // Lift C — bilateral consent + observer fan-out:
    //   * Propose (kind=73): slot 4 = contract_kind (low byte); slot 5 =
    //     expires_at_tick (0 = no expiry). target = recipient.
    //   * Announce (kind=74): slot 4 packs (radius_q8 << 8) |
    //     announcement_kind; slot 5 = 0. Self-origin: target = caster.
    ("EffectProposeApplied",  73),
    ("EffectAnnounceApplied", 74),
    // Lift D — knowledge / skills + obligation registry:
    //   * GainSkill (kind=75): slot 4 packs (amount_q8 << 8) | skill_id;
    //     slot 5 = 0. Self-cast: target = caster.
    //   * CreateObligation (kind=76): slot 4 packs (kind << 16) |
    //     obligation_id; slot 5 = 0. target = debtor / promisor.
    ("EffectGainSkillApplied",        75),
    ("EffectCreateObligationApplied", 76),
];

/// Look up the engine-defined `EventKindId` discriminant for an
/// event-declaration name. Returns `Some(id)` for the closed set of
/// chronicle-bearing engine events; `None` for any user-defined event
/// (which uses sequential allocation in
/// `dsl_compiler::cg::lower::driver::populate_event_kinds` /
/// `dsl_ast::resolve::resolve`).
pub fn engine_event_kind_id_for_name(name: &str) -> Option<u32> {
    ENGINE_EVENT_KIND_IDS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, id)| *id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_engine_names_resolve_to_engine_discriminants() {
        assert_eq!(engine_event_kind_id_for_name("EffectDamageApplied"), Some(26));
        assert_eq!(engine_event_kind_id_for_name("EffectHealApplied"), Some(27));
        assert_eq!(engine_event_kind_id_for_name("EffectShieldApplied"), Some(28));
        assert_eq!(engine_event_kind_id_for_name("EffectStunApplied"), Some(29));
        assert_eq!(engine_event_kind_id_for_name("EffectSlowApplied"), Some(30));
        assert_eq!(engine_event_kind_id_for_name("EffectGoldTransfer"), Some(31));
        assert_eq!(engine_event_kind_id_for_name("EffectStandingDelta"), Some(32));
        assert_eq!(engine_event_kind_id_for_name("EffectSelfDamageApplied"), Some(39));
        assert_eq!(engine_event_kind_id_for_name("EffectLifeStealApplied"), Some(40));
        assert_eq!(engine_event_kind_id_for_name("EffectDamageModifyApplied"), Some(41));
        assert_eq!(engine_event_kind_id_for_name("EffectExecuteApplied"), Some(42));
        assert_eq!(engine_event_kind_id_for_name("EffectRootApplied"), Some(43));
        assert_eq!(engine_event_kind_id_for_name("EffectSilenceApplied"), Some(44));
        assert_eq!(engine_event_kind_id_for_name("EffectFearApplied"), Some(45));
        assert_eq!(engine_event_kind_id_for_name("EffectTauntApplied"), Some(46));
        assert_eq!(engine_event_kind_id_for_name("EffectDashApplied"), Some(47));
        assert_eq!(engine_event_kind_id_for_name("EffectBlinkApplied"), Some(48));
        assert_eq!(engine_event_kind_id_for_name("EffectKnockbackApplied"), Some(49));
        assert_eq!(engine_event_kind_id_for_name("EffectPullApplied"), Some(50));
        assert_eq!(engine_event_kind_id_for_name("EffectDamageOverTimeApplied"), Some(51));
        assert_eq!(engine_event_kind_id_for_name("EffectHealOverTimeApplied"), Some(52));
        assert_eq!(engine_event_kind_id_for_name("EffectTimedShieldApplied"), Some(53));
        assert_eq!(engine_event_kind_id_for_name("EffectStealthApplied"), Some(54));
        assert_eq!(engine_event_kind_id_for_name("EffectCharmApplied"), Some(55));
        assert_eq!(engine_event_kind_id_for_name("EffectGroundedApplied"), Some(56));
        assert_eq!(engine_event_kind_id_for_name("EffectSuppressApplied"), Some(57));
        assert_eq!(engine_event_kind_id_for_name("EffectBuffApplied"), Some(58));
        assert_eq!(engine_event_kind_id_for_name("EffectHarvestApplied"), Some(59));
        assert_eq!(engine_event_kind_id_for_name("EffectPlaceVoxelApplied"), Some(60));
        assert_eq!(engine_event_kind_id_for_name("EffectReflectApplied"), Some(61));
        assert_eq!(engine_event_kind_id_for_name("EffectSummonApplied"), Some(62));
        assert_eq!(engine_event_kind_id_for_name("EffectPlantBeliefApplied"), Some(63));
        assert_eq!(engine_event_kind_id_for_name("EffectObserveApplied"), Some(64));
        assert_eq!(engine_event_kind_id_for_name("EffectScryApplied"), Some(65));
        assert_eq!(engine_event_kind_id_for_name("EffectRevealApplied"), Some(66));
        assert_eq!(engine_event_kind_id_for_name("EffectDisguiseApplied"), Some(67));
        assert_eq!(engine_event_kind_id_for_name("EffectDecoyApplied"), Some(68));
        assert_eq!(engine_event_kind_id_for_name("EffectEraseBeliefApplied"), Some(69));
        assert_eq!(engine_event_kind_id_for_name("EffectTravelToApplied"), Some(70));
        assert_eq!(engine_event_kind_id_for_name("EffectRecipeApplied"), Some(71));
        assert_eq!(engine_event_kind_id_for_name("EffectWearToolApplied"), Some(72));
        assert_eq!(engine_event_kind_id_for_name("EffectProposeApplied"), Some(73));
        assert_eq!(engine_event_kind_id_for_name("EffectAnnounceApplied"), Some(74));
        assert_eq!(engine_event_kind_id_for_name("EffectGainSkillApplied"), Some(75));
        assert_eq!(engine_event_kind_id_for_name("EffectCreateObligationApplied"), Some(76));
    }

    #[test]
    fn unknown_user_names_fall_through_to_none() {
        assert_eq!(engine_event_kind_id_for_name("Tick"), None);
        assert_eq!(engine_event_kind_id_for_name("MyCustomEvent"), None);
        assert_eq!(engine_event_kind_id_for_name(""), None);
    }
}
