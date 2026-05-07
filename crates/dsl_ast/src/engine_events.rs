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
    }

    #[test]
    fn unknown_user_names_fall_through_to_none() {
        assert_eq!(engine_event_kind_id_for_name("Tick"), None);
        assert_eq!(engine_event_kind_id_for_name("MyCustomEvent"), None);
        assert_eq!(engine_event_kind_id_for_name(""), None);
    }
}
