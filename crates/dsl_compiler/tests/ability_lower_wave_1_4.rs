//! Wave 1.4 lowering tests — verify the new body-block surfaces and
//! the two new headers from spec §4.2 / §4.4 / §9 error cleanly at the
//! lowering boundary, and that the Wave 1 corpus
//! (Strike / ShieldUp / Mend) still lowers without regression.
//!
//! Per `crates/dsl_compiler/src/ability_lower.rs` Wave 1.4 module-level
//! docs, lowering of `recast` / `recast_window` headers and
//! `morph { ... } into <Other>` body blocks still requires engine-side
//! schema work (multi-stage cast state, form-swap state). Until those
//! land, lowering surfaces:
//!   * `LowerError::HeaderNotImplemented { header: "recast" | "recast_window" }`
//!   * `LowerError::MorphBlockNotImplemented { ability, into, span }`
//!
//! Wave 2 piece 5/6 lifted `deliver { ... }` into engine IR — the
//! block now lowers into `Delivery::Method { kind, raw }` with
//! `kind: DeliveryMethodKind` validated against the engine's
//! 6-method vocabulary (projectile/channel/zone/chain/tether/trap).
//! Unknown method idents surface as `LowerError::UnknownDeliveryMethod`.
//!
//! The tests also exercise the spec §4.4 mutual-exclusion rule
//! (deliver + bare effects → `MixedBody`). The parser admits this
//! coexistence; lowering is the enforcer.

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_lower::{lower_ability_decl, LowerError};

/// Helper: parse one inline ability source and return the
/// `LowerError` from `lower_ability_decl`.
fn lower_inline(src: &str) -> LowerError {
    let file = parse_ability_file(src).expect("parser");
    lower_ability_decl(&file.abilities[0]).expect_err("lowering must error")
}

// ---------------------------------------------------------------------------
// 1. `recast:` and `recast_window:` headers — #129 captures into program
// ---------------------------------------------------------------------------
//
// Pre-#129: lowering erroed with `HeaderNotImplemented`, blocking the 39
// LoL files that declare `recast:` / `recast_window:`. Post-#129: the
// header value flows into `AbilityProgram.recast` (`RecastKind::Count`
// or `RecastKind::CooldownTicks`) and `AbilityProgram.recast_window_ticks`.
// Apply semantics (per-agent counter / window timer SoA) still arrive
// alongside the registry-driven dispatch (#125 family); this slice
// only unblocks the lowering.

use engine::ability::program::RecastKind;

fn lower_ok(src: &str) -> engine::ability::program::AbilityProgram {
    let file = parse_ability_file(src).expect("parser");
    lower_ability_decl(&file.abilities[0]).expect("lowering must succeed")
}

#[test]
fn lowering_recast_count_form_captures_count_into_program() {
    let prog = lower_ok("ability X { target: enemy cooldown: 1s recast: 3 damage 10 }");
    assert_eq!(prog.recast, Some(RecastKind::Count(3)));
    assert_eq!(prog.recast_window_ticks, None);
}

#[test]
fn lowering_recast_duration_form_captures_cooldown_ticks() {
    // `recast: 4s` -> RecastKind::CooldownTicks(40) at 100ms-per-tick.
    let prog = lower_ok("ability X { target: enemy cooldown: 1s recast: 4s damage 10 }");
    assert_eq!(prog.recast, Some(RecastKind::CooldownTicks(40)));
}

#[test]
fn lowering_recast_window_captures_ticks() {
    let prog = lower_ok(
        "ability X { target: enemy cooldown: 1s recast_window: 10s damage 10 }",
    );
    assert_eq!(prog.recast_window_ticks, Some(100)); // 10s @ 100ms/tick
}

#[test]
fn lowering_no_recast_keeps_program_fields_none() {
    let prog = lower_ok("ability X { target: enemy cooldown: 1s damage 10 }");
    assert!(prog.recast.is_none());
    assert!(prog.recast_window_ticks.is_none());
}

// ---------------------------------------------------------------------------
// 2. `deliver { ... }` body block
// ---------------------------------------------------------------------------

// Wave 2 piece 5/6 — the deliver block now lowers cleanly into
// `Delivery::Method { kind, raw }` instead of erroring with
// DeliverBlockNotImplemented. The `raw` capture preserves the verbatim
// source slice (params + body) for downstream apply handlers. Apply-
// handler dispatch (projectile travel, channel hold-over-time, etc.)
// is later infrastructure (#125 registry-driven dispatch).
#[test]
fn lowering_deliver_projectile_captures_method_and_raw() {
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::{Delivery, DeliveryMethodKind};
    let src = "ability X {
        target: enemy range: 5.0 cooldown: 1s
        deliver projectile { speed: 16.0 } { on_hit { damage 10 } }
    }";
    let file = parse_ability_file(src).expect("must parse");
    let prog = lower_ability_decl(&file.abilities[0])
        .expect("deliver projectile must lower");
    match &prog.delivery {
        Delivery::Method { kind, raw, hooks } => {
            assert_eq!(*kind, DeliveryMethodKind::Projectile);
            assert!(raw.contains("projectile"), "raw must include method ident: {raw}");
            assert!(raw.contains("on_hit"),     "raw must include body block: {raw}");
            // #139: deliver-body hooks now lower into structured IR.
            // The on_hit { damage 10 } block extracts to a single
            // DeliveryHook with one Damage(10) effect.
            assert_eq!(hooks.len(), 1, "expected one hook (on_hit), got {}", hooks.len());
            assert_eq!(
                hooks[0].kind,
                engine::ability::program::DeliveryHookKind::OnHit,
            );
            assert_eq!(hooks[0].effects.len(), 1, "on_hit damage 10 => 1 effect");
        }
        other => panic!("expected Delivery::Method(Projectile); got {other:?}"),
    }
}

#[test]
fn lowering_deliver_channel_captures_method_and_raw() {
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::{Delivery, DeliveryMethodKind};
    let src = "ability X {
        target: enemy range: 5.0 cooldown: 1s
        deliver channel { duration: 2s, tick: 500ms } { on_tick { damage 7 } }
    }";
    let file = parse_ability_file(src).expect("must parse");
    let prog = lower_ability_decl(&file.abilities[0]).expect("deliver channel must lower");
    match &prog.delivery {
        Delivery::Method { kind, .. } => {
            assert_eq!(*kind, DeliveryMethodKind::Channel);
        }
        other => panic!("expected Delivery::Method(Channel); got {other:?}"),
    }
}

#[test]
fn lowering_deliver_unknown_method_diagnostic() {
    let src = "ability X {
        target: enemy range: 5.0 cooldown: 1s
        deliver bouncepad { speed: 16.0 } { on_hit { damage 10 } }
    }";
    let err = lower_inline(src);
    match err {
        LowerError::UnknownDeliveryMethod { method, .. } => {
            assert_eq!(method, "bouncepad");
        }
        other => panic!("expected UnknownDeliveryMethod(bouncepad); got {other:?}"),
    }
    let msg = lower_inline(src).to_string();
    assert!(msg.contains("bouncepad"), "diagnostic must mention the offending method; got: {msg}");
    assert!(msg.contains("projectile"), "diagnostic must list valid methods; got: {msg}");
}

// ---------------------------------------------------------------------------
// 3. Spec §4.4 mutual exclusion: deliver + bare effects → MixedBody
//    (the parser admits this; lowering is the enforcer)
// ---------------------------------------------------------------------------

// MixedBody check relaxed (#128, post 49bbeee2). The LoL hero corpus
// uses the composite shape heavily — e.g. ArcaneShift =
// `deliver projectile {…} + dash to_target` (projectile fires on
// impact AND caster simultaneously dashes to the target point). With
// Delivery::Method capturing the payload separately from program.effects,
// both can coexist: trailing bare effects fire on the caster at
// cast-decide time, delivery payload fires on projectile resolution.
#[test]
fn mixed_deliver_and_bare_effects_lowers_into_both_slots() {
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::{Delivery, DeliveryMethodKind, EffectOp};
    // #139: hook-body inner stmts can't carry outer-aggregator
    // modifiers (in-shape / tags / chance / stacking / lifetime / scaling /
    // when / nested) — those slots only exist on top-level effects, so
    // a hook stmt with one would silently lose it. Recursive aggregator
    // capture is a future lift; #139 errors loudly via
    // NestedModifierDropped. Use a bare verb here.
    let src = "ability X {
        target: enemy range: 5.0 cooldown: 25s
        deliver projectile { speed: 12.0 } {
            on_hit { damage 15 }
        }
        damage 5
    }";
    let file = parse_ability_file(src).expect("must parse");
    let prog = lower_ability_decl(&file.abilities[0])
        .expect("composite deliver+bare lowers (was: MixedBody error)");
    // Delivery captures the projectile method + raw payload.
    match &prog.delivery {
        Delivery::Method { kind, .. } => {
            assert_eq!(*kind, DeliveryMethodKind::Projectile);
        }
        other => panic!("expected Delivery::Method(Projectile); got {other:?}"),
    }
    // Trailing bare effects land in program.effects.
    assert_eq!(prog.effects.len(), 1, "the trailing `damage 5` lands in effects");
    assert!(matches!(prog.effects[0], EffectOp::Damage { amount } if (amount - 5.0).abs() < 1e-6));
}

// ---------------------------------------------------------------------------
// 4. `morph { ... } into <Other>` body block
// ---------------------------------------------------------------------------

#[test]
fn lowering_morph_block_returns_unimplemented() {
    let src = "ability X {
        target: self cooldown: 8s
        morph { damage 30 } into Heatseeker
    }";
    let err = lower_inline(src);
    match err {
        LowerError::MorphBlockNotImplemented { ability, into, span } => {
            assert_eq!(ability, "X");
            assert_eq!(into, "Heatseeker");
            assert!(span.start < span.end);
        }
        other => panic!("expected MorphBlockNotImplemented; got {other:?}"),
    }
}

#[test]
fn lowering_morph_diagnostic_names_target_form() {
    let src = "ability X {
        target: self cooldown: 8s
        morph { damage 30 } into FireForm
    }";
    let msg = lower_inline(src).to_string();
    assert!(msg.contains("FireForm"), "diagnostic must name the target form; got: {msg}");
}

// ---------------------------------------------------------------------------
// 5. Wave 1 corpus regression: no deliver / morph / recast — still
//    lowers cleanly to a valid AbilityProgram.
// ---------------------------------------------------------------------------

#[test]
fn lowering_wave_1_corpus_still_works() {
    for (name, src) in [
        ("Strike", "ability Strike { target: enemy range: 5.0 cooldown: 1s hint: damage damage 15 }"),
        ("ShieldUp", "ability ShieldUp { target: self cooldown: 4s hint: defense shield 50 }"),
        ("Mend",     "ability Mend { target: self cooldown: 3s hint: heal heal 20 }"),
    ] {
        let file = parse_ability_file(src).unwrap_or_else(|e| panic!("{name} parses: {e}"));
        let prog = lower_ability_decl(&file.abilities[0])
            .unwrap_or_else(|e| panic!("{name} lowers: {e:?}"));
        assert_eq!(prog.effects.len(), 1, "{name} has one effect");
    }
}
