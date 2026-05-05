//! Wave 1.4 lowering tests — verify the new body-block surfaces and
//! the two new headers from spec §4.2 / §4.4 / §9 error cleanly at the
//! lowering boundary, and that the Wave 1 corpus
//! (Strike / ShieldUp / Mend) still lowers without regression.
//!
//! Per `crates/dsl_compiler/src/ability_lower.rs` Wave 1.4 module-level
//! docs, lowering of `recast` / `recast_window` headers,
//! `deliver { ... }` body blocks, and `morph { ... } into <Other>`
//! body blocks all require engine-side schema work (multi-stage cast
//! state, delivery-method SoA + on_*/on_arrival/on_tick hook dispatch,
//! form-swap state). Until those land, lowering surfaces:
//!   * `LowerError::HeaderNotImplemented { header: "recast" | "recast_window" }`
//!   * `LowerError::DeliverBlockNotImplemented { ability, method, span }`
//!   * `LowerError::MorphBlockNotImplemented { ability, into, span }`
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
// 1. `recast:` and `recast_window:` headers
// ---------------------------------------------------------------------------

#[test]
fn lowering_recast_header_returns_unimplemented() {
    let err = lower_inline("ability X { target: enemy cooldown: 1s recast: 3 damage 10 }");
    match err {
        LowerError::HeaderNotImplemented { header, span } => {
            assert_eq!(header, "recast");
            assert!(span.start < span.end, "span must be non-empty");
        }
        other => panic!("expected HeaderNotImplemented(recast); got {other:?}"),
    }
}

#[test]
fn lowering_recast_header_duration_form_returns_unimplemented() {
    // The duration form (`recast: 4s`) is the same parser surface;
    // lowering treats both the same way.
    let err = lower_inline("ability X { target: enemy cooldown: 1s recast: 4s damage 10 }");
    match err {
        LowerError::HeaderNotImplemented { header, .. } => assert_eq!(header, "recast"),
        other => panic!("expected HeaderNotImplemented(recast); got {other:?}"),
    }
}

#[test]
fn lowering_recast_window_header_returns_unimplemented() {
    let err = lower_inline(
        "ability X { target: enemy cooldown: 1s recast_window: 10s damage 10 }",
    );
    match err {
        LowerError::HeaderNotImplemented { header, span } => {
            assert_eq!(header, "recast_window");
            assert!(span.start < span.end);
        }
        other => panic!("expected HeaderNotImplemented(recast_window); got {other:?}"),
    }
}

#[test]
fn lowering_recast_diagnostic_message_mentions_the_header_key() {
    let err = lower_inline("ability X { target: enemy cooldown: 1s recast: 3 damage 10 }");
    let msg = err.to_string();
    assert!(
        msg.contains("recast"),
        "diagnostic must name the recast header; got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 2. `deliver { ... }` body block
// ---------------------------------------------------------------------------

#[test]
fn lowering_deliver_block_returns_unimplemented() {
    let src = "ability X {
        target: enemy range: 5.0 cooldown: 1s
        deliver projectile { speed: 16.0 } { on_hit { damage 10 } }
    }";
    let err = lower_inline(src);
    match err {
        LowerError::DeliverBlockNotImplemented { ability, method, span } => {
            assert_eq!(ability, "X");
            assert_eq!(method, "projectile");
            assert!(span.start < span.end);
        }
        other => panic!("expected DeliverBlockNotImplemented; got {other:?}"),
    }
}

#[test]
fn lowering_deliver_channel_block_carries_method() {
    let src = "ability X {
        target: enemy range: 5.0 cooldown: 1s
        deliver channel { duration: 2s, tick: 500ms } { on_tick { damage 7 } }
    }";
    let err = lower_inline(src);
    match err {
        LowerError::DeliverBlockNotImplemented { method, .. } => {
            assert_eq!(method, "channel");
        }
        other => panic!("expected DeliverBlockNotImplemented(channel); got {other:?}"),
    }
}

#[test]
fn lowering_deliver_diagnostic_mentions_method() {
    let src = "ability X {
        target: enemy range: 5.0 cooldown: 1s
        deliver projectile { speed: 16.0 } { on_hit { damage 10 } }
    }";
    let msg = lower_inline(src).to_string();
    assert!(msg.contains("deliver projectile"), "diagnostic must mention the method; got: {msg}");
}

// ---------------------------------------------------------------------------
// 3. Spec §4.4 mutual exclusion: deliver + bare effects → MixedBody
//    (the parser admits this; lowering is the enforcer)
// ---------------------------------------------------------------------------

#[test]
fn mixed_deliver_and_bare_effects_returns_mixed_body() {
    // Mirrors Ahri.SpiritRush's shape (deliver projectile + a trailing
    // bare effect). Uses target: enemy because Wave 1.6 lowering only
    // implements enemy/self targets — `ground` would short-circuit on
    // TargetModeReserved before the body check, hiding what we want
    // to assert here.
    let src = "ability X {
        target: enemy range: 5.0 cooldown: 25s
        deliver projectile { speed: 12.0 } {
            on_hit { damage 15 in circle(6.0) [MAGIC: 50] }
        }
        damage 5
    }";
    let err = lower_inline(src);
    match err {
        LowerError::MixedBody { ability, span } => {
            assert_eq!(ability, "X");
            assert!(span.start < span.end);
        }
        other => panic!("expected MixedBody; got {other:?}"),
    }
}

#[test]
fn mixed_body_diagnostic_mentions_both_shapes() {
    let src = "ability X {
        target: enemy range: 5.0 cooldown: 1s
        deliver projectile { speed: 16.0 } { on_hit { damage 10 } }
        damage 5
    }";
    let msg = lower_inline(src).to_string();
    assert!(msg.contains("deliver"), "diagnostic must name deliver; got: {msg}");
    assert!(msg.contains("effect") || msg.contains("body"), "diagnostic must name effects; got: {msg}");
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
