//! Pin: `stun N` (bare integer, no time suffix) lowers to the
//! typed `LowerError::EffectArgExpectedDuration` variant — NOT the
//! structurally-misleading `EffectArgMismatch { expected: 1, got: 1 }`.
//!
//! ## Why this matters (Gap squad_skirmish#A)
//!
//! Designers naturally reach for `stun 8` (intending "8 ticks") or
//! `stun 4` (intending "4 seconds"); both shapes were rejected with
//!
//! ```text
//! Lower(EffectArgMismatch { verb: "stun", expected: 1, got: 1, ... })
//! ```
//!
//! The "expected 1, got 1" half is structurally false (both sides ARE
//! 1) and gives the author no hint about the missing time suffix. The
//! root cause: the parser captures `stun 8` as
//! `EffectArg::Number(8.0)`, which clears the arity check but fails the
//! `EffectArg::Duration(_)` shape check inside `require_duration_arg`.
//!
//! Post-fix, `require_duration_arg` detects the bare-`Number` case and
//! surfaces `EffectArgExpectedDuration { verb, got_value, span }` with
//! a designer-facing message pointing at the missing time suffix.
//!
//! ## What this exercises
//!
//! Per status-effect verb (stun / root / silence / fear / taunt /
//! charm / grounded / suppress), parse a `<verb> 8` decl and assert:
//!   1. The error variant is `EffectArgExpectedDuration` (NOT
//!      `EffectArgMismatch`).
//!   2. The carried `verb` matches the source verb.
//!   3. The carried `got_value` renders as `"8"` (integer-valued, no
//!      `.0` tail) so the diagnostic reads `bare number 8`.
//!   4. The `Display` impl mentions both `1s` and `1ms` as suggestion
//!      shapes — the designer needs to know the suffix is missing.

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_lower::{lower_ability_decl, LowerError};

/// Helper: parse one inline ability source and return the
/// `LowerError` from `lower_ability_decl`. Panics if lowering succeeds —
/// every call site here is testing a rejection path.
fn lower_inline_err(src: &str) -> LowerError {
    let file = parse_ability_file(src).expect("parser must accept the source");
    lower_ability_decl(&file.abilities[0])
        .expect_err("lowering must reject `<verb> 8` (bare int)")
}

/// Pin the typed-diagnostic shape for a single duration-bearing verb.
/// Factored so each verb gets its own test with a clear failure label.
fn assert_bare_int_diagnostic(src: &str, expected_verb: &str) {
    let err = lower_inline_err(src);
    match &err {
        LowerError::EffectArgExpectedDuration { verb, got_value, .. } => {
            assert_eq!(
                verb, expected_verb,
                "expected verb `{expected_verb}`; got `{verb}`",
            );
            assert_eq!(
                got_value, "8",
                "got_value should render integer-valued f32 as `8` (no `.0`); got `{got_value}`",
            );
        }
        other => panic!(
            "expected EffectArgExpectedDuration for `{expected_verb} 8`; got {other:?}",
        ),
    }

    // Display must name the missing suffix and show both `1s` and `1ms`
    // as candidate shapes — the designer needs both options visible.
    let rendered = format!("{err}");
    assert!(
        rendered.contains("time-suffixed"),
        "Display must mention `time-suffixed`; got: {rendered}",
    );
    assert!(
        rendered.contains("8s") && rendered.contains("8ms"),
        "Display must suggest `8s` and `8ms`; got: {rendered}",
    );
    assert!(
        rendered.contains(expected_verb),
        "Display must name the offending verb `{expected_verb}`; got: {rendered}",
    );
}

#[test]
fn stun_bare_int_emits_typed_diagnostic() {
    // The canonical case from the squad_skirmish gap report.
    assert_bare_int_diagnostic(
        "ability X { target: enemy range: 5.0 cooldown: 1s stun 8 }",
        "stun",
    );
}

#[test]
fn root_bare_int_emits_typed_diagnostic() {
    assert_bare_int_diagnostic(
        "ability X { target: enemy range: 5.0 cooldown: 1s root 8 }",
        "root",
    );
}

#[test]
fn silence_bare_int_emits_typed_diagnostic() {
    assert_bare_int_diagnostic(
        "ability X { target: enemy range: 5.0 cooldown: 1s silence 8 }",
        "silence",
    );
}

#[test]
fn fear_bare_int_emits_typed_diagnostic() {
    assert_bare_int_diagnostic(
        "ability X { target: enemy range: 5.0 cooldown: 1s fear 8 }",
        "fear",
    );
}

#[test]
fn taunt_bare_int_emits_typed_diagnostic() {
    assert_bare_int_diagnostic(
        "ability X { target: enemy range: 5.0 cooldown: 1s taunt 8 }",
        "taunt",
    );
}

#[test]
fn charm_bare_int_emits_typed_diagnostic() {
    assert_bare_int_diagnostic(
        "ability X { target: enemy range: 5.0 cooldown: 1s charm 8 }",
        "charm",
    );
}

#[test]
fn grounded_bare_int_emits_typed_diagnostic() {
    assert_bare_int_diagnostic(
        "ability X { target: enemy range: 5.0 cooldown: 1s grounded 8 }",
        "grounded",
    );
}

#[test]
fn suppress_bare_int_emits_typed_diagnostic() {
    assert_bare_int_diagnostic(
        "ability X { target: enemy range: 5.0 cooldown: 1s suppress 8 }",
        "suppress",
    );
}

/// Regression guard: the time-suffixed shape (`stun 800ms`) MUST still
/// lower cleanly — the new variant is purely a diagnostic improvement
/// for the bare-int rejection path. Without this pin a future
/// require_duration_arg refactor could over-trigger the new variant.
#[test]
fn stun_with_time_suffix_still_lowers() {
    let file = parse_ability_file(
        "ability X { target: enemy range: 5.0 cooldown: 1s stun 800ms }",
    )
    .expect("parser");
    lower_ability_decl(&file.abilities[0])
        .expect("`stun 800ms` must lower cleanly post-fix");
}

/// Regression guard: the `for <duration>` modifier shape MUST still
/// lower cleanly — `extract_duration` short-circuits on the modifier
/// and never reaches `require_duration_arg`, so the new diagnostic
/// MUST NOT fire here.
#[test]
fn stun_with_for_modifier_still_lowers() {
    let file = parse_ability_file(
        "ability X { target: enemy range: 5.0 cooldown: 1s stun for 2s }",
    )
    .expect("parser");
    lower_ability_decl(&file.abilities[0])
        .expect("`stun for 2s` must lower cleanly post-fix");
}

/// Pin: a fractional bare number (`stun 8.5`) renders as `8.5` (NOT
/// `8.5_f32` and NOT truncated to `8`). The `format_bare_number`
/// helper is in charge of this; pin its boundary.
#[test]
fn stun_fractional_bare_number_renders_with_decimal() {
    let err = lower_inline_err(
        "ability X { target: enemy range: 5.0 cooldown: 1s stun 8.5 }",
    );
    match err {
        LowerError::EffectArgExpectedDuration { got_value, .. } => {
            assert_eq!(
                got_value, "8.5",
                "fractional bare number must keep the decimal; got `{got_value}`",
            );
        }
        other => panic!("expected EffectArgExpectedDuration; got {other:?}"),
    }
}
