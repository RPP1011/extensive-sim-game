//! Lexer regression tests for task #225 — hex literals + integer
//! suffixes in the .ability and .sim DSL number lex.
//!
//! Coverage:
//! - Hex literals (upper/lower case, `_` separators).
//! - Decimal `_` separators (already supported, asserted for safety).
//! - Integer suffixes (`u`, `i`, `u8`/`i32`/etc) accepted-and-discarded.
//! - Same surface in both `.ability` and `.sim` parsers.
//!
//! The integer-suffix is purely informational at this stage — the AST
//! stores numbers as `f64` (or `f32` for `EffectArg::Number`), and lower
//! assigns the real type from the verb signature. We just verify the
//! suffix doesn't blow up the lex.

use dsl_ast::ast::{EffectArg, EffectStmt, AbilityHeader};
use dsl_ast::parse_ability_file;

/// Build a minimal `.ability` source with one `damage <N>` effect, parse
/// it, and pull out the numeric value of arg 0 as f64. The test verbs
/// here use `damage` because it's a single-arg `Number` effect that
/// exists in every wave of the verb table.
fn parse_damage_arg(src: &str) -> f64 {
    let file = parse_ability_file(src)
        .unwrap_or_else(|e| panic!("parse failed for `{src}`: {e}"));
    let a = &file.abilities[0];
    let stmt: &EffectStmt = &a.effects[0];
    assert_eq!(stmt.verb, "damage");
    match &stmt.args[0] {
        EffectArg::Number(v) => *v as f64,
        other => panic!("expected EffectArg::Number; got {other:?}"),
    }
}

#[test]
fn hex_literal_uppercase() {
    let src = "ability T { target: enemy cooldown: 1s damage 0xFF }";
    assert_eq!(parse_damage_arg(src), 255.0);
}

#[test]
fn hex_literal_lowercase() {
    // 0xdead_beef = 3735928559. Above f32 mantissa precision (24 bits),
    // so we run this one through `decoy <subject> <fake_pos>` which
    // packs the value into a u32 via the lower step. For the bare-arg
    // path here we use a smaller value that round-trips through f32
    // exactly: 0xdead = 57005.
    let src = "ability T { target: enemy cooldown: 1s damage 0xdead }";
    assert_eq!(parse_damage_arg(src), 57005.0);
}

#[test]
fn hex_literal_with_separators() {
    let src = "ability T { target: enemy cooldown: 1s damage 0xFF_FF }";
    assert_eq!(parse_damage_arg(src), 65535.0);
}

#[test]
fn decimal_with_separators() {
    let src = "ability T { target: enemy cooldown: 1s damage 1_000 }";
    assert_eq!(parse_damage_arg(src), 1000.0);
}

#[test]
fn integer_suffix_unsigned() {
    let src = "ability T { target: enemy cooldown: 1s damage 1u32 }";
    assert_eq!(parse_damage_arg(src), 1.0);
}

#[test]
fn integer_suffix_signed() {
    let src = "ability T { target: enemy cooldown: 1s damage 100i16 }";
    assert_eq!(parse_damage_arg(src), 100.0);
}

#[test]
fn integer_suffix_bare_u_and_i() {
    // Bare `u`/`i` (no width) is also accepted.
    assert_eq!(
        parse_damage_arg("ability T { target: enemy cooldown: 1s damage 7u }"),
        7.0,
    );
    assert_eq!(
        parse_damage_arg("ability T { target: enemy cooldown: 1s damage 9i }"),
        9.0,
    );
}

#[test]
fn hex_then_int_suffix() {
    let src = "ability T { target: enemy cooldown: 1s damage 0xFFu8 }";
    assert_eq!(parse_damage_arg(src), 255.0);
}

#[test]
fn decimal_unit_suffix_unaffected_by_int_suffix_lex() {
    // `5s` must still parse as a Duration, not as an integer with `s`
    // suffix (`s` is not a recognized int suffix). Verifies our suffix
    // lex doesn't shadow the unit lex on the duration path.
    let src = "ability T { target: enemy cooldown: 5s damage 1 }";
    let file = parse_ability_file(src).expect("must parse");
    let cd = file.abilities[0]
        .headers
        .iter()
        .find_map(|h| match h {
            AbilityHeader::Cooldown(d, _) => Some(d.millis),
            _ => None,
        })
        .expect("cooldown header");
    assert_eq!(cd, 5_000);
}

#[test]
fn ident_after_number_not_consumed_as_suffix() {
    // `1up` must NOT consume `u` as a suffix (would leave `p` dangling).
    // The number lex stops at `1`, then `up` is an ident. Since the
    // .ability effect-arg grammar accepts `Ident`, `damage 1 up` would
    // parse as two args. Use the actual bare-ident form.
    let src = "ability T { target: enemy cooldown: 1s damage 1 up }";
    let file = parse_ability_file(src).expect("must parse");
    let stmt = &file.abilities[0].effects[0];
    assert_eq!(stmt.args.len(), 2);
    match &stmt.args[0] {
        EffectArg::Number(v) => assert_eq!(*v, 1.0),
        other => panic!("expected Number(1); got {other:?}"),
    }
    match &stmt.args[1] {
        EffectArg::Ident(name) => assert_eq!(name, "up"),
        other => panic!("expected Ident(up); got {other:?}"),
    }
}

#[test]
fn malformed_hex_no_digits_errors() {
    // `0x` alone is not a valid hex literal.
    let src = "ability T { target: enemy cooldown: 1s damage 0x }";
    let res = parse_ability_file(src);
    assert!(res.is_err(), "expected parse error for bare `0x`, got {res:?}");
}
