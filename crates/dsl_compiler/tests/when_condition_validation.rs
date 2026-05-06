//! #143: when-condition expression validation.
//!
//! Wave 1.5#7 captured the `when <cond>` predicate as opaque source
//! text terminated by the next modifier keyword / EOL / `}`. That
//! let typo'd operators and unbalanced sub-expressions silently ship
//! through to the program. #143 wires `dsl_ast::parser::parse_expression`
//! at lower time so syntax bugs in the captured slice surface as
//! `LowerError::WhenConditionParseError`. Field-name validation
//! against the agent schema is a follow-up.

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_lower::{lower_ability_decl, LowerError};

fn lower_first(src: &str) -> Result<(), LowerError> {
    let f = parse_ability_file(src).expect("source must parse");
    lower_ability_decl(&f.abilities[0]).map(|_| ())
}

#[test]
fn well_formed_when_lowers_clean() {
    // Sanity: the standard shape used across the corpus
    // (`when target.hp < 30`) still lowers without complaint.
    let src = "ability A {
        target: enemy range: 5.0 cooldown: 1s
        damage 10 when target.hp < 30
    }";
    lower_first(src).expect("well-formed when must lower");
}

#[test]
fn well_formed_when_else_lowers_clean() {
    // Both branches re-parse; ShadowSummoner-style two-branch shape.
    let src = "ability A {
        target: enemy range: 5.0 cooldown: 1s
        taunt 1500ms when target.alive else target.shield_hp == 0
    }";
    lower_first(src).expect("well-formed when/else must lower");
}

#[test]
fn typoed_operator_in_when_is_caught() {
    // `~` is not a valid binary operator. Pre-#143 this lowered
    // silently; now it must surface as WhenConditionParseError.
    let src = "ability A {
        target: enemy range: 5.0 cooldown: 1s
        damage 10 when target.hp ~ 30
    }";
    let err = lower_first(src).expect_err("typo'd operator must error");
    match err {
        LowerError::WhenConditionParseError { clause, predicate, .. } => {
            assert_eq!(clause, "when");
            assert!(
                predicate.contains("~"),
                "predicate should preserve the offending text: {predicate}"
            );
        }
        other => panic!("expected WhenConditionParseError; got {other:?}"),
    }
}

#[test]
fn malformed_else_clause_is_caught_and_attributed_to_else() {
    // The when branch is fine; the else branch has a stray operator.
    let src = "ability A {
        target: enemy range: 5.0 cooldown: 1s
        damage 10 when target.hp < 30 else target.hp ~ 50
    }";
    let err = lower_first(src).expect_err("typo'd else must error");
    match err {
        LowerError::WhenConditionParseError { clause, .. } => {
            assert_eq!(
                clause, "else",
                "the bad text is on the else branch, error must attribute it there"
            );
        }
        other => panic!("expected WhenConditionParseError; got {other:?}"),
    }
}

#[test]
fn trailing_junk_after_predicate_is_caught() {
    // The captured predicate must consume to its end. A dangling
    // identifier like `target.hp < 30 garbage` should not slip past.
    // (The parser captures up to the next modifier keyword / EOL —
    // `garbage` here is a bare ident on the same line, captured into
    // the predicate. Re-parse should reject the trailing token.)
    let src = "ability A {
        target: enemy range: 5.0 cooldown: 1s
        damage 10 when target.hp < 30 garbage_token
    }";
    let err = lower_first(src).expect_err("trailing junk must error");
    assert!(
        matches!(err, LowerError::WhenConditionParseError { .. }),
        "expected WhenConditionParseError, got {err:?}"
    );
}
