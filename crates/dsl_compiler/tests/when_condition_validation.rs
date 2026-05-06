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

// ---------------------------------------------------------------------
// #143 follow-up: field-name validation against AgentFieldId vocabulary
// ---------------------------------------------------------------------

#[test]
fn typoed_field_in_when_is_caught_against_agent_vocabulary() {
    // `target.htp` (typo of `hp`) parses cleanly as an expression,
    // so the syntax check passes. The follow-up walks the parsed
    // tree and surfaces this as `WhenConditionUnknownField`.
    let src = "ability A {
        target: enemy range: 5.0 cooldown: 1s
        damage 10 when target.htp < 30
    }";
    let err = lower_first(src).expect_err("typo'd field must error");
    match err {
        LowerError::WhenConditionUnknownField { clause, binder, field, .. } => {
            assert_eq!(clause, "when");
            assert_eq!(binder, "target");
            assert_eq!(field, "htp");
        }
        other => panic!("expected WhenConditionUnknownField; got {other:?}"),
    }
}

#[test]
fn typoed_field_on_self_is_also_caught() {
    let src = "ability A {
        target: enemy range: 5.0 cooldown: 1s
        damage 10 when self.aliv
    }";
    let err = lower_first(src).expect_err("typo'd self.aliv must error");
    match err {
        LowerError::WhenConditionUnknownField { binder, field, .. } => {
            assert_eq!(binder, "self");
            assert_eq!(field, "aliv");
        }
        other => panic!("expected WhenConditionUnknownField; got {other:?}"),
    }
}

#[test]
fn typoed_field_in_else_is_attributed_to_else_clause() {
    let src = "ability A {
        target: enemy range: 5.0 cooldown: 1s
        damage 10 when target.hp < 30 else target.bogus_field == 0
    }";
    let err = lower_first(src).expect_err("typo'd else field must error");
    match err {
        LowerError::WhenConditionUnknownField { clause, binder, field, .. } => {
            assert_eq!(clause, "else", "the bad field is on the else branch");
            assert_eq!(binder, "target");
            assert_eq!(field, "bogus_field");
        }
        other => panic!("expected WhenConditionUnknownField; got {other:?}"),
    }
}

#[test]
fn nested_typoed_field_inside_arithmetic_is_caught() {
    // Validator must recurse into Binary subtrees — the typo'd
    // `target.htp` is inside a nested arithmetic + comparison
    // chain, not a top-level Field node.
    let src = "ability A {
        target: enemy range: 5.0 cooldown: 1s
        damage 10 when (target.htp + 5) > 20
    }";
    let err = lower_first(src).expect_err("nested typo'd field must error");
    assert!(
        matches!(
            &err,
            LowerError::WhenConditionUnknownField { field, .. } if field == "htp"
        ),
        "expected WhenConditionUnknownField(htp); got {err:?}"
    );
}

#[test]
fn world_and_config_accessors_are_not_validated_as_agent_fields() {
    // `world.tick`, `config.foo` use binders the agent-field
    // validator deliberately skips — those have their own
    // resolution paths and aren't expected to round-trip through
    // AgentFieldId. This must lower without error.
    let src = "ability A {
        target: enemy range: 5.0 cooldown: 1s
        damage 10 when world.tick > 100
    }";
    lower_first(src).expect("world.tick must NOT trip the agent-field validator");
}

#[test]
fn valid_agent_fields_used_by_lol_corpus_lower_clean() {
    // Every field reference the LoL corpus uses on `target` /
    // `self` (hp / alive / shield_hp) lives in AgentFieldId. None
    // should false-positive. This test pins the canary's coverage
    // so an accidental rename of an AgentFieldId variant would
    // surface here as a test failure.
    for predicate in &["target.hp < 30", "target.alive", "target.shield_hp == 0"] {
        let src = format!(
            "ability A {{ target: enemy range: 5.0 cooldown: 1s damage 10 when {predicate} }}"
        );
        lower_first(&src).unwrap_or_else(|e| {
            panic!("LoL-corpus-style predicate `{predicate}` must lower clean; got {e:?}")
        });
    }
}
