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
fn well_formed_when_else_lowers_to_or() {
    // Task #228 (PR #51, 2026-05-09) accepted `when X else Y` as parse-
    // and-lower sugar for `when X || Y` — both branches lower to
    // identical RPN bytes. Previously this asserted the else clause
    // errored at lower time; now we assert it lowers cleanly.
    let src = "ability A {
        target: enemy range: 5.0 cooldown: 1s
        taunt 1500ms when target.hp < 50 else target.armor == 0
    }";
    lower_first(src).expect("well-formed `when X else Y` must lower (Task #228)");
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
fn malformed_else_clause_errors_as_parse_error_with_else_attribution() {
    // Task #228 (PR #51, 2026-05-09): the else branch now parses and
    // lowers, so malformed else-text surfaces as WhenConditionParseError
    // with `clause = "else"` attribution (the parse_when_branch helper
    // attributes errors to whichever clause they came from).
    let src = "ability A {
        target: enemy range: 5.0 cooldown: 1s
        damage 10 when target.hp < 30 else target.hp ~ 50
    }";
    let err = lower_first(src).expect_err("malformed else must error at lower");
    match err {
        LowerError::WhenConditionParseError { clause, predicate, .. } => {
            assert_eq!(clause, "else");
            assert!(
                predicate.contains("~"),
                "predicate should preserve the offending text: {predicate}"
            );
        }
        other => panic!("expected WhenConditionParseError{{clause:else}}; got {other:?}"),
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
fn typoed_field_in_else_surfaces_as_unknown_field() {
    // Task #228 (PR #51, 2026-05-09): else branches now parse + lower,
    // so field-validation runs on them too. A typo'd field in the else
    // branch surfaces as WhenConditionUnknownField, same as in the when
    // branch.
    let src = "ability A {
        target: enemy range: 5.0 cooldown: 1s
        damage 10 when target.hp < 30 else target.bogus_field == 0
    }";
    let err = lower_first(src).expect_err("typo'd else field must error");
    match err {
        LowerError::WhenConditionUnknownField { binder, field, .. } => {
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
fn world_and_config_accessors_error_at_lower() {
    // Wave 1.5#7 GPU eval (2026-05-07): the restricted predicate vocab
    // requires `<binder>.<field>` with binder ∈ {self, target}. The
    // `world.tick`, `config.foo` accessors are outside this vocab and
    // surface as WhenConditionUnsupported (they have their own
    // resolution paths but the apply_program predicate evaluator does
    // not honour them).
    let src = "ability A {
        target: enemy range: 5.0 cooldown: 1s
        damage 10 when world.tick > 100
    }";
    let err = lower_first(src).expect_err("world.tick must error at lower");
    match err {
        LowerError::WhenConditionUnsupported { clause, .. } => {
            assert_eq!(clause, "when");
        }
        other => panic!("expected WhenConditionUnsupported; got {other:?}"),
    }
}

#[test]
fn valid_agent_fields_used_by_lol_corpus_lower_clean() {
    // Wave 1.5#7 GPU eval (2026-05-07): the restricted predicate vocab
    // requires `<binder>.<field> <op> <literal>` with field in the
    // 8-field ScalingStatRef-shaped subset (attack_damage / ability_power
    // / max_hp / hp / armor / magic_resist / move_speed / mana). Bool
    // fields (target.alive) and non-ScalingStatRef fields (shield_hp)
    // now error at lower (deferred — open task #163-followup). The
    // surviving forms exercise the supported f32-stat subset.
    for predicate in &["target.hp < 30", "self.attack_damage >= 50", "target.armor < 10"] {
        let src = format!(
            "ability A {{ target: enemy range: 5.0 cooldown: 1s damage 10 when {predicate} }}"
        );
        lower_first(&src).unwrap_or_else(|e| {
            panic!("LoL-corpus-style predicate `{predicate}` must lower clean; got {e:?}")
        });
    }
}
