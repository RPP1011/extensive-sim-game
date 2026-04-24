//! GPU ability evaluation subsystem Phase 2 — per_ability scoring row.
//!
//! Covers the primitives landed incrementally across the PER-AB-1..5
//! sub-commits:
//! * PER-AB-1 — `row <name> per_ability { ... }` parses + resolves,
//!   lands on `ScoringIR::per_ability_rows` with guard/score/target
//!   slots populated.
//! * PER-AB-2 — `ability::tag(PHYSICAL)` lowers to `IrExpr::AbilityTag`.
//! * PER-AB-3 — `ability::hint == damage` lowers to
//!   `IrExpr::Binary(Eq, AbilityHint, AbilityHintLit(Damage))`.
//! * PER-AB-4 — `ability::range` + `ability::on_cooldown(ability)` lower
//!   to `IrExpr::AbilityRange` / `IrExpr::AbilityOnCooldown`.
//! * PER-AB-5 — the spec's full `pick_ability` row parses + resolves
//!   cleanly end-to-end.
//!
//! See `docs/superpowers/specs/2026-04-22-gpu-ability-evaluation-design.md`
//! §Architecture.

use dsl_compiler::ast::{BinOp, Decl, ExprKind};
use dsl_compiler::compile;
use dsl_compiler::ir::{AbilityHint, AbilityTag, IrExpr, ScoringRowKind};
use dsl_compiler::parse;

// ---------------------------------------------------------------------------
// PER-AB-1: parser + IR shape for `row <name> per_ability { ... }`
// ---------------------------------------------------------------------------

const SRC_MINIMAL: &str = r#"
scoring {
  row pick_ability per_ability {
    score: 1.0
  }
}
"#;

#[test]
fn per_ability_row_parses_on_scoring_block() {
    let program = parse(SRC_MINIMAL).expect("program should parse");
    let scoring_decl = program
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::Scoring(s) => Some(s),
            _ => None,
        })
        .expect("scoring block should be present");
    assert_eq!(
        scoring_decl.entries.len(),
        0,
        "per_ability rows should not land on the standard `entries` list"
    );
    assert_eq!(
        scoring_decl.per_ability_rows.len(),
        1,
        "one per_ability row should land on `per_ability_rows`"
    );
    let row = &scoring_decl.per_ability_rows[0];
    assert_eq!(row.name, "pick_ability");
    assert!(row.guard.is_none(), "guard is optional, should default to None");
    assert!(row.target.is_none(), "target is optional, should default to None");
    match &row.score.kind {
        ExprKind::Float(f) => assert_eq!(*f, 1.0),
        other => panic!("expected score to be Float(1.0); got {other:?}"),
    }
}

#[test]
fn per_ability_row_lowers_to_ir() {
    let comp = compile(SRC_MINIMAL).expect("compile should succeed");
    let scoring = comp
        .scoring
        .first()
        .expect("one scoring block should land in IR");
    assert!(
        scoring.entries.is_empty(),
        "standard entries list should stay empty for a pure per_ability block"
    );
    assert_eq!(scoring.per_ability_rows.len(), 1);
    let row = &scoring.per_ability_rows[0];
    assert_eq!(row.name, "pick_ability");
    assert_eq!(row.kind(), ScoringRowKind::PerAbility);
    assert!(row.guard.is_none());
    assert!(row.target.is_none());
}

// ---------------------------------------------------------------------------
// PER-AB-2: `ability::tag(TAG_NAME)` lowers to IrExpr::AbilityTag
// ---------------------------------------------------------------------------

#[test]
fn ability_tag_primitive_lowers_to_ir_variant() {
    const SRC: &str = r#"
scoring {
  row pick_ability per_ability {
    score: ability::tag(PHYSICAL)
  }
}
"#;
    let comp = compile(SRC).expect("compile should succeed");
    let row = &comp.scoring[0].per_ability_rows[0];
    match &row.score.kind {
        IrExpr::AbilityTag { tag } => {
            assert_eq!(*tag, AbilityTag::Physical);
        }
        other => panic!("expected IrExpr::AbilityTag(Physical), got {other:?}"),
    }
}

#[test]
fn ability_tag_primitive_recognises_every_tag_name() {
    for (ident, expected) in [
        ("PHYSICAL", AbilityTag::Physical),
        ("MAGICAL", AbilityTag::Magical),
        ("CROWD_CONTROL", AbilityTag::CrowdControl),
        ("HEAL", AbilityTag::Heal),
        ("DEFENSE", AbilityTag::Defense),
        ("UTILITY", AbilityTag::Utility),
    ] {
        let src = format!(
            r#"
scoring {{
  row r per_ability {{
    score: ability::tag({ident})
  }}
}}
"#,
        );
        let comp = compile(&src).unwrap_or_else(|e| panic!("compile `{ident}` failed: {e}"));
        let row = &comp.scoring[0].per_ability_rows[0];
        match &row.score.kind {
            IrExpr::AbilityTag { tag } => assert_eq!(*tag, expected, "tag `{ident}`"),
            other => panic!("tag `{ident}` lowered to unexpected IR: {other:?}"),
        }
    }
}

#[test]
fn ability_tag_primitive_rejects_unknown_tag_name() {
    const SRC: &str = r#"
scoring {
  row pick_ability per_ability {
    score: ability::tag(BOGUS_TAG)
  }
}
"#;
    let err = compile(SRC).expect_err("unknown tag should be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("BOGUS_TAG") || msg.contains("unknown ability tag"),
        "error should mention the unknown tag; got: {msg}"
    );
}

#[test]
fn ability_tag_primitive_rejects_wrong_arity() {
    const SRC: &str = r#"
scoring {
  row pick_ability per_ability {
    score: ability::tag(PHYSICAL, MAGICAL)
  }
}
"#;
    let err = compile(SRC).expect_err("two-arg `ability::tag` should be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("ability::tag") || msg.contains("argument"),
        "error should mention arity; got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// PER-AB-3: `ability::hint == <ident>` lowers via AbilityHint + AbilityHintLit
// ---------------------------------------------------------------------------

#[test]
fn ability_hint_compare_lowers_to_hint_and_hint_lit() {
    const SRC: &str = r#"
scoring {
  row pick_ability per_ability {
    score: if ability::hint == damage { 1.0 } else { 0.0 }
  }
}
"#;
    let comp = compile(SRC).expect("compile should succeed");
    let row = &comp.scoring[0].per_ability_rows[0];
    // Score body is `if ability::hint == damage { 1.0 } else { 0.0 }`.
    let IrExpr::If { cond, .. } = &row.score.kind else {
        panic!("expected `if` wrapper at the top; got {:?}", row.score.kind);
    };
    match &cond.kind {
        IrExpr::Binary(op, lhs, rhs) => {
            assert_eq!(*op, BinOp::Eq);
            assert!(
                matches!(lhs.kind, IrExpr::AbilityHint),
                "lhs should be AbilityHint; got {:?}",
                lhs.kind
            );
            match &rhs.kind {
                IrExpr::AbilityHintLit(AbilityHint::Damage) => {}
                other => panic!("rhs should be AbilityHintLit(Damage); got {other:?}"),
            }
        }
        other => panic!("expected Binary in cond; got {other:?}"),
    }
}

#[test]
fn ability_hint_compare_reversed_order_also_lowers() {
    // `damage == ability::hint` — mirror form.
    const SRC: &str = r#"
scoring {
  row pick_ability per_ability {
    score: if damage == ability::hint { 1.0 } else { 0.0 }
  }
}
"#;
    let comp = compile(SRC).expect("compile should succeed");
    let row = &comp.scoring[0].per_ability_rows[0];
    let IrExpr::If { cond, .. } = &row.score.kind else {
        panic!("expected `if` wrapper at the top; got {:?}", row.score.kind);
    };
    match &cond.kind {
        IrExpr::Binary(op, lhs, rhs) => {
            assert_eq!(*op, BinOp::Eq);
            // Source order preserved — lhs should be the literal, rhs the accessor.
            match &lhs.kind {
                IrExpr::AbilityHintLit(AbilityHint::Damage) => {}
                other => panic!("expected AbilityHintLit(Damage) on lhs; got {other:?}"),
            }
            assert!(
                matches!(rhs.kind, IrExpr::AbilityHint),
                "rhs should be AbilityHint accessor; got {:?}",
                rhs.kind
            );
        }
        other => panic!("expected Binary in cond; got {other:?}"),
    }
}

#[test]
fn ability_hint_compare_recognises_every_hint_name() {
    for (ident, expected) in [
        ("damage", AbilityHint::Damage),
        ("defense", AbilityHint::Defense),
        ("crowd_control", AbilityHint::CrowdControl),
        ("utility", AbilityHint::Utility),
    ] {
        let src = format!(
            r#"
scoring {{
  row r per_ability {{
    score: if ability::hint == {ident} {{ 1.0 }} else {{ 0.0 }}
  }}
}}
"#,
        );
        let comp = compile(&src).unwrap_or_else(|e| panic!("compile `{ident}` failed: {e}"));
        let row = &comp.scoring[0].per_ability_rows[0];
        let IrExpr::If { cond, .. } = &row.score.kind else {
            panic!("expected if wrapper for `{ident}`");
        };
        let IrExpr::Binary(_, _, rhs) = &cond.kind else {
            panic!("expected Binary for `{ident}`");
        };
        match &rhs.kind {
            IrExpr::AbilityHintLit(h) => assert_eq!(*h, expected, "hint `{ident}`"),
            other => panic!("hint `{ident}` lowered to unexpected IR: {other:?}"),
        }
    }
}

#[test]
fn ability_hint_accessor_parses_as_flattened_ident() {
    // Verify the AST side: `ability::hint` lands as a flattened
    // Ident("ability::hint") rather than any Field shape.
    const SRC: &str = r#"
scoring {
  row r per_ability {
    score: if ability::hint == damage { 1.0 } else { 0.0 }
  }
}
"#;
    let program = parse(SRC).expect("parse should succeed");
    let scoring = program
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::Scoring(s) => Some(s),
            _ => None,
        })
        .unwrap();
    let row = &scoring.per_ability_rows[0];
    let ExprKind::If { cond, .. } = &row.score.kind else {
        panic!("expected if wrapper at AST layer");
    };
    let ExprKind::Binary { lhs, .. } = &cond.kind else {
        panic!("expected Binary cond");
    };
    match &lhs.kind {
        ExprKind::Ident(s) => assert_eq!(s, "ability::hint"),
        other => panic!("expected flattened Ident(`ability::hint`); got {other:?}"),
    }
}
