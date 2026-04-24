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

use dsl_compiler::ast::{Decl, ExprKind};
use dsl_compiler::compile;
use dsl_compiler::ir::ScoringRowKind;
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
