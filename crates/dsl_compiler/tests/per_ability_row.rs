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

// ---------------------------------------------------------------------------
// PER-AB-4: `ability::range` + `ability::on_cooldown(ability)`
// ---------------------------------------------------------------------------

#[test]
fn ability_range_lowers_to_ir_variant() {
    const SRC: &str = r#"
scoring {
  row pick_ability per_ability {
    score: ability::range
  }
}
"#;
    let comp = compile(SRC).expect("compile should succeed");
    let row = &comp.scoring[0].per_ability_rows[0];
    match &row.score.kind {
        IrExpr::AbilityRange => {}
        other => panic!("expected IrExpr::AbilityRange; got {other:?}"),
    }
}

#[test]
fn ability_on_cooldown_lowers_to_ir_variant() {
    // `ability::on_cooldown(ability)` — the inner `ability` refers to
    // the row's implicit slot binder. Phase 2 resolves it as a bare
    // local reference; Phase 3 lowers the whole variant against the
    // per-(agent, slot) cooldown buffer.
    const SRC: &str = r#"
scoring {
  row pick_ability per_ability {
    guard: !ability::on_cooldown(ability)
    score: 1.0
  }
}
"#;
    let comp = compile(SRC).expect("compile should succeed");
    let row = &comp.scoring[0].per_ability_rows[0];
    let guard = row.guard.as_ref().expect("guard should be present");
    // Guard is `!ability::on_cooldown(ability)` — Unary(Not, ...).
    let IrExpr::Unary(_, inner) = &guard.kind else {
        panic!("expected Unary(Not, ...) at top of guard; got {:?}", guard.kind);
    };
    match &inner.kind {
        IrExpr::AbilityOnCooldown(slot) => {
            // The slot expression should resolve the implicit `ability`
            // local seeded on the per_ability scope.
            match &slot.kind {
                IrExpr::Local(_, name) => assert_eq!(name, "ability"),
                other => panic!(
                    "expected the inner arg to resolve to Local(ability); got {other:?}"
                ),
            }
        }
        other => panic!("expected IrExpr::AbilityOnCooldown(...); got {other:?}"),
    }
}

#[test]
fn ability_on_cooldown_accepts_literal_slot_index() {
    // Alternative shape: a literal slot index. Should also lower.
    const SRC: &str = r#"
scoring {
  row pick_ability per_ability {
    guard: !ability::on_cooldown(0)
    score: 1.0
  }
}
"#;
    let comp = compile(SRC).expect("compile should succeed");
    let row = &comp.scoring[0].per_ability_rows[0];
    let guard = row.guard.as_ref().unwrap();
    let IrExpr::Unary(_, inner) = &guard.kind else {
        panic!("expected Unary guard");
    };
    let IrExpr::AbilityOnCooldown(slot) = &inner.kind else {
        panic!("expected AbilityOnCooldown");
    };
    match &slot.kind {
        IrExpr::LitInt(n) => assert_eq!(*n, 0),
        other => panic!("expected LitInt(0); got {other:?}"),
    }
}

#[test]
fn ability_on_cooldown_rejects_wrong_arity() {
    const SRC: &str = r#"
scoring {
  row pick_ability per_ability {
    guard: ability::on_cooldown()
    score: 1.0
  }
}
"#;
    let err = compile(SRC).expect_err("zero-arg `ability::on_cooldown` should be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("on_cooldown") || msg.contains("argument"),
        "error should mention arity; got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// PER-AB-5: composite — the spec's full pick_ability row parses + resolves
// ---------------------------------------------------------------------------
//
// Exercises every Phase 2 primitive in one row:
//   * `row ... per_ability { ... }` (PER-AB-1)
//   * `ability::tag(PHYSICAL)` / tag(CROWD_CONTROL) / tag(DEFENSE) (PER-AB-2)
//   * `ability::on_cooldown(ability)` (PER-AB-4)
//   * `ability::range` (PER-AB-4)
//
// The target clause uses an unresolved wrapper (`nearest_hostile_in_range`)
// since the Phase 2 scope doesn't include target-query primitives — the
// resolver keeps the call as `IrExpr::UnresolvedCall` for a later pass
// (or a future stdlib addition) without failing. The scope also seeds
// `engaged_with_hostile` as an unresolved ident (a view call or local
// the spec fixture cites without declaring); `validate_scoring_body`
// tolerates it for the same reason.

// Stub `engaged_with_hostile` view so the spec fixture's expression
// `if engaged_with_hostile { 0.5 } else { 0.0 }` resolves cleanly.
// The body is a constant — this fixture exists only to prove the
// `per_ability` parser + resolver accept the spec's composite shape,
// not to exercise realistic view semantics. The view declaration
// takes a `(self: Agent)` parameter because `view <name>(...)` is
// the required grammar.
const SPEC_FIXTURE: &str = r#"
view engaged_with_hostile(a: Agent) -> bool { false }

scoring {
  row pick_ability per_ability {
    guard:  !ability::on_cooldown(ability)
    score:  ability::tag(PHYSICAL) * (1 - target.hp_frac)
          + ability::tag(CROWD_CONTROL) * (if engaged_with_hostile(self) { 0.5 } else { 0.0 })
          + ability::tag(DEFENSE) * (if self.hp_frac < 0.3 { 1.0 } else { 0.0 })
    target: nearest_hostile_in_range(ability::range)
  }
}
"#;

#[test]
fn spec_pick_ability_row_parses_end_to_end() {
    let program = parse(SPEC_FIXTURE).expect("parse should succeed");
    let scoring = program
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::Scoring(s) => Some(s),
            _ => None,
        })
        .unwrap();
    assert_eq!(scoring.per_ability_rows.len(), 1);
    assert_eq!(scoring.entries.len(), 0);
    let row = &scoring.per_ability_rows[0];
    assert_eq!(row.name, "pick_ability");
    assert!(row.guard.is_some());
    assert!(row.target.is_some());
}

#[test]
fn spec_pick_ability_row_resolves_end_to_end() {
    let comp = compile(SPEC_FIXTURE).expect("compile should succeed");
    let row = &comp.scoring[0].per_ability_rows[0];

    // Guard: `!ability::on_cooldown(ability)`.
    let guard = row.guard.as_ref().expect("guard present");
    let IrExpr::Unary(_, inner) = &guard.kind else {
        panic!("guard should be Unary(Not, ...); got {:?}", guard.kind);
    };
    assert!(
        matches!(inner.kind, IrExpr::AbilityOnCooldown(_)),
        "guard inner should be AbilityOnCooldown; got {:?}",
        inner.kind
    );

    // Score: sum of three tag-weighted terms. Walk left-associatively
    // to confirm each tag is present.
    let mut tags_seen: Vec<AbilityTag> = Vec::new();
    fn collect_tags(e: &IrExpr, out: &mut Vec<AbilityTag>) {
        match e {
            IrExpr::AbilityTag { tag } => out.push(*tag),
            IrExpr::Binary(_, lhs, rhs) => {
                collect_tags(&lhs.kind, out);
                collect_tags(&rhs.kind, out);
            }
            IrExpr::Unary(_, inner) => collect_tags(&inner.kind, out),
            IrExpr::If { cond, then_expr, else_expr } => {
                collect_tags(&cond.kind, out);
                collect_tags(&then_expr.kind, out);
                if let Some(e) = else_expr {
                    collect_tags(&e.kind, out);
                }
            }
            _ => {}
        }
    }
    collect_tags(&row.score.kind, &mut tags_seen);
    tags_seen.sort_by_key(|t| *t as u8);
    let mut expected = vec![
        AbilityTag::Physical,
        AbilityTag::CrowdControl,
        AbilityTag::Defense,
    ];
    expected.sort_by_key(|t| *t as u8);
    assert_eq!(
        tags_seen, expected,
        "score should mention PHYSICAL + CROWD_CONTROL + DEFENSE"
    );

    // Target: `nearest_hostile_in_range(ability::range)` — the inner
    // `ability::range` should have lowered. The outer call resolves
    // as an `UnresolvedCall` (the wrapper isn't a declared stdlib
    // method) whose lone argument is `IrExpr::AbilityRange`.
    let target = row.target.as_ref().expect("target present");
    match &target.kind {
        IrExpr::UnresolvedCall(name, args) => {
            assert_eq!(name, "nearest_hostile_in_range");
            assert_eq!(args.len(), 1);
            assert!(
                matches!(args[0].value.kind, IrExpr::AbilityRange),
                "target arg should be AbilityRange; got {:?}",
                args[0].value.kind
            );
        }
        other => panic!("expected UnresolvedCall for target; got {other:?}"),
    }
}

/// The composite fixture, when stored as the whole row, should not
/// disturb any other row (a regression guard — a later change that
/// accidentally drains `entries` or mis-associates rows would tank
/// this).
#[test]
fn spec_pick_ability_row_coexists_with_standard_rows() {
    const SRC: &str = r#"
scoring {
  Attack(t) = 1.0
  row pick_ability per_ability {
    score: ability::tag(PHYSICAL)
  }
  Hold = 0.1
}
"#;
    let comp = compile(SRC).expect("compile should succeed");
    let scoring = &comp.scoring[0];
    assert_eq!(
        scoring.entries.len(),
        2,
        "two standard rows (Attack, Hold) should survive"
    );
    assert_eq!(
        scoring.per_ability_rows.len(),
        1,
        "one per_ability row should land on its own list"
    );
    assert_eq!(scoring.per_ability_rows[0].name, "pick_ability");
}
