//! Unit tests for the `beliefs(observer).about(target).<field>` and related
//! expression READ primitives (Plan ToM Task 8).
//!
//! Covers:
//! - `beliefs(self).about(target).last_known_hp` → `IrExpr::BeliefsAccessor`
//! - `beliefs(self).confidence(target)` → `IrExpr::BeliefsConfidence`
//! - `beliefs(self).all_known(_)` → `IrExpr::BeliefsView`
//! - Unknown field in `.about(t).<bogus_field>` is rejected.
//! - The existing statement form (`beliefs(o).observe(t) with { ... }`)
//!   still parses correctly when `beliefs` appears in statement position.

use dsl_compiler::{compile, ir::IrExpr, ir::IrStmt};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Wrap an expression in a physics rule so the resolver has full scope.
fn physics_src(expr: &str) -> String {
    format!(
        r#"
event PhysEv {{
    actor: AgentId,
    target: AgentId,
}}

physics test_physics {{
    on PhysEv {{ actor: a, target: t }} {{
        let _x = {};
    }}
}}
"#,
        expr
    )
}

// ---------------------------------------------------------------------------
// Happy-path: BeliefsAccessor
// ---------------------------------------------------------------------------

#[test]
fn parses_beliefs_about_last_known_hp() {
    // Embed a beliefs read in a physics let-binding.
    let src = physics_src("beliefs(a).about(t).last_known_hp");
    let comp = compile(&src).expect("compile BeliefsAccessor");
    assert_eq!(comp.physics.len(), 1);
}

#[test]
fn parses_beliefs_about_all_valid_fields() {
    // Verify every field in the allowlist is accepted.
    for field in &[
        "last_known_pos",
        "last_known_hp",
        "last_known_max_hp",
        "last_known_creature_type",
        "last_updated_tick",
        "confidence",
    ] {
        let src = physics_src(&format!("beliefs(a).about(t).{field}"));
        compile(&src).unwrap_or_else(|e| panic!("field `{field}` rejected: {e}"));
    }
}

#[test]
fn beliefs_accessor_ir_variant() {
    // Parse a minimal physics rule with a `beliefs(a).about(t).last_known_hp`
    // read and confirm the IR carries `IrExpr::BeliefsAccessor`.
    let src = physics_src("beliefs(a).about(t).last_known_hp");
    let comp = compile(&src).expect("compile");
    let body = &comp.physics[0].handlers[0].body;
    // The let-binding wraps the expr.
    match &body[0] {
        IrStmt::Let { value, .. } => match &value.kind {
            IrExpr::BeliefsAccessor { field, .. } => {
                assert_eq!(field, "last_known_hp");
            }
            other => panic!("expected IrExpr::BeliefsAccessor, got {other:?}"),
        },
        other => panic!("expected IrStmt::Let, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Happy-path: BeliefsConfidence
// ---------------------------------------------------------------------------

#[test]
fn parses_beliefs_confidence() {
    let src = physics_src("beliefs(a).confidence(t)");
    let comp = compile(&src).expect("compile BeliefsConfidence");
    let body = &comp.physics[0].handlers[0].body;
    match &body[0] {
        IrStmt::Let { value, .. } => match &value.kind {
            IrExpr::BeliefsConfidence { .. } => {}
            other => panic!("expected IrExpr::BeliefsConfidence, got {other:?}"),
        },
        other => panic!("expected IrStmt::Let, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Happy-path: BeliefsView
// ---------------------------------------------------------------------------

#[test]
fn parses_beliefs_view() {
    let src = physics_src("beliefs(a).all_known(_)");
    let comp = compile(&src).expect("compile BeliefsView");
    let body = &comp.physics[0].handlers[0].body;
    match &body[0] {
        IrStmt::Let { value, .. } => match &value.kind {
            IrExpr::BeliefsView { view_name, .. } => {
                assert_eq!(view_name, "all_known");
            }
            other => panic!("expected IrExpr::BeliefsView, got {other:?}"),
        },
        other => panic!("expected IrStmt::Let, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Error: unknown belief field
// ---------------------------------------------------------------------------

#[test]
fn rejects_unknown_belief_field() {
    let src = physics_src("beliefs(a).about(t).bogus_field");
    let err = compile(&src).expect_err("should fail on unknown field");
    let msg = err.to_string();
    assert!(
        msg.contains("bogus_field"),
        "error should mention the unknown field name; got: {msg}"
    );
    assert!(
        msg.contains("last_known_pos"),
        "error should suggest valid fields; got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Statement form still works (regression guard)
// ---------------------------------------------------------------------------

#[test]
fn belief_observe_stmt_still_parses() {
    let src = r#"
event AgentMoved {
    actor: AgentId,
    location: Vec3,
}

physics observe_move {
    on AgentMoved { actor: a, location: loc } {
        beliefs(a).observe(a) with {
            last_known_pos: loc,
            confidence: 1.0,
        }
    }
}
"#;
    let comp = compile(src).expect("compile BeliefObserve statement");
    match &comp.physics[0].handlers[0].body[0] {
        IrStmt::BeliefObserve { fields, .. } => {
            assert_eq!(fields.len(), 2);
        }
        other => panic!("expected IrStmt::BeliefObserve, got {other:?}"),
    }
}
