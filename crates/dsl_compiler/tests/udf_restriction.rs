//! Negative + positive tests pinning the UDF-restriction contract for
//! `for` and `match` across body contexts. Task 155 landed `for` / `match`
//! lowering in physics handlers (commit `9ba805c6`); this file keeps the
//! mask / scoring / view-fold contexts honest: they compile to GPU-safe
//! kernels and must reject unbounded iteration + pattern-matching control
//! flow, citing the closed operator set.

use dsl_compiler::{CompileError, ResolveError};

// ---------------------------------------------------------------------------
// Negative: mask bodies reject `for` and `match`.
// ---------------------------------------------------------------------------

#[test]
fn mask_body_with_for_is_rejected() {
    // A `for` statement has no expression form, so the attempt to thread
    // it through the mask's predicate slot is caught by either the parser
    // (no `for` in expression position) or the resolver. Either way, the
    // error must mention `mask` and `for` so the author knows which
    // restriction was violated.
    let src = r#"
        mask Attack(t) when
          for x in agents.alive { x.alive }
    "#;
    let err = dsl_compiler::compile(src).expect_err("mask predicate with `for` must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("mask") && (msg.contains("`for`") || msg.contains("for ")),
        "expected mask `for` rejection diagnostic; got: {msg}",
    );
}

#[test]
fn mask_body_with_match_is_rejected() {
    // `match` *is* a valid expression, so it parses and reaches the
    // resolver. The mask-body validator must reject it with a clear UDF
    // restriction error pointing at the `match` span.
    let src = r#"
        enum Mood { Calm, Angry }
        mask Attack(t) when
          match t.mood { Mood::Calm => false, Mood::Angry => true }
    "#;
    let err = dsl_compiler::compile(src).expect_err("mask predicate with `match` must reject");
    match err {
        CompileError::Resolve(ResolveError::UdfInMaskBody {
            mask_name,
            offending_construct,
            ..
        }) => {
            assert_eq!(mask_name, "Attack");
            assert!(
                offending_construct.contains("match"),
                "offending construct should mention `match`; got: {offending_construct}",
            );
        }
        other => panic!("expected UdfInMaskBody(match); got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Negative: scoring bodies reject `for` and `match`.
// ---------------------------------------------------------------------------

#[test]
fn scoring_body_with_for_is_rejected() {
    let src = r#"
        scoring {
          Attack(t) = for x in agents.alive { x.alive }
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("scoring entry with `for` must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("scoring") || msg.contains("for"),
        "expected scoring `for` rejection diagnostic; got: {msg}",
    );
}

#[test]
fn scoring_body_with_match_is_rejected() {
    let src = r#"
        enum Mood { Calm, Angry }
        scoring {
          Attack(t) = match t.mood { Mood::Calm => 0.1, Mood::Angry => 0.9 }
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("scoring entry with `match` must reject");
    match err {
        CompileError::Resolve(ResolveError::UdfInScoringBody {
            offending_construct,
            ..
        }) => {
            assert!(
                offending_construct.contains("match"),
                "offending construct should mention `match`; got: {offending_construct}",
            );
        }
        other => panic!("expected UdfInScoringBody(match); got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Negative: view fold bodies reject `for` and `match`.
// (Task 135 already landed the view-fold validator; pin it here too.)
// ---------------------------------------------------------------------------

#[test]
fn view_fold_body_with_for_is_rejected() {
    // Pre-existing validator hit: `for` statement inside a fold-handler
    // body is an unbounded loop, forbidden by spec §2.3.
    let src = r#"
        @materialized(on_event = [AgentAttacked])
        view mood(a: Agent) -> f32 {
            initial: 0.0,
            on AgentAttacked{target: a} {
                for x in agents.alive { self += 0.1 }
            }
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("fold-body `for` must reject");
    match err {
        CompileError::Resolve(ResolveError::UdfInViewFoldBody {
            view_name,
            offending_construct,
            ..
        }) => {
            assert_eq!(view_name, "mood");
            assert!(
                offending_construct.contains("for"),
                "offending construct should mention `for`; got: {offending_construct}",
            );
        }
        other => panic!("expected UdfInViewFoldBody(for); got: {other:?}"),
    }
}

#[test]
fn view_fold_body_with_match_is_rejected() {
    // Fold bodies compile to a single commutative `apply_event` pass over
    // a persisted scalar; `match` control flow over event variants is
    // forbidden (the outer handler-pattern dispatch already names the
    // variant — inner `match` would be cross-variant plumbing).
    let src = r#"
        enum Mood { Calm, Angry }
        @materialized(on_event = [AgentAttacked])
        view mood(a: Agent) -> f32 {
            initial: 0.0,
            on AgentAttacked{target: a} {
                match Mood::Calm {
                    Mood::Calm => { self += 0.1 }
                    Mood::Angry => { self -= 0.1 }
                }
            }
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("fold-body `match` must reject");
    match err {
        CompileError::Resolve(ResolveError::UdfInViewFoldBody {
            view_name,
            offending_construct,
            ..
        }) => {
            assert_eq!(view_name, "mood");
            assert!(
                offending_construct.contains("match"),
                "offending construct should mention `match`; got: {offending_construct}",
            );
        }
        other => panic!("expected UdfInViewFoldBody(match); got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Positive: physics bodies accept `for` and `match` (task 155 contract).
// ---------------------------------------------------------------------------

#[test]
fn physics_body_with_for_compiles() {
    // Task 155 landed `for <binder> in <iter>` lowering in physics
    // handlers. Physics bodies run as CPU sequential cascade handlers;
    // the UDF restriction doesn't apply. Compile must succeed and the
    // resolved IR must carry an `IrStmt::For` inside the handler body.
    // Mirrors the shape of `for op in abilities.effects(ab)` from the
    // `cast` rule in `assets/sim/physics.sim` — task 155's canonical
    // example proving physics `for` works end-to-end.
    let src = r#"
        event Ping { actor: AgentId, ability: AbilityId }
        physics demo @phase(event) @terminating_in(1) {
            on Ping { actor: a, ability: ab } {
                for op in abilities.effects(ab) {
                    emit Ping { actor: a, ability: ab }
                }
            }
        }
    "#;
    let comp = dsl_compiler::compile(src).expect("physics `for` must compile");
    assert_eq!(comp.physics.len(), 1);
    let handler = &comp.physics[0].handlers[0];
    assert!(
        handler.body.iter().any(|s| matches!(s, dsl_compiler::ir::IrStmt::For { .. })),
        "expected an IrStmt::For in physics handler body: {:?}",
        handler.body,
    );
}

#[test]
fn physics_body_with_match_compiles() {
    // Task 155 also landed stdlib-enum-variant `match` lowering. A
    // physics body dispatching over a known enum must compile.
    let src = r#"
        enum Mood { Calm, Angry }
        event Ping { actor: AgentId, mood: Mood }
        physics demo @phase(event) @terminating_in(1) {
            on Ping { actor: a, mood: m } {
                match m {
                    Mood::Calm => { emit Ping { actor: a, mood: Mood::Calm } }
                    Mood::Angry => { emit Ping { actor: a, mood: Mood::Angry } }
                }
            }
        }
    "#;
    let comp = dsl_compiler::compile(src).expect("physics `match` must compile");
    let handler = &comp.physics[0].handlers[0];
    assert!(
        handler.body.iter().any(|s| matches!(s, dsl_compiler::ir::IrStmt::Match { .. })),
        "expected an IrStmt::Match in physics handler body: {:?}",
        handler.body,
    );
}
