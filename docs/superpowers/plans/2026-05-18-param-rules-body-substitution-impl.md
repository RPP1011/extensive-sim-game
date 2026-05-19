# Parameterised Rules — Body Substitution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the deferred body-substitution pass for parameterised rules so applying a template actually substitutes parameter references in the body with the application's argument values. Until this lands, the parameterised-rule feature can only ship empty-body templates (the chase.sim seed currently has no body).

**Architecture:** Two pure-functional AST walkers — `substitute_expr(&Expr, &HashMap<&str, ApplyArgValue>) -> Expr` and `substitute_stmt(&Stmt, &HashMap<&str, ApplyArgValue>) -> Stmt`. Walk every variant; the base case is `ExprKind::Ident(name)` — if `name` matches a param key, replace with a literal Expr derived from the arg value. Binder-introducing constructs (Quantifier / Fold / For / ForEachAgent / Block) shadow params by removing the binder name from the map before recursing into bodies. Wire the walkers into `monomorphise` (replacing the body-clone with a full substitution).

**Tech Stack:** Rust workspace; pure-functional walks in `crates/dsl_compiler/src/cg/lower/param_rules.rs`; tests in the same crate's `tests/` directory.

**Source spec:** `docs/superpowers/specs/2026-05-17-parameterised-rules-design.md` (the parameterised-rules spec's "Monomorphisation & lowering" section described this pass; T8 of the v1 plan deferred it).

---

## Architectural Impact Statement

- **Existing primitives searched:**
  - `dsl_ast::ast::Stmt` enum at `crates/dsl_ast/src/ast.rs:1179` (11 variants).
  - `dsl_ast::ast::Expr { kind: ExprKind, span: Span }` at `crates/dsl_ast/src/ast.rs:1325`.
  - `dsl_ast::ast::ExprKind` at `crates/dsl_ast/src/ast.rs:1332` (22 variants).
  - `dsl_compiler::cg::lower::param_rules::monomorphise` at `crates/dsl_compiler/src/cg/lower/param_rules.rs` (currently clones template body verbatim with a `TODO(param-rules-v2)` comment).
  - `ApplyArgValue` enum at `crates/dsl_ast/src/ast.rs` (F32, I32, U32, Bool, EntityKind).
  - Search: `rg`, direct `Read`.

- **Decision:** add `substitute_expr` + `substitute_stmt` helpers inside `param_rules.rs` and replace the body-clone in `monomorphise` with a full walk. No new public surface beyond `monomorphise`'s behaviour change.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: `crates/dsl_compiler/src/cg/lower/param_rules.rs` (substitution functions + wire into monomorphise).
  - Generated outputs re-emitted: `OUT_DIR/<fixture>/{generated.rs, runtime_core.rs}` for any fixture that uses parameterised rules — currently only `param_rule_smoke`.
  - Stdlib update: `stdlib/rules/chase.sim` body fleshed out to actually use params.

- **Hand-written downstream code:** NONE.

- **Constitution check:**
  - P1 (Compiler-First): PASS — substitution is compiler work, no rule-handler hand-writing.
  - P2 (Schema-Hash): N/A — no SimState SoA field changes.
  - P3 (Cross-Backend Parity): PASS — substitution runs before backend selection.
  - P4 (`EffectOp` Size): N/A.
  - P5 (Determinism via Keyed PCG): PASS — substitution is a pure-functional AST transformation; no RNG, no FS, no time. HashMap usage is `.get()` / `.remove()` only; no iteration.
  - P6 (Events Are the Mutation Channel): N/A.
  - P7 (Replayability Flagged): N/A.
  - P8 (AIS Required): PASS — this section.
  - P9 (Tasks Close With Verified Commit): PASS.
  - P10 (No Runtime Panic): PASS — substitution is build-time; errors are Results.
  - P11 (Reduction Determinism): N/A.

- **Runtime gate:** Task 4 fleshes out `stdlib/rules/chase.sim` with a real body that references params + Task 5 updates `assets/sim/param_rule_smoke.sim` to invoke it. The existing `param_rule_smoke` test in `crates/sims/tests/` continues to verify the module compiles end-to-end — which now requires substitution to be correct (otherwise the cloned body's unbound `aggro` / `target` / `speed` identifiers would fail the resolver). The build succeeding IS the runtime gate.

- **Re-evaluation:** [x] AIS reviewed at design phase.  [ ] AIS reviewed post-design.

---

## Files touched

- Modify: `crates/dsl_compiler/src/cg/lower/param_rules.rs` — add `substitute_expr`, `substitute_stmt`, `apply_arg_to_expr_kind`; replace `handlers.clone()` in `monomorphise` with the substituting walk.
- Modify: `stdlib/rules/chase.sim` — replace the empty body with a real body that references `aggro`, `target`, `speed`.
- (Optional) Modify: `assets/sim/param_rule_smoke.sim` — no change needed; the fixture already applies chase twice. The build going green with the real chase body IS the verification.
- Create: `crates/dsl_compiler/tests/param_rule_substitute.rs` — focused unit tests on the substitution behaviour.

---

## Task 1: `apply_arg_to_expr_kind` + `substitute_expr` (the Expr walker)

**Files:**
- Modify: `crates/dsl_compiler/src/cg/lower/param_rules.rs`
- Test: `crates/dsl_compiler/tests/param_rule_substitute.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_compiler/tests/param_rule_substitute.rs
use std::collections::HashMap;
use dsl_compiler::lower::param_rules::{substitute_expr, apply_arg_to_expr_kind};
use dsl_ast::ast::{Expr, ExprKind, ApplyArgValue, BinOp, Span};

fn ident(name: &str) -> Expr {
    Expr { kind: ExprKind::Ident(name.to_string()), span: Span::dummy() }
}

fn float_lit(v: f64) -> Expr {
    Expr { kind: ExprKind::Float(v), span: Span::dummy() }
}

fn args_map<'a>(items: &'a [(&'a str, ApplyArgValue)]) -> HashMap<&'a str, ApplyArgValue> {
    items.iter().cloned().collect()
}

#[test]
fn ident_matching_param_is_replaced_with_literal() {
    let expr = ident("aggro");
    let args = args_map(&[("aggro", ApplyArgValue::F32(15.0))]);
    let out = substitute_expr(&expr, &args);
    assert!(matches!(out.kind, ExprKind::Float(15.0)));
}

#[test]
fn ident_not_matching_param_is_unchanged() {
    let expr = ident("self");
    let args = args_map(&[("aggro", ApplyArgValue::F32(15.0))]);
    let out = substitute_expr(&expr, &args);
    assert!(matches!(out.kind, ExprKind::Ident(ref s) if s == "self"));
}

#[test]
fn binary_op_recurses_into_both_sides() {
    // aggro * aggro → 15.0 * 15.0
    let expr = Expr {
        kind: ExprKind::Binary {
            op: BinOp::Mul,
            lhs: Box::new(ident("aggro")),
            rhs: Box::new(ident("aggro")),
        },
        span: Span::dummy(),
    };
    let args = args_map(&[("aggro", ApplyArgValue::F32(15.0))]);
    let out = substitute_expr(&expr, &args);
    match out.kind {
        ExprKind::Binary { lhs, rhs, .. } => {
            assert!(matches!(lhs.kind, ExprKind::Float(15.0)));
            assert!(matches!(rhs.kind, ExprKind::Float(15.0)));
        }
        other => panic!("expected Binary, got {other:?}"),
    }
}

#[test]
fn entity_kind_arg_substitutes_as_ident() {
    // target → Wolf  (the resolver will later bind "Wolf" to an entity decl).
    let expr = ident("target");
    let args = args_map(&[("target", ApplyArgValue::EntityKind("Wolf".into()))]);
    let out = substitute_expr(&expr, &args);
    match out.kind {
        ExprKind::Ident(ref name) => assert_eq!(name, "Wolf"),
        other => panic!("expected Ident, got {other:?}"),
    }
}

#[test]
fn bool_arg_substitutes_to_bool_literal() {
    let expr = ident("flag");
    let args = args_map(&[("flag", ApplyArgValue::Bool(true))]);
    let out = substitute_expr(&expr, &args);
    assert!(matches!(out.kind, ExprKind::Bool(true)));
}

#[test]
fn int_arg_substitutes_to_int_literal() {
    let expr = ident("count");
    let args = args_map(&[("count", ApplyArgValue::I32(42))]);
    let out = substitute_expr(&expr, &args);
    assert!(matches!(out.kind, ExprKind::Int(42)));
}

#[test]
fn quantifier_binder_shadows_outer_param() {
    // forall aggro in foo: aggro > 0
    // The inner `aggro` is a binder, NOT a param — must NOT be substituted.
    let expr = Expr {
        kind: ExprKind::Quantifier {
            kind: dsl_ast::ast::QuantKind::Forall,
            binder: "aggro".into(),
            iter: Box::new(ident("foo")),
            body: Box::new(Expr {
                kind: ExprKind::Binary {
                    op: BinOp::Gt,
                    lhs: Box::new(ident("aggro")),  // binder use, not param
                    rhs: Box::new(Expr { kind: ExprKind::Int(0), span: Span::dummy() }),
                },
                span: Span::dummy(),
            }),
        },
        span: Span::dummy(),
    };
    let args = args_map(&[("aggro", ApplyArgValue::F32(15.0))]);
    let out = substitute_expr(&expr, &args);
    match out.kind {
        ExprKind::Quantifier { body, .. } => match body.kind {
            ExprKind::Binary { lhs, .. } => {
                // Inner aggro should still be Ident("aggro"), NOT Float(15.0).
                assert!(matches!(lhs.kind, ExprKind::Ident(ref s) if s == "aggro"),
                        "binder should shadow outer param, got: {:?}", lhs.kind);
            }
            other => panic!("expected Binary inside Quantifier body, got {other:?}"),
        }
        other => panic!("expected Quantifier, got {other:?}"),
    }
}

#[test]
fn apply_arg_to_expr_kind_covers_all_variants() {
    assert!(matches!(apply_arg_to_expr_kind(&ApplyArgValue::F32(1.5)), ExprKind::Float(_)));
    assert!(matches!(apply_arg_to_expr_kind(&ApplyArgValue::I32(7)), ExprKind::Int(7)));
    assert!(matches!(apply_arg_to_expr_kind(&ApplyArgValue::U32(7)), ExprKind::Int(7)));
    assert!(matches!(apply_arg_to_expr_kind(&ApplyArgValue::Bool(true)), ExprKind::Bool(true)));
    assert!(matches!(apply_arg_to_expr_kind(&ApplyArgValue::EntityKind("Wolf".into())),
                     ExprKind::Ident(_)));
}
```

(`Span::dummy()` was added in PR-T1 of the previous parameterised-rules plan; it's already available. If `dsl_ast::ast::Span` isn't re-exported, the test imports `dsl_ast::ast::Span` directly which works.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test param_rule_substitute`
Expected: FAIL — `substitute_expr` and `apply_arg_to_expr_kind` are private / don't exist yet.

- [ ] **Step 3: Implement substitute_expr + apply_arg_to_expr_kind**

In `crates/dsl_compiler/src/cg/lower/param_rules.rs`, append:

```rust
use dsl_ast::ast::{Expr, ExprKind};

/// Convert an `ApplyArgValue` into the corresponding `ExprKind` literal.
/// Used by `substitute_expr` when it finds an `Ident` matching a param name.
///
/// EntityKind args become `Ident(<entity_name>)` — the resolver will later
/// bind the entity name to its decl just like any other entity reference.
pub fn apply_arg_to_expr_kind(value: &ApplyArgValue) -> ExprKind {
    match value {
        ApplyArgValue::F32(v) => ExprKind::Float(*v as f64),
        ApplyArgValue::I32(v) => ExprKind::Int(*v as i64),
        ApplyArgValue::U32(v) => ExprKind::Int(*v as i64),
        ApplyArgValue::Bool(v) => ExprKind::Bool(*v),
        ApplyArgValue::EntityKind(name) => ExprKind::Ident(name.clone()),
    }
}

/// Substitute parameter references in an expression tree with the applied
/// arg values. Binder-introducing constructs (Quantifier / Fold / Block)
/// shadow params by removing the binder name from the map before recursing
/// into the body.
pub fn substitute_expr<'a>(expr: &Expr, args: &HashMap<&'a str, ApplyArgValue>) -> Expr {
    let new_kind = match &expr.kind {
        // ----- Base cases: literals are unchanged. -----
        ExprKind::Int(v)    => ExprKind::Int(*v),
        ExprKind::Float(v)  => ExprKind::Float(*v),
        ExprKind::Bool(v)   => ExprKind::Bool(*v),
        ExprKind::String(s) => ExprKind::String(s.clone()),

        // ----- THE substitution point. -----
        ExprKind::Ident(name) => {
            if let Some(value) = args.get(name.as_str()) {
                apply_arg_to_expr_kind(value)
            } else {
                ExprKind::Ident(name.clone())
            }
        }

        // ----- Pure-recursive variants. -----
        ExprKind::Field(inner, field) => ExprKind::Field(
            Box::new(substitute_expr(inner, args)),
            field.clone(),
        ),
        ExprKind::Index(a, b) => ExprKind::Index(
            Box::new(substitute_expr(a, args)),
            Box::new(substitute_expr(b, args)),
        ),
        ExprKind::Call(callee, call_args) => ExprKind::Call(
            Box::new(substitute_expr(callee, args)),
            call_args.iter().map(|ca| dsl_ast::ast::CallArg {
                name: ca.name.clone(),
                value: substitute_expr(&ca.value, args),
                span: ca.span,
            }).collect(),
        ),
        ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary {
            op: *op,
            lhs: Box::new(substitute_expr(lhs, args)),
            rhs: Box::new(substitute_expr(rhs, args)),
        },
        ExprKind::Unary { op, rhs } => ExprKind::Unary {
            op: *op,
            rhs: Box::new(substitute_expr(rhs, args)),
        },
        ExprKind::In { item, set } => ExprKind::In {
            item: Box::new(substitute_expr(item, args)),
            set: Box::new(substitute_expr(set, args)),
        },
        ExprKind::Contains { set, item } => ExprKind::Contains {
            set: Box::new(substitute_expr(set, args)),
            item: Box::new(substitute_expr(item, args)),
        },
        ExprKind::List(items) => ExprKind::List(
            items.iter().map(|e| substitute_expr(e, args)).collect()
        ),
        ExprKind::Tuple(items) => ExprKind::Tuple(
            items.iter().map(|e| substitute_expr(e, args)).collect()
        ),
        ExprKind::Struct { name, fields } => ExprKind::Struct {
            name: name.clone(),
            fields: fields.iter().map(|fi| dsl_ast::ast::FieldInit {
                name: fi.name.clone(),
                value: substitute_expr(&fi.value, args),
                span: fi.span,
            }).collect(),
        },
        ExprKind::Ctor { name, args: ctor_args } => ExprKind::Ctor {
            name: name.clone(),
            args: ctor_args.iter().map(|e| substitute_expr(e, args)).collect(),
        },
        ExprKind::Match { scrutinee, arms } => ExprKind::Match {
            scrutinee: Box::new(substitute_expr(scrutinee, args)),
            arms: arms.iter().map(|a| dsl_ast::ast::MatchExprArm {
                pattern: a.pattern.clone(),
                body: substitute_expr(&a.body, args),
                span: a.span,
            }).collect(),
        },
        ExprKind::If { cond, then_expr, else_expr } => ExprKind::If {
            cond: Box::new(substitute_expr(cond, args)),
            then_expr: Box::new(substitute_expr(then_expr, args)),
            else_expr: else_expr.as_ref().map(|e| Box::new(substitute_expr(e, args))),
        },
        ExprKind::PerUnit { expr: inner, delta } => ExprKind::PerUnit {
            expr: Box::new(substitute_expr(inner, args)),
            delta: Box::new(substitute_expr(delta, args)),
        },
        ExprKind::BeliefsAccessor { observer, target, field } => ExprKind::BeliefsAccessor {
            observer: Box::new(substitute_expr(observer, args)),
            target: Box::new(substitute_expr(target, args)),
            field: field.clone(),
        },
        ExprKind::BeliefsConfidence { observer, target } => ExprKind::BeliefsConfidence {
            observer: Box::new(substitute_expr(observer, args)),
            target: Box::new(substitute_expr(target, args)),
        },
        ExprKind::BeliefsView { observer, view_name } => ExprKind::BeliefsView {
            observer: Box::new(substitute_expr(observer, args)),
            view_name: view_name.clone(),
        },

        // ----- Binder-introducing variants (shadow params). -----
        ExprKind::Quantifier { kind, binder, iter, body } => {
            let new_iter = substitute_expr(iter, args);
            let mut inner_args = args.clone();
            inner_args.remove(binder.as_str());
            let new_body = substitute_expr(body, &inner_args);
            ExprKind::Quantifier {
                kind: *kind,
                binder: binder.clone(),
                iter: Box::new(new_iter),
                body: Box::new(new_body),
            }
        }
        ExprKind::Fold { kind, binder, iter, body } => {
            let new_iter = iter.as_ref().map(|i| Box::new(substitute_expr(i, args)));
            let mut inner_args = args.clone();
            if let Some(b) = binder.as_ref() {
                inner_args.remove(b.as_str());
            }
            let new_body = substitute_expr(body, &inner_args);
            ExprKind::Fold {
                kind: *kind,
                binder: binder.clone(),
                iter: new_iter,
                body: Box::new(new_body),
            }
        }
        ExprKind::Block { bindings, expr: tail } => {
            // Sequential let-binding semantics: each binding can shadow
            // earlier scope (and subsequent expressions see the shadowed name).
            let mut inner_args = args.clone();
            let new_bindings: Vec<(String, Expr)> = bindings.iter().map(|(name, value)| {
                let new_value = substitute_expr(value, &inner_args);
                inner_args.remove(name.as_str());
                (name.clone(), new_value)
            }).collect();
            let new_tail = substitute_expr(tail, &inner_args);
            ExprKind::Block {
                bindings: new_bindings,
                expr: Box::new(new_tail),
            }
        }
    };
    Expr { kind: new_kind, span: expr.span }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test param_rule_substitute`
Expected: 8 passed.

Also: prior tests still pass.
```
cargo test -p dsl_compiler --test param_rule_mono param_rule_validate_apply 2>&1 | grep "test result"
```

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src/cg/lower/param_rules.rs crates/dsl_compiler/tests/param_rule_substitute.rs
git commit -m "feat(dsl): substitute_expr + apply_arg_to_expr_kind — AST walk for body sub"
```

---

## Task 2: `substitute_stmt` — the Stmt walker

**Files:**
- Modify: `crates/dsl_compiler/src/cg/lower/param_rules.rs`
- Modify: `crates/dsl_compiler/tests/param_rule_substitute.rs` (add Stmt-level tests)

- [ ] **Step 1: Add failing tests at the Stmt level**

Append to `crates/dsl_compiler/tests/param_rule_substitute.rs`:

```rust
use dsl_compiler::lower::param_rules::substitute_stmt;
use dsl_ast::ast::Stmt;

#[test]
fn let_stmt_substitutes_in_value() {
    // let x = aggro;
    let stmt = Stmt::Let {
        name: "x".into(),
        value: ident("aggro"),
        span: Span::dummy(),
    };
    let args = args_map(&[("aggro", ApplyArgValue::F32(15.0))]);
    let out = substitute_stmt(&stmt, &args);
    match out {
        Stmt::Let { value, .. } => assert!(matches!(value.kind, ExprKind::Float(15.0))),
        other => panic!("expected Let, got {other:?}"),
    }
}

#[test]
fn for_each_agent_binder_does_not_shadow_unrelated_param() {
    // for_each_agent a { let _ = aggro; } — `a` doesn't match `aggro`, so
    // `aggro` substitutes normally.
    let stmt = Stmt::ForEachAgent {
        binder: "a".into(),
        body: vec![Stmt::Let {
            name: "_".into(),
            value: ident("aggro"),
            span: Span::dummy(),
        }],
        span: Span::dummy(),
    };
    let args = args_map(&[("aggro", ApplyArgValue::F32(15.0))]);
    let out = substitute_stmt(&stmt, &args);
    match out {
        Stmt::ForEachAgent { body, .. } => match &body[0] {
            Stmt::Let { value, .. } => assert!(matches!(value.kind, ExprKind::Float(15.0))),
            other => panic!("expected inner Let, got {other:?}"),
        }
        other => panic!("expected ForEachAgent, got {other:?}"),
    }
}

#[test]
fn for_each_agent_binder_shadows_matching_param() {
    // for_each_agent aggro { let _ = aggro; } — the inner `aggro` is the
    // binder, NOT the param.
    let stmt = Stmt::ForEachAgent {
        binder: "aggro".into(),
        body: vec![Stmt::Let {
            name: "_".into(),
            value: ident("aggro"),
            span: Span::dummy(),
        }],
        span: Span::dummy(),
    };
    let args = args_map(&[("aggro", ApplyArgValue::F32(15.0))]);
    let out = substitute_stmt(&stmt, &args);
    match out {
        Stmt::ForEachAgent { body, .. } => match &body[0] {
            Stmt::Let { value, .. } => assert!(matches!(value.kind, ExprKind::Ident(ref s) if s == "aggro"),
                                               "binder should shadow param"),
            other => panic!("expected inner Let, got {other:?}"),
        }
        other => panic!("expected ForEachAgent, got {other:?}"),
    }
}

#[test]
fn if_stmt_substitutes_in_cond_and_bodies() {
    // if aggro > 0.0 { let _ = aggro; } else { let _ = aggro; }
    let stmt = Stmt::If {
        cond: Expr {
            kind: ExprKind::Binary {
                op: BinOp::Gt,
                lhs: Box::new(ident("aggro")),
                rhs: Box::new(float_lit(0.0)),
            },
            span: Span::dummy(),
        },
        then_body: vec![Stmt::Let {
            name: "_".into(),
            value: ident("aggro"),
            span: Span::dummy(),
        }],
        else_body: Some(vec![Stmt::Let {
            name: "_".into(),
            value: ident("aggro"),
            span: Span::dummy(),
        }]),
        span: Span::dummy(),
    };
    let args = args_map(&[("aggro", ApplyArgValue::F32(15.0))]);
    let out = substitute_stmt(&stmt, &args);
    match out {
        Stmt::If { cond, then_body, else_body, .. } => {
            match cond.kind {
                ExprKind::Binary { lhs, .. } =>
                    assert!(matches!(lhs.kind, ExprKind::Float(15.0))),
                _ => panic!("cond should be Binary"),
            }
            match &then_body[0] {
                Stmt::Let { value, .. } =>
                    assert!(matches!(value.kind, ExprKind::Float(15.0))),
                _ => panic!("then_body[0] should be Let"),
            }
            match &else_body.as_ref().unwrap()[0] {
                Stmt::Let { value, .. } =>
                    assert!(matches!(value.kind, ExprKind::Float(15.0))),
                _ => panic!("else_body[0] should be Let"),
            }
        }
        other => panic!("expected If, got {other:?}"),
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test param_rule_substitute`
Expected: FAIL — `substitute_stmt` doesn't exist.

- [ ] **Step 3: Implement substitute_stmt**

Append to `crates/dsl_compiler/src/cg/lower/param_rules.rs`:

```rust
use dsl_ast::ast::Stmt;

pub fn substitute_stmt<'a>(stmt: &Stmt, args: &HashMap<&'a str, ApplyArgValue>) -> Stmt {
    match stmt {
        Stmt::Let { name, value, span } => Stmt::Let {
            name: name.clone(),
            value: substitute_expr(value, args),
            span: *span,
        },
        Stmt::Emit(es) => Stmt::Emit(dsl_ast::ast::EmitStmt {
            event_name: es.event_name.clone(),
            fields: es.fields.iter().map(|fi| dsl_ast::ast::FieldInit {
                name: fi.name.clone(),
                value: substitute_expr(&fi.value, args),
                span: fi.span,
            }).collect(),
            span: es.span,
        }),
        Stmt::For { binder, iter, filter, body, span } => {
            let new_iter = substitute_expr(iter, args);
            let new_filter = filter.as_ref().map(|f| substitute_expr(f, args));
            let mut inner_args = args.clone();
            inner_args.remove(binder.as_str());
            let new_body = body.iter().map(|s| substitute_stmt(s, &inner_args)).collect();
            Stmt::For {
                binder: binder.clone(),
                iter: new_iter,
                filter: new_filter,
                body: new_body,
                span: *span,
            }
        }
        Stmt::ForEachAgent { binder, body, span } => {
            let mut inner_args = args.clone();
            inner_args.remove(binder.as_str());
            let new_body = body.iter().map(|s| substitute_stmt(s, &inner_args)).collect();
            Stmt::ForEachAgent {
                binder: binder.clone(),
                body: new_body,
                span: *span,
            }
        }
        Stmt::If { cond, then_body, else_body, span } => Stmt::If {
            cond: substitute_expr(cond, args),
            then_body: then_body.iter().map(|s| substitute_stmt(s, args)).collect(),
            else_body: else_body.as_ref().map(|eb|
                eb.iter().map(|s| substitute_stmt(s, args)).collect()
            ),
            span: *span,
        },
        Stmt::Match { scrutinee, arms, span } => Stmt::Match {
            scrutinee: substitute_expr(scrutinee, args),
            arms: arms.iter().map(|a| dsl_ast::ast::MatchArm {
                pattern: a.pattern.clone(),
                body: a.body.iter().map(|s| substitute_stmt(s, args)).collect(),
                span: a.span,
            }).collect(),
            span: *span,
        },
        Stmt::SelfUpdate { op, value, span } => Stmt::SelfUpdate {
            op: op.clone(),
            value: substitute_expr(value, args),
            span: *span,
        },
        Stmt::SelfAppend { fields, span } => Stmt::SelfAppend {
            fields: fields.iter().map(|fi| dsl_ast::ast::FieldInit {
                name: fi.name.clone(),
                value: substitute_expr(&fi.value, args),
                span: fi.span,
            }).collect(),
            span: *span,
        },
        Stmt::Expr(e) => Stmt::Expr(substitute_expr(e, args)),
        Stmt::BeliefObserve(b) => Stmt::BeliefObserve(dsl_ast::ast::BeliefObserveStmt {
            observer: b.observer.clone(),
            target: b.target.clone(),
            fields: b.fields.iter().map(|fi| dsl_ast::ast::FieldInit {
                name: fi.name.clone(),
                value: substitute_expr(&fi.value, args),
                span: fi.span,
            }).collect(),
            span: b.span,
        }),
        Stmt::ApplyAbility(a) => Stmt::ApplyAbility(dsl_ast::ast::ApplyAbilityStmt {
            ability: substitute_expr(&a.ability, args),
            ability_name: a.ability_name.clone(),
            caster: a.caster.as_ref().map(|c| substitute_expr(c, args)),
            target: a.target.as_ref().map(|t| substitute_expr(t, args)),
            span: a.span,
        }),
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test param_rule_substitute`
Expected: 12 passed (8 from T1 + 4 from this task).

Also: prior tests still pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src/cg/lower/param_rules.rs crates/dsl_compiler/tests/param_rule_substitute.rs
git commit -m "feat(dsl): substitute_stmt — Stmt walker with binder shadowing for for/for_each_agent"
```

---

## Task 3: Wire substitution into `monomorphise`

**Files:**
- Modify: `crates/dsl_compiler/src/cg/lower/param_rules.rs` — replace `handlers.clone()` with the substituting walk.
- Modify: `crates/dsl_compiler/tests/param_rule_mono.rs` — add a test that verifies body substitution end-to-end.

- [ ] **Step 1: Write the failing test (end-to-end substitution)**

Append to `crates/dsl_compiler/tests/param_rule_mono.rs`:

```rust
use dsl_ast::ast::ExprKind;

#[test]
fn monomorphise_substitutes_param_refs_in_body() {
    // The template body uses `aggro` and `target` directly. After
    // monomorphisation, the concrete rule's body should contain literal
    // 15.0 (for aggro) and Ident("Wolf") (for target), NOT the original
    // Ident("aggro") / Ident("target").
    let src = r#"
entity Wolf : Agent {}

physics chase(target: EntityKind, aggro: f32) @phase(per_agent) {
  on Tick {} {
    let aggro_squared = aggro * aggro;
    let target_kind = target;
  }
}

physics HunterChase = chase(target: Wolf, aggro: 15.0);
"#;
    let mut program = parse(src).expect("parse");
    monomorphise(&mut program).expect("ok");

    let hunter = program.decls.iter().find_map(|d| match d {
        Decl::Physics(p) if p.name == "HunterChase" => Some(p),
        _ => None,
    }).expect("HunterChase should exist");

    let body = &hunter.handlers[0].body;
    // First stmt: let aggro_squared = aggro * aggro
    let first = &body[0];
    match first {
        dsl_ast::ast::Stmt::Let { name, value, .. } => {
            assert_eq!(name, "aggro_squared");
            match &value.kind {
                ExprKind::Binary { lhs, rhs, .. } => {
                    assert!(matches!(lhs.kind, ExprKind::Float(15.0)),
                            "lhs should be Float(15.0), got {:?}", lhs.kind);
                    assert!(matches!(rhs.kind, ExprKind::Float(15.0)),
                            "rhs should be Float(15.0), got {:?}", rhs.kind);
                }
                other => panic!("expected Binary, got {other:?}"),
            }
        }
        other => panic!("expected first stmt to be Let, got {other:?}"),
    }
    // Second stmt: let target_kind = target
    // After substitution, target → Ident("Wolf").
    let second = &body[1];
    match second {
        dsl_ast::ast::Stmt::Let { name, value, .. } => {
            assert_eq!(name, "target_kind");
            match &value.kind {
                ExprKind::Ident(s) => assert_eq!(s, "Wolf"),
                other => panic!("expected Ident, got {other:?}"),
            }
        }
        other => panic!("expected second stmt to be Let, got {other:?}"),
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test param_rule_mono`
Expected: FAIL — current monomorphise clones the body verbatim; the substituted Float / Ident("Wolf") values aren't in the output.

- [ ] **Step 3: Replace `handlers.clone()` with the substituting walk**

In `crates/dsl_compiler/src/cg/lower/param_rules.rs`, find the line:

```rust
let handlers: Vec<PhysicsHandler> = template.handlers.clone();
```

Replace with:

```rust
// Build the per-decl substitution map.
let arg_map: HashMap<&str, ApplyArgValue> = apply.args.iter()
    .map(|a| (a.name.as_str(), a.value.clone()))
    .collect();
// Substitute param refs in every handler's where_clause + body.
let handlers: Vec<PhysicsHandler> = template.handlers.iter().map(|h| {
    PhysicsHandler {
        pattern: h.pattern.clone(),
        where_clause: h.where_clause.as_ref().map(|w| substitute_expr(w, &arg_map)),
        body: h.body.iter().map(|s| substitute_stmt(s, &arg_map)).collect(),
        span: h.span,
    }
}).collect();
```

Also remove the `TODO(param-rules-v2)` comment that flagged this deferral.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test param_rule_mono`
Expected: 4 passed (3 from before + 1 new).

Also run all prior param_rule tests + substitution tests: `cargo test -p dsl_compiler --test param_rule_substitute param_rule_mono param_rule_validate_apply param_rule_validate_decl param_rule_collision param_rule_parse_apply param_rule_parse_decl 2>&1 | grep "test result"`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src/cg/lower/param_rules.rs crates/dsl_compiler/tests/param_rule_mono.rs
git commit -m "feat(dsl): monomorphise substitutes param refs in handler bodies"
```

---

## Task 4: Flesh out `stdlib/rules/chase.sim` with a real body

**Files:**
- Modify: `stdlib/rules/chase.sim`

- [ ] **Step 1: Replace the empty body with one that uses params**

```text
// stdlib/rules/chase.sim
// Chase a target entity within an aggro radius.
// Imported via `import std/rules/chase.sim;` and applied via
// `physics MyChase = chase(target: SomeEntity, aggro: 10.0, speed: 1.0);`.
//
// After monomorphisation each apply site gets a fully-substituted body
// with literal values baked in — the emitted WGSL kernel contains the
// constants directly (e.g. aggro*aggro folds to 225.0 for aggro=15.0).

physics chase(target: EntityKind, aggro: f32, speed: f32) @phase(per_agent) {
  on Tick {} {
    // Param refs (aggro / target / speed) become literal values via
    // monomorphisation. The body below uses simple bindings that exist
    // in the DSL today — fancier nearest-target / move-toward logic
    // is left to the user to extend per-fixture.
    let _aggro_squared = aggro * aggro;
    let _target_kind   = target;
    let _move_speed    = speed;
  }
}
```

(This is a near-trivial body. The point is to exercise the substitution end-to-end. Real chase logic — `nearest_of_kind(target)`, `agents.set_vel(self, ...)` etc. — can be added per-fixture as the DSL surface grows. For v1, we just want to prove "param refs in body get substituted to concrete values in the emitted kernel.")

- [ ] **Step 2: Verify the smoke fixture build still succeeds**

Run: `cargo check -p sims 2>&1 | tail -10`
Expected: clean. The `param_rule_smoke` fixture imports stdlib/rules/chase.sim and applies it twice; with substitution working, both `HunterChase` and `WolfChase` produce concrete-rule kernels whose body contains the substituted literal values.

Run: `cargo test -p sims --test param_rule_smoke 2>&1 | tail -3`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add stdlib/rules/chase.sim
git commit -m "feat(stdlib): chase.sim body uses params now that substitution works"
```

---

## Task 5: Workspace test sweep

**Files:** none (verification only).

- [ ] **Step 1: Run the targeted sweep**

Run: `RUST_MIN_STACK=33554432 cargo test -p dsl_ast -p dsl_compiler -p sims --tests 2>&1 | grep -E "^test result:|FAILED|error\[" | tail -40`
Expected: all green. Should see new tests `param_rule_substitute` and the new `monomorphise_substitutes_param_refs_in_body` test from T3.

- [ ] **Step 2: Address any failures**

Most likely causes:
- A variant of `Stmt` or `ExprKind` not handled in `substitute_*` (compiler error: non-exhaustive match). Add the missing arm.
- A test that relied on the *non-substituting* monomorphise behaviour. Update it.

- [ ] **Step 3: Commit fixes if any were needed**

```bash
git status
# If clean: nothing to commit.
# If files changed:
git add <files>
git commit -m "fix(param-rules-substitution): workspace test fallout"
```

---

## Plan complete — exit criteria

- [ ] `substitute_expr` walks every `ExprKind` variant and substitutes `Ident` refs matching param names with literal Exprs.
- [ ] `substitute_stmt` walks every `Stmt` variant and uses `substitute_expr` for all sub-expressions.
- [ ] Binder-introducing constructs (Quantifier / Fold / Block / For / ForEachAgent) shadow params by removing the binder name from the args map before recursing.
- [ ] `monomorphise` produces concrete rules whose body has param refs replaced with literal values.
- [ ] `stdlib/rules/chase.sim` has a non-empty body that references params and compiles end-to-end via the smoke fixture.
- [ ] Full affected-crate `cargo test` passes.

## Follow-up plans (out of scope here)

1. **Body uses richer DSL primitives** — once the spec ships `nearest_of_kind(target)`, `agents.set_vel(self, direction, speed)`, the stdlib chase body becomes useful for real fixtures.
2. **Stdlib expansion** — `wander.sim`, `flock.sim`, `regen.sim` and friends, now that templates with real bodies are usable.
3. **`Span` attribution through substitution** — currently substituted Exprs carry the template's spans; error messages should point at both the apply site and the template line. Threading dual spans is a separate refactor.
4. **Constant-folding sanity test** — manually inspect emitted WGSL to confirm `15.0 * 15.0` folds to `225.0` in the shader. Optional; the shader compiler does this regardless.
