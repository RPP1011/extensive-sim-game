use std::collections::HashMap;
use dsl_compiler::lower::param_rules::{substitute_expr, apply_arg_to_expr_kind};
use dsl_ast::ast::{Expr, ExprKind, ApplyArgValue, BinOp, Span};

fn ident(name: &str) -> Expr {
    Expr { kind: ExprKind::Ident(name.to_string()), span: Span::dummy() }
}

#[allow(dead_code)]
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
    let expr = ident("target");
    let items = [("target", ApplyArgValue::EntityKind("Wolf".into()))];
    let args = args_map(&items);
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
    let expr = Expr {
        kind: ExprKind::Quantifier {
            kind: dsl_ast::ast::QuantKind::Forall,
            binder: "aggro".into(),
            iter: Box::new(ident("foo")),
            body: Box::new(Expr {
                kind: ExprKind::Binary {
                    op: BinOp::Gt,
                    lhs: Box::new(ident("aggro")),
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
    let wolf = ApplyArgValue::EntityKind("Wolf".into());
    assert!(matches!(apply_arg_to_expr_kind(&wolf), ExprKind::Ident(_)));
}
