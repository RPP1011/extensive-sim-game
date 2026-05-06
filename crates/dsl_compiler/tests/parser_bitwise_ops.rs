//! #159: parser support for bitwise binary operators (`|`, `^`, `&`).
//!
//! These were added to the AST + CG IR + WGSL emit so per-agent
//! recipe / skill bitsets can be merged and probed from the DSL
//! surface (`recipes_known | RECIPE_BREAD`,
//! `(skills & WOODCUTTING) != 0u`). Coverage:
//!
//! 1. The three new tokens lex into the right `BinOp::*` variants.
//! 2. Operator precedence: bitwise ops bind tighter than logical
//!    (`&&` / `||`) and looser than comparison (`==`, `!=`, `<`, …) —
//!    so `(mask & FLAG) != 0` parses as `(mask & FLAG) != 0` and
//!    `mask & FLAG != 0` parses as `(mask & FLAG) != 0` too (NOT
//!    `mask & (FLAG != 0)` per Rust / C convention).
//! 3. The lexer doesn't conflate `|` with `||` or `&` with `&&`.

use dsl_ast::ast::{BinOp, Decl, ExprKind};
use dsl_ast::parser::parse_program;

/// Pull a spatial_query's filter expression (the `= <expr>` part) out
/// of a parsed program. spatial_query is the simplest decl shape that
/// keeps a free-form expression around for inspection.
fn extract_filter<'a>(src: &'a str) -> dsl_ast::ast::Expr {
    let prog = parse_program(src).expect("parses");
    let q = prog
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::SpatialQuery(s) => Some(s),
            _ => None,
        })
        .expect("spatial_query decl");
    q.filter.clone()
}

#[test]
fn bit_or_token_lexes_to_bitor() {
    let e = extract_filter(
        "spatial_query q(self: AgentId, candidate: AgentId) = self.recipes | candidate.recipes",
    );
    match e.kind {
        ExprKind::Binary { op, .. } => assert_eq!(op, BinOp::BitOr),
        other => panic!("expected Binary; got {other:?}"),
    }
}

#[test]
fn bit_xor_token_lexes_to_bitxor() {
    let e = extract_filter(
        "spatial_query q(self: AgentId, candidate: AgentId) = self.skills ^ candidate.skills",
    );
    match e.kind {
        ExprKind::Binary { op, .. } => assert_eq!(op, BinOp::BitXor),
        other => panic!("expected Binary; got {other:?}"),
    }
}

#[test]
fn bit_and_token_lexes_to_bitand() {
    let e = extract_filter(
        "spatial_query q(self: AgentId, candidate: AgentId) = self.skills & candidate.skills",
    );
    match e.kind {
        ExprKind::Binary { op, .. } => assert_eq!(op, BinOp::BitAnd),
        other => panic!("expected Binary; got {other:?}"),
    }
}

#[test]
fn bit_ops_do_not_shadow_logical() {
    // `&&` must keep binding as logical-and, not as `&` followed by `&`.
    let e = extract_filter(
        "spatial_query q(self: AgentId, candidate: AgentId) = self.alive && candidate.alive",
    );
    match e.kind {
        ExprKind::Binary { op, .. } => assert_eq!(op, BinOp::And),
        other => panic!("expected logical And; got {other:?}"),
    }
}

#[test]
fn bit_op_binds_tighter_than_logical_and() {
    // `a & b && c` parses as `(a & b) && c` because `&` (prec 5) binds
    // tighter than `&&` (prec 2). The outer Binary should be And; its
    // lhs should be a Binary BitAnd.
    let e = extract_filter(
        "spatial_query q(self: AgentId, candidate: AgentId) = \
         self.skills & candidate.skills && self.alive",
    );
    match &e.kind {
        ExprKind::Binary { op, lhs, .. } => {
            assert_eq!(*op, BinOp::And, "outer must be logical And");
            match &lhs.kind {
                ExprKind::Binary { op: inner, .. } => {
                    assert_eq!(*inner, BinOp::BitAnd, "lhs must be BitAnd");
                }
                other => panic!("lhs not Binary; got {other:?}"),
            }
        }
        other => panic!("expected Binary; got {other:?}"),
    }
}

#[test]
fn bit_op_binds_tighter_than_comparison() {
    // `a & FLAG != 0` parses as `(a & FLAG) != 0` because `&` (prec 7)
    // binds tighter than `!=` (prec 3). The outer Binary should be NotEq;
    // its lhs should be a Binary BitAnd. Mirrors Rust convention so
    // `recipes & RECIPE_BREAD != 0` reads naturally without parens. (#159)
    let e = extract_filter(
        "spatial_query q(self: AgentId, candidate: AgentId) = \
         self.skills & candidate.skills != 0",
    );
    match &e.kind {
        ExprKind::Binary { op, lhs, .. } => {
            assert_eq!(*op, BinOp::NotEq, "outer must be NotEq");
            match &lhs.kind {
                ExprKind::Binary { op: inner, .. } => {
                    assert_eq!(*inner, BinOp::BitAnd, "lhs must be BitAnd");
                }
                other => panic!("lhs not Binary; got {other:?}"),
            }
        }
        other => panic!("expected Binary; got {other:?}"),
    }
}
