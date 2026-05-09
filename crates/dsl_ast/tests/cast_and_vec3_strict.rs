//! Surface-form tests for `vec3(x, y, z)` strict-f32 typing and
//! explicit `f32(...)` / `u32(...)` / `i32(...)` casts.
//!
//! These tests exercise the parser → resolver pipeline only — the CG
//! lowering of the same constructs is covered by inline tests in
//! `crates/dsl_compiler/src/cg/lower/expr.rs::tests`. Strict-f32
//! enforcement on `vec3` happens at CG lowering, not at resolve time,
//! so the resolver shape is unchanged from before.

use dsl_ast::compile;
use dsl_ast::ir::{Builtin, IrExpr, IrStmt};

/// Wrap a single statement in a `physics` body so we have a complete
/// `.sim` source the parser + resolver will accept. Returns the
/// resolved Compilation. The fixture defines a single `Tick` event
/// and a `Probe` entity with a `pos: vec3` field so the body can
/// reference `self.pos` if needed.
fn compile_physics_body(stmt_src: &str) -> dsl_ast::Compilation {
    let src = format!(
        "event Tick {{ }}\n\
         entity Probe : Agent {{ pos: vec3, }}\n\
         physics P @phase(per_agent) {{\n\
           on Tick {{}} where (self.alive) {{\n\
             {stmt_src}\n\
           }}\n\
         }}\n"
    );
    compile(&src).unwrap_or_else(|e| panic!("compile failed for body `{stmt_src}`:\n{src}\nerror: {e}"))
}

/// Pull the first `Let` statement out of the resolved physics body and
/// return the let's value expression.
fn first_let_value(comp: &dsl_ast::Compilation) -> &IrExpr {
    let physics = comp
        .physics
        .first()
        .expect("expected at least one physics rule");
    let handler = physics
        .handlers
        .first()
        .expect("expected at least one handler");
    let stmt = handler
        .body
        .first()
        .expect("expected at least one body stmt");
    match stmt {
        IrStmt::Let { value, .. } => &value.kind,
        other => panic!("expected first body stmt to be Let, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// f32 / u32 / i32 surface casts
// ---------------------------------------------------------------------------

#[test]
fn f32_cast_lexes_and_parses() {
    let comp = compile_physics_body("let x = f32(1);");
    match first_let_value(&comp) {
        IrExpr::BuiltinCall(Builtin::F32Cast, args) => {
            assert_eq!(args.len(), 1, "f32(...) takes one arg");
        }
        other => panic!("expected BuiltinCall(F32Cast, _), got {other:?}"),
    }
}

#[test]
fn u32_cast_lexes_and_parses() {
    let comp = compile_physics_body("let x = u32(1.5);");
    match first_let_value(&comp) {
        IrExpr::BuiltinCall(Builtin::U32Cast, args) => {
            assert_eq!(args.len(), 1, "u32(...) takes one arg");
        }
        other => panic!("expected BuiltinCall(U32Cast, _), got {other:?}"),
    }
}

#[test]
fn i32_cast_lexes_and_parses() {
    let comp = compile_physics_body("let x = i32(1.5);");
    match first_let_value(&comp) {
        IrExpr::BuiltinCall(Builtin::I32Cast, args) => {
            assert_eq!(args.len(), 1, "i32(...) takes one arg");
        }
        other => panic!("expected BuiltinCall(I32Cast, _), got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// vec3 strict-f32 — surface parse always succeeds; CG lowering enforces.
// ---------------------------------------------------------------------------

#[test]
fn vec3_with_f32_literals_parses() {
    let comp = compile_physics_body("let x = vec3(1.0, 2.0, 3.0);");
    match first_let_value(&comp) {
        IrExpr::BuiltinCall(Builtin::Vec3, args) => {
            assert_eq!(args.len(), 3, "vec3(...) takes three args");
        }
        other => panic!("expected BuiltinCall(Vec3, _), got {other:?}"),
    }
}

#[test]
fn f32_cast_inside_vec3_parses() {
    // Resolves to a Vec3 builtin whose three children are each an
    // F32Cast builtin. CG lowering of the same shape is covered by
    // a sibling inline test in dsl_compiler.
    let comp = compile_physics_body("let x = vec3(f32(1), f32(2), f32(3));");
    let val = first_let_value(&comp);
    let args = match val {
        IrExpr::BuiltinCall(Builtin::Vec3, args) => args,
        other => panic!("expected BuiltinCall(Vec3, _), got {other:?}"),
    };
    assert_eq!(args.len(), 3);
    for (i, a) in args.iter().enumerate() {
        match &a.value.kind {
            IrExpr::BuiltinCall(Builtin::F32Cast, _) => {}
            other => panic!("vec3 arg[{i}]: expected BuiltinCall(F32Cast, _), got {other:?}"),
        }
    }
}
