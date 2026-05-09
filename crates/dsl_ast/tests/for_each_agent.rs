//! Surface-form tests for the `for_each_agent <binder>` body-shape
//! primitive (Task #229). Exercises the parser → resolver path; the CG
//! lowering of the same construct is covered by the inline tests in
//! `crates/dsl_compiler/src/cg/lower/physics.rs::tests` and the
//! end-to-end behavioural pin lives in
//! `crates/for_each_agent_probe_runtime`.

use dsl_ast::ast::{Decl, Stmt};
use dsl_ast::ir::IrStmt;

/// Wrap one or more statements in a per-agent physics body so the
/// parser + resolver have a complete `.sim` source to chew on. The
/// fixture defines a single `Tick` event and a `Probe` entity (with
/// `mana: f32` so the body can read/write the field via the
/// `agents.set_mana` setter the lowering recognises).
fn parse_physics_body(body_src: &str) -> dsl_ast::Program {
    let src = format!(
        "event Tick {{ }}\n\
         entity Probe : Agent {{ pos: vec3, mana: f32, max_mana: f32, }}\n\
         physics P @phase(per_agent) {{\n\
           on Tick {{}} where (self.alive) {{\n\
             {body_src}\n\
           }}\n\
         }}\n"
    );
    dsl_ast::parse(&src).unwrap_or_else(|e| {
        panic!("parse failed for body `{body_src}`:\n{src}\nerror: {e}")
    })
}

/// Compile (parse + resolve) the physics body so the IR is available
/// for further structural checks.
fn compile_physics_body(body_src: &str) -> dsl_ast::Compilation {
    let src = format!(
        "event Tick {{ }}\n\
         entity Probe : Agent {{ pos: vec3, mana: f32, max_mana: f32, }}\n\
         physics P @phase(per_agent) {{\n\
           on Tick {{}} where (self.alive) {{\n\
             {body_src}\n\
           }}\n\
         }}\n"
    );
    dsl_ast::compile(&src).unwrap_or_else(|e| {
        panic!("compile failed for body `{body_src}`:\n{src}\nerror: {e}")
    })
}

/// Pull the first physics handler's body out of the parsed program.
fn first_physics_handler_body(prog: &dsl_ast::Program) -> &[Stmt] {
    let physics = prog.decls.iter().find_map(|d| match d {
        Decl::Physics(p) => Some(p),
        _ => None,
    }).expect("expected a physics decl");
    let handler = physics
        .handlers
        .first()
        .expect("expected at least one handler");
    &handler.body
}

#[test]
fn for_each_agent_parses_with_binder() {
    // Smallest accepted form: a single setter inside the body.
    let prog = parse_physics_body(
        "for_each_agent a {\n\
           agents.set_mana(a, agents.mana(a) + 1.0)\n\
         }",
    );
    let body = first_physics_handler_body(&prog);
    let stmt = body.first().expect("expected at least one statement");
    match stmt {
        Stmt::ForEachAgent { binder, body, .. } => {
            assert_eq!(binder, "a", "binder name round-trips through the parser");
            assert!(
                !body.is_empty(),
                "the body must carry the inner statements"
            );
        }
        other => panic!(
            "expected Stmt::ForEachAgent, got {other:?}\nthe parser must \
             accept `for_each_agent <binder> {{ <body> }}`",
        ),
    }
}

#[test]
fn for_each_agent_with_body_stmts() {
    // Multi-statement body: a let, then a setter using both the binder
    // AND the let-bound value. Exercises both the binder-name scope and
    // the body-statement parsing inside `for_each_agent`.
    let comp = compile_physics_body(
        "for_each_agent slot {\n\
           let cap = agents.max_mana(slot)\n\
           agents.set_mana(slot, cap)\n\
         }",
    );
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
        .expect("expected at least one resolved statement");
    match stmt {
        IrStmt::ForEachAgent {
            binder_name,
            body,
            ..
        } => {
            assert_eq!(
                binder_name, "slot",
                "the binder name must resolve through to the IR for diagnostics"
            );
            assert_eq!(
                body.len(),
                2,
                "the inner body should carry both the let and the setter call \
                 — got {body:?}"
            );
        }
        other => panic!(
            "expected IrStmt::ForEachAgent at top-level, got {other:?}\n\
             the resolver must mirror the AST shape into IR",
        ),
    }
}
