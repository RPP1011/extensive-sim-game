//! Wave 3 ToM Phase 3.5 — `agents.beliefs_<field>(observer, subject)`
//! 2-arg view-call lowering. The 6 reader methods + 6 setter methods
//! registered in `populate_namespace_registry` make scry/reveal consumer
//! rules typecheck and lower; this test confirms the call sites land as
//! `IrExpr::NamespaceCall` after resolve and then `CgExpr::NamespaceCall`
//! after lowering — no `UnsupportedNamespaceCall` fall-through.
//!
//! The WGSL stubs are placeholders today (each function returns 0 /
//! sentinel); the actual SoA cell access lives in the runtime CPU
//! consumer (`tom_probe_runtime::scry` / `::reveal`) until a future
//! phase emits a WGSL kernel from the chronicle stream.

use dsl_ast::ir::{IrExpr, NamespaceId};
use dsl_ast::parser::parse_program;
use dsl_ast::resolve::resolve;
use dsl_compiler::cg::expr::{CgExpr, CgTy};
use dsl_compiler::cg::lower::lower_compilation_to_cg;

/// Per-field smoke that walks a synthetic `.sim` source whose physics
/// rule body reads `agents.beliefs_pos(self, self)` and `agents.set_beliefs_confidence(self, self, 200u)`.
/// The lowering must not surface `UnsupportedNamespaceCall`.
#[test]
fn agents_beliefs_2arg_reader_and_setter_lower_cleanly() {
    let src = r#"
        event Tick { }
        entity Agent_ : Agent { }

        physics ScryProbe @phase(per_agent) {
          on Tick {} where (self.alive) {
            let p = agents.beliefs_pos(self, self)
            let f = agents.beliefs_flags(self, self)
            let c = agents.beliefs_creature_type(self, self)
            let t = agents.beliefs_last_seen_tick(self, self)
            let conf = agents.beliefs_confidence(self, self)
            let susp = agents.beliefs_suspicion(self, self)
            let _ack0 = agents.set_beliefs_pos(self, self, p)
            let _ack1 = agents.set_beliefs_flags(self, self, f)
            let _ack2 = agents.set_beliefs_creature_type(self, self, c)
            let _ack3 = agents.set_beliefs_last_seen_tick(self, self, t)
            let _ack4 = agents.set_beliefs_confidence(self, self, conf)
            let _ack5 = agents.set_beliefs_suspicion(self, self, susp)
          }
        }
    "#;
    let program = parse_program(src).expect("parses");
    let comp = resolve(program).expect("resolves");

    // After resolve, the body should carry IrExpr::NamespaceCall nodes
    // for each agents.beliefs_* / agents.set_beliefs_* call site.
    let physics = comp
        .physics
        .iter()
        .find(|p| p.name == "ScryProbe")
        .expect("physics rule present");
    let body = &physics.handlers[0].body;
    let mut beliefs_call_count = 0usize;
    fn walk(expr: &dsl_ast::ir::IrExprNode, count: &mut usize) {
        if let IrExpr::NamespaceCall { ns: NamespaceId::Agents, method, .. } = &expr.kind {
            if method.starts_with("beliefs_") || method.starts_with("set_beliefs_") {
                *count += 1;
            }
        }
    }
    for stmt in body {
        if let dsl_ast::ir::IrStmt::Let { value, .. } = stmt {
            walk(value, &mut beliefs_call_count);
        }
    }
    assert_eq!(
        beliefs_call_count, 12,
        "expected 12 NamespaceCall nodes (6 readers + 6 setters); got {beliefs_call_count}",
    );

    // Lower the program — the registry-fallback arm of
    // `lower_namespace_call` should resolve each call against the
    // populated registry without surfacing `UnsupportedNamespaceCall`.
    let cg = lower_compilation_to_cg(&comp).expect("lowers");

    // Spot-check: pull every CgExpr::NamespaceCall out of the program
    // and verify the 12 we expect are present with correct arity +
    // return types.
    let mut found_readers = std::collections::BTreeSet::<String>::new();
    let mut found_setters = std::collections::BTreeSet::<String>::new();
    for expr in &cg.exprs {
        if let CgExpr::NamespaceCall { ns: NamespaceId::Agents, method, args, ty } = expr {
            if method.starts_with("set_beliefs_") {
                assert_eq!(args.len(), 3, "{method} setter is 3-arg");
                assert_eq!(*ty, CgTy::Bool, "{method} returns bool ack");
                found_setters.insert(method.clone());
            } else if method.starts_with("beliefs_") {
                assert_eq!(args.len(), 2, "{method} reader is 2-arg");
                found_readers.insert(method.clone());
            }
        }
    }

    let expected_readers: std::collections::BTreeSet<String> = [
        "beliefs_pos",
        "beliefs_creature_type",
        "beliefs_last_seen_tick",
        "beliefs_confidence",
        "beliefs_suspicion",
        "beliefs_flags",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(found_readers, expected_readers, "all 6 readers should lower");

    let expected_setters: std::collections::BTreeSet<String> = [
        "set_beliefs_pos",
        "set_beliefs_creature_type",
        "set_beliefs_last_seen_tick",
        "set_beliefs_confidence",
        "set_beliefs_suspicion",
        "set_beliefs_flags",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(found_setters, expected_setters, "all 6 setters should lower");
}
