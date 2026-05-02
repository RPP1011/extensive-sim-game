//! Integration tests for the `spatial_query <name>(self, candidate, …) =
//! <filter>` grammar + resolver registry (Phase 7 Task 4).
//!
//! Covers the four scenarios spec'd in the task plan:
//!   - Parser accepts the minimal `(self, candidate)` shape.
//!   - Parser accepts a typed-arg form (`radius: f32`).
//!   - Resolver registers the decl in `Compilation::spatial_queries`
//!     and routes `from spatial.<name>(...)` references through
//!     `IrExpr::NamespaceCall { ns: Spatial, … }`.
//!   - Resolver rejects the wrong first-two-binders convention.
//!   - Resolver rejects `from spatial.<unknown>(...)` references.

use dsl_ast::{ast::Decl, ir::IrExpr, parse, parser::parse_program};
use dsl_ast::ir::NamespaceId;
use dsl_ast::resolve::resolve;
use dsl_ast::resolve_error::ResolveError;

#[test]
fn parse_spatial_query_decl_minimal() {
    let src = r#"
        spatial_query nearby_alive_other(self: AgentId, candidate: AgentId) =
          self != candidate
    "#;
    let prog = parse_program(src).expect("parses");
    let decl = prog
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::SpatialQuery(s) if s.name == "nearby_alive_other" => Some(s),
            _ => None,
        })
        .expect("decl present");
    assert_eq!(decl.params.len(), 2);
    assert_eq!(decl.params[0].name, "self");
    assert_eq!(decl.params[1].name, "candidate");
}

#[test]
fn parse_spatial_query_decl_with_typed_arg() {
    let src = r#"
        spatial_query nearby_in_radius(self: AgentId, candidate: AgentId, radius: f32) =
          self != candidate
    "#;
    let prog = parse_program(src).expect("parses");
    let decl = prog
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::SpatialQuery(s) if s.name == "nearby_in_radius" => Some(s),
            _ => None,
        })
        .expect("decl present");
    assert_eq!(decl.params.len(), 3);
    assert_eq!(decl.params[2].name, "radius");
}

#[test]
fn resolve_spatial_query_decl_registers_in_compilation() {
    let src = r#"
        spatial_query nearby_alive_other(self: AgentId, candidate: AgentId) =
          self != candidate
    "#;
    let prog = parse(src).expect("parses");
    let comp = resolve(prog).expect("resolves");
    let decl = comp
        .spatial_queries
        .iter()
        .find(|d| d.name == "nearby_alive_other")
        .expect("registered");
    assert_eq!(decl.params.len(), 2);
    assert_eq!(decl.params[0].name, "self");
    assert_eq!(decl.params[1].name, "candidate");
}

#[test]
fn resolve_spatial_query_call_resolves_against_registry() {
    // The mask body's `from spatial.my_query(self)` clause must resolve
    // to a `NamespaceCall { ns: Spatial, method: "my_query", … }` once
    // the matching `spatial_query` declaration is registered.
    //
    // The from-clause is resolved BEFORE the head's `target` binder is
    // bound (resolve.rs Task 138 comment: "the enumeration source can
    // only reference `self` — the target binding is what this
    // expression *produces*, not a free variable"), so the call site
    // passes `self` only; the `candidate` parameter binds
    // positionally inside the spatial_query body and corresponds to
    // the per-iteration target at lowering time (Phase 7 Task 5).
    let src = r#"
        spatial_query my_query(self: AgentId, candidate: AgentId) =
          self != candidate

        mask MyMask(target)
          from spatial.my_query(self)
          when target != self
    "#;
    let prog = parse(src).expect("parses");
    let comp = resolve(prog).expect("resolves");
    let mask = comp
        .masks
        .iter()
        .find(|m| m.head.name == "MyMask")
        .expect("mask present");
    let cs = mask
        .candidate_source
        .as_ref()
        .expect("candidate_source set");
    match &cs.kind {
        IrExpr::NamespaceCall { ns, method, .. } => {
            assert_eq!(*ns, NamespaceId::Spatial);
            assert_eq!(method, "my_query");
        }
        other => panic!("expected NamespaceCall(Spatial, ...), got {other:?}"),
    }
}

#[test]
fn resolve_spatial_query_missing_self_candidate_binders_errors() {
    let src = r#"
        spatial_query bad(x: AgentId, y: AgentId) =
          x != y
    "#;
    let prog = parse(src).expect("parses");
    let result = resolve(prog);
    assert!(
        matches!(
            result,
            Err(ResolveError::SpatialQueryRequiresSelfCandidateBinders { .. })
        ),
        "expected SpatialQueryRequiresSelfCandidateBinders, got {result:?}"
    );
}

#[test]
fn resolve_unknown_spatial_query_reference_errors() {
    let src = r#"
        mask MyMask(target)
          from spatial.does_not_exist(self)
          when target != self
    "#;
    let prog = parse(src).expect("parses");
    let result = resolve(prog);
    assert!(
        matches!(result, Err(ResolveError::UnknownSpatialQuery { .. })),
        "expected UnknownSpatialQuery, got {result:?}"
    );
}
