//! Plan I — slice I.2 resolver tests for the `belief` keyword.
//!
//! Pins the resolver surface:
//!   * Supported signatures lower into the reserved `ViewIR` slot with
//!     `kind = ViewKind::Belief` and the resolved param + return type.
//!   * Social-merge clauses round-trip into `ViewIR::social_merges`
//!     with the source-agent identifier bound to a LocalRef via the
//!     event pattern's binder.
//!   * Unsupported signatures (no params, >2 params, non-Agent first
//!     param, Vec3 return) surface as `ResolveError::UnsupportedBeliefSignature`.
//!   * A `merge from <ident>` that names an out-of-scope identifier
//!     surfaces as `ResolveError::UnknownSocialMergeSource`.

use dsl_ast::compile;
use dsl_ast::ir::{MergeOp, ViewKind};
use dsl_ast::resolve_error::ResolveError;

fn compile_err(src: &str) -> ResolveError {
    let prog = dsl_ast::parser::parse_program(src)
        .unwrap_or_else(|e| panic!("parse failed:\n{src}\nerror: {e}"));
    dsl_ast::resolve::resolve(prog).expect_err("expected resolve error")
}

#[test]
fn supported_signature_lowers_to_belief_view_slot() {
    // (observer: Agent, subject: Agent) -> bool with one propagation
    // handler + one bit_or social-merge clause.
    let src = "\
        event SubjectSeen { observer: Agent, subject: Agent }\n\
        event AllyDied { dead: Agent }\n\
        belief detected_subject(observer: Agent, subject: Agent) -> bool {\n\
          initial: false,\n\
          on SubjectSeen { observer: obs, subject: subj } { true }\n\
          on AllyDied { dead: d } merge from d: bit_or\n\
        }\n";
    let comp = compile(src).unwrap_or_else(|e| panic!("compile failed: {e}"));
    let view = comp
        .views
        .iter()
        .find(|v| v.name == "detected_subject")
        .expect("belief should appear in comp.views");
    assert_eq!(view.kind, ViewKind::Belief, "kind must be Belief, not Lazy/Materialized");
    assert_eq!(view.params.len(), 2);
    assert_eq!(view.params[0].name, "observer");
    assert_eq!(view.params[1].name, "subject");
    assert_eq!(view.social_merges.len(), 1, "one social-merge clause expected");
    assert_eq!(view.social_merges[0].op, MergeOp::BitOr);
    // source_agent's LocalRef must be the `d` binder from the event
    // pattern — its raw u16 should be greater than the param LocalRefs
    // (params bound first into the outer scope, event binders bound
    // into the inner scope on top of those).
    let src_local = view.social_merges[0].source_agent;
    let param_max = view
        .params
        .iter()
        .map(|p| p.local.0)
        .max()
        .unwrap_or(0);
    assert!(
        src_local.0 > param_max,
        "source_agent local must come from the inner event-pattern scope, after the params"
    );
}

#[test]
fn no_params_rejected() {
    let src = "\
        belief flag() -> bool {\n\
          initial: false,\n\
        }\n";
    let err = compile_err(src);
    match err {
        ResolveError::UnsupportedBeliefSignature { belief_name, detail, .. } => {
            assert_eq!(belief_name, "flag");
            assert!(detail.contains("observer"), "detail should mention the observer requirement; got: {detail}");
        }
        other => panic!("expected UnsupportedBeliefSignature, got {other:?}"),
    }
}

#[test]
fn too_many_params_rejected() {
    let src = "\
        belief flag(observer: Agent, subject: Agent, extra: u32) -> bool {\n\
          initial: false,\n\
        }\n";
    let err = compile_err(src);
    match err {
        ResolveError::UnsupportedBeliefSignature { belief_name, detail, .. } => {
            assert_eq!(belief_name, "flag");
            assert!(detail.contains("max is 2"), "detail should cite the 2-param max; got: {detail}");
        }
        other => panic!("expected UnsupportedBeliefSignature, got {other:?}"),
    }
}

#[test]
fn non_agent_first_param_rejected() {
    // First param is `u32` instead of `Agent` — should be rejected.
    let src = "\
        belief flag(observer: u32) -> bool {\n\
          initial: false,\n\
        }\n";
    let err = compile_err(src);
    match err {
        ResolveError::UnsupportedBeliefSignature { belief_name, detail, .. } => {
            assert_eq!(belief_name, "flag");
            assert!(detail.contains("Agent"), "detail should cite the Agent requirement; got: {detail}");
        }
        other => panic!("expected UnsupportedBeliefSignature, got {other:?}"),
    }
}

#[test]
fn unsupported_return_type_rejected() {
    // Vec3 return is outside the cell-storage matrix.
    let src = "\
        belief vector_belief(observer: Agent) -> Vec3 {\n\
          initial: vec3(0.0, 0.0, 0.0),\n\
        }\n";
    let err = compile_err(src);
    match err {
        ResolveError::UnsupportedBeliefSignature { belief_name, detail, .. } => {
            assert_eq!(belief_name, "vector_belief");
            assert!(
                detail.contains("Vec3") || detail.contains("return"),
                "detail should cite the unsupported return type; got: {detail}",
            );
        }
        other => panic!("expected UnsupportedBeliefSignature, got {other:?}"),
    }
}

#[test]
fn out_of_scope_social_merge_source_rejected() {
    // `merge from ghost: bit_or` — `ghost` is not bound by the event
    // pattern's binders (only `dead` is).
    let src = "\
        event AllyDied { dead: Agent }\n\
        belief flag(observer: Agent) -> u32 {\n\
          initial: 0,\n\
          on AllyDied { dead: d } merge from ghost: bit_or\n\
        }\n";
    let err = compile_err(src);
    match err {
        ResolveError::UnknownSocialMergeSource {
            belief_name,
            source_name,
            bound,
            ..
        } => {
            assert_eq!(belief_name, "flag");
            assert_eq!(source_name, "ghost");
            assert!(
                bound.iter().any(|n| n == "d") || bound.iter().any(|n| n == "observer"),
                "bound list should surface in-scope binders; got: {bound:?}",
            );
        }
        other => panic!("expected UnknownSocialMergeSource, got {other:?}"),
    }
}

#[test]
fn all_four_merge_ops_resolve_to_ir_variants() {
    let cases = [
        ("bit_or", MergeOp::BitOr),
        ("max", MergeOp::Max),
        ("min", MergeOp::Min),
        ("replace", MergeOp::Replace),
    ];
    for (op_text, expected) in cases {
        let src = format!(
            "\
            event Tick {{ giver: Agent }}\n\
            belief flag(observer: Agent) -> u32 {{\n\
              initial: 0,\n\
              on Tick {{ giver: g }} merge from g: {op_text}\n\
            }}\n"
        );
        let comp = compile(&src).unwrap_or_else(|e| panic!("compile failed for {op_text}: {e}"));
        let view = comp.views.iter().find(|v| v.name == "flag").expect("belief in views");
        assert_eq!(view.social_merges.len(), 1);
        assert_eq!(view.social_merges[0].op, expected, "op text was `{op_text}`");
    }
}

#[test]
fn belief_shares_view_namespace_with_views() {
    // Beliefs occupy slots in `comp.views` (not a sibling vec) so name
    // collisions with `view` decls are caught by the same DuplicateDecl
    // check.
    let src = "\
        view collide(observer: Agent) -> u32 { initial: 0 }\n\
        belief collide(observer: Agent) -> u32 { initial: 0 }\n";
    let err = compile_err(src);
    match err {
        ResolveError::DuplicateDecl { kind, name, .. } => {
            assert_eq!(kind, "view");
            assert_eq!(name, "collide");
        }
        other => panic!("expected DuplicateDecl, got {other:?}"),
    }
}
