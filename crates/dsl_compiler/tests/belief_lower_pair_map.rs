//! Plan I — slice I.3 lowering tests for the `belief` keyword.
//!
//! Pins:
//!   * `belief detected_subject(observer: Agent, subject: Agent) -> u32`
//!     infers `StorageHint::PairMap` and produces a `ViewFold` compute
//!     op for the propagation handler.
//!   * Single-key `(observer: Agent) -> T` and key-typed `(observer:
//!     Agent, key: u32) -> T` shapes surface as
//!     `LoweringError::UnsupportedBeliefShape` with a slice-pointer
//!     detail (I.3a).
//!   * The belief's view signature is registered with the CG context
//!     so call-site lowering (`belief.<name>(o, s)`) can resolve via
//!     the same `BuiltinId::ViewCall` path as `view.<name>(...)`.
//!   * Social-merge clauses round-trip on the `ViewIR` for the I.4
//!     emit pass to consume — they do NOT yet produce additional
//!     compute ops.

use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::op::ComputeOpKind;
use dsl_compiler::cg::program::CgProgram;

fn lower_str(src: &str) -> Result<CgProgram, Vec<String>> {
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    match lower_compilation_to_cg(&comp) {
        Ok(p) => Ok(p),
        Err(outcome) => Err(outcome.diagnostics.iter().map(|d| format!("{d}")).collect()),
    }
}

#[test]
fn pair_keyed_belief_lowers_to_view_fold_op() {
    let src = "\
        event SubjectSeen { observer: Agent, subject: Agent, flags: u32 }\n\
        belief detected_subject(observer: Agent, subject: Agent) -> u32 {\n\
          initial: 0,\n\
          on SubjectSeen { observer: o, subject: s, flags: f }\n\
            where (o == observer) && (s == subject)\n\
            { self |= f }\n\
        }\n";
    let cg = lower_str(src).unwrap_or_else(|diags| {
        panic!("lower failed with diagnostics: {diags:?}");
    });
    // Expect at least one ViewFold op for the propagation handler.
    let fold_op_count = cg
        .ops
        .iter()
        .filter(|op| matches!(op.kind, ComputeOpKind::ViewFold { .. }))
        .count();
    assert!(
        fold_op_count >= 1,
        "expected ≥1 ViewFold op for the belief's propagation handler; got {fold_op_count}"
    );
}

#[test]
fn single_key_belief_surfaces_unsupported_shape_diagnostic() {
    let src = "\
        event Tick { observer: Agent }\n\
        belief flag(observer: Agent) -> bool {\n\
          initial: false,\n\
          on Tick { observer: o } { true }\n\
        }\n";
    let diags = lower_str(src).expect_err("expected lowering diagnostics");
    let joined = diags.join(" | ");
    assert!(
        joined.contains("single-key") && joined.contains("I.3a"),
        "expected UnsupportedBeliefShape with slice pointer; got: {joined}"
    );
}

#[test]
fn key_typed_second_param_surfaces_unsupported_shape_diagnostic() {
    let src = "\
        event RoomEntered { observer: Agent, room: u32 }\n\
        belief room_known(observer: Agent, room: u32) -> bool {\n\
          initial: false,\n\
          on RoomEntered { observer: o, room: r } { true }\n\
        }\n";
    let diags = lower_str(src).expect_err("expected lowering diagnostics");
    let joined = diags.join(" | ");
    assert!(
        joined.contains("key-typed") && joined.contains("I.3a"),
        "expected UnsupportedBeliefShape with slice pointer; got: {joined}"
    );
}

#[test]
fn belief_with_social_merge_lowers_propagation_handler_only() {
    // Social-merge clauses sit on ViewIR::social_merges for I.4 emit
    // pass to consume — they do NOT yet add extra compute ops.
    let src = "\
        event SubjectSeen { observer: Agent, subject: Agent, flags: u32 }\n\
        event AllyDied { dead: Agent }\n\
        belief detected_subject(observer: Agent, subject: Agent) -> u32 {\n\
          initial: 0,\n\
          on SubjectSeen { observer: o, subject: s, flags: f }\n\
            where (o == observer) && (s == subject)\n\
            { self |= f }\n\
          on AllyDied { dead: d } merge from d: bit_or\n\
        }\n";
    let cg = lower_str(src).expect("lower succeeds");
    let fold_op_count = cg
        .ops
        .iter()
        .filter(|op| matches!(op.kind, ComputeOpKind::ViewFold { .. }))
        .count();
    // Exactly one ViewFold op — the propagation handler. The social-
    // merge clause does NOT yet produce its own op (slice I.4
    // territory).
    assert_eq!(
        fold_op_count, 1,
        "exactly one ViewFold op expected (propagation handler); social-merge stays on IR until I.4"
    );
}

#[test]
fn belief_interned_under_its_source_name() {
    // The lowering interns view names on the builder so call-site
    // lookup can resolve `BuiltinId::ViewCall { view: ViewId }`. The
    // pair-keyed belief MUST appear in the interned view-name map.
    let src = "\
        event SubjectSeen { observer: Agent, subject: Agent, flags: u32 }\n\
        belief detected_subject(observer: Agent, subject: Agent) -> u32 {\n\
          initial: 0,\n\
          on SubjectSeen { observer: o, subject: s, flags: f }\n\
            where (o == observer) && (s == subject)\n\
            { self |= f }\n\
        }\n";
    let cg = lower_str(src).expect("lower succeeds");
    // The belief is the only view in the source so it sits at ViewId(0).
    let name = cg
        .interner
        .get_view_name(dsl_compiler::cg::data_handle::ViewId(0))
        .expect("view name interned for ViewId(0)");
    assert_eq!(
        name, "detected_subject",
        "belief decl should intern its source-level name on the CG builder",
    );
}
