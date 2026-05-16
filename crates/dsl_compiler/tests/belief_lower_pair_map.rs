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
fn single_key_belief_lowers_via_pair_map_hint() {
    // Slice I.3a — `(observer: Agent) -> T` beliefs now ride the
    // same PairMap hint as pair-keyed beliefs. The per-view sizing
    // path (`build_helper::detect_pair_keyed_second_key`) skips
    // 1-param views, so the buffer collapses to single-key (N cells)
    // at allocation time. End-to-end: the lower call must succeed
    // and produce at least one ViewFold op for the propagation
    // handler.
    let src = "\
        event Tick { observer: Agent }\n\
        belief flag(observer: Agent) -> u32 {\n\
          initial: 0,\n\
          on Tick { observer: o } { self |= 1 }\n\
        }\n";
    let cg = lower_str(src).expect("single-key belief should lower cleanly");
    let fold_op_count = cg
        .ops
        .iter()
        .filter(|op| matches!(op.kind, ComputeOpKind::ViewFold { .. }))
        .count();
    assert!(
        fold_op_count >= 1,
        "expected ≥1 ViewFold op for the single-key belief's propagation handler"
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
        joined.contains("key-typed") && joined.contains("I.3b"),
        "expected UnsupportedBeliefShape with slice pointer; got: {joined}"
    );
}

#[test]
fn belief_with_social_merge_lowers_one_view_fold_plus_one_merge_op() {
    // Plan I slice I.4 (IR wiring landed) — social-merge clauses now
    // produce one `ComputeOpKind::BeliefSocialMerge` op each, in
    // addition to the propagation handler's ViewFold op. Kernel body
    // is a stub (TODO marker) until I.4b.
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
    let merge_op_count = cg
        .ops
        .iter()
        .filter(|op| matches!(op.kind, ComputeOpKind::BeliefSocialMerge { .. }))
        .count();
    assert_eq!(fold_op_count, 1, "one ViewFold op for the propagation handler");
    assert_eq!(merge_op_count, 1, "one BeliefSocialMerge op for the merge clause");
    // Verify the merge op carries the bit_or discriminant (=0).
    let merge_op = cg
        .ops
        .iter()
        .find(|op| matches!(op.kind, ComputeOpKind::BeliefSocialMerge { .. }))
        .expect("BeliefSocialMerge present");
    if let ComputeOpKind::BeliefSocialMerge { op, .. } = &merge_op.kind {
        assert_eq!(*op, 0, "bit_or merge op must serialize to discriminant 0");
    }
}

#[test]
fn belief_with_multiple_merge_ops_produces_all_three_variants() {
    // Three social-merge clauses with distinct ops → three ops with
    // distinct discriminants.
    let src = "\
        event Tick { giver: Agent }\n\
        event Tock { giver: Agent }\n\
        event Tack { giver: Agent }\n\
        belief test(observer: Agent, subject: Agent) -> u32 {\n\
          initial: 0,\n\
          on Tick { giver: g } merge from g: bit_or\n\
          on Tock { giver: g } merge from g: max\n\
          on Tack { giver: g } merge from g: replace\n\
        }\n";
    let cg = lower_str(src).expect("lower succeeds");
    let mut ops: Vec<u8> = cg
        .ops
        .iter()
        .filter_map(|op| match &op.kind {
            ComputeOpKind::BeliefSocialMerge { op, .. } => Some(*op),
            _ => None,
        })
        .collect();
    ops.sort();
    assert_eq!(ops, vec![0u8, 1u8, 3u8], "expect bit_or(0) + max(1) + replace(3)");
}

#[test]
fn every_merge_op_emits_matching_atomic_primitive() {
    // Plan I.4b — verify the WGSL kernel emit picks the right
    // `atomicOr`/`atomicMax`/`atomicMin`/`atomicStore` per op
    // discriminant.
    use dsl_compiler::cg::emit::emit_cg_program;
    use dsl_compiler::cg::schedule::{synthesize_schedule, ScheduleStrategy};

    let cases = [
        ("bit_or", "atomicOr"),
        ("max", "atomicMax"),
        ("min", "atomicMin"),
        ("replace", "atomicStore"),
    ];
    for (op_name, expected_atomic) in cases {
        let src = format!(
            "\
            event Trigger {{ giver: Agent }}\n\
            belief b(observer: Agent, subject: Agent) -> u32 {{\n\
              initial: 0,\n\
              on Trigger {{ giver: g }} merge from g: {op_name}\n\
            }}\n"
        );
        let cg = lower_str(&src).unwrap_or_else(|diags| {
            panic!("op {op_name}: lower failed: {diags:?}")
        });
        let sched = synthesize_schedule(&cg, ScheduleStrategy::Default);
        let art = emit_cg_program(&sched.schedule, &cg)
            .unwrap_or_else(|e| panic!("op {op_name}: emit failed: {e:?}"));
        let merge = art
            .wgsl_files
            .iter()
            .find(|(n, _)| n.contains(&format!("merge_b_trigger_{op_name}")))
            .unwrap_or_else(|| {
                let names: Vec<&str> = art.wgsl_files.iter().map(|(n, _)| n.as_str()).collect();
                panic!("op {op_name}: merge kernel not found; emitted: {names:?}")
            });
        let body = &merge.1;
        assert!(
            body.contains(expected_atomic),
            "op {op_name}: kernel body should call `{expected_atomic}`; \
             body excerpt: {}",
            &body[..body.len().min(800)]
        );
    }
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
