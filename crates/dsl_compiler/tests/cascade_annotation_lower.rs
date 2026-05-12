//! Surface-level coverage for the `@cascade(max_iter=N)` annotation on
//! physics rules (Gap forest_fire#F).
//!
//! The annotation is parsed generically by the AST layer (no parser
//! change required — annotations are a generic name+arg surface), then
//! interpreted by the lowering driver via
//! [`dsl_compiler::cg::lower::driver::cascade_max_iter_authored`].
//! The driver records the rule's `max_iter` into a per-program side-
//! table ([`CgProgram::cascade_max_iter`]), which the schedule
//! synthesizer reads when classifying physics topologies into
//! `DispatchOp::FixedPoint { kernel, max_iter }` entries.
//!
//! These tests pin:
//!
//! 1. `parse + resolve` succeeds for `@cascade(max_iter=N)` on a
//!    physics rule (annotations pass through the resolver without an
//!    allowlist; this guards against a future regression that adds
//!    one).
//! 2. `lower_compilation_to_cg` populates `cascade_max_iter` with the
//!    authored value, accessible via
//!    `CgProgram::cascade_max_iter_for(PhysicsRuleId)`.
//! 3. The synthesized SCHEDULE emit carries the per-rule `max_iter`
//!    in the `DispatchOp::FixedPoint { ..., max_iter: N }` entry —
//!    distinct from the legacy hardcoded default of `8`.
//! 4. Without the annotation, the synthesizer keeps emitting the
//!    legacy `max_iter: 8` default — non-regression for existing
//!    fixtures.
//!
//! **Runtime status (2026-05-12).** The runtime `DispatchOp::FixedPoint`
//! arm is still a catch-all no-op (see
//! `crates/dsl_compiler/src/build_helper.rs` near the
//! "DispatchOp::FixedPoint, DispatchOp::GatedBy" comment block). These
//! tests cover the parser+resolver+lower+schedule-emit surface only;
//! threading `max_iter` through to the host's per-tick dispatch loop
//! is a separately-scoped follow-up.

use dsl_compiler::cg::op::PhysicsRuleId;

const TEMPLATE: &str = r#"
event Tick { }

@replayable
@gpu_amenable
event Spread {
  target: AgentId,
}

entity Particle : Agent {
  pos: vec3,
  vel: vec3,
}

ANNOTATION_PLACEHOLDER
physics Tickle @phase(per_agent) {
  on Tick {} {
    emit Spread { target: self }
  }
}
"#;

fn render_src(annotation: &str) -> String {
    TEMPLATE.replace("ANNOTATION_PLACEHOLDER", annotation)
}

/// Lower the fixture and return the resulting [`CgProgram`].
fn lower(annotation: &str) -> dsl_compiler::cg::program::CgProgram {
    let src = render_src(annotation);
    let program = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("lower")
}

/// Synthesize the SCHEDULE Rust source for the lowered fixture and
/// return it so individual tests can pattern-match the emitted entry
/// shape.
fn synth_schedule_src(prog: &dsl_compiler::cg::program::CgProgram) -> String {
    let plan = dsl_compiler::cg::schedule::synthesize_schedule(
        prog,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::cross_cutting::synthesize_schedule(&plan.schedule, prog)
}

// -- Surface check #1: parse + resolve succeed. --------------------------

#[test]
fn cascade_annotation_parses_and_resolves() {
    let src = render_src("@cascade(max_iter = 8)");
    let program = dsl_compiler::parse(&src).expect("parse must accept @cascade");
    dsl_ast::resolve::resolve(program).expect("resolve must accept @cascade");
}

// -- Surface check #2: lower populates cascade_max_iter. -----------------

#[test]
fn cascade_annotation_recorded_in_cg_program() {
    let prog = lower("@cascade(max_iter = 8)");
    // `Tickle` is the only physics rule in the fixture → PhysicsRuleId(0).
    assert_eq!(
        prog.cascade_max_iter_for(PhysicsRuleId(0)),
        Some(8),
        "lowering must populate cascade_max_iter from the annotation"
    );
}

#[test]
fn cascade_annotation_records_authored_value_not_default() {
    // Pick a value distinct from the legacy default of 8 so we can
    // distinguish "the annotation took effect" from "the default
    // happened to coincide".
    let prog = lower("@cascade(max_iter = 12)");
    assert_eq!(
        prog.cascade_max_iter_for(PhysicsRuleId(0)),
        Some(12),
        "lowering must propagate the authored max_iter, not the default"
    );
}

// -- Surface check #3: synthesized SCHEDULE carries the per-rule max_iter. --

/// The fixture's `Tickle` physics rule lowers to a kernel named
/// `physics_tickle` (per `physics_rule_fused_kernel_name`'s
/// `physics_<rule>` convention), pascal-cased to `PhysicsTickle` in
/// the emitted `KernelId` reference.
const PHYSICS_KERNEL_PASCAL: &str = "PhysicsTickle";

#[test]
fn cascade_annotation_flows_through_to_schedule_emit() {
    let prog = lower("@cascade(max_iter = 12)");
    let src = synth_schedule_src(&prog);
    let want = format!(
        "DispatchOp::FixedPoint {{ kernel: KernelId::{PHYSICS_KERNEL_PASCAL}, max_iter: 12 }}"
    );
    assert!(
        src.contains(&want),
        "synthesized SCHEDULE must emit the authored max_iter (want substring `{want}`); got:\n{src}"
    );
    // Sanity: the legacy default did NOT slip through alongside it.
    assert!(
        !src.contains("max_iter: 8"),
        "legacy max_iter: 8 must not appear when the rule is annotated; got:\n{src}"
    );
}

// -- Surface check #4: non-regression — no annotation keeps the regular
//    `DispatchOp::Kernel(...)` shape (no FixedPoint upgrade). The legacy
//    `max_iter = 8` default lives behind the dead `kernel_name ==
//    "physics"` guard in [`classify_topology_for_schedule`]; in the CG
//    path that guard never fires, so a non-annotated physics rule
//    emits as a plain Kernel entry.

#[test]
fn no_cascade_annotation_emits_plain_kernel_entry() {
    let prog = lower("");
    // No rule carries the annotation.
    assert_eq!(
        prog.cascade_max_iter_for(PhysicsRuleId(0)),
        None,
        "rule without @cascade must not appear in cascade_max_iter side-table"
    );
    let src = synth_schedule_src(&prog);
    let want = format!("DispatchOp::Kernel(KernelId::{PHYSICS_KERNEL_PASCAL})");
    assert!(
        src.contains(&want),
        "non-annotated rule must emit a plain Kernel(...) entry (want substring `{want}`); got:\n{src}"
    );
    // The `DispatchOp` enum *declaration* always contains a
    // `FixedPoint { ... }` variant; what we want is that no SCHEDULE
    // *entry* uses it. `DispatchOp::FixedPoint {` is the entry-form
    // prefix the synthesizer emits.
    assert!(
        !src.contains("DispatchOp::FixedPoint {"),
        "no FixedPoint entry should appear without @cascade annotation; got:\n{src}"
    );
}

// -- Edge cases on the annotation parser ---------------------------------

#[test]
fn cascade_with_zero_max_iter_falls_back_to_default() {
    // `0` is a no-op iteration count; the lower helper rejects it
    // (returns `None`) so the schedule keeps the legacy default
    // rather than silently disabling the kernel. Resolve still
    // accepts it (no allowlist), so the only observable behaviour is
    // "no entry in the side-table".
    let prog = lower("@cascade(max_iter = 0)");
    assert_eq!(
        prog.cascade_max_iter_for(PhysicsRuleId(0)),
        None,
        "max_iter = 0 must not populate the side-table"
    );
}

#[test]
fn cascade_with_negative_max_iter_rejected_by_parser() {
    // The annotation parser does not accept a leading minus sign on
    // numeric annotation values (`expected annotation value` parse
    // error). Pin that contract — it's the parser, not the lower
    // helper, that fences out negatives, so the lower helper's
    // `*n > 0` guard is belt-and-braces.
    let src = render_src("@cascade(max_iter = -3)");
    let err = dsl_compiler::parse(&src)
        .err()
        .expect("parser must reject negative annotation literals");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("expected annotation value"),
        "expected `expected annotation value` parse error; got: {msg}"
    );
}

#[test]
fn cascade_with_large_max_iter_is_threaded_through() {
    // u32 ceiling — pin a representative large value so a future
    // narrowing of the type (e.g. u16) trips this test.
    let prog = lower("@cascade(max_iter = 1024)");
    assert_eq!(prog.cascade_max_iter_for(PhysicsRuleId(0)), Some(1024));
    let src = synth_schedule_src(&prog);
    assert!(
        src.contains("max_iter: 1024"),
        "large max_iter must round-trip through the schedule emit; got:\n{src}"
    );
}
