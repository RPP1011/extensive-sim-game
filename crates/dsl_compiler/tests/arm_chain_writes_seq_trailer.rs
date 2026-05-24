//! Pin: emit_chronicle_arm_chain (apply_ability dispatcher arm chain)
//! writes the seq trailer at the last payload word (offset 10), matching
//! emit_chronicle_append_skeleton's shape (stride 11 = 2 header + 8 payload + 1 seq).
//!
//! Pre-fix: the arm chain used stride 10 and did not write the seq word,
//! so sort kernels in Slice 3 would read garbage from the uninitialized
//! word for events emitted through the apply_ability path.
//!
//! Post-fix: every arm in emit_chronicle_arm_chain:
//!   1. Uses `_slot * 11u` (stride 11).
//!   2. Writes the seq trailer at `[_slot * 11u + 10u]`.
//!
//! This test pins both properties by compiling an apply_ability fixture
//! and asserting on the emitted WGSL.

use dsl_compiler::cg::emit::EmittedArtifacts;

fn compile_fixture(src: &str) -> EmittedArtifacts {
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .unwrap_or_else(|o| o.program);
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit")
}

/// Load and compile the canonical apply_ability_smoke fixture, then assert
/// that the dispatcher's arm chain uses stride 11 and writes the seq trailer.
#[test]
fn apply_ability_dispatch_writes_seq_trailer_per_arm() {
    let src = std::fs::read_to_string("../../assets/sim/apply_ability_smoke.sim")
        .expect("read apply_ability_smoke.sim");
    let art = compile_fixture(&src);

    // Find the kernel that contains the apply_ability dispatcher.
    let dispatcher_body = art
        .wgsl_files
        .iter()
        .find(|(_, body)| {
            body.contains("apply_program") || body.contains("apply_ability")
                || body.contains("EffectDamageApplied")
                || body.contains("ability_registry_effect_kinds")
        })
        .map(|(_, b)| b.as_str())
        .expect("an apply_ability dispatcher kernel must be emitted");

    // Stride must be 11 — no old * 10u stride in arm writes.
    assert!(
        !dispatcher_body.contains("_slot * 10u + "),
        "arm chain must NOT use the old * 10u stride;\n\
         got body excerpt:\n{}",
        &dispatcher_body[..dispatcher_body.len().min(1200)],
    );

    // Seq trailer at offset 10 (stride 11) must appear after each arm's
    // last payload store. Pin a count >= 40 to catch complete omission
    // while leaving headroom for minor arm-set changes.
    let trailer_count = dispatcher_body
        .matches("atomicStore(&event_ring[_slot * 11u + 10u]")
        .count();
    assert!(
        trailer_count >= 40,
        "arm chain must write seq trailer at [_slot * 11u + 10u] \
         for every arm; found {trailer_count} trailer writes;\n\
         body excerpt:\n{}",
        &dispatcher_body[..dispatcher_body.len().min(1200)],
    );
}
