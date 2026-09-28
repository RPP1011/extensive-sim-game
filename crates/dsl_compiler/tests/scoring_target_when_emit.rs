//! Compile-side pins: scoring re-applies a target-bearing verb's `when`
//! per candidate (the row guard, wired by
//! `cg::lower::driver::wire_scoring_pair_guards`) and always runs the
//! candidate loop for such verbs, even when the score never reads
//! `target`. GPU behaviour is pinned in
//! `crates/sims/tests/target_when_probe_pin.rs`.

use dsl_compiler::cg::emit::EmittedArtifacts;

fn workspace_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(rel)
}

fn compile(stem: &str) -> EmittedArtifacts {
    let src = std::fs::read_to_string(workspace_path(&format!("assets/sim/{stem}.sim"))).unwrap();
    let program = dsl_compiler::parse(&src).expect("parse");
    let _ = dsl_compiler::custom_agent_fields::populate(&program);
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        panic!("{stem} lowering: {:#?}", o.diagnostics.iter().map(|d| d.to_string()).collect::<Vec<_>>())
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit")
}

fn naga_ok(art: &EmittedArtifacts) {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    for (name, body) in &art.wgsl_files {
        let m = naga::front::wgsl::parse_str(body).unwrap_or_else(|e| panic!("{name}: {e}\n{body}"));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&m)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    }
}

#[test]
fn scoring_filters_candidates_by_the_verb_when() {
    let art = compile("target_when_probe");
    let s = &art.wgsl_files["scoring.wgsl"];
    // Both rows are guarded pair rows now.
    assert_eq!(s.matches("(guarded)").count(), 2, "{s}");
    assert_eq!(s.matches("for (var per_pair_candidate: u32 = 0u;").count(), 2, "{s}");
    // PickAny's constant score still yields a real target.
    assert_eq!(s.matches("best_target = per_pair_candidate;").count(), 2, "{s}");
    assert!(!s.contains("best_target = 0xFFFFFFFFu;"), "a row still emits NO_TARGET:\n{s}");
    // The guard is the verb's `when` evaluated for the candidate.
    assert!(s.contains("((agent_idx[per_pair_candidate] % 2u) == 0u)"), "{s}");
    assert!(s.contains("((agent_idx[per_pair_candidate] % 3u) == 1u)"), "{s}");
    naga_ok(&art);
}

#[test]
fn self_only_verbs_keep_the_actor_level_row() {
    // A verb without a target parameter has a PerAgent mask: its row is
    // unchanged (no candidate loop, no guard).
    let art = compile("verb_fire_probe");
    let s = art
        .wgsl_files
        .values()
        .find(|b| b.contains("// scoring_argmax:"))
        .expect("a kernel carrying the scoring argmax");
    assert!(!s.contains("(guarded)"), "{s}");
    assert!(!s.contains("per_pair_candidate"), "{s}");
}
