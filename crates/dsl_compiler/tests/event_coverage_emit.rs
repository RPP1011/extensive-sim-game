//! Compile-side pin: per-event kernels (chronicle consumers, CAS view
//! folds) walk the whole live ring with a grid stride instead of owning
//! one ring slot per thread — their dispatch is agent-sized, so slots
//! past the thread count used to be dropped. GPU behaviour is pinned in
//! `crates/sims/tests/event_flood_probe_pin.rs`.

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

const ONE_SLOT_PREAMBLE: &str = "let event_idx = gid.x;\nif (event_idx >= cfg.event_count) { return; }";

#[test]
fn per_event_consumer_walks_the_whole_ring() {
    let art = compile("event_flood_probe");
    let k = &art.wgsl_files["physics_CountPings.wgsl"];
    assert!(!k.contains(ONE_SLOT_PREAMBLE), "one-slot-per-thread preamble survived:\n{k}");
    for needle in [
        "@builtin(num_workgroups) _ev_nwg: vec3<u32>",
        "for (var _ev_i: u32 = gid.x; _ev_i < min(cfg.event_count, 1048576u); _ev_i = _ev_i + _ev_nwg.x * 64u) {",
        "let event_idx = _ev_i;",
    ] {
        assert!(k.contains(needle), "missing `{needle}`:\n{k}");
    }
    // No per-event kernel anywhere keeps the one-slot preamble.
    for (name, body) in &art.wgsl_files {
        assert!(!body.contains(ONE_SLOT_PREAMBLE), "{name} keeps the one-slot preamble");
    }
    naga_ok(&art);
}
