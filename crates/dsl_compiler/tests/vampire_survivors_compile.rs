//! Compile-gate tests for the vampire_survivors DSL benchmark fixture.
//! Drives assets/sim/vampire_survivors.sim through
//! parse -> resolve -> lower -> schedule -> emit and asserts emitted
//! kernel shapes. A failing lower IS the gap signal (spec §8 ledger).

use dsl_compiler::cg::emit::EmittedArtifacts;

fn workspace_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(rel)
}

fn compile_sim(path: &std::path::Path) -> Result<EmittedArtifacts, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let program = dsl_compiler::parse(&src).map_err(|e| format!("parse: {e:?}"))?;
    let comp = dsl_ast::resolve::resolve(program).map_err(|e| format!("resolve: {e:?}"))?;
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .map_err(|e| format!("lower: {e:?}"))?;
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .map_err(|e| format!("emit: {e:?}"))
}

#[allow(dead_code)]
fn kernel_body_containing<'a>(art: &'a EmittedArtifacts, needle: &str) -> Option<&'a str> {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle))
        .map(|(_, body)| body.as_str())
}

#[test]
fn vampire_survivors_compiles() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| panic!("vampire_survivors.sim failed at: {e}"));
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    eprintln!(
        "[vampire_survivors] {} kernels: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

#[test]
fn enemy_chase_emits_neighbour_walk() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    let body = kernel_body_containing(&art, "ChasePlayer")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| panic!("no chase kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(
        body.contains("spatial_grid_offsets") || body.contains("grid_starts"),
        "expected bounded-neighbour walk in ChasePlayer body; got:\n{body}",
    );
}
