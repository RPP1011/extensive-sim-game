//! Pin: fixtures with no f32 view-fold accumulator do NOT get the 15
//! radix sort kernels emitted. Pure-u32 / Or folds are trivial —
//! sort overhead is wasteful for them.
//!
//! The 15 radix sort kernels are injected by `build_helper::inject_sort_kernels`
//! only when `build_helper::fixture_needs_sort_kernels` returns `true`.
//! That predicate is the opt-in gate: it returns `true` iff the CG program
//! contains at least one ViewFold op whose view has an f32 result and an
//! Add or Sub fold operator.
//!
//! This test pins both directions:
//!   - sanity: f32-fold fixtures are detected as needing sort kernels.
//!   - pin:    non-f32-fold fixtures are detected as NOT needing them.

use std::path::Path;

use dsl_compiler::build_helper::fixture_needs_sort_kernels;
use dsl_compiler::cg::lower::lower_compilation_to_cg;

fn compile_cg(stem: &str) -> dsl_compiler::cg::program::CgProgram {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/dsl_compiler");
    let path = workspace_root.join("assets/sim").join(format!("{stem}.sim"));
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    lower_compilation_to_cg(&comp).unwrap_or_else(|o| o.program)
}

#[test]
fn forest_fire_has_sort_kernels_sanity() {
    let cg = compile_cg("forest_fire");
    assert!(
        fixture_needs_sort_kernels(&cg),
        "forest_fire has f32 view folds — sort kernels MUST be emitted (sanity check)",
    );
}

#[test]
fn f32_reduction_probe_has_sort_kernels_sanity() {
    let cg = compile_cg("f32_reduction_probe");
    assert!(
        fixture_needs_sort_kernels(&cg),
        "f32_reduction_probe has f32 view folds — sort kernels MUST be emitted",
    );
}

#[test]
fn boids_has_no_sort_kernels() {
    let cg = compile_cg("boids");
    assert!(
        !fixture_needs_sort_kernels(&cg),
        "boids has no f32 view folds — sort kernels must be omitted",
    );
}

#[test]
fn apply_ability_smoke_has_no_sort_kernels() {
    let cg = compile_cg("apply_ability_smoke");
    assert!(
        !fixture_needs_sort_kernels(&cg),
        "apply_ability_smoke has no f32 view folds — sort kernels must be omitted",
    );
}
