//! Integration test: the `assets/sim/crafting_diffusion.sim` file
//! parses + resolves cleanly. Lower may surface diagnostics for the
//! sidestepped bitwise-operator gap (the runtime owns the merge);
//! we only assert parse + resolve succeed so the declarative shape
//! doesn't bit-rot while the gap stays open.

use std::path::PathBuf;

#[test]
fn crafting_diffusion_sim_parses_and_resolves() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let sim_path = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/")
        .join("assets/sim/crafting_diffusion.sim");
    let src = std::fs::read_to_string(&sim_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", sim_path.display()));
    let program = dsl_compiler::parse(&src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let _comp = dsl_ast::resolve::resolve(program)
        .unwrap_or_else(|e| panic!("resolve failed: {e:?}"));
}
