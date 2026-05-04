//! `particle_collision_runtime` build script. Mirrors
//! `predator_prey_runtime/build.rs` verbatim except for the input
//! fixture path (`particle_collision_min.sim`).

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/particle_collision_runtime");
    let sim_path = workspace_root.join("assets/sim/particle_collision_min.sim");

    println!("cargo:rerun-if-changed={}", sim_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let src = fs::read_to_string(&sim_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", sim_path.display()));
    let program = dsl_compiler::parse(&src).expect("parse particle_collision_min.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve particle_collision_min.sim");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("lower particle_collision_min.sim to CG");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit particle_collision CG program");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!(
        "cargo:warning=[particle_collision emit-stats] {} kernels, schedule has {} stages",
        artifacts.kernel_index.len(),
        schedule_result.schedule.stages.len(),
    );
    let kernel_keys: Vec<String> = artifacts
        .kernel_index
        .iter()
        .map(|name| format!("{name}.wgsl"))
        .collect();
    for (kernel_name, key) in artifacts.kernel_index.iter().zip(kernel_keys.iter()) {
        let body = match artifacts.wgsl_files.get(key) {
            Some(b) => b,
            None => continue,
        };
        let bytes = body.len();
        let lines = body.lines().count();
        let bindings = body.matches("@binding(").count();
        let inline_consts = body.matches("\nconst config_").count()
            + body.starts_with("const config_") as usize;
        println!(
            "cargo:warning=[particle_collision emit-stats]   {kernel_name}: {bytes} B, {lines} lines, {bindings} bindings, {inline_consts} inline_consts",
        );
    }

    for (name, body) in &artifacts.wgsl_files {
        fs::write(out_dir.join(name), body)
            .unwrap_or_else(|e| panic!("write {}: {e}", name));
    }

    let mut generated = String::new();
    generated.push_str(
        "// AUTO-CONCATENATED from compiler-emitted artifacts by particle_collision_runtime/build.rs.\n\
         // Do not edit. Regenerate by editing assets/sim/particle_collision_min.sim and rebuilding.\n\n",
    );
    let mut wrap_module = |name: &str, content: &str| {
        generated.push_str(
            "#[allow(non_snake_case, unused_imports, unused_variables, clippy::all)]\n",
        );
        generated.push_str(&format!("pub mod {name} {{\n"));
        generated.push_str(content);
        generated.push_str("\n}\n\n");
    };
    for kernel_name in &artifacts.kernel_index {
        let key = format!("{kernel_name}.rs");
        let content = artifacts
            .rust_files
            .get(&key)
            .unwrap_or_else(|| panic!("missing rust file {key} for kernel {kernel_name}"));
        wrap_module(kernel_name, content);
    }
    for sibling in ["schedule", "dispatch"] {
        let key = format!("{sibling}.rs");
        if let Some(content) = artifacts.rust_files.get(&key) {
            wrap_module(sibling, content);
        }
    }
    if let Some(lib_content) = artifacts.rust_files.get("lib.rs") {
        for line in lib_content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub mod ") || trimmed.starts_with("#![") {
                continue;
            }
            generated.push_str(line);
            generated.push('\n');
        }
    }

    fs::write(out_dir.join("generated.rs"), generated)
        .unwrap_or_else(|e| panic!("write generated.rs: {e}"));
}
