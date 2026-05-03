//! `boids_runtime` build script.
//!
//! Runs `dsl_compiler` at compile time against `assets/sim/boids.sim`
//! and writes the per-kernel WGSL + Rust dispatch modules into
//! `OUT_DIR`. `src/lib.rs` pulls them in via
//! `include!(concat!(env!("OUT_DIR"), "/generated.rs"))`.
//!
//! ## Output shape in OUT_DIR
//!
//! ```text
//! OUT_DIR/
//!   generated.rs                  ← single concatenated file with
//!                                   `pub mod <kernel> { ... }` blocks
//!                                   wrapping each emitted Rust module
//!   physics_MoveBoid.wgsl         ← per-kernel WGSL (referenced via
//!   upload_sim_cfg.wgsl              `include_str!("<name>.wgsl")` from
//!   fused_pack_agents.wgsl           inside the inlined mod blocks)
//!   kick_snapshot.wgsl
//! ```
//!
//! Each per-kernel Rust module's `include_str!("<name>.wgsl")` call
//! resolves relative to the source location of the macro invocation.
//! Once concatenated into `OUT_DIR/generated.rs`, that source location
//! IS `OUT_DIR/generated.rs` — so the WGSL files alongside it resolve
//! correctly.
//!
//! ## When this re-runs
//!
//! cargo re-runs build.rs whenever a tracked input changes. We track:
//!   - `assets/sim/boids.sim` (the DSL fixture)
//!   - `build.rs` itself (cargo's default)
//!   - `Cargo.toml` (cargo's default)
//!
//! The dsl_compiler crate's source is not tracked — if the compiler
//! itself changes, cargo rebuilds dsl_compiler (since boids_runtime
//! depends on it as a build-dep), and the resulting fresh dsl_compiler
//! re-runs this script with whatever changes it carries.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/boids_runtime");
    let boids_sim = workspace_root.join("assets/sim/boids.sim");

    println!("cargo:rerun-if-changed={}", boids_sim.display());
    println!("cargo:rerun-if-changed=build.rs");

    let src = fs::read_to_string(&boids_sim)
        .unwrap_or_else(|e| panic!("read {}: {e}", boids_sim.display()));
    let program = dsl_compiler::parse(&src).expect("parse boids.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve boids.sim");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("lower boids.sim to CG");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit boids CG program");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    // Per-kernel emit stats — surfaced via `cargo:warning=` so they
    // appear in the build output for any `cargo build` / `cargo test`
    // run. Free at runtime; the only cost is a few build-output
    // lines. The numbers are derived directly from the rendered WGSL
    // (no additional compiler-side instrumentation), so they reflect
    // exactly what naga consumes:
    //
    // - `wgsl_bytes` / `wgsl_lines` — overall WGSL size. Use as a
    //   regression signal: if a refactor 3×s the bytes-per-kernel,
    //   something probably stopped fusing.
    // - `bindings` — count of `@binding(` declarations. Catches
    //   accidental duplicate / missing binding emission.
    // - `inline_consts` — count of `const config_<id>:` lines, i.e.
    //   how many DSL-side `config.*.*` reads landed as compile-time
    //   constants. Higher = more values baked at WGSL compile.
    // - `fold_walks` — count of fused-fold cell walks (substring
    //   match on `for (var dz: i32`). For boids today this should be
    //   1 per kernel that does any spatial fold; 4 (= one per
    //   accumulator) would mean fold fusion regressed.
    //
    // Walk in `kernel_index` order so output is deterministic and
    // matches the schedule's emit order rather than alphabetical.
    let kernel_keys: Vec<String> = artifacts
        .kernel_index
        .iter()
        .map(|name| format!("{name}.wgsl"))
        .collect();
    println!(
        "cargo:warning=[boids emit-stats] {} kernels, schedule has {} stages",
        artifacts.kernel_index.len(),
        schedule_result.schedule.stages.len(),
    );
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
        let fold_walks = body.matches("for (var dz: i32").count();
        println!(
            "cargo:warning=[boids emit-stats]   {kernel_name}: {bytes} B, {lines} lines, {bindings} bindings, {inline_consts} inline_consts, {fold_walks} fold_walks",
        );
    }

    // Write WGSL files verbatim. include_str! inside the per-kernel
    // mod blocks resolves to OUT_DIR/<name>.wgsl.
    for (name, body) in &artifacts.wgsl_files {
        fs::write(out_dir.join(name), body)
            .unwrap_or_else(|e| panic!("write {}: {e}", name));
    }

    // Concatenate every emitted Rust file into a single generated.rs.
    // For each kernel module + the schedule + dispatch modules we
    // wrap the file content in `pub mod <name> { ... }` so the
    // include!() at lib.rs surfaces them as submodules of
    // boids_runtime. The compiler's lib.rs content (KernelId enum,
    // BufferRef enum, pub use re-exports) appends after the modules.
    // No `#![allow(...)]` inner attributes here: `include!()` splices
    // these tokens into lib.rs and inner attributes can't apply across
    // an include boundary the way they would at the top of a real
    // module. Each `pub mod` below carries its own outer
    // `#[allow(...)]` instead, and lib.rs adds crate-level allows
    // around the include site for anything the modules' bodies need.
    let mut generated = String::new();
    generated.push_str(
        "// AUTO-CONCATENATED from compiler-emitted artifacts by boids_runtime/build.rs.\n\
         // Do not edit. Regenerate by editing assets/sim/boids.sim and rebuilding.\n\n",
    );

    let mut wrap_module = |name: &str, content: &str| {
        generated.push_str(
            "#[allow(non_snake_case, unused_imports, unused_variables, clippy::all)]\n",
        );
        generated.push_str(&format!("pub mod {name} {{\n"));
        generated.push_str(content);
        generated.push_str("\n}\n\n");
    };

    // Per-kernel modules first (the schedule/dispatch modules
    // reference them via `crate::<kernel>::*`).
    for kernel_name in &artifacts.kernel_index {
        let key = format!("{kernel_name}.rs");
        let content = artifacts
            .rust_files
            .get(&key)
            .unwrap_or_else(|| panic!("missing rust file {key} for kernel {kernel_name}"));
        wrap_module(kernel_name, content);
    }

    // schedule + dispatch.
    for sibling in ["schedule", "dispatch"] {
        let key = format!("{sibling}.rs");
        if let Some(content) = artifacts.rust_files.get(&key) {
            wrap_module(sibling, content);
        }
    }

    // Append the compiler-emitted lib.rs's TOP-LEVEL content (KernelId
    // enum, BufferRef enum, pub use lines). Strip the `pub mod`
    // declarations because we've inlined them above; strip `#![...]`
    // inner attributes because they only parse at crate / module
    // top level — once spliced into the middle of generated.rs they'd
    // be a syntax error.
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
