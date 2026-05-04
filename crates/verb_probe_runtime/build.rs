//! `verb_probe_runtime` build script. Mirrors `predator_prey_runtime`'s
//! build.rs verbatim except for the input fixture path.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/verb_probe_runtime");
    let sim_path = workspace_root.join("assets/sim/verb_fire_probe.sim");

    println!("cargo:rerun-if-changed={}", sim_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let src = fs::read_to_string(&sim_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", sim_path.display()));
    let program = dsl_compiler::parse(&src).expect("parse verb_fire_probe.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve verb_fire_probe.sim");
    // Tolerate lower diagnostics — discovery: surface what's emitted
    // even if some downstream physics rules failed to lower.
    let cg = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => {
            for d in &o.diagnostics {
                println!("cargo:warning=[verb_probe lower diag] {d}");
            }
            o.program
        }
    };
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit verb_fire_probe CG program");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!(
        "cargo:warning=[verb_probe emit-stats] {} kernels, schedule has {} stages",
        artifacts.kernel_index.len(),
        schedule_result.schedule.stages.len(),
    );
    for kernel_name in &artifacts.kernel_index {
        let key = format!("{kernel_name}.wgsl");
        let body = match artifacts.wgsl_files.get(&key) {
            Some(b) => b,
            None => continue,
        };
        let bytes = body.len();
        let bindings = body.matches("@binding(").count();
        println!(
            "cargo:warning=[verb_probe emit-stats]   {kernel_name}: {bytes} B, {bindings} bindings",
        );
    }

    for (name, body) in &artifacts.wgsl_files {
        // Runtime-side workarounds for known compiler emitter gaps
        // in `physics_verb_chronicle_Pray.wgsl`. The chronicle is
        // synthesised by the verb-cascade lowering but its WGSL body
        // has three issues that wgpu validation rejects:
        //
        //   (1) `event_ring` is declared `array<u32>` (non-atomic)
        //       but its emit body writes via `atomicStore` — wgpu
        //       errors with "atomic operation is done on a pointer
        //       to a non-atomic". Producer-side kernels (e.g.
        //       scoring) declare the same buffer as
        //       `array<atomic<u32>>`; chronicle just inherited the
        //       wrong qualifier.
        //   (2) The body references `tick` but never binds
        //       `let tick = cfg.tick;`. Other handler bodies (mask,
        //       scoring, fold) all declare it before the op block.
        //   (3) Once (1) flips the binding to atomic, the load
        //       lines that fetch payload words must use
        //       `atomicLoad` rather than plain `event_ring[...]`
        //       indexing — WGSL forbids non-atomic access through
        //       an atomic-typed binding.
        //
        // Patch the WGSL source here so the runtime can dispatch
        // the chronicle without touching the dsl_compiler. If the
        // upstream emitter ever fixes any of these, the affected
        // `replace` falls through (no-op) — no panic. This is a
        // targeted unblock for Gap #4; the long-term fix lives in
        // the compiler.
        let patched = if name == "physics_verb_chronicle_Pray.wgsl" {
            let p1 = body.replace(
                "var<storage, read_write> event_ring: array<u32>;",
                "var<storage, read_write> event_ring: array<atomic<u32>>;",
            );
            let p2 = p1.replace(
                "let event_idx = gid.x;",
                "let event_idx = gid.x;\nlet tick = cfg.tick;",
            );
            // Replace plain reads with atomicLoad. Hand-coded for
            // the three offsets the current emit body uses (+2, +3,
            // +4). Each occurs exactly once.
            p2.replace(
                "let local_2: u32 = event_ring[event_idx * 10u + 2u];",
                "let local_2: u32 = atomicLoad(&event_ring[event_idx * 10u + 2u]);",
            )
            .replace(
                "let local_1: u32 = event_ring[event_idx * 10u + 3u];",
                "let local_1: u32 = atomicLoad(&event_ring[event_idx * 10u + 3u]);",
            )
            .replace(
                "let local_0: u32 = event_ring[event_idx * 10u + 4u];",
                "let local_0: u32 = atomicLoad(&event_ring[event_idx * 10u + 4u]);",
            )
        } else {
            body.clone()
        };
        fs::write(out_dir.join(name), &patched)
            .unwrap_or_else(|e| panic!("write {}: {e}", name));
    }

    let mut generated = String::new();
    generated.push_str(
        "// AUTO-CONCATENATED from compiler-emitted artifacts by verb_probe_runtime/build.rs.\n\
         // Do not edit. Regenerate by editing assets/sim/verb_fire_probe.sim and rebuilding.\n\n",
    );
    let mut wrap_module = |name: &str, content: &str| {
        generated.push_str(
            "#[allow(non_snake_case, unused_imports, unused_variables, dead_code, clippy::all)]\n",
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
    for sibling in ["schedule", "dispatch", "invariants", "metrics", "probes"] {
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
