//! `apply_ability_chronicle_consumer_runtime` build script.
//!
//! Lowers `assets/sim/apply_ability_chronicle_consumer.sim` through the DSL
//! compiler pipeline (parse → resolve → CG lower → schedule → emit). The
//! resulting WGSL + Rust files land in `OUT_DIR/<kernel>.{wgsl,rs}` and are
//! concatenated into `OUT_DIR/generated.rs` for `include!` into `src/lib.rs`.
//!
//! Tolerates lower diagnostics — the consumer rule trips the same P6
//! well_formed diagnostic that `duel_abilities.sim::ApplyDamage` trips
//! (PerEvent + `agents.set_hp` is flagged because P6's strict reading is
//! "events are the mutation channel"). Mirrors
//! `duel_abilities_runtime/build.rs`'s `DriverOutcome::Err → o.program`
//! pattern: surface the diag as a cargo:warning, but keep the emitted
//! kernels.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/apply_ability_chronicle_consumer_runtime");
    let sim_path = workspace_root.join("assets/sim/apply_ability_chronicle_consumer.sim");

    println!("cargo:rerun-if-changed={}", sim_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let src = fs::read_to_string(&sim_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", sim_path.display()));
    let program =
        dsl_compiler::parse(&src).expect("parse apply_ability_chronicle_consumer.sim");
    let comp =
        dsl_ast::resolve::resolve(program).expect("resolve apply_ability_chronicle_consumer.sim");
    // Tolerate lower diagnostics — the consumer rule trips P6 (PerEvent +
    // agents.set_hp). Mirror `duel_abilities_runtime/build.rs`: surface the
    // diag, take `o.program` anyway.
    let cg = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => {
            for d in &o.diagnostics {
                println!(
                    "cargo:warning=[apply_ability_chronicle_consumer lower diag] {d}"
                );
            }
            o.program
        }
    };
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit apply_ability_chronicle_consumer CG program");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!(
        "cargo:warning=[apply_ability_chronicle_consumer emit-stats] {} kernels, schedule has {} stages",
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
            "cargo:warning=[apply_ability_chronicle_consumer emit-stats]   {kernel_name}: {bytes} B, {bindings} bindings",
        );
    }

    for (name, body) in &artifacts.wgsl_files {
        // Architectural friction (documented in src/lib.rs preamble):
        // the consumer kernel filters by the .sim's declaration-index
        // event id (`EffectDamageApplied` is event #1 in this fixture
        // after `Tick` at #0). The dispatcher writes the engine's
        // hardcoded `EventKindId::EffectDamageApplied = 26`. Without a
        // fixup the closed loop is silently broken at the kind-tag
        // layer. Rewrite the consumer's filter so the per-tick damage
        // application actually fires.
        //
        // Proper fix: compiler should resolve well-known engine-event
        // names to their hardcoded EventKindId instead of declaration
        // index. Tracked alongside task #138 follow-ups.
        let patched = if name == "physics_ApplyChronicleDamage.wgsl" {
            let needle = "atomicLoad(&event_ring[event_idx * 10u + 0u]) == 1u";
            let replacement = "atomicLoad(&event_ring[event_idx * 10u + 0u]) == 26u";
            if body.contains(needle) {
                println!(
                    "cargo:warning=[apply_ability_chronicle_consumer build.rs] \
                     patching consumer kind filter: '== 1u' -> '== 26u' \
                     (compiler emits .sim-local id; dispatcher writes engine id 26)"
                );
                body.replace(needle, replacement)
            } else {
                println!(
                    "cargo:warning=[apply_ability_chronicle_consumer build.rs] \
                     consumer kind-filter needle not found in WGSL — emit shape \
                     may have changed. Loop will be broken until the fixup is \
                     re-targeted."
                );
                body.clone()
            }
        } else {
            body.clone()
        };
        fs::write(out_dir.join(name), &patched)
            .unwrap_or_else(|e| panic!("write {}: {e}", name));
    }

    let mut generated = String::new();
    generated.push_str(
        "// AUTO-CONCATENATED from compiler-emitted artifacts by apply_ability_chronicle_consumer_runtime/build.rs.\n\
         // Do not edit. Regenerate by editing assets/sim/apply_ability_chronicle_consumer.sim and rebuilding.\n\n",
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
