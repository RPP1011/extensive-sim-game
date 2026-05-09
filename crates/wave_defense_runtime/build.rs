//! `wave_defense_runtime` build script. Mirrors `village_economy_runtime`'s
//! shape verbatim except for the input fixture path, the .ability corpus
//! filenames, and the AOE Path B opt-in.
//!
//! Lowers `assets/sim/wave_defense.sim` through the DSL compiler pipeline
//! (parse → resolve → CG lower → schedule → emit). The resulting WGSL +
//! Rust files land in OUT_DIR/<kernel>.{wgsl,rs} and are concatenated
//! into `OUT_DIR/generated.rs` for `include!` into `src/lib.rs`.
//!
//! AOE Path B opt-in (`LowerOpts.aoe_dispatch=true`) is ON because
//! MonsterCleave declares `damage 4.0 in spread(2.0, 8)`; without the
//! flag the WGSL dispatcher collapses to single-target shape and Spread
//! never fires. The runtime test pin
//! `settlement_falls_within_budget` would still terminate (settlers
//! die from sheer wave count) but with mis-attributed AOE behavior.
//!
//! Tolerates lower diagnostics — wave_defense inherits the
//! duel_25v25 / village_economy known-deferred well_formed warnings
//! (P6 + cycle: chronicle physics writes agent.hp / mana / pos /
//! alive on the same ring scoring/mask might read).

use std::env;
use std::fs;
use std::path::PathBuf;

// Task #249 polish slice: 5 abilities (was 2 in foundation slice).
// MonsterCleave + 4 tier-keyed Spawn abilities for wave-size ramping.
// Alphabetised — registry slot order matches.
const ABILITY_NAMES: &[&str] = &[
    "MonsterCleave.ability",
    "SpawnHorde.ability",
    "SpawnLarge.ability",
    "SpawnMedium.ability",
    "SpawnSmall.ability",
];

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/wave_defense_runtime");
    let sim_path = workspace_root.join("assets/sim/wave_defense.sim");

    println!("cargo:rerun-if-changed={}", sim_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let corpus_dir = workspace_root.join("assets/ability_test/wave_defense");
    for name in ABILITY_NAMES {
        println!("cargo:rerun-if-changed={}", corpus_dir.join(name).display());
    }

    let src = fs::read_to_string(&sim_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", sim_path.display()));
    let program = dsl_compiler::parse(&src).expect("parse wave_defense.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve wave_defense.sim");

    // AOE Path B opt-in (#121, 2026-05-07): MonsterCleave's program
    // declares `damage 4.0 in spread(2.0, 8)`. Without the opt-in the
    // dispatcher collapses to single-target and Spread never fires.
    let opts = dsl_compiler::cg::lower::LowerOpts {
        aoe_dispatch: true,
        belief_state: false,
        ..dsl_compiler::cg::lower::LowerOpts::default()
    };
    let cg = match dsl_compiler::cg::lower::lower_compilation_to_cg_with_opts(&comp, opts) {
        Ok(p) => p,
        Err(o) => {
            for d in &o.diagnostics {
                println!("cargo:warning=[wave_defense lower diag] {d}");
            }
            o.program
        }
    };

    // Build the AbilityRegistry at build time so the schedule synthesizer's
    // fusion analyzer can resolve `apply_ability <literal>` to the
    // chronicle event kinds the WGSL dispatcher will write — same Task
    // #235 dependency village_economy uses.
    let ability_files: Vec<(String, _)> = ABILITY_NAMES
        .iter()
        .map(|name| {
            let path = corpus_dir.join(name);
            let src = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let parsed = dsl_ast::parse_ability_file(&src)
                .unwrap_or_else(|e| panic!("parse {name}: {e:?}"));
            (name.to_string(), parsed)
        })
        .collect();
    let built_registry =
        dsl_compiler::ability_registry::build_registry(&ability_files)
            .expect("build wave_defense AbilityRegistry");

    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule_with_registry(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
        Some(&built_registry.registry),
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit wave_defense CG program");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!(
        "cargo:warning=[wave_defense emit-stats] {} kernels, schedule has {} stages",
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
            "cargo:warning=[wave_defense emit-stats]   {kernel_name}: {bytes} B, {bindings} bindings",
        );
    }

    for (name, body) in &artifacts.wgsl_files {
        fs::write(out_dir.join(name), body)
            .unwrap_or_else(|e| panic!("write {}: {e}", name));
    }

    let mut generated = String::new();
    generated.push_str(
        "// AUTO-CONCATENATED from compiler-emitted artifacts by wave_defense_runtime/build.rs.\n\
         // Do not edit. Regenerate by editing assets/sim/wave_defense.sim and rebuilding.\n\n",
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
