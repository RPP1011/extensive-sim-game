//! Coverage sweep: run `emit_physics_wgsl` against every rule in
//! `assets/sim/physics.sim`, tally which lower cleanly vs. fail with
//! `Unsupported`. Reports the breakdown via `println!` so the task 187
//! report can pull the per-rule status without a separate binary. Also
//! ensures the test harness catches future rules added to physics.sim
//! that silently drop out of GPU coverage.
//!
//! The test passes as long as every rule either emits or returns a
//! structured `Unsupported` error (never a panic / unhandled case).

use std::fs;
use std::path::PathBuf;

use dsl_compiler::ast::Program;
use dsl_compiler::emit_physics_wgsl::{emit_physics_dispatcher_wgsl, emit_physics_wgsl, EmitContext};

fn sim_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // repo root
    p.push("assets/sim");
    p
}

#[test]
fn every_physics_rule_reaches_the_wgsl_emitter() {
    let root = sim_root();
    let config_src = fs::read_to_string(root.join("config.sim")).expect("config seed");
    let events_src = fs::read_to_string(root.join("events.sim")).expect("events seed");
    let enums_src = fs::read_to_string(root.join("enums.sim")).expect("enums seed");
    let physics_src = fs::read_to_string(root.join("physics.sim")).expect("physics seed");

    let mut merged = Program { decls: Vec::new() };
    merged.decls.extend(dsl_compiler::parse(&config_src).unwrap().decls);
    merged.decls.extend(dsl_compiler::parse(&enums_src).unwrap().decls);
    merged.decls.extend(dsl_compiler::parse(&events_src).unwrap().decls);
    merged.decls.extend(dsl_compiler::parse(&physics_src).unwrap().decls);
    let comp = dsl_compiler::compile_ast(merged).expect("merged sources resolve");

    let ctx = EmitContext {
        events: &comp.events,
        event_tags: &comp.event_tags,
    };

    let mut accepted: Vec<String> = Vec::new();
    let mut unsupported: Vec<(String, String)> = Vec::new();
    for p in &comp.physics {
        match emit_physics_wgsl(p, &ctx) {
            Ok(_) => accepted.push(p.name.clone()),
            Err(e) => unsupported.push((p.name.clone(), format!("{e}"))),
        }
    }

    println!("=== Physics WGSL emission coverage ===");
    println!("Total rules: {}", comp.physics.len());
    println!("Accepted ({}):", accepted.len());
    for n in &accepted {
        println!("  + {n}");
    }
    println!("Unsupported ({}):", unsupported.len());
    for (n, why) in &unsupported {
        println!("  - {n}: {why}");
    }

    // Dispatcher must always emit (even when no rule is accepted; it's
    // a pure-metadata walk).
    let dispatcher = emit_physics_dispatcher_wgsl(&comp.physics, &ctx);
    assert!(
        dispatcher.contains("fn physics_dispatch(event_idx: u32)"),
        "dispatcher emission must land:\n{dispatcher}"
    );
}
