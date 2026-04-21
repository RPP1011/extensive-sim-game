//! Debug dump: print the emitted WGSL for a handful of physics rules so
//! a human reviewer can spot-check the shape. Marked `#[ignore]` so it
//! only runs on explicit request (`cargo test -- --ignored`).

use std::fs;
use std::path::PathBuf;

use dsl_compiler::ast::Program;
use dsl_compiler::emit_physics_wgsl::{emit_physics_dispatcher_wgsl, emit_physics_wgsl, EmitContext};

fn sim_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push("assets/sim");
    p
}

#[test]
#[ignore]
fn dump_physics_wgsl_for_selected_rules() {
    let root = sim_root();
    let mut merged = Program { decls: Vec::new() };
    for f in &["config.sim", "enums.sim", "events.sim", "physics.sim"] {
        let src = fs::read_to_string(root.join(f)).unwrap();
        merged.decls.extend(dsl_compiler::parse(&src).unwrap().decls);
    }
    let comp = dsl_compiler::compile_ast(merged).unwrap();
    let ctx = EmitContext {
        events: &comp.events,
        event_tags: &comp.event_tags,
    };

    for rule_name in &["damage", "heal", "fear_spread_on_death", "cast", "engagement_on_move"] {
        if let Some(p) = comp.physics.iter().find(|p| p.name == *rule_name) {
            let out = emit_physics_wgsl(p, &ctx).unwrap();
            println!("\n// ===== rule `{rule_name}` =====\n{out}");
        }
    }

    let dispatcher = emit_physics_dispatcher_wgsl(&comp.physics, &ctx);
    println!("\n// ===== dispatcher =====\n{dispatcher}");
}
