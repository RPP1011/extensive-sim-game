//! Plan I-step-3 (hot-reload) — end-to-end pin.
//!
//! Closes the "Ideally, you could do this by testing hot reload"
//! item from the recurring prompt. Demonstrates the full hot-reload
//! cycle on a single ability:
//!
//!   1. Parse + lower v1 ("damage 10"). Register into an
//!      `AbilityRegistry`. Confirm the registry reports `damage 10`.
//!   2. Parse + lower v2 ("damage 25") — the same `.ability` source
//!      with one literal changed (the simplest hot-reload edit).
//!   3. Build a NEW registry from v1 via
//!      `AbilityRegistry::with_program_replaced(id, v2_program)`.
//!   4. Confirm the new registry reflects `damage 25` at the same
//!      slot id, and the original v1 registry is untouched (immutable
//!      contract preserved — any in-flight Arc<AbilityRegistry> stays
//!      consistent).
//!
//! This is the primitive a runtime would use as the swap step of a
//! file-watch-driven hot-reload loop. The watch trigger (e.g.
//! `notify` crate) is plumbing the runtime owns; the load-bearing
//! piece is the parse → lower → registry-swap chain proven here.

use dsl_ast::ability_parser::parse_ability_file;
use dsl_compiler::ability_lower::lower_ability_decl;
use engine::ability::{
    AbilityProgram, AbilityRegistryBuilder, EffectOp,
};

fn parse_and_lower_one(src: &str) -> AbilityProgram {
    let file = parse_ability_file(src).expect("parse");
    let decl = file
        .abilities
        .first()
        .expect("at least one ability decl in source");
    lower_ability_decl(decl).expect("lower")
}

#[test]
fn ability_source_edit_propagates_through_hot_reload_swap() {
    // --- v1: original source.
    let src_v1 = "ability HotReloadProbe {\n    target: enemy\n    range: 5.0\n    damage 10\n}\n";
    let prog_v1 = parse_and_lower_one(src_v1);

    let mut builder = AbilityRegistryBuilder::new();
    let id = builder.register(prog_v1);
    let registry_v1 = builder.build();

    let initial = registry_v1.get(id).expect("v1 program registered");
    let v1_amount = match initial.effects.first().expect("at least one effect") {
        EffectOp::Damage { amount } => *amount,
        other => panic!("v1 effect must be Damage, got {other:?}"),
    };
    assert_eq!(v1_amount, 10.0, "v1 source declares damage 10");

    // --- v2: same ability source, one literal edited (simulating
    // the author saving the file with a new value).
    let src_v2 = "ability HotReloadProbe {\n    target: enemy\n    range: 5.0\n    damage 25\n}\n";
    let prog_v2 = parse_and_lower_one(src_v2);

    // --- The hot-reload step itself: produce a new registry with
    // the same slot id but the new program.
    let registry_v2 = registry_v1
        .with_program_replaced(id, prog_v2)
        .expect("hot-reload swap on a known id must succeed");

    // --- Confirm: new registry reflects v2.
    let after = registry_v2.get(id).expect("v2 program present at same id");
    let v2_amount = match after.effects.first().expect("at least one effect") {
        EffectOp::Damage { amount } => *amount,
        other => panic!("v2 effect must be Damage, got {other:?}"),
    };
    assert_eq!(
        v2_amount, 25.0,
        "v2 source declares damage 25; hot-reload must propagate the new literal"
    );

    // --- Confirm: original registry untouched (immutable contract).
    let still = registry_v1.get(id).expect("v1 still has program");
    let still_amount = match still.effects.first().expect("at least one effect") {
        EffectOp::Damage { amount } => *amount,
        other => panic!("v1 effect remains Damage, got {other:?}"),
    };
    assert_eq!(
        still_amount, 10.0,
        "v1 registry must remain immutable across the swap; got {still_amount}"
    );

    // --- Slot id stays stable across the swap (the load-bearing
    // contract for hot-reload — any cached AbilityId references in
    // simulation state stay valid).
    assert_eq!(registry_v1.len(), registry_v2.len());
}
