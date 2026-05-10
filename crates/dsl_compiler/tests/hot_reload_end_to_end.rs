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
    AbilityId, AbilityProgram, AbilityRegistry, AbilityRegistryBuilder, EffectOp,
    PackedAbilityRegistry,
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

/// **The behavioral pin.** Earlier tests proved registry-lookup
/// returns the new program; this proves the change PROPAGATES through
/// a simulated tick loop that dispatches the ability each tick.
///
/// Models a 10-tick sim where one ability fires at a target every
/// tick, accumulating hp damage. At tick 5 we hot-swap the ability
/// from `damage 10` to `damage 25`. Total damage taken should be:
///   ticks 0..5 (5 ticks): 5 × 10 = 50
///   ticks 5..10 (5 ticks): 5 × 25 = 125
///   total: 175
///
/// Without the hot swap (baseline run): 10 × 10 = 100.
///
/// The 75-damage delta IS the load-bearing proof that hot-reload
/// produces a behavioural change inside a running sim, not just a
/// registry-lookup change.
#[test]
fn hot_reload_propagates_through_sim_loop() {
    // Tiny dispatcher: read the program at `id`, sum damage from its
    // effects (the only EffectOp variant this test uses).
    fn dispatch_one_tick(reg: &AbilityRegistry, id: AbilityId) -> f32 {
        let prog = reg.get(id).expect("ability registered");
        prog.effects
            .iter()
            .map(|e| match e {
                EffectOp::Damage { amount } => *amount,
                _ => 0.0,
            })
            .sum()
    }

    let prog_v1 = parse_and_lower_one(
        "ability HotSwapMidLoop {\n    target: enemy\n    range: 5.0\n    damage 10\n}\n",
    );
    let prog_v2 = parse_and_lower_one(
        "ability HotSwapMidLoop {\n    target: enemy\n    range: 5.0\n    damage 25\n}\n",
    );

    let mut builder = AbilityRegistryBuilder::new();
    let id = builder.register(prog_v1.clone());
    let mut reg = builder.build();

    // --- Run with a hot swap at tick 5.
    let mut hp_taken_with_swap: f32 = 0.0;
    for tick in 0..10 {
        if tick == 5 {
            reg = reg
                .with_program_replaced(id, prog_v2.clone())
                .expect("swap at known id");
        }
        hp_taken_with_swap += dispatch_one_tick(&reg, id);
    }
    assert_eq!(
        hp_taken_with_swap, 175.0,
        "with hot swap at tick 5: 5 × 10 + 5 × 25 = 175; got {hp_taken_with_swap}"
    );

    // --- Baseline: no swap. 10 × 10 = 100.
    let mut baseline_builder = AbilityRegistryBuilder::new();
    let id_b = baseline_builder.register(prog_v1);
    let baseline_reg = baseline_builder.build();
    let mut hp_taken_baseline: f32 = 0.0;
    for _ in 0..10 {
        hp_taken_baseline += dispatch_one_tick(&baseline_reg, id_b);
    }
    assert_eq!(
        hp_taken_baseline, 100.0,
        "baseline (no swap): 10 × 10 = 100; got {hp_taken_baseline}"
    );

    // --- The behavioural delta IS the proof.
    let delta = hp_taken_with_swap - hp_taken_baseline;
    assert_eq!(
        delta, 75.0,
        "hot reload at tick 5 must shift total damage by +75 (5 × (25-10)); got {delta}"
    );
}

/// **Pack-the-swap pin.** The CPU registry change has to flow all the
/// way through `PackedAbilityRegistry::pack` to be visible to the GPU
/// upload step (`PackedAbilityRegistryGpu::upload` reads the packed
/// columns). This test proves the swap propagates: pack v1, hot-swap,
/// pack v2, and confirm the byte-level change shows up in the
/// effect-payload column at the swapped slot.
///
/// The columns matter because that's exactly what the GPU upload
/// would carry over the wire on a hot-reload re-upload — `effect_payload_a`
/// at slot 0 stride 0 carries `bitcast<u32>(damage_amount)` for the
/// Damage variant. v1 packs `bitcast<u32>(10.0)`; v2 packs
/// `bitcast<u32>(25.0)`. They differ → the GPU would see the new
/// value on the next re-upload.
#[test]
fn hot_reload_propagates_through_pack_for_gpu_reupload() {
    let prog_v1 = parse_and_lower_one(
        "ability HotReloadGpuPath {\n    target: enemy\n    range: 5.0\n    damage 10\n}\n",
    );
    let prog_v2 = parse_and_lower_one(
        "ability HotReloadGpuPath {\n    target: enemy\n    range: 5.0\n    damage 25\n}\n",
    );

    let mut builder = AbilityRegistryBuilder::new();
    let id = builder.register(prog_v1);
    let registry_v1 = builder.build();
    let registry_v2 = registry_v1
        .with_program_replaced(id, prog_v2)
        .expect("known id");

    let packed_v1 = PackedAbilityRegistry::pack(&registry_v1);
    let packed_v2 = PackedAbilityRegistry::pack(&registry_v2);

    // Slot 0, effect 0 → effect_payload_a[0]. For Damage{amount},
    // pack_effect stores `bitcast<u32>(amount)` here. Confirm both
    // the v1 byte pattern and the v2 byte pattern match the literal
    // amount, and that they differ (no aliasing through caching).
    let v1_word = packed_v1.effect_payload_a[0];
    let v2_word = packed_v2.effect_payload_a[0];
    assert_eq!(
        v1_word,
        f32::to_bits(10.0),
        "v1 effect_payload_a[0] should be bitcast<u32>(10.0); got 0x{v1_word:08X}"
    );
    assert_eq!(
        v2_word,
        f32::to_bits(25.0),
        "v2 effect_payload_a[0] should be bitcast<u32>(25.0); got 0x{v2_word:08X}"
    );
    assert_ne!(
        v1_word, v2_word,
        "packed bytes MUST change after the swap — otherwise GPU re-upload is a no-op"
    );

    // Sanity: swap doesn't affect n_abilities or column lengths.
    assert_eq!(packed_v1.n_abilities, packed_v2.n_abilities);
    assert_eq!(packed_v1.effect_kinds.len(), packed_v2.effect_kinds.len());
    assert_eq!(packed_v1.effect_payload_a.len(), packed_v2.effect_payload_a.len());
}
