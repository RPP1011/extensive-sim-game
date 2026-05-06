//! Pack-the-full-LoL-corpus smoke test — drive all 172
//! `dataset/abilities/lol_heroes/*.ability` files through
//! `dsl_compiler::build_registry` and `PackedAbilityRegistry::pack`,
//! and verify the SoA encoding survives the full EffectOp vocabulary.
//!
//! Coverage rationale:
//!   * The existing `packed_registry_corpus` test pins the bit-exact
//!     packing of the 8-file `duel_abilities` corpus — it covers
//!     ~5 EffectOp variants out of the 30+ that ship today.
//!   * Whenever a new EffectOp variant lands (Wave 2 piece 7 stealth,
//!     piece 8 charm/grounded/suppress/reflect, etc.) the pack arm in
//!     `engine::ability::packed::pack_effect_op` needs a parallel
//!     entry. Forgetting that arm panics with an `unimplemented!` /
//!     match-non-exhaustive at pack time the FIRST time a corpus
//!     ability uses it. The pack-corpus test catches this systemically:
//!     every variant the LoL canary lowers, this test exercises.
//!   * Beyond no-panic, this test also asserts every non-empty program
//!     packs at least one non-EMPTY effect_kind slot — a packing bug
//!     that silently zero-extends a discriminant would silently lose
//!     the verb without a test like this.
//!
//! Companion to `dsl_compiler/tests/lol_corpus_lowering.rs`'s baseline
//! (172/172 lower clean as of Wave 2 piece 8). This test re-exercises
//! the exact same files end-to-end through the SoA pack stage.

use std::fs;
use std::path::PathBuf;

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_registry::build_registry;
use engine::ability::{EFFECT_KIND_EMPTY, MAX_EFFECTS_PER_PROGRAM, PackedAbilityRegistry};

fn lol_corpus_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .expect("crates/")
        .parent()
        .expect("repo root")
        .join("dataset/abilities/lol_heroes")
}

#[test]
fn pack_full_lol_corpus_does_not_panic_and_every_program_carries_at_least_one_effect() {
    let dir = lol_corpus_root();
    if !dir.is_dir() {
        eprintln!("dataset/abilities/lol_heroes not found at {}", dir.display());
        return;
    }

    let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "ability"))
        .collect();
    paths.sort();

    // Walk every file, keeping only those that parse + pass the canary's
    // `build_registry`. Files that lower with errors (none, post-canary
    // saturation) are skipped without aborting the test. The corpus is
    // currently saturated at 172/172 lowering clean.
    let mut files: Vec<(String, dsl_ast::AbilityFile)> = Vec::new();
    for p in &paths {
        let src = match fs::read_to_string(p) { Ok(s) => s, Err(_) => continue };
        let parsed = match parse_ability_file(&src) { Ok(f) => f, Err(_) => continue };
        let label = p.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        files.push((label, parsed));
    }
    assert!(
        files.len() >= 170,
        "LoL canary expected ≥170 parseable files; got {} — corpus regression?",
        files.len()
    );

    // build_registry is the same path production code uses to build the
    // GPU-facing SoA table. If it fails, the cause is upstream of the
    // pack step; surface that loudly.
    let built = build_registry(&files).expect("LoL canary must build_registry clean");
    assert_eq!(
        built.registry.len(),
        files.iter().map(|(_, f)| f.abilities.len()).sum::<usize>(),
        "every parseable .ability decl must reach the registry"
    );

    // The actual pack call. This is the test's primary purpose — any
    // newly-added EffectOp variant that forgets to wire its pack arm
    // panics here. With 30+ variants today, the surface area is real.
    let packed = PackedAbilityRegistry::pack(&built.registry);

    assert_eq!(packed.n_abilities, built.registry.len());
    let n = packed.n_abilities;

    // SoA invariants: every per-program column has length n; every
    // per-effect column has length n * MAX_EFFECTS_PER_PROGRAM.
    assert_eq!(packed.cooldown_ticks.len(), n, "cooldown_ticks SoA stride");
    assert_eq!(packed.range.len(), n, "range SoA stride");
    assert_eq!(packed.gate_flags.len(), n, "gate_flags SoA stride");
    assert_eq!(packed.delivery_kind.len(), n, "delivery_kind SoA stride");
    assert_eq!(
        packed.effect_kinds.len(),
        n * MAX_EFFECTS_PER_PROGRAM,
        "effect_kinds per-effect SoA stride",
    );
    assert_eq!(
        packed.effect_payload_a.len(),
        n * MAX_EFFECTS_PER_PROGRAM,
        "effect_payload_a per-effect SoA stride",
    );
    assert_eq!(
        packed.effect_payload_b.len(),
        n * MAX_EFFECTS_PER_PROGRAM,
        "effect_payload_b per-effect SoA stride",
    );

    // Every program with a non-empty `effects` SmallVec at the IR level
    // must pack at least one non-EMPTY discriminant. A silent zero-fill
    // bug in the packer would otherwise drop the verb without a peep.
    let mut programs_with_effects = 0usize;
    let mut packed_effect_slots_filled = 0usize;
    for slot in 0..n {
        let raw = (slot as u32) + 1;
        let id = engine::ability::AbilityId::new(raw)
            .expect("AbilityId::new(slot+1) is non-zero");
        let prog = built.registry.get(id).expect("slot in range");
        if prog.effects.is_empty() {
            continue;
        }
        programs_with_effects += 1;
        let base = slot * MAX_EFFECTS_PER_PROGRAM;
        let mut filled_in_this_program = 0;
        for i in 0..prog.effects.len() {
            assert_ne!(
                packed.effect_kinds[base + i],
                EFFECT_KIND_EMPTY,
                "program slot {slot} effect {i}: packer left an EMPTY discriminant \
                 even though the IR-level effect exists (look for a missing arm in \
                 engine::ability::packed::pack_effect_op)",
            );
            filled_in_this_program += 1;
        }
        packed_effect_slots_filled += filled_in_this_program;
        // Slots beyond the program's effect count must be the EMPTY
        // sentinel — verifies the packer doesn't write past the end.
        for i in prog.effects.len()..MAX_EFFECTS_PER_PROGRAM {
            assert_eq!(
                packed.effect_kinds[base + i],
                EFFECT_KIND_EMPTY,
                "program slot {slot} effect {i}: trailing slot must be EMPTY sentinel",
            );
        }
    }

    eprintln!(
        "Packed {n} LoL ability programs ({programs_with_effects} carry instant effects, \
         {packed_effect_slots_filled} EffectOp slots filled)"
    );
    assert!(
        programs_with_effects >= 1,
        "at least one program must carry instant effects in this corpus"
    );
}
