//! Wave 1.3 lowering tests — the new `structure` top-level block parses
//! but errors cleanly at the lowering boundary, and the Wave 1 corpus
//! (Strike / ShieldUp / Mend / Bleed / Reap / Vampirize / Fortify) still
//! lowers without regression.
//!
//! Per `crates/dsl_compiler/src/ability_lower.rs` Wave 1.3 module
//! contract, lowering surfaces:
//!   * `LowerError::StructureBlockNotImplemented { name, span }` for any
//!     non-empty `file.structures`.
//!
//! Voxel rasterization + `StructureRegistry` (spec §12.2 GPU work) is
//! Wave 2+.

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_lower::{lower_ability_decl, lower_ability_file, LowerError};

// ---------------------------------------------------------------------------
// 1. `structure <Name>(<params>) { ... }` top-level — StructureBlockNotImplemented
// ---------------------------------------------------------------------------

#[test]
fn lowering_file_with_structure_returns_unimplemented_error() {
    let src = r#"
structure Castle(wall_mat: Material = stone, height: int = 8) {
    bounds: box(20, $height, 20)
    place $wall_mat in box(20, 1, 20)
}
"#;
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_file(&file).expect_err("lowering must error");
    match err {
        LowerError::StructureBlockNotImplemented { name, span } => {
            assert_eq!(name, "Castle");
            assert!(span.start < span.end, "non-empty span");
            assert!(span.end <= src.len(), "in-source span");
        }
        other => panic!("expected StructureBlockNotImplemented; got {other:?}"),
    }
}

#[test]
fn lowering_structure_block_diagnostic_names_the_structure() {
    let src = "structure Empty() { }";
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_file(&file).expect_err("lowering must error");
    let msg = err.to_string();
    assert!(
        msg.contains("Empty"),
        "diagnostic must name the structure; got: {msg}"
    );
    assert!(
        msg.contains("structure"),
        "diagnostic must mention `structure`; got: {msg}"
    );
}

#[test]
fn lowering_first_structure_short_circuits() {
    // Multiple structures — the first one in source order should be the
    // surfaced diagnostic so authors get a stable error location.
    let src = r#"
structure FirstOne() { }
structure SecondOne() { place stone in box(1, 1, 1) }
"#;
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_file(&file).expect_err("lowering must error");
    match err {
        LowerError::StructureBlockNotImplemented { name, .. } => {
            assert_eq!(name, "FirstOne", "first structure wins");
        }
        other => panic!("expected StructureBlockNotImplemented; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 2. Wave 1 corpus regression — no structures, lowers cleanly
// ---------------------------------------------------------------------------

#[test]
fn lowering_wave_1_corpus_still_works() {
    // The seven-ability Wave 1 corpus uses neither templates,
    // instantiations, nor structures, so the lowering pipeline should
    // produce a valid AbilityProgram for each.
    for (name, src) in [
        ("Strike",
         "ability Strike { target: enemy range: 5.0 cooldown: 1s hint: damage damage 15 }"),
        ("ShieldUp",
         "ability ShieldUp { target: self cooldown: 4s hint: defense shield 50 }"),
        ("Mend",
         "ability Mend { target: self cooldown: 3s hint: heal heal 20 }"),
        ("Bleed",
         "ability Bleed { target: enemy range: 3.0 cooldown: 2s hint: damage damage 5 }"),
        ("Reap",
         "ability Reap { target: enemy range: 4.0 cooldown: 5s hint: damage execute 10.0 }"),
        ("Vampirize",
         "ability Vampirize { target: self cooldown: 6s hint: utility lifesteal 0.5 4s }"),
        ("Fortify",
         "ability Fortify { target: self cooldown: 5s hint: defense damage_modify 0.5 5s }"),
    ] {
        let file = parse_ability_file(src).unwrap_or_else(|e| panic!("{name} parses: {e}"));
        let prog = lower_ability_decl(&file.abilities[0])
            .unwrap_or_else(|e| panic!("{name} lowers: {e:?}"));
        assert_eq!(prog.effects.len(), 1, "{name} should have one effect");
    }
}
