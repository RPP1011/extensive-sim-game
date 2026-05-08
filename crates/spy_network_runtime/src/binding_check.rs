//! spy_network acceptance binding check — re-parses
//! `assets/ability_test/spy_network/*.ability` at fixture-construction
//! time, lowers them through the .ability pipeline, and asserts every
//! program landed at the expected AbilityId slot. Drift surfaces here
//! at startup rather than as silent wrong-slot dispatch downstream.
//!
//! Unlike `duel_abilities_runtime`, the spy_network .sim verb bodies
//! do NOT use `apply_ability N` dispatch — they `emit` chronicle events
//! directly. The slot pins are still authored so a future migration to
//! the apply_ability path lands without re-doing the binding-check
//! shape, AND so the registry-build pipeline gets exercised by every
//! cargo build of this crate (catches Wave 1.x lowering regressions
//! before they spread).

use std::path::PathBuf;

use engine::ability::AbilityId;

/// Disguise lands first (alphabetised list head: Disguise / Slander /
/// Strike). The runtime asserts the registered slot at startup so any
/// drift in `build_registry`'s ordering surfaces as a panic.
pub const DISGUISE_EXPECTED_ABILITY_ID: u32 = 1;

/// Slander — second slot in the alphabetised file list.
pub const SLANDER_EXPECTED_ABILITY_ID: u32 = 2;

/// Strike — third slot.
pub const STRIKE_EXPECTED_ABILITY_ID: u32 = 3;

/// Read + parse + build the AbilityRegistry over every .ability file
/// under `assets/ability_test/spy_network/`. Mirrors
/// `duel_abilities_runtime::binding_check::build_duel_abilities_registry`.
pub(crate) fn build_spy_network_registry() -> dsl_compiler::ability_registry::BuiltRegistry {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set by cargo");
    let corpus = PathBuf::from(manifest)
        .join("..")
        .join("..")
        .join("assets")
        .join("ability_test")
        .join("spy_network");

    let read = |name: &str| {
        let path = corpus.join(name);
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    };
    let parse = |name: &str, src: &str| {
        dsl_ast::parse_ability_file(src)
            .unwrap_or_else(|e| panic!("parse {name}: {e:?}"))
    };

    // Source-order names list — also the registry's slot order. Pinned
    // by the constants above; reorder them and the asserts in
    // `assert_ability_registry_matches_sim_constants` will fire.
    let names = [
        "Disguise.ability",
        "Slander.ability",
        "Strike.ability",
    ];
    let files: Vec<(String, _)> = names
        .iter()
        .map(|name| {
            let src = read(name);
            (name.to_string(), parse(name, &src))
        })
        .collect();

    dsl_compiler::ability_registry::build_registry(&files)
        .expect("build_registry over spy_network corpus")
}

/// Single binding-check entry point — one slot pin per .ability.
/// Called from `SpyNetworkState::new` at fixture construction time.
pub fn assert_ability_registry_matches_sim_constants() {
    let built = build_spy_network_registry();

    let disguise_id = *built
        .names
        .get("Disguise")
        .expect("Disguise registered in name table");
    assert_eq!(
        disguise_id,
        AbilityId::new(DISGUISE_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "Disguise's AbilityId drifted from slot {} — re-check the \
         `names` literal in `build_spy_network_registry()`.",
        DISGUISE_EXPECTED_ABILITY_ID,
    );

    let slander_id = *built
        .names
        .get("Slander")
        .expect("Slander registered in name table");
    assert_eq!(
        slander_id,
        AbilityId::new(SLANDER_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "Slander's AbilityId drifted from slot {} — re-check the \
         `names` literal in `build_spy_network_registry()`.",
        SLANDER_EXPECTED_ABILITY_ID,
    );

    let strike_id = *built
        .names
        .get("Strike")
        .expect("Strike registered in name table");
    assert_eq!(
        strike_id,
        AbilityId::new(STRIKE_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "Strike's AbilityId drifted from slot {} — re-check the \
         `names` literal in `build_spy_network_registry()`.",
        STRIKE_EXPECTED_ABILITY_ID,
    );
}
