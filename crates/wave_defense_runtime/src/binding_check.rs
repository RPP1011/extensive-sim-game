//! wave_defense acceptance binding check — re-parses
//! `assets/ability_test/wave_defense/*.ability` at fixture-construction
//! time, lowers them through the .ability pipeline, and asserts every
//! program landed at the expected AbilityId slot. Drift surfaces here
//! at startup rather than as silent wrong-slot dispatch downstream.
//!
//! Slot pinning order (alphabetised filenames in `names`):
//!   MonsterCleave  → AbilityId(1)   (MonsterCleaveScan dispatches `apply_ability 1`)
//!   SpawnWave      → AbilityId(2)   (SpawnWave verb dispatches `apply_ability 2`)
//!
//! Each .sim verb body's `apply_ability N by self target X` literal
//! must agree with these constants — drift surfaces as silent
//! wrong-program dispatch (chronicle records carry the wrong payloads).

use std::path::PathBuf;

use engine::ability::AbilityId;

pub const MONSTER_CLEAVE_EXPECTED_ABILITY_ID: u32 = 1;
pub const SPAWN_WAVE_EXPECTED_ABILITY_ID: u32 = 2;

/// Read + parse + build the AbilityRegistry over every .ability file
/// under `assets/ability_test/wave_defense/`. Mirrors
/// `village_economy_runtime::binding_check::build_village_economy_registry`.
pub(crate) fn build_wave_defense_registry()
-> dsl_compiler::ability_registry::BuiltRegistry {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set by cargo");
    let corpus = PathBuf::from(manifest)
        .join("..")
        .join("..")
        .join("assets")
        .join("ability_test")
        .join("wave_defense");

    let read = |name: &str| {
        let path = corpus.join(name);
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    };
    let parse = |name: &str, src: &str| {
        dsl_ast::parse_ability_file(src)
            .unwrap_or_else(|e| panic!("parse {name}: {e:?}"))
    };

    // Source-order names list — also the registry's slot order.
    // MonsterCleave + SpawnWave alphabetised → (1, 2) AbilityIds.
    let names = ["MonsterCleave.ability", "SpawnWave.ability"];
    let files: Vec<(String, _)> = names
        .iter()
        .map(|name| {
            let src = read(name);
            (name.to_string(), parse(name, &src))
        })
        .collect();

    dsl_compiler::ability_registry::build_registry(&files)
        .expect("build_registry over wave_defense corpus")
}

/// Single binding-check entry point — one slot pin per .ability.
/// Called from `WaveDefenseState::new` at fixture construction time.
pub fn assert_ability_registry_matches_sim_constants() {
    let built = build_wave_defense_registry();

    let cleave_id = *built
        .names
        .get("MonsterCleave")
        .expect("MonsterCleave registered in name table");
    assert_eq!(
        cleave_id,
        AbilityId::new(MONSTER_CLEAVE_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "MonsterCleave's AbilityId drifted from slot {} — re-check the \
         `names` literal in `build_wave_defense_registry()`.",
        MONSTER_CLEAVE_EXPECTED_ABILITY_ID,
    );

    let spawn_id = *built
        .names
        .get("SpawnWave")
        .expect("SpawnWave registered in name table");
    assert_eq!(
        spawn_id,
        AbilityId::new(SPAWN_WAVE_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "SpawnWave's AbilityId drifted from slot {} — re-check the \
         `names` literal in `build_wave_defense_registry()`.",
        SPAWN_WAVE_EXPECTED_ABILITY_ID,
    );
}
