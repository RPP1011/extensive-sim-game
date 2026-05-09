//! wave_defense acceptance binding check — re-parses
//! `assets/ability_test/wave_defense/*.ability` at fixture-construction
//! time, lowers them through the .ability pipeline, and asserts every
//! program landed at the expected AbilityId slot. Drift surfaces here
//! at startup rather than as silent wrong-slot dispatch downstream.
//!
//! Slot pinning order (alphabetised filenames in `names`) — Phase E
//! voxel-engine integration added BuildPalisade as a 6th ability:
//!   BuildPalisade  → AbilityId(1)   (BuildPalisade verb dispatches `apply_ability 1`; place_voxel "palisade")
//!   MonsterCleave  → AbilityId(2)   (MonsterCleaveScan dispatches `apply_ability 2`)
//!   SpawnHorde     → AbilityId(3)   (SpawnHorde verb dispatches `apply_ability 3`; count=64)
//!   SpawnLarge     → AbilityId(4)   (SpawnLarge verb dispatches `apply_ability 4`; count=32)
//!   SpawnMedium    → AbilityId(5)   (SpawnMedium verb dispatches `apply_ability 5`; count=16)
//!   SpawnSmall     → AbilityId(6)   (SpawnSmall verb dispatches `apply_ability 6`; count=8)
//!
//! Each .sim verb body's `apply_ability N by self target X` literal
//! must agree with these constants — drift surfaces as silent
//! wrong-program dispatch (chronicle records carry the wrong payloads).

use std::path::PathBuf;

use engine::ability::AbilityId;

pub const BUILD_PALISADE_EXPECTED_ABILITY_ID: u32 = 1;
pub const MONSTER_CLEAVE_EXPECTED_ABILITY_ID: u32 = 2;
pub const SPAWN_HORDE_EXPECTED_ABILITY_ID: u32 = 3;
pub const SPAWN_LARGE_EXPECTED_ABILITY_ID: u32 = 4;
pub const SPAWN_MEDIUM_EXPECTED_ABILITY_ID: u32 = 5;
pub const SPAWN_SMALL_EXPECTED_ABILITY_ID: u32 = 6;

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
    // Alphabetised: BuildPalisade + MonsterCleave + SpawnHorde +
    // SpawnLarge + SpawnMedium + SpawnSmall → AbilityIds (1, 2, 3, 4, 5, 6).
    let names = [
        "BuildPalisade.ability",
        "MonsterCleave.ability",
        "SpawnHorde.ability",
        "SpawnLarge.ability",
        "SpawnMedium.ability",
        "SpawnSmall.ability",
    ];
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

    let assert_slot = |name: &str, expected: u32| {
        let id = *built
            .names
            .get(name)
            .unwrap_or_else(|| panic!("{name} registered in name table"));
        assert_eq!(
            id,
            AbilityId::new(expected).expect("non-zero AbilityId"),
            "{name}'s AbilityId drifted from slot {expected} — re-check the \
             `names` literal in `build_wave_defense_registry()`.",
        );
    };

    assert_slot("BuildPalisade", BUILD_PALISADE_EXPECTED_ABILITY_ID);
    assert_slot("MonsterCleave", MONSTER_CLEAVE_EXPECTED_ABILITY_ID);
    assert_slot("SpawnHorde",    SPAWN_HORDE_EXPECTED_ABILITY_ID);
    assert_slot("SpawnLarge",    SPAWN_LARGE_EXPECTED_ABILITY_ID);
    assert_slot("SpawnMedium",   SPAWN_MEDIUM_EXPECTED_ABILITY_ID);
    assert_slot("SpawnSmall",    SPAWN_SMALL_EXPECTED_ABILITY_ID);
}
