//! village_economy acceptance binding check — re-parses
//! `assets/ability_test/village_economy/*.ability` at fixture-construction
//! time, lowers them through the .ability pipeline, and asserts every
//! program landed at the expected AbilityId slot. Drift surfaces here
//! at startup rather than as silent wrong-slot dispatch downstream.
//!
//! Slot pinning order (alphabetised filenames in `names`):
//!   ForgeIron      → AbilityId(1)
//!   OfferDeal      → AbilityId(2)
//!   WalkToWorkshop → AbilityId(3)
//!
//! Each .sim verb body's `apply_ability N by self target X` literal must
//! agree with these constants — drift surfaces as silent wrong-program
//! dispatch (chronicle records carry the wrong payloads).

use std::path::PathBuf;

use engine::ability::AbilityId;

pub const FORGE_IRON_EXPECTED_ABILITY_ID: u32 = 1;
pub const OFFER_DEAL_EXPECTED_ABILITY_ID: u32 = 2;
pub const WALK_TO_WORKSHOP_EXPECTED_ABILITY_ID: u32 = 3;

/// Read + parse + build the AbilityRegistry over every .ability file
/// under `assets/ability_test/village_economy/`. Mirrors
/// `spy_network_runtime::binding_check::build_spy_network_registry`.
pub(crate) fn build_village_economy_registry() -> dsl_compiler::ability_registry::BuiltRegistry {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set by cargo");
    let corpus = PathBuf::from(manifest)
        .join("..")
        .join("..")
        .join("assets")
        .join("ability_test")
        .join("village_economy");

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
    // ForgeIron, OfferDeal, WalkToWorkshop alphabetised.
    let names = [
        "ForgeIron.ability",
        "OfferDeal.ability",
        "WalkToWorkshop.ability",
    ];
    let files: Vec<(String, _)> = names
        .iter()
        .map(|name| {
            let src = read(name);
            (name.to_string(), parse(name, &src))
        })
        .collect();

    dsl_compiler::ability_registry::build_registry(&files)
        .expect("build_registry over village_economy corpus")
}

/// Single binding-check entry point — one slot pin per .ability.
/// Called from `VillageEconomyState::new` at fixture construction time.
pub fn assert_ability_registry_matches_sim_constants() {
    let built = build_village_economy_registry();

    let forge_id = *built
        .names
        .get("ForgeIron")
        .expect("ForgeIron registered in name table");
    assert_eq!(
        forge_id,
        AbilityId::new(FORGE_IRON_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "ForgeIron's AbilityId drifted from slot {} — re-check the \
         `names` literal in `build_village_economy_registry()`.",
        FORGE_IRON_EXPECTED_ABILITY_ID,
    );

    let offer_id = *built
        .names
        .get("OfferDeal")
        .expect("OfferDeal registered in name table");
    assert_eq!(
        offer_id,
        AbilityId::new(OFFER_DEAL_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "OfferDeal's AbilityId drifted from slot {} — re-check the \
         `names` literal in `build_village_economy_registry()`.",
        OFFER_DEAL_EXPECTED_ABILITY_ID,
    );

    let walk_id = *built
        .names
        .get("WalkToWorkshop")
        .expect("WalkToWorkshop registered in name table");
    assert_eq!(
        walk_id,
        AbilityId::new(WALK_TO_WORKSHOP_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "WalkToWorkshop's AbilityId drifted from slot {} — re-check the \
         `names` literal in `build_village_economy_registry()`.",
        WALK_TO_WORKSHOP_EXPECTED_ABILITY_ID,
    );
}
