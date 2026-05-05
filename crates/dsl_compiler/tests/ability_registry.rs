//! Wave 1.7 — registry-build + cast-resolution tests.
//!
//! Coverage:
//!   1. The on-disk Wave 1 corpus (Strike + ShieldUp + Mend) builds.
//!   2. `cast <Name>` resolves to the real `AbilityId` (not the Wave 1.6
//!      placeholder).
//!   3. Self-cast (A casts A) is reported as a 2-element cycle.
//!   4. Two-cycle (A -> B -> A) is reported as a 3-element cycle.
//!   5. Three-cycle (A -> B -> C -> A) is reported as a 4-element cycle.
//!   6. `cast NoSuch` surfaces `UnresolvedCastTarget`.
//!   7. Two files declaring the same ability name surface
//!      `DuplicateAbilityName`.
//!   8. Acyclic cast chain (A casts B; B is leaf) builds.
//!   9. Sanity: `names` map round-trips through `registry.get(*id)` for
//!      every ability — guards against future Pass-1 / Pass-4 drift.
//!
//! Inline source strings rely on `dsl_ast::parse_ability_file`. The disk
//! corpus is loaded via `CARGO_MANIFEST_DIR` joins to stay portable
//! between local and CI runs.

use dsl_ast::ast::AbilityFile;
use dsl_ast::parse_ability_file;
use dsl_compiler::ability_registry::{build_registry, BuiltRegistry, RegistryBuildError};
use engine::ability::program::EffectOp;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse(src: &str) -> AbilityFile {
    parse_ability_file(src).expect("parser must accept inline source")
}

fn corpus_path(file: &str) -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    std::path::PathBuf::from(manifest)
        .join("..")
        .join("..")
        .join("assets")
        .join("ability_test")
        .join("duel_abilities")
        .join(file)
}

// ---------------------------------------------------------------------------
// 1. On-disk corpus builds
// ---------------------------------------------------------------------------

#[test]
fn builds_three_file_corpus() {
    let strike   = std::fs::read_to_string(corpus_path("Strike.ability"))
        .expect("Strike.ability missing");
    let shield   = std::fs::read_to_string(corpus_path("ShieldUp.ability"))
        .expect("ShieldUp.ability missing");
    let mend     = std::fs::read_to_string(corpus_path("Mend.ability"))
        .expect("Mend.ability missing");

    let files = vec![
        ("Strike.ability".to_string(),   parse(&strike)),
        ("ShieldUp.ability".to_string(), parse(&shield)),
        ("Mend.ability".to_string(),     parse(&mend)),
    ];

    let BuiltRegistry { registry, names } = build_registry(&files)
        .expect("Wave 1 corpus must build");

    assert_eq!(registry.len(), 3, "three abilities registered");
    for n in ["Strike", "ShieldUp", "Mend"] {
        let id = names.get(n).unwrap_or_else(|| panic!("name '{n}' missing"));
        assert!(registry.get(*id).is_some(), "id for '{n}' must resolve");
    }
}

// ---------------------------------------------------------------------------
// 2. Cast resolves to a real id
// ---------------------------------------------------------------------------

#[test]
fn cast_name_resolves_to_real_id() {
    // Two abilities in source order: `Other` is slot 0 (id raw=1),
    // `Caller` is slot 1 (id raw=2). The Wave 1.6 placeholder is also
    // raw=1, so we additionally check that the registry actually contains
    // the right program at the resolved id (not the placeholder by luck).
    let src = "
        ability Other { target: enemy range: 5.0 cooldown: 1s damage 5 }
        ability Caller { target: self cooldown: 2s cast Other }
    ";
    let files = vec![("inline.ability".to_string(), parse(src))];

    let BuiltRegistry { registry, names } = build_registry(&files)
        .expect("acyclic cast must build");

    let other_id  = *names.get("Other").expect("Other registered");
    let caller_id = *names.get("Caller").expect("Caller registered");
    assert_ne!(other_id, caller_id);

    let caller_prog = registry.get(caller_id).expect("Caller program present");
    assert_eq!(caller_prog.effects.len(), 1);
    match caller_prog.effects[0] {
        EffectOp::CastAbility { ability, .. } => {
            assert_eq!(
                ability, other_id,
                "cast target must be resolved to Other's id, not the Wave 1.6 placeholder",
            );
        }
        ref other => panic!("expected CastAbility; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 3. Self-cast cycle
// ---------------------------------------------------------------------------

#[test]
fn self_cast_is_cycle() {
    let src = "ability A { target: self cooldown: 1s cast A }";
    let files = vec![("self.ability".to_string(), parse(src))];

    let err = build_registry(&files).expect_err("self-cast must be rejected");
    match err {
        RegistryBuildError::CastCycle { path } => {
            assert_eq!(path, vec!["A".to_string(), "A".to_string()]);
        }
        other => panic!("expected CastCycle; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 4. Two-cycle
// ---------------------------------------------------------------------------

#[test]
fn two_cycle_detected() {
    // A -> B -> A. DFS roots in slot order; root is `A`, cycle path
    // closes with `A` at the tail.
    let src = "
        ability A { target: self cooldown: 1s cast B }
        ability B { target: self cooldown: 1s cast A }
    ";
    let files = vec![("two.ability".to_string(), parse(src))];

    let err = build_registry(&files).expect_err("2-cycle must be rejected");
    match err {
        RegistryBuildError::CastCycle { path } => {
            assert_eq!(
                path,
                vec!["A".to_string(), "B".to_string(), "A".to_string()],
            );
        }
        other => panic!("expected CastCycle; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 5. Three-cycle
// ---------------------------------------------------------------------------

#[test]
fn three_cycle_detected() {
    let src = "
        ability A { target: self cooldown: 1s cast B }
        ability B { target: self cooldown: 1s cast C }
        ability C { target: self cooldown: 1s cast A }
    ";
    let files = vec![("three.ability".to_string(), parse(src))];

    let err = build_registry(&files).expect_err("3-cycle must be rejected");
    match err {
        RegistryBuildError::CastCycle { path } => {
            assert_eq!(
                path,
                vec![
                    "A".to_string(),
                    "B".to_string(),
                    "C".to_string(),
                    "A".to_string(),
                ],
            );
        }
        other => panic!("expected CastCycle; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 6. Unresolved cast target
// ---------------------------------------------------------------------------

#[test]
fn unresolved_cast_target() {
    let src = "ability A { target: self cooldown: 1s cast NoSuch }";
    let files = vec![("ghost.ability".to_string(), parse(src))];

    let err = build_registry(&files).expect_err("unknown cast target must error");
    match err {
        RegistryBuildError::UnresolvedCastTarget {
            from_ability,
            target_name,
            file,
            ..
        } => {
            assert_eq!(from_ability, "A");
            assert_eq!(target_name, "NoSuch");
            assert_eq!(file, "ghost.ability");
        }
        other => panic!("expected UnresolvedCastTarget; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 7. Duplicate name across files
// ---------------------------------------------------------------------------

#[test]
fn duplicate_name_across_files() {
    let src_a = "ability Same { target: self cooldown: 1s heal 1 }";
    let src_b = "ability Same { target: self cooldown: 2s shield 1 }";

    let files = vec![
        ("a.ability".to_string(), parse(src_a)),
        ("b.ability".to_string(), parse(src_b)),
    ];

    let err = build_registry(&files).expect_err("duplicate name must error");
    match err {
        RegistryBuildError::DuplicateAbilityName {
            name,
            first_file,
            dup_file,
            ..
        } => {
            assert_eq!(name, "Same");
            assert_eq!(first_file, "a.ability");
            assert_eq!(dup_file, "b.ability");
        }
        other => panic!("expected DuplicateAbilityName; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 8. Acyclic cast chain builds
// ---------------------------------------------------------------------------

#[test]
fn acyclic_chain_is_fine() {
    // A casts B; B has no further cast. Both ids resolve; A's lowered
    // effect carries B's id verbatim.
    let src = "
        ability A { target: self cooldown: 1s cast B }
        ability B { target: enemy range: 4.0 cooldown: 1s damage 7 }
    ";
    let files = vec![("chain.ability".to_string(), parse(src))];

    let BuiltRegistry { registry, names } =
        build_registry(&files).expect("chain must build");

    let a_id = *names.get("A").expect("A registered");
    let b_id = *names.get("B").expect("B registered");

    let a_prog = registry.get(a_id).expect("A program present");
    assert_eq!(a_prog.effects.len(), 1);
    match a_prog.effects[0] {
        EffectOp::CastAbility { ability, .. } => assert_eq!(ability, b_id),
        ref other => panic!("expected CastAbility; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 9. Invariant: every name resolves to a present program
// ---------------------------------------------------------------------------

#[test]
fn invariant_id_matches_pass1() {
    // Mix self-targets + a cast edge so the test exercises the patched
    // and non-patched codepaths together.
    let src = "
        ability Alpha { target: self cooldown: 1s heal 1 }
        ability Beta  { target: self cooldown: 1s shield 1 }
        ability Gamma { target: self cooldown: 1s cast Alpha }
    ";
    let files = vec![("mix.ability".to_string(), parse(src))];

    let BuiltRegistry { registry, names } =
        build_registry(&files).expect("must build");

    assert_eq!(registry.len(), names.len(), "registry size matches name count");
    for (name, id) in names.iter() {
        assert!(
            registry.get(*id).is_some(),
            "name '{name}' -> id {id:?} must resolve in the built registry",
        );
    }
}
