//! Pin: AOE dispatch detection survives a partial .ability corpus
//! failure — one decl that fails to lower MUST NOT silently disable
//! AOE Path B for its peers.
//!
//! ## Why this matters (Gap squad_skirmish#B)
//!
//! Pre-fix `crates/dsl_compiler/src/build_helper.rs` set
//!
//! ```text
//! let aoe_dispatch = built_registry
//!     .as_ref()
//!     .map(|br| ... per_effect_areas.iter().any(|a| a.is_some()) ...)
//!     .unwrap_or(false);
//! ```
//!
//! `build_registry` returns `Err(_)` if ANY .ability in the input set
//! fails to lower. The closure-on-`Some(_)` then never runs — the
//! whole detection collapses to `false`. In the squad_skirmish gap,
//! `Daze.ability` carried the bare `stun 8` (Gap A) which rejected at
//! the lower step and dragged the entire registry down with it; the
//! `aoe_dispatch=false` result then disabled AOE Path B emit even
//! though the peer `Volley.ability` declared
//! `damage 6 in spread(4.0, 8)`. Two unrelated authoring errors fused
//! into one near-impossible-to-debug runtime symptom (no AOE damage
//! despite a clearly AOE ability).
//!
//! Post-fix the AOE detection iterates whatever programs DID lower
//! (per-decl, NOT per-file — `lower_ability_file` short-circuits on
//! the first decl error, so we go via `lower_ability_decl`). A single
//! broken .ability no longer disables AOE for its peers.
//!
//! ## What this exercises
//!
//! Hand-build a 2-file `.ability` corpus where:
//!   * `Volley.ability` declares an AOE shape (`damage 6 in
//!     spread(4.0, 8)`).
//!   * `Daze.ability` carries the broken `stun 8` (no time suffix —
//!     Gap A territory).
//!
//! Then drive `build_helper::detect_aoe_dispatch` directly with the
//! parsed corpus + the corresponding (failed) registry build outcome
//! and assert:
//!   1. `build_registry` returns `Err(_)` because of `Daze.ability`'s
//!      bad `stun 8`.
//!   2. `detect_aoe_dispatch(None, &files)` returns `true` — the
//!      partial-failure fallback walks per-decl and sees Volley's AOE
//!      shape despite Daze rejecting.
//!   3. The all-clean corpus path (`Volley.ability` alone) ALSO
//!      returns `true` — the happy path is unchanged.

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_registry::build_registry;
use dsl_compiler::build_helper::detect_aoe_dispatch;

const VOLLEY_AOE_ABILITY: &str = "
ability Volley {
    target: enemy
    range: 12.0
    cooldown: 18
    hint: damage

    damage 6.0 in spread(4.0, 8)
}
";

const DAZE_BROKEN_STUN: &str = "
ability Daze {
    target: enemy
    range: 6.0
    cooldown: 20
    hint: crowd_control

    stun 8
}
";

const DAZE_GOOD_STUN: &str = "
ability Daze {
    target: enemy
    range: 6.0
    cooldown: 20
    hint: crowd_control

    stun 800ms
}
";

fn parse(label: &str, src: &str) -> (String, dsl_ast::AbilityFile) {
    (
        label.to_string(),
        parse_ability_file(src)
            .unwrap_or_else(|e| panic!("parse {label}: {e:?}")),
    )
}

/// Pin the canonical Gap B repro: registry build fails because of
/// Daze, yet AOE detection still picks up Volley's `spread`.
#[test]
fn aoe_dispatch_true_when_one_peer_fails_to_lower() {
    let files = vec![
        parse("Daze.ability", DAZE_BROKEN_STUN),
        parse("Volley.ability", VOLLEY_AOE_ABILITY),
    ];

    // Pre-condition: the registry build MUST fail because of Daze's
    // bare `stun 8`. If this assertion ever flips, the regression
    // surface has moved (e.g. the lowering pass started accepting
    // bare ints) and this test no longer pins what it claims to pin.
    let registry_err = build_registry(&files)
        .err()
        .expect("registry build must fail because of `stun 8` in Daze");
    let rendered = format!("{registry_err:?}");
    assert!(
        rendered.contains("EffectArgExpectedDuration") || rendered.contains("stun"),
        "registry build error should name the stun-bare-int culprit; got: {rendered}",
    );

    // Post-fix surface: detect_aoe_dispatch with `None` registry MUST
    // walk per-decl and find Volley's AOE shape.
    let aoe = detect_aoe_dispatch(None, &files);
    assert!(
        aoe,
        "AOE detection must return true even when Daze fails to lower — \
         Volley.ability carries `damage 6.0 in spread(4.0, 8)` and that \
         signal MUST survive the partial corpus failure",
    );
}

/// Regression guard for the happy path: when the WHOLE corpus lowers
/// cleanly, `built_registry` is `Some(_)` and we still return `true`
/// for any AOE-bearing program. Without this pin, a refactor that
/// over-applied the per-decl fallback could regress the happy-path
/// surface.
#[test]
fn aoe_dispatch_true_when_corpus_lowers_cleanly() {
    let files = vec![
        parse("Daze.ability", DAZE_GOOD_STUN),
        parse("Volley.ability", VOLLEY_AOE_ABILITY),
    ];

    let registry = build_registry(&files)
        .expect("clean corpus must build a complete registry");
    let aoe = detect_aoe_dispatch(Some(&registry), &files);
    assert!(
        aoe,
        "AOE detection must return true when the corpus is clean and \
         contains an AOE ability",
    );
}

/// Pin the negative case: a corpus with NO AOE shapes (and no
/// failures) returns `false`. Otherwise `aoe_dispatch=true` would be
/// the trivially-correct test for ALL inputs and the post-fix path
/// would not actually be exercising the per-decl walk.
#[test]
fn aoe_dispatch_false_when_no_corpus_uses_aoe() {
    let single_target_only = "
ability Strike {
    target: enemy
    range: 4.0
    cooldown: 6
    hint: damage

    damage 12.0
}
";
    let files = vec![parse("Strike.ability", single_target_only)];
    let registry = build_registry(&files)
        .expect("clean single-target corpus must build a complete registry");
    let aoe = detect_aoe_dispatch(Some(&registry), &files);
    assert!(
        !aoe,
        "AOE detection must return false when no program declares an \
         AOE shape",
    );
}

/// Pin: the empty-corpus case (no .ability files at all) returns
/// `false`. This is the path every fixture without an
/// `assets/ability_test/<fixture>/` directory takes.
#[test]
fn aoe_dispatch_false_when_corpus_empty() {
    let files: Vec<(String, dsl_ast::AbilityFile)> = Vec::new();
    let aoe = detect_aoe_dispatch(None, &files);
    assert!(
        !aoe,
        "AOE detection must return false for an empty corpus",
    );
}

/// Cross-gap pin: when the ENTIRE corpus has only the broken file
/// (Daze with bare `stun 8`), AOE detection returns `false` because
/// no program declared an AOE shape — NOT because Daze failed to
/// lower. The post-fix code MUST distinguish "no AOE" from "could
/// not check for AOE".
#[test]
fn aoe_dispatch_false_when_only_broken_decls_and_no_aoe() {
    let files = vec![parse("Daze.ability", DAZE_BROKEN_STUN)];
    // build_registry fails here too.
    assert!(build_registry(&files).is_err());
    let aoe = detect_aoe_dispatch(None, &files);
    assert!(
        !aoe,
        "AOE detection must return false when the only file fails to \
         lower AND declares no AOE shape (no false positive on broken \
         non-AOE decls)",
    );
}
