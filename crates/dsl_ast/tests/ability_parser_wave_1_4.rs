//! Wave 1.4 `.ability` parser surface tests for the three body-block
//! forms beyond bare effect statements (per
//! `docs/spec/ability_dsl_unified.md` §4.2 / §4.4 / §9):
//!
//! 1. `recast: <int|dur>` and `recast_window: <duration>` headers
//!    (spec §4.2: "recast / recast_window | int / dur").
//! 2. `deliver <method> { params } { body }` body block — captured
//!    opaquely (spec §9 delivery hooks owned by Wave 2+).
//! 3. `morph { effects } into <Other>` body block.
//!
//! Plus an LoL-corpus parse-rate sanity check at the bottom that
//! tracks the pre-/post-Wave-1.4 unblock signal.

use dsl_ast::ast::{AbilityHeader, RecastValue};
use dsl_ast::parse_ability_file;

// ---------------------------------------------------------------------------
// 1. `recast:` and `recast_window:` headers
// ---------------------------------------------------------------------------

#[test]
fn parses_recast_header_count() {
    let src = "ability X { target: enemy cooldown: 1s recast: 3 damage 10 }";
    let file = parse_ability_file(src).expect("must parse");
    let decl = &file.abilities[0];
    let recast = decl
        .headers
        .iter()
        .find_map(|h| match h {
            AbilityHeader::Recast(v) => Some(*v),
            _ => None,
        })
        .expect("recast header present");
    assert_eq!(recast, RecastValue::Count(3));
}

#[test]
fn parses_recast_header_duration_seconds() {
    // Per spec §4.2 the duration form is also valid (`int / dur`).
    let src = "ability X { target: enemy cooldown: 1s recast: 4s damage 10 }";
    let file = parse_ability_file(src).expect("must parse");
    let recast = file.abilities[0]
        .headers
        .iter()
        .find_map(|h| match h {
            AbilityHeader::Recast(v) => Some(*v),
            _ => None,
        })
        .unwrap();
    match recast {
        RecastValue::Duration(d) => assert_eq!(d.millis, 4_000),
        other => panic!("expected RecastValue::Duration; got {other:?}"),
    }
}

#[test]
fn parses_recast_header_duration_millis() {
    let src = "ability X { target: enemy cooldown: 1s recast: 250ms damage 10 }";
    let file = parse_ability_file(src).expect("must parse");
    let recast = file.abilities[0]
        .headers
        .iter()
        .find_map(|h| match h {
            AbilityHeader::Recast(v) => Some(*v),
            _ => None,
        })
        .unwrap();
    assert_eq!(recast, RecastValue::Duration(dsl_ast::ast::Duration { millis: 250 }));
}

#[test]
fn parses_recast_window_header() {
    let src = "ability X { target: enemy cooldown: 1s recast_window: 10s damage 10 }";
    let file = parse_ability_file(src).expect("must parse");
    let win = file.abilities[0]
        .headers
        .iter()
        .find_map(|h| match h {
            AbilityHeader::RecastWindow(d) => Some(*d),
            _ => None,
        })
        .expect("recast_window header present");
    assert_eq!(win.millis, 10_000);
}

#[test]
fn parses_recast_with_recast_window_combined() {
    // Both headers in the same ability — common pattern in the LoL
    // corpus (Aatrox / Anivia / Ahri all do this).
    let src = "ability X { target: enemy cooldown: 1s recast: 3 recast_window: 15s damage 10 }";
    let file = parse_ability_file(src).expect("must parse");
    let decl = &file.abilities[0];
    let has_recast = decl
        .headers
        .iter()
        .any(|h| matches!(h, AbilityHeader::Recast(_)));
    let has_window = decl
        .headers
        .iter()
        .any(|h| matches!(h, AbilityHeader::RecastWindow(_)));
    assert!(has_recast && has_window, "both headers landed");
}

#[test]
fn duplicate_recast_header_is_error() {
    let src = "ability X { target: enemy cooldown: 1s recast: 1 recast: 2 damage 10 }";
    let err = parse_ability_file(src).expect_err("duplicate recast must error");
    let msg = err.to_string();
    assert!(
        msg.contains("duplicate") && msg.contains("recast"),
        "diagnostic must name the duplicate `recast:` header; got: {msg}"
    );
}

#[test]
fn negative_recast_count_is_error() {
    let src = "ability X { target: enemy cooldown: 1s recast: -1 damage 10 }";
    let err = parse_ability_file(src).expect_err("negative recast count must error");
    let msg = err.to_string();
    assert!(
        msg.contains("recast") && msg.contains("non-negative integer"),
        "diagnostic must explain integer/duration requirement; got: {msg}"
    );
}

#[test]
fn fractional_recast_count_is_error() {
    // `recast: 1.5` is not a duration (no unit) and not a valid count.
    let src = "ability X { target: enemy cooldown: 1s recast: 1.5 damage 10 }";
    let err = parse_ability_file(src).expect_err("fractional recast count must error");
    assert!(err.to_string().contains("recast"));
}

#[test]
fn recast_header_span_is_non_empty() {
    let src = "ability X { target: enemy cooldown: 1s recast: 4s damage 10 }";
    let file = parse_ability_file(src).expect("must parse");
    // Header value spans aren't carried on AbilityHeader::Recast variant
    // by design (the headers Vec is positional). Instead we sanity-check
    // the surrounding decl span covers the recast token.
    let decl = &file.abilities[0];
    let slice = &src[decl.span.start..decl.span.end];
    assert!(slice.contains("recast: 4s"));
}

// ---------------------------------------------------------------------------
// 2. `deliver { ... }` body block
// ---------------------------------------------------------------------------

#[test]
fn parses_minimal_deliver_block() {
    // Spec-shape minimal: `deliver projectile { speed: 10.0 } { on_hit { damage 5 } }`.
    // The brief sketch had `deliver { effects } cast_on enemy`, but the
    // real LoL corpus uses `deliver <method> { params } { body }` per
    // spec §9. We capture the whole invocation opaquely.
    let src = "ability X {
        target: enemy range: 5.0 cooldown: 1s
        deliver projectile { speed: 10.0 } { on_hit { damage 5 } }
    }";
    let file = parse_ability_file(src).expect("must parse");
    let decl = &file.abilities[0];
    let block = decl.deliver.as_ref().expect("deliver block present");
    assert_eq!(block.method, "projectile");
    assert!(block.raw.starts_with("deliver projectile"));
    assert!(block.raw.contains("speed: 10.0"));
    assert!(block.raw.contains("on_hit"));
    assert!(block.raw.contains("damage 5"));
    // Span is non-empty + bounds the raw slice.
    assert!(block.span.start < block.span.end);
    assert_eq!(&src[block.span.start..block.span.end], block.raw);
}

#[test]
fn parses_deliver_channel_block() {
    // Different delivery method — same opaque-capture machinery.
    let src = "ability X {
        target: enemy range: 5.0 cooldown: 1s
        deliver channel { duration: 2s, tick: 500ms } { on_tick { damage 7 } }
    }";
    let file = parse_ability_file(src).expect("must parse");
    let block = file.abilities[0].deliver.as_ref().unwrap();
    assert_eq!(block.method, "channel");
    assert!(block.raw.contains("duration: 2s"));
    assert!(block.raw.contains("on_tick"));
}

#[test]
fn parses_deliver_with_modifier_rich_inner() {
    // Inner body uses Wave 1.5 modifiers — our opaque capture must
    // preserve them verbatim (the inner is NOT parsed).
    let src = "ability X {
        target: enemy range: 5.0 cooldown: 1s
        deliver projectile { speed: 14.0 } {
            on_hit {
                damage 50 in circle(2.5) [PHYSICAL: 60]
                slow 0.3 for 2s
            }
        }
    }";
    let file = parse_ability_file(src).expect("must parse");
    let block = file.abilities[0].deliver.as_ref().unwrap();
    assert!(block.raw.contains("in circle(2.5)"));
    assert!(block.raw.contains("[PHYSICAL: 60]"));
    assert!(block.raw.contains("slow 0.3 for 2s"));
}

#[test]
fn parses_deliver_then_bare_effect_coexistence() {
    // Per AbilityDecl docs / Wave 1.4 module docs: spec §4.4 says
    // these are mutually exclusive, but the LoL corpus contradicts
    // (Ahri.SpiritRush has `deliver projectile {…}{…} dash to_target`).
    // We admit coexistence at parse time so the corpus parses; the
    // mutual-exclusion check moves to lowering (`MixedBody`).
    let src = "ability X {
        target: ground range: 5.0 cooldown: 25s
        deliver projectile { speed: 12.0 } {
            on_hit { damage 15 in circle(6.0) [MAGIC: 50] }
        }
        dash to_target
    }";
    let file = parse_ability_file(src).expect("must parse");
    let decl = &file.abilities[0];
    assert!(decl.deliver.is_some(), "deliver block landed");
    assert_eq!(decl.effects.len(), 1, "trailing dash effect landed");
    assert_eq!(decl.effects[0].verb, "dash");
}

#[test]
fn duplicate_deliver_block_is_error() {
    let src = "ability X {
        target: enemy range: 5.0 cooldown: 1s
        deliver projectile { speed: 5.0 } { on_hit { damage 5 } }
        deliver projectile { speed: 6.0 } { on_hit { damage 6 } }
    }";
    let err = parse_ability_file(src).expect_err("duplicate deliver must error");
    assert!(
        err.to_string().contains("more than one `deliver`"),
        "diagnostic must name the duplicate; got: {err}"
    );
}

#[test]
fn deliver_with_unbalanced_braces_errors_cleanly() {
    let src = "ability X {
        target: enemy range: 5.0 cooldown: 1s
        deliver projectile { speed: 10.0 } { on_hit { damage 5
    }";
    let err = parse_ability_file(src).expect_err("unbalanced inner brace must error");
    assert!(
        err.to_string().contains("end of input") || err.to_string().contains("unexpected"),
        "diagnostic must call out the unterminated block; got: {err}"
    );
}

#[test]
fn parses_aatrox_recast_header() {
    // Brief's named canary. Aatrox.TheDarkinBlade uses `recast: 1` and
    // `recast_window: 4s`; Aatrox.InfernalChains uses
    // `deliver projectile { … } { on_hit { … } }`. Aatrox also uses
    // `passive DeathbringerStance { trigger: periodic(5s) … }` — Wave
    // 1.4 stretched the trigger parser to accept `(args)` to keep this
    // canary green.
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("dataset")
        .join("abilities")
        .join("lol_heroes")
        .join("Aatrox.ability");
    let src = std::fs::read_to_string(&path).expect("Aatrox.ability readable");
    let file = parse_ability_file(&src).expect("Aatrox.ability parses cleanly post-1.4");
    // Spot-check: TheDarkinBlade carries the recast headers.
    let dark = file
        .abilities
        .iter()
        .find(|a| a.name == "TheDarkinBlade")
        .expect("TheDarkinBlade decl present");
    assert!(
        dark.headers.iter().any(|h| matches!(h, AbilityHeader::Recast(_))),
        "TheDarkinBlade has `recast:` header"
    );
    assert!(
        dark.headers.iter().any(|h| matches!(h, AbilityHeader::RecastWindow(_))),
        "TheDarkinBlade has `recast_window:` header"
    );
    // Spot-check: InfernalChains carries the deliver block.
    let chains = file
        .abilities
        .iter()
        .find(|a| a.name == "InfernalChains")
        .expect("InfernalChains decl present");
    let block = chains.deliver.as_ref().expect("InfernalChains has a deliver block");
    assert_eq!(block.method, "projectile");
}

#[test]
fn parses_three_more_lol_files_with_deliver() {
    // Sanity-check three more LoL files that lean on `deliver {…}`
    // for both projectile and other delivery methods. These should
    // ALL parse post-Wave-1.4 since their deliver blocks are
    // captured opaquely. If a future corpus drift breaks one, the
    // failure points at the file.
    let names = ["Brand", "Anivia", "Ahri"];
    for name in &names {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("dataset")
            .join("abilities")
            .join("lol_heroes")
            .join(format!("{name}.ability"));
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{name}.ability readable: {e}"));
        parse_ability_file(&src)
            .unwrap_or_else(|e| panic!("{name}.ability parses post-1.4: {e}"));
    }
}

// ---------------------------------------------------------------------------
// 3. `morph { ... } into <Other>` body block
// ---------------------------------------------------------------------------

#[test]
fn parses_morph_block_with_into() {
    let src = "ability X {
        target: self cooldown: 8s
        morph { damage 30 } into Heatseeker
    }";
    let file = parse_ability_file(src).expect("must parse");
    let block = file.abilities[0].morph.as_ref().expect("morph block present");
    assert_eq!(block.into, "Heatseeker");
    assert_eq!(block.effects.len(), 1);
    assert_eq!(block.effects[0].verb, "damage");
    assert!(block.span.start < block.span.end);
}

#[test]
fn parses_morph_block_with_multi_effect_body() {
    let src = "ability X {
        target: self cooldown: 8s
        morph {
            damage 30 in circle(3.0) [MAGIC: 60]
            heal 10
            slow 0.3 for 2s
        } into FireForm
    }";
    let file = parse_ability_file(src).expect("must parse");
    let block = file.abilities[0].morph.as_ref().unwrap();
    assert_eq!(block.into, "FireForm");
    assert_eq!(block.effects.len(), 3);
    // Inner effects ARE re-parsed via parse_nested_block, so Wave 1.5
    // modifiers should land on each statement.
    assert!(block.effects[0].area.is_some(), "first effect kept its `in circle(3.0)`");
    assert_eq!(block.effects[0].tags.len(), 1);
    assert_eq!(block.effects[0].tags[0].name, "MAGIC");
}

#[test]
fn morph_without_into_is_error() {
    let src = "ability X {
        target: self cooldown: 8s
        morph { damage 30 }
    }";
    let err = parse_ability_file(src).expect_err("morph without `into` must error");
    assert!(
        err.to_string().contains("into"),
        "diagnostic must mention `into`; got: {err}"
    );
}

#[test]
fn duplicate_morph_block_is_error() {
    let src = "ability X {
        target: self cooldown: 8s
        morph { damage 30 } into A
        morph { damage 40 } into B
    }";
    let err = parse_ability_file(src).expect_err("duplicate morph must error");
    assert!(
        err.to_string().contains("more than one `morph`"),
        "diagnostic must name the duplicate; got: {err}"
    );
}

// ---------------------------------------------------------------------------
// 4. LoL corpus parse-rate sanity check
// ---------------------------------------------------------------------------

#[test]
fn lol_corpus_parse_rate_post_wave_1_4() {
    // Tracking signal — emits the parse rate to stderr but DOES NOT
    // assert on a target rate. Wave 1.5 baseline: 6/172. Brief's
    // expected post-1.4 range: ~28-138/172 (110 files unblocked by
    // deliver, 28 by recast, with overlap and `buff` hint /
    // `break_on_damage` flag still failing). Run with --nocapture to
    // see the number.
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("dataset")
        .join("abilities")
        .join("lol_heroes");
    if !dir.is_dir() {
        eprintln!("dataset/abilities/lol_heroes not found at {}", dir.display());
        return;
    }
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "ability"))
        .collect();
    files.sort();
    let mut ok = 0usize;
    let mut errs = std::collections::BTreeMap::<String, usize>::new();
    for path in &files {
        let src = std::fs::read_to_string(path).unwrap();
        match parse_ability_file(&src) {
            Ok(_) => ok += 1,
            Err(e) => {
                let line = e.to_string().lines().next().unwrap_or("").to_string();
                *errs.entry(line).or_insert(0) += 1;
            }
        }
    }
    eprintln!(
        "---- LoL corpus parse rate post-Wave-1.4 ----\n  parsed: {}/{}",
        ok,
        files.len()
    );
    let mut top: Vec<_> = errs.iter().collect();
    top.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (e, c) in top.iter().take(5) {
        eprintln!("  {} : {}", c, e);
    }
}
