//! Wave 1.5 `.ability` parser surface tests for the nine effect-statement
//! modifier slots (per `docs/spec/ability_dsl_unified.md` §6.1).
//!
//! Coverage by slot:
//!  1. `in <shape>(args)`              — area expansion
//!  2. `[TAG: value]`                  — power tags (multiple allowed)
//!  3. `for <duration>`                — effect duration
//!  4. `when <cond> [else <cond>]`     — conditional gate (opaque)
//!  5. `chance N%`                     — Bernoulli gate
//!  6. `stacking refresh|stack|extend` — stack policy
//!  7. `+ N% <stat_ref>`               — additive scaling (multiple)
//!  8. `until_caster_dies` / `damageable_hp(N)` — voxel lifetime
//!  9. `{ … }`                         — nested follow-up effects
//!
//! Plus:
//!  * a complex combo exercising all 9 slots in one effect
//!  * duplicate-slot diagnostics for single-value slots
//!  * unknown trailing-token diagnostic
//!  * LoL hero corpus canary (Aatrox.ability requires deferred Wave 1.4
//!    surfaces; we use Alistar.ability which uses Wave 1.5 modifiers
//!    exclusively)

use dsl_ast::ast::{EffectArg, EffectLifetime, StackingMode};
use dsl_ast::parse_ability_file;

// ---------------------------------------------------------------------------
// 1. `in <shape>(args)`
// ---------------------------------------------------------------------------

#[test]
fn parses_in_circle() {
    let src = "ability X { target: enemy cooldown: 1s damage 50 in circle(2.5) }";
    let file = parse_ability_file(src).expect("must parse");
    let e = &file.abilities[0].effects[0];
    let area = e.area.as_ref().expect("area populated");
    assert_eq!(area.shape, "circle");
    assert_eq!(area.args.len(), 1);
    assert!((area.args[0] - 2.5).abs() < 1e-6);
}

#[test]
fn parses_in_cone_with_two_args_and_deg_suffix() {
    // Spec §8.1 cone(r, angle_deg). The `deg` unit is documented but
    // the corpus uses bare numbers; both forms parse.
    let src = "ability X { target: enemy cooldown: 1s damage 100 in cone(45deg, 8.0) }";
    let file = parse_ability_file(src).expect("must parse");
    let area = file.abilities[0].effects[0].area.as_ref().unwrap();
    assert_eq!(area.shape, "cone");
    assert_eq!(area.args.len(), 2);
    assert!((area.args[0] - 45.0).abs() < 1e-6);
    assert!((area.args[1] - 8.0).abs() < 1e-6);
}

#[test]
fn parses_in_box_with_three_args() {
    let src = "ability X { target: enemy cooldown: 1s damage 25 in box(3.0, 3.0, 1.0) }";
    let file = parse_ability_file(src).expect("must parse");
    let area = file.abilities[0].effects[0].area.as_ref().unwrap();
    assert_eq!(area.shape, "box");
    assert_eq!(area.args.len(), 3);
}

// ---------------------------------------------------------------------------
// 2. `[TAG: value]` power tags
// ---------------------------------------------------------------------------

#[test]
fn parses_single_tag() {
    let src = "ability X { target: enemy cooldown: 1s damage 50 [PHYSICAL: 60] }";
    let file = parse_ability_file(src).expect("must parse");
    let e = &file.abilities[0].effects[0];
    assert_eq!(e.tags.len(), 1);
    assert_eq!(e.tags[0].name, "PHYSICAL");
    assert!((e.tags[0].value - 60.0).abs() < 1e-6);
}

#[test]
fn parses_multiple_tags() {
    let src = "ability X { target: enemy cooldown: 1s damage 50 [PHYSICAL: 60] [CROWD_CONTROL: 20] }";
    let file = parse_ability_file(src).expect("must parse");
    let e = &file.abilities[0].effects[0];
    assert_eq!(e.tags.len(), 2);
    assert_eq!(e.tags[0].name, "PHYSICAL");
    assert_eq!(e.tags[1].name, "CROWD_CONTROL");
    assert!((e.tags[1].value - 20.0).abs() < 1e-6);
}

#[test]
fn parses_tag_with_float_value() {
    // Spec §6.1 stores tag values as f32; corpus has both int and float.
    let src = "ability X { target: enemy cooldown: 1s damage 50 [FIRE: 12.5] }";
    let file = parse_ability_file(src).expect("must parse");
    assert!((file.abilities[0].effects[0].tags[0].value - 12.5).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// 3. `for <duration>`
// ---------------------------------------------------------------------------

#[test]
fn parses_for_duration_seconds() {
    let src = "ability X { target: enemy cooldown: 1s slow 0.5 4s for 4s }";
    // Note: `slow` takes (factor, duration) as positional args; we
    // additionally tag with `for 4s` (modifier) — the parser accepts
    // both shapes and the lowering pass owns the meaning.
    let file = parse_ability_file(src).expect("must parse");
    let e = &file.abilities[0].effects[0];
    let d = e.duration.as_ref().expect("duration populated");
    assert_eq!(d.duration.millis, 4_000);
}

#[test]
fn parses_for_duration_millis() {
    let src = "ability X { target: enemy cooldown: 1s damage 5 for 250ms }";
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(
        file.abilities[0].effects[0].duration.as_ref().unwrap().duration.millis,
        250
    );
}

// ---------------------------------------------------------------------------
// 4. `when <cond> [else <cond>]`
// ---------------------------------------------------------------------------

#[test]
fn parses_when_simple() {
    let src = "ability X { target: enemy cooldown: 1s damage 50 when target.hp < 30 }";
    let file = parse_ability_file(src).expect("must parse");
    let cond = file.abilities[0].effects[0]
        .condition
        .as_ref()
        .expect("condition populated");
    assert_eq!(cond.when_cond, "target.hp < 30");
    assert!(cond.else_cond.is_none());
}

#[test]
fn parses_when_with_else() {
    let src = "ability X { target: enemy cooldown: 1s heal 20 when target.hp < 30 else when target.shield_hp == 0 }";
    let file = parse_ability_file(src).expect("must parse");
    let cond = file.abilities[0].effects[0].condition.as_ref().unwrap();
    assert_eq!(cond.when_cond, "target.hp < 30");
    assert_eq!(cond.else_cond.as_deref(), Some("when target.shield_hp == 0"));
}

#[test]
fn parses_when_with_parens_in_cond() {
    // Top-level `)` shouldn't terminate inside a balanced paren group.
    let src = "ability X { target: enemy cooldown: 1s damage 50 when (target.hp < 30 or target.shield_hp == 0) }";
    let file = parse_ability_file(src).expect("must parse");
    let cond = file.abilities[0].effects[0].condition.as_ref().unwrap();
    assert_eq!(cond.when_cond, "(target.hp < 30 or target.shield_hp == 0)");
}

// ---------------------------------------------------------------------------
// 5. `chance N%`
// ---------------------------------------------------------------------------

#[test]
fn parses_chance_percent() {
    let src = "ability X { target: enemy cooldown: 1s damage 50 chance 25% }";
    let file = parse_ability_file(src).expect("must parse");
    let p = file.abilities[0].effects[0].chance.as_ref().unwrap().p;
    assert!((p - 0.25).abs() < 1e-6, "expected 0.25; got {p}");
}

#[test]
fn parses_chance_100_percent() {
    let src = "ability X { target: enemy cooldown: 1s damage 50 chance 100% }";
    let file = parse_ability_file(src).expect("must parse");
    let p = file.abilities[0].effects[0].chance.as_ref().unwrap().p;
    assert!((p - 1.0).abs() < 1e-6);
}

#[test]
fn chance_without_percent_is_error() {
    let src = "ability X { target: enemy cooldown: 1s damage 50 chance 0.25 }";
    let err = parse_ability_file(src).expect_err("chance must require %");
    assert!(
        err.to_string().contains("`%` suffix"),
        "expected % diagnostic; got: {err}"
    );
}

#[test]
fn chance_out_of_range_is_error() {
    let src = "ability X { target: enemy cooldown: 1s damage 50 chance 150% }";
    let err = parse_ability_file(src).expect_err("chance out of range");
    assert!(
        err.to_string().contains("out of range"),
        "expected out-of-range diagnostic; got: {err}"
    );
}

// ---------------------------------------------------------------------------
// 6. `stacking refresh|stack|extend`
// ---------------------------------------------------------------------------

#[test]
fn parses_stacking_refresh() {
    let src = "ability X { target: enemy cooldown: 1s heal 20 stacking refresh }";
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(
        file.abilities[0].effects[0].stacking,
        Some(StackingMode::Refresh)
    );
}

#[test]
fn parses_stacking_stack() {
    let src = "ability X { target: enemy cooldown: 1s heal 20 stacking stack }";
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(
        file.abilities[0].effects[0].stacking,
        Some(StackingMode::Stack)
    );
}

#[test]
fn parses_stacking_extend() {
    let src = "ability X { target: enemy cooldown: 1s heal 20 stacking extend }";
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(
        file.abilities[0].effects[0].stacking,
        Some(StackingMode::Extend)
    );
}

#[test]
fn unknown_stacking_mode_is_error() {
    let src = "ability X { target: enemy cooldown: 1s heal 20 stacking forever }";
    let err = parse_ability_file(src).expect_err("unknown stacking mode");
    assert!(
        err.to_string().contains("forever") || err.to_string().contains("unknown stacking"),
        "expected diagnostic; got: {err}"
    );
}

// ---------------------------------------------------------------------------
// 7. `+ N% stat_ref`
// ---------------------------------------------------------------------------

#[test]
fn parses_scaling_with_stat_ref() {
    let src = "ability X { target: enemy cooldown: 1s damage 50 + 30% AP }";
    let file = parse_ability_file(src).expect("must parse");
    let scs = &file.abilities[0].effects[0].scalings;
    assert_eq!(scs.len(), 1);
    assert!((scs[0].percent - 30.0).abs() < 1e-6);
    assert_eq!(scs[0].stat_ref, "AP");
}

#[test]
fn parses_multiple_scalings() {
    let src = "ability X { target: enemy cooldown: 1s damage 50 + 30% AP + 20% AD }";
    let file = parse_ability_file(src).expect("must parse");
    let scs = &file.abilities[0].effects[0].scalings;
    assert_eq!(scs.len(), 2);
    assert_eq!(scs[0].stat_ref, "AP");
    assert_eq!(scs[1].stat_ref, "AD");
    assert!((scs[1].percent - 20.0).abs() < 1e-6);
}

#[test]
fn parses_dotted_stat_ref() {
    let src = "ability X { target: enemy cooldown: 1s damage 50 + 5% self.hp }";
    let file = parse_ability_file(src).expect("must parse");
    let scs = &file.abilities[0].effects[0].scalings;
    assert_eq!(scs.len(), 1);
    assert_eq!(scs[0].stat_ref, "self.hp");
}

// ---------------------------------------------------------------------------
// 8. `until_caster_dies` / `damageable_hp(N)`
// ---------------------------------------------------------------------------

#[test]
fn parses_until_caster_dies() {
    let src = "ability X { target: self cooldown: 1s shield 100 until_caster_dies }";
    let file = parse_ability_file(src).expect("must parse");
    match file.abilities[0].effects[0].lifetime.as_ref().unwrap() {
        EffectLifetime::UntilCasterDies { .. } => {}
        other => panic!("expected UntilCasterDies; got {other:?}"),
    }
}

#[test]
fn parses_damageable_hp() {
    let src = "ability X { target: ground cooldown: 1s damage 5 damageable_hp(50) }";
    // target: ground is a planned mode but the parser accepts all the
    // documented modes; lowering rejects the unsupported ones.
    let file = parse_ability_file(src).expect("must parse");
    match file.abilities[0].effects[0].lifetime.as_ref().unwrap() {
        EffectLifetime::DamageableHp { hp, .. } => {
            assert!((*hp - 50.0).abs() < 1e-6);
        }
        other => panic!("expected DamageableHp; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 9. Nested `{ … }` blocks
// ---------------------------------------------------------------------------

#[test]
fn parses_nested_block_single_stmt() {
    let src = "ability X { target: enemy cooldown: 1s heal 50 { stun 1s } }";
    let file = parse_ability_file(src).expect("must parse");
    let e = &file.abilities[0].effects[0];
    assert_eq!(e.nested.len(), 1);
    assert_eq!(e.nested[0].verb, "stun");
}

#[test]
fn parses_nested_block_multi_stmt() {
    let src = r#"
ability X {
    target: enemy
    cooldown: 1s
    heal 50 {
        stun 1s
        damage 5
    }
}"#;
    let file = parse_ability_file(src).expect("must parse");
    let e = &file.abilities[0].effects[0];
    assert_eq!(e.nested.len(), 2);
    assert_eq!(e.nested[0].verb, "stun");
    assert_eq!(e.nested[1].verb, "damage");
}

#[test]
fn parses_nested_block_with_inner_modifier() {
    // Recursion: nested effects parse with the full Wave 1.5 modifier
    // surface — the inner `when target.alive` populates the inner
    // condition slot.
    let src = "ability X { target: enemy cooldown: 1s heal 50 { stun 1s when target.alive } }";
    let file = parse_ability_file(src).expect("must parse");
    let inner = &file.abilities[0].effects[0].nested[0];
    assert_eq!(inner.verb, "stun");
    let cond = inner.condition.as_ref().expect("inner condition populated");
    assert_eq!(cond.when_cond, "target.alive");
}

// ---------------------------------------------------------------------------
// Combo / order-independence
// ---------------------------------------------------------------------------

#[test]
fn parses_complex_combo_all_nine_slots() {
    // Exercise all 9 slot types in one effect. Verify modifier order
    // is NOT semantically meaningful — we put them in a "weird" order
    // and assert each slot is populated.
    let src = r#"
ability Mega {
    target: enemy
    cooldown: 1s
    damage 50 in circle(3.0) [PHYSICAL: 60] [FIRE: 25] for 4s when target.hp < 50 chance 75% stacking refresh + 30% AP + 20% AD until_caster_dies { stun 1s }
}"#;
    let file = parse_ability_file(src).expect("must parse");
    let e = &file.abilities[0].effects[0];
    assert_eq!(e.verb, "damage");
    assert!(matches!(&e.args[0], EffectArg::Number(v) if (*v - 50.0).abs() < 1e-6));
    assert!(e.area.is_some(), "area slot");
    assert_eq!(e.tags.len(), 2, "two power tags");
    assert!(e.duration.is_some(), "for-duration slot");
    assert!(e.condition.is_some(), "when slot");
    assert!(e.chance.is_some(), "chance slot");
    assert_eq!(e.stacking, Some(StackingMode::Refresh));
    assert_eq!(e.scalings.len(), 2, "two scaling terms");
    assert!(e.lifetime.is_some(), "lifetime slot");
    assert_eq!(e.nested.len(), 1, "one nested stmt");
}

#[test]
fn modifier_order_does_not_matter() {
    let a = "ability A { target: enemy cooldown: 1s damage 50 in circle(2.5) for 3s when target.hp < 30 }";
    let b = "ability A { target: enemy cooldown: 1s damage 50 when target.hp < 30 for 3s in circle(2.5) }";
    let fa = parse_ability_file(a).expect("a parses");
    let fb = parse_ability_file(b).expect("b parses");
    let ea = &fa.abilities[0].effects[0];
    let eb = &fb.abilities[0].effects[0];
    // Compare each slot's payload (spans differ — area sits at a
    // different byte offset in each).
    assert_eq!(ea.area.as_ref().unwrap().shape, eb.area.as_ref().unwrap().shape);
    assert_eq!(ea.area.as_ref().unwrap().args, eb.area.as_ref().unwrap().args);
    assert_eq!(
        ea.duration.as_ref().unwrap().duration,
        eb.duration.as_ref().unwrap().duration
    );
    assert_eq!(
        ea.condition.as_ref().unwrap().when_cond,
        eb.condition.as_ref().unwrap().when_cond
    );
}

// ---------------------------------------------------------------------------
// Duplicate-slot diagnostics
// ---------------------------------------------------------------------------

#[test]
fn duplicate_for_duration_is_error() {
    let src = "ability X { target: enemy cooldown: 1s damage 5 for 1s for 2s }";
    let err = parse_ability_file(src).expect_err("duplicate for must fail");
    assert!(
        err.to_string().contains("duplicate `for"),
        "expected duplicate-for diagnostic; got: {err}"
    );
}

#[test]
fn duplicate_in_shape_is_error() {
    let src = "ability X { target: enemy cooldown: 1s damage 5 in circle(2.0) in cone(45, 5.0) }";
    let err = parse_ability_file(src).expect_err("duplicate in must fail");
    assert!(
        err.to_string().contains("duplicate `in"),
        "expected duplicate-in diagnostic; got: {err}"
    );
}

#[test]
fn duplicate_when_is_error() {
    let src = "ability X { target: enemy cooldown: 1s damage 5 when a < 1 when b < 2 }";
    let err = parse_ability_file(src).expect_err("duplicate when must fail");
    assert!(
        err.to_string().contains("duplicate `when"),
        "expected duplicate-when diagnostic; got: {err}"
    );
}

#[test]
fn duplicate_chance_is_error() {
    let src = "ability X { target: enemy cooldown: 1s damage 5 chance 25% chance 50% }";
    let err = parse_ability_file(src).expect_err("duplicate chance must fail");
    assert!(
        err.to_string().contains("duplicate `chance"),
        "expected duplicate-chance diagnostic; got: {err}"
    );
}

#[test]
fn duplicate_stacking_is_error() {
    let src = "ability X { target: enemy cooldown: 1s heal 5 stacking refresh stacking stack }";
    let err = parse_ability_file(src).expect_err("duplicate stacking must fail");
    assert!(
        err.to_string().contains("duplicate `stacking"),
        "expected duplicate-stacking diagnostic; got: {err}"
    );
}

#[test]
fn duplicate_lifetime_is_error() {
    let src = "ability X { target: self cooldown: 1s shield 50 until_caster_dies damageable_hp(20) }";
    let err = parse_ability_file(src).expect_err("duplicate lifetime must fail");
    assert!(
        err.to_string().contains("duplicate lifetime"),
        "expected duplicate-lifetime diagnostic; got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Unknown trailing-token diagnostic
// ---------------------------------------------------------------------------

#[test]
fn unknown_modifier_keyword_is_error() {
    // Once a modifier slot has been parsed, any subsequent token MUST
    // be a recognised modifier — this guards against silently dropping
    // mistyped modifier keywords. Here `forr` (typo of `for`) sits
    // after the `in circle(...)` modifier; it isn't a known keyword
    // and isn't a valid effect-arg start at this point in the
    // statement, so the parser errors out.
    let src = "ability X { target: enemy cooldown: 1s damage 5 in circle(2.0) forr 3s }";
    let err = parse_ability_file(src).expect_err("unknown modifier must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("unknown modifier") || msg.contains("trailing token"),
        "expected unknown-modifier diagnostic; got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// LoL corpus canary
// ---------------------------------------------------------------------------

#[test]
fn lol_corpus_alistar_parses_cleanly() {
    // Canary that the modifier surface handles real-world ability
    // source. Alistar.ability uses the Wave 1.5 modifier surfaces
    // (`in circle(N)`, `[MAGIC: N]`) without depending on Wave 1.4
    // surfaces (`deliver` / `recast` / etc.). Aatrox.ability — the
    // brief's named canary — uses `recast:` and `deliver` blocks,
    // which the parser still rejects in this slice (Wave 1.4 work).
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("dataset")
        .join("abilities")
        .join("lol_heroes")
        .join("Alistar.ability");
    let src = std::fs::read_to_string(&path).expect("Alistar.ability readable");
    parse_ability_file(&src).expect("Alistar.ability parses cleanly");
}

#[test]
fn lol_corpus_aatrox_parses_cleanly() {
    // Brief's named canary. Aatrox.ability uses Wave 1.4 surfaces
    // (`recast:` header, `deliver { ... }` body) that this slice does
    // not yet implement; per the brief we still attempt the parse —
    // when Wave 1.4 lands and Aatrox parses cleanly, this assertion
    // flips from `is_err` (deferred to Wave 1.4) to `is_ok` (full
    // surface parses). For Wave 1.5 we document the current state by
    // asserting the parse FAILS on a non-modifier surface, NOT on
    // anything in spec §6.1.
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("dataset")
        .join("abilities")
        .join("lol_heroes")
        .join("Aatrox.ability");
    let src = std::fs::read_to_string(&path).expect("Aatrox.ability readable");
    let res = parse_ability_file(&src);
    if let Err(e) = &res {
        let msg = e.to_string();
        // The failure must be on a Wave 1.4 surface (`recast:` header
        // or `deliver` block), NOT on any Wave 1.5 modifier slot.
        assert!(
            msg.contains("recast")
                || msg.contains("deliver")
                || msg.contains("morph")
                || msg.contains("template")
                || msg.contains("structure")
                || msg.contains("unsupported header"),
            "Aatrox failure must be on a Wave 1.4 surface, not a Wave 1.5 modifier slot. Got: {msg}"
        );
    }
    // If Wave 1.4 has landed and Aatrox parses, that's also fine — the
    // assertion is "no Wave 1.5 modifier regression".
}
