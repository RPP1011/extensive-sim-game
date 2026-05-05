//! Wave 1.6 lowering tests — `.ability` AST -> `AbilityProgram`.
//!
//! Coverage:
//!  1. Inline minimal `Strike` (target: enemy + range + cooldown + damage).
//!  2. Inline `ShieldUp` (target: self + cooldown + shield).
//!  3. Inline `Mend` (target: self + cooldown + heal).
//!  4. The Wave 1 corpus at
//!     `assets/ability_test/duel_abilities/{Strike,ShieldUp,Mend}.ability`
//!     — all three lower cleanly.
//!  5. Error cases:
//!       - `target: ground` -> `TargetModeReserved`
//!       - unknown verb `whirl 5` -> `UnknownEffectVerb`
//!       - 5 effects -> `BudgetExceeded` (max is 4)

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_lower::{lower_ability_decl, lower_ability_file, LowerError};
use engine::ability::program::{AbilityHint, Area, Delivery, EffectOp};

// ---------------------------------------------------------------------------
// Inline-source happy path
// ---------------------------------------------------------------------------

#[test]
fn lower_minimal_strike_inline() {
    let src = "ability Strike { target: enemy range: 5.0 cooldown: 1s hint: damage damage 15 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");

    assert!(matches!(prog.delivery, Delivery::Instant));
    match prog.area {
        Area::SingleTarget { range } => assert!((range - 5.0).abs() < 1e-6),
    }
    assert_eq!(prog.gate.cooldown_ticks, 10, "1s @ 100ms = 10 ticks");
    assert!(prog.gate.hostile_only, "target: enemy must set hostile_only");
    assert_eq!(prog.hint, Some(AbilityHint::Damage));
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::Damage { amount } => assert!((amount - 15.0).abs() < 1e-6),
        ref other => panic!("expected Damage; got {other:?}"),
    }
}

#[test]
fn lower_minimal_shield_up_inline() {
    let src = "ability ShieldUp { target: self cooldown: 4s hint: defense shield 50 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");

    match prog.area {
        // Self-target with no `range:` header -> default 0.0.
        Area::SingleTarget { range } => assert_eq!(range, 0.0),
    }
    assert_eq!(prog.gate.cooldown_ticks, 40, "4s @ 100ms = 40 ticks");
    assert!(!prog.gate.hostile_only, "target: self must clear hostile_only");
    assert_eq!(prog.hint, Some(AbilityHint::Defense));
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::Shield { amount } => assert!((amount - 50.0).abs() < 1e-6),
        ref other => panic!("expected Shield; got {other:?}"),
    }
}

#[test]
fn lower_minimal_mend_inline() {
    let src = "ability Mend { target: self cooldown: 3s hint: heal heal 25 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");

    assert_eq!(prog.gate.cooldown_ticks, 30);
    assert!(!prog.gate.hostile_only);
    // `heal` hint maps to `Defense` today — see `map_hint` in
    // `ability_lower.rs` for rationale (engine `AbilityHint` lacks a
    // `Heal` variant in this slice).
    assert_eq!(prog.hint, Some(AbilityHint::Defense));
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::Heal { amount } => assert!((amount - 25.0).abs() < 1e-6),
        ref other => panic!("expected Heal; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// On-disk Wave 1 corpus — three real `.ability` files lower without errors.
// ---------------------------------------------------------------------------

fn corpus_path(file: &str) -> std::path::PathBuf {
    // `CARGO_MANIFEST_DIR` is the dsl_compiler crate dir; the corpus lives
    // at the workspace root under `assets/ability_test/duel_abilities/`.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    std::path::PathBuf::from(manifest)
        .join("..")
        .join("..")
        .join("assets")
        .join("ability_test")
        .join("duel_abilities")
        .join(file)
}

#[test]
fn lower_wave1_corpus_strike() {
    let path = corpus_path("Strike.ability");
    let src = std::fs::read_to_string(&path).expect("Strike.ability missing");
    let file = parse_ability_file(&src).expect("parser");
    let progs = lower_ability_file(&file).expect("lowering");
    assert_eq!(progs.len(), 1);
    let p = &progs[0];
    match p.area {
        Area::SingleTarget { range } => assert!((range - 5.0).abs() < 1e-6),
    }
    assert_eq!(p.gate.cooldown_ticks, 10);
    assert!(p.gate.hostile_only);
    assert_eq!(p.hint, Some(AbilityHint::Damage));
    assert!(matches!(p.effects[0], EffectOp::Damage { .. }));
}

#[test]
fn lower_wave1_corpus_shield_up() {
    let path = corpus_path("ShieldUp.ability");
    let src = std::fs::read_to_string(&path).expect("ShieldUp.ability missing");
    let file = parse_ability_file(&src).expect("parser");
    let progs = lower_ability_file(&file).expect("lowering");
    assert_eq!(progs.len(), 1);
    let p = &progs[0];
    assert_eq!(p.gate.cooldown_ticks, 40);
    assert!(!p.gate.hostile_only);
    assert!(matches!(p.effects[0], EffectOp::Shield { .. }));
}

#[test]
fn lower_wave1_corpus_mend() {
    let path = corpus_path("Mend.ability");
    let src = std::fs::read_to_string(&path).expect("Mend.ability missing");
    let file = parse_ability_file(&src).expect("parser");
    let progs = lower_ability_file(&file).expect("lowering");
    assert_eq!(progs.len(), 1);
    let p = &progs[0];
    assert_eq!(p.gate.cooldown_ticks, 30);
    assert!(!p.gate.hostile_only);
    assert!(matches!(p.effects[0], EffectOp::Heal { .. }));
}

// ---------------------------------------------------------------------------
// Wave 2 piece 1 — control verbs (root / silence / fear / taunt).
// All four mirror `stun`'s shape: one `<duration>` arg → `EffectOp::*
// { duration_ticks }` with the same 100ms/tick conversion.
// ---------------------------------------------------------------------------

#[test]
fn lowers_root() {
    let src = "ability Snare { target: enemy range: 5 cooldown: 1s root 2s }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::Root { duration_ticks } => assert_eq!(duration_ticks, 20, "2s @ 100ms = 20 ticks"),
        ref other => panic!("expected Root; got {other:?}"),
    }
}

#[test]
fn lowers_silence() {
    let src = "ability Hush { target: enemy range: 6 cooldown: 1s silence 3s }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::Silence { duration_ticks } => assert_eq!(duration_ticks, 30, "3s @ 100ms = 30 ticks"),
        ref other => panic!("expected Silence; got {other:?}"),
    }
}

#[test]
fn lowers_fear() {
    let src = "ability Howl { target: enemy range: 4 cooldown: 1s fear 1500ms }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::Fear { duration_ticks } => assert_eq!(duration_ticks, 15, "1500ms @ 100ms = 15 ticks"),
        ref other => panic!("expected Fear; got {other:?}"),
    }
}

#[test]
fn lowers_taunt() {
    let src = "ability Provoke { target: enemy range: 3 cooldown: 1s taunt 4s }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::Taunt { duration_ticks } => assert_eq!(duration_ticks, 40, "4s @ 100ms = 40 ticks"),
        ref other => panic!("expected Taunt; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

#[test]
fn target_ground_is_reserved() {
    // Parser accepts `ground`; lowering rejects it as planned/reserved.
    let src = "ability Boulder { target: ground cooldown: 5s damage 10 }";
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_decl(&file.abilities[0]).expect_err("must reject ground");
    match err {
        LowerError::TargetModeReserved { mode, .. } => assert_eq!(mode, "ground"),
        other => panic!("expected TargetModeReserved(ground); got {other:?}"),
    }
}

#[test]
fn unknown_verb_is_rejected() {
    // `whirl` isn't in the Wave 1.6 catalog. (Wave 1.0 parser captures
    // any bare ident as a verb name; lowering is the gate.)
    let src = "ability Mystery { target: enemy cooldown: 1s whirl 5 }";
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_decl(&file.abilities[0]).expect_err("must reject unknown verb");
    match err {
        LowerError::UnknownEffectVerb { verb, .. } => assert_eq!(verb, "whirl"),
        other => panic!("expected UnknownEffectVerb(whirl); got {other:?}"),
    }
}

#[test]
fn budget_exceeded_when_more_than_four_effects() {
    // Five bare effects -> per-program budget breach. Effects must live
    // on their own lines because the parser only ends an effect statement
    // at a newline or `}` (see `parse_effect` in `ability_parser.rs`).
    let src = r#"
ability TooMany {
    target: enemy
    cooldown: 1s
    damage 1
    damage 1
    damage 1
    damage 1
    damage 1
}
"#;
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_decl(&file.abilities[0]).expect_err("must reject 5 effects");
    match err {
        LowerError::BudgetExceeded { count, max, ability, .. } => {
            assert_eq!(count, 5);
            assert_eq!(max, 4);
            assert_eq!(ability, "TooMany");
        }
        other => panic!("expected BudgetExceeded; got {other:?}"),
    }
}
