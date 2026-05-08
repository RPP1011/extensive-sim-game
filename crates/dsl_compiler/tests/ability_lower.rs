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
    // #142: `heal` hint maps to `AbilityHint::Heal` (was `Defense`
    // before the variant landed). Distinct scoring bucket from defense.
    assert_eq!(prog.hint, Some(AbilityHint::Heal));
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
    let outcome = lower_ability_file(&file).expect("lowering");
    assert!(outcome.skipped.is_empty(), "no top-level skips on Wave 1 corpus");
    assert_eq!(outcome.programs.len(), 1);
    let p = &outcome.programs[0];
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
    let outcome = lower_ability_file(&file).expect("lowering");
    assert!(outcome.skipped.is_empty(), "no top-level skips on Wave 1 corpus");
    assert_eq!(outcome.programs.len(), 1);
    let p = &outcome.programs[0];
    assert_eq!(p.gate.cooldown_ticks, 40);
    assert!(!p.gate.hostile_only);
    assert!(matches!(p.effects[0], EffectOp::Shield { .. }));
}

#[test]
fn lower_wave1_corpus_mend() {
    let path = corpus_path("Mend.ability");
    let src = std::fs::read_to_string(&path).expect("Mend.ability missing");
    let file = parse_ability_file(&src).expect("parser");
    let outcome = lower_ability_file(&file).expect("lowering");
    assert!(outcome.skipped.is_empty(), "no top-level skips on Wave 1 corpus");
    assert_eq!(outcome.programs.len(), 1);
    let p = &outcome.programs[0];
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
// Wave 2 piece 2 — movement verbs (dash / blink / knockback / pull).
// All four mirror `damage`'s shape: one `<distance:f32>` positional arg
// → `EffectOp::* { distance }`. The runtime apply handlers (compute
// facing / away / toward vectors and update `hot_pos`) land in a
// follow-up Wave 2 piece.
// ---------------------------------------------------------------------------

#[test]
fn lowers_dash() {
    let src = "ability Lunge { target: self cooldown: 1s dash 4.5 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::Dash { distance } => assert!((distance - 4.5).abs() < 1e-6, "dash distance"),
        ref other => panic!("expected Dash; got {other:?}"),
    }
}

// Lift A — `travel_to <x> <y> for <duration>` multi-tick travel.
// Self-cast: the caster initiates a multi-tick walk to the destination.
// q8 packing: 5.0 → 1280 (= 5 * 256). 5s @ 100ms = 50 ticks.
#[test]
fn lowers_travel_to() {
    let src = "ability Walk { target: self cooldown: 1s travel_to 5 5 for 5s }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::TravelTo { dest_x_q8, dest_y_q8, eta_ticks } => {
            assert_eq!(dest_x_q8, 1280, "5.0 packed q8");
            assert_eq!(dest_y_q8, 1280, "5.0 packed q8");
            assert_eq!(eta_ticks, 50, "5s @ 100ms = 50 ticks");
        }
        ref other => panic!("expected TravelTo; got {other:?}"),
    }
}

// Lift A — travel_to with optional Z arg accepted (today ignored in
// the EffectOp payload — 2D-flat sims dominate; the SoA cell carries Z
// independently via the consumer rule).
#[test]
fn lowers_travel_to_with_optional_z() {
    let src = "ability WalkZ { target: self cooldown: 1s travel_to 5 5 0 for 5s }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    assert!(matches!(
        prog.effects[0],
        EffectOp::TravelTo { dest_x_q8: 1280, dest_y_q8: 1280, eta_ticks: 50 }
    ));
}

// Lift A — travel_to without `for` modifier is a hard error. Travel
// without an ETA is meaningless — that's just `blink` (instant teleport).
#[test]
fn travel_to_without_duration_errors() {
    let src = "ability NoEta { target: self cooldown: 1s travel_to 5 5 }";
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_decl(&file.abilities[0])
        .expect_err("missing for-duration must error");
    // Should surface as EffectArgMismatch (verb expects 2 args + for).
    match err {
        LowerError::EffectArgMismatch { verb, .. } => assert_eq!(verb, "travel_to"),
        other => panic!("expected EffectArgMismatch on travel_to; got {other:?}"),
    }
}

// Lift A — q8 round-trip: 0.5 → 128, -1.0 → -256, etc. Pin the packing
// rule so a future change to the q8 convention surfaces here.
#[test]
fn lowers_travel_to_q8_packing_round_trips() {
    let src = "ability Walk { target: self cooldown: 1s travel_to 0.5 -1.0 for 1s }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    match prog.effects[0] {
        EffectOp::TravelTo { dest_x_q8, dest_y_q8, eta_ticks } => {
            assert_eq!(dest_x_q8, 128, "0.5 * 256 = 128");
            assert_eq!(dest_y_q8, -256, "-1.0 * 256 = -256");
            assert_eq!(eta_ticks, 10);
        }
        ref other => panic!("expected TravelTo; got {other:?}"),
    }
}

#[test]
fn lowers_blink() {
    let src = "ability Flash { target: enemy range: 6 cooldown: 1s blink 6 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::Blink { distance } => assert!((distance - 6.0).abs() < 1e-6, "blink distance"),
        ref other => panic!("expected Blink; got {other:?}"),
    }
}

#[test]
fn lowers_knockback() {
    let src = "ability Shove { target: enemy range: 2 cooldown: 1s knockback 3 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::Knockback { distance } => assert!((distance - 3.0).abs() < 1e-6, "knockback distance"),
        ref other => panic!("expected Knockback; got {other:?}"),
    }
}

#[test]
fn lowers_pull() {
    let src = "ability Yank { target: enemy range: 5 cooldown: 1s pull 2.5 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::Pull { distance } => assert!((distance - 2.5).abs() < 1e-6, "pull distance"),
        ref other => panic!("expected Pull; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Wave 2 piece 3 — advanced verbs (execute / self_damage).
// Both mirror `damage`'s shape: one `<f32>` positional arg →
// `EffectOp::* { ... }`. NEITHER adds new SoA fields; per-fixture apply
// handlers (Execute → emit Defeated when hp < threshold; SelfDamage →
// emit Damaged{source=target=caster, amount}) land in Wave 2 piece N.
// ---------------------------------------------------------------------------

#[test]
fn lowers_execute() {
    let src = "ability Finisher { target: enemy range: 4 cooldown: 1s execute 25 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::Execute { hp_threshold } => {
            assert!((hp_threshold - 25.0).abs() < 1e-6, "execute hp_threshold")
        }
        ref other => panic!("expected Execute; got {other:?}"),
    }
}

#[test]
fn lowers_self_damage() {
    let src = "ability BloodPact { target: self cooldown: 1s self_damage 7.5 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::SelfDamage { amount } => {
            assert!((amount - 7.5).abs() < 1e-6, "self_damage amount")
        }
        ref other => panic!("expected SelfDamage; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Wave 2 piece 4 — buff verbs (lifesteal / damage_modify).
// Both mirror `slow`'s shape: `<f32 magnitude> <duration>` →
// `EffectOp::* { duration_ticks, <field>_q8 }`. q8 packing is
// `magnitude * 256` clamped to `i16::MIN..=i16::MAX`. Apply-handler
// stacking is documented on the SoA fields (single per-agent slot,
// max-with-duration-tiebreak); the apply handlers themselves land in a
// follow-up Wave 2 piece.
// ---------------------------------------------------------------------------

#[test]
fn lowers_lifesteal() {
    // 0.5 * 256 = 128. 4s @ 100ms/tick = 40 ticks.
    let src = "ability VampStrike { target: self cooldown: 1s lifesteal 0.5 4s }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::LifeSteal { duration_ticks, fraction_q8 } => {
            assert_eq!(duration_ticks, 40, "4s @ 100ms = 40 ticks");
            assert_eq!(fraction_q8, 128, "0.5 * 256 = 128");
        }
        ref other => panic!("expected LifeSteal; got {other:?}"),
    }
}

#[test]
fn lowers_lifesteal_full_fraction_rounds_to_256() {
    // Edge: 1.0 → 256 (canonical "heal-for-full-damage" buff).
    let src = "ability FullVamp { target: self cooldown: 1s lifesteal 1.0 2s }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::LifeSteal { duration_ticks, fraction_q8 } => {
            assert_eq!(duration_ticks, 20, "2s @ 100ms = 20 ticks");
            assert_eq!(fraction_q8, 256, "1.0 * 256 = 256");
        }
        ref other => panic!("expected LifeSteal; got {other:?}"),
    }
}

#[test]
fn lowers_damage_modify() {
    // 1.5 * 256 = 384. 3s @ 100ms/tick = 30 ticks.
    let src = "ability Vulnerable { target: enemy range: 5 cooldown: 1s damage_modify 1.5 3s }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::DamageModify { duration_ticks, multiplier_q8 } => {
            assert_eq!(duration_ticks, 30, "3s @ 100ms = 30 ticks");
            assert_eq!(multiplier_q8, 384, "1.5 * 256 = 384");
        }
        ref other => panic!("expected DamageModify; got {other:?}"),
    }
}

#[test]
fn lowers_damage_modify_half_multiplier() {
    // 0.5 * 256 = 128 ("take half damage" — canonical defensive buff).
    let src = "ability Steeled { target: self cooldown: 1s damage_modify 0.5 1500ms }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::DamageModify { duration_ticks, multiplier_q8 } => {
            assert_eq!(duration_ticks, 15, "1500ms @ 100ms = 15 ticks");
            assert_eq!(multiplier_q8, 128, "0.5 * 256 = 128");
        }
        ref other => panic!("expected DamageModify; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

#[test]
fn target_ground_lowers_to_target_mode_kind_ground() {
    use engine::ability::program::TargetModeKind;
    // Wave 2 follow-on (#127): all eight target modes lower into
    // TargetModeKind. Apply dispatch for position-targeted modes
    // wires later via registry-driven infrastructure (#125).
    let src = "ability Boulder { target: ground cooldown: 5s damage 10 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0])
        .expect("ground target mode must lower (was: TargetModeReserved)");
    assert_eq!(prog.target_mode, TargetModeKind::Ground);
    // hostile_only defaults to false for non-pair-targeted modes —
    // there's no "the other guy" to friendly-fire-check.
    assert!(!prog.gate.hostile_only);
}

// ---------------------------------------------------------------------------
// Wave 3 phase 3.5 — Theory-of-Mind `observe` verb. Single positional
// `<id:u8>` arg (agent slot id) → `EffectOp::Observe { target_observer }`.
// Mirrors `dash`'s single-positional shape; engine op kind = 33.
// ---------------------------------------------------------------------------

#[test]
fn lowers_observe() {
    let src = "ability Spy { target: self cooldown: 1s observe 7 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::Observe { target_observer } => {
            assert_eq!(target_observer, 7, "observe id payload");
        }
        ref other => panic!("expected Observe; got {other:?}"),
    }
}

#[test]
fn lower_erase_belief_with_field_bitset() {
    // Wave 3 ToM Phase 4 — `erase_belief <subject_idx> <fields>`. The
    // .ability parser doesn't lex hex (Wave 1.0), so the bitset arrives
    // as a plain decimal: `7` == 0b00000111 == pos|type|tick.
    let src = "ability Wipe { target: enemy range: 5.0 cooldown: 1s erase_belief 5 7 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::EraseBelief { subject_idx, fields } => {
            assert_eq!(subject_idx, 5);
            assert_eq!(fields, 0x07, "0b00000111 = pos|type|tick");
        }
        ref other => panic!("expected EraseBelief; got {other:?}"),
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
    // MAX_EFFECTS_PER_PROGRAM bumped 4 → 6 (#131-followup) to fit
    // LoL hero ultimates (5–6 effects each). Test now needs 7 to
    // overrun the budget.
    let src = r#"
ability TooMany {
    target: enemy
    cooldown: 1s
    damage 1
    damage 1
    damage 1
    damage 1
    damage 1
    damage 1
    damage 1
}
"#;
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_decl(&file.abilities[0]).expect_err("must reject 7 effects");
    match err {
        LowerError::BudgetExceeded { count, max, ability, .. } => {
            assert_eq!(count, 7);
            assert_eq!(max, 6);
            assert_eq!(ability, "TooMany");
        }
        other => panic!("expected BudgetExceeded; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Wave 3 ToM Phase 4 — `disguise <fake_type> for <duration>` (spy_network).
// ---------------------------------------------------------------------------

#[test]
fn lower_disguise_for_duration() {
    let src = "ability Spy { target: self cooldown: 20s disguise 3 for 10s }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("disguise must lower");
    assert_eq!(prog.effects.len(), 1);
    match &prog.effects[0] {
        EffectOp::Disguise { fake_type, duration_ticks } => {
            assert_eq!(*fake_type, 3);
            assert_eq!(*duration_ticks, 100);
        }
        other => panic!("expected Disguise; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Wave 3 ToM Phase 3.5 — `reveal <subject_idx>` belief broadcast.
// ---------------------------------------------------------------------------

#[test]
fn lowers_reveal() {
    let src = "ability Reveal { target: self cooldown: 1s reveal 5 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lowering");
    assert_eq!(prog.effects.len(), 1);
    match prog.effects[0] {
        EffectOp::Reveal { subject_idx } => assert_eq!(subject_idx, 5),
        ref other => panic!("expected Reveal; got {other:?}"),
    }
}
