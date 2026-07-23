//! Webband port S5-prep — the 10-spec ability subset from
//! `F:\MB\src\battle\abilities\catalog.ts`, translated into
//! `dataset/abilities/webband/*.ability` (see that directory's README
//! for the round→seconds conversion table and the gap list).
//!
//! Walks the webband dataset dir on the `lol_corpus_lowering.rs`
//! harness pattern and asserts:
//!   1. every file is LF-only (the parser rejects CRLF; the corpus
//!      rule from the port plan),
//!   2. every file parses,
//!   3. every ability lowers to an `AbilityProgram` without error,
//!   4. per-ability expected IR: op kinds + amounts, delivery shape,
//!      per-effect AoE shapes/args, when-predicates, cooldown ticks —
//!      so a lowering change that silently reshapes the Webband kit
//!      surfaces here, not in the S5 fixture.

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_lower::lower_ability_decl;
use engine::ability::program::{
    AbilityProgram, Delivery, DeliveryHookKind, DeliveryMethodKind, EffectOp, ShapeKind,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn webband_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("dataset")
        .join("abilities")
        .join("webband")
}

fn load_programs() -> BTreeMap<String, AbilityProgram> {
    let dir = webband_dir();
    assert!(dir.is_dir(), "dataset/abilities/webband missing at {}", dir.display());

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "ability"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .ability files in {}", dir.display());

    let mut programs = BTreeMap::new();
    for path in &files {
        let src = std::fs::read_to_string(path).expect("read .ability");
        assert!(
            !src.contains('\r'),
            "{} contains CRLF — .ability files must be LF",
            path.display()
        );
        let file = parse_ability_file(&src)
            .unwrap_or_else(|e| panic!("{} failed to parse: {e}", path.display()));
        for decl in &file.abilities {
            let prog = lower_ability_decl(decl)
                .unwrap_or_else(|e| panic!("{}: ability '{}' failed to lower: {e:?}", path.display(), decl.name));
            let prior = programs.insert(decl.name.clone(), prog);
            assert!(prior.is_none(), "duplicate ability name '{}'", decl.name);
        }
    }
    programs
}

/// Shorthand: the per-effect area shape at effect index `i`, if any.
fn area_of(p: &AbilityProgram, i: usize) -> Option<(ShapeKind, [f32; 4])> {
    p.per_effect_areas
        .get(i)
        .and_then(|o| o.as_ref())
        .map(|s| (s.kind, s.args))
}

#[test]
fn webband_subset_parses_and_lowers() {
    let programs = load_programs();
    let names: Vec<&str> = programs.keys().map(|s| s.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "WebbandBallistaBolt",
            "WebbandCleavingBlow",
            "WebbandCripplingShot",
            "WebbandFieldDressing",
            "WebbandLunge",
            "WebbandPowerStrike",
            "WebbandSecondWind",
            "WebbandShieldWall",
            "WebbandWarlordSweep",
            "WebbandWhirlwind",
        ],
        "the 10-spec Webband subset (sorted) — a rename or a missing spec lands here"
    );
}

#[test]
fn webband_power_strike_ir() {
    // power_strike: damage 46, cd 3 rounds -> 6s -> 60 ticks.
    let p = &load_programs()["WebbandPowerStrike"];
    assert_eq!(p.delivery, Delivery::Instant);
    assert_eq!(p.gate.cooldown_ticks, 60);
    assert!(p.gate.hostile_only);
    assert_eq!(p.effects.as_slice(), &[EffectOp::Damage { amount: 46.0 }]);
    assert!(area_of(p, 0).is_none());
}

#[test]
fn webband_crippling_shot_ir() {
    // crippling_shot: projectile, on_hit damage 28 + slow x0.5 for
    // 2 rounds -> 4s -> 40 ticks. factor_q8 = 0.5 * 256 = 128.
    let p = &load_programs()["WebbandCripplingShot"];
    assert_eq!(p.gate.cooldown_ticks, 40);
    match &p.delivery {
        Delivery::Method { kind, hooks, .. } => {
            assert_eq!(*kind, DeliveryMethodKind::Projectile);
            assert_eq!(hooks.len(), 1, "exactly one on_hit hook");
            assert_eq!(hooks[0].kind, DeliveryHookKind::OnHit);
            assert_eq!(
                hooks[0].effects.as_slice(),
                &[
                    EffectOp::Damage { amount: 28.0 },
                    EffectOp::Slow { duration_ticks: 40, factor_q8: 128 },
                ]
            );
        }
        other => panic!("expected projectile delivery, got {other:?}"),
    }
}

#[test]
fn webband_field_dressing_ir() {
    // field_dressing: heal 40, ally target, cd 2 rounds -> 40 ticks.
    let p = &load_programs()["WebbandFieldDressing"];
    assert_eq!(p.delivery, Delivery::Instant);
    assert_eq!(p.gate.cooldown_ticks, 40);
    assert!(!p.gate.hostile_only, "ally-targeted — never hostile_only");
    assert_eq!(p.effects.as_slice(), &[EffectOp::Heal { amount: 40.0 }]);
}

#[test]
fn webband_shield_wall_ir() {
    // shield_wall: shield 35 for 4 rounds -> 8s -> TimedShield(80 ticks).
    let p = &load_programs()["WebbandShieldWall"];
    assert_eq!(p.gate.cooldown_ticks, 80);
    assert_eq!(
        p.effects.as_slice(),
        &[EffectOp::TimedShield { amount: 35.0, duration_ticks: 80 }]
    );
}

#[test]
fn webband_whirlwind_ir() {
    // whirlwind: damage 30 + knockback 2, both in circle(3.2), cd 80 ticks.
    let p = &load_programs()["WebbandWhirlwind"];
    assert_eq!(p.gate.cooldown_ticks, 80);
    assert_eq!(
        p.effects.as_slice(),
        &[EffectOp::Damage { amount: 30.0 }, EffectOp::Knockback { distance: 2.0 }]
    );
    for i in 0..2 {
        let (kind, args) = area_of(p, i).expect("both effects carry the circle");
        assert_eq!(kind, ShapeKind::Circle);
        assert_eq!(args[0], 3.2);
    }
}

#[test]
fn webband_cleaving_blow_ir() {
    // cleaving_blow: damage 38 in a 100-degree cone, reach 2.8.
    // Engine cone args are [half_angle_deg, range] (apply.rs:955-958).
    let p = &load_programs()["WebbandCleavingBlow"];
    assert_eq!(p.gate.cooldown_ticks, 60);
    assert_eq!(p.effects.as_slice(), &[EffectOp::Damage { amount: 38.0 }]);
    let (kind, args) = area_of(p, 0).expect("cone shape");
    assert_eq!(kind, ShapeKind::Cone);
    assert_eq!(args[0], 50.0, "half-angle: Webband areaDeg 100 / 2");
    assert_eq!(args[1], 2.8, "range: Webband areaR");
}

#[test]
fn webband_lunge_ir() {
    // lunge: dash 7 then damage 34 — order preserved (Webband resolves
    // the dash pre-strike).
    let p = &load_programs()["WebbandLunge"];
    assert_eq!(p.gate.cooldown_ticks, 60);
    assert_eq!(
        p.effects.as_slice(),
        &[EffectOp::Dash { distance: 7.0 }, EffectOp::Damage { amount: 34.0 }]
    );
}

#[test]
fn webband_second_wind_ir() {
    // second_wind: heal 45 gated on self.hp < 50 (the caster_hp_below
    // approximation — see README), shield 25 for 3 rounds -> 60 ticks.
    let p = &load_programs()["WebbandSecondWind"];
    assert_eq!(p.gate.cooldown_ticks, 120);
    assert_eq!(
        p.effects.as_slice(),
        &[
            EffectOp::Heal { amount: 45.0 },
            EffectOp::TimedShield { amount: 25.0, duration_ticks: 60 },
        ]
    );
    let whens = &p.when_per_effect;
    assert_eq!(whens.len(), 2, "populated when slice has one slot per effect");
    let gate = whens[0].as_ref().expect("heal carries the hp gate");
    assert!(
        gate.when_compiled.is_some(),
        "simple atom mirrors onto when_compiled"
    );
    assert!(whens[1].is_none(), "the shield is unconditional");
}

#[test]
fn webband_warlord_sweep_ir() {
    // warlord_sweep: damage 40 + knockback 2 in circle(3.5).
    let p = &load_programs()["WebbandWarlordSweep"];
    assert_eq!(p.gate.cooldown_ticks, 60);
    assert_eq!(
        p.effects.as_slice(),
        &[EffectOp::Damage { amount: 40.0 }, EffectOp::Knockback { distance: 2.0 }]
    );
    for i in 0..2 {
        let (kind, args) = area_of(p, i).expect("circle on both effects");
        assert_eq!(kind, ShapeKind::Circle);
        assert_eq!(args[0], 3.5);
    }
}

#[test]
fn webband_ballista_bolt_ir() {
    // ballista_bolt: damage 32 down an 11m lane, width 1.8 (Webband's
    // fixed 0.9 half-width doubled). Instant on purpose — Webband's
    // projectile delivery was visual-only and hook-stmt in-shapes are
    // dropped by the lowerer today (see README).
    let p = &load_programs()["WebbandBallistaBolt"];
    assert_eq!(p.delivery, Delivery::Instant);
    assert_eq!(p.gate.cooldown_ticks, 60);
    assert_eq!(p.effects.as_slice(), &[EffectOp::Damage { amount: 32.0 }]);
    let (kind, args) = area_of(p, 0).expect("line shape");
    assert_eq!(kind, ShapeKind::Line);
    assert_eq!(args[0], 11.0, "length: Webband areaR");
    assert_eq!(args[1], 1.8, "full width: 2 x Webband's 0.9 half-width");
}
