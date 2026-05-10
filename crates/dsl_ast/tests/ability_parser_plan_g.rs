//! Plan G grammar tests — `cast { … }` block, `effect { … }` block,
//! `interrupts:` syntax with set ops, `cooldown @ phase` qualifier,
//! AbilityProgramStep composition, backwards compat with legacy
//! ability shape.
//!
//! See `docs/superpowers/plans/2026-05-09-cast-state-and-threat-zones.md`.

use dsl_ast::ast::*;
use dsl_ast::parse_ability_file;

fn parse_one(src: &str) -> AbilityDecl {
    let file = parse_ability_file(src).unwrap_or_else(|e| {
        panic!("parse failed:\n{e}\nsource:\n{src}");
    });
    assert_eq!(file.abilities.len(), 1, "expected exactly one ability");
    file.abilities.into_iter().next().unwrap()
}

// ---------------------------------------------------------------------------
// Backwards compatibility — legacy ability shape still parses with
// program: None.
// ---------------------------------------------------------------------------

#[test]
fn legacy_ability_keeps_program_none() {
    let src = r"
        ability Strike {
            target: enemy
            range: 100
            cooldown: 2s

            damage 30
        }
    ";
    let a = parse_one(src);
    assert!(
        a.program.is_none(),
        "legacy ability without `cast {{` block must produce program=None; got {:?}",
        a.program,
    );
    assert_eq!(a.effects.len(), 1, "legacy bare effect should land in effects");
}

// ---------------------------------------------------------------------------
// Cooldown @ phase qualifier
// ---------------------------------------------------------------------------

#[test]
fn cooldown_default_phase_is_none() {
    let src = r"
        ability Strike {
            target: enemy
            cooldown: 5s
            damage 10
        }
    ";
    let a = parse_one(src);
    let phase = a.headers.iter().find_map(|h| match h {
        AbilityHeader::Cooldown(_, p) => Some(*p),
        _ => None,
    });
    assert_eq!(phase, Some(None), "bare `cooldown:` must store phase=None");
}

#[test]
fn cooldown_at_resolve_qualifier_parses() {
    let src = r"
        ability Strike {
            target: enemy
            cooldown: 5s @ resolve
            damage 10
        }
    ";
    let a = parse_one(src);
    let phase = a.headers.iter().find_map(|h| match h {
        AbilityHeader::Cooldown(_, p) => *p,
        _ => None,
    });
    assert_eq!(phase, Some(CooldownPhase::Resolve));
}

#[test]
fn cooldown_at_cast_qualifier_parses() {
    let src = r"
        ability Strike {
            target: enemy
            cooldown: 5s @ cast
            damage 10
        }
    ";
    let a = parse_one(src);
    let phase = a.headers.iter().find_map(|h| match h {
        AbilityHeader::Cooldown(_, p) => *p,
        _ => None,
    });
    assert_eq!(phase, Some(CooldownPhase::Cast));
}

#[test]
fn cooldown_at_interrupt_qualifier_parses() {
    let src = r"
        ability Strike {
            target: enemy
            cooldown: 5s @ interrupt
            damage 10
        }
    ";
    let a = parse_one(src);
    let phase = a.headers.iter().find_map(|h| match h {
        AbilityHeader::Cooldown(_, p) => *p,
        _ => None,
    });
    assert_eq!(phase, Some(CooldownPhase::Interrupt));
}

#[test]
fn cooldown_unknown_phase_errors() {
    let src = r"
        ability Strike {
            cooldown: 5s @ banana
            damage 10
        }
    ";
    let err = parse_ability_file(src).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("banana"), "error must mention the unknown phase: {msg}");
}

// ---------------------------------------------------------------------------
// cast { … } block
// ---------------------------------------------------------------------------

#[test]
fn cast_block_minimal() {
    let src = r"
        ability Firebolt {
            target: enemy
            range: 30
            cast { duration: 3t }
            effect { damage 25 }
        }
    ";
    let a = parse_one(src);
    let program = a.program.expect("cast {} block should populate program");
    assert_eq!(program.len(), 2, "expected one Cast step + one Effects step");
    match &program[0] {
        AbilityProgramStep::Cast(spec) => {
            assert_eq!(spec.duration_ticks, 3);
            assert!(spec.telegraph.is_none());
            assert_eq!(spec.interrupts, InterruptSet::Standard);
        }
        other => panic!("expected Cast step first, got {other:?}"),
    }
    match &program[1] {
        AbilityProgramStep::Effects(effs) => {
            assert_eq!(effs.len(), 1);
            assert_eq!(effs[0].verb, "damage");
        }
        other => panic!("expected Effects step second, got {other:?}"),
    }
}

#[test]
fn cast_block_with_telegraph_and_interrupts() {
    let src = r"
        ability Firebolt {
            target: enemy
            cast {
                duration: 3t;
                telegraph: line(self.pos, target.pos, width: 2);
                interrupts: standard
            }
            effect { damage 25 }
        }
    ";
    let a = parse_one(src);
    let program = a.program.unwrap();
    let cast = match &program[0] {
        AbilityProgramStep::Cast(s) => s.clone(),
        _ => unreachable!(),
    };
    assert_eq!(cast.duration_ticks, 3);
    assert!(cast.telegraph.as_deref().unwrap().contains("line"));
    assert_eq!(cast.interrupts, InterruptSet::Standard);
}

#[test]
fn cast_block_requires_duration() {
    let src = r"
        ability Bad { cast { interrupts: none } effect { damage 1 } }
    ";
    let err = parse_ability_file(src).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("duration"), "error must mention missing duration: {msg}");
}

#[test]
fn cast_block_unknown_field_errors() {
    let src = r"
        ability Bad { cast { duration: 3t; flavour: hot } effect { damage 1 } }
    ";
    let err = parse_ability_file(src).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("flavour"), "error must mention unknown field: {msg}");
}

// ---------------------------------------------------------------------------
// interrupts: set syntax
// ---------------------------------------------------------------------------

#[test]
fn interrupts_explicit_subset() {
    let src = r"
        ability F {
            cast { duration: 1t; interrupts: { damage, stun } }
            effect { damage 1 }
        }
    ";
    let a = parse_one(src);
    let cast = match &a.program.unwrap()[0] {
        AbilityProgramStep::Cast(s) => s.clone(),
        _ => unreachable!(),
    };
    match &cast.interrupts {
        InterruptSet::Subset(kinds) => {
            assert_eq!(kinds.len(), 2);
            assert!(kinds.contains(&InterruptKind::Damage));
            assert!(kinds.contains(&InterruptKind::Stun));
        }
        other => panic!("expected Subset, got {other:?}"),
    }
}

#[test]
fn interrupts_none_uninterruptible() {
    let src = r"
        ability BindSoul {
            cast { duration: 10t; interrupts: none }
            effect { damage 999 }
        }
    ";
    let a = parse_one(src);
    let cast = match &a.program.unwrap()[0] {
        AbilityProgramStep::Cast(s) => s.clone(),
        _ => unreachable!(),
    };
    assert_eq!(cast.interrupts, InterruptSet::None);
}

#[test]
fn interrupts_standard_plus_movement() {
    let src = r"
        ability FocusFire {
            cast { duration: 2t; interrupts: standard + { movement } }
            effect { damage 5 }
        }
    ";
    let a = parse_one(src);
    let cast = match &a.program.unwrap()[0] {
        AbilityProgramStep::Cast(s) => s.clone(),
        _ => unreachable!(),
    };
    match &cast.interrupts {
        InterruptSet::StandardPlus(kinds) => {
            assert_eq!(kinds, &vec![InterruptKind::Movement]);
        }
        other => panic!("expected StandardPlus, got {other:?}"),
    }
}

#[test]
fn interrupts_standard_minus_damage() {
    let src = r"
        ability HardyForage {
            cast { duration: 5t; interrupts: standard - { damage } }
            effect { damage 0 }
        }
    ";
    let a = parse_one(src);
    let cast = match &a.program.unwrap()[0] {
        AbilityProgramStep::Cast(s) => s.clone(),
        _ => unreachable!(),
    };
    match &cast.interrupts {
        InterruptSet::StandardMinus(kinds) => {
            assert_eq!(kinds, &vec![InterruptKind::Damage]);
        }
        other => panic!("expected StandardMinus, got {other:?}"),
    }
}

#[test]
fn interrupts_unknown_kind_errors() {
    let src = r"
        ability Bad {
            cast { duration: 1t; interrupts: { tickled } }
            effect { damage 1 }
        }
    ";
    let err = parse_ability_file(src).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("tickled"), "error must mention unknown interrupt kind: {msg}");
}

// ---------------------------------------------------------------------------
// Multi-stage program
// ---------------------------------------------------------------------------

#[test]
fn multi_stage_program_chains_cast_effect() {
    let src = r"
        ability Channelbeam {
            target: enemy

            cast { duration: 1t; interrupts: standard }
            effect { damage 5 }

            cast { duration: 1t; interrupts: standard }
            effect { damage 8 }

            cast { duration: 1t; interrupts: standard }
            effect { damage 15 }
        }
    ";
    let a = parse_one(src);
    let program = a.program.unwrap();
    assert_eq!(program.len(), 6, "expected three Cast + three Effects steps in alternation");
    for (i, step) in program.iter().enumerate() {
        if i % 2 == 0 {
            assert!(matches!(step, AbilityProgramStep::Cast(_)), "step {i} should be Cast");
        } else {
            assert!(matches!(step, AbilityProgramStep::Effects(_)), "step {i} should be Effects");
        }
    }
}

// ---------------------------------------------------------------------------
// Mutual exclusion: bare effects + cast/effect blocks reject
// ---------------------------------------------------------------------------

#[test]
fn mixing_bare_effects_with_program_blocks_errors() {
    let src = r"
        ability Bad {
            target: enemy

            damage 10                          # bare effect (legacy)

            cast { duration: 1t }
            effect { damage 5 }
        }
    ";
    let err = parse_ability_file(src).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("mixes")
            || msg.contains("bare")
            || msg.contains("ONE shape"),
        "error must flag the mixed shape: {msg}",
    );
}
