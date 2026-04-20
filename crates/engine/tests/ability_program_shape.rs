//! Combat Foundation Task 7 — pins `EffectOp` size + `AbilityProgram`
//! constructor shapes. The size budget blocks silent growth; the construction
//! tests pin variant field names so a rename trips the test before it trips
//! a caller.

use engine::ability::{
    AbilityId, AbilityProgram, Area, Delivery, EffectOp, Gate,
    TargetSelector, MAX_EFFECTS_PER_PROGRAM,
};

#[test]
fn effect_op_size_under_16_bytes() {
    let sz = std::mem::size_of::<EffectOp>();
    assert!(
        sz <= 16,
        "EffectOp grew past 16B budget: {sz}. Extend the `16` only with an \
         accompanying schema-hash bump AND a cache-locality review of the \
         cast-dispatch hot path."
    );
}

#[test]
fn max_effects_per_program_bounded() {
    // The smallvec inline budget is 4 effects; growing this changes the
    // schema hash and the cast-dispatch hot-path footprint.
    assert_eq!(MAX_EFFECTS_PER_PROGRAM, 4);
}

#[test]
fn single_effect_damage_program() {
    let gate = Gate { cooldown_ticks: 20, hostile_only: true, line_of_sight: false };
    let p = AbilityProgram::new_single_target(6.0, gate, [EffectOp::Damage { amount: 50.0 }]);
    assert!(matches!(p.delivery, Delivery::Instant));
    assert!(matches!(p.area, Area::SingleTarget { range: 6.0 }));
    assert_eq!(p.effects.len(), 1);
    assert!(matches!(p.effects[0], EffectOp::Damage { amount: 50.0 }));
    assert_eq!(p.gate.cooldown_ticks, 20);
    assert!(p.gate.hostile_only);
    assert!(!p.gate.line_of_sight);
}

#[test]
fn damage_plus_stun_program() {
    let gate = Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false };
    let p = AbilityProgram::new_single_target(
        4.0,
        gate,
        [
            EffectOp::Damage { amount: 20.0 },
            EffectOp::Stun { duration_ticks: 5 },
        ],
    );
    assert_eq!(p.effects.len(), 2);
    assert!(matches!(p.effects[0], EffectOp::Damage { amount: 20.0 }));
    assert!(matches!(p.effects[1], EffectOp::Stun { duration_ticks: 5 }));
}

#[test]
fn world_effects_program() {
    let gate = Gate { cooldown_ticks: 100, hostile_only: false, line_of_sight: false };
    let p = AbilityProgram::new_single_target(
        3.0,
        gate,
        [
            EffectOp::TransferGold { amount: -50 },
            EffectOp::ModifyStanding { delta: -20 },
        ],
    );
    assert_eq!(p.effects.len(), 2);
    assert!(matches!(p.effects[0], EffectOp::TransferGold { amount: -50 }));
    assert!(matches!(p.effects[1], EffectOp::ModifyStanding { delta: -20 }));
}

#[test]
fn recursive_chain_program() {
    let nested = AbilityId::new(3).unwrap();
    let gate = Gate { cooldown_ticks: 1, hostile_only: true, line_of_sight: false };
    let p = AbilityProgram::new_single_target(
        5.0,
        gate,
        [
            EffectOp::Damage { amount: 10.0 },
            EffectOp::CastAbility { ability: nested, selector: TargetSelector::Target },
            EffectOp::CastAbility { ability: nested, selector: TargetSelector::Caster },
        ],
    );
    assert_eq!(p.effects.len(), 3);
    match p.effects[1] {
        EffectOp::CastAbility { ability, selector } => {
            assert_eq!(ability.raw(), 3);
            assert_eq!(selector, TargetSelector::Target);
        }
        _ => panic!("expected CastAbility at index 1"),
    }
    match p.effects[2] {
        EffectOp::CastAbility { selector, .. } => assert_eq!(selector, TargetSelector::Caster),
        _ => panic!("expected CastAbility at index 2"),
    }
}
