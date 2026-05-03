//! Combat Foundation Tasks 6 + 8.
//!
//! Pins `AbilityId` newtype semantics, then (Task 8) the `AbilityRegistry`
//! append-only slot-stable builder contract.

use engine::ability::{
    AbilityId, AbilityProgram, AbilityRegistry, AbilityRegistryBuilder,
    EffectOp, Gate,
};

// ---------- Task 6: AbilityId ----------

#[test]
fn ability_id_round_trips_through_new() {
    let a = AbilityId::new(1).unwrap();
    assert_eq!(a.raw(), 1);
    assert_eq!(a.slot(), 0);
    let b = AbilityId::new(42).unwrap();
    assert_eq!(b.raw(), 42);
    assert_eq!(b.slot(), 41);
}

#[test]
fn ability_id_rejects_zero() {
    assert!(AbilityId::new(0).is_none());
}

#[test]
fn option_ability_id_niche_optimized() {
    assert_eq!(std::mem::size_of::<Option<AbilityId>>(), 4);
}

// ---------- Task 8: registry + builder ----------

fn gate_hostile() -> Gate {
    Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false }
}

#[test]
fn empty_registry_get_returns_none() {
    let reg = AbilityRegistry::new();
    assert!(reg.is_empty());
    assert_eq!(reg.len(), 0);
    // Any id will miss.
    let id = AbilityId::new(1).unwrap();
    assert!(reg.get(id).is_none());
}

#[test]
fn builder_register_returns_monotonic_ids() {
    let mut b = AbilityRegistryBuilder::new();
    let a1 = b.register(AbilityProgram::new_single_target(
        5.0, gate_hostile(), [EffectOp::Damage { amount: 10.0 }],
    ));
    let a2 = b.register(AbilityProgram::new_single_target(
        5.0, gate_hostile(), [EffectOp::Heal { amount: 20.0 }],
    ));
    let a3 = b.register(AbilityProgram::new_single_target(
        3.0, gate_hostile(), [EffectOp::Shield { amount: 30.0 }],
    ));
    assert_eq!(a1.raw(), 1);
    assert_eq!(a2.raw(), 2);
    assert_eq!(a3.raw(), 3);
    assert_eq!(a1.slot(), 0);
    assert_eq!(a2.slot(), 1);
    assert_eq!(a3.slot(), 2);
}

#[test]
fn built_registry_lookup_round_trips() {
    let mut b = AbilityRegistryBuilder::new();
    let dmg = b.register(AbilityProgram::new_single_target(
        6.0, gate_hostile(), [EffectOp::Damage { amount: 50.0 }],
    ));
    let heal = b.register(AbilityProgram::new_single_target(
        4.0, gate_hostile(), [EffectOp::Heal { amount: 25.0 }],
    ));
    let reg = b.build();
    assert_eq!(reg.len(), 2);

    let p_dmg = reg.get(dmg).expect("damage program must exist");
    assert_eq!(p_dmg.effects.len(), 1);
    assert!(matches!(p_dmg.effects[0], EffectOp::Damage { amount: 50.0 }));

    let p_heal = reg.get(heal).expect("heal program must exist");
    assert!(matches!(p_heal.effects[0], EffectOp::Heal { amount: 25.0 }));
}

#[test]
fn registry_get_out_of_range_returns_none() {
    let mut b = AbilityRegistryBuilder::new();
    let _ = b.register(AbilityProgram::new_single_target(
        5.0, gate_hostile(), [EffectOp::Damage { amount: 10.0 }],
    ));
    let reg = b.build();
    // Slot 0 exists; slot 999 (id=1000) doesn't.
    let oob = AbilityId::new(1000).unwrap();
    assert!(reg.get(oob).is_none());
}

#[test]
fn default_registry_is_empty() {
    let reg = AbilityRegistry::default();
    assert_eq!(reg.len(), 0);
    assert!(reg.is_empty());
}
