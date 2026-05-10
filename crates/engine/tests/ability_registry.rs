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

// ---------- Plan I-step-3: hot-reload primitive ----------

/// `with_program_replaced` returns a NEW registry with one slot
/// swapped. The original registry is unchanged (immutable contract
/// preserved); slot ids stay stable across the swap so any in-flight
/// `Arc<AbilityRegistry>` holders keep seeing valid lookups.
#[test]
fn with_program_replaced_swaps_one_slot_and_leaves_others() {
    let mut b = AbilityRegistryBuilder::new();
    let dmg = b.register(AbilityProgram::new_single_target(
        5.0, gate_hostile(), [EffectOp::Damage { amount: 10.0 }],
    ));
    let heal = b.register(AbilityProgram::new_single_target(
        5.0, gate_hostile(), [EffectOp::Heal { amount: 20.0 }],
    ));
    let v1 = b.build();

    // Hot-reload the damage program: 10.0 → 25.0 (simulating the
    // author editing `damage 10` → `damage 25` and re-saving).
    let v2 = v1
        .with_program_replaced(
            dmg,
            AbilityProgram::new_single_target(
                5.0, gate_hostile(), [EffectOp::Damage { amount: 25.0 }],
            ),
        )
        .expect("dmg slot is in range");

    // Original registry unchanged — damage still at 10.0.
    let p_old = v1.get(dmg).expect("v1 still has damage program");
    assert!(
        matches!(p_old.effects[0], EffectOp::Damage { amount: 10.0 }),
        "original registry must be untouched; got {:?}",
        p_old.effects[0],
    );

    // New registry sees the swapped value at the SAME id.
    let p_new = v2.get(dmg).expect("v2 has new damage program");
    assert!(
        matches!(p_new.effects[0], EffectOp::Damage { amount: 25.0 }),
        "new registry must reflect hot-reloaded value; got {:?}",
        p_new.effects[0],
    );

    // Other slots cloned through unchanged.
    let p_heal_v1 = v1.get(heal).unwrap();
    let p_heal_v2 = v2.get(heal).unwrap();
    assert!(matches!(p_heal_v1.effects[0], EffectOp::Heal { amount: 20.0 }));
    assert!(matches!(p_heal_v2.effects[0], EffectOp::Heal { amount: 20.0 }));

    // Length identical — no slot reorder.
    assert_eq!(v1.len(), 2);
    assert_eq!(v2.len(), 2);
}

/// Out-of-range ids surface as `None` rather than panicking. A
/// hot-reload runtime should fall back to a full rebuild when this
/// happens (the slot table changed, e.g. a `register()` reorder).
#[test]
fn with_program_replaced_returns_none_on_out_of_range_id() {
    let mut b = AbilityRegistryBuilder::new();
    let _ = b.register(AbilityProgram::new_single_target(
        5.0, gate_hostile(), [EffectOp::Damage { amount: 10.0 }],
    ));
    let reg = b.build();

    let oob = AbilityId::new(999).unwrap();
    let result = reg.with_program_replaced(
        oob,
        AbilityProgram::new_single_target(
            5.0, gate_hostile(), [EffectOp::Heal { amount: 1.0 }],
        ),
    );
    assert!(result.is_none(), "OOR id must surface None for runtime fallback");
}
