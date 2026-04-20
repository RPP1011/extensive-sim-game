//! Combat Foundation Task 6 scaffold + Task 8 extension.
//!
//! Pins `AbilityId` newtype semantics and (after Task 8) the `AbilityRegistry`
//! append-only slot-stable builder contract.

use engine::ability::AbilityId;

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
    // Niche optimisation: Option<AbilityId> packs to 4 bytes.
    assert_eq!(std::mem::size_of::<Option<AbilityId>>(), 4);
}
