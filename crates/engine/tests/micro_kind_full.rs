use engine::mask::MicroKind;

#[test]
fn all_variants_present() {
    let all = MicroKind::ALL;
    assert_eq!(all.len(), 18);
}

#[test]
fn specific_ordinal_assignments() {
    // Movement (3)
    assert_eq!(MicroKind::Hold        as u8, 0);
    assert_eq!(MicroKind::MoveToward  as u8, 1);
    assert_eq!(MicroKind::Flee        as u8, 2);
    // Combat (3)
    assert_eq!(MicroKind::Attack      as u8, 3);
    assert_eq!(MicroKind::Cast        as u8, 4);
    assert_eq!(MicroKind::UseItem     as u8, 5);
    // Resource (4)
    assert_eq!(MicroKind::Harvest     as u8, 6);
    assert_eq!(MicroKind::Eat         as u8, 7);
    assert_eq!(MicroKind::Drink       as u8, 8);
    assert_eq!(MicroKind::Rest        as u8, 9);
    // Construction (3)
    assert_eq!(MicroKind::PlaceTile    as u8, 10);
    assert_eq!(MicroKind::PlaceVoxel   as u8, 11);
    assert_eq!(MicroKind::HarvestVoxel as u8, 12);
    // Social (2)
    assert_eq!(MicroKind::Converse     as u8, 13);
    assert_eq!(MicroKind::ShareStory   as u8, 14);
    // Info push + pull (2)
    assert_eq!(MicroKind::Communicate  as u8, 15);
    assert_eq!(MicroKind::Ask          as u8, 16);
    // Memory (1)
    assert_eq!(MicroKind::Remember     as u8, 17);
}

#[test]
fn all_slice_matches_ordinal_order() {
    for (i, k) in MicroKind::ALL.iter().enumerate() {
        assert_eq!(*k as u8, i as u8);
    }
}
