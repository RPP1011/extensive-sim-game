use engine_voxel::{MaterialTable, MaterialRow};

#[test]
fn lookup_by_id() {
    static ROWS: [MaterialRow; 3] = [
        MaterialRow { id: 1, walkable: true,  hardness: 1, biome_tag_hash: 0, color: 0x4A8B3A, movement_cost: 1.0 },
        MaterialRow { id: 2, walkable: false, hardness: 8, biome_tag_hash: 0, color: 0x808080, movement_cost: 1.0 },
        MaterialRow { id: 3, walkable: true,  hardness: 2, biome_tag_hash: 0, color: 0xD9C28A, movement_cost: 1.5 },
    ];
    let t = MaterialTable::new(&ROWS);
    assert_eq!(t.get(1).unwrap().color, 0x4A8B3A);
    assert_eq!(t.get(2).unwrap().walkable, false);
    assert!((t.get(3).unwrap().movement_cost - 1.5).abs() < 1e-6);
    assert!(t.get(0).is_none());     // air sentinel
    assert!(t.get(99).is_none());    // unknown id
}
