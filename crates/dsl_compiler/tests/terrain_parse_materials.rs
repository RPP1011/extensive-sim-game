use dsl_compiler::parse;

#[test]
fn parses_materials_block_with_one_entry() {
    let src = r#"
terrain {
  extent: 64
  cell_size: 1.0
  seed_purpose: 0x1
  materials {
    grass { id: 1, walkable: true, hardness: 1, color: 0x4A8B3A }
  }
}
"#;
    let t = parse(src).unwrap().terrain.unwrap();
    assert_eq!(t.materials.len(), 1);
    let g = &t.materials[0];
    assert_eq!(g.name, "grass");
    assert_eq!(g.id, 1);
    assert!(g.walkable);
    assert_eq!(g.hardness, 1);
    assert_eq!(g.color, 0x4A8B3A);
    assert!((g.movement_cost - 1.0).abs() < 1e-6); // default
    assert!(g.biome_tag.is_none());                 // default
}

#[test]
fn rejects_duplicate_material_id() {
    let src = r#"
terrain {
  extent: 64
  cell_size: 1.0
  seed_purpose: 0x1
  materials {
    a { id: 1, walkable: true,  hardness: 1, color: 0xFFFFFF }
    b { id: 1, walkable: false, hardness: 2, color: 0x000000 }
  }
}
"#;
    let err = parse(src).err().unwrap();
    assert!(format!("{err}").contains("duplicate material id"), "got: {err}");
}

#[test]
fn rejects_material_id_zero() {
    let src = r#"
terrain {
  extent: 64
  cell_size: 1.0
  seed_purpose: 0x1
  materials {
    void { id: 0, walkable: true, hardness: 0, color: 0 }
  }
}
"#;
    let err = parse(src).err().unwrap();
    assert!(format!("{err}").contains("material id must be 1..=255"), "got: {err}");
}
