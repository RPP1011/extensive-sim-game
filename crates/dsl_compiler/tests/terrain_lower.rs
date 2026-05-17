use dsl_compiler::{lower::lower_terrain, parse, LowerError};

#[test]
fn lower_resolves_fill_layer_material_to_id() {
    let src = r#"
terrain {
  extent: 32
  cell_size: 1.0
  seed_purpose: 0x1
  materials {
    grass { id: 7, walkable: true, hardness: 1, color: 0x4A8B3A }
  }
  layer fill { material: grass }
}
"#;
    let program = parse(src).unwrap();
    let lowered = lower_terrain(&program.terrain.unwrap()).unwrap();
    assert_eq!(lowered.layers.len(), 1);
    match &lowered.layers[0] {
        dsl_compiler::lower::TerrainLayerIr::Fill { material_id } => {
            assert_eq!(*material_id, 7);
        }
    }
}

#[test]
fn lower_rejects_unknown_material_ref() {
    let src = r#"
terrain {
  extent: 32
  cell_size: 1.0
  seed_purpose: 0x1
  materials { grass { id: 1, walkable: true, hardness: 1, color: 0x4A8B3A } }
  layer fill { material: ghost }
}
"#;
    let program = parse(src).unwrap();
    let err = lower_terrain(&program.terrain.unwrap()).err().unwrap();
    assert!(
        matches!(err, LowerError::UnknownMaterial(ref name) if name == "ghost"),
        "got: {err:?}"
    );
}
