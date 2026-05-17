use dsl_compiler::parse;
use dsl_ast::terrain::LayerKind;

#[test]
fn parses_single_fill_layer() {
    let src = r#"
terrain {
  extent: 32
  cell_size: 1.0
  seed_purpose: 0x1
  materials { grass { id: 1, walkable: true, hardness: 1, color: 0x4A8B3A } }
  layer fill { material: grass }
}
"#;
    let t = parse(src).unwrap().terrain.unwrap();
    assert_eq!(t.layers.len(), 1);
    assert_eq!(t.layers[0].index, 1);
    match &t.layers[0].kind {
        LayerKind::Fill { material } => assert_eq!(material, "grass"),
    }
}

#[test]
fn layer_indices_are_source_order_one_based() {
    let src = r#"
terrain {
  extent: 32
  cell_size: 1.0
  seed_purpose: 0x1
  materials {
    a { id: 1, walkable: true, hardness: 1, color: 0xFF0000 }
    b { id: 2, walkable: true, hardness: 1, color: 0x00FF00 }
  }
  layer fill { material: a }
  layer fill { material: b }
}
"#;
    let t = parse(src).unwrap().terrain.unwrap();
    assert_eq!(t.layers.len(), 2);
    assert_eq!(t.layers[0].index, 1);
    assert_eq!(t.layers[1].index, 2);
}

#[test]
fn rejects_unknown_material_in_fill_layer() {
    // Material name resolution is a lowering concern, not parsing.
    // This test asserts the *parser* still produces an AST with the
    // unresolved name — the failure surfaces in Task 5 (lowering).
    let src = r#"
terrain {
  extent: 32
  cell_size: 1.0
  seed_purpose: 0x1
  materials { grass { id: 1, walkable: true, hardness: 1, color: 0x4A8B3A } }
  layer fill { material: ghost }
}
"#;
    let t = parse(src).unwrap().terrain.unwrap();
    match &t.layers[0].kind {
        LayerKind::Fill { material } => assert_eq!(material, "ghost"),
    }
}
