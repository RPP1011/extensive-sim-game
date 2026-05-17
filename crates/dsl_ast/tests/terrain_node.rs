use dsl_ast::terrain::{TerrainBlock, MaterialDecl, LayerDecl, LayerKind};

#[test]
fn terrain_block_construct_and_field_access() {
    let block = TerrainBlock {
        extent: 128,
        cell_size: 0.25,
        seed_purpose: 0xBA5E_7E55,
        materials: vec![],
        layers: vec![],
    };
    assert_eq!(block.extent, 128);
    assert!((block.cell_size - 0.25).abs() < 1e-6);
    assert_eq!(block.seed_purpose, 0xBA5E_7E55);
    assert!(block.materials.is_empty());
    assert!(block.layers.is_empty());
}

#[test]
fn material_decl_defaults() {
    let m = MaterialDecl {
        name: "grass".into(),
        id: 1,
        walkable: true,
        hardness: 1,
        biome_tag: None,
        color: 0x4A8B3A,
        movement_cost: 1.0,
    };
    assert_eq!(m.id, 1);
    assert_eq!(m.color, 0x4A8B3A);
}

#[test]
fn layer_fill_decl() {
    let l = LayerDecl {
        index: 1,
        kind: LayerKind::Fill { material: "grass".into() },
    };
    assert_eq!(l.index, 1);
    matches!(l.kind, LayerKind::Fill { .. });
}
