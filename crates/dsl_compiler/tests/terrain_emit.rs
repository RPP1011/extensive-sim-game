use dsl_compiler::{parse, lower::lower_terrain, emit::emit_terrain};

#[test]
fn emit_contains_extent_materials_and_fill_call() {
    let src = r#"
terrain {
  extent: 16
  cell_size: 0.5
  seed_purpose: 0xBA5E_7E55
  materials {
    grass { id: 1, walkable: true,  hardness: 1, color: 0x4A8B3A }
    stone { id: 2, walkable: false, hardness: 8, color: 0x808080 }
  }
  layer fill { material: stone }
}
"#;
    let ir = lower_terrain(&parse(src).unwrap().terrain.unwrap()).unwrap();
    let rust_src = emit_terrain(&ir);

    // Smoke checks on emitted text. Goldens are *not* committed
    // (ahash drift caveat); these substring assertions are stable.
    assert!(rust_src.contains("pub const EXTENT: u32 = 16"), "extent missing");
    assert!(rust_src.contains("pub const CELL_SIZE: f32 = 0.5"), "cell_size missing");
    assert!(rust_src.contains("pub const SEED_PURPOSE: u32 = 0xBA5E_7E55"),
            "seed_purpose missing");
    assert!(rust_src.contains("pub static MATERIALS: ::engine_voxel::MaterialTable"),
            "MATERIALS static missing");
    assert!(rust_src.contains("id: 1u8") && rust_src.contains("id: 2u8"),
            "material ids missing");
    assert!(rust_src.contains("pub fn generate_terrain(world_seed: u64)"),
            "generate_terrain signature missing");
    assert!(rust_src.contains("::engine_voxel::terrain_layers::layer_fill"),
            "layer_fill call missing");
    assert!(rust_src.contains("2u8"), "fill material id 2 missing in call");
}
