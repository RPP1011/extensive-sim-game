//! Runtime gate: the merged Program's materials come from stdlib via
//! the import system, and the emitted generate_terrain produces a
//! VoxelTerrain reflecting those materials.
//!
//! # Environment
//! `WORLDSIM_STDLIB_ROOT` is optional: `crates/sims/build.rs` (via
//! `dsl_compiler::build_helper`) defaults to `<workspace>/stdlib` when the
//! env var is absent. Tests run without any extra env setup.

use engine_voxel::TerrainGenHandle;

#[test]
fn imported_materials_reach_generate_terrain() {
    let handle = TerrainGenHandle::spawn(42, sims::terrain_probe_imported::generate_terrain);
    let terrain = handle.block_until_ready().expect("gen succeeds");

    // The stdlib basic.sim declares grass (id=1), stone (id=2), sand (id=3)
    // and layer fill { material: stone }. So every cell should be stone (id=2).
    assert_eq!(terrain.extent(), 8);
    assert_eq!(terrain.cell_at(0, 0, 0), 2, "fill from stdlib should be stone");
    assert_eq!(terrain.cell_at(7, 7, 7), 2, "fill from stdlib should be stone");

    // Materials table contains the stdlib materials by id.
    let mtable = terrain.materials();
    let grass = mtable.get(1).expect("grass id=1");
    assert!(grass.walkable);
    let stone = mtable.get(2).expect("stone id=2");
    assert!(!stone.walkable);
    let sand = mtable.get(3).expect("sand id=3");
    assert!((sand.movement_cost - 1.5).abs() < 1e-6);
}
