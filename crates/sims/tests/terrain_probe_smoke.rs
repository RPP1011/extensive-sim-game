//! Runtime gate (per plan AIS): boot terrain gen on a worker thread,
//! await readiness, install into Arc<dyn TerrainQuery>, assert the
//! trait surface reflects the declared materials.

use std::sync::Arc;
use engine::terrain::TerrainQuery;
use engine_voxel::TerrainGenHandle;
use glam::Vec3;

#[test]
fn async_gen_installs_and_serves_terrain_query() {
    // Generate on a worker thread.
    let handle = TerrainGenHandle::spawn(42, sims::terrain_probe::generate_terrain);
    let terrain = handle.block_until_ready().expect("gen succeeds");

    // Post-condition 1: declared extent + cell value.
    assert_eq!(terrain.extent(), 8);
    assert_eq!(terrain.cell_at(0, 0, 0), 2, "fill stone id=2 should occupy origin");
    assert_eq!(terrain.cell_at(7, 7, 7), 2, "fill stone id=2 should occupy max corner");

    // Post-condition 2: materials table reachable + correct.
    // Bind the table first — `materials()` returns by value (Copy) and
    // `get()` borrows from it, so we need a longer-lived binding.
    let mtable = terrain.materials();
    let stone = mtable.get(2).expect("stone in table");
    assert_eq!(stone.walkable, false);
    assert_eq!(stone.hardness, 8);

    // Post-condition 3: install into Arc<dyn TerrainQuery> and
    // exercise the trait surface — this is the actual changed
    // code path the runtime gate must cover.
    let arc: Arc<dyn TerrainQuery + Send + Sync> = Arc::new(terrain);

    // `height_at` over the filled region should be > 0 (solid below).
    // The fill layer fills all 8 z-levels; top face of z=7 cell = 8.0.
    let h = arc.height_at(4.0, 4.0);
    assert!(h > 0.0, "expected non-zero ground height over filled region, got {h}");

    // `walkable` for a Walk-mode agent must reflect the stone
    // declaration (walkable: false → false). The VoxelTerrain impl
    // returns `cell == 0` for Walk mode; stone fill (cell=2) → false.
    let walkable = arc.walkable(
        Vec3::new(4.0, 4.0, 4.0),
        engine_voxel::MovementMode::Walk,
    );
    assert!(!walkable, "stone fill must block Walk mode");

    // `line_of_sight` across the solid region must be blocked.
    let los = arc.line_of_sight(Vec3::new(0.5, 4.0, 4.0), Vec3::new(7.5, 4.0, 4.0));
    assert!(!los, "stone fill must block LOS");
}
