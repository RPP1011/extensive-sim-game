use engine_voxel::{VoxelTerrain, terrain_layers::layer_fill};

#[test]
fn layer_fill_writes_every_cell() {
    let mut terrain = VoxelTerrain::with_extent(4);
    layer_fill(&mut terrain, /*material_id=*/ 2);
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                assert_eq!(terrain.cell_at(x, y, z), 2, "cell ({x},{y},{z})");
            }
        }
    }
}

#[test]
fn layer_fill_with_air_id_zero_clears() {
    let mut terrain = VoxelTerrain::with_extent(2);
    layer_fill(&mut terrain, 5);
    layer_fill(&mut terrain, 0);
    assert_eq!(terrain.cell_at(0, 0, 0), 0);
    assert_eq!(terrain.cell_at(1, 1, 1), 0);
}
