use glam::Vec3;
use voxel_engine::voxel::grid::VoxelGrid;
use viz::grid_paint::{
    clear_above_ground, grid_index_of, paint_agent, paint_ground_plane,
    paint_line, paint_ring, GRID_SIDE, GROUND_Y,
};
use viz::palette::{PAL_AIR, PAL_ATTACK, PAL_GROUND, PAL_HUMAN};

fn fresh_grid() -> VoxelGrid { VoxelGrid::new(GRID_SIDE, GRID_SIDE, GRID_SIDE) }

#[test]
fn ground_plane_fills_its_layer() {
    let mut g = fresh_grid();
    paint_ground_plane(&mut g);
    for x in [0u32, 1, GRID_SIDE - 1] {
        for z in [0u32, 1, GRID_SIDE - 1] {
            assert_eq!(g.get(x, GROUND_Y, z), Some(PAL_GROUND));
        }
    }
    assert_eq!(g.get(5, GROUND_Y + 1, 5), Some(PAL_AIR));
    assert_eq!(g.get(5, GROUND_Y - 1, 5), Some(PAL_AIR));
}

#[test]
fn clear_above_ground_erases_only_above_layer() {
    let mut g = fresh_grid();
    paint_ground_plane(&mut g);
    g.set(5, GROUND_Y + 1, 5, PAL_HUMAN);
    g.set(5, GROUND_Y - 1, 5, PAL_GROUND);
    clear_above_ground(&mut g);
    assert_eq!(g.get(5, GROUND_Y + 1, 5), Some(PAL_AIR));
    assert_eq!(g.get(5, GROUND_Y, 5),     Some(PAL_GROUND));
    assert_eq!(g.get(5, GROUND_Y - 1, 5), Some(PAL_GROUND));
}

#[test]
fn paint_agent_stamps_one_voxel() {
    let mut g = fresh_grid();
    paint_agent(&mut g, Vec3::new(10.5, 4.0, 20.0), PAL_HUMAN);
    assert_eq!(g.get(10, 4, 20), Some(PAL_HUMAN));
    assert_eq!(g.get(11, 4, 20), Some(PAL_AIR));
}

#[test]
fn paint_line_draws_3_4_5_triangle() {
    let mut g = fresh_grid();
    paint_line(&mut g,
        Vec3::new(0.0, 10.0, 0.0),
        Vec3::new(3.0, 10.0, 4.0),
        PAL_ATTACK);
    assert_eq!(g.get(0, 10, 0), Some(PAL_ATTACK));
    assert_eq!(g.get(3, 10, 4), Some(PAL_ATTACK));
    let mut count = 0;
    for x in 0..GRID_SIDE { for z in 0..GRID_SIDE {
        if g.get(x, 10, z) == Some(PAL_ATTACK) { count += 1; }
    }}
    assert!(count >= 4, "expected >=4 line voxels, got {}", count);
}

#[test]
fn paint_ring_stays_on_the_circle() {
    let mut g = fresh_grid();
    let center = Vec3::new(40.0, 10.0, 40.0);
    paint_ring(&mut g, center, 10.0, PAL_ATTACK);
    let mut any = false;
    for x in 0..GRID_SIDE { for z in 0..GRID_SIDE {
        if g.get(x, 10, z) == Some(PAL_ATTACK) {
            let dx = x as f32 - 40.0;
            let dz = z as f32 - 40.0;
            let d  = (dx*dx + dz*dz).sqrt();
            assert!((d - 10.0).abs() < 1.5, "ring voxel at ({},{}) d={:.2}", x, z, d);
            any = true;
        }
    }}
    assert!(any, "paint_ring stamped zero voxels");
}

#[test]
fn grid_index_of_rejects_out_of_bounds() {
    let g = fresh_grid();
    assert!(grid_index_of(Vec3::new(-0.1, 0.0, 0.0), &g).is_none());
    assert!(grid_index_of(Vec3::new(GRID_SIDE as f32 + 0.1, 0.0, 0.0), &g).is_none());
    assert!(grid_index_of(Vec3::new(0.0, -1.0, 0.0), &g).is_none());
    assert!(grid_index_of(Vec3::new(GRID_SIDE as f32, 0.0, 0.0), &g).is_none());
    assert!(grid_index_of(Vec3::new((GRID_SIDE - 1) as f32, 0.0, 0.0), &g).is_some());
}
