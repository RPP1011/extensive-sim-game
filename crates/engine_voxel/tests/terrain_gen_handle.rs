use engine_voxel::{TerrainGenHandle, TerrainGenError, VoxelTerrain};

fn gen_simple(_seed: u64) -> VoxelTerrain {
    let mut t = VoxelTerrain::with_extent(4);
    for x in 0..4 { for y in 0..4 { for z in 0..4 { t.set_cell(x, y, z, 1); } } }
    t
}

fn gen_panics(_seed: u64) -> VoxelTerrain {
    panic!("intentional panic for catch_unwind test")
}

#[test]
fn block_until_ready_returns_terrain() {
    let handle = TerrainGenHandle::spawn(42, gen_simple);
    let terrain = handle.block_until_ready().expect("ok");
    assert_eq!(terrain.cell_at(0, 0, 0), 1);
    assert_eq!(terrain.cell_at(3, 3, 3), 1);
}

#[test]
fn try_take_returns_none_then_some() {
    let mut handle = TerrainGenHandle::spawn(7, gen_simple);
    // Spin until the worker reports done. Bounded ~1s to avoid hang.
    let mut got = None;
    for _ in 0..1000 {
        if let Some(r) = handle.try_take() { got = Some(r); break; }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let terrain = got.expect("worker completed within bound").expect("ok");
    assert_eq!(terrain.cell_at(0, 0, 0), 1);
}

#[test]
fn panic_in_gen_surfaces_as_error_not_abort() {
    let handle = TerrainGenHandle::spawn(0, gen_panics);
    let err = handle.block_until_ready().expect_err("gen panicked");
    match err {
        TerrainGenError::Panicked(msg) => {
            assert!(msg.contains("intentional panic"), "got: {msg}");
        }
    }
}

#[test]
fn two_runs_same_seed_byte_identical_for_deterministic_gen() {
    let a = TerrainGenHandle::spawn(123, gen_simple).block_until_ready().unwrap();
    let b = TerrainGenHandle::spawn(123, gen_simple).block_until_ready().unwrap();
    for x in 0..4 { for y in 0..4 { for z in 0..4 {
        assert_eq!(a.cell_at(x, y, z), b.cell_at(x, y, z));
    }}}
}
