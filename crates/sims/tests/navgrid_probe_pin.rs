//! Phase 4b end-to-end pin for `navgrid.walkable(cx, cz)` — proves
//! the host-built navgrid actually drives a GPU-side physics rule.
//!
//! Topology:
//!   * 4×1×4 navgrid built host-side via `engine_voxel::build_navgrid`.
//!   * Wall column at (cx=2, cz=2) marked non-walkable in the
//!     host-built `NavgridIndex` (the column reaches the scan
//!     ceiling, so its top-of-column has no air-above → walkable=false).
//!   * 4 agents seeded at cells (0, 0), (1, 1), (3, 3), (2, 2). The
//!     first three sit on walkable floor cells; slot 3 sits on the
//!     wall column.
//!
//! Per-tick: the `NavgridProbe` physics rule reads
//! `navgrid.walkable(cx, cz)` at each agent's pos-derived cell and
//! stamps the result (1 or 0) into `agents.busy_with_ability_id`.
//! The pin reads the SoA back and asserts the kernel saw exactly
//! what the host uploaded — 1, 1, 1, 0.
//!
//! Outcome contract:
//!   1. `navgrid_buf` + `navgrid_cfg_buf` exist on `GeneratedRuntime`
//!      (compile-time check via field access).
//!   2. `upload_navgrid(&idx)` writes the cells + extent uniform.
//!   3. The kernel's `voxel_navgrid_walkable(cx, cz)` reads match
//!      the host-built `NavgridIndex::cell_at(cx, cz)`.

#![allow(non_snake_case)]

use sims::navgrid_probe::GeneratedRuntime;
use engine_voxel::{build_navgrid, Aabb, VoxelRegion, VoxelRegionBounds, VoxelRegionId, VoxelRegionKind};

const SEED: u64 = 0x4_AB1E_CE11_5_F00D;
const N: u32 = 4;
const GRID_SIZE: i32 = 4;
const WALL_CX: i32 = 2;
const WALL_CZ: i32 = 2;

#[test]
fn navgrid_walkable_reads_in_physics_rule_match_host_index() {
    let mut state = match GeneratedRuntime::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[navgrid_probe] skipping: no wgpu adapter");
            return;
        }
    };

    // --- Host-build the navgrid for a 4×1×4 region. Floor at y=0
    // everywhere; wall column at (x=WALL_CX, z=WALL_CZ) reaching
    // through the whole y range so build_navgrid's classify pass
    // marks the column non-walkable (top-of-column == scan_max_y - 1).
    let region = VoxelRegion {
        id: VoxelRegionId::from_raw_for_test(),
        bounds: VoxelRegionBounds::Aabb(Aabb {
            min: [0.0, 0.0, 0.0],
            max: [GRID_SIZE as f32, 4.0, GRID_SIZE as f32],
        }),
        kind: VoxelRegionKind(0),
        created_at_tick: 0,
    };
    let navgrid = build_navgrid(&region, |x, y, z| {
        // Ground at y=0, plus a wall column at (WALL_CX, *, WALL_CZ).
        y == 0 || (x == WALL_CX && z == WALL_CZ && (0..4).contains(&y))
    })
    .expect("build navgrid");

    println!(
        "[navgrid_probe] host navgrid: {}×{}, cells:",
        navgrid.size_x, navgrid.size_z,
    );
    for cz in 0..navgrid.size_z {
        let row: Vec<&str> = (0..navgrid.size_x)
            .map(|cx| {
                let cell = navgrid.cell_at(cx, cz).expect("cell in bounds");
                if cell.walkable() { "." } else { "#" }
            })
            .collect();
        println!("    cz={cz}: {}", row.join(""));
    }
    let host_walk: Vec<bool> = (0..N as usize)
        .map(|slot| {
            let (cx, cz) = slot_cell(slot);
            navgrid
                .cell_at(cx, cz)
                .map(|c| c.walkable())
                .unwrap_or(false)
        })
        .collect();
    println!("[navgrid_probe] host expects: {host_walk:?}");

    // --- Upload the host-built navgrid into the runtime's GPU buffer.
    state.upload_navgrid(&navgrid);

    // --- Seed agent positions matching the slot→cell mapping above.
    seed_topology(&mut state);

    // --- Step once. The NavgridProbe rule fires per-agent and stamps
    // `busy_with_ability_id` with 1 (walkable) or 0 (blocked).
    state.step();

    // --- Read back.
    let busy = read_busy_with_ability_id(&mut state);
    let gpu_walk: Vec<bool> = busy.iter().map(|&v| v != 0).collect();
    println!("[navgrid_probe] GPU saw:     {gpu_walk:?}");

    // --- Per-slot pins.
    for slot in 0..N as usize {
        assert_eq!(
            gpu_walk[slot], host_walk[slot],
            "slot {slot} at cell {:?}: GPU = {} but host navgrid = {}",
            slot_cell(slot),
            gpu_walk[slot],
            host_walk[slot],
        );
    }

    // Sanity: at least one walkable and one blocked across the run,
    // so we know the pin would actually fail if the kernel returned
    // a constant.
    assert!(
        gpu_walk.iter().any(|&w| w) && gpu_walk.iter().any(|&w| !w),
        "expected at least one walkable and one blocked outcome; got {gpu_walk:?}",
    );
}

fn slot_cell(slot: usize) -> (u32, u32) {
    // Slot → (cx, cz) mapping. Slots 0..2 land on walkable floor;
    // slot 3 sits on the wall column at (WALL_CX, WALL_CZ).
    match slot {
        0 => (0, 0),
        1 => (1, 1),
        2 => (3, 3),
        3 => (WALL_CX as u32, WALL_CZ as u32),
        _ => unreachable!(),
    }
}

fn seed_topology(state: &mut GeneratedRuntime) {
    let n = N as usize;
    // Position each slot at the centre of its assigned cell so the
    // NavgridProbe's u32(i32(pos.x)) cast resolves to the right cx/cz.
    let mut positions: Vec<[f32; 4]> = Vec::with_capacity(n);
    for slot in 0..n {
        let (cx, cz) = slot_cell(slot);
        positions.push([cx as f32 + 0.5, 0.0, cz as f32 + 0.5, 0.0]);
    }
    state.gpu.queue.write_buffer(
        &state.agent_pos_buf,
        0,
        bytemuck::cast_slice(&positions),
    );
    let alive: Vec<u32> = vec![1u32; n];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );
}

fn read_busy_with_ability_id(state: &mut GeneratedRuntime) -> Vec<u32> {
    let count = N as usize;
    let bytes = (count as u64) * 4;
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("navgrid_probe::busy_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("navgrid_probe::busy_readback"),
        },
    );
    encoder.copy_buffer_to_buffer(
        &state.agent_busy_with_ability_id_buf,
        0,
        &staging,
        0,
        bytes,
    );
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out: Vec<u32> = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}
