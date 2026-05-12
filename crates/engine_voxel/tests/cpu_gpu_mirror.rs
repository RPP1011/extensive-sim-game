//! CPU↔GPU divergence pins for the [`VoxelMirror`] surface and the
//! WGSL `voxel_line_of_sight` helper text. Salvaged from
//! `voxel_probe_runtime::voxel_pins::{cpu_gpu_voxel_state_matches,
//! gpu_terrain_query_matches_cpu}` when that crate was deleted as part
//! of the Plan E mega-crate sweep.
//!
//! The original tests sat on top of a hand-written `*_runtime` crate
//! whose only purpose was to drain voxel chronicle records on the
//! host. With that crate gone, the end-to-end "verb fires → chronicle
//! drain → terrain mutation" coverage is gone (it lives in git history
//! at `crates/voxel_probe_runtime`). What's preserved here is the
//! load-bearing CPU↔GPU divergence catcher: that the GPU mirror's
//! cell encoding and the WGSL `voxel_line_of_sight` body agree with
//! the CPU [`VoxelGrid`] + `voxel_engine::ray_cast_grid`. Those are
//! the two specific failure modes that aren't already covered by the
//! `engine_voxel` unit tests.
//!
//! These tests don't need a per-fixture runtime — they construct a
//! [`VoxelMirror`] directly via the public API, mutate cells through
//! `VoxelTerrain::set_cell` + `VoxelMirror::mark_dirty` +
//! `VoxelMirror::flush_dirty`, then run a tiny standalone compute
//! kernel against `mirror.buffer()` to read the GPU's view back.
//!
//! ## Inline WGSL source-of-truth
//!
//! The compute shaders below are verbatim copies of the helper text
//! the dsl_compiler emits into production kernels (the `voxel_at` +
//! `voxel_line_of_sight` body in
//! `dsl_compiler::cg::emit::program::VOXEL_GRID_WGSL_PRELUDE`). A
//! sister text-equality pin in `dsl_compiler` keeps the helper text
//! pinned so this test's local copy can't drift undetected. Don't
//! reformat or re-author this text — diff it byte-for-byte against
//! the prelude when bumping.

use engine::terrain::TerrainQuery;
use engine::GpuContext;
use engine_voxel::{VoxelMirror, VoxelTerrain};
use glam::{IVec3, Vec3};

const READBACK_WGSL: &str = r#"
struct DimsCfg {
    width: u32,
    height: u32,
    depth: u32,
    n: u32,
};
@group(0) @binding(0) var<storage, read> voxel_grid: array<u32>;
@group(0) @binding(1) var<storage, read> probe_cells: array<u32>;
@group(0) @binding(2) var<storage, read_write> probe_results: array<u32>;
@group(0) @binding(3) var<uniform> cfg: DimsCfg;

fn voxel_at(x: i32, y: i32, z: i32) -> u32 {
    if (x < 0 || y < 0 || z < 0) {
        return 0u;
    }
    let ux: u32 = u32(x);
    let uy: u32 = u32(y);
    let uz: u32 = u32(z);
    if (ux >= cfg.width || uy >= cfg.height || uz >= cfg.depth) {
        return 0u;
    }
    let idx: u32 = uz * cfg.height * cfg.width + uy * cfg.width + ux;
    return voxel_grid[idx];
}

@compute @workgroup_size(64)
fn cs_voxel_readback(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) {
        return;
    }
    let x = i32(probe_cells[i * 3u + 0u]);
    let y = i32(probe_cells[i * 3u + 1u]);
    let z = i32(probe_cells[i * 3u + 2u]);
    probe_results[i] = voxel_at(x, y, z);
}
"#;

const LOS_PROBE_WGSL: &str = r#"
struct LosCfg {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(0) var<storage, read> voxel_grid: array<u32>;
@group(0) @binding(1) var<storage, read> los_inputs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> los_results: array<u32>;
@group(0) @binding(3) var<uniform> cfg: LosCfg;

const VOXEL_GRID_DIM: u32 = 16u;

fn voxel_grid_dim() -> u32 {
    return VOXEL_GRID_DIM;
}

fn voxel_at(x: i32, y: i32, z: i32) -> u32 {
    if (x < 0 || y < 0 || z < 0) {
        return 0u;
    }
    let ux: u32 = u32(x);
    let uy: u32 = u32(y);
    let uz: u32 = u32(z);
    let dim: u32 = voxel_grid_dim();
    if (ux >= dim || uy >= dim || uz >= dim) {
        return 0u;
    }
    let idx: u32 = uz * dim * dim + uy * dim + ux;
    return voxel_grid[idx];
}

fn voxel_line_of_sight(seg_from: vec3<f32>, seg_to: vec3<f32>) -> bool {
    let delta: vec3<f32> = seg_to - seg_from;
    let length_sq: f32 = dot(delta, delta);
    if (length_sq < 1.0e-12) {
        return true;
    }
    let length: f32 = sqrt(length_sq);
    let dir: vec3<f32> = delta / length;

    var cx: i32 = i32(floor(seg_from.x));
    var cy: i32 = i32(floor(seg_from.y));
    var cz: i32 = i32(floor(seg_from.z));

    let step_x: i32 = select(select(0, -1, dir.x < 0.0), 1, dir.x > 0.0);
    let step_y: i32 = select(select(0, -1, dir.y < 0.0), 1, dir.y > 0.0);
    let step_z: i32 = select(select(0, -1, dir.z < 0.0), 1, dir.z > 0.0);

    let inv_dx: f32 = select(1.0e30, 1.0 / dir.x, abs(dir.x) > 1.0e-20);
    let inv_dy: f32 = select(1.0e30, 1.0 / dir.y, abs(dir.y) > 1.0e-20);
    let inv_dz: f32 = select(1.0e30, 1.0 / dir.z, abs(dir.z) > 1.0e-20);

    var t_max_x: f32 = 1.0e30;
    if (step_x > 0) {
        t_max_x = (f32(cx + 1) - seg_from.x) * inv_dx;
    } else if (step_x < 0) {
        t_max_x = (seg_from.x - f32(cx)) * (-inv_dx);
    }
    var t_max_y: f32 = 1.0e30;
    if (step_y > 0) {
        t_max_y = (f32(cy + 1) - seg_from.y) * inv_dy;
    } else if (step_y < 0) {
        t_max_y = (seg_from.y - f32(cy)) * (-inv_dy);
    }
    var t_max_z: f32 = 1.0e30;
    if (step_z > 0) {
        t_max_z = (f32(cz + 1) - seg_from.z) * inv_dz;
    } else if (step_z < 0) {
        t_max_z = (seg_from.z - f32(cz)) * (-inv_dz);
    }

    let t_delta_x: f32 = select(1.0e30, abs(inv_dx), step_x != 0);
    let t_delta_y: f32 = select(1.0e30, abs(inv_dy), step_y != 0);
    let t_delta_z: f32 = select(1.0e30, abs(inv_dz), step_z != 0);

    if (voxel_at(cx, cy, cz) != 0u) {
        return false;
    }

    let max_steps: u32 = 3u * voxel_grid_dim();
    var i: u32 = 0u;
    loop {
        if (i >= max_steps) {
            break;
        }
        var t_next: f32;
        if (t_max_x <= t_max_y && t_max_x <= t_max_z) {
            cx = cx + step_x;
            t_next = t_max_x;
            t_max_x = t_max_x + t_delta_x;
        } else if (t_max_y <= t_max_z) {
            cy = cy + step_y;
            t_next = t_max_y;
            t_max_y = t_max_y + t_delta_y;
        } else {
            cz = cz + step_z;
            t_next = t_max_z;
            t_max_z = t_max_z + t_delta_z;
        }
        if (t_next > length) {
            return true;
        }
        if (voxel_at(cx, cy, cz) != 0u) {
            return false;
        }
        i = i + 1u;
    }
    return true;
}

@compute @workgroup_size(64)
fn cs_los_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) {
        return;
    }
    let from_v: vec4<f32> = los_inputs[i * 2u];
    let to_v: vec4<f32>   = los_inputs[i * 2u + 1u];
    let seg_from = vec3<f32>(from_v.x, from_v.y, from_v.z);
    let seg_to   = vec3<f32>(to_v.x,   to_v.y,   to_v.z);
    let clear: bool = voxel_line_of_sight(seg_from, seg_to);
    los_results[i] = select(0u, 1u, clear);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DimsCfg {
    width: u32,
    height: u32,
    depth: u32,
    n: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LosCfg {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Read `probes.len()` cells from `mirror.buffer()` via a one-shot
/// compute dispatch. Returns the values the GPU sees at each `(x,y,z)`.
fn gpu_read_cells(
    gpu: &GpuContext,
    mirror: &VoxelMirror,
    probes: &[(i32, i32, i32)],
) -> Vec<u32> {
    use wgpu::util::DeviceExt;
    let n = probes.len() as u32;
    let mut cells_data: Vec<u32> = Vec::with_capacity(probes.len() * 3);
    for (x, y, z) in probes {
        cells_data.extend_from_slice(&[*x as u32, *y as u32, *z as u32]);
    }
    while cells_data.len() < 4 {
        cells_data.push(0);
    }
    let cells_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("engine_voxel_test::probe_cells"),
        contents: bytemuck::cast_slice(&cells_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let results_bytes = ((n as u64) * 4).max(16);
    let results_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine_voxel_test::probe_results"),
        size: results_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let (w, h, d) = mirror.dimensions();
    let cfg = DimsCfg { width: w, height: h, depth: d, n };
    let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("engine_voxel_test::dims_cfg"),
        contents: bytemuck::bytes_of(&cfg),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine_voxel_test::probe_results_staging"),
        size: results_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine_voxel_test::readback_shader"),
        source: wgpu::ShaderSource::Wgsl(READBACK_WGSL.into()),
    });
    let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("engine_voxel_test::readback_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("engine_voxel_test::readback_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("engine_voxel_test::readback_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("cs_voxel_readback"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("engine_voxel_test::readback_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: mirror.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: results_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: cfg_buf.as_entire_binding() },
        ],
    });

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("engine_voxel_test::readback_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine_voxel_test::readback_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let groups = (n + 63) / 64;
        pass.dispatch_workgroups(groups.max(1), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&results_buf, 0, &staging, 0, results_bytes);
    gpu.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    receiver.recv().expect("readback map result").expect("readback map ok");
    let mapped = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&mapped);
    let out: Vec<u32> = words[..n as usize].to_vec();
    drop(mapped);
    staging.unmap();
    out
}

/// Dispatch the inline LOS-probe kernel against `mirror.buffer()` for
/// each `(from, to)` pair. Returns the per-pair bool the GPU computed.
fn gpu_line_of_sight_batch(
    gpu: &GpuContext,
    mirror: &VoxelMirror,
    pairs: &[(Vec3, Vec3)],
) -> Vec<bool> {
    use wgpu::util::DeviceExt;
    let n = pairs.len() as u32;
    let mut packed: Vec<f32> = Vec::with_capacity((pairs.len() * 8).max(8));
    for (from, to) in pairs {
        packed.extend_from_slice(&[from.x, from.y, from.z, 0.0]);
        packed.extend_from_slice(&[to.x, to.y, to.z, 0.0]);
    }
    while packed.len() < 8 {
        packed.push(0.0);
    }
    let inputs_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("engine_voxel_test::los_inputs"),
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let results_bytes = ((n as u64) * 4).max(16);
    let results_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine_voxel_test::los_results"),
        size: results_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let cfg = LosCfg { n, _pad0: 0, _pad1: 0, _pad2: 0 };
    let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("engine_voxel_test::los_cfg"),
        contents: bytemuck::bytes_of(&cfg),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine_voxel_test::los_results_staging"),
        size: results_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine_voxel_test::los_probe_shader"),
        source: wgpu::ShaderSource::Wgsl(LOS_PROBE_WGSL.into()),
    });
    let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("engine_voxel_test::los_probe_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("engine_voxel_test::los_probe_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("engine_voxel_test::los_probe_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("cs_los_probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("engine_voxel_test::los_probe_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: mirror.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: inputs_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: results_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: cfg_buf.as_entire_binding() },
        ],
    });

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("engine_voxel_test::los_probe_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine_voxel_test::los_probe_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let groups = (n + 63) / 64;
        pass.dispatch_workgroups(groups.max(1), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&results_buf, 0, &staging, 0, results_bytes);
    gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    receiver.recv().expect("los map result").expect("los map ok");
    let mapped = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&mapped);
    let out: Vec<bool> = words[..n as usize].iter().map(|&w| w != 0).collect();
    drop(mapped);
    staging.unmap();
    out
}

/// Drive a deterministic cell-mutation pattern through `(terrain,
/// mirror)` so the GPU mirror sees the same byte encoding the CPU
/// `VoxelTerrain` stores. Then sample 10 cells across the 16³ extent
/// and compare CPU `cell_at` vs. GPU `voxel_at` at every probe.
///
/// Catches:
/// - `mark_dirty` mis-mapping cell → chunk.
/// - `flush_dirty` skipping a chunk (off-by-one in BTreeSet drain or
///   chunk-bounds calculation).
/// - The cell→flat-index encoding diverging between `VoxelGrid::index`
///   (host: `z*H*W + y*W + x`) and the WGSL `voxel_at` helper.
///
/// Don't substitute "voxel_count_on_gpu > 0" — that pin passes for
/// any non-empty mirror state regardless of which cells are populated.
#[test]
fn cpu_gpu_voxel_state_matches() {
    let gpu = match GpuContext::new_blocking() {
        Ok(g) => g,
        Err(_) => {
            eprintln!(
                "[engine_voxel cpu_gpu_voxel_state_matches] skipping: no wgpu adapter."
            );
            return;
        }
    };
    let mut terrain = VoxelTerrain::with_extent(16);
    let mut mirror = VoxelMirror::new(&gpu, terrain.grid());

    // Mutation pattern that mirrors what the deleted voxel_probe
    // chronicle drain produced over 30 ticks: place at origin, then
    // clear it, then place + leave it across a few non-origin cells.
    // The exact pattern doesn't matter — what matters is that we
    // dirty several chunks and exercise both place + clear.
    let writes: &[(u32, u32, u32, u8)] = &[
        (0, 0, 0, 0x79), // place at origin
        (1, 0, 0, 0),    // air (no-op write)
        (5, 5, 5, 0x42), // far chunk
        (8, 4, 2, 0x01), // another chunk
        (2, 8, 4, 0xFE), // another chunk
        (0, 0, 0, 0),    // clear origin (harvest equivalent)
    ];
    for (x, y, z, v) in writes {
        terrain.set_cell(*x, *y, *z, *v);
        mirror.mark_dirty(IVec3::new(*x as i32, *y as i32, *z as i32));
    }
    mirror.flush_dirty(&gpu, terrain.grid());

    let probes: Vec<(i32, i32, i32)> = vec![
        (0, 0, 0),    // place + clear target — final state: air
        (1, 0, 0),    // adjacent — air
        (0, 1, 0),
        (0, 0, 1),
        (3, 3, 3),    // far from probes — should be air
        (5, 5, 5),    // populated
        (15, 15, 15), // corner — should be air
        (8, 4, 2),    // populated
        (2, 8, 4),    // populated
        (4, 2, 8),    // air
    ];
    let gpu_values = gpu_read_cells(&gpu, &mirror, &probes);
    for (i, (x, y, z)) in probes.iter().enumerate() {
        let cpu_val = terrain.cell_at(*x, *y, *z) as u32;
        let gpu_val = gpu_values[i];
        assert_eq!(
            cpu_val, gpu_val,
            "CPU/GPU mirror divergence at cell ({x}, {y}, {z}): \
             CPU={cpu_val}, GPU={gpu_val}. Either `mark_dirty` mis-\
             mapped the cell to a chunk, `flush_dirty` skipped the \
             chunk, or the cell→flat-index encoding diverges between \
             host (z*H*W + y*W + x) and the WGSL `voxel_at` helper."
        );
    }
    eprintln!(
        "[engine_voxel] cpu_gpu_voxel_state_matches: {n} cells matched. \
         Mirror buffer = {bytes} B ({kib:.1} KiB).",
        n = probes.len(),
        bytes = mirror.buffer_bytes(),
        kib = mirror.buffer_bytes() as f64 / 1024.0,
    );
}

/// Place a deterministic mix of voxels (10 cells across the 16³
/// extent), then sample 8 (from, to) segments through the populated
/// region. For each segment, compare the GPU helper's
/// `voxel_line_of_sight(from, to)` answer to the CPU
/// `TerrainQuery::line_of_sight(from, to)`. CPU != GPU at any segment
/// fails the pin.
///
/// Catches:
/// - GPU `voxel_at` index formula drifts from
///   `VoxelGrid::index = z*H*W + y*W + x`.
/// - DDA step direction signs disagree (CPU uses voxel_engine's
///   `ray_cast_grid`; GPU uses the inline DDA helper).
/// - OOB defaults diverge (one returns false, the other true).
///
/// Don't substitute "GPU returns at least one true and one false" —
/// that pin passes for any helper that produces mixed output
/// regardless of cell-level correctness.
#[test]
fn gpu_terrain_query_matches_cpu() {
    let gpu = match GpuContext::new_blocking() {
        Ok(g) => g,
        Err(_) => {
            eprintln!(
                "[engine_voxel gpu_terrain_query_matches_cpu] skipping: no wgpu adapter."
            );
            return;
        }
    };
    let mut terrain = VoxelTerrain::with_extent(16);
    let mut mirror = VoxelMirror::new(&gpu, terrain.grid());

    // Wall-like cluster around y=5 covering x ∈ [3, 7].
    for x in 3u32..=7 {
        terrain.set_cell(x, 5, 5, 1);
    }
    // Vertical pillar at (10, 10).
    for z in 4u32..=6 {
        terrain.set_cell(10, 10, z, 1);
    }
    // Stray cells at corners.
    terrain.set_cell(2, 12, 8, 1);
    terrain.set_cell(14, 2, 12, 1);

    for (x, y, z) in [
        (3, 5, 5), (4, 5, 5), (5, 5, 5), (6, 5, 5), (7, 5, 5),
        (10, 10, 4), (10, 10, 5), (10, 10, 6),
        (2, 12, 8), (14, 2, 12),
    ] {
        mirror.mark_dirty(IVec3::new(x, y, z));
    }
    mirror.flush_dirty(&gpu, terrain.grid());

    // 8 deterministic (from, to) pairs spanning blocked + clear cases
    // through the populated region. Mix axis-aligned with diagonals so
    // the DDA exercises all three step directions.
    let pairs: Vec<(Vec3, Vec3)> = vec![
        (Vec3::new(0.5, 5.5, 5.5),  Vec3::new(15.5, 5.5, 5.5)),  // crosses wall — blocked
        (Vec3::new(0.5, 8.5, 5.5),  Vec3::new(15.5, 8.5, 5.5)),  // parallel ray — clear
        (Vec3::new(0.5, 10.5, 5.5), Vec3::new(15.5, 10.5, 5.5)), // through pillar — blocked
        (Vec3::new(1.5, 1.5, 0.5),  Vec3::new(1.5, 1.5, 15.5)),  // vertical clear — clear
        (Vec3::new(0.5, 0.5, 0.5),  Vec3::new(15.5, 15.5, 15.5)), // diagonal — depends
        (Vec3::new(13.5, 13.5, 1.5), Vec3::new(13.5, 13.5, 3.5)), // empty short — clear
        (Vec3::new(2.5, 12.5, 0.5), Vec3::new(2.5, 12.5, 15.5)), // through stray — blocked
        (Vec3::new(0.5, 2.5, 12.5), Vec3::new(15.5, 2.5, 12.5)), // through stray — blocked
    ];

    let cpu_results: Vec<bool> = pairs
        .iter()
        .map(|(from, to)| terrain.line_of_sight(*from, *to))
        .collect();
    let gpu_results = gpu_line_of_sight_batch(&gpu, &mirror, &pairs);

    let mut diffs = Vec::new();
    for (i, ((from, to), (cpu, gpu_v))) in pairs
        .iter()
        .zip(cpu_results.iter().zip(gpu_results.iter()))
        .enumerate()
    {
        if cpu != gpu_v {
            diffs.push(format!(
                "  pair[{i}]: from=({:.1},{:.1},{:.1}) to=({:.1},{:.1},{:.1}) \
                 CPU={cpu} GPU={gpu_v}",
                from.x, from.y, from.z, to.x, to.y, to.z
            ));
        }
    }
    assert!(
        diffs.is_empty(),
        "CPU/GPU terrain-query divergence on {n}/{tot} segments:\n{lines}\n\
         Either the GPU `voxel_line_of_sight` helper diverges from \
         voxel_engine's `ray_cast_grid` (DDA step direction, OOB \
         defaults, cell-floor semantics) OR the GPU mirror's cell \
         contents differ from the host VoxelGrid. The \
         cpu_gpu_voxel_state_matches pin catches the latter; this \
         pin catches the former.",
        n = diffs.len(),
        tot = pairs.len(),
        lines = diffs.join("\n"),
    );
    eprintln!(
        "[engine_voxel] gpu_terrain_query_matches_cpu: {n} segments \
         agreed (CPU == GPU). cpu={cpu_results:?}",
        n = pairs.len(),
    );
}
