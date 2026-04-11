//! Windowed voxel renderer for the world sim.
//!
//! Merges 4×4×4 sim chunks into 64³ mega-grid GPU textures to reduce draw calls.
//! Only uploads mega-chunks within camera radius. Uses GPU blit presentation.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::time::Instant;

use anyhow::{Context, Result};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use voxel_engine::camera::FreeCamera;
use voxel_engine::render::VoxelRenderer;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::swapchain::SwapchainContext;
use voxel_engine::terrain_compute::LoadedChunkView;
use voxel_engine::vulkan::voxel_gpu::{self, GpuVoxelTexture};
use voxel_engine::voxel::grid::VoxelGrid;

use super::runtime::WorldSim;
use super::voxel::{Chunk, ChunkPos, CHUNK_SIZE, CHUNK_VOLUME};
use super::voxel_bridge::VoxelBridge;

use super::constants::{MEGA, MEGA_VOXELS, LOAD_RADIUS, RENDER_WIDTH, RENDER_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT};

// ---------------------------------------------------------------------------
// MegaChunkPos — groups 4×4×4 sim chunks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MegaPos {
    x: i32,
    y: i32,
    z: i32,
}

impl MegaPos {
    fn from_chunk(cp: ChunkPos) -> Self {
        Self {
            x: cp.x.div_euclid(MEGA),
            y: cp.y.div_euclid(MEGA),
            z: cp.z.div_euclid(MEGA),
        }
    }

    /// World-space center of this mega-chunk (in engine coords: x, y-up, z).
    fn world_center(&self) -> [f32; 3] {
        let half = (MEGA_VOXELS as f32) / 2.0;
        [
            self.x as f32 * MEGA_VOXELS as f32 + half,
            self.z as f32 * MEGA_VOXELS as f32 + half, // sim z → engine y
            self.y as f32 * MEGA_VOXELS as f32 + half, // sim y → engine z
        ]
    }

    /// World-space position (min corner) in engine coords.
    fn world_position(&self) -> [f32; 3] {
        [
            (self.x * MEGA * CHUNK_SIZE as i32) as f32,
            (self.z * MEGA * CHUNK_SIZE as i32) as f32,
            (self.y * MEGA * CHUNK_SIZE as i32) as f32,
        ]
    }
}

// ---------------------------------------------------------------------------
// GPU mega-chunk
// ---------------------------------------------------------------------------

struct GpuMega {
    texture: GpuVoxelTexture,
    position: [f32; 3],
}

// ---------------------------------------------------------------------------
// AppState
// ---------------------------------------------------------------------------

struct AppState {
    window: Window,
    ctx: VulkanContext,
    alloc: VulkanAllocator,
    swapchain: SwapchainContext,
    renderer: VoxelRenderer,
    camera: FreeCamera,

    bridge: VoxelBridge,
    sim: WorldSim,

    /// GPU compute pipeline for on-demand chunk terrain generation.
    terrain_compute: voxel_engine::terrain_compute::TerrainComputePipeline,

    gpu_megas: HashMap<MegaPos, GpuMega>,
    /// Tracks which mega-chunks have been dirtied since last upload.
    dirty_megas: HashSet<MegaPos>,

    paused: bool,
    /// When true, each phase of run_frame is individually timed. Off by
    /// default — the 14 Instant::now() calls cost ~2 µs/frame at 230 k+
    /// FPS, which is ~half the frame. Set VOXEL_PERF_DETAILED=1 to
    /// re-enable per-phase timings for debugging.
    detailed_perf: bool,
    /// Sim-speed multiplier. ticks_per_frame == 1 means the sim ticks at
    /// SIM_BASE_HZ; doubling it doubles the sim-time rate (1 real second
    /// → 2 sim seconds of advancement). Despite the legacy name, it no
    /// longer corresponds to render frames — the sim is now decoupled
    /// from the render loop via `sim_accumulator`.
    ticks_per_frame: u32,
    last_frame: Instant,
    /// Wall-clock leftover since the last `sim.tick()` call. Accumulates
    /// the render frame `dt`; when it exceeds `SIM_BASE_DT / ticks_per_frame`
    /// we advance the sim and subtract. Lets the sim run at a fixed
    /// wall-clock rate regardless of render FPS.
    sim_accumulator: f32,

    keys_held: HashSet<KeyCode>,
    mouse_captured: bool,
    last_mouse: Option<(f64, f64)>,
    move_speed: f32,

    // FPS tracking
    frame_count: u32,
    fps_timer: Instant,
    last_fps: f32,
    last_visible_megas: usize,
    settlement_jump_idx: usize,

    // Pool throughput diagnostics (reset each second in the perf log).
    gen_submitted_this_sec: u32,
    drain_completed_this_sec: u32,
    gen_short_circuit_this_sec: u32,

    /// (eye_x, eye_y, eye_z, fwd_x, fwd_y, fwd_z) of the camera at the end of
    /// the last generate_camera_chunks call that submitted 0 chunks. If the
    /// camera hasn't moved since and the pool has no in-flight dispatches,
    /// gen short-circuits immediately — the spiral has already converged and
    /// will just find dedup no-ops for the entire 13-chunk radius.
    last_gen_converged_cam: Option<[f32; 6]>,

    /// Reused per-frame buffers for the render's visible-chunk list + the
    /// stripped version passed to `render_frame_pool`. Pre-allocated to
    /// avoid heap churn on the 5000 FPS hot path (the old code was
    /// allocating two ~256-entry Vecs every single frame).
    visible_buf: Vec<(LoadedChunkView, [f32; 4], [f32; 3], [f32; 3], f32)>,
    pool_views_buf: Vec<(LoadedChunkView, [f32; 4], [f32; 3], [f32; 3])>,
    /// Cull-cache key: `(camera eye+center, terrain_compute.pool_generation())`
    /// at the time `pool_views_buf` was last rebuilt. When both components
    /// match the current frame, `pool_views_buf` is still correct and the
    /// whole cull loop (frustum test + sort + mark_touched copy) is
    /// skipped.
    last_cull_cam_key: Option<[f32; 6]>,
    last_cull_pool_gen: u64,

    // Per-phase timing EMAs (exponential moving avg, ms). Alpha=0.1.
    ema_drain_cpu_ms: f32,
    ema_drain_gpu_ms: f32,
    ema_gen_ms: f32,
    ema_update_cam_ms: f32,
    ema_tick_sim_ms: f32,
    ema_render_ms: f32,
    ema_cull_ms: f32,       // sub-phase of render: frustum cull + sort
    ema_wait_ms: f32,       // sub-phase of render: CPU wait for previous GPU frame
    ema_raycast_ms: f32,    // sub-phase of render: record/submit gbuffer+shadow+light (no wait)
    ema_present_ms: f32,    // sub-phase of render: present_blit
    ema_frame_ms: f32,

    // Background chunk generation
    chunk_rx: mpsc::Receiver<(ChunkPos, Chunk)>,
    chunks_pending: usize,
    chunks_loaded: usize,

    /// Maps an in-flight GPU dispatch request id → target ChunkPos, so we can
    /// reconstruct the chunk once the dispatch completes (Task 14).
    pending_chunk_requests: HashMap<u64, ChunkPos>,
}

impl AppState {
    fn new(window: Window, sim: WorldSim) -> Result<Self> {
        let t0 = Instant::now();
        let ctx = VulkanContext::new_with_surface_extensions(&window)?;
        eprintln!("[voxel] Vulkan context: {:.1}ms", t0.elapsed().as_secs_f32() * 1000.0);
        let mut alloc = VulkanAllocator::new(&ctx)?;
        let mut swapchain = SwapchainContext::new(&ctx, &window)?;
        let renderer = VoxelRenderer::new(&ctx, RENDER_WIDTH, RENDER_HEIGHT)?;
        eprintln!("[voxel] Renderer ready: {:.1}ms total", t0.elapsed().as_secs_f32() * 1000.0);

        // Pre-record the blit command buffers for each swapchain image now
        // that we know the renderer's output image handle. present_blit
        // will use the pre-recorded fast path instead of re-recording the
        // same barrier/blit/barrier sequence every frame.
        swapchain.prepare_blit_commands(
            &ctx,
            renderer.gbuffer_output_image(),
            RENDER_WIDTH,
            RENDER_HEIGHT,
        )?;

        // Show a blank frame immediately so the window isn't frozen.
        let _ = swapchain.present_cleared_frame(&ctx, [0.05, 0.05, 0.08, 1.0]);

        let bridge = VoxelBridge::new();

        // Pre-generate terrain only around settlements (small radius for sim queries).
        // Rendering chunks are generated on-demand as the camera moves.
        let state = sim.state();
        let seed = state.rng_state;
        let plan = state.voxel_world.region_plan.clone();

        // Create GPU terrain compute pipeline and upload region plan + rivers.
        let mut terrain_compute =
            voxel_engine::terrain_compute::TerrainComputePipeline::new(&ctx, &mut alloc)
                .context("create terrain compute pipeline")?;
        if let Some(p) = plan.as_ref() {
            let gpu_cells = p.to_gpu_cells();
            terrain_compute
                .upload_region_plan(
                    &ctx,
                    &mut alloc,
                    p.cols as u32,
                    p.rows as u32,
                    crate::world_sim::terrain::CELL_SIZE as u32,
                    &gpu_cells,
                )
                .context("upload region plan")?;
            let (river_points, river_headers) = p.to_gpu_rivers();
            terrain_compute
                .upload_rivers(&ctx, &mut alloc, &river_points, &river_headers)
                .context("upload rivers")?;
            eprintln!(
                "[voxel] GPU terrain compute initialized: {} cells, {} rivers",
                gpu_cells.len(),
                river_headers.len()
            );
        }

        // Upload the shared palette to the compute pool once. The pool render
        // entry point samples this for every chunk draw; no per-mega palette
        // uploads on the hot path.
        {
            let palette_rgba = bridge.palette_rgba();
            terrain_compute
                .upload_palette(&ctx, &mut alloc, palette_rgba)
                .context("upload palette to pool")?;
        }

        let mut settlement_chunks: Vec<ChunkPos> = Vec::new();
        if let Some(ref plan) = plan {
            use crate::world_sim::terrain::MAX_SURFACE_Z;
            let cs = CHUNK_SIZE as f32;
            let radius = 3i32; // 3 chunks each direction around each settlement

            for settlement in &state.settlements {
                // Settlement pos is already in voxel space.
                let (sx, sy) = settlement.pos;
                let h = plan.interpolate_height(sx, sy);
                let surface_cz = (h * MAX_SURFACE_Z as f32) as i32 / CHUNK_SIZE as i32;
                let center_cx = (sx / cs).floor() as i32;
                let center_cy = (sy / cs).floor() as i32;

                for dx in -radius..=radius {
                    for dy in -radius..=radius {
                        for dz in -2..=2 {
                            let cp = ChunkPos::new(center_cx + dx, center_cy + dy, surface_cz + dz);
                            settlement_chunks.push(cp);
                        }
                    }
                }
            }
            settlement_chunks.sort_by(|a, b| {
                (a.x, a.y, a.z).cmp(&(b.x, b.y, b.z))
            });
            settlement_chunks.dedup();
        }

        let total_settlement_chunks = settlement_chunks.len();
        eprintln!("[voxel] Pre-generating {} chunks around {} settlements",
            total_settlement_chunks, state.settlements.len());

        // Spawn background thread for settlement chunk pre-generation.
        let (chunk_tx, chunk_rx) = mpsc::channel::<(ChunkPos, Chunk)>();
        if let Some(plan_clone) = plan.clone() {
            std::thread::spawn(move || {
                for (i, cp) in settlement_chunks.iter().enumerate() {
                    let chunk = crate::world_sim::terrain::materialize_chunk(*cp, &plan_clone, seed);
                    if chunk_tx.send((*cp, chunk)).is_err() {
                        break;
                    }
                    if (i + 1) % 200 == 0 {
                        eprintln!("[voxel] Pre-gen {}/{}", i + 1, total_settlement_chunks);
                    }
                }
                eprintln!("[voxel] Settlement pre-gen complete: {} chunks", total_settlement_chunks);
            });
        }

        // Camera placement: pick a scenic settlement (prefer Forest/Jungle for
        // visual interest — lots of trees and varied terrain). Fall back to
        // Plains, then Mountains, then whatever is first.
        //
        // First-person height (~1.8m = 18 voxels at 10cm/voxel) with a slight
        // downward tilt so the horizon is visible.
        //
        // Plan-based surface_height_at (analytical, no voxel data required).
        let plan = sim.state().voxel_world.region_plan.clone();
        let world_seed = sim.state().rng_state;
        let (cam_pos, cam_target) = if let (Some(plan_ref), settlements) =
            (plan.as_ref(), &sim.state().settlements)
        {
            use crate::world_sim::state::Terrain;
            // Score settlements by biome interestingness. Higher = better.
            let score_biome = |t: Terrain| -> i32 {
                match t {
                    Terrain::Forest => 100,
                    Terrain::Jungle => 95,
                    Terrain::Plains => 80,
                    Terrain::Badlands => 70,
                    Terrain::Swamp => 65,
                    Terrain::Tundra => 60,
                    Terrain::Mountains => 50,
                    Terrain::Desert => 55,
                    _ => 20,
                }
            };
            // Look up each settlement's terrain via the region plan.
            let best = settlements.iter().max_by_key(|s| {
                let (cell, _, _) = plan_ref.sample(s.pos.0, s.pos.1);
                score_biome(cell.terrain)
            });
            match best {
                Some(s) => {
                    let vx = s.pos.0;
                    let vy = s.pos.1;
                    let (cell, _, _) = plan_ref.sample(vx, vy);
                    let surface_z = crate::world_sim::terrain::surface_height_at(vx, vy, plan_ref, world_seed);
                    let eye = glam::Vec3::new(vx, (surface_z + 18) as f32, vy);
                    let target = eye + glam::Vec3::new(0.0, -3.0, 30.0);
                    eprintln!("[voxel] Camera at settlement '{}' ({:?}) surface_z={} eye_y={}",
                        s.name, cell.terrain, surface_z, surface_z + 18);
                    (eye, target)
                }
                None => (glam::Vec3::new(0.0, 500.0, 0.0), glam::Vec3::new(0.0, 497.0, 30.0)),
            }
        } else {
            (glam::Vec3::new(0.0, 500.0, 0.0), glam::Vec3::new(0.0, 497.0, 30.0))
        };
        let mut camera = FreeCamera::new(cam_pos, cam_target);
        camera.set_move_speed(50.0);

        Ok(Self {
            window, ctx, alloc, swapchain, renderer, camera,
            bridge, sim,
            terrain_compute,
            gpu_megas: HashMap::new(),
            dirty_megas: HashSet::new(),
            paused: true, // start paused so sim doesn't run during chunk loading
            detailed_perf: std::env::var("VOXEL_PERF_DETAILED")
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(false),
            ticks_per_frame: 10,
            sim_accumulator: 0.0,
            last_frame: Instant::now(),
            keys_held: HashSet::new(),
            mouse_captured: false,
            last_mouse: None,
            move_speed: 50.0,
            frame_count: 0,
            fps_timer: Instant::now(),
            last_fps: 0.0,
            last_visible_megas: 0,
            settlement_jump_idx: 0,
            gen_submitted_this_sec: 0,
            drain_completed_this_sec: 0,
            gen_short_circuit_this_sec: 0,
            chunk_rx,
            chunks_pending: total_settlement_chunks,
            chunks_loaded: 0,
            pending_chunk_requests: HashMap::new(),
            last_gen_converged_cam: None,
            visible_buf: Vec::with_capacity(320),
            pool_views_buf: Vec::with_capacity(320),
            last_cull_cam_key: None,
            last_cull_pool_gen: 0,
            ema_drain_cpu_ms: 0.0,
            ema_drain_gpu_ms: 0.0,
            ema_gen_ms: 0.0,
            ema_update_cam_ms: 0.0,
            ema_tick_sim_ms: 0.0,
            ema_render_ms: 0.0,
            ema_cull_ms: 0.0,
            ema_wait_ms: 0.0,
            ema_raycast_ms: 0.0,
            ema_present_ms: 0.0,
            ema_frame_ms: 0.0,
        })
    }

    /// Build a 64³ mega-grid from up to 4×4×4 sim chunks.
    fn build_mega_grid(&self, mp: MegaPos) -> Option<VoxelGrid> {
        let world = &self.sim.state().voxel_world;
        let mut any_solid = false;
        let mut all_full = true;
        let mut all_loaded = true;

        for dz in 0..MEGA {
            for dy in 0..MEGA {
                for dx in 0..MEGA {
                    let cp = ChunkPos::new(mp.x * MEGA + dx, mp.y * MEGA + dy, mp.z * MEGA + dz);
                    match world.chunks.get(&cp) {
                        Some(c) => {
                            if !any_solid {
                                any_solid = c.voxels.iter().any(|v| v.material.is_solid());
                            }
                            if all_full && c.voxels.iter().any(|v| !v.material.is_solid()) {
                                all_full = false;
                            }
                        }
                        None => { all_loaded = false; all_full = false; }
                    }
                }
            }
        }
        if !any_solid { return None; }
        let all_solid = all_full && all_loaded;

        // Check neighbor mega-chunks for occlusion.
        if all_solid {
            let neighbors_solid = [
                MegaPos { x: mp.x - 1, y: mp.y, z: mp.z },
                MegaPos { x: mp.x + 1, y: mp.y, z: mp.z },
                MegaPos { x: mp.x, y: mp.y - 1, z: mp.z },
                MegaPos { x: mp.x, y: mp.y + 1, z: mp.z },
                MegaPos { x: mp.x, y: mp.y, z: mp.z - 1 },
                MegaPos { x: mp.x, y: mp.y, z: mp.z + 1 },
            ].iter().all(|nmp| {
                (0..MEGA).all(|dz| (0..MEGA).all(|dy| (0..MEGA).all(|dx| {
                    let cp = ChunkPos::new(nmp.x * MEGA + dx, nmp.y * MEGA + dy, nmp.z * MEGA + dz);
                    world.chunks.get(&cp).map_or(false, |c| c.solid_count() == CHUNK_VOLUME)
                })))
            });
            if neighbors_solid { return None; }
        }

        let mut grid = VoxelGrid::new(MEGA_VOXELS, MEGA_VOXELS, MEGA_VOXELS);

        for dz in 0..MEGA {
            for dy in 0..MEGA {
                for dx in 0..MEGA {
                    let cp = ChunkPos::new(mp.x * MEGA + dx, mp.y * MEGA + dy, mp.z * MEGA + dz);
                    let chunk = match world.chunks.get(&cp) {
                        Some(c) => c,
                        None => continue,
                    };

                    let base_x = (dx * CHUNK_SIZE as i32) as u32;
                    let base_y = (dy * CHUNK_SIZE as i32) as u32;
                    let base_z = (dz * CHUNK_SIZE as i32) as u32;

                    for lz in 0..CHUNK_SIZE {
                        for ly in 0..CHUNK_SIZE {
                            for lx in 0..CHUNK_SIZE {
                                let voxel = chunk.get(lx, ly, lz);
                                let mat = voxel.material as u8;
                                if mat != 0 {
                                    // Swap Y↔Z for engine coords.
                                    grid.set(
                                        base_x + lx as u32,
                                        base_z + lz as u32, // sim z → engine y
                                        base_y + ly as u32, // sim y → engine z
                                        mat,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        Some(grid)
    }

    /// Drain ready chunks from the background generation thread.
    /// Processes up to `budget` chunks per call to avoid stalling the frame.
    fn drain_ready_chunks(&mut self, budget: usize) {
        let mut count = 0;
        while count < budget {
            match self.chunk_rx.try_recv() {
                Ok((cp, chunk)) => {
                    self.sim.state_mut().voxel_world.chunks.insert(cp, chunk);
                    self.dirty_megas.insert(MegaPos::from_chunk(cp));
                    self.chunks_loaded += 1;
                    count += 1;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
    }

    /// Submit chunks near the camera to the GPU terrain compute pipeline.
    /// Dispatches are asynchronous (Task 14): we submit up to `budget` new
    /// chunks per call, each consuming one free slot in the compute pipeline's
    /// in-flight ring. Completed chunks are picked up by
    /// [`drain_completed_gpu_chunks`] on a later frame.
    ///
    /// To avoid saturating the GPU, we also cap the total number of in-flight
    /// dispatches (MAX_INFLIGHT). With the current halo-based feature
    /// stamping the shader is expensive, and submitting hundreds of
    /// dispatches at once means none complete in any reasonable time —
    /// present_blit stalls behind the compute queue backlog and the frame
    /// rate collapses.
    fn generate_camera_chunks(&mut self, budget: usize) {
        // Cap in-flight compute dispatches. With the render→present semaphore
        // handoff (voxel_engine 4364161) graphics no longer CPU-blocks on
        // compute, so this cap just throttles how fast we grow the queue.
        // Bumped from 4 → 32: with the old cap, the pool filled at ~2
        // chunks/sec because the queue was drained before gen could refill
        // it; the new cap lets us keep the compute queue saturated while
        // still bounding the backlog.
        const MAX_INFLIGHT: usize = 32;

        let (_free, in_flight, _loaded) = self.terrain_compute.pool_stats();
        if in_flight >= MAX_INFLIGHT {
            self.gen_short_circuit_this_sec += 1;
            return;
        }

        // Converged-spiral short-circuit — FIRST, before any expensive
        // work. Once the pool has been filled with every in-frustum
        // chunk the spiral can find, re-running the spiral every frame
        // for the same camera pose produces zero new submissions, so
        // nothing matters except the camera + in_flight state. Compare
        // directly against cached engine-space eye + center (6 floats)
        // so we can bail before cloning the region plan.
        //
        // The previous ordering did: pool_stats() → plan.clone() →
        // axis-swap → forward-vector sqrt → short-circuit check. The
        // plan.clone() deep-copies three large Vecs (cells/rivers/roads)
        // on every frame at ~250 k FPS, which was the single biggest
        // hidden cost in run_frame's `other` bucket.
        let cam = self.camera.eye_position();
        let cam_center_v = self.camera.center();
        let cam_key = [
            cam[0], cam[1], cam[2],
            cam_center_v.x, cam_center_v.y, cam_center_v.z,
        ];
        if in_flight == 0
            && self.last_gen_converged_cam.map(|c| c == cam_key).unwrap_or(false)
        {
            self.gen_short_circuit_this_sec += 1;
            return;
        }

        // Slow path: the spiral might actually submit something, so we
        // need the region plan and the sim-space conversions.
        let budget = budget.min(MAX_INFLIGHT - in_flight);
        let has_plan = self.sim.state().voxel_world.region_plan.is_some();
        static LOGGED_PLAN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED_PLAN.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("[voxel] generate_camera_chunks: has_plan={}", has_plan);
        }
        let plan = match &self.sim.state().voxel_world.region_plan {
            Some(p) => p.clone(),
            None => return,
        };
        let seed = self.sim.state().rng_state;
        // Convert engine coords (x, y-up, z) back to sim coords (x, y, z-up).
        let cam_vx = cam[0];
        let cam_vy = cam[2]; // engine z → sim y
        let cam_vz = cam[1]; // engine y → sim z

        // Camera forward in sim coords (same axis swap).
        let fwd_raw = [cam_center_v.x - cam[0], cam_center_v.y - cam[1], cam_center_v.z - cam[2]];
        let fwd_len2 = fwd_raw[0] * fwd_raw[0] + fwd_raw[1] * fwd_raw[1] + fwd_raw[2] * fwd_raw[2];
        let (fwd_vx, fwd_vy, fwd_vz) = if fwd_len2 > 1e-6 {
            let inv = 1.0 / fwd_len2.sqrt();
            // engine (x, y-up, z) → sim (x, z, y)
            (fwd_raw[0] * inv, fwd_raw[2] * inv, fwd_raw[1] * inv)
        } else {
            (1.0, 0.0, 0.0)
        };

        // One-time debug: log camera position and what biome we're looking at
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let (cell, _, _) = plan.sample(cam_vx, cam_vy);
            let surface = crate::world_sim::terrain::surface_height_at(cam_vx, cam_vy, &plan, seed);
            eprintln!("[voxel] Camera sim coords: ({:.0}, {:.0}, {:.0}), surface_z={}, terrain={:?}, height={:.3}",
                cam_vx, cam_vy, cam_vz, surface, cell.terrain, cell.height);
        }

        let cs = CHUNK_SIZE as f32;
        let center_cx = (cam_vx / cs).floor() as i32;
        let center_cy = (cam_vy / cs).floor() as i32;
        let center_cz = (cam_vz / cs).floor() as i32;

        // Radius in chunks to check around camera.
        let radius = (LOAD_RADIUS / (MEGA as f32 * cs)).ceil() as i32 + 1;

        // Build the render frustum planes once so the spiral can reject
        // candidates that won't pass the render-side frustum test anyway.
        // Previously we only dropped chunks more than ~107° off-axis via a
        // forward-alignment dot product, which was way wider than the
        // render frustum (~90° × 45°). The spiral was happily submitting
        // ~75% of all chunks within LOAD_RADIUS, most of which would never
        // be visible. Once the pool filled, LRU eviction kept churning
        // these off-screen chunks — compute stayed 100% busy, hogging the
        // GPU SMs and starving the render queue.
        //
        // With real frustum culling here + in the render path, a
        // stationary camera reaches a steady state where all in-frustum
        // chunks are loaded, the spiral finds nothing new to submit,
        // compute goes idle, and graphics gets the full GPU.
        let aspect = RENDER_WIDTH as f32 / RENDER_HEIGHT as f32;
        let vp = frustum_vp_matrix(&self.camera, aspect);
        let frustum_planes = extract_frustum_planes(&vp);

        let mut submitted = 0;
        let mut skipped_behind = 0u32;
        let _ = fwd_vx; let _ = fwd_vy; let _ = fwd_vz; // superseded by frustum
        // Spiral outward from camera for priority ordering.
        'outer: for dist in 0..=radius {
            for dx in -dist..=dist {
                for dy in -dist..=dist {
                    // Narrow vertical range: the old -2..=2 spent 40% of each
                    // shell's budget on chunks 128 voxels above/below the
                    // camera — subterranean rock or empty sky that's almost
                    // never visible from a surface camera. -1..=1 covers the
                    // camera chunk plus one above/below for cave/aerial
                    // peeks while freeing 40% of the compute budget for
                    // same-layer forward chunks.
                    for dz in -1..=1 {
                        if submitted >= budget { break 'outer; }
                        // Only process shell of current distance.
                        if dx.abs() != dist && dy.abs() != dist { continue; }

                        // Frustum cull ALL chunks — including shells 0-1.
                        // Previously shells 0-1 were always submitted as
                        // insurance for camera rotation, but on a
                        // stationary camera those non-visible shell-0/1
                        // chunks were getting LRU-churned every frame:
                        // submitted → evicted → re-submitted → evicted →
                        // ... keeping the compute queue at 100% even
                        // though the pool already held every in-frustum
                        // chunk the render needed. The camera chunk
                        // itself (dist=0) always passes frustum because
                        // the camera is inside its AABB.
                        //
                        // Sim chunk (cx, cy, cz) → engine-space AABB.
                        // Engine coords: sim.x → eng.x, sim.z → eng.y (up),
                        // sim.y → eng.z. So a chunk spanning sim
                        // (cx..cx+1, cy..cy+1, cz..cz+1) becomes engine
                        // (cx..cx+1, cz..cz+1, cy..cy+1).
                        {
                            let scx = center_cx + dx;
                            let scy = center_cy + dy;
                            let scz = center_cz + dz;
                            let min = [
                                scx as f32 * cs,
                                scz as f32 * cs,
                                scy as f32 * cs,
                            ];
                            let max = [min[0] + cs, min[1] + cs, min[2] + cs];
                            if !aabb_vs_frustum(&frustum_planes, &min, &max) {
                                skipped_behind += 1;
                                continue;
                            }
                        }

                        let cp = ChunkPos::new(center_cx + dx, center_cy + dy, center_cz + dz);
                        // NOTE: we intentionally do NOT skip chunks that exist
                        // in sim.voxel_world.chunks — those are inserted by the
                        // settlement pre-gen CPU thread for simulation queries,
                        // but the GPU pool is a separate rendering cache.
                        // Phase 3 removed the upload_megas path, so without
                        // submitting here the renderer sees nothing for
                        // settlement areas. submit_chunk itself is a no-op for
                        // chunks already Loaded or InFlight in the pool.
                        if self.pending_chunk_requests.values().any(|p| *p == cp) { continue; }

                        match self.terrain_compute.submit_chunk_with_frame(
                            &self.ctx,
                            [cp.x, cp.y, cp.z],
                            seed as u32,
                            self.frame_count as u64,
                        ) {
                            Ok(Some(req_id)) => {
                                self.pending_chunk_requests.insert(req_id, cp);
                                submitted += 1;
                                self.gen_submitted_this_sec += 1;
                            }
                            Ok(None) => {
                                // Chunk is already Loaded or InFlight in the
                                // pool — deduplication, not a failure. Skip
                                // and keep scanning for new chunks to submit.
                                // (Also returned when all 256 slots are
                                // InFlight, which is rare; the next frame
                                // will drain some and retry.)
                                continue;
                            }
                            Err(e) => {
                                eprintln!("[voxel] submit_chunk failed for {:?}: {}", cp, e);
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        let _ = skipped_behind; // currently unused; could surface in perf log

        // If the spiral found nothing new to submit, remember the current
        // camera state so the next frame can skip the whole spiral if the
        // camera hasn't moved. Note we only mark convergence when the
        // spiral ran to completion (not when it bailed early via
        // `break 'outer` from an error) — `submitted == 0` at this point
        // implies a complete sweep that couldn't find anything.
        if submitted == 0 {
            self.last_gen_converged_cam = Some(cam_key);
        } else {
            self.last_gen_converged_cam = None;
        }
    }

    /// Submit chunks in a bulk radius around the camera. Invoked by the F key.
    /// For each (cx, cy) column in the horizontal radius, submits chunks at
    /// the actual *surface* vertical position (not the camera's), plus one
    /// above and one below. This keeps the total submission count small
    /// enough that the pool's 256-slot capacity isn't overrun — which would
    /// otherwise LRU-evict currently-visible chunks and produce visible
    /// occlusion while the new dispatches complete.
    ///
    /// At `radius = 4` this submits at most `(2*4+1)^2 * 3 = 243` chunks,
    /// just under the 256-slot pool size. Already-loaded chunks return
    /// `Ok(None)` from `submit_chunk` and aren't counted as new submissions.
    fn fill_chunks_around_camera(&mut self, radius: i32) -> usize {
        let plan = match &self.sim.state().voxel_world.region_plan {
            Some(p) => p.clone(),
            None => return 0,
        };
        let seed = self.sim.state().rng_state;
        let cam = self.camera.eye_position();
        let cam_vx = cam[0];
        let cam_vy = cam[2]; // engine z → sim y
        let cs = CHUNK_SIZE as f32;
        let center_cx = (cam_vx / cs).floor() as i32;
        let center_cy = (cam_vy / cs).floor() as i32;

        let mut submitted = 0usize;
        // Spiral outward so closer chunks come first (better LRU ordering).
        'outer: for dist in 0..=radius {
            for dx in -dist..=dist {
                for dy in -dist..=dist {
                    // Shell filter (only process cells at the current dist).
                    if dx.abs() != dist && dy.abs() != dist { continue; }

                    let cx = center_cx + dx;
                    let cy = center_cy + dy;

                    // Per-column surface → chunk z. Load surface_cz ± 1 so
                    // tree canopies above the surface and the underlying
                    // rock just below are covered.
                    let col_vx = (cx as f32 + 0.5) * cs;
                    let col_vy = (cy as f32 + 0.5) * cs;
                    let surface_z = crate::world_sim::terrain::surface_height_at(
                        col_vx, col_vy, &plan, seed,
                    );
                    let surface_cz = surface_z.div_euclid(CHUNK_SIZE as i32);

                    for dz in -1..=1 {
                        let cp = ChunkPos::new(cx, cy, surface_cz + dz);
                        if self.pending_chunk_requests.values().any(|p| *p == cp) { continue; }

                        match self.terrain_compute.submit_chunk(
                            &self.ctx,
                            [cp.x, cp.y, cp.z],
                            seed as u32,
                        ) {
                            Ok(Some(req_id)) => {
                                self.pending_chunk_requests.insert(req_id, cp);
                                submitted += 1;
                            }
                            Ok(None) => {
                                // Already loaded in the pool — skip.
                            }
                            Err(e) => {
                                eprintln!("[voxel] fill submit_chunk failed for {:?}: {}", cp, e);
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        submitted
    }

    /// Poll the GPU terrain compute pipeline for completed dispatches.
    /// Phase 3: the chunk texture stays GPU-resident in the pool, and the
    /// renderer samples it directly via `loaded_chunk_views()` — no CPU
    /// readback or `VoxelWorld` insert happens here. Settlement chunks
    /// (CPU-generated in a background thread) are handled by
    /// `drain_ready_chunks` and still go through the CPU path.
    fn drain_completed_gpu_chunks(&mut self) {
        let completed = match self
            .terrain_compute
            .try_take_completed_with_frame(&self.ctx, self.frame_count as u64)
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[voxel] try_take_completed failed: {}", e);
                return;
            }
        };
        self.drain_completed_this_sec += completed.len() as u32;
        for (req_id, _chunk_pos) in completed {
            self.pending_chunk_requests.remove(&req_id);
        }
    }

    /// Upload dirty mega-chunks within camera radius to the GPU.
    /// Budgeted: uploads at most `MAX_MEGA_UPLOADS_PER_FRAME` megas per call
    /// to avoid stalling the render loop.
    fn upload_megas(&mut self) -> Result<()> {
        const MAX_MEGA_UPLOADS_PER_FRAME: usize = 2;

        let palette_rgba = self.bridge.palette_rgba();
        let cam_pos = self.camera.eye_position();
        let r2 = LOAD_RADIUS * LOAD_RADIUS;

        // Sort dirty megas by distance to camera (closest first), filtering
        // out-of-range entries. Cheap loop — no grid building yet.
        let mut candidates: Vec<(MegaPos, f32)> = self.dirty_megas.iter()
            .filter_map(|&mp| {
                let c = mp.world_center();
                let dx = cam_pos[0] - c[0];
                let dy = cam_pos[1] - c[1];
                let dz = cam_pos[2] - c[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 <= r2 { Some((mp, d2)) } else { None }
            })
            .collect();
        candidates.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Only build/upload up to MAX_MEGA_UPLOADS_PER_FRAME closest megas.
        // Out-of-range entries stay in dirty_megas (no churn).
        let mut uploaded = 0;
        for (mp, _dist2) in candidates {
            if uploaded >= MAX_MEGA_UPLOADS_PER_FRAME {
                break;
            }

            // Remove from dirty before processing (we'll re-dirty on failure).
            self.dirty_megas.remove(&mp);

            let grid = match self.build_mega_grid(mp) {
                Some(g) => g,
                None => {
                    // Fully occluded or empty — evict if previously uploaded.
                    if let Some(old) = self.gpu_megas.remove(&mp) {
                        old.texture.destroy(&self.ctx, &mut self.alloc);
                    }
                    continue;
                }
            };

            let texture = voxel_gpu::upload_grid_to_gpu(
                &self.ctx, &mut self.alloc, &grid, palette_rgba,
            )?;

            if let Some(old) = self.gpu_megas.remove(&mp) {
                old.texture.destroy(&self.ctx, &mut self.alloc);
            }

            self.gpu_megas.insert(mp, GpuMega {
                texture,
                position: mp.world_position(),
            });
            uploaded += 1;
        }

        // Evict mega-chunks that are too far from camera.
        let far_keys: Vec<MegaPos> = self.gpu_megas.keys()
            .filter(|mp| {
                let c = mp.world_center();
                let dx = cam_pos[0] - c[0];
                let dy = cam_pos[1] - c[1];
                let dz = cam_pos[2] - c[2];
                dx * dx + dy * dy + dz * dz > r2 * 1.5 // hysteresis
            })
            .copied()
            .collect();
        for mp in far_keys {
            if let Some(old) = self.gpu_megas.remove(&mp) {
                old.texture.destroy(&self.ctx, &mut self.alloc);
            }
            // Re-mark dirty so it reloads when camera returns to range.
            // (Only re-dirty if not already dirty to avoid churn.)
            self.dirty_megas.insert(mp);
        }

        Ok(())
    }

    /// Render one frame. Returns sub-phase timings in ms:
    /// `(cull_ms, wait_ms, raycast_ms, present_ms)`.
    fn render(&mut self) -> Result<(f32, f32, f32, f32)> {
        // Only pay for an Instant::now() when the caller is going to
        // read the sub-phase timings. On the steady-state fast path
        // (detailed_perf == false), both the start-of-cull and
        // end-of-cull timestamps are discarded, so we skip them.
        let detailed = self.detailed_perf;
        let t_cull = if detailed { Some(Instant::now()) } else { None };

        // Stable-scene ultra-fast path: if the camera is unchanged AND the
        // chunk pool hasn't mutated since the last cull, `pool_views_buf`
        // still holds the correct visibility set in the correct order.
        // Skip the cull loop entirely (no frustum test, no sort, no
        // rebuild, no per-slot touch refresh).
        //
        // Correctness of skipping mark_touched_slot relies on a chain of
        // invariants:
        //   1. Cull cache hit ⟺ camera + pool_generation unchanged.
        //   2. Camera + pool stable ⟹ generate_camera_chunks hits its
        //      own `in_flight == 0 && cam == converged_cam` short-circuit
        //      and makes no submit_chunk_with_frame calls.
        //   3. No submit ⟹ LRU guard is never consulted, so stale
        //      last_touched_frame values on loaded slots don't matter.
        //
        // When the camera finally moves:
        //   - cull_cache_hit goes false the same frame → full cull runs
        //     and writes fresh last_touched_frame on every visible slot
        //     (same frame_count value we'd pass here).
        //   - Next frame's gen uses the NEW camera pose; spiral runs.
        //     The LRU guard's +1-frame tolerance
        //     (`oldest_frame + 1 >= current_frame`) comfortably covers
        //     the 1-frame gap between cull and submit.
        //
        // Previously we called mark_touched_slot on every visible slot
        // every cache-hit frame as insurance. On a 256-slot pool that
        // was 256 scattered cache-line writes per frame, the single
        // biggest remaining render-loop cost at ~185k FPS.
        let cur_cam_pos = self.camera.eye_position();
        let cur_cam_center = self.camera.center();
        let cur_cam_key = [
            cur_cam_pos[0], cur_cam_pos[1], cur_cam_pos[2],
            cur_cam_center.x, cur_cam_center.y, cur_cam_center.z,
        ];
        let cur_pool_gen = self.terrain_compute.pool_generation();
        let cull_cache_hit = self
            .last_cull_cam_key
            .map(|k| k == cur_cam_key)
            .unwrap_or(false)
            && self.last_cull_pool_gen == cur_pool_gen
            && !self.pool_views_buf.is_empty();
        if cull_cache_hit {
            // Single O(1) bulk touch instead of walking the 256-slot
            // pool_views_buf writing last_touched_frame on every entry.
            // Keeps the LRU guard happy so evictions don't fire during
            // transient states (e.g. the last few InFlight chunks
            // draining while the camera is already parked at steady
            // state, which previously kicked off a 20-chunk/sec churn).
            self.terrain_compute.bulk_touch_all_loaded(self.frame_count as u64);
            let cull_ms = t_cull.map(|t| t.elapsed().as_secs_f32() * 1000.0).unwrap_or(0.0);
            return Ok((cull_ms, 0.0, 0.0, 0.0));
        }

        let dims = MEGA_VOXELS as f32;
        let aspect = RENDER_WIDTH as f32 / RENDER_HEIGHT as f32;
        let vp = frustum_vp_matrix(&self.camera, aspect);
        let planes = extract_frustum_planes(&vp);
        let cam_pos = self.camera.eye_position();

        // --- Phase 3: render directly from the GPU chunk pool ---
        // Reuse pre-allocated Vecs to avoid per-frame heap churn on the
        // 5000 FPS hot path.
        let chunk_size_f = CHUNK_SIZE as f32;
        self.visible_buf.clear();
        for v in self.terrain_compute.loaded_chunk_views() {
            let pos = [
                v.chunk_pos[0] as f32 * chunk_size_f,
                v.chunk_pos[2] as f32 * chunk_size_f, // sim z → engine y (up)
                v.chunk_pos[1] as f32 * chunk_size_f, // sim y → engine z
            ];
            let max = [pos[0] + dims, pos[1] + dims, pos[2] + dims];
            if !aabb_vs_frustum(&planes, &pos, &max) {
                continue;
            }
            let cx = pos[0] + dims * 0.5 - cam_pos[0];
            let cy = pos[1] + dims * 0.5 - cam_pos[1];
            let cz = pos[2] + dims * 0.5 - cam_pos[2];
            let dist2 = cx * cx + cy * cy + cz * cz;
            if dist2 > LOAD_RADIUS * LOAD_RADIUS * 4.0 {
                continue;
            }
            self.visible_buf
                .push((v, [1.0f32, 1.0, 1.0, 1.0], pos, [dims, dims, dims], dist2));
        }
        self.visible_buf
            .sort_unstable_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));

        let cur_frame = self.frame_count as u64;
        for (v, _, _, _, _) in self.visible_buf.iter() {
            self.terrain_compute.mark_touched_slot(v.slot_idx, cur_frame);
        }

        self.pool_views_buf.clear();
        self.pool_views_buf
            .extend(self.visible_buf.iter().map(|(v, c, p, d, _)| (*v, *c, *p, *d)));
        self.last_visible_megas = self.pool_views_buf.len();

        // Stamp the cull cache so subsequent frames with the same camera
        // + pool generation can skip the whole visibility pass.
        self.last_cull_cam_key = Some(cur_cam_key);
        self.last_cull_pool_gen = cur_pool_gen;

        let cull_ms = t_cull.map(|t| t.elapsed().as_secs_f32() * 1000.0).unwrap_or(0.0);

        if self.pool_views_buf.is_empty() {
            let t_present = Instant::now();
            self.swapchain.present_cleared_frame(&self.ctx, [0.1, 0.1, 0.15, 1.0])?;
            let present_ms = t_present.elapsed().as_secs_f32() * 1000.0;
            return Ok((cull_ms, 0.0, 0.0, present_ms));
        }

        // Stable-scene super-fast path: if the camera and pool are
        // identical to the previous frame, the renderer's output AND the
        // last presented swapchain image are already correct. Skip the
        // entire wait→render→present chain — no fence sync, no queue
        // submit, no blit, no vkQueuePresentKHR. The display keeps
        // scanning out the last presented content in IMMEDIATE mode, so
        // this is a pure win (~0.06 ms of fence + presentation overhead
        // eliminated on cache-hit frames).
        if self.renderer.cache_matches(&self.camera, &self.pool_views_buf) {
            return Ok((cull_ms, 0.0, 0.0, 0.0));
        }

        // Split out the CPU wait on last frame's render_fence so we can see
        // whether the old "raycast" bucket was dominated by the wait (GPU
        // behind because compute is saturating shared resources) or by
        // command-buffer recording/submit cost.
        let t_wait = Instant::now();
        self.renderer.wait_for_previous_frame(&self.ctx)?;
        let wait_ms = t_wait.elapsed().as_secs_f32() * 1000.0;

        let t_raycast = Instant::now();
        let palette_view = self.terrain_compute.palette_view();
        self.renderer
            .render_frame_pool(&self.ctx, &self.camera, &self.pool_views_buf, palette_view)?;
        let raycast_ms = t_raycast.elapsed().as_secs_f32() * 1000.0;

        let t_present = Instant::now();
        // Lighting was merged into the gbuffer fragment shader, so the
        // final post-light color lives in the gbuffer's single RT.
        let src = self.renderer.gbuffer_output_image();
        // GPU-side wait: the blit reads the renderer's output, which
        // the renderer queued but did NOT CPU-wait for. The semaphore makes
        // present_blit's submit block on the GPU side until render is done.
        let render_done = self.renderer.render_done_semaphore();
        self.swapchain.present_blit_with_wait(
            &self.ctx, src, RENDER_WIDTH, RENDER_HEIGHT, render_done,
        )?;
        let present_ms = t_present.elapsed().as_secs_f32() * 1000.0;

        Ok((cull_ms, wait_ms, raycast_ms, present_ms))
    }

    /// One frame's worth of game work (drain → gen → update_cam → tick_sim
    /// → render). `dt` is pre-computed by the batch driver in
    /// `about_to_wait`. Per-frame timing, EMA updates, frame-count and
    /// fps logging all live in the batch driver too, so on the fast
    /// path this function makes zero `Instant::now()` calls.
    ///
    /// Set `VOXEL_PERF_DETAILED=1` to re-enable the per-phase timers
    /// for debugging a specific bucket — that path is untouched by the
    /// batching change.
    fn run_frame(&mut self, dt: f32) {
        if self.detailed_perf {
            // Detailed debug path: per-phase timers. Pays ~14 clock
            // reads per frame, only used when investigating a specific
            // bucket.
            let t_frame_start = Instant::now();

            let t_drain_cpu = Instant::now();
            self.drain_ready_chunks(64);
            let drain_cpu_ms = t_drain_cpu.elapsed().as_secs_f32() * 1000.0;

            let t_drain_gpu = Instant::now();
            self.drain_completed_gpu_chunks();
            let drain_gpu_ms = t_drain_gpu.elapsed().as_secs_f32() * 1000.0;

            let t_gen = Instant::now();
            self.generate_camera_chunks(8);
            let gen_ms = t_gen.elapsed().as_secs_f32() * 1000.0;

            let t_cam = Instant::now();
            self.update_camera(dt);
            let update_cam_ms = t_cam.elapsed().as_secs_f32() * 1000.0;

            let t_sim = Instant::now();
            self.tick_sim(dt);
            let tick_sim_ms = t_sim.elapsed().as_secs_f32() * 1000.0;

            let t_render = Instant::now();
            let (cull_ms, wait_ms, raycast_ms, present_ms) = match self.render() {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[voxel] render error: {}", e);
                    (0.0, 0.0, 0.0, 0.0)
                }
            };
            let render_ms = t_render.elapsed().as_secs_f32() * 1000.0;

            let frame_ms = t_frame_start.elapsed().as_secs_f32() * 1000.0;

            let a = 0.1;
            let lerp = |old: f32, new: f32| old * (1.0 - a) + new * a;
            self.ema_frame_ms = lerp(self.ema_frame_ms, frame_ms);
            self.ema_drain_cpu_ms = lerp(self.ema_drain_cpu_ms, drain_cpu_ms);
            self.ema_drain_gpu_ms = lerp(self.ema_drain_gpu_ms, drain_gpu_ms);
            self.ema_gen_ms = lerp(self.ema_gen_ms, gen_ms);
            self.ema_update_cam_ms = lerp(self.ema_update_cam_ms, update_cam_ms);
            self.ema_tick_sim_ms = lerp(self.ema_tick_sim_ms, tick_sim_ms);
            self.ema_render_ms = lerp(self.ema_render_ms, render_ms);
            self.ema_cull_ms = lerp(self.ema_cull_ms, cull_ms);
            self.ema_wait_ms = lerp(self.ema_wait_ms, wait_ms);
            self.ema_raycast_ms = lerp(self.ema_raycast_ms, raycast_ms);
            self.ema_present_ms = lerp(self.ema_present_ms, present_ms);
        } else {
            // Fast path: zero clock reads. Work only.
            self.drain_ready_chunks(64);
            self.drain_completed_gpu_chunks();
            self.generate_camera_chunks(8);
            self.update_camera(dt);
            self.tick_sim(dt);
            if let Err(e) = self.render() {
                eprintln!("[voxel] render error: {}", e);
            }
        }
    }

    /// Called once per `about_to_wait` visit (i.e. once per batch of
    /// FRAME_BATCH run_frame() calls). Updates frame_count, does the
    /// 1 Hz perf log, and updates ema_frame_ms from the batch-level
    /// measurement. Batch-level timing is applied to every frame in
    /// the batch as an average, which is exactly what the EMA wants.
    fn record_batch_stats(&mut self, batch_start: Instant, batch_ms: f32, frames_in_batch: usize) {
        // Per-frame timing averaged over the batch.
        let per_frame_ms = batch_ms / frames_in_batch as f32;
        // In the fast path we only update ema_frame_ms; the detailed
        // path does per-phase EMAs inside run_frame.
        if !self.detailed_perf {
            self.ema_frame_ms = self.ema_frame_ms * 0.9 + per_frame_ms * 0.1;
        }

        self.frame_count += frames_in_batch as u32;
        let fps_elapsed = batch_start.duration_since(self.fps_timer).as_secs_f32();
        if fps_elapsed >= 1.0 {
            self.last_fps = self.frame_count as f32 / fps_elapsed;
            self.frame_count = 0;
            self.fps_timer = batch_start;
            let cam = self.camera.eye_position();
            let status = if self.paused { "PAUSED" } else { "RUNNING" };
            let loading = if self.chunks_loaded < self.chunks_pending {
                format!(" | loading {}/{}", self.chunks_loaded, self.chunks_pending)
            } else {
                String::new()
            };
            self.window.set_title(&format!(
                "World Sim — {:.0} FPS | {} | {:.0},{:.0},{:.0} | {}/{} megas | speed {}x{}",
                self.last_fps, status, cam[0], cam[1], cam[2],
                self.last_visible_megas, self.gpu_megas.len(), self.ticks_per_frame, loading,
            ));

            let accounted = self.ema_drain_cpu_ms + self.ema_drain_gpu_ms + self.ema_gen_ms
                + self.ema_update_cam_ms + self.ema_tick_sim_ms + self.ema_render_ms;
            let overhead = (self.ema_frame_ms - accounted).max(0.0);
            let (free, in_flight, loaded) = self.terrain_compute.pool_stats();
            eprintln!(
                "[perf] {:.1} FPS frame={:.2}ms | drain_cpu={:.2} drain_gpu={:.2} gen={:.2} cam={:.2} sim={:.2} render={:.2} [cull={:.2} wait={:.2} raycast={:.2} present={:.2}] other={:.2} | visible={} pool=free:{}/inflight:{}/loaded:{} | throughput: sub={}/s drained={}/s gen_short_circuit={}/s",
                self.last_fps,
                self.ema_frame_ms,
                self.ema_drain_cpu_ms,
                self.ema_drain_gpu_ms,
                self.ema_gen_ms,
                self.ema_update_cam_ms,
                self.ema_tick_sim_ms,
                self.ema_render_ms,
                self.ema_cull_ms,
                self.ema_wait_ms,
                self.ema_raycast_ms,
                self.ema_present_ms,
                overhead,
                self.last_visible_megas,
                free, in_flight, loaded,
                self.gen_submitted_this_sec,
                self.drain_completed_this_sec,
                self.gen_short_circuit_this_sec,
            );
            self.gen_submitted_this_sec = 0;
            self.drain_completed_this_sec = 0;
            self.gen_short_circuit_this_sec = 0;
        }
    }

    fn tick_sim(&mut self, dt: f32) {
        if self.paused { return; }

        // Wall-clock sim scheduling. Base rate is SIM_BASE_HZ; the user's
        // ticks_per_frame multiplier scales the effective sim rate so
        // "ticks_per_frame=2" means the sim advances twice as fast in
        // wall time. Previously the sim ran literally once per render
        // frame × ticks_per_frame, which meant at 194 k FPS × 10 it
        // was burning ~1.9 million sim ticks per second — almost all
        // of the remaining render-loop cost.
        const SIM_BASE_HZ: f32 = 10.0;
        let sim_dt = 1.0 / (SIM_BASE_HZ * self.ticks_per_frame.max(1) as f32);

        self.sim_accumulator += dt;
        // Cap burst to avoid a death spiral if the render loop stalled.
        let mut bursts = 0;
        const MAX_BURST: u32 = 4;
        let mut ticked = false;
        while self.sim_accumulator >= sim_dt && bursts < MAX_BURST {
            self.sim.tick();
            self.sim_accumulator -= sim_dt;
            bursts += 1;
            ticked = true;
        }
        if self.sim_accumulator > sim_dt * MAX_BURST as f32 {
            // Discard excess to prevent runaway after a long hitch.
            self.sim_accumulator = 0.0;
        }
        if !ticked { return; }

        self.bridge.sync_all(self.sim.state_mut());

        // Mark mega-chunks dirty for any sim chunks that changed,
        // then clear the chunk dirty flag so we don't re-upload every frame.
        for chunk in self.sim.state_mut().voxel_world.chunks.values_mut() {
            if chunk.dirty {
                self.dirty_megas.insert(MegaPos::from_chunk(chunk.pos));
                chunk.dirty = false;
            }
        }
    }

    fn handle_key(&mut self, key: KeyCode, pressed: bool) {
        if pressed {
            self.keys_held.insert(key);
        } else {
            self.keys_held.remove(&key);
        }

        if !pressed { return; }
        match key {
            KeyCode::Space => {
                self.paused = !self.paused;
                eprintln!("[voxel] {}", if self.paused { "PAUSED" } else { "RUNNING" });
            }
            KeyCode::Equal | KeyCode::NumpadAdd => {
                self.ticks_per_frame = (self.ticks_per_frame * 2).min(1000);
                eprintln!("[voxel] speed: {} ticks/frame", self.ticks_per_frame);
            }
            KeyCode::Minus | KeyCode::NumpadSubtract => {
                self.ticks_per_frame = (self.ticks_per_frame / 2).max(1);
                eprintln!("[voxel] speed: {} ticks/frame", self.ticks_per_frame);
            }
            KeyCode::KeyF => {
                // Preload chunks around the camera at the actual terrain
                // surface height. Radius 4 caps the submission at 243 chunks,
                // under the pool's 256-slot limit, so currently-visible chunks
                // don't get LRU-evicted while the new dispatches are pending.
                let submitted = self.fill_chunks_around_camera(4);
                eprintln!("[voxel] Fill radius: submitted {} chunks", submitted);
            }
            KeyCode::Tab => {
                // Teleport to next settlement at first-person surface height.
                // Uses plan-based surface_height_at (voxel_world.surface_height
                // is stale — only scans z=0..63 which is below the new terrain).
                let settlements = self.sim.state().settlements.clone();
                if settlements.is_empty() { return; }
                let plan = match self.sim.state().voxel_world.region_plan.clone() {
                    Some(p) => p,
                    None => return,
                };
                let seed = self.sim.state().rng_state;
                self.settlement_jump_idx = self.settlement_jump_idx % settlements.len();
                let s = &settlements[self.settlement_jump_idx];
                let vx = s.pos.0;
                let vy = s.pos.1;
                let surface_z = crate::world_sim::terrain::surface_height_at(vx, vy, &plan, seed);
                // First-person standing position: 18 voxels (~1.8m) above surface.
                let cam_pos = glam::Vec3::new(vx, (surface_z + 18) as f32, vy);
                self.camera.set_position(cam_pos);
                eprintln!("[voxel] Jumped to '{}' ({:.0},{:.0}) surface_z={}", s.name, vx, vy, surface_z);
                self.settlement_jump_idx += 1;
            }
            _ => {}
        }
    }

    fn update_camera(&mut self, dt: f32) {
        use voxel_engine::camera::{CameraController, InputState};

        // Fast path: if no movement keys are held, nothing about the
        // camera changes this frame. Mouse deltas are applied directly
        // from WindowEvent::CursorMoved (not through update_camera), so
        // an empty keys_held set means truly nothing to update — avoid
        // the HashSet lookups AND the per-call view-matrix rebuild in
        // FreeCamera::update (~0.5 µs/frame at 360 k FPS).
        if self.keys_held.is_empty() {
            return;
        }

        let forward = if self.keys_held.contains(&KeyCode::KeyW) { 1.0 }
            else if self.keys_held.contains(&KeyCode::KeyS) { -1.0 }
            else { 0.0 };
        let right = if self.keys_held.contains(&KeyCode::KeyD) { 1.0 }
            else if self.keys_held.contains(&KeyCode::KeyA) { -1.0 }
            else { 0.0 };
        let up = if self.keys_held.contains(&KeyCode::KeyE) { 1.0 }
            else if self.keys_held.contains(&KeyCode::KeyQ) { -1.0 }
            else { 0.0 };

        self.camera.update(&InputState {
            move_forward: forward,
            move_right: right,
            move_up: up,
            mouse_dx: 0.0,
            mouse_dy: 0.0,
            scroll_delta: 0.0,
        }, dt);
    }
}

// ---------------------------------------------------------------------------
// Winit ApplicationHandler
// ---------------------------------------------------------------------------

struct WorldSimVoxelApp {
    state: Option<AppState>,
    sim: Option<WorldSim>,
}

impl ApplicationHandler for WorldSimVoxelApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let attrs = Window::default_attributes()
            .with_title("World Sim — Voxel Renderer")
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));

        let window = match event_loop.create_window(attrs) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("[voxel] Failed to create window: {}", e);
                event_loop.exit();
                return;
            }
        };

        let sim = self.sim.take().expect("sim should be set");
        match AppState::new(window, sim) {
            Ok(app) => {
                eprintln!("[voxel] Initialized, loading terrain in background...");
                self.state = Some(app);
            }
            Err(e) => {
                eprintln!("[voxel] Failed to initialize: {}", e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let app = match &mut self.state {
            Some(a) => a,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if key == KeyCode::Escape && event.state == ElementState::Pressed {
                        event_loop.exit();
                        return;
                    }
                    app.handle_key(key, event.state == ElementState::Pressed);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    app.mouse_captured = state == ElementState::Pressed;
                    if !app.mouse_captured { app.last_mouse = None; }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if app.mouse_captured {
                    if let Some((lx, ly)) = app.last_mouse {
                        use voxel_engine::camera::{CameraController, InputState};
                        app.camera.update(&InputState {
                            move_forward: 0.0, move_right: 0.0, move_up: 0.0,
                            mouse_dx: (position.x - lx) as f32,
                            mouse_dy: (position.y - ly) as f32,
                            scroll_delta: 0.0,
                        }, 0.0);
                    }
                    app.last_mouse = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                app.move_speed = (app.move_speed + scroll * 10.0).clamp(5.0, 500.0);
                app.camera.set_move_speed(app.move_speed);
            }
            // Frames are now driven by `about_to_wait` (see below), not
            // by the winit redraw queue. Any RedrawRequested event from
            // the compositor (e.g. window expose) is harmless here —
            // the next about_to_wait will render a fresh frame anyway.
            WindowEvent::RedrawRequested => {}
            _ => {}
        }
    }

    /// Called by winit after the event queue is drained, before the
    /// event loop sleeps/polls again. With `ControlFlow::Poll` this
    /// fires on every iteration, so we drive frame rendering here
    /// instead of through `RedrawRequested` + `request_redraw()`. That
    /// eliminates the per-frame round trip through the window redraw
    /// queue and the cross-process event dispatch that comes with it.
    ///
    /// Additionally, we batch multiple run_frame() calls per
    /// about_to_wait invocation. At 367 k FPS, each trip out to winit
    /// costs ~2 µs of event-loop iteration overhead (epoll_wait syscall,
    /// event-queue scan, trait dispatch) that dominates the frame — pure
    /// fast-path work inside run_frame is only ~0.7 µs. Batching amortizes
    /// the winit overhead across FRAME_BATCH frames. Input latency is
    /// bounded by `FRAME_BATCH × frame_time` — at ~2 µs/frame × 8 = 16 µs,
    /// far below human perception and responsive enough that pending
    /// events still get processed every ~16 µs.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        const FRAME_BATCH: usize = 8;
        if let Some(app) = &mut self.state {
            // Batch-level timing. One Instant::now() pair per 8 frames
            // instead of one per frame — at ~25 ns each that was 12 %
            // of the 0.4 µs frame. Per-frame `dt` is the batch average;
            // since the sim accumulator just sums incoming dts, giving
            // each frame `batch_dt / FRAME_BATCH` advances the sim by
            // the same total wall time as the old per-frame measurement.
            let batch_start = Instant::now();
            let dt_total = (batch_start - app.last_frame).as_secs_f32().min(0.1);
            app.last_frame = batch_start;
            let per_frame_dt = dt_total / FRAME_BATCH as f32;

            for _ in 0..FRAME_BATCH {
                app.run_frame(per_frame_dt);
            }

            let batch_ms = batch_start.elapsed().as_secs_f32() * 1000.0;
            app.record_batch_stats(batch_start, batch_ms, FRAME_BATCH);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU material-byte → VoxelMaterial
// ---------------------------------------------------------------------------

/// Maps a material byte from the GPU terrain compute shader to the
/// corresponding `VoxelMaterial` enum variant. Must stay in lockstep with the
/// `repr(u8)` ordering in `voxel.rs` and the material ids the shader writes.
fn voxel_material_from_u8(b: u8) -> crate::world_sim::voxel::VoxelMaterial {
    use crate::world_sim::voxel::VoxelMaterial;
    match b {
        0 => VoxelMaterial::Air,
        1 => VoxelMaterial::Dirt,
        2 => VoxelMaterial::Stone,
        3 => VoxelMaterial::Granite,
        4 => VoxelMaterial::Sand,
        5 => VoxelMaterial::Clay,
        6 => VoxelMaterial::Gravel,
        7 => VoxelMaterial::Grass,
        8 => VoxelMaterial::Water,
        9 => VoxelMaterial::Lava,
        10 => VoxelMaterial::Ice,
        11 => VoxelMaterial::Snow,
        12 => VoxelMaterial::IronOre,
        13 => VoxelMaterial::CopperOre,
        14 => VoxelMaterial::GoldOre,
        15 => VoxelMaterial::Coal,
        16 => VoxelMaterial::Crystal,
        17 => VoxelMaterial::WoodLog,
        18 => VoxelMaterial::WoodPlanks,
        19 => VoxelMaterial::StoneBlock,
        20 => VoxelMaterial::StoneBrick,
        21 => VoxelMaterial::Thatch,
        22 => VoxelMaterial::Iron,
        23 => VoxelMaterial::Glass,
        24 => VoxelMaterial::Farmland,
        25 => VoxelMaterial::Crop,
        26 => VoxelMaterial::Basalt,
        27 => VoxelMaterial::Sandstone,
        28 => VoxelMaterial::Marble,
        29 => VoxelMaterial::Bone,
        30 => VoxelMaterial::Brick,
        31 => VoxelMaterial::CutStone,
        32 => VoxelMaterial::Concrete,
        33 => VoxelMaterial::Ceramic,
        34 => VoxelMaterial::Steel,
        35 => VoxelMaterial::Bronze,
        36 => VoxelMaterial::Obsidian,
        37 => VoxelMaterial::JungleMoss,
        38 => VoxelMaterial::MudGrass,
        39 => VoxelMaterial::RedSand,
        40 => VoxelMaterial::Peat,
        41 => VoxelMaterial::TallGrass,
        42 => VoxelMaterial::Leaves,
        _ => VoxelMaterial::Air, // unknown → air
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run_with_renderer(sim: WorldSim) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = WorldSimVoxelApp {
        state: None,
        sim: Some(sim),
    };

    event_loop.run_app(&mut app)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Frustum culling helpers
// ---------------------------------------------------------------------------

/// Build View-Projection matrix from camera (column-major, glam layout).
fn frustum_vp_matrix(cam: &FreeCamera, aspect: f32) -> [f32; 16] {
    let v = cam.view_matrix_array();
    let p = cam.projection_matrix_array(aspect);
    mat4_mul_cols(&p, &v)
}

/// Column-major 4×4 multiply.
fn mat4_mul_cols(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut r = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            r[col * 4 + row] = sum;
        }
    }
    r
}

/// Extract 6 frustum planes from a column-major VP matrix.
/// Each plane is [a, b, c, d] where ax + by + cz + d >= 0 is inside.
fn extract_frustum_planes(vp: &[f32; 16]) -> [[f32; 4]; 6] {
    let row = |i: usize| -> [f32; 4] { [vp[i], vp[4 + i], vp[8 + i], vp[12 + i]] };
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);

    let add = |a: &[f32; 4], b: &[f32; 4]| -> [f32; 4] { [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]] };
    let sub = |a: &[f32; 4], b: &[f32; 4]| -> [f32; 4] { [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]] };

    let mut planes = [
        add(&r3, &r0), // left
        sub(&r3, &r0), // right
        add(&r3, &r1), // bottom
        sub(&r3, &r1), // top
        add(&r3, &r2), // near
        sub(&r3, &r2), // far
    ];

    for p in &mut planes {
        let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if len > 1e-8 {
            p[0] /= len; p[1] /= len; p[2] /= len; p[3] /= len;
        }
    }
    planes
}

/// Test AABB against frustum planes. Returns true if potentially visible.
fn aabb_vs_frustum(planes: &[[f32; 4]; 6], min: &[f32; 3], max: &[f32; 3]) -> bool {
    for p in planes {
        let px = if p[0] >= 0.0 { max[0] } else { min[0] };
        let py = if p[1] >= 0.0 { max[1] } else { min[1] };
        let pz = if p[2] >= 0.0 { max[2] } else { min[2] };
        if p[0] * px + p[1] * py + p[2] * pz + p[3] < 0.0 {
            return false;
        }
    }
    true
}
