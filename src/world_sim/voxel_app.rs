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
    ticks_per_frame: u32,
    last_frame: Instant,

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

        // Camera directly above first settlement, looking down.
        // Settlement pos is already in voxel space (not world units).
        let (cam_pos, cam_target) = if let Some(s) = sim.state().settlements.first() {
            let vx = s.pos.0 as i32;
            let vy = s.pos.1 as i32;
            let surface_z = sim.state().voxel_world.surface_height(vx, vy);
            // Engine coords: sim x → engine x, sim z → engine y (up), sim y → engine z
            let target = glam::Vec3::new(vx as f32, surface_z as f32, vy as f32);
            let eye = target + glam::Vec3::new(0.0, 150.0, -100.0);
            eprintln!("[voxel] Camera above settlement '{}' at voxel ({}, {}, {})", s.name, vx, vy, surface_z);
            (eye, target)
        } else {
            (glam::Vec3::new(0.0, 200.0, 0.0), glam::Vec3::ZERO)
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
            ticks_per_frame: 10,
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
            chunk_rx,
            chunks_pending: total_settlement_chunks,
            chunks_loaded: 0,
            pending_chunk_requests: HashMap::new(),
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
    fn generate_camera_chunks(&mut self, budget: usize) {
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
        let cam = self.camera.eye_position();
        // Convert engine coords (x, y-up, z) back to sim coords (x, y, z-up).
        let cam_vx = cam[0];
        let cam_vy = cam[2]; // engine z → sim y
        let cam_vz = cam[1]; // engine y → sim z

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

        let mut submitted = 0;
        // Spiral outward from camera for priority ordering.
        'outer: for dist in 0..=radius {
            for dx in -dist..=dist {
                for dy in -dist..=dist {
                    for dz in -2..=2 { // limited vertical range
                        if submitted >= budget { break 'outer; }
                        // Only process shell of current distance.
                        if dx.abs() != dist && dy.abs() != dist { continue; }

                        let cp = ChunkPos::new(center_cx + dx, center_cy + dy, center_cz + dz);
                        if self.sim.state().voxel_world.chunks.contains_key(&cp) { continue; }
                        // Skip if already dispatched and waiting on GPU.
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
                                // No free slots in the ring — stop submitting
                                // for this frame.
                                break 'outer;
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
    }

    /// Poll the GPU terrain compute pipeline for completed dispatches and
    /// convert each one into a sim `Chunk`. Non-blocking — if nothing is ready,
    /// this is a cheap fence status check per slot.
    fn drain_completed_gpu_chunks(&mut self) {
        // Phase 2: the pool keeps chunks GPU-resident. Until Phase 3 wires
        // the renderer to sample the pool directly, we still need the bytes
        // on the CPU side, so use the deprecated compat shim that performs
        // an explicit readback per completed chunk.
        let completed = match self
            .terrain_compute
            .try_take_completed_with_bytes(&self.ctx)
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[voxel] try_take_completed_with_bytes failed: {}", e);
                return;
            }
        };
        let cs = CHUNK_SIZE;
        for (req_id, chunk_pos, gpu_mats) in completed {
            self.pending_chunk_requests.remove(&req_id);
            let cp = ChunkPos::new(chunk_pos[0], chunk_pos[1], chunk_pos[2]);
            // A pre-generated chunk may have been inserted between submit and
            // drain (e.g. the settlement thread); don't clobber it.
            if self.sim.state().voxel_world.chunks.contains_key(&cp) {
                continue;
            }
            let mut chunk = crate::world_sim::voxel::Chunk::new_air(cp);
            for lz in 0..cs {
                for ly in 0..cs {
                    for lx in 0..cs {
                        let mat_byte = gpu_mats[lz * cs * cs + ly * cs + lx];
                        if mat_byte != 0 {
                            let mat = voxel_material_from_u8(mat_byte);
                            chunk.voxels[crate::world_sim::voxel::local_index(lx, ly, lz)] =
                                crate::world_sim::voxel::Voxel::new(mat);
                        }
                    }
                }
            }
            chunk.dirty = true;
            self.sim.state_mut().voxel_world.chunks.insert(cp, chunk);
            self.dirty_megas.insert(MegaPos::from_chunk(cp));
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

    /// Render one frame.
    fn render(&mut self) -> Result<()> {
        if self.gpu_megas.is_empty() {
            self.swapchain.present_cleared_frame(&self.ctx, [0.1, 0.1, 0.15, 1.0])?;
            return Ok(());
        }

        let dims = MEGA_VOXELS as f32;
        let aspect = RENDER_WIDTH as f32 / RENDER_HEIGHT as f32;
        let vp = frustum_vp_matrix(&self.camera, aspect);
        let planes = extract_frustum_planes(&vp);
        let cam_pos = self.camera.eye_position();

        // Frustum cull + collect visible mega-chunks.
        let mut visible: Vec<(&GpuVoxelTexture, [f32; 4], [f32; 3], [f32; 3], f32)> =
            self.gpu_megas.values()
                .filter(|gm| {
                    let min = gm.position;
                    let max = [min[0] + dims, min[1] + dims, min[2] + dims];
                    aabb_vs_frustum(&planes, &min, &max)
                })
                .map(|gm| {
                    let cx = gm.position[0] + dims * 0.5 - cam_pos[0];
                    let cy = gm.position[1] + dims * 0.5 - cam_pos[1];
                    let cz = gm.position[2] + dims * 0.5 - cam_pos[2];
                    let dist2 = cx * cx + cy * cy + cz * cz;
                    (&gm.texture, [1.0f32, 1.0, 1.0, 1.0], gm.position, [dims, dims, dims], dist2)
                })
                .collect();

        // Sort front-to-back for early-z rejection.
        visible.sort_unstable_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));

        let objects: Vec<(&GpuVoxelTexture, [f32; 4], [f32; 3], [f32; 3])> =
            visible.into_iter().map(|(t, c, p, d, _)| (t, c, p, d)).collect();

        self.last_visible_megas = objects.len();

        // GPU render + blit to swapchain (no CPU readback).
        self.renderer.render_frame_gpu(&self.ctx, &self.camera, &objects)?;
        let src = self.renderer.light_output_image();
        self.swapchain.present_blit(&self.ctx, src, RENDER_WIDTH, RENDER_HEIGHT)?;

        Ok(())
    }

    fn tick_sim(&mut self) {
        if self.paused { return; }

        for _ in 0..self.ticks_per_frame {
            self.sim.tick();
        }

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
            KeyCode::Tab => {
                // Settlement pos is already in voxel space.
                let settlements = &self.sim.state().settlements;
                if settlements.is_empty() { return; }
                self.settlement_jump_idx = self.settlement_jump_idx % settlements.len();
                let s = &settlements[self.settlement_jump_idx];
                let vx = s.pos.0 as i32;
                let vy = s.pos.1 as i32;
                let surface_z = self.sim.state().voxel_world.surface_height(vx, vy);
                let cam_pos = glam::Vec3::new(
                    vx as f32,
                    (surface_z + 80) as f32,
                    vy as f32 - 60.0,
                );
                self.camera.set_position(cam_pos);
                eprintln!("[voxel] Jumped to '{}' ({:.0},{:.0})", s.name, s.pos.0, s.pos.1);
                self.settlement_jump_idx += 1;
            }
            _ => {}
        }
    }

    fn update_camera(&mut self, dt: f32) {
        use voxel_engine::camera::{CameraController, InputState};

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
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - app.last_frame).as_secs_f32().min(0.1);
                app.last_frame = now;

                // Drain background-generated settlement chunks.
                app.drain_ready_chunks(64);

                // Drain GPU compute chunks whose fences are signaled (Task 14).
                // Cheap fence polling — only blocks on memcpy+conversion for
                // chunks that are actually ready.
                let t_drain = Instant::now();
                app.drain_completed_gpu_chunks();
                let _drain_ms = t_drain.elapsed().as_secs_f32() * 1000.0;

                // Submit new GPU terrain dispatches (Task 14). Up to 8 in
                // flight; each frame we top up the ring without blocking.
                let t_gen = Instant::now();
                app.generate_camera_chunks(8);
                let gen_ms = t_gen.elapsed().as_secs_f32() * 1000.0;

                // FPS tracking
                app.frame_count += 1;
                let fps_elapsed = now.duration_since(app.fps_timer).as_secs_f32();
                if fps_elapsed >= 1.0 {
                    app.last_fps = app.frame_count as f32 / fps_elapsed;
                    app.frame_count = 0;
                    app.fps_timer = now;
                    let cam = app.camera.eye_position();
                    let status = if app.paused { "PAUSED" } else { "RUNNING" };
                    let loading = if app.chunks_loaded < app.chunks_pending {
                        format!(" | loading {}/{}", app.chunks_loaded, app.chunks_pending)
                    } else {
                        String::new()
                    };
                    app.window.set_title(&format!(
                        "World Sim — {:.0} FPS | {} | {:.0},{:.0},{:.0} | {}/{} megas | speed {}x{}",
                        app.last_fps, status, cam[0], cam[1], cam[2],
                        app.last_visible_megas, app.gpu_megas.len(), app.ticks_per_frame, loading,
                    ));
                }

                app.update_camera(dt);
                app.tick_sim();

                let t_upload = Instant::now();
                if let Err(e) = app.upload_megas() {
                    eprintln!("[voxel] upload error: {}", e);
                }
                let t_render = Instant::now();
                if let Err(e) = app.render() {
                    eprintln!("[voxel] render error: {}", e);
                }
                let t_done = Instant::now();

                // Log frame breakdown once per second (alongside FPS update)
                if fps_elapsed >= 1.0 {
                    let upload_ms = t_render.duration_since(t_upload).as_secs_f32() * 1000.0;
                    let render_ms = t_done.duration_since(t_render).as_secs_f32() * 1000.0;
                    let total_ms = 1000.0 / app.last_fps.max(0.1);
                    eprintln!("[perf] {:.1} FPS | gen {:.1}ms upload {:.1}ms render {:.1}ms total {:.1}ms | {}/{} megas",
                        app.last_fps, gen_ms, upload_ms, render_ms, total_ms,
                        app.last_visible_megas, app.gpu_megas.len());
                }

                app.window.request_redraw();
            }
            _ => {}
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
