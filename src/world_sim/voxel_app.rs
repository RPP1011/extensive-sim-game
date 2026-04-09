//! Windowed voxel renderer for the world sim.
//!
//! Merges 4×4×4 sim chunks into 64³ mega-grid GPU textures to reduce draw calls.
//! Only uploads mega-chunks within camera radius. Uses GPU blit presentation.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::time::Instant;

use anyhow::Result;
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

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

/// How many sim chunks per mega-chunk axis. 4 × 16 = 64 voxels per side.
const MEGA: i32 = 4;
const MEGA_VOXELS: u32 = (MEGA as u32) * (CHUNK_SIZE as u32); // 64

/// Maximum distance (in world units) from camera to mega-chunk center for it to be loaded.
const LOAD_RADIUS: f32 = 2048.0;

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

    // Background chunk generation
    chunk_rx: mpsc::Receiver<(ChunkPos, Chunk)>,
    chunks_pending: usize,
    chunks_loaded: usize,
}

impl AppState {
    fn new(window: Window, sim: WorldSim) -> Result<Self> {
        let t0 = Instant::now();
        let ctx = VulkanContext::new_with_surface_extensions(&window)?;
        eprintln!("[voxel] Vulkan context: {:.1}ms", t0.elapsed().as_secs_f32() * 1000.0);
        let alloc = VulkanAllocator::new(&ctx)?;
        let mut swapchain = SwapchainContext::new(&ctx, &window)?;
        let renderer = VoxelRenderer::new(&ctx, WIDTH, HEIGHT)?;
        eprintln!("[voxel] Renderer ready: {:.1}ms total", t0.elapsed().as_secs_f32() * 1000.0);

        // Show a blank frame immediately so the window isn't frozen.
        let _ = swapchain.present_cleared_frame(&ctx, [0.05, 0.05, 0.08, 1.0]);

        let bridge = VoxelBridge::new();

        // Compute chunk positions to generate, but don't block — spawn background thread.
        let state = sim.state();
        let seed = state.rng_state;
        let plan = state.voxel_world.region_plan.clone();

        let mut chunk_positions: Vec<ChunkPos> = Vec::new();
        let mut positions: Vec<(f32, f32)> = state.settlements.iter()
            .map(|s| s.pos)
            .collect();
        for e in &state.entities {
            if e.alive { positions.push(e.pos); }
        }
        if !positions.is_empty() {
            let min_x = positions.iter().map(|p| p.0).fold(f32::MAX, f32::min);
            let max_x = positions.iter().map(|p| p.0).fold(f32::MIN, f32::max);
            let min_y = positions.iter().map(|p| p.1).fold(f32::MAX, f32::min);
            let max_y = positions.iter().map(|p| p.1).fold(f32::MIN, f32::max);

            let cs = CHUNK_SIZE as f32;
            let pad = 2;
            let c_min_x = (min_x / cs).floor() as i32 - pad;
            let c_max_x = (max_x / cs).ceil() as i32 + pad;
            let c_min_y = (min_y / cs).floor() as i32 - pad;
            let c_max_y = (max_y / cs).ceil() as i32 + pad;

            let (c_min_z, c_max_z) = if let Some(ref plan) = plan {
                use crate::world_sim::terrain::MAX_SURFACE_Z;
                let mut min_h = f32::MAX;
                let mut max_h = f32::MIN;
                for &(px, py) in &positions {
                    let h = plan.interpolate_height(px, py);
                    min_h = min_h.min(h);
                    max_h = max_h.max(h);
                }
                let surface_min = (min_h * MAX_SURFACE_Z as f32) as i32;
                let surface_max = (max_h * MAX_SURFACE_Z as f32) as i32;
                let z_lo = (surface_min / CHUNK_SIZE as i32) - 3;
                let z_hi = (surface_max / CHUNK_SIZE as i32) + 3;
                (z_lo, z_hi)
            } else {
                (0, 3)
            };

            for cx in c_min_x..=c_max_x {
                for cy in c_min_y..=c_max_y {
                    for cz in c_min_z..=c_max_z {
                        chunk_positions.push(ChunkPos::new(cx, cy, cz));
                    }
                }
            }
        }

        let total_chunks = chunk_positions.len();
        eprintln!("[voxel] Queued {} chunks for background generation", total_chunks);

        // Sort by distance from camera so nearby chunks load first.
        // Camera will be above first settlement.
        let cam_center = if let Some(s) = state.settlements.first() {
            (s.pos.0, s.pos.1)
        } else {
            (0.0, 0.0)
        };
        chunk_positions.sort_by(|a, b| {
            let s = CHUNK_SIZE as f32;
            let da = (a.x as f32 * s - cam_center.0).powi(2) + (a.y as f32 * s - cam_center.1).powi(2);
            let db = (b.x as f32 * s - cam_center.0).powi(2) + (b.y as f32 * s - cam_center.1).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Spawn background thread to generate chunks.
        let (chunk_tx, chunk_rx) = mpsc::channel::<(ChunkPos, Chunk)>();
        if let Some(plan_clone) = plan.clone() {
            std::thread::spawn(move || {
                for (i, cp) in chunk_positions.iter().enumerate() {
                    let chunk = crate::world_sim::terrain::materialize_chunk(*cp, &plan_clone, seed);
                    if chunk_tx.send((*cp, chunk)).is_err() {
                        break; // receiver dropped, app closed
                    }
                    if (i + 1) % 500 == 0 {
                        eprintln!("[voxel] Generated {}/{} chunks", i + 1, total_chunks);
                    }
                }
                eprintln!("[voxel] Background generation complete: {} chunks", total_chunks);
            });
        }

        // Camera above first settlement.
        let (cam_pos, cam_target) = if let Some(s) = sim.state().settlements.first() {
            let z = if let Some(ref plan) = sim.state().voxel_world.region_plan {
                let h = plan.interpolate_height(s.pos.0, s.pos.1);
                h * crate::world_sim::terrain::MAX_SURFACE_Z as f32
            } else {
                80.0
            };
            eprintln!("[voxel] Camera target: settlement at ({:.0}, {:.0}), surface z={:.0}", s.pos.0, s.pos.1, z);
            let target = glam::Vec3::new(s.pos.0, z, s.pos.1);
            let eye = target + glam::Vec3::new(-80.0, 120.0, -80.0);
            (eye, target)
        } else {
            (glam::Vec3::new(0.0, 200.0, 0.0), glam::Vec3::ZERO)
        };
        let mut camera = FreeCamera::new(cam_pos, cam_target);
        camera.set_move_speed(50.0);

        Ok(Self {
            window, ctx, alloc, swapchain, renderer, camera,
            bridge, sim,
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
            chunk_rx,
            chunks_pending: total_chunks,
            chunks_loaded: 0,
        })
    }

    /// Build a 64³ mega-grid from up to 4×4×4 sim chunks.
    fn build_mega_grid(&self, mp: MegaPos) -> Option<VoxelGrid> {
        let world = &self.sim.state().voxel_world;
        let mut any_content = false;

        // Check if this mega-chunk is fully occluded (all constituent chunks solid
        // and all 6 mega-neighbors also fully solid).
        let mut all_solid = true;
        for dz in 0..MEGA {
            for dy in 0..MEGA {
                for dx in 0..MEGA {
                    let cp = ChunkPos::new(mp.x * MEGA + dx, mp.y * MEGA + dy, mp.z * MEGA + dz);
                    match world.chunks.get(&cp) {
                        Some(c) => {
                            if c.solid_count() != CHUNK_VOLUME { all_solid = false; }
                            any_content = true;
                        }
                        None => { all_solid = false; }
                    }
                }
            }
        }
        if !any_content { return None; }

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

    /// Upload dirty mega-chunks within camera radius to the GPU.
    fn upload_megas(&mut self) -> Result<()> {
        let palette_rgba = self.bridge.palette_rgba();
        let cam_pos = self.camera.eye_position();
        let r2 = LOAD_RADIUS * LOAD_RADIUS;

        // Collect dirty megas to process (can't borrow self mutably in loop).
        let to_process: Vec<MegaPos> = self.dirty_megas.drain().collect();

        for mp in to_process {
            // Camera distance check.
            let center = mp.world_center();
            let dx = cam_pos[0] - center[0];
            let dy = cam_pos[1] - center[1];
            let dz = cam_pos[2] - center[2];
            if dx * dx + dy * dy + dz * dz > r2 {
                // Out of range — mark dirty again so it uploads when we get closer.
                self.dirty_megas.insert(mp);
                continue;
            }

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
            self.dirty_megas.insert(mp); // re-dirty so it reloads when close again
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

        let objects: Vec<(&GpuVoxelTexture, [f32; 4], [f32; 3], [f32; 3])> =
            self.gpu_megas.values()
                .map(|gm| (&gm.texture, [1.0, 1.0, 1.0, 1.0], gm.position, [dims, dims, dims]))
                .collect();

        // GPU render + blit to swapchain (no CPU readback).
        self.renderer.render_frame_gpu(&self.ctx, &self.camera, &objects)?;
        let src = self.renderer.light_output_image();
        self.swapchain.present_blit(&self.ctx, src, WIDTH, HEIGHT)?;

        Ok(())
    }

    fn tick_sim(&mut self) {
        if self.paused { return; }

        for _ in 0..self.ticks_per_frame {
            self.sim.tick();
        }

        self.bridge.sync_all(self.sim.state_mut());

        // Mark mega-chunks dirty for any sim chunks that changed.
        for chunk in self.sim.state().voxel_world.chunks.values() {
            if chunk.dirty {
                self.dirty_megas.insert(MegaPos::from_chunk(chunk.pos));
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
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT));

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

                // Drain background-generated chunks (up to 64 per frame to stay responsive).
                app.drain_ready_chunks(64);

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
                        "World Sim — {:.0} FPS | {} | {:.0},{:.0},{:.0} | {} megas | speed {}x{}",
                        app.last_fps, status, cam[0], cam[1], cam[2],
                        app.gpu_megas.len(), app.ticks_per_frame, loading,
                    ));
                }

                app.update_camera(dt);
                app.tick_sim();

                if let Err(e) = app.upload_megas() {
                    eprintln!("[voxel] upload error: {}", e);
                }
                if let Err(e) = app.render() {
                    eprintln!("[voxel] render error: {}", e);
                }

                app.window.request_redraw();
            }
            _ => {}
        }
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
