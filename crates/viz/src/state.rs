//! App state. Templated after `src/world_sim/voxel_app.rs::AppState`
//! but far smaller: one GpuVoxelTexture, no chunk pool, no terrain
//! compute, no mega-chunk bookkeeping.

use std::collections::HashSet;
use std::time::Instant;

use anyhow::{Context, Result};
use glam::Vec3;
use voxel_engine::camera::FreeCamera;
use voxel_engine::render::VoxelRenderer;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::swapchain::SwapchainContext;
use voxel_engine::vulkan::voxel_gpu::{self, GpuVoxelTexture};
use voxel_engine::voxel::grid::VoxelGrid;
use winit::keyboard::KeyCode;
use winit::window::Window;

use viz::grid_paint::{clear_above_ground, paint_ground_plane, GRID_SIDE};
use viz::palette::build_palette_rgba;

// Matches src/world_sim/constants.rs for visual consistency with legacy renderer.
pub const RENDER_WIDTH:  u32 =  480;
pub const RENDER_HEIGHT: u32 =  270;
pub const WINDOW_WIDTH:  u32 = 1280;
pub const WINDOW_HEIGHT: u32 =  720;

pub struct AppState {
    // Held to keep the window alive for the lifetime of the Vulkan surface.
    // Never read directly after construction — the swapchain owns the surface.
    #[allow(dead_code)]
    pub window:    Window,
    pub ctx:       VulkanContext,
    pub alloc:     VulkanAllocator,
    pub swapchain: SwapchainContext,
    pub renderer:  VoxelRenderer,

    pub camera:         FreeCamera,
    pub keys_held:      HashSet<KeyCode>,
    pub mouse_captured: bool,
    pub last_mouse:     Option<(f64, f64)>,

    pub grid:        VoxelGrid,
    pub gpu_texture: Option<GpuVoxelTexture>,

    pub sim:       engine::state::SimState,
    // Wired into `engine::step::step` in Task 3 — held on AppState now so
    // the SimState/backend pair is plumbed through the scenario loader.
    #[allow(dead_code)]
    pub scratch:   engine::step::SimScratch,
    #[allow(dead_code)]
    pub events:    engine::event::EventRing,
    #[allow(dead_code)]
    pub cascade:   engine::cascade::CascadeRegistry,
    #[allow(dead_code)]
    pub backend:   engine::policy::UtilityBackend,
    #[allow(dead_code)]
    pub agent_ids: Vec<engine::ids::AgentId>,

    pub last_frame:  Instant,
}

impl AppState {
    pub fn new(window: Window, scenario: viz::scenario::Scenario) -> Result<Self> {
        let ctx       = VulkanContext::new_with_surface_extensions(&window).context("Vulkan context")?;
        let alloc     = VulkanAllocator::new(&ctx).context("allocator")?;
        let mut swapchain = SwapchainContext::new(&ctx, &window).context("swapchain")?;
        let renderer  = VoxelRenderer::new(&ctx, RENDER_WIDTH, RENDER_HEIGHT).context("renderer")?;
        swapchain.prepare_blit_commands(&ctx, renderer.gbuffer_output_image(), RENDER_WIDTH, RENDER_HEIGHT)
            .context("prepare blit")?;
        let _ = swapchain.present_cleared_frame(&ctx, [0.05, 0.05, 0.12, 1.0]);

        let mut grid = VoxelGrid::new(GRID_SIDE, GRID_SIDE, GRID_SIDE);
        paint_ground_plane(&mut grid);

        // Expand cap to at least the scenario's agent count so a
        // miscounted cap can't silently refuse a spawn.
        let needed = scenario.agent.len().max(1) as u32;
        let cap = scenario.world.agent_cap.max(needed);
        let mut sim = engine::state::SimState::new(cap, scenario.world.seed);
        let scratch = engine::step::SimScratch::new(cap as usize);
        let events  = engine::event::EventRing::with_cap(4096);
        let cascade = engine::cascade::CascadeRegistry::new();
        let backend = engine::policy::UtilityBackend;

        let mut agent_ids = Vec::with_capacity(scenario.agent.len());
        for spec in &scenario.agent {
            let ct = spec.creature()?;
            let spawn = engine::state::AgentSpawn { creature_type: ct, pos: spec.position(), hp: spec.hp };
            match sim.spawn_agent(spawn) {
                Some(id) => agent_ids.push(id),
                None => anyhow::bail!("spawn_agent returned None — agent_cap {} exhausted", cap),
            }
        }
        eprintln!("[viz] Spawned {} agents (cap {})", agent_ids.len(), cap);

        // Camera aims at the spawn centroid; falls back to grid center if empty.
        let center: Vec3 = if scenario.agent.is_empty() {
            Vec3::new((GRID_SIDE as f32) * 0.5, 2.0, (GRID_SIDE as f32) * 0.5)
        } else {
            let sum: Vec3 = scenario.agent.iter().map(|a| a.position()).sum();
            sum / (scenario.agent.len() as f32)
        };
        let eye = Vec3::new(center.x, center.y + 30.0, center.z - 30.0);
        let target = Vec3::new(center.x, 2.0, center.z);
        let mut camera = FreeCamera::new(eye, target);
        camera.set_move_speed(20.0);

        Ok(Self {
            window, ctx, alloc, swapchain, renderer, camera,
            keys_held: HashSet::new(),
            mouse_captured: false, last_mouse: None,
            grid, gpu_texture: None,
            sim, scratch, events, cascade, backend, agent_ids,
            last_frame: Instant::now(),
        })
    }

    pub fn upload_grid(&mut self) -> Result<()> {
        let palette = build_palette_rgba();
        let new_tex = voxel_gpu::upload_grid_to_gpu(
            &self.ctx, &mut self.alloc, &self.grid, &palette,
        ).context("upload voxel grid")?;
        if let Some(old) = self.gpu_texture.take() {
            old.destroy(&self.ctx, &mut self.alloc);
        }
        self.gpu_texture = Some(new_tex);
        Ok(())
    }

    /// One rendered frame via `render_frame_gpu` + `present_blit`.
    /// `render_frame_gpu` CPU-waits on its fence internally so the
    /// subsequent blit can read the gbuffer RT without a semaphore.
    #[allow(clippy::type_complexity)]
    pub fn render(&mut self) -> Result<()> {
        clear_above_ground(&mut self.grid);
        for id in self.sim.agents_alive() {
            let pos = match self.sim.agent_pos(id) { Some(p) => p, None => continue };
            let ct  = self.sim.agent_creature_type(id).unwrap_or(engine::creature::CreatureType::Human);
            let idx = viz::palette::creature_palette_index(ct);
            viz::grid_paint::paint_agent(&mut self.grid, pos, idx);
        }
        self.upload_grid()?;
        let tex = self.gpu_texture.as_ref().expect("upload_grid always populates gpu_texture");
        let dims = [GRID_SIDE as f32; 3];
        let position = [0.0f32, 0.0, 0.0];
        let palette_color = [1.0f32, 1.0, 1.0, 1.0];
        let objects: [(&GpuVoxelTexture, [f32; 4], [f32; 3], [f32; 3]); 1] =
            [(tex, palette_color, position, dims)];
        self.renderer.render_frame_gpu(&self.ctx, &self.camera, &objects)
            .context("render_frame_gpu")?;
        let src = self.renderer.gbuffer_output_image();
        self.swapchain.present_blit(&self.ctx, src, RENDER_WIDTH, RENDER_HEIGHT)
            .context("present_blit")?;
        Ok(())
    }

    pub fn handle_key(&mut self, key: KeyCode, pressed: bool) {
        if pressed { self.keys_held.insert(key); } else { self.keys_held.remove(&key); }
    }

    pub fn update_camera(&mut self, dt: f32) {
        use voxel_engine::camera::{CameraController, InputState};
        if self.keys_held.is_empty() { return; }
        let forward = if self.keys_held.contains(&KeyCode::KeyW) { 1.0 }
            else if self.keys_held.contains(&KeyCode::KeyS) { -1.0 } else { 0.0 };
        let right = if self.keys_held.contains(&KeyCode::KeyD) { 1.0 }
            else if self.keys_held.contains(&KeyCode::KeyA) { -1.0 } else { 0.0 };
        let up = if self.keys_held.contains(&KeyCode::KeyE) { 1.0 }
            else if self.keys_held.contains(&KeyCode::KeyQ) { -1.0 } else { 0.0 };
        self.camera.update(&InputState {
            move_forward: forward, move_right: right, move_up: up,
            mouse_dx: 0.0, mouse_dy: 0.0, scroll_delta: 0.0,
        }, dt);
    }

    pub fn run_frame(&mut self) {
        let now = Instant::now();
        let dt  = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.update_camera(dt);
        if let Err(e) = self.render() {
            eprintln!("[viz] render error: {}", e);
        }
    }
}
