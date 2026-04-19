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

use viz::grid_paint::{paint_ground_plane, GRID_SIDE};
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

    pub last_frame:  Instant,
}

impl AppState {
    pub fn new(window: Window) -> Result<Self> {
        let ctx = VulkanContext::new_with_surface_extensions(&window)
            .context("create Vulkan context")?;
        let alloc = VulkanAllocator::new(&ctx)
            .context("create Vulkan allocator")?;
        let mut swapchain = SwapchainContext::new(&ctx, &window)
            .context("create swapchain")?;
        let renderer = VoxelRenderer::new(&ctx, RENDER_WIDTH, RENDER_HEIGHT)
            .context("create voxel renderer")?;

        // Pre-record blit cmds for the stable output image — enables the
        // SwapchainContext fast path (matches voxel_app.rs:273).
        swapchain.prepare_blit_commands(
            &ctx, renderer.gbuffer_output_image(),
            RENDER_WIDTH, RENDER_HEIGHT,
        ).context("prepare blit commands")?;

        // Blank dark-blue first frame while startup finishes (matches voxel_app.rs:281).
        let _ = swapchain.present_cleared_frame(&ctx, [0.05, 0.05, 0.12, 1.0]);

        let mut grid = VoxelGrid::new(GRID_SIDE, GRID_SIDE, GRID_SIDE);
        paint_ground_plane(&mut grid);

        // Camera: 30 m above + 30 m south of the grid center, looking at the ground.
        let cx = (GRID_SIDE as f32) * 0.5;
        let eye = Vec3::new(cx, 40.0, cx - 30.0);
        let target = Vec3::new(cx, 2.0, cx);
        let mut camera = FreeCamera::new(eye, target);
        camera.set_move_speed(20.0);

        Ok(Self {
            window, ctx, alloc, swapchain, renderer, camera,
            keys_held: HashSet::new(),
            mouse_captured: false,
            last_mouse: None,
            grid,
            gpu_texture: None,
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
        self.upload_grid()?;
        let tex = self.gpu_texture.as_ref()
            .expect("upload_grid always populates gpu_texture");
        let dims = [GRID_SIDE as f32; 3];
        let position = [0.0f32, 0.0, 0.0];
        let palette_color = [1.0f32, 1.0, 1.0, 1.0];
        let objects: [(&GpuVoxelTexture, [f32; 4], [f32; 3], [f32; 3]); 1] =
            [(tex, palette_color, position, dims)];
        self.renderer
            .render_frame_gpu(&self.ctx, &self.camera, &objects)
            .context("render_frame_gpu")?;
        let src = self.renderer.gbuffer_output_image();
        self.swapchain
            .present_blit(&self.ctx, src, RENDER_WIDTH, RENDER_HEIGHT)
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
