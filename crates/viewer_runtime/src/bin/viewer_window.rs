//! `viewer_window` — Phase A.5 windowed driver for `viewer_runtime`.
//!
//! Opens a winit window, drives the Vulkan-backed `voxel_engine`
//! renderer through `Renderer::render_frame_gpu` + the swapchain's
//! `present_blit`, and ticks the sim on a 100ms fixed-step cadence
//! independent of the render rate.
//!
//! Phase A.5 deliberately renders **no scene content** — the
//! Scene-to-renderer bridge (per-tick voxel grid upload covering all
//! agents + terrain) is Phase B's slice. This binary proves the
//! windowing + present pipeline works against an empty objects list,
//! which produces the renderer's sky/sun pass with no obstructing
//! geometry. The window's title bar surfaces sim state every fixed
//! tick: `seed | tick | settlers | monsters | score`.
//!
//! Run with:
//!
//! ```text
//! cargo run -p viewer_runtime --bin viewer_window --release [seed]
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use viewer_runtime::ViewerApp;
use voxel_engine::app::App as _;
use voxel_engine::camera::OrbitCamera;
use voxel_engine::render::{RendererConfig, VoxelRenderer};
use voxel_engine::scene::config::SceneConfig;
use voxel_engine::scene::Scene;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::swapchain::SwapchainContext;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const SIM_TICK_PERIOD: Duration = Duration::from_millis(100);
const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;

/// Camera observer position. Looking at origin from (60, 60, 40) puts
/// the wave_defense settler ring in the center of the frame with
/// monsters approaching from outside the ring visible at the edges.
fn observer_camera() -> OrbitCamera {
    OrbitCamera::new(glam::Vec3::ZERO, 80.0)
}

struct WindowedViewer {
    seed: u64,
    app: ViewerApp,
    scene: Scene,
    camera: OrbitCamera,
    last_tick: Instant,
    /// Constructed lazily on first `resumed()` — winit 0.30 doesn't
    /// give a window until the event loop is running.
    gfx: Option<Gfx>,
}

struct Gfx {
    window: Arc<Window>,
    ctx: VulkanContext,
    swapchain: SwapchainContext,
    renderer: VoxelRenderer,
}

impl ApplicationHandler for WindowedViewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(self.title_for_tick(0))
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_W, WINDOW_H));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("create_window failed"),
        );

        let ctx = VulkanContext::new_with_surface_extensions(&window)
            .expect("VulkanContext::new_with_surface_extensions failed");
        let swapchain = SwapchainContext::new(&ctx, &window)
            .expect("SwapchainContext::new failed");
        let renderer = VoxelRenderer::new(&ctx, WINDOW_W, WINDOW_H)
            .expect("VoxelRenderer::new failed");

        // Phase A.5 setup: prime the Scene with whatever Phase A's
        // ViewerApp wants pre-populated. The Scene currently has no
        // role in render_frame_gpu (Phase B will bridge), but we keep
        // the call to validate the data pipeline against a real
        // Scene allocation.
        self.app
            .setup(&mut self.scene)
            .expect("ViewerApp::setup failed");

        self.gfx = Some(Gfx {
            window,
            ctx,
            swapchain,
            renderer,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                eprintln!(
                    "[viewer_window] CloseRequested — exit at tick={}",
                    self.app.sim_tick(),
                );
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Catch up on fixed-step sim ticks. If many ticks
                // elapsed (e.g. window was hidden), bound the catch-up
                // at 4 to avoid runaway after a long pause — better to
                // skip than to freeze.
                let mut ticks_this_frame: u32 = 0;
                while self.last_tick.elapsed() >= SIM_TICK_PERIOD && ticks_this_frame < 4 {
                    self.app.tick(&mut self.scene, 0.1);
                    self.last_tick += SIM_TICK_PERIOD;
                    ticks_this_frame += 1;
                }
                // Build the title before re-borrowing `self.gfx` mutably
                // (winit's set_title is borrow-conservative).
                let title = self.title_for_tick(self.app.sim_tick());
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.window.set_title(&title);

                    // Empty objects list — Phase B fills this. Renderer
                    // still produces a sky/sun pass which we present.
                    let objects: Vec<(
                        &voxel_engine::vulkan::voxel_gpu::GpuVoxelTexture,
                        [f32; 4],
                        [f32; 3],
                        [f32; 3],
                    )> = Vec::new();
                    if let Err(e) = gfx
                        .renderer
                        .render_frame_gpu(&gfx.ctx, &self.camera, &objects)
                    {
                        eprintln!("[viewer_window] render_frame_gpu failed: {e}");
                        return;
                    }
                    if let Err(e) = gfx.swapchain.present_blit(
                        &gfx.ctx,
                        gfx.renderer.light_output_image(),
                        WINDOW_W,
                        WINDOW_H,
                    ) {
                        eprintln!("[viewer_window] present_blit failed: {e}");
                        return;
                    }
                    gfx.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl WindowedViewer {
    fn title_for_tick(&self, tick: u64) -> String {
        format!(
            "viewer_window — seed={} tick={} settlers={} monsters={} score={:.1}",
            self.seed,
            tick,
            self.app.alive_settlers(),
            self.app.alive_monsters(),
            self.app.score(),
        )
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

    eprintln!(
        "[viewer_window] seed={seed} — opening window {WINDOW_W}x{WINDOW_H} \
         (Phase A.5: empty scene, no entities rendered yet)"
    );

    let event_loop = EventLoop::new().expect("EventLoop::new failed");
    let app = ViewerApp::new(seed);
    let scene = Scene::new_headless(SceneConfig::default());
    let camera = observer_camera();

    let mut viewer = WindowedViewer {
        seed,
        app,
        scene,
        camera,
        last_tick: Instant::now(),
        gfx: None,
    };

    event_loop
        .run_app(&mut viewer)
        .expect("event_loop.run_app failed");
}

// Suppress unused — RendererConfig is the type Phase B will read
// from CLI args / config; pull it in now so the import doesn't churn
// the file later.
#[allow(dead_code)]
fn _phase_b_marker() -> RendererConfig {
    RendererConfig::default()
}
