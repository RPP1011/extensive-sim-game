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

use viewer_runtime::{ViewerApp, VoxelBridge};
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

/// Cell count along each axis of the voxel-bridge grid. 128³ × 1B
/// per cell = 2 MB of CPU + GPU memory; ~10 µs to upload at PCIe 4.0
/// bandwidth, called once per fixed-step tick.
const VOXEL_GRID_DIM: u32 = 128;
/// World-space length the bridge grid covers along each axis.
/// Wave_defense uses ±64 around origin (settlers at radius 8,
/// spawners at radius 60); 128 covers it with 1 unit per cell.
const VOXEL_WORLD_EXTENT: f32 = 128.0;

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
    /// Phase B: world-grid bridge. `Option` because it's allocated
    /// after `app.setup()` populates the snapshot — first refresh
    /// happens before the first render so the grid isn't empty.
    bridge: VoxelBridge,
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

        // Setup primes ViewerApp's snapshot caches via
        // refresh_snapshot — needed before the first bridge.refresh
        // so the initial paint reflects post-setup state, not
        // pre-init zeros.
        self.app
            .setup(&mut self.scene)
            .expect("ViewerApp::setup failed");

        // Allocate the world-grid bridge + paint the initial frame
        // so render_frame_gpu has something to draw before the first
        // sim tick fires (otherwise the first ~100ms is empty).
        let mut bridge = VoxelBridge::new(
            &ctx,
            self.app.palette(),
            VOXEL_GRID_DIM,
            VOXEL_WORLD_EXTENT,
        )
        .expect("VoxelBridge::new failed");
        bridge
            .refresh(&ctx, &self.app)
            .expect("VoxelBridge::refresh (initial) failed");

        // Kick off the redraw loop. winit 0.30 doesn't send
        // RedrawRequested on its own past the initial expose;
        // about_to_wait below keeps it pumping.
        window.request_redraw();

        self.gfx = Some(Gfx {
            window,
            ctx,
            swapchain,
            renderer,
            bridge,
        });
    }

    /// Drive continuous animation. winit calls about_to_wait whenever
    /// the event queue drains; without an explicit request_redraw
    /// here, RedrawRequested fires only on OS-driven invalidation
    /// (resize, expose) and the title bar / sim tick freeze.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gfx) = self.gfx.as_ref() {
            gfx.window.request_redraw();
        }
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
                // Full cleanup before VulkanContext drops. Each
                // owned Vulkan resource has a `destroy(ctx)` method
                // (no Drop impls upstream — they all want the
                // VulkanContext, which Rust's Drop signature can't
                // pass). Order: wait_idle → bridge → swapchain →
                // renderer → drop ctx implicitly. wait_idle ensures
                // no in-flight commands are still touching these
                // resources when we destroy them.
                if let Some(mut gfx) = self.gfx.take() {
                    let _ = unsafe { gfx.ctx.device().device_wait_idle() };
                    gfx.bridge.destroy(&gfx.ctx);
                    gfx.swapchain.destroy(&gfx.ctx);
                    gfx.renderer.destroy(&gfx.ctx);
                    // ctx + window drop here naturally.
                }
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Catch up on fixed-step sim ticks. If many ticks
                // elapsed (e.g. window was hidden), bound the catch-up
                // at 4 to avoid runaway after a long pause — better to
                // skip than to freeze. Each tick refreshes the voxel
                // bridge so the rendered cells reflect the new
                // positions.
                let mut ticks_this_frame: u32 = 0;
                while self.last_tick.elapsed() >= SIM_TICK_PERIOD && ticks_this_frame < 4 {
                    self.app.tick(&mut self.scene, 0.1);
                    self.last_tick += SIM_TICK_PERIOD;
                    ticks_this_frame += 1;
                    if let Some(gfx) = self.gfx.as_mut() {
                        if let Err(e) = gfx.bridge.refresh(&gfx.ctx, &self.app) {
                            eprintln!("[viewer_window] VoxelBridge::refresh failed: {e}");
                        }
                    }
                }
                // Build the title before re-borrowing `self.gfx` mutably
                // (winit's set_title is borrow-conservative).
                let title = self.title_for_tick(self.app.sim_tick());
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.window.set_title(&title);

                    // Single render object — voxel_engine's fragment
                    // shader resolves per-cell colour through
                    // `palette_tex[voxel_id]`, so all four
                    // creature types render with their distinct
                    // palette colours from one draw.
                    let objects: Vec<_> = gfx.bridge.render_object().into_iter().collect();
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
                    // about_to_wait handles the next request_redraw —
                    // no need to schedule it again here.
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
