//! `viewer_app` — windowed viewer for `sims::dungeon_horde`.
//!
//! Opens a winit window, drives voxel_engine's Vulkan renderer, and
//! ticks the sim at 100ms wall-clock cadence. Top-down camera framed
//! on the dungeon centroid.
//!
//! Usage:
//!
//! ```text
//! cargo run -p viewer_runtime --bin viewer_app --release [SEED]
//! ```
//!
//! With no SEED, defaults to `0xD007_BEEF_5_7EA1` — the same seed the
//! `dungeon_horde_pin` test uses, so the rendered scene matches the
//! pinned test layout.
//!
//! What you should see:
//!
//! - A top-down view of a multi-room voxel dungeon (6×6 grid of room
//!   slots, ~22 rooms connected via doorways).
//! - Gray walls (stone) outlining the rooms; tan floors inside.
//! - Five hero cubes (warrior=brown, cleric=white, ranger=green,
//!   mage=blue, rogue=purple) at the spawn-room centroid.
//! - Hundreds of enemy cubes scattered through the rooms: orange
//!   archers, dim-green goblins, dark-red brutes.
//! - Per-tick (100ms wall clock) the agents move toward each other
//!   and start fighting; dead agents disappear from the scene.

use std::sync::Arc;
use std::time::{Duration, Instant};

use viewer_runtime::{ViewerApp, VoxelBridge, BRIDGE_DIM_X, BRIDGE_DIM_Z};
use voxel_engine::camera::FreeCamera;
use voxel_engine::render::VoxelRenderer;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::swapchain::SwapchainContext;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const SIM_TICK_PERIOD: Duration = Duration::from_millis(100);
const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;

/// Default seed — same as `dungeon_horde_pin.rs::SEED_U64` so the
/// rendered scene matches the pinned test layout.
const DEFAULT_SEED: u64 = 0xD007_BEEF_5_7EA1;

/// Top-down observer camera, framed on the dungeon centroid. The
/// dungeon is `GRID_X × GRID_Y` cells covering world `[0, GRID_X) × [0, GRID_Y)`,
/// so the centroid is at `(GRID_X/2, GRID_Y/2, 1)`. Camera height
/// (Y, since voxel_engine is Y-up) is enough to frame the whole
/// dungeon vertically.
fn observer_camera() -> FreeCamera {
    // Bridge grid is (X horizontal, Y vertical, Z depth) post axis-swap.
    // Centroid in the X-Z plane; camera sits above on the Y axis.
    let cx = BRIDGE_DIM_X as f32 / 2.0;
    let cz = BRIDGE_DIM_Z as f32 / 2.0;
    // Height covers the full X-Z floor at the renderer's default ~60° FOV.
    let height = BRIDGE_DIM_X.max(BRIDGE_DIM_Z) as f32 + 8.0;
    // Tilt the camera off perfect vertical to avoid gimbal lock — straight-down
    // (-Y forward) with up=+Y makes Mat4::look_at_rh degenerate. Pulling the eye
    // back along +Z gives a high-angle bird's-eye view.
    FreeCamera::new(
        glam::Vec3::new(cx, height, cz + height * 0.4),
        glam::Vec3::new(cx, 0.0, cz),
    )
}

struct WindowedViewer {
    seed: u64,
    app: ViewerApp,
    camera: FreeCamera,
    last_tick: Instant,
    /// Constructed lazily on first `resumed()` — winit 0.30 doesn't
    /// give a window until the event loop is running.
    gfx: Option<Gfx>,
    /// Wall-clock instant at which the current run terminated.
    /// `None` while the run is still in progress. Used to delay
    /// auto-restart so the user can read the verdict + see the
    /// outcome tint before the next dungeon rolls in.
    terminated_at_wall: Option<Instant>,
    /// Total runs completed in this session (incremented on each
    /// auto-restart). Drives the cross-run aggregate line.
    session_runs: u32,
    /// Subset of `session_runs` that ended in DUNGEON CLEARED.
    session_wins: u32,
    /// Sum of `app.sim_tick()` at termination across all completed
    /// runs. Divided by `session_runs` for the mean-ticks metric.
    session_tick_total: u64,
}

struct Gfx {
    window: Arc<Window>,
    ctx: VulkanContext,
    swapchain: SwapchainContext,
    renderer: VoxelRenderer,
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

        let mut bridge =
            VoxelBridge::new(&ctx, &self.app).expect("VoxelBridge::new failed");
        bridge
            .refresh(&ctx, &self.app)
            .expect("VoxelBridge::refresh (initial) failed");

        window.request_redraw();

        self.gfx = Some(Gfx { window, ctx, swapchain, renderer, bridge });
    }

    /// Drive continuous animation. winit calls about_to_wait whenever
    /// the event queue drains; without an explicit request_redraw
    /// here, RedrawRequested fires only on OS-driven invalidation.
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
                    "[viewer_app] CloseRequested — exit at tick={}",
                    self.app.sim_tick(),
                );
                if self.session_runs > 0 {
                    let mean_ticks = self.session_tick_total / self.session_runs as u64;
                    eprintln!(
                        "[viewer_app] final session: {} runs ({} cleared, {} TPK, mean {} ticks, win-rate {:.0}%)",
                        self.session_runs,
                        self.session_wins,
                        self.session_runs - self.session_wins,
                        mean_ticks,
                        100.0 * (self.session_wins as f32) / (self.session_runs as f32),
                    );
                }
                if let Some(mut gfx) = self.gfx.take() {
                    let _ = unsafe { gfx.ctx.device().device_wait_idle() };
                    gfx.bridge.destroy(&gfx.ctx);
                    gfx.swapchain.destroy(&gfx.ctx);
                    gfx.renderer.destroy(&gfx.ctx);
                }
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Catch up on fixed-step sim ticks. Bound at 4 to
                // avoid runaway after a long pause.
                let mut ticks_this_frame: u32 = 0;
                while self.last_tick.elapsed() >= SIM_TICK_PERIOD && ticks_this_frame < 4 {
                    self.app.step();
                    self.last_tick += SIM_TICK_PERIOD;
                    ticks_this_frame += 1;
                    if let Some(gfx) = self.gfx.as_mut() {
                        if let Err(e) = gfx.bridge.refresh(&gfx.ctx, &self.app) {
                            eprintln!("[viewer_app] VoxelBridge::refresh failed: {e}");
                        }
                    }
                }

                // Auto-restart with the next seed once the user has had a
                // few seconds to read the outcome. Bumping the seed gives
                // a fresh dungeon roll without the user having to relaunch
                // the binary — the viewer becomes a continuous demo reel.
                const POST_TERMINATION_HOLD: Duration = Duration::from_secs(3);
                if self.app.terminated_at_tick.is_some() && self.terminated_at_wall.is_none() {
                    self.terminated_at_wall = Some(Instant::now());
                }
                if let Some(t) = self.terminated_at_wall {
                    if t.elapsed() >= POST_TERMINATION_HOLD {
                        // Capture this run's contribution to the session aggregate
                        // before we tear it down. `outcome` is Some by construction
                        // here (we only enter the hold path after termination).
                        let won = self.app.outcome.unwrap_or(false);
                        self.session_runs += 1;
                        if won { self.session_wins += 1; }
                        self.session_tick_total += self.app.sim_tick();
                        let mean_ticks = self.session_tick_total / self.session_runs as u64;
                        eprintln!(
                            "[viewer_app] session: {} runs ({} cleared, {} TPK, mean {} ticks)",
                            self.session_runs,
                            self.session_wins,
                            self.session_runs - self.session_wins,
                            mean_ticks,
                        );

                        let next_seed = self.seed.wrapping_add(1);
                        eprintln!(
                            "[viewer_app] auto-restart: seed 0x{:X} -> 0x{:X}",
                            self.seed, next_seed,
                        );
                        if let Some(new_app) = ViewerApp::try_new(next_seed) {
                            self.app = new_app;
                            self.seed = next_seed;
                            self.last_tick = Instant::now();
                            self.terminated_at_wall = None;
                            // Rebuild the bridge so the dungeon walls/floor
                            // get re-uploaded from the new ViewerApp's grid.
                            if let Some(gfx) = self.gfx.as_mut() {
                                let _ = unsafe { gfx.ctx.device().device_wait_idle() };
                                let old_bridge = std::mem::replace(
                                    &mut gfx.bridge,
                                    VoxelBridge::new(&gfx.ctx, &self.app)
                                        .expect("VoxelBridge::new (auto-restart) failed"),
                                );
                                old_bridge.destroy(&gfx.ctx);
                                if let Err(e) = gfx.bridge.refresh(&gfx.ctx, &self.app) {
                                    eprintln!("[viewer_app] post-restart refresh failed: {e}");
                                }
                            }
                        } else {
                            eprintln!(
                                "[viewer_app] auto-restart failed (no wgpu adapter on next try).                                  Holding on the current run."
                            );
                            // Push the hold deadline forward so we don't hammer the failure path.
                            self.terminated_at_wall = Some(Instant::now());
                        }
                    }
                }
                let title = self.title_for_tick(self.app.sim_tick());
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.window.set_title(&title);
                    let objects: Vec<_> = gfx.bridge.render_object().into_iter().collect();
                    if let Err(e) = gfx
                        .renderer
                        .render_frame_gpu(&gfx.ctx, &self.camera, &objects)
                    {
                        eprintln!("[viewer_app] render_frame_gpu failed: {e}");
                        return;
                    }
                    if let Err(e) = gfx.swapchain.present_blit(
                        &gfx.ctx,
                        gfx.renderer.light_output_image(),
                        WINDOW_W,
                        WINDOW_H,
                    ) {
                        eprintln!("[viewer_app] present_blit failed: {e}");
                        return;
                    }
                }
            }
            _ => {}
        }
    }
}

impl WindowedViewer {
    fn title_for_tick(&self, tick: u64) -> String {
        let mut alive_h = 0u32;
        let mut alive_e = 0u32;
        for a in self.app.agents() {
            if !a.alive { continue; }
            if a.creature_type == viewer_runtime::dungeon::CT_HERO {
                alive_h += 1;
            } else {
                alive_e += 1;
            }
        }
        format!(
            "viewer_runtime — seed=0x{:X} tick={} heroes={}/5 enemies={}",
            self.seed, tick, alive_h, alive_e,
        )
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args
        .get(1)
        .and_then(|s| {
            if let Some(stripped) = s.strip_prefix("0x") {
                u64::from_str_radix(stripped, 16).ok()
            } else {
                s.parse().ok()
            }
        })
        .unwrap_or(DEFAULT_SEED);

    eprintln!(
        "[viewer_app] seed=0x{seed:X} — opening window {WINDOW_W}x{WINDOW_H}"
    );
    eprintln!(
        "[viewer_app] What you should see: top-down view of a voxel dungeon \
         (gray walls, tan floors, ~22 rooms). 5 hero cubes at spawn (brown=Warrior, \
         white=Cleric, green=Ranger, blue=Mage, purple=Rogue). Hundreds of enemy \
         cubes (orange=Archer, dim-green=Goblin, dark-red=Brute) in deeper rooms. \
         Tick advances every 100ms; agents move + die over time."
    );

    let app = match ViewerApp::try_new(seed) {
        Some(a) => a,
        None => {
            eprintln!(
                "[viewer_app] no compatible wgpu adapter found for sim runtime — \
                 cannot initialize. Try running on a host with a discrete GPU \
                 or software fallback (lavapipe, etc.)."
            );
            std::process::exit(2);
        }
    };
    let camera = observer_camera();

    let event_loop = EventLoop::new().expect("EventLoop::new failed");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut viewer = WindowedViewer {
        seed,
        app,
        camera,
        last_tick: Instant::now(),
        gfx: None,
        terminated_at_wall: None,
        session_runs: 0,
        session_wins: 0,
        session_tick_total: 0,
    };
    event_loop
        .run_app(&mut viewer)
        .expect("event_loop.run_app failed");
}
